"""File-free worker for source-local Host snapshots and remote NIXL READs.

The group controller owns request routing and lifecycle.  This class owns only
one TP rank's physical registration/READ handle and reports a result after the
real NIXL fence.  It never scans a directory or makes a scheduling decision.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import threading
import time
import uuid

from sglang.srt.disaggregation.agentic_remote_host import (
    HostShard,
    ReadReceipt,
    RemoteHostTransport,
    group_ack,
    layout_fingerprint,
    mha_page_read_spans,
    mha_read_spans,
)
from sglang.srt.disaggregation.agentic_remote_hybrid import (
    CompleteHostMapping,
    HybridWireLayout,
)


class UnfencedRemoteRead(RuntimeError):
    """The destination and source must remain quarantined."""


class DrainedRemoteRead(RuntimeError):
    """A failed/cancelled READ with an authoritative no-inflight receipt."""

    def __init__(self, message: str, receipt: ReadReceipt):
        super().__init__(message)
        self.receipt = receipt


def _remote_host_progress_thread_count() -> int:
    value = int(
        os.getenv(
            "SGLANG_AGENTIC_REMOTE_HOST_NIXL_THREADS",
            os.getenv("SGLANG_AGENTIC_NIXL_PROGRESS_THREADS", "8"),
        )
    )
    if value < 1:
        raise ValueError("remote Host NIXL progress threads must be positive")
    return value


class RemoteHostRankWorker:
    """One rank's data-plane adapter; safe only on a background I/O worker."""

    def __init__(
        self,
        device_pool,
        page_size: int,
        tp_rank: int,
        tp_size: int,
        direction: str,
        *,
        transport_factory=None,
        source_arena=None,
    ):
        if direction not in {"p2d", "d2p"}:
            raise ValueError("invalid Host transfer direction")
        self.hybrid = None
        full_pool = device_pool
        if hasattr(device_pool, "mamba_pool"):
            self.hybrid = HybridWireLayout(
                device_pool, 2 if direction == "p2d" else 1
            )
            full_pool = self.hybrid.attention
        if not all(
            hasattr(full_pool, name)
            for name in ("k_buffer", "v_buffer", "head_num", "head_dim")
        ):
            raise ValueError("remote Host worker requires an MHA attention pool")
        if int(tp_size) < 1 or not 0 <= int(tp_rank) < int(tp_size):
            raise ValueError("invalid TP rank/size")

        self.device_pool = device_pool
        self.pool = full_pool
        self.page_size = int(page_size)
        self.rank, self.size = int(tp_rank), int(tp_size)
        self.direction = direction
        schema = {
            "kind": "mha",
            "dtype": str(full_pool.store_dtype),
            "layers": int(full_pool.layer_num),
            "heads": int(full_pool.head_num),
            "head_dim": int(full_pool.head_dim),
            "page_size": self.page_size,
            "tp_size": self.size,
        }
        if self.hybrid is not None:
            schema["hybrid"] = self.hybrid.schema()
        self.layout = layout_fingerprint(schema)
        self.item_size = (
            int(full_pool.head_num)
            * int(full_pool.head_dim)
            * int(full_pool.store_dtype.itemsize)
        )
        self._factory = transport_factory
        self._source_arena = source_arena
        self._arena_mapping = None
        self._arena_base = 0
        self._arena_registration = None
        self._arena_metadata = None
        self._transport = None
        self._destination_registered = False
        self._init_lock = threading.RLock()
        self._exports = {}
        self._source_views = {}
        # Includes exports whose NIXL publication failed after a partial DRAM
        # registration.  The backing arena extent cannot be reclaimed while
        # its keepalive remains quarantined in RemoteHostTransport.
        self._export_keepalives = {}

    @property
    def gpu_id(self) -> int:
        return int(self.pool.k_buffer[0].device.index)

    def _transport_for_worker(self, *, destination: bool) -> RemoteHostTransport:
        with self._init_lock:
            if self._transport is None:
                if self._factory is not None:
                    self._transport = self._factory()
                else:
                    import torch
                    from nixl._api import nixl_agent, nixl_agent_config

                    torch.cuda.set_device(self.gpu_id)
                    progress_threads = _remote_host_progress_thread_count()
                    config_kwargs = {
                        "backends": ["UCX"],
                        "num_threads": progress_threads,
                    }
                    try:
                        from nixl._api import nixl_thread_sync_t

                        config_kwargs["sync_mode"] = (
                            nixl_thread_sync_t.NIXL_THREAD_SYNC_RW
                        )
                    except (ImportError, AttributeError):
                        pass
                    try:
                        config = nixl_agent_config(**config_kwargs)
                    except TypeError:
                        # Compatibility with NIXL releases predating
                        # ``sync_mode``; RemoteHostTransport still serializes
                        # Python API calls with its lifecycle lock.
                        config_kwargs.pop("sync_mode", None)
                        config = nixl_agent_config(**config_kwargs)
                    agent = nixl_agent("dualpd-host-" + uuid.uuid4().hex, config)
                    self._transport = RemoteHostTransport(agent)
            if destination and not self._destination_registered:
                agent = self._transport.agent
                addresses, lengths, _ = self.pool.get_contiguous_buf_infos()
                regions = [
                    (int(address), int(length), self.gpu_id, "")
                    for address, length in zip(addresses, lengths)
                ]
                if self.hybrid is not None:
                    addresses, lengths, _ = self.hybrid.pool.get_state_buf_infos()
                    regions.extend(
                        (int(address), int(length), self.gpu_id, "")
                        for address, length in zip(addresses, lengths)
                    )
                agent.register_memory(regions, "VRAM", backends=["UCX"])
                self._destination_registered = True
            return self._transport

    def prewarm_source_arena(self) -> None:
        """Register a recyclable source arena once, before serving requests."""
        if self._source_arena is None:
            return
        with self._init_lock:
            if self._arena_registration is not None:
                return
            arena = self._source_arena
            fd = os.open(arena.path, os.O_RDWR)
            try:
                mapping = mmap.mmap(
                    fd, int(arena.capacity_bytes), access=mmap.ACCESS_WRITE
                )
            finally:
                os.close(fd)
            try:
                base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                # Retain the mapping even if NIXL reports an ambiguous partial
                # registration failure; only process teardown may then unmap it.
                self._arena_mapping = mapping
                transport = self._transport_for_worker(destination=False)
                registration, metadata = transport.register_shared_arena(
                    base, int(arena.capacity_bytes), mapping
                )
            except Exception:
                raise
            self._arena_base = base
            self._arena_registration = registration
            self._arena_metadata = metadata

    def export_snapshot(self, snapshot_id: str, snapshot) -> HostShard:
        """Export after the source-local D2H event has completed."""
        with self._init_lock:
            existing = self._exports.get(snapshot_id)
            if existing is not None:
                return existing.shard
            if snapshot_id in self._export_keepalives:
                raise RuntimeError(
                    f"previous Host export is still quarantined: {snapshot_id}"
                )
            keepalive = snapshot
            address = int(snapshot.kv_buffer.data_ptr()) if self._source_arena is None and self.hybrid is None else 0
            shared = self._source_arena is not None
            if shared:
                self.prewarm_source_arena()
                offset = int(snapshot.file_offset)
                if (
                    snapshot.path != self._source_arena.path
                    or offset < 0
                    or offset + int(snapshot.byte_size)
                    > int(self._source_arena.capacity_bytes)
                ):
                    raise ValueError("snapshot is outside its registered Host arena")
                address = self._arena_base + offset
            elif self.hybrid is not None:
                expected = self.hybrid.layout(snapshot.token_count).total_bytes
                if int(snapshot.byte_size) != expected:
                    raise ValueError("incomplete hybrid Host snapshot")
                keepalive = CompleteHostMapping(snapshot)
                address = keepalive.address
                self._source_views[snapshot_id] = keepalive
            transport = self._transport_for_worker(destination=False)
            self._export_keepalives[snapshot_id] = keepalive
            exported = transport.export(
                snapshot_id=snapshot_id,
                tp_rank=self.rank,
                tp_size=self.size,
                layout=self.layout,
                token_count=int(snapshot.token_count),
                address=address,
                byte_size=int(snapshot.byte_size),
                keepalive=keepalive,
                shared_registration=(self._arena_registration if shared else None),
                shared_metadata=(self._arena_metadata if shared else None),
            )
            self._exports[snapshot_id] = exported
            return exported.shard

    def claim_export(self, snapshot_id: str, attempt_id: str) -> HostShard:
        with self._init_lock:
            exported = self._exports.get(snapshot_id)
            if exported is None:
                raise KeyError(f"unknown Host snapshot {snapshot_id}")
            return exported.claim(attempt_id)

    def reserve_export_eviction(self, snapshot_id: str, eviction_id: str) -> bool:
        with self._init_lock:
            exported = self._exports.get(snapshot_id)
            return bool(
                exported is not None and exported.reserve_eviction(eviction_id)
            )

    def cancel_export_eviction(self, snapshot_id: str, eviction_id: str) -> bool:
        with self._init_lock:
            exported = self._exports.get(snapshot_id)
            return bool(
                exported is not None and exported.cancel_eviction(eviction_id)
            )

    def finish_export_eviction(self, snapshot_id: str, eviction_id: str) -> bool:
        with self._init_lock:
            exported = self._exports.get(snapshot_id)
            if exported is None or not exported.finish_eviction(eviction_id):
                return False
            self._exports.pop(snapshot_id, None)
            self._export_keepalives.pop(snapshot_id, None)
            view = self._source_views.pop(snapshot_id, None)
            if view is not None:
                view.close()
            return True

    def discard_unclaimed_export(self, snapshot_id: str) -> bool:
        """Abort a Host-store commit before any recovery reader claims it."""

        with self._init_lock:
            exported = self._exports.get(snapshot_id)
            if exported is None:
                keepalive = self._export_keepalives.get(snapshot_id)
                if keepalive is None:
                    return True
                transport = self._transport
                if transport is None or not transport.discard_unpublished_keepalive(
                    keepalive
                ):
                    return False
            else:
                if not exported.discard_unclaimed():
                    return False
                self._exports.pop(snapshot_id, None)
            self._export_keepalives.pop(snapshot_id, None)
            view = self._source_views.pop(snapshot_id, None)
            if view is not None:
                view.close()
            return True

    def _attention_spans(
        self,
        token_count: int,
        *,
        device_indices=None,
        page_indices=None,
    ) -> list[tuple[int, int, int]]:
        bases = [
            int(tensor.data_ptr())
            for tensor in tuple(self.pool.k_buffer) + tuple(self.pool.v_buffer)
        ]
        if page_indices is not None:
            return mha_page_read_spans(
                layer_bases=bases,
                page_indices=page_indices,
                page_size=self.page_size,
                token_count=token_count,
                bytes_per_token=self.item_size,
            )
        if device_indices is None:
            raise ValueError("destination token or page indices are required")
        if hasattr(device_indices, "detach"):
            device_indices = device_indices.detach().cpu().tolist()
        if len(device_indices) != token_count:
            raise ValueError("destination must cover the complete snapshot")
        return mha_read_spans(
            layer_bases=bases,
            token_indices=device_indices,
            bytes_per_token=self.item_size,
        )

    def load(
        self,
        shard_value,
        *,
        attempt_id: str,
        device_indices=None,
        page_indices=None,
        state_indices=None,
        cancel_check=None,
    ) -> ReadReceipt:
        """Read one complete rank shard and return only after its real fence."""
        shard = (
            shard_value
            if isinstance(shard_value, HostShard)
            else HostShard.from_dict(shard_value)
        )
        if shard.snapshot_id == "" or shard.token_count <= 0:
            raise ValueError("invalid remote Host shard")
        spans = self._attention_spans(
            shard.token_count,
            device_indices=device_indices,
            page_indices=page_indices,
        )
        payload_ranges = None
        if self.hybrid is not None:
            spans.extend(self.hybrid.state_spans(shard.token_count, state_indices))
            payload_ranges = self.hybrid.payload_ranges(shard.token_count)
        transport = self._transport_for_worker(destination=True)
        pending = None
        try:
            pending = transport.prepare_read(
                shard,
                read_id=str(attempt_id),
                tp_rank=self.rank,
                tp_size=self.size,
                layout=self.layout,
                gpu_id=self.gpu_id,
                spans=spans,
                payload_ranges=payload_ranges,
            )
            if pending.state == "PREPARED":
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("remote Host load cancelled before post")
                pending.start()
            while True:
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("remote Host load cancelled")
                receipt = pending.poll()
                if receipt is not None:
                    transport.retire_read(shard.export_id, str(attempt_id))
                    return receipt
                if pending.state in {"ERR", "UNKNOWN"}:
                    raise RuntimeError("remote Host READ failed")
                time.sleep(0.001)
        except Exception as error:
            if pending is None:
                receipt = ReadReceipt(
                    shard.snapshot_id,
                    shard.export_id,
                    shard.tp_rank,
                    shard.tp_size,
                    shard.layout,
                    shard.token_count,
                    str(attempt_id),
                    "drained",
                )
                raise DrainedRemoteRead(str(error), receipt) from error
            try:
                receipt = pending.drain_failure()
                transport.retire_read(shard.export_id, str(attempt_id))
            except Exception as fence_error:
                raise UnfencedRemoteRead(
                    "remote Host READ fence is unresolved"
                ) from fence_error
            raise DrainedRemoteRead(str(error), receipt) from error

    def release_export(
        self,
        snapshot_id: str,
        shards,
        receipts,
        *,
        committed_attempt: str,
    ) -> bool:
        """Release source DRAM only after the group ownership commit."""
        group_ack(shards, receipts)
        with self._init_lock:
            exported = self._exports.get(snapshot_id)
            if exported is None:
                return False
            changed = exported.release_after_group_ack(
                shards, receipts, committed_read_id=committed_attempt
            )
            self._exports.pop(snapshot_id, None)
            self._export_keepalives.pop(snapshot_id, None)
            view = self._source_views.pop(snapshot_id, None)
            if view is not None:
                view.close()
            return changed

    def cancel_export(self, snapshot_id: str, shards, receipts) -> bool:
        """Retain Host data after every rank has physically drained the read."""
        group_ack(shards, receipts, cancelled=True)
        with self._init_lock:
            exported = self._exports.get(snapshot_id)
            if exported is None:
                return False
            return exported.unclaim_after_group_cancel(shards, receipts)
