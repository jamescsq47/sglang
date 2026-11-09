from __future__ import annotations

"""D-side bounded Host writer and optional NUMA Relay.

Physical snapshot buffers and the shared ledger have separate canonical owners.
This role never frees request HBM: only the caller may release it after the
existing full-snapshot durable fence and group-level commit.
"""
import hashlib
import logging
import math
import os
import time
from concurrent.futures import Future
from typing import Any, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.utils import kv_to_page_indices
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotManifest, page_namespace
from sglang.srt.disaggregation.agentic_host_ledger import (
    HostStageState, PermanentHostStageError, SharedHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_host_buffers import (
    H2DLaunchFence, HostCopyWorkerPool, LayerFirstD2HStaging,
    PinnedMHAHostBounce, SharedMHAHostSnapshot, _device_indices_to_host,
)
from sglang.srt.disaggregation.agentic_host_dma import _cuda_driver_batch_memcpy

# Keep historical diagnostics stable after moving the implementation.
logger = logging.getLogger("sglang.srt.disaggregation.agentic_host_staging")

_LEDGER_ENTRY_UNSET = object()


def _cleanup_nixl_room(manager, room: int, bootstrap_addr: Optional[str] = None) -> None:
    for name in (
        "request_status",
        "failure_records",
        "required_prefill_response_num_table",
        "prefill_response_tracker",
        "transfer_infos",
        "transfer_statuses",
    ):
        table = getattr(manager, name, None)
        if table is not None:
            table.pop(room, None)
    if bootstrap_addr:
        # Only DECODE-role managers own addr_to_rooms_tracker.  Relay sources
        # are PREFILL-role reverse managers and have no receiver-side tracker.
        tracker = getattr(manager, "addr_to_rooms_tracker", None)
        rooms = None if tracker is None else tracker.get(bootstrap_addr)
        if rooms is not None:
            rooms.discard(room)


class AgenticDRelayWorker:
    """Arena-local D that relays remote-D KV through fixed HBM staging slots."""

    def __init__(
        self,
        *,
        ledger: SharedHostStagingLedger,
        relay_id: str,
        numa_node: int,
        device_pool,
        token_allocator,
        receiver_runtime,
        page_size: int,
        slot_mib: int,
        slot_count: int,
        d2h_gib_per_second: float,
        relay_aux_index: int,
    ):
        self.ledger = ledger
        self.relay_id = str(relay_id)
        self.numa_node = int(numa_node)
        self.device_pool = device_pool
        self.token_allocator = token_allocator
        self.runtime = receiver_runtime
        self.page_size = int(page_size)
        self.slot_count = max(1, int(slot_count))
        # Reusing the stock P->D NIXL manager also reuses its registered
        # MetadataBuffers.  One row is permanently removed from the normal
        # request allocator and used only as the relay completion mailbox, so
        # the reverse path can never overwrite live request metadata.
        self.relay_aux_index = int(relay_aux_index)
        bytes_per_page = sum(
            int(value) for value in receiver_runtime.manager.kv_args.kv_item_lens
        )
        requested_bytes = max(1, int(slot_mib)) * 1024**2
        slot_pages = max(1, requested_bytes // bytes_per_page)
        self.slot_token_count = int(slot_pages * self.page_size)
        # These allocator entries are intentionally pinned for the lifetime of
        # the relay.  Expose the exact count so the scheduler's strict memory
        # checker can distinguish them from leaked request KV.
        self.reserved_token_count = self.slot_count * self.slot_token_count
        self.slots = []
        try:
            for _ in range(self.slot_count):
                indices = token_allocator.alloc(self.slot_token_count)
                if indices is None:
                    raise RuntimeError("not enough D HBM for relay staging slots")
                self.slots.append(indices)
        except Exception:
            for indices in self.slots:
                token_allocator.free(indices)
            self.slots.clear()
            raise
        # Decode is the latency/throughput-critical stage.  A high-priority
        # D2H gather stream preempts Decode kernels whenever several complete
        # snapshots are offloaded together, turning an otherwise bandwidth-
        # sufficient slow path into visible Decode bubbles.  Keep D2H at the
        # normal (lowest supported on A100) priority by default; deployments
        # can still override it explicitly for latency experiments.
        d2h_priority = int(
            os.getenv("SGLANG_AGENTIC_KV_D2H_STREAM_PRIORITY", "0")
        )
        self._d2h_stream = torch.cuda.Stream(
            device=torch.cuda.current_device(), priority=d2h_priority
        )
        self._d2h_staging = LayerFirstD2HStaging(
            self.device_pool,
            int(os.getenv("SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS", "1024")),
        )
        self._d2h_host_bounce = PinnedMHAHostBounce(
            self.device_pool, self.slot_token_count
        )
        self.active: Optional[dict[str, Any]] = None
        self._last_heartbeat = 0.0
        self.ledger.register_relay(
            relay_id=self.relay_id,
            pid=os.getpid(),
            numa_node=self.numa_node,
            slot_token_count=self.slot_token_count,
            slot_count=self.slot_count,
            d2h_gib_per_second=float(d2h_gib_per_second),
        )
        logger.info(
            "AgenticKV relay_register relay=%s numa=%d slots=%d "
            "tokens_per_slot=%d reserved_mib=%.1f",
            self.relay_id,
            self.numa_node,
            self.slot_count,
            self.slot_token_count,
            self.slot_count * self.slot_token_count * bytes_per_page
            / self.page_size
            / 1024**2,
        )

    @staticmethod
    def _room(snapshot_id: str, seq: int, first_room: int) -> int:
        if seq == 0:
            return int(first_room)
        raw = hashlib.sha256(f"relay:{snapshot_id}:{seq}".encode()).digest()[:8]
        return int.from_bytes(raw, "little") & ((1 << 63) - 1)

    def _close_receiver(self, active: dict[str, Any]) -> None:
        receiver = active.pop("receiver", None)
        if receiver is None:
            return
        try:
            receiver.clear()
        except Exception:
            logger.debug("Relay receiver clear failed", exc_info=True)
        _cleanup_nixl_room(
            receiver.kv_mgr,
            int(active.get("room", 0)),
            active["entry"].get("source_bootstrap_addr"),
        )

    def _finish_active(self) -> None:
        active = self.active
        if active is None:
            return
        self._close_receiver(active)
        event = active.pop("event", None)
        if event is not None:
            event.synchronize()
        snapshot = active.pop("snapshot", None)
        if snapshot is not None:
            snapshot.close(unlink=False)
        self.active = None

    def _fail_active(self, reason: str) -> None:
        active = self.active
        if active is None:
            return
        snapshot_id = active["entry"]["snapshot_id"]
        try:
            self._finish_active()
        finally:
            self.ledger.relay_fail_to_direct(snapshot_id, self.relay_id, reason)
        logger.warning(
            "AgenticKV relay_fallback_direct snapshot=%s relay=%s reason=%s",
            snapshot_id,
            self.relay_id,
            reason,
        )

    def _claim(self) -> None:
        entry = self.ledger.claim_relay_job(self.relay_id, os.getpid())
        if entry is None:
            return
        grants = entry.get("grants", [])
        if len(grants) != 1 or grants[0].get("kind") != "shared_host_extent":
            self.ledger.relay_fail_to_direct(
                entry["snapshot_id"], self.relay_id, "invalid_host_extent"
            )
            return
        grant = grants[0]
        try:
            snapshot = SharedMHAHostSnapshot(
                path=str(grant["arena_path"]),
                token_count=int(grant["token_count"]),
                device_pool=self.device_pool,
                byte_size=int(grant["byte_size"]),
                create=False,
                file_offset=int(grant.get("arena_offset", 0)),
            )
        except Exception:
            logger.exception("Relay could not map Host extent for %s", entry["snapshot_id"])
            self.ledger.relay_fail_to_direct(
                entry["snapshot_id"], self.relay_id, "relay_host_map_failed"
            )
            return
        self.active = {
            "entry": entry,
            "snapshot": snapshot,
            "seq": int(entry.get("relay_completed_tokens", 0))
            // self.slot_token_count,
            "completed_tokens": int(entry.get("relay_completed_tokens", 0)),
            "phase": "prepare",
        }
        logger.info(
            "AgenticKV relay_claim snapshot=%s relay=%s bytes=%d",
            entry["snapshot_id"],
            self.relay_id,
            int(entry["byte_size"]),
        )

    def _prepare_chunk(self, active: dict[str, Any]) -> None:
        entry = active["entry"]
        start = int(active["completed_tokens"])
        remaining = int(entry["token_count"]) - start
        count = min(self.slot_token_count, remaining)
        seq = int(active["seq"])
        slot = self.slots[seq % self.slot_count][:count]
        room = self._room(entry["snapshot_id"], seq, int(entry["source_room"]))
        bootstrap_addr = str(entry["source_bootstrap_addr"])
        try:
            if not self.runtime.manager.try_ensure_parallel_info(bootstrap_addr):
                return
            receiver = self.runtime.receiver_class(
                mgr=self.runtime.manager,
                bootstrap_addr=bootstrap_addr,
                bootstrap_room=room,
            )
            receiver.init(prefill_dp_rank=0)
            if receiver.poll() == KVPoll.Failed:
                raise RuntimeError("relay NIXL receiver init failed")
            pages = kv_to_page_indices(slot.cpu().numpy(), self.page_size)
            receiver.send_metadata(pages, aux_index=self.relay_aux_index)
            if not self.ledger.relay_prepare_chunk(
                entry["snapshot_id"],
                relay_id=self.relay_id,
                seq=seq,
                start_token=start,
                token_count=count,
                room=room,
            ):
                receiver.clear()
                _cleanup_nixl_room(receiver.kv_mgr, room, bootstrap_addr)
                raise RuntimeError("relay chunk publication was rejected")
        except Exception:
            logger.exception("Relay chunk setup failed for %s", entry["snapshot_id"])
            self._fail_active("relay_chunk_setup_failed")
            return
        active.update(
            phase="receiving",
            receiver=receiver,
            room=room,
            slot=slot,
            chunk_start=start,
            chunk_count=count,
        )
        logger.info(
            "AgenticKV relay_chunk_ready snapshot=%s relay=%s seq=%d tokens=%d",
            entry["snapshot_id"],
            self.relay_id,
            seq,
            count,
        )

    def _poll_active(self) -> None:
        active = self.active
        if active is None:
            return
        if active["phase"] == "prepare":
            self._prepare_chunk(active)
            return
        if active["phase"] == "receiving":
            try:
                result = active["receiver"].poll()
            except Exception:
                logger.exception("Relay NIXL receive failed")
                result = KVPoll.Failed
            if result == KVPoll.Failed:
                self._fail_active("relay_nixl_receive_failed")
                return
            if result != KVPoll.Success:
                return
            self._close_receiver(active)
            try:
                event, refs = active["snapshot"].start_backup_range_from_device(
                    active["slot"],
                    destination_start=active["chunk_start"],
                    stream=self._d2h_stream,
                    staging=self._d2h_staging,
                    host_bounce=self._d2h_host_bounce,
                )
            except Exception:
                logger.exception("Relay D2H launch failed")
                self._fail_active("relay_d2h_launch_failed")
                return
            active.update(phase="d2h", event=event, copy_refs=refs)
            return
        if active["phase"] != "d2h" or not active["event"].query():
            return
        active["event"].synchronize()
        d2h_elapsed_ms = active["snapshot"]._last_d2h_start_event.elapsed_time(
            active["event"]
        )
        active["snapshot"].commit_backup_range_from_bounce(
            self._d2h_host_bounce,
            destination_start=active["chunk_start"],
            token_count=active["chunk_count"],
        )
        active.pop("event", None)
        active.pop("copy_refs", None)
        entry = active["entry"]
        seq = int(active["seq"])
        if not self.ledger.relay_complete_chunk(
            entry["snapshot_id"], self.relay_id, seq
        ):
            self._fail_active("relay_chunk_commit_failed")
            return
        active["completed_tokens"] += int(active["chunk_count"])
        logger.info(
            "AgenticKV relay_chunk_complete snapshot=%s relay=%s seq=%d tokens=%d "
            "elapsed_ms=%.3f gib_per_s=%.3f",
            entry["snapshot_id"],
            self.relay_id,
            seq,
            int(active["chunk_count"]),
            d2h_elapsed_ms,
            2
            * int(active["chunk_count"])
            * active["snapshot"].layout_dim
            / max(d2h_elapsed_ms / 1000.0, 1e-9)
            / (1024**3),
        )
        if active["completed_tokens"] >= int(entry["token_count"]):
            logger.info(
                "AgenticKV relay_host_complete snapshot=%s relay=%s",
                entry["snapshot_id"],
                self.relay_id,
            )
            self._finish_active()
            return
        active["seq"] += 1
        active["phase"] = "prepare"

    def poll(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= 0.5:
            self.ledger.heartbeat_relay(self.relay_id, os.getpid())
            self._last_heartbeat = now
        if self.active is None:
            self._claim()
        self._poll_active()


class AgenticDHostStagingClient:
    """D-side direct writer for P-owned shared Host extents.

    This object never frees request HBM itself.  It returns ``host_ready`` only
    after the D CUDA event completed, the shared mapping was unregistered, and
    the ledger atomically published the complete snapshot.
    """

    def __init__(
        self,
        ledger: SharedHostStagingLedger,
        device_pool,
        page_size: int,
        *,
        direct_runtime=None,
        relay_enabled: bool = False,
        source_numa_node: int = -1,
        arena_numa_node: int = -1,
        arena_domain: int = -1,
        tp_rank: int = 0,
        tp_size: int = 1,
        retain_logical_hashes: bool = False,
        direct_cross_numa_gib_per_second: float = 7.45,
        nvlink_gib_per_second: float = 220.0,
        relay_stale_seconds: float = 5.0,
    ):
        self.ledger = ledger
        self.device_pool = device_pool
        self.page_size = int(page_size)
        self.direct_runtime = direct_runtime
        self.relay_enabled = bool(relay_enabled and direct_runtime is not None)
        self.source_numa_node = int(source_numa_node)
        self.arena_numa_node = int(arena_numa_node)
        self.arena_domain = int(arena_domain)
        # Every TP scheduler owns a different KV shard and must publish that
        # shard under its own ledger rank.  Leaving these unset makes
        # ``offer()`` fall back to rank 0 in every process, so the second
        # scheduler looks like a non-idempotent rewrite of rank 0 instead of
        # the rank-1 half of the same logical snapshot.
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid D Host staging TP rank")
        # Page hashes are needed only by the optional storage-spill backend.
        # Shared-Arena control should not carry hundreds of unused hashes.
        self.retain_logical_hashes = bool(retain_logical_hashes)
        self.direct_cross_numa_gib_per_second = float(
            direct_cross_numa_gib_per_second
        )
        self.nvlink_gib_per_second = float(nvlink_gib_per_second)
        self.relay_stale_seconds = float(relay_stale_seconds)
        d2h_priority = int(
            os.getenv("SGLANG_AGENTIC_KV_D2H_STREAM_PRIORITY", "0")
        )
        # Never enqueue an entire multi-GiB snapshot on the D copy stream.
        # CUDA stream priority only chooses among *pending* kernels/copies; a
        # long queue of gather+D2H work can otherwise keep winning between
        # Decode forwards.  Submit one bounded chunk at a time and let the
        # agentic progress worker yield back to Decode before the next chunk.
        self._d2h_chunk_tokens = max(
            self.page_size,
            int(os.getenv("SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS", "256")),
        )
        self._d2h_chunk_tokens = (
            self._d2h_chunk_tokens // self.page_size * self.page_size
        )
        lane_count = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_D2H_INFLIGHT", "4"))
        )
        self._d2h_dma_limit = lane_count
        self._d2h_bounce_depth = max(
            2, int(os.getenv("SGLANG_AGENTIC_KV_D2H_BOUNCE_DEPTH", "2"))
        )
        self._d2h_host_copy_pool = HostCopyWorkerPool(
            f"agentic-d-d2h-host-copy-{os.getpid()}",
            max(
                1,
                int(
                    os.getenv(
                        "SGLANG_AGENTIC_KV_D2H_HOST_COPY_WORKERS",
                        str(lane_count),
                    )
                ),
            ),
        )
        # A CUDA submission without an armed completion event has no safe
        # allocator or mapping-release boundary. Such records are retained for
        # process lifetime instead of converting uncertainty into use-after-free.
        self._d2h_dma_quarantine: list[tuple[Any, ...]] = []
        staging_tokens = int(
            os.getenv("SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS", "256")
        )
        # Each physical DMA lane owns one HBM gather buffer/stream but several
        # pinned bounce slots.  After DMA into bounce[0], its CPU commit may run
        # while the same stream and gather buffer service bounce[1].  This
        # doubles pipeline depth without doubling the expensive HBM staging.
        self._d2h_lanes = []
        for dma_group in range(lane_count):
            stream = torch.cuda.Stream(
                device=torch.cuda.current_device(), priority=d2h_priority
            )
            staging = LayerFirstD2HStaging(self.device_pool, staging_tokens)
            for _ in range(self._d2h_bounce_depth):
                self._d2h_lanes.append(
                    {
                        "stream": stream,
                        "staging": staging,
                        "host_bounce": PinnedMHAHostBounce(
                            self.device_pool, self._d2h_chunk_tokens
                        ),
                        "snapshot_id": None,
                        "phase": "free",
                        "dma_group": dma_group,
                    }
                )
        # A bounce slot is pipeline depth, not an independent snapshot slot.
        # Counting every bounce here admitted eight snapshots onto four DMA
        # groups and round-robin progress kept all eight complete generations
        # resident in D HBM.  One active snapshot per physical DMA group lets
        # bounce[0] commit on CPU while bounce[1] receives the next chunk,
        # without increasing the number of concurrent gather/DMA operations
        # that can contend with Decode Forward.
        self.max_active_writes = lane_count

    def set_target(self, *, prefill_domain: int, arena_numa_node: int) -> None:
        """Apply TP0's route before this rank publishes its shard offer."""

        if any(lane["snapshot_id"] is not None for lane in self._d2h_lanes):
            raise RuntimeError("cannot retarget an active Host snapshot")
        self.arena_domain = int(prefill_domain)
        self.arena_numa_node = int(arena_numa_node)

    def offer(
        self,
        *,
        manifest: SnapshotManifest,
        metadata,
        token_count: int,
        token_digest: str,
        logical_hashes: list[str],
        byte_size: int,
        arena_domain: Optional[int] = None,
        arena_numa_node: Optional[int] = None,
    ) -> dict[str, Any]:
        # The destination is immutable for this request-generation.  Do not
        # keep it as mutable client state: several snapshots may be copying at
        # once and each may have selected a different logical P.
        target_domain = (
            self.arena_domain if arena_domain is None else int(arena_domain)
        )
        target_numa = (
            self.arena_numa_node
            if arena_numa_node is None
            else int(arena_numa_node)
        )
        offer = {
            "snapshot_id": manifest.snapshot_id,
            "request_id": metadata.current.request_id,
            "generation": metadata.current.generation,
            "token_count": int(token_count),
            "token_digest": token_digest,
            "byte_size": int(byte_size),
            "storage_namespace": page_namespace(metadata.current),
            "tool_type": metadata.tool_type,
            "tool_started_at": manifest.tool_started_at or time.time(),
            "d_pid": os.getpid(),
            "source_numa_node": self.source_numa_node,
            "arena_numa_node": target_numa,
            "arena_domain": target_domain,
            "source_bootstrap_addr": (
                None
                if self.direct_runtime is None
                else self.direct_runtime.bootstrap_addr
            ),
            "source_room": manifest.direct_room,
            "tp_rank": int(getattr(self, "tp_rank", 0)),
            "tp_size": int(getattr(self, "tp_size", 1)),
            "kv_layout_hash": manifest.kv_layout_hash,
        }
        if self.retain_logical_hashes:
            offer["logical_hashes"] = list(logical_hashes)
        return self.ledger.offer(offer)

    @staticmethod
    def has_active_local_write(candidate: dict[str, Any]) -> bool:
        """Whether D2H can advance using only its local CUDA fence.

        Once the Host extent has been granted and the first chunk has started,
        P cannot consume it before D publishes HOST_READY.  Re-reading the
        shared ledger for every 512-token chunk therefore adds only global
        flock/JSON contention.  A concurrent abort remains safe: D retains its
        source pages, finishes (or drains) the bounded local I/O, and the final
        atomic ledger commit rejects the stale writer.
        """

        return bool(
            candidate.get("arena_write") is not None
            and not candidate.get("rank_host_write_complete", False)
        )

    def _cleanup_write(self, candidate: dict[str, Any]) -> bool:
        write = candidate.get("arena_write")
        if write is None:
            return True
        chunks = write.get("chunks", {})
        physically_quiesced = True
        for lane_id, chunk in tuple(chunks.items()):
            cpu_future = chunk.get("cpu_future")
            if cpu_future is not None:
                cpu_future.cancel()
                if not cpu_future.done():
                    physically_quiesced = False
                    continue
                try:
                    cpu_future.result()
                except BaseException:
                    pass
            launch_fence = chunk.get("launch_fence")
            if launch_fence is not None and launch_fence.submitted:
                if launch_fence.unavailable or not launch_fence.armed:
                    if not chunk.get("dma_quarantined"):
                        chunk["dma_quarantined"] = True
                        quarantine = getattr(self, "_d2h_dma_quarantine", None)
                        if quarantine is None:
                            quarantine = self._d2h_dma_quarantine = []
                        quarantine.append(
                            (
                                write.get("snapshot"),
                                launch_fence,
                                chunk.get("copy_refs"),
                                candidate,
                            )
                        )
                        logger.error(
                            "AgenticKV D2H cleanup found an unfenced CUDA "
                            "submission snapshot=%s; retaining D source, Host "
                            "mapping, bounce, and lane for process lifetime",
                            candidate["manifest"].snapshot_id,
                        )
                    physically_quiesced = False
                    continue
                if not launch_fence.event.query():
                    physically_quiesced = False
                    continue
            event = chunk.get("event")
            if event is not None and not event.query():
                physically_quiesced = False
                continue
        if not physically_quiesced:
            return False
        try:
            write["snapshot"].close(unlink=False)
        except Exception:
            # The physically fenced snapshot remains the cleanup carrier.  In
            # particular, an unregister failure must not make its lane or mmap
            # reusable until a later cleanup visit succeeds.
            logger.exception(
                "AgenticKV D2H registered extent cleanup retry snapshot=%s",
                candidate["manifest"].snapshot_id,
            )
            return False
        candidate.pop("arena_write", None)
        for lane_id in tuple(chunks):
            lane = self._d2h_lanes[int(lane_id)]
            if lane["snapshot_id"] == candidate["manifest"].snapshot_id:
                lane["snapshot_id"] = None
                lane["phase"] = "free"
        return True

    def _cleanup_relay_senders(self, candidate: dict[str, Any]) -> None:
        senders = candidate.pop("relay_senders", {})
        for state in senders.values():
            sender = state.get("sender")
            if sender is None:
                continue
            _cleanup_nixl_room(
                sender.kv_mgr,
                int(state["room"]),
                state.get("bootstrap_addr"),
            )

    def _progress_relay_source(
        self,
        candidate: dict[str, Any],
        entry: dict[str, Any],
        source_token_indices,
    ) -> str:
        chunk = entry.get("relay_chunk")
        if chunk is None:
            return "waiting"
        seq = int(chunk["seq"])
        senders = candidate.setdefault("relay_senders", {})
        state = senders.get(seq)
        if state is None:
            room = int(chunk["room"])
            if seq == 0 and int(candidate["manifest"].direct_room) == room:
                sender = candidate["sender"]
            else:
                sender = self.direct_runtime.sender_class(
                    mgr=self.direct_runtime.manager,
                    bootstrap_addr=self.direct_runtime.bootstrap_addr,
                    bootstrap_room=room,
                    dest_tp_ranks=[0],
                    pp_rank=0,
                )
            state = {
                "sender": sender,
                "room": room,
                "bootstrap_addr": self.direct_runtime.bootstrap_addr,
                "sent": False,
            }
            senders[seq] = state
        sender = state["sender"]
        try:
            poll = sender.poll()
            if not state["sent"] and poll == KVPoll.WaitingForInput:
                # The relay receiver intentionally reuses the stock P->D NIXL
                # manager, whose registered MetadataBuffers contain ten
                # fields.  Relay chunks require only a completion notification;
                # copy the reverse runtime's one-byte sentinel into the first
                # pointer of the relay's *reserved* metadata row.  The row
                # index is carried by TransferInfo.dst_aux_index.
                manager = sender.kv_mgr
                source_aux_count = len(manager.kv_args.aux_data_ptrs)
                if source_aux_count != 1:
                    raise RuntimeError("relay source requires one sentinel aux item")
                for transfer in manager.transfer_infos.get(state["room"], {}).values():
                    remote = manager.decode_kv_args_table[transfer.agent_name]
                    if len(remote.dst_aux_ptrs) > source_aux_count:
                        remote.dst_aux_ptrs = remote.dst_aux_ptrs[:source_aux_count]
                start = int(chunk["start_token"])
                count = int(chunk["token_count"])
                pages = kv_to_page_indices(
                    source_token_indices[start : start + count].cpu().numpy(),
                    self.page_size,
                )
                sender.init(len(pages), aux_index=0)
                sender.send(pages)
                state["sent"] = True
                if not self.ledger.relay_mark_source_sent(
                    candidate["manifest"].snapshot_id, seq, os.getpid()
                ):
                    raise RuntimeError("relay source-send marker was rejected")
                logger.info(
                    "AgenticKV relay_nvlink_start snapshot=%s relay=%s "
                    "seq=%d tokens=%d",
                    candidate["manifest"].snapshot_id,
                    entry.get("relay_id"),
                    seq,
                    count,
                )
                return "waiting"
            if state["sent"] and poll == KVPoll.Success:
                _cleanup_nixl_room(
                    sender.kv_mgr,
                    int(state["room"]),
                    state.get("bootstrap_addr"),
                )
                senders.pop(seq, None)
                logger.info(
                    "AgenticKV relay_nvlink_complete snapshot=%s relay=%s seq=%d",
                    candidate["manifest"].snapshot_id,
                    entry.get("relay_id"),
                    seq,
                )
            elif poll == KVPoll.Failed:
                raise RuntimeError("relay source NIXL sender failed")
        except Exception:
            # The relay receiver owns the only possible in-flight destination.
            # It will drain/fail the job before changing write_mode to direct.
            logger.exception(
                "Agentic relay source transfer failed for %s",
                candidate["manifest"].snapshot_id,
            )
        return "waiting"

    def _start_write(
        self,
        candidate: dict[str, Any],
        entry: dict[str, Any],
        source_token_indices,
    ) -> None:
        grants = entry.get("grants", [])
        matching = [
            grant
            for grant in grants
            if int(grant.get("tp_rank", 0)) == self.tp_rank
            and grant.get("kind") == "shared_host_extent"
        ]
        if len(matching) != 1:
            raise PermanentHostStageError(
                f"P did not publish Host extent for TP rank {self.tp_rank}"
            )
        grant = matching[0]
        expected_tokens = len(source_token_indices)
        if int(grant.get("token_count", -1)) != expected_tokens:
            raise PermanentHostStageError(
                "Shared Host grant token count does not match D snapshot: "
                f"grant={grant.get('token_count')} source={expected_tokens}"
            )
        snapshot = SharedMHAHostSnapshot(
            path=str(grant["arena_path"]),
            token_count=int(grant["token_count"]),
            device_pool=self.device_pool,
            byte_size=int(grant["byte_size"]),
            create=False,
            file_offset=int(grant.get("arena_offset", 0)),
        )
        if hasattr(self.device_pool, "mamba_pool"):
            snapshot.set_state_indices(candidate["source_state_indices"])
        write = candidate["arena_write"] = {
            "snapshot": snapshot,
            "chunks": {},
            "next_offset": 0,
            "retry_ranges": [],
            "committed_tokens": 0,
            "gpu_elapsed_ms": 0.0,
            "host_copy_elapsed_seconds": 0.0,
            "wall_started_at": time.monotonic(),
        }
        if _cuda_driver_batch_memcpy() is not None:
            if torch.is_tensor(source_token_indices) and source_token_indices.is_cuda:
                copy_pool = getattr(self, "_d2h_host_copy_pool", None)
                if copy_pool is not None:
                    def populate_index_mirror():
                        write["source_indices_host"] = _device_indices_to_host(
                            source_token_indices
                        )

                    write["index_mirror_future"] = copy_pool.submit(
                        populate_index_mirror
                    )
            else:
                write["source_indices_host"] = _device_indices_to_host(
                    source_token_indices
                )
        logger.info(
            "AgenticKV shared_host_d2h_start snapshot=%s bytes=%d "
            "chunk_tokens=%d",
            candidate["manifest"].snapshot_id,
            snapshot.byte_size,
            self._d2h_chunk_tokens,
        )

    def _start_write_chunk(
        self,
        candidate: dict[str, Any],
        source_token_indices,
    ) -> bool:
        """Submit one bounded chunk on any free independent D2H lane."""

        snapshot_id = candidate["manifest"].snapshot_id
        write = candidate["arena_write"]
        index_future = write.get("index_mirror_future")
        if index_future is not None:
            if not index_future.done():
                return False
            write.pop("index_mirror_future", None)
            try:
                index_future.result()
            except BaseException:
                # Index mirroring is only a fast-path preparation.  The
                # existing gather kernel remains a complete fallback.
                write.pop("source_indices_host", None)
                logger.exception(
                    "AgenticKV D2H index mirror failed snapshot=%s; "
                    "using gather fallback",
                    snapshot_id,
                )
        chunks = write["chunks"]
        if len(chunks) >= int(getattr(self, "_d2h_bounce_depth", 2)):
            return False
        # Multiple DMA submissions for one snapshot do not improve the CPU
        # commit pipeline, but they do increase gather pressure on Forward.
        # Wait only for the DMA boundary (not CPU commit) before submitting
        # the next chunk from this snapshot.
        if any(chunk.get("phase") == "dma" for chunk in chunks.values()):
            return False
        dma_inflight = sum(
            lane.get("phase") == "dma" for lane in self._d2h_lanes
        )
        if dma_inflight >= int(getattr(self, "_d2h_dma_limit", len(self._d2h_lanes))):
            return False
        busy_dma_groups = {
            int(lane.get("dma_group", index))
            for index, lane in enumerate(self._d2h_lanes)
            if lane.get("phase") == "dma"
        }
        lane_id = next(
            (
                index
                for index, lane in enumerate(self._d2h_lanes)
                if lane["snapshot_id"] is None and lane.get("phase", "free") == "free"
                and int(lane.get("dma_group", index)) not in busy_dma_groups
            ),
            None,
        )
        if lane_id is None:
            return False
        lane = self._d2h_lanes[lane_id]
        retry_ranges = write["retry_ranges"]
        if retry_ranges:
            start, end = retry_ranges[0]
            range_kind = "retry"
        else:
            start = int(write["next_offset"])
            if start >= len(source_token_indices):
                return False
            end = min(start + self._d2h_chunk_tokens, len(source_token_indices))
            range_kind = "new"
        launch_fence = H2DLaunchFence(
            event=torch.cuda.Event(enable_timing=True)
        )
        lane["snapshot_id"] = snapshot_id
        lane["phase"] = "dma"
        chunk = {
            "start": int(start),
            "end": int(end),
            "phase": "dma",
            "event": None,
            "copy_refs": None,
            "launch_fence": launch_fence,
        }
        try:
            event, refs = write["snapshot"].start_backup_range_from_device(
                source_token_indices[start:end],
                destination_start=start,
                stream=lane["stream"],
                staging=lane["staging"],
                host_bounce=lane["host_bounce"],
                launch_fence=launch_fence,
                source_indices_host=(
                    write["source_indices_host"][start:end]
                    if write.get("source_indices_host") is not None
                    else None
                ),
            )
        except Exception as exc:
            if launch_fence.submitted:
                # Preserve the lane, source refs, and completion authority.
                # No pageable Host bytes are committed from this failed
                # bounce. Once the fence drains, progress() reopens the same
                # complete extent and retries from the unchanged offset.
                chunk["launch_error"] = exc
                chunk["copy_refs"] = launch_fence.copy_refs
                chunk["event"] = (
                    launch_fence.event if launch_fence.armed else None
                )
                chunks[lane_id] = chunk
                if range_kind == "retry":
                    retry_ranges.pop(0)
                else:
                    write["next_offset"] = int(end)
                if launch_fence.unavailable:
                    logger.error(
                        "AgenticKV D2H launch has no completion fence "
                        "snapshot=%s; quarantining D source and Host extent",
                        snapshot_id,
                    )
            else:
                lane["snapshot_id"] = None
                lane["phase"] = "free"
                candidate["arena_write_retry_at"] = time.monotonic() + 0.05
                logger.exception(
                    "AgenticKV D2H launch failed before submission "
                    "snapshot=%s; retrying",
                    snapshot_id,
                )
            return False
        chunk["event"] = event
        chunk["copy_refs"] = refs
        chunk["start_event"] = getattr(
            write["snapshot"], "_last_d2h_start_event", None
        )
        chunks[lane_id] = chunk
        if range_kind == "retry":
            retry_ranges.pop(0)
        else:
            write["next_offset"] = int(end)
        return True

    def progress(
        self,
        candidate: dict[str, Any],
        source_token_indices,
        *,
        entry_snapshot=_LEDGER_ENTRY_UNSET,
        local_write_only: bool = False,
    ) -> str:
        snapshot_id = candidate["manifest"].snapshot_id
        if local_write_only:
            if not self.has_active_local_write(candidate):
                raise ValueError("local D2H progress requires an active Host write")
            # The immutable extent and transfer mode were captured before the
            # first chunk.  No remote state is needed until the final commit.
            entry = {
                "state": HostStageState.HOST_WRITING.value,
                "write_mode": "direct_local_progress",
            }
        else:
            entry = (
                self.ledger.get(snapshot_id)
                if entry_snapshot is _LEDGER_ENTRY_UNSET
                else entry_snapshot
            )
        if entry is None:
            if not self._cleanup_write(candidate):
                return "waiting"
            # Active entries are never pruned; an absent record therefore
            # means metadata corruption or an unproven terminal transition.
            # Fail closed and retain the complete D source.  Recreating an
            # offer here could resurrect an already-consumed generation after
            # its terminal tombstone was pruned.
            if not candidate.get("missing_host_ledger_logged"):
                candidate["missing_host_ledger_logged"] = True
                logger.error(
                    "AgenticKV active Host ledger entry missing snapshot=%s; "
                    "retaining D source",
                    snapshot_id,
                )
            return "waiting"
        candidate.pop("missing_host_ledger_logged", None)
        state = entry.get("state")
        durable_states = {
            HostStageState.HOST_READY.value,
            HostStageState.H2D_LOADING.value,
            # Pressure eviction can race D's observation of the HOST_READY
            # fence.  Both states prove that the complete Host copy was
            # durable before intentional recomputation was chosen, so D may
            # still release its obsolete source pages.
            HostStageState.EVICTING.value,
            HostStageState.RECOMPUTE_REQUIRED.value,
            HostStageState.SPILLING.value,
            HostStageState.MOONCAKE_READY.value,
            HostStageState.CONSUMED.value,
        }
        if state in {
            HostStageState.REJECTED.value,
            HostStageState.FAILED.value,
        }:
            if not self._cleanup_write(candidate):
                return "waiting"
            self._cleanup_relay_senders(candidate)
            return "failed"
        if state == HostStageState.ABORTING.value:
            try:
                if not self._cleanup_write(candidate):
                    return "waiting"
                self.ledger.mark_writer_rank_drained(
                    snapshot_id,
                    os.getpid(),
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
            except Exception:
                logger.exception("Failed to drain shared Host writer for %s", snapshot_id)
                # The local DMA/write has been made physically quiescent, but
                # the shared lifecycle ACK is still authoritative.  Retain the
                # D source and retry that idempotent ACK; treating a transient
                # ledger error as terminal would discard the only complete
                # request-generation snapshot.
                return "waiting"
            return "waiting"
        if candidate.get("rank_host_write_complete"):
            return "host_ready" if state in durable_states else "waiting"
        if state in durable_states:
            # HOST_READY is a monotonic durability boundary.  P may already be
            # loading or spilling by the time D polls; all later states still
            # imply that a complete non-D copy exists.
            if not self._cleanup_write(candidate):
                return "waiting"
            self._cleanup_relay_senders(candidate)
            return "host_ready"
        write = candidate.get("arena_write")
        if state != HostStageState.HOST_WRITING.value:
            return "waiting"
        write_mode = entry.get("write_mode")
        if not write_mode:
            if self.relay_enabled:
                entry = self.ledger.assign_transfer_path(
                    snapshot_id,
                    source_pid=os.getpid(),
                    source_numa_node=self.source_numa_node,
                    arena_numa_node=int(
                        entry.get("arena_numa_node", self.arena_numa_node)
                    ),
                    direct_cross_numa_gib_per_second=(
                        self.direct_cross_numa_gib_per_second
                    ),
                    nvlink_gib_per_second=self.nvlink_gib_per_second,
                    relay_stale_seconds=self.relay_stale_seconds,
                )
                if entry is None:
                    return "failed"
                write_mode = entry.get("write_mode")
                logger.info(
                    "AgenticKV host_path_select snapshot=%s mode=%s relay=%s "
                    "relay_eta_ms=%.3f direct_eta_ms=%.3f",
                    snapshot_id,
                    write_mode,
                    entry.get("relay_id"),
                    float(entry.get("relay_predicted_seconds", 0.0)) * 1000,
                    float(entry.get("direct_predicted_seconds", 0.0)) * 1000,
                )
            else:
                write_mode = "direct"
        if write_mode == "relay":
            return self._progress_relay_source(
                candidate, entry, source_token_indices
            )
        if write_mode == "direct_cross_numa_fallback":
            self._cleanup_relay_senders(candidate)
        if write is None:
            if time.monotonic() < float(
                candidate.get("arena_write_retry_at", 0.0)
            ):
                return "waiting"
            try:
                self._start_write(candidate, entry, source_token_indices)
            except PermanentHostStageError:
                logger.exception(
                    "Agentic shared Host D2H grant is incompatible for %s",
                    snapshot_id,
                )
                self.ledger.fail_host_write(
                    snapshot_id,
                    os.getpid(),
                    "shared_host_d2h_start_failed",
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return "failed"
            except Exception:
                # No CUDA copy has started and D still owns the complete HBM
                # source. mmap/open/registration failures are transient;
                # retain both the grant and D source and retry independently
                # instead of converting infrastructure jitter into Prefill.
                candidate["arena_write_retry_at"] = time.monotonic() + 0.05
                logger.exception(
                    "Agentic shared Host D2H start transient failure for %s; "
                    "retrying",
                    snapshot_id,
                )
                return "waiting"
            write = candidate["arena_write"]
            candidate.pop("arena_write_retry_at", None)
        chunks = write["chunks"]

        # Retire at most one physical stage per progress visit.  This keeps the
        # control worker bounded while allowing DMA(N+1) to run as soon as the
        # previous DMA hands bounce[N] to an independent CPU commit worker.
        for lane_id, chunk in tuple(chunks.items()):
            cpu_future = chunk.get("cpu_future")
            if cpu_future is None or not cpu_future.done():
                continue
            lane = self._d2h_lanes[int(lane_id)]
            try:
                cpu_elapsed = float(cpu_future.result())
            except BaseException:
                # D remains the unique authoritative owner.  Re-copy this
                # disjoint range; a partial pageable write is never published.
                write["retry_ranges"].append(
                    (int(chunk["start"]), int(chunk["end"]))
                )
                candidate["arena_write_retry_at"] = time.monotonic() + 0.05
                logger.exception(
                    "AgenticKV pageable Host D2H commit failed "
                    "snapshot=%s offset=%d; retaining D source and retrying",
                    snapshot_id,
                    int(chunk["start"]),
                )
            else:
                write["host_copy_elapsed_seconds"] = float(
                    write.get("host_copy_elapsed_seconds", 0.0)
                ) + cpu_elapsed
                write["committed_tokens"] = int(
                    write.get("committed_tokens", 0)
                ) + int(chunk["end"]) - int(chunk["start"])
                candidate.pop("arena_write_retry_at", None)
            chunks.pop(lane_id, None)
            lane["snapshot_id"] = None
            lane["phase"] = "free"
            return "waiting"

        for lane_id, chunk in tuple(chunks.items()):
            if chunk.get("launch_error") is None:
                continue
            launch_fence = chunk.get("launch_fence")
            if launch_fence is None or launch_fence.unavailable:
                return "waiting"
            event = chunk.get("event")
            if event is None or not event.query():
                return "waiting"
            event.synchronize()
            lane = self._d2h_lanes[int(lane_id)]
            lane["snapshot_id"] = None
            lane["phase"] = "free"
            chunks.pop(lane_id, None)
            write["retry_ranges"].append(
                (int(chunk["start"]), int(chunk["end"]))
            )
            candidate["arena_write_retry_at"] = time.monotonic() + 0.05
            logger.warning(
                "AgenticKV drained partial D2H launch snapshot=%s; retrying "
                "complete chunk from D source",
                snapshot_id,
            )
            return "waiting"

        for lane_id, chunk in tuple(chunks.items()):
            if chunk.get("phase") != "dma":
                continue
            event = chunk.get("event")
            if event is None or not event.query():
                continue
            start_event = chunk.get("start_event")
            if start_event is not None:
                chunk_elapsed_ms = float(start_event.elapsed_time(event))
                if math.isfinite(chunk_elapsed_ms):
                    write["gpu_elapsed_ms"] += chunk_elapsed_ms
            lane = self._d2h_lanes[int(lane_id)]
            chunk["event"] = None
            chunk["phase"] = "cpu"
            lane["phase"] = "cpu"
            copy_pool = getattr(self, "_d2h_host_copy_pool", None)
            copy_args = (lane["host_bounce"],)
            copy_kwargs = {
                "destination_start": int(chunk["start"]),
                "token_count": int(chunk["end"]) - int(chunk["start"]),
            }
            if copy_pool is None:
                future = Future()
                started_at = time.perf_counter()
                try:
                    write["snapshot"].commit_backup_range_from_bounce(
                        *copy_args, **copy_kwargs
                    )
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(time.perf_counter() - started_at)
            else:
                future = copy_pool.submit(
                    write["snapshot"].commit_backup_range_from_bounce,
                    *copy_args,
                    **copy_kwargs,
                )
            chunk["cpu_future"] = future
            return "waiting"

        has_unassigned_range = bool(write["retry_ranges"]) or int(
            write["next_offset"]
        ) < len(source_token_indices)
        if has_unassigned_range:
            if time.monotonic() >= float(
                candidate.get("arena_write_retry_at", 0.0)
            ) and self._start_write_chunk(candidate, source_token_indices):
                return "waiting"
            if chunks:
                return "waiting"
            return "waiting"

        if chunks:
            return "waiting"
        if int(write.get("committed_tokens", 0)) != len(source_token_indices):
            logger.error(
                "AgenticKV D2H commit accounting mismatch snapshot=%s "
                "committed=%d expected=%d; retaining D source",
                snapshot_id,
                int(write.get("committed_tokens", 0)),
                len(source_token_indices),
            )
            return "waiting"

        d2h_elapsed_ms = float(write["gpu_elapsed_ms"])
        d2h_bytes = write["snapshot"].byte_size
        d2h_wall_seconds = max(
            0.0,
            time.monotonic()
            - float(write.get("wall_started_at", time.monotonic())),
        )
        elapsed_ready = time.time()
        if not self._cleanup_write(candidate):
            return "waiting"
        if not self.ledger.complete_host_write(
            snapshot_id,
            os.getpid(),
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        ):
            self.ledger.mark_writer_rank_drained(
                snapshot_id,
                os.getpid(),
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
            return "failed"
        logger.info(
            "AgenticKV shared_host_d2h_complete snapshot=%s completed_at=%.6f "
            "elapsed_ms=%.3f host_copy_ms=%.3f wall_ms=%.3f gib_per_s=%.3f "
            "e2e_gib_per_s=%.3f",
            snapshot_id,
            elapsed_ready,
            d2h_elapsed_ms,
            float(write.get("host_copy_elapsed_seconds", 0.0)) * 1000.0,
            d2h_wall_seconds * 1000.0,
            d2h_bytes / max(d2h_elapsed_ms / 1000.0, 1e-9) / (1024**3),
            d2h_bytes / max(d2h_wall_seconds, 1e-9) / (1024**3),
        )
        if self.tp_size == 1:
            ready = True
        else:
            committed = self.ledger.get(snapshot_id)
            ready = (
                committed is not None
                and committed.get("state") == HostStageState.HOST_READY.value
            )
        if not ready:
            candidate["rank_host_write_complete"] = True
        return "host_ready" if ready else "waiting"
