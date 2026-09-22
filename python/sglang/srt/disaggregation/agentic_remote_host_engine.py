"""Existing Host lifecycle adapter for source-local DRAM → remote MHA HBM.

All methods that touch NIXL run on the existing Host I/O workers, never Forward.
Socket control stores descriptors/receipts in broker DRAM. The legacy backend
uses a shared directory for those records only, never KV payload bytes.
The existing ledger remains the authority for claim, eviction and TP commit.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from sglang.srt.disaggregation.agentic_multinode import load_multinode_config
from sglang.srt.disaggregation.agentic_remote_host import (
    HostShard, ReadReceipt, RemoteHostTransport, group_ack,
    layout_fingerprint, mha_read_spans,
)

logger = logging.getLogger(__name__)


class UnfencedRemoteRead(RuntimeError):
    """Destination and source must remain quarantined; not a retry signal."""


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as out:
            json.dump(value, out, separators=(",", ":"))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def create_remote_host_bridge(device_pool, page_size, tp_rank, tp_size, direction):
    config = load_multinode_config()
    if config is None:
        return None
    return RemoteHostEngineBridge(device_pool, page_size, tp_rank, tp_size, direction, config)


class RemoteHostEngineBridge:
    def __init__(self, device_pool, page_size, tp_rank, tp_size, direction, config,
                 *, transport_factory=None, control=None):
        self.hybrid = None
        if hasattr(device_pool, "mamba_pool"):
            from sglang.srt.disaggregation.agentic_remote_hybrid import HybridWireLayout
            self.hybrid = HybridWireLayout(device_pool, 2 if direction == "p2d" else 1)
            device_pool = self.hybrid.attention
        if not all(
            hasattr(device_pool, field) for field in ("k_buffer", "v_buffer", "head_num", "head_dim")
        ):
            raise ValueError("multi-node Host adapter requires an MHA Attention pool")
        if direction not in {"p2d", "d2p"}:
            raise ValueError("invalid Host direction")
        if int(tp_size) != config.tp_size or not 0 <= int(tp_rank) < int(tp_size):
            raise ValueError("Host adapter TP topology mismatch")
        if int(getattr(device_pool, "v_head_dim", device_pool.head_dim)) != int(device_pool.head_dim):
            raise ValueError("multi-node Host requires matching K/V head dimensions")
        self.pool = device_pool
        self.page_size = int(page_size)
        self.rank, self.size = int(tp_rank), int(tp_size)
        self.config = config
        self.direction = direction
        from sglang.srt.disaggregation.agentic_remote_host_control import create_remote_host_control_client
        self._control = control if control is not None else create_remote_host_control_client(direction)
        self.root = Path(config.control_directory) / "remote-host" / direction
        schema = {
            "kind": "mha", "dtype": str(device_pool.store_dtype),
            "layers": int(device_pool.layer_num), "heads": int(device_pool.head_num),
            "head_dim": int(device_pool.head_dim), "page_size": self.page_size,
            "tp_size": self.size,
        }
        if self.hybrid is not None:
            schema["hybrid"] = self.hybrid.schema()
        self.layout = layout_fingerprint(schema)
        self.item_size = int(device_pool.head_num) * int(device_pool.head_dim) * int(device_pool.store_dtype.itemsize)
        self._factory = transport_factory
        self._transport = None
        self._destination_registered = False
        self._registration_descriptors = None
        self._initialization_error = None
        self._init_lock = threading.Lock()
        self._exports = {}
        self._failed_exports = {}
        self.incarnation = uuid.uuid4().hex
        self._read_locks = {}
        self._cleanup_locks = {}
        self._source_views = {}
        self._read_timing_lock = threading.Lock()
        self._read_timing_count = 0
        self._read_timing_totals = {}

    def _record_read_timing(self, **seconds):
        """Bounded worker-only diagnostics; never poll files or synchronize CUDA.

        A Host worker's elapsed time is not DMA time. Aggregate the existing
        preparation, transport and receipt boundaries so the next run can
        distinguish control-store waits from network transfer waits.
        """
        with self._read_timing_lock:
            self._read_timing_count += 1
            for key, value in seconds.items():
                self._read_timing_totals[key] = self._read_timing_totals.get(key, 0.0) + value
            if self._read_timing_count < 64:
                return
            count, totals = self._read_timing_count, self._read_timing_totals
            self._read_timing_count, self._read_timing_totals = 0, {}
        logger.info(
            "AgenticKV remote_host_read_timing direction=%s rank=%d reads=%d %s",
            self.direction, self.rank, count,
            " ".join(f"{key}_ms={value * 1000 / count:.3f}" for key, value in totals.items()),
        )

    def _directory(self, snapshot_id):
        return self.root / hashlib.sha256(snapshot_id.encode()).hexdigest()

    def _register_destination_for_worker(self, agent, device_id):
        """Register receiver VRAM once, rolling back a partial backend init.

        No READ has been posted at this boundary. On rollback failure keep the
        agent/descriptors alive and fail closed; never create agents on retries.
        Source exporters only register their CPU extent in transport.export().
        """
        addresses, lengths, _ = self.pool.get_contiguous_buf_infos()
        regions = [(int(a), int(n), device_id, "") for a, n in zip(addresses, lengths)]
        if self.hybrid is not None:
            addresses, lengths, _ = self.hybrid.pool.get_state_buf_infos()
            regions += [(int(a), int(n), device_id, "") for a, n in zip(addresses, lengths)]
        descriptors = agent.get_reg_descs(regions, "VRAM")
        self._registration_descriptors = descriptors
        try:
            agent.register_memory(descriptors, backends=["UCX"])
        except Exception:
            try:
                agent.deregister_memory(descriptors, backends=["UCX"])
            except Exception as cleanup_error:
                raise RuntimeError("receiver VRAM registration cleanup failed; restart worker") from cleanup_error
            self._registration_descriptors = None
            raise
        self._destination_registered = True

    def _transport_for_worker(self, *, destination=False):
        if self._factory is None:
            import torch
            torch.cuda.set_device(int(self.pool.k_buffer[0].device.index))
        with self._init_lock:
            if self._initialization_error is not None:
                raise RuntimeError(self._initialization_error)
            if self._transport is None:
                if self._factory is not None:
                    self._transport = self._factory()
                else:
                    from nixl._api import nixl_agent, nixl_agent_config
                    # Only Host I/O uses this agent; registrations cannot hold
                    # the Direct agent's lock or a scheduler/Forward lock.
                    agent = nixl_agent("dualpd-host-" + uuid.uuid4().hex,
                        nixl_agent_config(backends=["UCX"], num_threads=2))
                    self._transport = RemoteHostTransport(agent)
            if destination and not self._destination_registered and self._factory is None:
                try:
                    self._register_destination_for_worker(self._transport.agent,
                        int(self.pool.k_buffer[0].device.index))
                except Exception as exc:
                    self._initialization_error = f"remote Host receiver initialization failed: {exc}"
                    raise
            return self._transport

    def cancel_unstarted(self, snapshot_id, *, attempt_id):
        """Receipt for a lifecycle-cancelled worker that provably never posted.

        Caller must first cancel its unstarted Future or own a no-I/O recovery
        record. This never waits behind a running READ and never fences one.
        """
        with self._init_lock:
            lock = self._read_locks.setdefault((snapshot_id, attempt_id), threading.Lock())
        if not lock.acquire(blocking=False):
            raise UnfencedRemoteRead("cannot cancel an executing remote Host worker")
        try:
            return self._cancel_unstarted_locked(snapshot_id, attempt_id)
        finally:
            lock.release()

    def _cancel_unstarted_locked(self, snapshot_id, attempt_id):
        if self._control is not None:
            try:
                return ReadReceipt(**self._control.call("cancel_unstarted", snapshot_id,
                    rank=self.rank, attempt_id=str(attempt_id), engine_id=self.config.engine_id))
            except Exception as error:
                raise UnfencedRemoteRead("cannot establish unstarted remote Host cancellation") from error
        directory = self._directory(snapshot_id)
        attempt_dir = directory / hashlib.sha256(str(attempt_id).encode()).hexdigest()
        prior = _read_json(attempt_dir / f"receipt-{self.rank}.json")
        if prior is not None:
            return ReadReceipt(**prior)
        if (attempt_dir / f"destination-{self.rank}.json").exists():
            raise UnfencedRemoteRead("remote Host worker already acquired its destination")
        record = _read_json(directory / f"rank-{self.rank}.json")
        if record is None:
            raise RuntimeError("cancelled Host snapshot is missing its source descriptor")
        shard = HostShard.from_dict(record["shard"])
        self._claim_attempt(snapshot_id, str(attempt_id))
        receipt = ReadReceipt(snapshot_id, shard.export_id, self.rank, self.size,
                              shard.layout, shard.token_count, str(attempt_id), "drained")
        self._publish_receipt(snapshot_id, receipt)
        return receipt

    def _publish_export(self, snapshot_id, shard):
        record = {"node_id": self.config.node_id, "engine_id": self.config.engine_id,
                  "shard": shard.to_dict()}
        if self._control is not None:
            self._control.call("export", snapshot_id, record=record)
        else:
            _write_json(self._directory(snapshot_id) / f"rank-{self.rank}.json", record)

    def _source_descriptor(self, snapshot_id):
        if self._control is not None:
            return self._control.call("descriptor", snapshot_id, rank=self.rank)
        return _read_json(self._directory(snapshot_id) / f"rank-{self.rank}.json")

    def export_snapshot(self, snapshot_id, snapshot):
        """After D2H fence; source arena must retain snapshot until cleanup."""
        transport = self._transport_for_worker()
        with self._init_lock:
            existing = self._exports.get(snapshot_id)
            if existing is not None:
                self._publish_export(snapshot_id, existing.shard)
                return existing.shard.to_dict()
            if snapshot_id in self._failed_exports:
                raise RuntimeError("source Host export has unresolved registration cleanup")
            if self.hybrid is not None:
                from sglang.srt.disaggregation.agentic_remote_hybrid import CompleteHostMapping
                if snapshot.byte_size != self.hybrid.layout(snapshot.token_count).total_bytes:
                    raise ValueError("incomplete remote hybrid source extent")
                view = CompleteHostMapping(snapshot)
                self._source_views[snapshot_id] = view
                address = view.address
                keepalive = view
            else:
                address = int(snapshot.kv_buffer.data_ptr())
                keepalive = snapshot
            try:
                exported = transport.export(
                    snapshot_id=snapshot_id, tp_rank=self.rank, tp_size=self.size,
                    layout=self.layout, token_count=int(snapshot.token_count),
                    address=address, byte_size=int(snapshot.byte_size),
                    keepalive=keepalive,
                )
            except Exception:
                # Preserve the arena lease even when registration failed
                # before a publishable RemoteHostExport object existed.
                self._failed_exports[snapshot_id] = keepalive
                raise
            self._exports[snapshot_id] = exported
            self._publish_export(snapshot_id, exported.shard)
            return exported.shard.to_dict()

    def _shards(self, snapshot_id):
        if self._control is not None:
            records = self._control.call("shards", snapshot_id)
            if records is None:
                return None
            shards = [HostShard.from_dict(record) for record in records]
            if len(shards) != self.size or any(
                shard.snapshot_id != snapshot_id or shard.tp_rank != rank or shard.tp_size != self.size
                for rank, shard in enumerate(shards)
            ):
                raise RuntimeError("remote Host descriptor identity mismatch")
            return shards
        result = []
        for rank in range(self.size):
            record = _read_json(self._directory(snapshot_id) / f"rank-{rank}.json")
            if record is None:
                return None
            shard = HostShard.from_dict(record["shard"])
            if shard.snapshot_id != snapshot_id or shard.tp_rank != rank or shard.tp_size != self.size:
                raise RuntimeError("remote Host descriptor identity mismatch")
            result.append(shard)
        return result

    def _receipts(self, snapshot_id, attempt_id, terminal_entry=None, shards=None):
        if self._control is not None:
            records = self._control.call("receipts", snapshot_id, attempt_id=attempt_id)
            return None if records is None else [ReadReceipt(**r) for r in records]
        key = hashlib.sha256(attempt_id.encode()).hexdigest()
        records = [_read_json(self._directory(snapshot_id) / key / f"receipt-{rank}.json")
                   for rank in range(self.size)]
        # P->D may cancel a peer before its worker ever enters the bridge.
        # Only an authoritative all-rank physically-drained FAILED state plus
        # absence of a published destination permits a no-READ receipt.
        if terminal_entry and terminal_entry.get("state") == "failed" and shards:
            drained = set(terminal_entry.get("loader_drained_ranks", []))
            for rank in range(self.size):
                destination = self._directory(snapshot_id) / key / f"destination-{rank}.json"
                if records[rank] is None and rank in drained and not destination.exists():
                    shard = shards[rank]
                    records[rank] = asdict(ReadReceipt(snapshot_id, shard.export_id,
                        rank, self.size, shard.layout, shard.token_count, attempt_id, "drained"))
        return None if any(record is None for record in records) else [ReadReceipt(**r) for r in records]

    def _publish_receipt(self, snapshot_id, receipt):
        if self._control is not None:
            self._control.call("receipt", snapshot_id, engine_id=self.config.engine_id,
                               receipt=asdict(receipt))
            return
        key = hashlib.sha256(receipt.read_id.encode()).hexdigest()
        _write_json(self._directory(snapshot_id) / key / f"receipt-{self.rank}.json", asdict(receipt))

    def _claim_attempt(self, snapshot_id, attempt_id):
        if self._control is not None:
            self._control.call("claim_attempt", snapshot_id, attempt_id=attempt_id,
                               engine_id=self.config.engine_id)
            return
        directory = self._directory(snapshot_id)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "attempt.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            old = _read_json(directory / "attempt.json")
            wanted = {"attempt_id": str(attempt_id), "engine_id": self.config.engine_id}
            if old == wanted:
                return
            if old is not None:
                shards = self._shards(snapshot_id)
                receipts = self._receipts(snapshot_id, old["attempt_id"])
                if shards is None or receipts is None:
                    raise RuntimeError("previous remote Host attempt is not fully fenced")
                group_ack(shards, receipts, cancelled=True)
            # Caller has already atomically acquired the current ledger recovery
            # claim; this marker prevents independent receiver engines racing.
            _write_json(directory / "attempt.json", wanted)

    def load(self, snapshot_id, grant, device_indices, *, attempt_id, cancel_check=None, state_indices=None):
        with self._init_lock:
            lock = self._read_locks.setdefault((snapshot_id, attempt_id), threading.Lock())
        with lock:
            try:
                return self._load(snapshot_id, grant, device_indices,
                                  attempt_id=attempt_id, cancel_check=cancel_check, state_indices=state_indices)
            except UnfencedRemoteRead:
                raise
            except Exception:
                # Layout/slot validation can fail before a destination/READ is
                # published. Peers still need this rank's no-I/O receipt to
                # retire the group attempt. Never invent a fence after a post.
                try:
                    self._cancel_unstarted_locked(snapshot_id, attempt_id)
                except Exception as error:
                    raise UnfencedRemoteRead("cannot establish remote READ drain proof") from error
                raise

    def _load(self, snapshot_id, grant, device_indices, *, attempt_id, cancel_check=None, state_indices=None):
        """Whole-shard network READ; call only on a Host I/O worker.

        Ordinary errors have a published drained receipt. UnfencedRemoteRead
        forbids freeing/retrying the destination workset or source Host extent.
        """
        started_at = time.perf_counter()
        if not attempt_id:
            raise ValueError("remote recovery requires one group-level attempt ID")
        transport = self._transport_for_worker(destination=True)
        directory = self._directory(snapshot_id)
        record = self._source_descriptor(snapshot_id)
        if record is None:
            raise RuntimeError("HOST_READY is missing its remote shard descriptor")
        if record.get("node_id") != grant.get("remote_host_node"):
            raise RuntimeError("remote Host source node does not match the grant")
        if grant.get("remote_host_engine") and record.get("engine_id") != grant["remote_host_engine"]:
            raise RuntimeError("remote Host source engine does not match the grant")
        shard = HostShard.from_dict(record["shard"])
        if shard.snapshot_id != snapshot_id or shard.token_count != int(grant["token_count"]):
            raise RuntimeError("remote Host snapshot/token identity mismatch")
        if shard.byte_size != int(grant["byte_size"]):
            raise RuntimeError("remote Host extent size mismatch")
        if hasattr(device_indices, "detach"):
            indices = device_indices.detach().cpu().tolist()
        else:
            indices = list(device_indices)
        if len(indices) != shard.token_count:
            raise ValueError("remote destination must contain the whole snapshot")
        if any(int(index) < 0 or int(index) >= int(self.pool.k_buffer[0].shape[0])
               for index in indices):
            raise ValueError("remote destination token index outside registered KV pool")
        spans = mha_read_spans(layer_bases=[int(t.data_ptr()) for t in
            tuple(self.pool.k_buffer) + tuple(self.pool.v_buffer)],
            token_indices=indices, bytes_per_token=self.item_size)
        ranges = None
        if self.hybrid is not None:
            if shard.byte_size != self.hybrid.layout(shard.token_count).total_bytes:
                raise ValueError("remote hybrid extent has incomplete state")
            spans += self.hybrid.spans(shard.token_count, state_indices)
            ranges = self.hybrid.ranges(shard.token_count)
        descriptors_at = time.perf_counter()
        self._claim_attempt(snapshot_id, str(attempt_id))
        claimed_at = time.perf_counter()
        signature = {"engine": self.config.engine_id, "incarnation": self.incarnation,
                     "shard": shard.to_dict(), "spans": [list(span) for span in spans]}
        if self._control is not None:
            # Destination identity is a fence, not transfer data. Keep large
            # fragmented page lists local to the NIXL I/O worker.
            signature["spans_digest"] = hashlib.sha256(json.dumps(
                signature.pop("spans"), separators=(",", ":"),
                allow_nan=False).encode()).hexdigest()
            signature["span_count"] = len(spans)
            try:
                prior_receipt = self._control.call("reserve_destination", snapshot_id,
                    rank=self.rank, attempt_id=str(attempt_id), engine_id=self.config.engine_id,
                    signature=signature)
            except Exception as error:
                # An ambiguous RPC or changed incarnation must not manufacture
                # a drain receipt for another worker's possibly active READ.
                raise UnfencedRemoteRead("cannot own remote Host destination") from error
        else:
            attempt_dir = directory / hashlib.sha256(str(attempt_id).encode()).hexdigest()
            attempt_dir.mkdir(parents=True, exist_ok=True)
            with (attempt_dir / f"rank-{self.rank}.lock").open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                previous = _read_json(attempt_dir / f"destination-{self.rank}.json")
                if previous is not None and previous != signature:
                    raise UnfencedRemoteRead("remote Host attempt changed destination/process incarnation")
                try:
                    _write_json(attempt_dir / f"destination-{self.rank}.json", signature)
                except Exception as error:
                    # This worker has not initialized/posted a READ yet. Publish
                    # its physical no-I/O proof or quarantine on control-store loss.
                    try:
                        self._publish_receipt(snapshot_id, ReadReceipt(snapshot_id,
                            shard.export_id, self.rank, self.size, shard.layout,
                            shard.token_count, str(attempt_id), "drained"))
                    except Exception as publication_error:
                        raise UnfencedRemoteRead("cannot publish unstarted READ cancellation") from publication_error
                    raise error
            prior_receipt = _read_json(attempt_dir / f"receipt-{self.rank}.json")
        if prior_receipt is not None:
            prior_receipt = ReadReceipt(**prior_receipt)
            if prior_receipt.outcome != "loaded":
                raise RuntimeError("remote Host attempt already cancelled")
            return prior_receipt
        pending = None
        destination_at = time.perf_counter()
        try:
            pending = transport.prepare_read(shard, read_id=str(attempt_id),
                tp_rank=self.rank, tp_size=self.size, layout=self.layout,
                gpu_id=int(self.pool.k_buffer[0].device.index), spans=spans,
                payload_ranges=ranges)
            prepared_at = time.perf_counter()
            if pending.state == "PREPARED":
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("remote Host load cancelled before post")
                pending.start()
            while True:
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("remote Host load cancelled")
                receipt = pending.poll()
                if receipt is not None:
                    transferred_at = time.perf_counter()
                    self._publish_receipt(snapshot_id, receipt)
                    published_at = time.perf_counter()
                    # The persistent same-attempt destination/receipt tombstone
                    # prevents repost after retiring this physically fenced handle.
                    transport.retire_read(shard.export_id, str(attempt_id))
                    try:
                        self._record_read_timing(
                            descriptors=descriptors_at - started_at,
                            claim=claimed_at - descriptors_at,
                            destination=destination_at - claimed_at,
                            prepare=prepared_at - destination_at,
                            transfer=transferred_at - prepared_at,
                            receipt=published_at - transferred_at,
                            retire=time.perf_counter() - published_at,
                        )
                    except Exception:
                        # Diagnostics must not turn an already-fenced and
                        # published successful READ into a cancellation.
                        pass
                    return receipt
                if pending.state in {"ERR", "UNKNOWN"}:
                    raise RuntimeError("remote Host READ failed")
                time.sleep(0.001)
        except Exception as error:
            if pending is not None:
                try:
                    receipt = pending.drain_failure()
                    self._publish_receipt(snapshot_id, receipt)
                    transport.retire_read(shard.export_id, str(attempt_id))
                except Exception as fence_error:
                    raise UnfencedRemoteRead("remote Host READ fence unresolved") from fence_error
            else:
                self._publish_receipt(snapshot_id, ReadReceipt(shard.snapshot_id,
                    shard.export_id, self.rank, self.size, shard.layout,
                    shard.token_count, str(attempt_id), "drained"))
            raise error

    def cleanup_source(self, snapshot_id, ledger_entry):
        """Return True only when source registration is safely gone.

        Existing ledger all-rank terminal state is the authority. No consumer
        attempt allows FAILED/REJECTED/EVICTING cleanup without a remote fence;
        all published attempts instead require complete loaded/drained receipts.
        """
        with self._init_lock:
            exported = self._exports.get(snapshot_id)
            failed_snapshot = self._failed_exports.get(snapshot_id)
            if exported is None and failed_snapshot is None:
                return True
        self._transport_for_worker()  # select the rank's CUDA device on cleanup thread too
        state = (ledger_entry or {}).get("state")
        success = state == "consumed"
        if state not in {"consumed", "failed", "rejected", "evicting", "recompute_required"}:
            return False
        if failed_snapshot is not None:
            transport = self._transport_for_worker()
            with transport.lock:
                transport.retry_unpublished_cleanup()
                if any(item.get("keepalive") is failed_snapshot
                       for item in transport.quarantined.values()):
                    return False
            with self._init_lock:
                self._failed_exports.pop(snapshot_id, None)
                self._close_source_view(snapshot_id)
            return True
        if self._control is not None:
            with self._init_lock:
                cleanup_lock = self._cleanup_locks.setdefault(snapshot_id, threading.Lock())
            with cleanup_lock:
                return self._cleanup_control_source(snapshot_id, exported, ledger_entry)
        directory = self._directory(snapshot_id)
        with (directory / "attempt.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if exported.closed:
                with self._init_lock:
                    self._exports.pop(snapshot_id, None)
                    self._close_source_view(snapshot_id)
                return True
            attempt = _read_json(directory / "attempt.json")
            if attempt is not None:
                shards = self._shards(snapshot_id)
                receipts = self._receipts(snapshot_id, attempt["attempt_id"], ledger_entry, shards)
                if shards is None or receipts is None:
                    return False
                group_ack(shards, receipts, cancelled=not success)
                exported.claim(attempt["attempt_id"])
                if success:
                    exported.release_after_group_ack(shards, receipts,
                        committed_read_id=attempt["attempt_id"])
                else:
                    exported.unclaim_after_group_cancel(shards, receipts)
                    with exported.transport.lock:
                        exported.transport.agent.deregister_memory(exported.registration,
                            backends=exported.transport.backends)
                        exported.closed = True
                        exported.keepalive = None
                        exported.transport.exports.pop(exported.shard.export_id, None)
            else:
                if success and (ledger_entry.get("loader_acks") or ledger_entry.get("loading_ranks")
                                or ledger_entry.get("recovery_claims")):
                    return False
                with exported.transport.lock:
                    exported.transport.agent.deregister_memory(exported.registration,
                        backends=exported.transport.backends)
                    exported.closed = True
                    exported.keepalive = None
                    exported.transport.exports.pop(exported.shard.export_id, None)
            with self._init_lock:
                self._exports.pop(snapshot_id, None)
                self._close_source_view(snapshot_id)
            return True

    def _cleanup_control_source(self, snapshot_id, exported, ledger_entry):
        # The broker seals a fully fenced generation before returning this
        # immutable plan. No remote mutex is held while NIXL deregisters memory.
        plan = self._control.call("prepare_cleanup", snapshot_id, rank=self.rank,
            export_id=exported.shard.export_id, terminal_entry=ledger_entry or {})
        if plan is None:
            return False
        if not exported.closed:
            attempt = plan["attempt"]
            if attempt is not None:
                shards = [HostShard.from_dict(s) for s in plan["shards"]]
                receipts = [ReadReceipt(**r) for r in plan["receipts"]]
                group_ack(shards, receipts, cancelled=not plan["success"])
                exported.claim(attempt["attempt_id"])
                if plan["success"]:
                    exported.release_after_group_ack(shards, receipts,
                        committed_read_id=attempt["attempt_id"])
                else:
                    exported.unclaim_after_group_cancel(shards, receipts)
            if not exported.closed:
                with exported.transport.lock:
                    exported.transport.agent.deregister_memory(exported.registration,
                        backends=exported.transport.backends)
                    exported.closed = True
                    exported.keepalive = None
                    exported.transport.exports.pop(exported.shard.export_id, None)
        self._control.call("complete_cleanup", snapshot_id, rank=self.rank,
                           export_id=exported.shard.export_id)
        with self._init_lock:
            self._exports.pop(snapshot_id, None)
            self._close_source_view(snapshot_id)
        return True

    def _close_source_view(self, snapshot_id):
        view = self._source_views.get(snapshot_id)
        if view is not None:
            view.close()
            self._source_views.pop(snapshot_id)
