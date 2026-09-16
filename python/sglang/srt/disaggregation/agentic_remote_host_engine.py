"""Existing Host lifecycle adapter for source-local DRAM → remote MHA HBM.

All methods that touch NIXL run on the existing Host I/O workers, never Forward.
The shared directory contains descriptors/receipts only, never KV payload bytes.
The existing ledger remains the authority for claim, eviction and TP commit.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
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
                 *, transport_factory=None):
        if hasattr(device_pool, "mamba_pool") or not all(
            hasattr(device_pool, field) for field in ("k_buffer", "v_buffer", "head_num", "head_dim")
        ):
            raise ValueError("multi-node Host adapter currently supports dense MHA only, not Mamba/MLA")
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
        self.root = Path(config.control_directory) / "remote-host" / direction
        self.layout = layout_fingerprint({
            "kind": "mha", "dtype": str(device_pool.store_dtype),
            "layers": int(device_pool.layer_num), "heads": int(device_pool.head_num),
            "head_dim": int(device_pool.head_dim), "page_size": self.page_size,
            "tp_size": self.size,
        })
        self.item_size = int(device_pool.head_num) * int(device_pool.head_dim) * int(device_pool.store_dtype.itemsize)
        self._factory = transport_factory
        self._transport = None
        self._init_lock = threading.Lock()
        self._exports = {}
        self._failed_exports = {}
        self.incarnation = uuid.uuid4().hex
        self._read_locks = {}

    def _directory(self, snapshot_id):
        return self.root / hashlib.sha256(snapshot_id.encode()).hexdigest()

    def _transport_for_worker(self):
        if self._factory is None:
            import torch
            torch.cuda.set_device(int(self.pool.k_buffer[0].device.index))
        with self._init_lock:
            if self._transport is not None:
                return self._transport
            if self._factory is not None:
                self._transport = self._factory()
                return self._transport
            import torch
            from nixl._api import nixl_agent, nixl_agent_config
            device_id = int(self.pool.k_buffer[0].device.index)
            torch.cuda.set_device(device_id)
            # Only Host I/O uses this agent; registrations cannot hold the
            # Direct agent's lock or a scheduler/Forward lock.
            agent = nixl_agent("dualpd-host-" + uuid.uuid4().hex,
                               nixl_agent_config(backends=["UCX"], num_threads=2))
            addresses, lengths, _ = self.pool.get_contiguous_buf_infos()
            agent.register_memory([(int(address), int(length), device_id, "")
                                   for address, length in zip(addresses, lengths)],
                                  "VRAM", backends=["UCX"])
            self._transport = RemoteHostTransport(agent)
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
        finally:
            lock.release()

    def export_snapshot(self, snapshot_id, snapshot):
        """After D2H fence; source arena must retain snapshot until cleanup."""
        transport = self._transport_for_worker()
        with self._init_lock:
            existing = self._exports.get(snapshot_id)
            if existing is not None:
                _write_json(self._directory(snapshot_id) / f"rank-{self.rank}.json",
                            {"node_id": self.config.node_id, "engine_id": self.config.engine_id,
                             "shard": existing.shard.to_dict()})
                return existing.shard.to_dict()
            if snapshot_id in self._failed_exports:
                raise RuntimeError("source Host export has unresolved registration cleanup")
            try:
                exported = transport.export(
                    snapshot_id=snapshot_id, tp_rank=self.rank, tp_size=self.size,
                    layout=self.layout, token_count=int(snapshot.token_count),
                    address=int(snapshot.kv_buffer.data_ptr()), byte_size=int(snapshot.byte_size),
                    keepalive=snapshot,
                )
            except Exception:
                # Preserve the arena lease even when registration failed
                # before a publishable RemoteHostExport object existed.
                self._failed_exports[snapshot_id] = snapshot
                raise
            self._exports[snapshot_id] = exported
            _write_json(self._directory(snapshot_id) / f"rank-{self.rank}.json",
                        {"node_id": self.config.node_id, "engine_id": self.config.engine_id,
                         "shard": exported.shard.to_dict()})
            return exported.shard.to_dict()

    def _shards(self, snapshot_id):
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
        key = hashlib.sha256(receipt.read_id.encode()).hexdigest()
        _write_json(self._directory(snapshot_id) / key / f"receipt-{self.rank}.json", asdict(receipt))

    def _claim_attempt(self, snapshot_id, attempt_id):
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

    def load(self, snapshot_id, grant, device_indices, *, attempt_id, cancel_check=None):
        with self._init_lock:
            lock = self._read_locks.setdefault((snapshot_id, attempt_id), threading.Lock())
        with lock:
            return self._load(snapshot_id, grant, device_indices,
                              attempt_id=attempt_id, cancel_check=cancel_check)

    def _load(self, snapshot_id, grant, device_indices, *, attempt_id, cancel_check=None):
        """Whole-shard network READ; call only on a Host I/O worker.

        Ordinary errors have a published drained receipt. UnfencedRemoteRead
        forbids freeing/retrying the destination workset or source Host extent.
        """
        if not attempt_id:
            raise ValueError("remote recovery requires one group-level attempt ID")
        transport = self._transport_for_worker()
        directory = self._directory(snapshot_id)
        record = _read_json(directory / f"rank-{self.rank}.json")
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
        self._claim_attempt(snapshot_id, str(attempt_id))
        attempt_dir = directory / hashlib.sha256(str(attempt_id).encode()).hexdigest()
        attempt_dir.mkdir(parents=True, exist_ok=True)
        signature = {"engine": self.config.engine_id, "incarnation": self.incarnation,
                     "shard": shard.to_dict(), "spans": [list(span) for span in spans]}
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
        try:
            pending = transport.prepare_read(shard, read_id=str(attempt_id),
                tp_rank=self.rank, tp_size=self.size, layout=self.layout,
                gpu_id=int(self.pool.k_buffer[0].device.index), spans=spans)
            if pending.state == "PREPARED":
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("remote Host load cancelled before post")
                pending.start()
            while True:
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("remote Host load cancelled")
                receipt = pending.poll()
                if receipt is not None:
                    self._publish_receipt(snapshot_id, receipt)
                    # The persistent same-attempt destination/receipt tombstone
                    # prevents repost after retiring this physically fenced handle.
                    transport.retire_read(shard.export_id, str(attempt_id))
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
            return True
        directory = self._directory(snapshot_id)
        with (directory / "attempt.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if exported.closed:
                with self._init_lock:
                    self._exports.pop(snapshot_id, None)
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
            return True
