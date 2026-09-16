"""Source-node D→Host ownership adapter for opt-in multi-node serving.

The local D2H implementation, workset allocator, TP ledger barriers and P Radix
commit are unchanged. This object only relocates the Host owner to the source D
node and retains exported bytes until the existing group lifecycle permits reuse.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


class RemoteOnlyArena:
    """P has no source D→P Arena in multi-node mode (D owns those bytes)."""

    path = None
    capacity_bytes = used_bytes = committed_bytes = 0
    preallocation_seconds = 0.0
    backend = "source-node-memfd"

    def __init__(self, directory):
        self.directory = directory

    def usage(self):
        return 0.0

    def can_reserve(self, *args):
        return False

    def close(self):
        pass

    def release(self, snapshot):
        raise RuntimeError("P cannot free source-node D Host extents")


class RemoteHostDescriptor:
    """Metadata-only parent; never opens the other node's memfd path."""

    cuda_host_registered = False

    def __init__(self, grant):
        self.grant = dict(grant)
        self.byte_size = int(grant["byte_size"])
        self.token_count = int(grant["token_count"])
        self._materialized = self

    def materialize(self):
        return self

    def close(self, *, unlink=False):
        # The receiver does not own Host. Source cleanup watches group commit.
        pass


class SourceLocalD2PArena:
    def __init__(self, *, client, config, arena=None, bridge=None, start_thread=True):
        from sglang.srt.disaggregation.agentic_host_staging import SharedHostSnapshotArena
        from sglang.srt.disaggregation.agentic_remote_host_engine import create_remote_host_bridge

        self.client = client
        self.config = config
        self.ledger = client.ledger
        self.tp_rank, self.tp_size = client.tp_rank, client.tp_size
        self.owner = "d-host:" + config.engine_id
        self.device_pool = client.device_pool
        self.storage_spill_enabled = False
        self.workset_broker = None
        self._lock = threading.RLock()
        self.records = {}
        self._pressure = False
        self._required_bytes = 0
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._thread = None
        directory = os.path.join(
            os.getenv("SGLANG_AGENTIC_MULTINODE_LOCAL_ROOT", "/dev/shm/dualpd"),
            config.run_id, config.engine_id, "d2p", f"rank-{self.tp_rank}",
        )
        capacity = int(float(os.getenv("SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB", "8")) * 1024**3)
        if capacity <= 0:
            raise ValueError("source D→P Host Arena capacity must be positive")
        self.bridge = bridge if bridge is not None else create_remote_host_bridge(
            self.device_pool, client.page_size, self.tp_rank, self.tp_size, "d2p"
        )
        if self.bridge is None:
            raise RuntimeError("multi-node D2P requires remote Host transport")
        self.arena = arena if arena is not None else SharedHostSnapshotArena(
            directory, capacity, backend="memfd"
        )
        if start_thread:
            self._thread = threading.Thread(target=self._worker,
                name=f"agentic-d-source-host-{self.tp_rank}", daemon=True)
            self._thread.start()

    def ensure_grant(self, entry):
        """Called by D background progress; no P allocation round trip."""
        if entry is None or entry.get("source_host_node") != self.config.node_id:
            return entry
        if entry.get("source_host_engine") != self.config.engine_id:
            raise RuntimeError("source Host engine identity mismatch")
        if entry.get("state") not in {"offered", "host_reserved"}:
            return entry
        sid = str(entry["snapshot_id"])
        with self._lock:
            record = self.records.get(sid)
            offer = entry.get("rank_offers", {}).get(str(self.tp_rank), entry)
            byte_size = int(offer.get("byte_size", 0))
            if byte_size <= 0 or byte_size > self.arena.capacity_bytes:
                self.ledger.reject_unclaimed_offer(sid, reason="source_host_snapshot_oversize")
                return self.ledger.get(sid)
            if record is None and not self.arena.can_reserve(byte_size, 1.0):
                self.ledger.set_source_host_pressure_rank(sid, self.config.engine_id,
                    tp_rank=self.tp_rank, required_bytes=byte_size)
                self._pressure = True
                self._required_bytes = max(self._required_bytes, byte_size)
                self._wakeup.set()
                return entry
            claimed = self.ledger.claim_rank(sid, self.owner,
                tp_rank=self.tp_rank, tp_size=self.tp_size)
            if claimed is None:
                return self.ledger.get(sid)
            if record is None:
                try:
                    snapshot = self.arena.create(sid, int(entry["token_count"]), self.device_pool, byte_size)
                except Exception:
                    from sglang.srt.disaggregation.agentic_host_staging import HostStageState
                    # No D2H has started on this rank; peers observe ABORTING
                    # and drain before the original D ownership is released.
                    self.records[sid] = {"snapshot": None, "exported": False, "released": True}
                    self.ledger.transition(sid, HostStageState.ABORTING,
                        owner=self.owner, reason="source_host_extent_allocation_failed")
                    raise
                record = {"snapshot": snapshot, "exported": False, "released": False}
                self.records[sid] = record
            if entry.get("source_host_pressure_ranks", {}).get(str(self.tp_rank)):
                self.ledger.set_source_host_pressure_rank(sid, self.config.engine_id,
                    tp_rank=self.tp_rank, required_bytes=0)
            snapshot = record["snapshot"]
            grant = {"kind": "shared_host_extent", "seq": 0,
                "tp_rank": self.tp_rank, "arena_path": snapshot.path,
                "arena_offset": int(getattr(snapshot, "file_offset", 0)),
                "byte_size": byte_size, "token_count": int(entry["token_count"]),
                "arena_numa_node": self.client.source_numa_node,
                "remote_host_node": self.config.node_id,
                "remote_host_engine": self.config.engine_id}
            if not self.ledger.publish_rank_grant(sid, self.owner, grant,
                tp_rank=self.tp_rank, tp_size=self.tp_size):
                # Retain claimed extent if publication raced an abort; cleanup
                # observes the all-rank writer drain state before freeing it.
                return self.ledger.get(sid)
        return self.ledger.get(sid)

    def export_after_d2h(self, snapshot_id):
        """D2H fence is complete; export before HOST_READY publication."""
        with self._lock:
            record = self.records[snapshot_id]
            if not record["exported"]:
                snapshot = record["snapshot"].materialize()
                record["export_attempted"] = True
                self.bridge.export_snapshot(snapshot_id, snapshot)
                record["exported"] = True
        self._wakeup.set()

    def _release_evicted_host_rank(self, snapshot_id, entry):
        return self._cleanup_one(snapshot_id, entry)

    def _cleanup_one(self, sid, entry):
        from sglang.srt.disaggregation.agentic_host_staging import HostStageState

        state = entry.get("state")
        terminal = {HostStageState.CONSUMED.value, HostStageState.FAILED.value,
                    HostStageState.EVICTING.value, HostStageState.RECOMPUTE_REQUIRED.value}
        if state not in terminal:
            return False
        with self._lock:
            record = self.records.get(sid)
            if record is None:
                return False
            if not record["released"]:
                if not record["exported"] and not (
                    entry.get("writer_drained") or
                    set(entry.get("writer_acks", [])) == set(range(self.tp_size))
                ):
                    # A FAILED label alone is not proof that another rank's
                    # producer DMA stopped touching this complete Host object.
                    return False
                if record.get("export_attempted") and not self.bridge.cleanup_source(sid, entry):
                    return False
                if not self.arena.release(record["snapshot"]):
                    return False
                record["released"] = True
            if not self.ledger.complete_source_host_release_rank(sid, self.owner,
                tp_rank=self.tp_rank, tp_size=self.tp_size):
                return False
            if state == HostStageState.EVICTING.value:
                if not self.ledger.complete_host_eviction_rank(sid, self.owner,
                    tp_rank=self.tp_rank, tp_size=self.tp_size):
                    return False
            self.records.pop(sid, None)
            return True

    def progress(self):
        from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager

        with self._lock:
            ids = tuple(self.records)
        entries = {sid: self.ledger.get(sid) for sid in ids}
        entries = {sid: entry for sid, entry in entries.items() if entry is not None}
        for sid, entry in tuple(entries.items()):
            if (entry.get("state") == "aborting" and entry.get("writer_drained")
                    and not entry.get("h2d_abort_started")):
                from sglang.srt.disaggregation.agentic_host_staging import HostStageState
                self.ledger.transition(sid, HostStageState.FAILED,
                    owner=self.owner, reason="source_host_writer_drained")
                entries[sid] = self.ledger.get(sid)
        # Reuse late application-final cleanup and its exact ownership CAS.
        AgenticPHostStagingManager._progress_final_host_cleanup(self, entries)
        for sid, entry in entries.items():
            self._cleanup_one(sid, entry)
        if self.tp_rank != 0:
            return
        peer_required = max((int(size) for entry in entries.values()
            for size in entry.get("source_host_pressure_ranks", {}).values()), default=0)
        if peer_required:
            self._pressure = True
            self._required_bytes = max(self._required_bytes, peer_required)
        if self.arena.usage() >= .90:
            self._pressure = True
        if not self._pressure:
            return
        target = min(int(self.arena.capacity_bytes * .75),
                     self.arena.capacity_bytes - self._required_bytes)
        remaining = self.arena.used_bytes
        if peer_required:
            # A peer may retain an old extent after this rank has freed its
            # shard. Its pressure must still trigger a leader-only eviction.
            target = min(target, max(0, remaining - peer_required))
        remaining -= sum(int(entry.get("rank_offers", {}).get("0", entry).get("byte_size", 0))
            for entry in entries.values() if entry.get("state") == "evicting")
        candidates = sorted(entries.items(), key=lambda pair:
            (int(pair[1].get("token_count", 0)), float(pair[1].get("created_at", 0))))
        for sid, entry in candidates:
            if remaining <= target:
                break
            if self.ledger.begin_host_eviction(sid, self.owner, tp_size=self.tp_size,
                reason="source_d_host_pressure"):
                rank = entry.get("rank_offers", {}).get("0", entry)
                remaining -= int(rank.get("byte_size", 0))
        if self.arena.used_bytes <= target:
            self._pressure = False
            self._required_bytes = 0

    def _worker(self):
        while not self._stop.is_set():
            self._wakeup.wait(.1)
            self._wakeup.clear()
            try:
                self.progress()
            except Exception:
                logger.exception("Source-local D Host progress failed; retaining Arena ownership")

    def close(self):
        self._stop.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        # No live extents are reclaimed solely because the worker was stopped.
        if not self.records:
            self.arena.close()
