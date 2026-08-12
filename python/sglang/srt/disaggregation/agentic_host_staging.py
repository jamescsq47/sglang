from __future__ import annotations

"""Request-generation D-HBM -> shared P-Host arena for agentic PD serving.

The control plane is deliberately tiny and node-local.  KV bytes never pass
through it: D publishes an offer in /dev/shm, P atomically reserves one complete
snapshot extent in a shared pinned Host arena, and D writes that extent with its
own CUDA D2H engine.  P later restores it with its own H2D engine.  D may release
its source snapshot only after its CUDA event completed and HOST_READY was
committed.  Cold Host snapshots retain the existing Mooncake spill lifecycle.
"""

import fcntl
import hashlib
import json
import logging
import math
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from enum import Enum
from typing import Any, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    SharedSnapshotEvictionController,
    SnapshotManifest,
    SnapshotState,
    page_namespace,
)
from sglang.srt.disaggregation.utils import kv_to_page_indices
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo

logger = logging.getLogger(__name__)
_LEDGER_ENTRY_UNSET = object()


class HostStageState(str, Enum):
    OFFERED = "offered"
    HOST_RESERVED = "host_reserved"
    HOST_WRITING = "host_writing"
    ABORTING = "aborting"
    HOST_READY = "host_ready"
    H2D_LOADING = "h2d_loading"
    SPILLING = "spilling"
    MOONCAKE_READY = "mooncake_ready"
    CONSUMED = "consumed"
    REJECTED = "rejected"
    FAILED = "failed"


_TERMINAL_STATES = {
    HostStageState.CONSUMED.value,
    HostStageState.REJECTED.value,
    HostStageState.FAILED.value,
}

_ALLOWED_STAGE_TRANSITIONS = {
    HostStageState.OFFERED.value: {
        HostStageState.HOST_RESERVED.value,
        HostStageState.REJECTED.value,
        HostStageState.FAILED.value,
    },
    HostStageState.HOST_RESERVED.value: {
        HostStageState.HOST_WRITING.value,
        HostStageState.ABORTING.value,
        HostStageState.REJECTED.value,
        HostStageState.FAILED.value,
    },
    HostStageState.HOST_WRITING.value: {
        HostStageState.ABORTING.value,
        HostStageState.FAILED.value,
    },
    HostStageState.ABORTING.value: {HostStageState.FAILED.value},
    HostStageState.HOST_READY.value: {
        HostStageState.H2D_LOADING.value,
        HostStageState.SPILLING.value,
        HostStageState.CONSUMED.value,
        HostStageState.FAILED.value,
    },
    HostStageState.H2D_LOADING.value: {
        HostStageState.HOST_READY.value,
        HostStageState.CONSUMED.value,
        HostStageState.FAILED.value,
    },
    HostStageState.SPILLING.value: {
        HostStageState.HOST_READY.value,
        HostStageState.MOONCAKE_READY.value,
        HostStageState.FAILED.value,
    },
}


class SharedHostStagingLedger:
    """Flock-protected, idempotent request-generation staging control plane."""

    VERSION = 1

    def __init__(self, path: str):
        if not path:
            raise ValueError("host staging ledger path is required")
        directory = os.path.dirname(path) or "."
        if directory != "/dev/shm" and not directory.startswith("/dev/shm/"):
            raise ValueError("host staging ledger must reside in /dev/shm")
        os.makedirs(directory, exist_ok=True)
        self.path = path
        # The ledger contains page hashes for every live request-generation.
        # Re-decoding the complete JSON document on every P/D scheduler step
        # makes Decode CPU-bound after a few hundred long-lived snapshots.
        # Cache only reads; every mutation performed by this process
        # invalidates immediately, while cross-process changes become visible
        # after this short, configurable control-plane interval.
        self._snapshot_cache_seconds = max(
            0.0,
            float(os.getenv("SGLANG_AGENTIC_KV_LEDGER_CACHE_SECONDS", "0")),
        )
        self._snapshot_cache_at = 0.0
        self._snapshot_cache: dict[str, dict[str, Any]] = {}
        self._snapshot_cache_lock = threading.Lock()
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            file_obj.seek(0)
            raw = file_obj.read()
            if not raw:
                self._write_locked(
                    file_obj,
                    {"version": self.VERSION, "entries": {}, "relays": {}},
                )
            else:
                data = json.loads(raw)
                if data.get("version") != self.VERSION:
                    raise ValueError("unsupported host staging ledger version")
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_locked(file_obj, value: dict[str, Any]) -> None:
        file_obj.seek(0)
        json.dump(value, file_obj, separators=(",", ":"), sort_keys=True)
        file_obj.truncate()
        file_obj.flush()
        # /dev/shm is memory-backed; fsync is intentionally omitted from the hot path.

    def _mutate(self, callback):
        with open(self.path, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            try:
                file_obj.seek(0)
                data = json.loads(file_obj.read() or "{}")
                if data.get("version") != self.VERSION:
                    raise ValueError("corrupt host staging ledger")
                result, changed = callback(data.setdefault("entries", {}))
                if changed:
                    self._write_locked(file_obj, data)
                    with self._snapshot_cache_lock:
                        self._snapshot_cache_at = 0.0
                        self._snapshot_cache = {}
                return result
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    def _mutate_document(self, callback):
        """Mutate entries plus the node-local relay registry atomically."""

        with open(self.path, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            try:
                file_obj.seek(0)
                data = json.loads(file_obj.read() or "{}")
                if data.get("version") != self.VERSION:
                    raise ValueError("corrupt host staging ledger")
                data.setdefault("entries", {})
                data.setdefault("relays", {})
                result, changed = callback(data)
                if changed:
                    self._write_locked(file_obj, data)
                    with self._snapshot_cache_lock:
                        self._snapshot_cache_at = 0.0
                        self._snapshot_cache = {}
                return result
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    def offer(self, entry: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = str(entry["snapshot_id"])

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is not None:
                return dict(current), False
            now = time.time()
            value = dict(entry)
            value.update(
                state=HostStageState.OFFERED.value,
                created_at=float(value.get("created_at", now)),
                updated_at=now,
                grants=[],
                acked_chunks=[],
                sent_chunks=[],
            )
            entries[snapshot_id] = value
            return dict(value), True

        return self._mutate(callback)

    def get(self, snapshot_id: str) -> Optional[dict[str, Any]]:
        def callback(entries):
            value = entries.get(snapshot_id)
            return (None if value is None else dict(value)), False

        return self._mutate(callback)

    def snapshot_entries(self) -> dict[str, dict[str, Any]]:
        """Read all staging states under one shared lock.

        Scheduler callers commonly need the state of tens of snapshots at
        once.  Reading and JSON-decoding the complete ledger separately for
        every snapshot makes control-plane cost quadratic in the number of
        in-flight agent turns and serializes all P/D workers on one flock.
        """

        now = time.monotonic()
        with self._snapshot_cache_lock:
            if (
                self._snapshot_cache_seconds > 0
                and now - self._snapshot_cache_at < self._snapshot_cache_seconds
            ):
                return {
                    key: dict(value) for key, value in self._snapshot_cache.items()
                }
        with open(self.path, "r", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_SH)
            try:
                data = json.loads(file_obj.read() or "{}")
                if data.get("version") != self.VERSION:
                    raise ValueError("corrupt host staging ledger")
                entries = {
                    str(key): dict(value)
                    for key, value in data.get("entries", {}).items()
                }
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        with self._snapshot_cache_lock:
            self._snapshot_cache = entries
            self._snapshot_cache_at = time.monotonic()
        return {key: dict(value) for key, value in entries.items()}

    def register_relay(
        self,
        *,
        relay_id: str,
        pid: int,
        numa_node: int,
        slot_token_count: int,
        slot_count: int,
        d2h_gib_per_second: float,
    ) -> dict[str, Any]:
        """Publish one Arena-local D relay without putting bytes in the ledger."""

        def callback(data):
            relays = data["relays"]
            previous = relays.get(relay_id, {})
            value = {
                "relay_id": str(relay_id),
                "pid": int(pid),
                "numa_node": int(numa_node),
                "slot_token_count": int(slot_token_count),
                "slot_count": int(slot_count),
                "d2h_gib_per_second": float(d2h_gib_per_second),
                "queued_bytes": int(previous.get("queued_bytes", 0)),
                "active_snapshot": previous.get("active_snapshot"),
                "updated_at": time.time(),
            }
            relays[str(relay_id)] = value
            return dict(value), True

        return self._mutate_document(callback)

    def heartbeat_relay(self, relay_id: str, pid: int) -> bool:
        def callback(data):
            relay = data["relays"].get(str(relay_id))
            if relay is None or int(relay.get("pid", -1)) != int(pid):
                return False, False
            relay["updated_at"] = time.time()
            return True, True

        return bool(self._mutate_document(callback))

    def list_relays(self) -> list[dict[str, Any]]:
        def callback(data):
            values = [dict(value) for value in data["relays"].values()]
            values.sort(key=lambda item: item["relay_id"])
            return values, False

        return self._mutate_document(callback)

    def assign_transfer_path(
        self,
        snapshot_id: str,
        *,
        source_pid: int,
        source_numa_node: int,
        arena_numa_node: int,
        direct_cross_numa_gib_per_second: float,
        nvlink_gib_per_second: float,
        relay_stale_seconds: float,
    ) -> Optional[dict[str, Any]]:
        """Atomically choose an Arena-local relay or the direct D2H fallback.

        Selection uses queued bytes divided by each relay's measured D2H
        bandwidth.  A relay is used only when its predicted completion is
        earlier than writing the same snapshot directly across NUMA.
        """

        now = time.time()

        def callback(data):
            entries = data["entries"]
            current = entries.get(snapshot_id)
            if current is None or int(current.get("d_pid", -1)) != int(source_pid):
                return None, False
            if current.get("state") != HostStageState.HOST_WRITING.value:
                return dict(current), False
            if current.get("write_mode"):
                return dict(current), False

            byte_size = int(current.get("byte_size", 0))
            # A source already local to the Arena owns the best PCIe endpoint;
            # an extra NVLink hop cannot improve it.
            if int(source_numa_node) == int(arena_numa_node):
                current["write_mode"] = "direct_local"
                current["updated_at"] = now
                return dict(current), True

            direct_bw = max(float(direct_cross_numa_gib_per_second), 1e-6)
            direct_seconds = byte_size / (direct_bw * (1024**3))
            best = None
            for relay in data["relays"].values():
                if int(relay.get("pid", -1)) == int(source_pid):
                    continue
                if int(relay.get("numa_node", -1)) != int(arena_numa_node):
                    continue
                if now - float(relay.get("updated_at", 0.0)) > relay_stale_seconds:
                    continue
                slot_tokens = int(relay.get("slot_token_count", 0))
                if slot_tokens <= 0:
                    continue
                relay_bw = max(float(relay.get("d2h_gib_per_second", 0.0)), 1e-6)
                queued = max(0, int(relay.get("queued_bytes", 0)))
                predicted = (queued + byte_size) / (relay_bw * (1024**3))
                predicted += byte_size / (
                    max(float(nvlink_gib_per_second), 1e-6) * (1024**3)
                )
                candidate = (predicted, queued, str(relay["relay_id"]), relay)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate

            if best is None or best[0] >= direct_seconds:
                current["write_mode"] = "direct_cross_numa"
                current["direct_predicted_seconds"] = direct_seconds
                current["updated_at"] = now
                return dict(current), True

            relay = best[3]
            relay["queued_bytes"] = int(relay.get("queued_bytes", 0)) + byte_size
            relay["updated_at"] = now
            slot_tokens = int(relay["slot_token_count"])
            current.update(
                write_mode="relay",
                relay_id=str(relay["relay_id"]),
                relay_pid=int(relay["pid"]),
                relay_job_state="queued",
                relay_completed_tokens=0,
                relay_total_chunks=math.ceil(
                    int(current["token_count"]) / slot_tokens
                ),
                relay_predicted_seconds=float(best[0]),
                direct_predicted_seconds=direct_seconds,
                updated_at=now,
            )
            return dict(current), True

        return self._mutate_document(callback)

    def claim_relay_job(self, relay_id: str, pid: int) -> Optional[dict[str, Any]]:
        def callback(data):
            relay = data["relays"].get(str(relay_id))
            if relay is None or int(relay.get("pid", -1)) != int(pid):
                return None, False
            active = relay.get("active_snapshot")
            if active:
                current = data["entries"].get(active)
                return (None if current is None else dict(current)), False
            queued = [
                value
                for value in data["entries"].values()
                if value.get("relay_id") == str(relay_id)
                and value.get("relay_job_state") == "queued"
                and value.get("state") == HostStageState.HOST_WRITING.value
            ]
            if not queued:
                return None, False
            queued.sort(key=lambda item: (item.get("created_at", 0.0), item["snapshot_id"]))
            current = queued[0]
            current["relay_job_state"] = "claimed"
            current["updated_at"] = time.time()
            relay["active_snapshot"] = current["snapshot_id"]
            relay["updated_at"] = time.time()
            return dict(current), True

        return self._mutate_document(callback)

    def relay_prepare_chunk(
        self,
        snapshot_id: str,
        *,
        relay_id: str,
        seq: int,
        start_token: int,
        token_count: int,
        room: int,
    ) -> bool:
        def callback(data):
            current = data["entries"].get(snapshot_id)
            if (
                current is None
                or current.get("relay_id") != str(relay_id)
                or current.get("relay_job_state") not in {"claimed", "transferring"}
                or current.get("relay_chunk") is not None
            ):
                return False, False
            current["relay_job_state"] = "transferring"
            current["relay_chunk"] = {
                "seq": int(seq),
                "start_token": int(start_token),
                "token_count": int(token_count),
                "room": int(room),
                "state": "receiver_ready",
            }
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate_document(callback))

    def relay_mark_source_sent(self, snapshot_id: str, seq: int, source_pid: int) -> bool:
        def callback(data):
            current = data["entries"].get(snapshot_id)
            chunk = None if current is None else current.get("relay_chunk")
            if (
                current is None
                or int(current.get("d_pid", -1)) != int(source_pid)
                or chunk is None
                or int(chunk.get("seq", -1)) != int(seq)
            ):
                return False, False
            if chunk.get("state") == "source_sent":
                return True, False
            if chunk.get("state") != "receiver_ready":
                return False, False
            chunk["state"] = "source_sent"
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate_document(callback))

    def relay_complete_chunk(self, snapshot_id: str, relay_id: str, seq: int) -> bool:
        """Commit one relay D2H chunk; the final chunk publishes HOST_READY."""

        def callback(data):
            current = data["entries"].get(snapshot_id)
            chunk = None if current is None else current.get("relay_chunk")
            if (
                current is None
                or current.get("relay_id") != str(relay_id)
                or chunk is None
                or int(chunk.get("seq", -1)) != int(seq)
            ):
                return False, False
            completed = int(current.get("relay_completed_tokens", 0)) + int(
                chunk["token_count"]
            )
            current["relay_completed_tokens"] = completed
            current["relay_chunk"] = None
            current["updated_at"] = time.time()
            if completed >= int(current["token_count"]):
                current["relay_job_state"] = "complete"
                current["sent_chunks"] = list(range(int(current["relay_total_chunks"])))
                current["acked_chunks"] = list(range(int(current["relay_total_chunks"])))
                current["state"] = HostStageState.HOST_READY.value
                relay = data["relays"].get(str(relay_id))
                if relay is not None:
                    relay["queued_bytes"] = max(
                        0,
                        int(relay.get("queued_bytes", 0))
                        - int(current.get("byte_size", 0)),
                    )
                    relay["active_snapshot"] = None
                    relay["updated_at"] = time.time()
            return True, True

        return bool(self._mutate_document(callback))

    def relay_fail_to_direct(
        self, snapshot_id: str, relay_id: str, reason: str
    ) -> bool:
        """After relay DMA is drained, return ownership to the source D."""

        def callback(data):
            current = data["entries"].get(snapshot_id)
            if current is None or current.get("relay_id") != str(relay_id):
                return False, False
            relay = data["relays"].get(str(relay_id))
            if relay is not None:
                relay["queued_bytes"] = max(
                    0,
                    int(relay.get("queued_bytes", 0))
                    - int(current.get("byte_size", 0)),
                )
                if relay.get("active_snapshot") == snapshot_id:
                    relay["active_snapshot"] = None
                relay["updated_at"] = time.time()
            current["write_mode"] = "direct_cross_numa_fallback"
            current["relay_job_state"] = "failed"
            current["relay_chunk"] = None
            current["reason"] = str(reason)[:256]
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate_document(callback))

    def list_state(self, *states: HostStageState) -> list[dict[str, Any]]:
        wanted = {state.value for state in states}

        def callback(entries):
            values = [dict(v) for v in entries.values() if v.get("state") in wanted]
            values.sort(key=lambda item: (item.get("created_at", 0.0), item["snapshot_id"]))
            return values, False

        return self._mutate(callback)

    def claim(self, snapshot_id: str, owner: str) -> Optional[dict[str, Any]]:
        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("state") != HostStageState.OFFERED.value:
                return None, False
            current["state"] = HostStageState.HOST_RESERVED.value
            current["p_owner"] = owner
            current["updated_at"] = time.time()
            return dict(current), True

        return self._mutate(callback)

    def publish_grants(self, snapshot_id: str, owner: str, grants: list[dict[str, Any]]) -> bool:
        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("p_owner") != owner:
                return False, False
            if current.get("state") not in {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
            }:
                return False, False
            current["state"] = HostStageState.HOST_WRITING.value
            current["grants"] = [dict(grant) for grant in grants]
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback))

    def complete_host_write(self, snapshot_id: str, d_pid: int) -> bool:
        """Atomically publish a complete D-written shared-Host snapshot."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or int(current.get("d_pid", -1)) != int(d_pid):
                return False, False
            if current.get("state") == HostStageState.HOST_READY.value:
                return True, False
            if current.get("state") != HostStageState.HOST_WRITING.value:
                return False, False
            grants = current.get("grants", [])
            if len(grants) != 1 or grants[0].get("kind") != "shared_host_extent":
                return False, False
            current["sent_chunks"] = [0]
            current["acked_chunks"] = [0]
            current["state"] = HostStageState.HOST_READY.value
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback))

    def mark_writer_drained(self, snapshot_id: str, d_pid: int) -> bool:
        """ACK that no D-side DMA can still target an aborting extent."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or int(current.get("d_pid", -1)) != int(d_pid):
                return False, False
            if current.get("state") != HostStageState.ABORTING.value:
                return False, False
            if current.get("writer_drained"):
                return True, False
            current["writer_drained"] = True
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback))

    def fail_host_write(self, snapshot_id: str, d_pid: int, reason: str) -> bool:
        """Fail closed after D has stopped touching the shared extent."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or int(current.get("d_pid", -1)) != int(d_pid):
                return False, False
            if current.get("state") not in {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
                HostStageState.ABORTING.value,
            }:
                return False, False
            current["state"] = HostStageState.ABORTING.value
            current["writer_drained"] = True
            current["reason"] = str(reason)[:256]
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback))

    def mark_sent(self, snapshot_id: str, seq: int) -> bool:
        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("state") != HostStageState.HOST_WRITING.value:
                return False, False
            sent = set(int(x) for x in current.get("sent_chunks", []))
            if seq in sent:
                return True, False
            sent.add(int(seq))
            current["sent_chunks"] = sorted(sent)
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback))

    def ack_chunk(self, snapshot_id: str, owner: str, seq: int) -> bool:
        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("p_owner") != owner:
                return False, False
            if current.get("state") != HostStageState.HOST_WRITING.value:
                return False, False
            acked = set(int(x) for x in current.get("acked_chunks", []))
            if seq in acked:
                return True, False
            acked.add(int(seq))
            current["acked_chunks"] = sorted(acked)
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback))

    def mark_host_ready(self, snapshot_id: str, owner: str, total_chunks: int) -> bool:
        """Commit visibility only when every expected chunk has a D2H ACK."""

        expected = set(range(int(total_chunks)))

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("p_owner") != owner:
                return False, False
            if current.get("state") != HostStageState.HOST_WRITING.value:
                return False, False
            acked = set(int(x) for x in current.get("acked_chunks", []))
            if acked != expected:
                return False, False
            current["state"] = HostStageState.HOST_READY.value
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback))

    def transition(
        self,
        snapshot_id: str,
        state: HostStageState,
        *,
        owner: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> bool:
        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or (owner is not None and current.get("p_owner") != owner):
                return False, False
            current_state = current.get("state")
            if state.value == current_state:
                return True, False
            if state.value not in _ALLOWED_STAGE_TRANSITIONS.get(current_state, set()):
                return False, False
            current["state"] = state.value
            current["updated_at"] = time.time()
            if reason:
                current["reason"] = str(reason)[:256]
            return True, True

        return bool(self._mutate(callback))

    def prune(
        self,
        older_than_seconds: float = 600.0,
        consumed_older_than_seconds: float = 5.0,
    ) -> None:
        cutoff = time.time() - max(0.0, older_than_seconds)
        consumed_cutoff = time.time() - max(0.0, consumed_older_than_seconds)

        def callback(entries):
            doomed = [
                key
                for key, value in entries.items()
                if (
                    value.get("state") == HostStageState.CONSUMED.value
                    and float(value.get("updated_at", 0.0)) < consumed_cutoff
                )
                or (
                    value.get("state") in _TERMINAL_STATES
                    and float(value.get("updated_at", 0.0)) < cutoff
                )
            ]
            for key in doomed:
                entries.pop(key, None)
            return None, bool(doomed)

        self._mutate(callback)


class LayerFirstD2HStaging:
    """Small reusable HBM gather buffer feeding contiguous PCIe D2H DMA."""

    def __init__(self, device_pool, token_capacity: int):
        self.token_capacity = max(1, int(token_capacity))
        self.layer_num = int(device_pool.layer_num)
        self.head_num = int(device_pool.head_num)
        self.head_dim = int(device_pool.head_dim)
        self.v_head_dim = int(getattr(device_pool, "v_head_dim", self.head_dim))
        self.dtype = device_pool.store_dtype
        self.device = device_pool.device
        self.k_buffer = [
            torch.empty(
                (self.token_capacity, self.head_num, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.empty(
                (self.token_capacity, self.head_num, self.v_head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.k_data_ptrs = torch.tensor(
            [value.data_ptr() for value in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [value.data_ptr() for value in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.local_indices = torch.arange(
            self.token_capacity, dtype=torch.int64, device=self.device
        )

    @property
    def byte_size(self) -> int:
        return (
            2
            * self.token_capacity
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )


class SharedMHAHostSnapshot:
    """One request-generation extent mapped by both P and its owning D.

    The hot on-disk shape is layer-first and snapshot-local:
    ``[K/V, layer, token, head, head_dim]``.  This lets D gather scattered KV
    into a small reusable HBM slot and then use contiguous PCIe DMA into Host.
    The file lives in tmpfs, and the
    mapping is CUDA-registered independently in each process.  It therefore
    contains exactly one physical Host copy even though P and D have different
    virtual addresses.
    """

    def __init__(
        self,
        *,
        path: str,
        token_count: int,
        device_pool,
        byte_size: int,
        create: bool,
    ):
        if not path.startswith("/dev/shm/"):
            raise ValueError("shared Host snapshot must reside in /dev/shm")
        if not hasattr(device_pool, "k_buffer") or not hasattr(device_pool, "v_buffer"):
            raise ValueError("shared Host arena V1 requires an MHA KV pool")
        self.path = path
        self.token_count = int(token_count)
        self.device_pool = device_pool
        self.layer_num = int(device_pool.layer_num)
        self.head_num = int(device_pool.head_num)
        self.head_dim = int(device_pool.head_dim)
        self.v_head_dim = int(getattr(device_pool, "v_head_dim", self.head_dim))
        if self.v_head_dim != self.head_dim:
            raise ValueError("shared Host arena V1 requires equal K/V head dimensions")
        self.dtype = device_pool.store_dtype
        self.item_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.item_size * self.layer_num
        expected_bytes = 2 * self.token_count * self.layout_dim
        if int(byte_size) != expected_bytes:
            raise ValueError(
                f"shared Host extent size mismatch: expected={expected_bytes} actual={byte_size}"
            )
        self.byte_size = expected_bytes
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, self.byte_size)
            elif os.fstat(fd).st_size != self.byte_size:
                raise ValueError("shared Host extent file has the wrong size")
            self.mapping = mmap.mmap(fd, self.byte_size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        raw = torch.frombuffer(self.mapping, dtype=torch.uint8, count=self.byte_size)
        self.kv_buffer = raw.view(self.dtype).view(
            2,
            self.layer_num,
            self.token_count,
            self.head_num,
            self.head_dim,
        )
        result = torch.cuda.cudart().cudaHostRegister(
            self.kv_buffer.data_ptr(), self.byte_size, 0
        )
        if result != torch.cuda.cudart().cudaError.success:
            self.kv_buffer = None
            raw = None
            self.mapping.close()
            raise RuntimeError(f"cudaHostRegister failed: {result}")
        self._raw = raw
        self.k_data_ptrs = torch.tensor(
            [self.kv_buffer[0, layer].data_ptr() for layer in range(self.layer_num)],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [self.kv_buffer[1, layer].data_ptr() for layer in range(self.layer_num)],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self._closed = False

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def start_backup_from_device(self, source_indices, stream, *, staging=None):
        """Launch D-HBM -> this Host extent on the D GPU's own stream."""

        if len(source_indices) != self.token_count:
            raise ValueError("source token count does not match shared Host extent")
        return self.start_backup_range_from_device(
            source_indices, destination_start=0, stream=stream, staging=staging
        )

    def start_backup_range_from_device(
        self, source_indices, *, destination_start: int, stream, staging=None
    ):
        """Launch one relay staging chunk into a range of this Host extent."""

        from sgl_kernel.kvcacheio import transfer_kv_all_layer

        destination_start = int(destination_start)
        if destination_start < 0 or destination_start + len(source_indices) > self.token_count:
            raise ValueError("relay chunk falls outside shared Host extent")
        # req_to_token uses int32 to save HBM, while the kvcacheio gather
        # kernel deliberately accepts only int64 indices.  Normalize here so
        # every caller (including the scheduler's native req_to_token view)
        # gets the same contract.
        original_source_indices = source_indices
        if staging is None:
            staging = LayerFirstD2HStaging(
                self.device_pool,
                int(os.getenv("SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS", "1024")),
            )
        start_event = torch.cuda.Event(enable_timing=True)
        event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            if not source_indices.is_cuda or source_indices.dtype != torch.int64:
                source_indices = source_indices.to(
                    device=self.device_pool.device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
            start_event.record(stream)
            for start in range(0, len(source_indices), staging.token_capacity):
                count = min(staging.token_capacity, len(source_indices) - start)
                source_chunk = source_indices[start : start + count]
                local_indices = staging.local_indices[:count]
                transfer_kv_all_layer(
                    src_k_layers=self.device_pool.k_data_ptrs,
                    dst_k_layers=staging.k_data_ptrs,
                    src_v_layers=self.device_pool.v_data_ptrs,
                    dst_v_layers=staging.v_data_ptrs,
                    src_indices=source_chunk,
                    dst_indices=local_indices,
                    item_size=self.item_size,
                    num_layers=self.layer_num,
                    block_quota=8,
                    num_warps_per_block=32,
                )
                host_start = destination_start + start
                host_end = host_start + count
                for layer_id in range(self.layer_num):
                    self.k_buffer[layer_id, host_start:host_end].copy_(
                        staging.k_buffer[layer_id][:count], non_blocking=True
                    )
                    self.v_buffer[layer_id, host_start:host_end].copy_(
                        staging.v_buffer[layer_id][:count], non_blocking=True
                    )
            event.record(stream)
            source_indices.record_stream(stream)
            if hasattr(original_source_indices, "record_stream"):
                original_source_indices.record_stream(stream)
        self._last_d2h_start_event = start_event
        return event, (source_indices, original_source_indices, staging, start_event)

    def start_load_to_device(
        self,
        device_indices,
        stream,
        *,
        chunk_tokens: int = 4096,
        staging=None,
    ):
        """Launch this Host extent -> P-HBM on the P GPU's own stream.

        Do not let the scatter kernel dereference CUDA-registered mmap
        addresses directly.  That UVA path is fast in an isolated benchmark,
        but under concurrent Prefill it can enter replayable-fault handling
        and has produced illegal-address failures.  Mirror the stable D2H
        path instead: contiguous Host->device DMA into one reusable layer-first
        staging buffer, followed by a device-only scatter into the KV pool.
        All users share one H2D stream, so reusing the buffer across queued
        snapshots is safe by stream ordering.
        """

        from sgl_kernel.kvcacheio import transfer_kv_all_layer

        if len(device_indices) != self.token_count:
            raise ValueError("destination token count does not match shared Host extent")
        original_device_indices = device_indices
        chunk_tokens = max(1, int(chunk_tokens))
        if staging is None:
            staging = LayerFirstD2HStaging(self.device_pool, chunk_tokens)
        if staging.token_capacity < chunk_tokens:
            raise ValueError("H2D staging buffer is smaller than chunk_tokens")
        start_event = torch.cuda.Event(enable_timing=True)
        event = torch.cuda.Event(enable_timing=True)
        copy_refs = [device_indices, staging]
        with torch.cuda.stream(stream):
            if not device_indices.is_cuda or device_indices.dtype != torch.int64:
                device_indices = device_indices.to(
                    device=self.device_pool.device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
            start_event.record(stream)
            # Bound each DMA/scatter launch while retaining the layer-first
            # Host layout.  The PCIe leg is contiguous for every layer; only
            # the device-to-device leg touches scattered KV-pool indices.
            for start in range(0, self.token_count, chunk_tokens):
                end = min(start + chunk_tokens, self.token_count)
                count = end - start
                source_indices = staging.local_indices[:count]
                destination_chunk = device_indices[start:end]
                for layer_id in range(self.layer_num):
                    staging.k_buffer[layer_id][:count].copy_(
                        self.k_buffer[layer_id, start:end], non_blocking=True
                    )
                    staging.v_buffer[layer_id][:count].copy_(
                        self.v_buffer[layer_id, start:end], non_blocking=True
                    )
                transfer_kv_all_layer(
                    src_k_layers=staging.k_data_ptrs,
                    dst_k_layers=self.device_pool.k_data_ptrs,
                    src_v_layers=staging.v_data_ptrs,
                    dst_v_layers=self.device_pool.v_data_ptrs,
                    src_indices=source_indices,
                    dst_indices=destination_chunk,
                    item_size=self.item_size,
                    num_layers=self.layer_num,
                    block_quota=4,
                    num_warps_per_block=32,
                )
                source_indices.record_stream(stream)
                destination_chunk.record_stream(stream)
                copy_refs.extend((source_indices, destination_chunk))
            event.record(stream)
            device_indices.record_stream(stream)
            if hasattr(original_device_indices, "record_stream"):
                original_device_indices.record_stream(stream)
        self._last_h2d_start_event = start_event
        copy_refs.extend((original_device_indices, start_event))
        return event, tuple(copy_refs)

    def copy_into_hicache(self, host_pool, host_indices, page_size: int) -> None:
        """CPU copy used only when a cold extent is selected for Mooncake."""

        if len(host_indices) != self.token_count:
            raise ValueError("Mooncake spill allocation has the wrong size")
        for start in range(0, self.token_count, int(page_size)):
            destination = int(host_indices[start].item())
            # HiCache's page-first pool expects [K/V, token, layer, head, dim].
            page = (
                self.kv_buffer[:, :, start : start + int(page_size)]
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .flatten()
            )
            host_pool.set_from_flat_data_page(destination, page)

    def close(self, *, unlink: bool = False) -> None:
        if self._closed:
            return
        result = torch.cuda.cudart().cudaHostUnregister(self.kv_buffer.data_ptr())
        if result != torch.cuda.cudart().cudaError.success:
            raise RuntimeError(f"cudaHostUnregister failed: {result}")
        self.k_data_ptrs = None
        self.v_data_ptrs = None
        self.kv_buffer = None
        self._raw = None
        self.mapping.close()
        self._closed = True
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


class LazySharedMHAHostSnapshot:
    """A granted tmpfs extent whose P-side pinned mapping is built later.

    Creating and CUDA-registering a multi-GiB mapping can take seconds under
    burst load.  D needs only an existing file of the correct size in order to
    open its own mapping and start D2H, so coupling P registration to grant
    publication unnecessarily pins the complete source KV on D.  This object
    creates the sparse tmpfs extent immediately and materializes P's mapping
    independently after D has begun writing it.
    """

    def __init__(self, *, path: str, token_count: int, device_pool, byte_size: int):
        if not path.startswith("/dev/shm/"):
            raise ValueError("shared Host snapshot must reside in /dev/shm")
        self.path = path
        self.token_count = int(token_count)
        self.device_pool = device_pool
        self.byte_size = int(byte_size)
        self._materialized = None
        self._closed = False
        self._lock = threading.Lock()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, self.byte_size)
        finally:
            os.close(fd)

    def materialize(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot materialize a released Host extent")
            if self._materialized is None:
                self._materialized = SharedMHAHostSnapshot(
                    path=self.path,
                    token_count=self.token_count,
                    device_pool=self.device_pool,
                    byte_size=self.byte_size,
                    create=False,
                )
            return self

    def __getattr__(self, name):
        materialized = object.__getattribute__(self, "_materialized")
        if materialized is None:
            raise RuntimeError(
                f"P Host extent is not materialized; cannot access {name}"
            )
        return getattr(materialized, name)

    def close(self, *, unlink: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            if self._materialized is not None:
                self._materialized.close(unlink=unlink)
            elif unlink:
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
            self._closed = True


class SharedHostSnapshotArena:
    """P-owned capacity and lifecycle for snapshot-scoped shared extents."""

    def __init__(self, directory: str, capacity_bytes: int):
        if not directory.startswith("/dev/shm/"):
            raise ValueError("shared Host arena must reside in /dev/shm")
        self.directory = directory.rstrip("/")
        self.capacity_bytes = int(capacity_bytes)
        if self.capacity_bytes <= 0:
            raise ValueError("shared Host arena capacity must be positive")
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        self.used_bytes = 0
        self._lock = threading.Lock()

    def path_for(self, snapshot_id: str) -> str:
        digest = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
        return os.path.join(self.directory, f"{digest}.kv")

    def can_reserve(self, byte_size: int, hard_watermark: float) -> bool:
        with self._lock:
            return (
                self.used_bytes + int(byte_size)
                <= int(self.capacity_bytes * float(hard_watermark))
            )

    def create(self, snapshot_id: str, token_count: int, device_pool, byte_size: int):
        snapshot = LazySharedMHAHostSnapshot(
            path=self.path_for(snapshot_id),
            token_count=token_count,
            device_pool=device_pool,
            byte_size=byte_size,
        )
        with self._lock:
            self.used_bytes += snapshot.byte_size
        return snapshot

    def release(self, snapshot: SharedMHAHostSnapshot) -> None:
        if snapshot._closed:
            return
        byte_size = snapshot.byte_size
        snapshot.close(unlink=True)
        with self._lock:
            self.used_bytes = max(0, self.used_bytes - byte_size)

    def usage(self) -> float:
        with self._lock:
            return self.used_bytes / max(1, self.capacity_bytes)


class AgenticPHostStagingManager:
    """P-side shared Host arena, demand restore, and cold-spill manager."""

    def _get_state_lock(self):
        # A few focused lifecycle tests construct the manager with __new__ to
        # avoid allocating CUDA buffers.  Lazy creation also keeps those
        # lightweight control-plane uses backward compatible.
        lock = getattr(self, "_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._state_lock = lock
        return lock

    def __init__(
        self,
        *,
        ledger: SharedHostStagingLedger,
        runtime,
        token_allocator,
        cache_controller,
        tree_cache,
        page_size: int,
        arena_directory: str,
        arena_capacity_bytes: int,
        high_watermark: float = 0.80,
        low_watermark: float = 0.70,
        hard_watermark: float = 0.90,
        arena_numa_node: int = -1,
        expected_tool_seconds: Optional[dict[str, float]] = None,
        eviction_controller: Optional[SharedSnapshotEvictionController] = None,
    ):
        if not (0 < low_watermark <= high_watermark < hard_watermark <= 1):
            raise ValueError("host watermarks must satisfy 0 < low <= high < hard <= 1")
        self.ledger = ledger
        self.runtime = runtime
        self.token_allocator = token_allocator
        self.cache_controller = cache_controller
        self.tree_cache = tree_cache
        self.host_pool = cache_controller.mem_pool_host
        self.device_pool = token_allocator.get_kvcache()
        self.page_size = int(page_size)
        self.owner = f"p:{os.getpid()}"
        self.high_watermark = float(high_watermark)
        self.low_watermark = float(low_watermark)
        self.hard_watermark = float(hard_watermark)
        self.arena_numa_node = int(arena_numa_node)
        self.expected_tool_seconds = expected_tool_seconds or {}
        self.eviction_controller = eviction_controller
        self.arena = SharedHostSnapshotArena(
            arena_directory, int(arena_capacity_bytes)
        )
        # Kept for scheduler runtime accounting compatibility.  The new slow
        # path reserves no fixed P-HBM staging slots.
        self.reserved_hbm_bytes = 0
        self.reserved_token_count = 0
        self.active: dict[str, dict[str, Any]] = {}
        self.aborting: dict[str, dict[str, Any]] = {}
        self.host_ready: dict[str, dict[str, Any]] = {}
        self.loads: dict[str, dict[str, Any]] = {}
        self.spills: dict[str, dict[str, Any]] = {}
        self._ledger_entries_cache: dict[str, dict[str, Any]] = {}
        self.max_h2d_inflight = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT", "1"))
        )
        self.h2d_chunk_tokens = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS", "4096"))
        )
        # Slow ingress is launched by each D on its own CUDA stream.  P owns
        # only the latency-sensitive demand H2D stream.
        current_device = torch.cuda.current_device()
        self._h2d_stream = torch.cuda.Stream(device=current_device, priority=-1)
        self._h2d_staging = LayerFirstD2HStaging(
            self.device_pool, self.h2d_chunk_tokens
        )
        self._spill_threads: dict[str, threading.Thread] = {}
        self._spilling_pressure = False
        self._last_prune = 0.0
        self._next_poll_at = 0.0
        self._poll_interval = max(
            0.0,
            float(os.getenv("SGLANG_AGENTIC_KV_P_CONTROL_POLL_SECONDS", "0")),
        )
        # Filesystem-ledger discovery, Host-arena admission, D2H completion
        # discovery, and Mooncake spill bookkeeping must not depend on the P
        # scheduler returning from a (potentially long) Prefill forward.  In
        # particular, D keeps the complete source KV pinned until P publishes
        # a shared-Host grant and observes HOST_READY.  Progressing this work
        # from the scheduler created a feedback loop in which busy P delayed D
        # release, shrinking Decode batches and eventually starving P again.
        #
        # The worker never touches the GPU token allocator, Radix cache, or a
        # request object.  Those ownership-changing operations remain in
        # gate_request() on the scheduler thread.
        self._state_lock = threading.RLock()
        self._async_control = os.getenv(
            "SGLANG_AGENTIC_KV_P_ASYNC_CONTROL", "1"
        ).lower() not in {"0", "false", "no", "off"}
        self._control_interval = max(
            0.001,
            float(
                os.getenv(
                    "SGLANG_AGENTIC_KV_P_ASYNC_CONTROL_INTERVAL_SECONDS",
                    "0.005",
                )
            ),
        )
        self._admission_batch = max(
            1,
            int(os.getenv("SGLANG_AGENTIC_KV_P_HOST_ADMISSION_BATCH", "16")),
        )
        self._materialize_workers = max(
            1,
            int(
                os.getenv(
                    "SGLANG_AGENTIC_KV_P_HOST_MATERIALIZE_WORKERS", "4"
                )
            ),
        )
        self._materialize_pool = ThreadPoolExecutor(
            max_workers=self._materialize_workers,
            thread_name_prefix=f"agentic-p-map-{os.getpid()}",
        )
        self._control_wakeup = threading.Event()
        self._control_cycles = 0
        self._control_errors = 0
        self._control_total_seconds = 0.0
        self._control_max_seconds = 0.0
        self._control_last_stats = time.monotonic()
        self._control_thread = None
        logger.info(
            "Agentic shared Host arena enabled directory=%s capacity_gib=%.1f "
            "reserved_hbm_mib=0 h2d_priority=%d h2d_max_inflight=%d "
            "h2d_chunk_tokens=%d",
            self.arena.directory,
            self.arena.capacity_bytes / (1024**3),
            self._h2d_stream.priority,
            self.max_h2d_inflight,
            self.h2d_chunk_tokens,
        )
        if self._async_control:
            self._control_thread = threading.Thread(
                target=self._control_worker,
                name=f"agentic-p-control-{os.getpid()}",
                daemon=True,
            )
            self._control_thread.start()
            logger.info(
                "Agentic P async control enabled interval_ms=%.3f "
                "admission_batch=%d materialize_workers=%d",
                self._control_interval * 1000.0,
                self._admission_batch,
                self._materialize_workers,
            )

    def _host_usage(self) -> float:
        return self.arena.usage()

    def _can_admit(self, byte_size: int) -> bool:
        return self.arena.can_reserve(byte_size, self.hard_watermark)

    def _reject(self, offer: dict[str, Any], reason: str) -> None:
        self.ledger.transition(
            offer["snapshot_id"], HostStageState.REJECTED, owner=self.owner, reason=reason
        )
        logger.warning("Agentic host staging rejected %s: %s", offer["snapshot_id"], reason)

    def _admit_one(self, ledger_entries=None) -> bool:
        if ledger_entries is None:
            offers = self.ledger.list_state(HostStageState.OFFERED)
        else:
            offers = [
                dict(value)
                for value in ledger_entries.values()
                if value.get("state") == HostStageState.OFFERED.value
            ]
            offers.sort(
                key=lambda item: (item.get("created_at", 0.0), item["snapshot_id"])
            )
        if self.arena_numa_node >= 0:
            offers = [
                offer
                for offer in offers
                if int(offer.get("arena_numa_node", -1)) == self.arena_numa_node
            ]
        if not offers:
            return False
        offer = offers[0]
        claimed = self.ledger.claim(offer["snapshot_id"], self.owner)
        if claimed is None:
            return False
        token_count = int(claimed["token_count"])
        if token_count <= 0 or token_count % self.page_size:
            self._reject(claimed, "unaligned_token_count")
            return True
        byte_size = int(claimed.get("byte_size", 0))
        if byte_size <= 0 or not self._can_admit(byte_size):
            # Do not accept a partial snapshot.  Spill is progressed first and
            # D retains its complete HBM copy while this offer is rejected.
            self._reject(claimed, "p_host_hard_watermark")
            return True
        try:
            snapshot = self.arena.create(
                claimed["snapshot_id"], token_count, self.device_pool, byte_size
            )
        except Exception:
            logger.exception("Failed to allocate shared Host extent for %s", claimed["snapshot_id"])
            self._reject(claimed, "shared_host_extent_allocation_failed")
            return True
        with self._get_state_lock():
            self.active[claimed["snapshot_id"]] = {
                "offer": claimed,
                "snapshot": snapshot,
                "loading": False,
            }
        grant = {
            "kind": "shared_host_extent",
            "seq": 0,
            "arena_path": snapshot.path,
            "byte_size": snapshot.byte_size,
            "token_count": snapshot.token_count,
        }
        if not self.ledger.publish_grants(
            claimed["snapshot_id"], self.owner, [grant]
        ):
            with self._get_state_lock():
                self.active.pop(claimed["snapshot_id"], None)
            self.arena.release(snapshot)
            self._reject(claimed, "shared_host_grant_publish_failed")
        return True

    def _admit_batch(self, ledger_entries=None) -> int:
        """Admit several complete snapshots per control cycle.

        The old one-offer-per-scheduler-tick rule was visible as tens of D
        requests retaining HBM while P had abundant Host capacity.  Each
        admission still owns one complete request-generation extent; this is
        batching of control operations, not partial KV admission.
        """

        if ledger_entries is None:
            offers = self.ledger.list_state(HostStageState.OFFERED)
        else:
            offers = [
                dict(value)
                for value in ledger_entries.values()
                if value.get("state") == HostStageState.OFFERED.value
            ]
            offers.sort(
                key=lambda item: (item.get("created_at", 0.0), item["snapshot_id"])
            )
        if self.arena_numa_node >= 0:
            offers = [
                offer
                for offer in offers
                if int(offer.get("arena_numa_node", -1)) == self.arena_numa_node
            ]
        admitted = 0
        # Pass a single-entry view so _admit_one preserves its validation and
        # atomic claim/publish behavior without repeatedly sorting the ledger.
        for offer in offers[: self._admission_batch]:
            if self._admit_one({offer["snapshot_id"]: offer}):
                admitted += 1
        return admitted

    def _release_record(self, record: dict[str, Any]) -> None:
        snapshot = record.pop("snapshot", None)
        if snapshot is not None:
            self.arena.release(snapshot)

    def _fail_active(self, snapshot_id: str, reason: str, *, free_host: bool = True) -> None:
        with self._get_state_lock():
            entry = self.active.pop(snapshot_id, None)
        if entry is not None:
            entry["failure_reason"] = reason
            entry["free_host_on_abort"] = bool(free_host)
            with self._get_state_lock():
                self.aborting[snapshot_id] = entry
            # ABORTING is visible before unlink.  D drains its local D2H event
            # and unregisters the mapping, then marks writer_drained.
            self.ledger.transition(
                snapshot_id, HostStageState.ABORTING, owner=self.owner, reason=reason
            )

    def _poll_aborting(self, ledger_entries=None) -> None:
        with self._get_state_lock():
            aborting = list(self.aborting.items())
        for snapshot_id, entry in aborting:
            ledger_entry = (
                self.ledger.get(snapshot_id)
                if ledger_entries is None
                else ledger_entries.get(snapshot_id)
            ) or {}
            if not ledger_entry.get("writer_drained"):
                continue
            if entry.get("free_host_on_abort", True):
                self._release_record(entry)
            self.ledger.transition(
                snapshot_id,
                HostStageState.FAILED,
                owner=self.owner,
                reason=entry.get("failure_reason", "staging_failed"),
            )
            with self._get_state_lock():
                self.aborting.pop(snapshot_id, None)

    def _poll_active(self, ledger_entries=None) -> None:
        with self._get_state_lock():
            active = list(self.active.items())
        for snapshot_id, entry in active:
            ledger_entry = (
                self.ledger.get(snapshot_id)
                if ledger_entries is None
                else ledger_entries.get(snapshot_id)
            )
            if ledger_entry is None:
                self._fail_active(snapshot_id, "shared_host_ledger_missing")
                continue
            state = ledger_entry.get("state")
            if state == HostStageState.HOST_READY.value:
                future = entry.get("materialize_future")
                if future is None:
                    entry["materialize_started_at"] = time.monotonic()
                    entry["materialize_future"] = self._materialize_pool.submit(
                        entry["snapshot"].materialize
                    )
                    logger.info(
                        "AgenticKV shared_host_materialize_start snapshot=%s "
                        "tokens=%d bytes=%d",
                        snapshot_id,
                        entry["offer"]["token_count"],
                        entry["offer"]["byte_size"],
                    )
                    continue
                if not future.done():
                    continue
                try:
                    future.result()
                except Exception:
                    logger.exception(
                        "Failed to materialize P Host mapping for %s", snapshot_id
                    )
                    self._fail_active(
                        snapshot_id, "p_host_materialization_failed"
                    )
                    continue
                entry["ready_at"] = time.time()
                with self._get_state_lock():
                    # gate_request may observe the ledger before this move,
                    # but it treats HOST_READY as deferred and retries.
                    self.host_ready[snapshot_id] = entry
                    self.active.pop(snapshot_id, None)
                logger.info(
                    "AgenticKV shared_host_ready snapshot=%s tokens=%d bytes=%d "
                    "materialize_ms=%.3f",
                    snapshot_id,
                    entry["offer"]["token_count"],
                    entry["offer"]["byte_size"],
                    (
                        time.monotonic()
                        - float(entry.get("materialize_started_at", time.monotonic()))
                    )
                    * 1000.0,
                )
            elif state in {
                HostStageState.REJECTED.value,
                HostStageState.FAILED.value,
            }:
                self._release_record(entry)
                with self._get_state_lock():
                    self.active.pop(snapshot_id, None)
            elif state == HostStageState.ABORTING.value:
                entry["failure_reason"] = ledger_entry.get(
                    "reason", "shared_host_writer_failed"
                )
                entry["free_host_on_abort"] = True
                with self._get_state_lock():
                    self.active.pop(snapshot_id, None)
                    self.aborting[snapshot_id] = entry

    def _spill_score(self, record: dict[str, Any], now: float) -> float:
        offer = record["offer"]
        expected = float(self.expected_tool_seconds.get(offer.get("tool_type"), 0.0))
        elapsed = max(0.0, now - float(offer.get("tool_started_at", now)))
        remaining = max(expected - elapsed, 0.0)
        return float(offer.get("byte_size", 0)) * remaining

    def _spill_worker(self, snapshot_id: str, record: dict[str, Any]) -> None:
        offer = record["offer"]
        logical_hashes = list(offer.get("logical_hashes") or [])
        spill_indices = None
        reserved_manifest = None
        publishing_manifest = None
        fallback_manifest = None
        try:
            if len(logical_hashes) * self.page_size != int(offer["token_count"]):
                raise ValueError("spill is missing the complete logical hash list")
            spill_indices = self.host_pool.alloc(int(offer["token_count"]))
            if spill_indices is None:
                raise MemoryError("P HiCache has no temporary room for Mooncake spill")
            record["snapshot"].copy_into_hicache(
                self.host_pool, spill_indices, self.page_size
            )
            backend = self.cache_controller.storage_backend
            namespace = str(offer["storage_namespace"])
            physical_keys, byte_size = backend.agentic_snapshot_layout(
                logical_hashes, spill_indices, namespace
            )
            snapshot_store = backend.agentic_snapshot_store()
            current = snapshot_store.load_request_generation(
                offer["request_id"], int(offer["generation"]), require_ready=False
            ) if hasattr(snapshot_store, "load_request_generation") else None
            if current is None:
                from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration

                request = RequestGeneration(offer["request_id"], int(offer["generation"]))
                current = snapshot_store.load(request, require_ready=False)
            if current is None or current.state is not SnapshotState.SLOW_FALLBACK:
                raise RuntimeError("spill requires an owned SLOW_FALLBACK manifest")
            fallback_manifest = current
            manifest = replace(
                current,
                page_keys=tuple(physical_keys),
                byte_size=int(byte_size),
            ).transition(SnapshotState.OFFLOADING)
            if self.eviction_controller is not None:
                if not self.eviction_controller.reserve(manifest):
                    raise MemoryError(
                        "request-level Mooncake capacity could not reserve spill"
                    )
                reserved_manifest = manifest
            snapshot_store.update(manifest)
            publishing_manifest = manifest
            extra = HiCacheStorageExtraInfo(
                extra_info={"agentic_page_namespace": namespace}
            )
            batch_pages = max(1, int(self.cache_controller.storage_batch_size))
            for start in range(0, len(logical_hashes), batch_pages):
                hashes = logical_hashes[start : start + batch_pages]
                start_token = start * self.page_size
                end_token = start_token + len(hashes) * self.page_size
                results = backend.batch_set_v1(
                    hashes, spill_indices[start_token:end_token], extra
                )
                if not all(results):
                    raise RuntimeError("incomplete Mooncake spill Put")
            ready = snapshot_store.commit_publish(manifest.request)
            if self.eviction_controller is not None:
                self.eviction_controller.commit(ready)
                reserved_manifest = None
            record["spill_result"] = (True, ready, None)
        except Exception as exc:
            if publishing_manifest is not None:
                try:
                    observed = snapshot_store.load(
                        publishing_manifest.request, require_ready=False
                    )
                    if observed is not None and observed.state is SnapshotState.OFFLOADING:
                        snapshot_store.store.batch_remove(
                            list(observed.page_keys), force=False
                        )
                        # The complete P-Host copy remains authoritative and
                        # can be retried after pressure subsides.  Restore the
                        # invisible placeholder instead of leaving a FAILED
                        # generation that would make every retry impossible.
                        snapshot_store.update(fallback_manifest)
                except Exception:
                    logger.exception(
                        "Failed to clean incomplete P Host spill for %s", snapshot_id
                    )
            if reserved_manifest is not None and self.eviction_controller is not None:
                try:
                    self.eviction_controller.cancel(reserved_manifest)
                except Exception:
                    logger.exception(
                        "Failed to cancel P Host spill reservation for %s", snapshot_id
                    )
            logger.exception("Agentic P Host spill failed for %s", snapshot_id)
            record["spill_result"] = (False, None, str(exc))
        finally:
            if spill_indices is not None:
                self.host_pool.free(spill_indices)

    def _progress_spills(self) -> None:
        with self._get_state_lock():
            spills = list(self.spills.items())
        for snapshot_id, record in spills:
            result = record.get("spill_result")
            if result is None:
                continue
            success, _, reason = result
            thread = self._spill_threads.pop(snapshot_id, None)
            if thread is not None:
                thread.join(timeout=0)
            if success:
                # Mooncake commit is the replacement-complete ACK.
                self._release_record(record)
                with self._get_state_lock():
                    self.host_ready.pop(snapshot_id, None)
                self.ledger.transition(
                    snapshot_id, HostStageState.MOONCAKE_READY, owner=self.owner
                )
            else:
                record["loading"] = False
                with self._get_state_lock():
                    self.host_ready[snapshot_id] = record
                self.ledger.transition(
                    snapshot_id,
                    HostStageState.HOST_READY,
                    owner=self.owner,
                    reason=f"spill_failed:{reason}",
                )
            with self._get_state_lock():
                self.spills.pop(snapshot_id, None)

    def _maybe_spill(self) -> None:
        self._progress_spills()
        usage = self._host_usage()
        if not self._spilling_pressure and usage >= self.high_watermark:
            self._spilling_pressure = True
        if self._spilling_pressure and usage <= self.low_watermark:
            self._spilling_pressure = False
        with self._get_state_lock():
            if not self._spilling_pressure or self.spills:
                return
        now = time.time()
        with self._get_state_lock():
            candidates = [
                (self._spill_score(record, now), snapshot_id, record)
                for snapshot_id, record in self.host_ready.items()
                if not record.get("loading")
            ]
            if not candidates:
                return
            _, snapshot_id, record = max(
                candidates, key=lambda item: (item[0], item[1])
            )
            # Atomic ownership claim against gate_request.
            record["loading"] = "spill"
            self.host_ready.pop(snapshot_id, None)
            self.spills[snapshot_id] = record
        self.ledger.transition(
            snapshot_id, HostStageState.SPILLING, owner=self.owner
        )
        thread = threading.Thread(
            target=self._spill_worker,
            args=(snapshot_id, record),
            name=f"agentic-spill-{snapshot_id}",
            daemon=True,
        )
        self._spill_threads[snapshot_id] = thread
        thread.start()

    def _poll_once(self) -> None:
        now = time.monotonic()
        if now < self._next_poll_at:
            return
        self._next_poll_at = now + self._poll_interval
        # One shared read per P scheduler tick replaces one complete JSON read
        # per active snapshot (and another per aborting snapshot/request gate).
        ledger_entries = self.ledger.snapshot_entries()
        self._ledger_entries_cache = ledger_entries
        self._progress_spills()
        self._poll_active(ledger_entries)
        self._poll_aborting(ledger_entries)
        self._maybe_spill()
        self._admit_batch(ledger_entries)
        if time.monotonic() - self._last_prune > 5.0:
            self.ledger.prune()
            self._last_prune = time.monotonic()

    def _control_worker(self) -> None:
        while True:
            started = time.monotonic()
            try:
                self._poll_once()
            except Exception:
                self._control_errors += 1
                logger.exception("Agentic P async control progress failed")
            elapsed = time.monotonic() - started
            self._control_cycles += 1
            self._control_total_seconds += elapsed
            self._control_max_seconds = max(self._control_max_seconds, elapsed)
            now = time.monotonic()
            if now - self._control_last_stats >= 30.0:
                with self._get_state_lock():
                    active = len(self.active)
                    host_ready = len(self.host_ready)
                    spills = len(self.spills)
                logger.info(
                    "Agentic P async control stats cycles=%d avg_us=%.1f "
                    "max_ms=%.3f active=%d host_ready=%d spills=%d errors=%d",
                    self._control_cycles,
                    self._control_total_seconds
                    / max(self._control_cycles, 1)
                    * 1e6,
                    self._control_max_seconds * 1e3,
                    active,
                    host_ready,
                    spills,
                    self._control_errors,
                )
                self._control_cycles = 0
                self._control_total_seconds = 0.0
                self._control_max_seconds = 0.0
                self._control_last_stats = now
            self._control_wakeup.wait(self._control_interval)
            self._control_wakeup.clear()

    def poll(self) -> None:
        """Retain synchronous behavior only when async control is disabled.

        The scheduler calls this method both while busy and while spinning
        idle.  Waking the worker on every call defeats its fixed-rate bound
        and can turn an idle scheduler into tens of thousands of ledger scans
        per minute, so async mode is intentionally a true O(1) no-op.
        """

        if self._async_control:
            return
        self._poll_once()

    def snapshot_ready(self, request_generation) -> bool:
        """Return whether P already owns a readable Host snapshot."""

        with self._get_state_lock():
            return request_generation.snapshot_id in self.host_ready

    def gate_request(
        self, req, request_generation, *, allow_start: bool = True
    ) -> Optional[bool]:
        """Return True to defer, False when loaded, None when not owned.

        ``allow_start`` separates discovery from admission.  Polling an H2D
        copy that is already in flight is always permitted, but a new HBM
        allocation and copy are started only after the P scheduler selects
        this request from its priority queues.
        """

        snapshot_id = request_generation.snapshot_id
        with self._get_state_lock():
            load = self.loads.get(req.rid)
        if load is not None:
            if not load["event"].query():
                return True
            load["event"].synchronize()
            record = load["record"]
            h2d_start_event = getattr(
                record["snapshot"], "_last_h2d_start_event", None
            )
            h2d_elapsed_ms = (
                float("nan")
                if h2d_start_event is None
                else h2d_start_event.elapsed_time(load["event"])
            )
            device_indices = load["device_indices"]
            keys = req.origin_input_ids[: int(record["offer"]["token_count"])]
            try:
                from sglang.srt.mem_cache.base_prefix_cache import (
                    InsertParams,
                    MatchPrefixParams,
                )
                from sglang.srt.mem_cache.radix_cache import RadixKey

                radix_key = RadixKey(keys, req.extra_key)
                result = self.tree_cache.insert(
                    InsertParams(
                        key=radix_key,
                        value=device_indices,
                        priority=getattr(req, "priority", 0) or 0,
                    )
                )
                if result.prefix_len:
                    self.token_allocator.free(device_indices[: result.prefix_len])
                matched = self.tree_cache.match_prefix(
                    MatchPrefixParams(key=radix_key, req=req)
                )
                if len(matched.device_indices) != len(keys):
                    raise RuntimeError("P Host H2D Radix insert is incomplete")
                self.tree_cache.inc_lock_ref(matched.last_device_node)
                req._agentic_kv_host_pin_node = matched.last_device_node
            except Exception:
                self.token_allocator.free(device_indices)
                record["loading"] = False
                with self._get_state_lock():
                    self.host_ready[snapshot_id] = record
                    self.loads.pop(req.rid, None)
                logger.exception("Failed to insert P Host snapshot for %s", req.rid)
                return None
            # The complete GPU copy is now explicitly protected in Radix.  The
            # scheduler releases this temporary reference immediately after
            # _prefetch_kvcache rematches and acquires the request's own lock.
            self._release_record(record)
            with self._get_state_lock():
                self.host_ready.pop(snapshot_id, None)
                self.loads.pop(req.rid, None)
            self.ledger.transition(snapshot_id, HostStageState.CONSUMED, owner=self.owner)
            logger.info(
                "AgenticKV shared_host_h2d_complete snapshot=%s tokens=%d "
                "elapsed_ms=%.3f gib_per_s=%.3f arena_released=true",
                snapshot_id,
                int(record["offer"]["token_count"]),
                h2d_elapsed_ms,
                0.0
                if not math.isfinite(h2d_elapsed_ms)
                else int(record["offer"]["byte_size"])
                / max(h2d_elapsed_ms / 1000.0, 1e-9)
                / (1024**3),
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_host_hit_tokens = int(record["offer"]["token_count"])
            return False

        with self._get_state_lock():
            record = self.host_ready.get(snapshot_id)
            control_owned = snapshot_id in getattr(
                self, "active", {}
            ) or snapshot_id in getattr(self, "aborting", {})
        if control_owned:
            return True
        ledger_cache = getattr(self, "_ledger_entries_cache", None)
        ledger_entry = (
            self.ledger.get(snapshot_id)
            if ledger_cache is None
            else ledger_cache.get(snapshot_id)
        )
        if record is None:
            if ledger_entry is not None and ledger_entry.get("state") in {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
                HostStageState.ABORTING.value,
                HostStageState.HOST_READY.value,
                HostStageState.SPILLING.value,
                HostStageState.H2D_LOADING.value,
            }:
                return True
            return None
        with self._get_state_lock():
            # Re-read under the ownership lock: the background spill worker
            # may have claimed the record after the first discovery read.
            record = self.host_ready.get(snapshot_id)
            if record is None or record.get("loading"):
                return True
            if not allow_start:
                return True
            if len(self.loads) >= getattr(self, "max_h2d_inflight", 1):
                return True
            # Claim before touching the GPU allocator so spill selection and
            # H2D admission can never own the same snapshot concurrently.
            record["loading"] = "h2d_reserving"
        # There is one priority H2D stream, so copies remain ordered.  Keep a
        # small queue on that stream: each DMA takes only milliseconds, while
        # the scheduler may not revisit the waiting queue for several seconds
        # under long Prefill kernels.  Layer-first contiguous mappings make
        # this bounded queue safe and remove those otherwise idle PCIe gaps.
        offer = record["offer"]
        parent_tokens = req.origin_input_ids[: int(offer["token_count"])]
        from sglang.srt.disaggregation.agentic_kv_lifecycle import token_ids_digest

        if len(parent_tokens) != int(offer["token_count"]) or token_ids_digest(
            parent_tokens
        ) != offer.get("token_digest"):
            with self._get_state_lock():
                record["loading"] = False
            return None
        device_indices = self.token_allocator.alloc(int(offer["token_count"]))
        if device_indices is None:
            with self._get_state_lock():
                record["loading"] = False
            return True
        try:
            event, copy_refs = record["snapshot"].start_load_to_device(
                device_indices,
                self._h2d_stream,
                chunk_tokens=getattr(self, "h2d_chunk_tokens", 4096),
                staging=self._h2d_staging,
            )
        except Exception:
            self.token_allocator.free(device_indices)
            with self._get_state_lock():
                record["loading"] = False
            raise
        with self._get_state_lock():
            record["loading"] = "h2d"
            self.loads[req.rid] = {
                "record": record,
                "device_indices": device_indices,
                "event": event,
                "copy_refs": copy_refs,
            }
        self.ledger.transition(snapshot_id, HostStageState.H2D_LOADING, owner=self.owner)
        logger.info(
            "AgenticKV shared_host_h2d_start snapshot=%s tokens=%d bytes=%d",
            snapshot_id,
            int(offer["token_count"]),
            int(offer["byte_size"]),
        )
        return True

    def release_request_pin(self, req) -> None:
        node = getattr(req, "_agentic_kv_host_pin_node", None)
        if node is None:
            return
        self.tree_cache.dec_lock_ref(node)
        delattr(req, "_agentic_kv_host_pin_node")


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
        self.direct_cross_numa_gib_per_second = float(
            direct_cross_numa_gib_per_second
        )
        self.nvlink_gib_per_second = float(nvlink_gib_per_second)
        self.relay_stale_seconds = float(relay_stale_seconds)
        d2h_priority = int(
            os.getenv("SGLANG_AGENTIC_KV_D2H_STREAM_PRIORITY", "0")
        )
        self._d2h_stream = torch.cuda.Stream(
            device=torch.cuda.current_device(), priority=d2h_priority
        )
        self._d2h_staging = LayerFirstD2HStaging(
            self.device_pool,
            int(os.getenv("SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS", "256")),
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
        self._active_write_snapshot_id: Optional[str] = None

    def offer(
        self,
        *,
        manifest: SnapshotManifest,
        metadata,
        token_count: int,
        token_digest: str,
        logical_hashes: list[str],
        byte_size: int,
    ) -> dict[str, Any]:
        return self.ledger.offer(
            {
                "snapshot_id": manifest.snapshot_id,
                "request_id": metadata.current.request_id,
                "generation": metadata.current.generation,
                "token_count": int(token_count),
                "token_digest": token_digest,
                "logical_hashes": list(logical_hashes),
                "byte_size": int(byte_size),
                "storage_namespace": page_namespace(metadata.current),
                "tool_type": metadata.tool_type,
                "tool_started_at": manifest.tool_started_at or time.time(),
                "d_pid": os.getpid(),
                "source_numa_node": self.source_numa_node,
                "arena_numa_node": self.arena_numa_node,
                "source_bootstrap_addr": (
                    None
                    if self.direct_runtime is None
                    else self.direct_runtime.bootstrap_addr
                ),
                "source_room": manifest.direct_room,
            }
        )

    def _cleanup_write(self, candidate: dict[str, Any]) -> bool:
        write = candidate.get("arena_write")
        if write is None:
            return True
        event = write.get("event")
        if event is not None and not event.query():
            return False
        candidate.pop("arena_write", None)
        if getattr(self, "_active_write_snapshot_id", None) == candidate[
            "manifest"
        ].snapshot_id:
            self._active_write_snapshot_id = None
        write["snapshot"].close(unlink=False)
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
        if len(grants) != 1 or grants[0].get("kind") != "shared_host_extent":
            raise RuntimeError("P did not publish one complete shared Host extent")
        grant = grants[0]
        snapshot = SharedMHAHostSnapshot(
            path=str(grant["arena_path"]),
            token_count=int(grant["token_count"]),
            device_pool=self.device_pool,
            byte_size=int(grant["byte_size"]),
            create=False,
        )
        candidate["arena_write"] = {
            "snapshot": snapshot,
            "event": None,
            "copy_refs": None,
            "offset": 0,
            "chunk_end": 0,
            "gpu_elapsed_ms": 0.0,
        }
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
        """Submit at most one bounded D2H chunk across all candidates."""

        snapshot_id = candidate["manifest"].snapshot_id
        if self._active_write_snapshot_id not in {None, snapshot_id}:
            return False
        write = candidate["arena_write"]
        if write["event"] is not None:
            return False
        start = int(write["offset"])
        if start >= len(source_token_indices):
            return False
        end = min(start + self._d2h_chunk_tokens, len(source_token_indices))
        event, refs = write["snapshot"].start_backup_range_from_device(
            source_token_indices[start:end],
            destination_start=start,
            stream=self._d2h_stream,
            staging=self._d2h_staging,
        )
        write["event"] = event
        write["copy_refs"] = refs
        write["chunk_end"] = end
        self._active_write_snapshot_id = snapshot_id
        return True

    def progress(
        self,
        candidate: dict[str, Any],
        source_token_indices,
        *,
        entry_snapshot=_LEDGER_ENTRY_UNSET,
    ) -> str:
        snapshot_id = candidate["manifest"].snapshot_id
        entry = (
            self.ledger.get(snapshot_id)
            if entry_snapshot is _LEDGER_ENTRY_UNSET
            else entry_snapshot
        )
        if entry is None:
            if not self._cleanup_write(candidate):
                return "waiting"
            return "failed"
        state = entry.get("state")
        if state in {
            HostStageState.HOST_READY.value,
            HostStageState.H2D_LOADING.value,
            HostStageState.SPILLING.value,
            HostStageState.MOONCAKE_READY.value,
            HostStageState.CONSUMED.value,
        }:
            # HOST_READY is a monotonic durability boundary.  P may already be
            # loading or spilling by the time D polls; all later states still
            # imply that a complete non-D copy exists.
            if not self._cleanup_write(candidate):
                return "waiting"
            self._cleanup_relay_senders(candidate)
            return "host_ready"
        if state in {
            HostStageState.REJECTED.value,
            HostStageState.FAILED.value,
        }:
            if not self._cleanup_write(candidate):
                return "waiting"
            self._cleanup_relay_senders(candidate)
            return "failed"
        write = candidate.get("arena_write")
        if state == HostStageState.ABORTING.value:
            try:
                if not self._cleanup_write(candidate):
                    return "waiting"
                self.ledger.mark_writer_drained(snapshot_id, os.getpid())
            except Exception:
                logger.exception("Failed to drain shared Host writer for %s", snapshot_id)
                return "failed"
            return "waiting"
        if state != HostStageState.HOST_WRITING.value:
            return "waiting"
        write_mode = entry.get("write_mode")
        if not write_mode:
            if self.relay_enabled:
                entry = self.ledger.assign_transfer_path(
                    snapshot_id,
                    source_pid=os.getpid(),
                    source_numa_node=self.source_numa_node,
                    arena_numa_node=self.arena_numa_node,
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
            try:
                self._start_write(candidate, entry, source_token_indices)
            except Exception:
                logger.exception("Agentic shared Host D2H start failed for %s", snapshot_id)
                self.ledger.fail_host_write(
                    snapshot_id, os.getpid(), "shared_host_d2h_start_failed"
                )
                return "failed"
            write = candidate["arena_write"]
        if write["event"] is None:
            self._start_write_chunk(candidate, source_token_indices)
            return "waiting"
        if not write["event"].query():
            return "waiting"
        chunk_elapsed_ms = write["snapshot"]._last_d2h_start_event.elapsed_time(
            write["event"]
        )
        write["gpu_elapsed_ms"] += chunk_elapsed_ms
        write["offset"] = int(write["chunk_end"])
        write["event"] = None
        write["copy_refs"] = None
        self._active_write_snapshot_id = None
        if write["offset"] < len(source_token_indices):
            # Do not submit the next chunk in this call.  Returning to the
            # progress loop gives the default-priority Decode stream a chance
            # to enqueue its next forward before more background copy work.
            return "waiting"
        d2h_elapsed_ms = float(write["gpu_elapsed_ms"])
        d2h_bytes = write["snapshot"].byte_size
        elapsed_ready = time.time()
        if not self._cleanup_write(candidate):
            return "waiting"
        if not self.ledger.complete_host_write(snapshot_id, os.getpid()):
            self.ledger.mark_writer_drained(snapshot_id, os.getpid())
            return "failed"
        logger.info(
            "AgenticKV shared_host_d2h_complete snapshot=%s completed_at=%.6f "
            "elapsed_ms=%.3f gib_per_s=%.3f",
            snapshot_id,
            elapsed_ready,
            d2h_elapsed_ms,
            d2h_bytes / max(d2h_elapsed_ms / 1000.0, 1e-9) / (1024**3),
        )
        return "host_ready"
