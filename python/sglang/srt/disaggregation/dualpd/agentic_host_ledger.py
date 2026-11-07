"""Node-local Host ledger: request-generation states, CAS and fenced ownership.

This module is deliberately independent of CUDA, DMA and worker construction.
Import public classes through agentic_host_staging for historical compatibility.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional

_LEDGER_OWNERSHIP_CHANGED = object()


class HostStageState(str, Enum):
    OFFERED = "offered"
    HOST_RESERVED = "host_reserved"
    HOST_WRITING = "host_writing"
    ABORTING = "aborting"
    HOST_READY = "host_ready"
    H2D_LOADING = "h2d_loading"
    # Every P rank owns a complete HBM copy, but the child request has not yet
    # inserted and pinned it in Radix. Host remains authoritative here.
    HBM_READY = "hbm_ready"
    RETRY_PENDING = "retry_pending"
    # Request-generation eviction is a two-phase group operation.  The Host
    # snapshot remains authoritative in EVICTING until every TP rank has
    # released its local shard. Pressure eviction then permits recomputation;
    # application-final reclamation instead commits CONSUMED (no next turn).
    EVICTING = "evicting"
    RECOMPUTE_REQUIRED = "recompute_required"
    SPILLING = "spilling"
    MOONCAKE_READY = "mooncake_ready"
    CONSUMED = "consumed"
    REJECTED = "rejected"
    FAILED = "failed"


class PermanentHostStageError(RuntimeError):
    """Non-retryable request-generation/Shared-Arena incompatibility."""


_TERMINAL_STATES = {
    HostStageState.CONSUMED.value,
    HostStageState.REJECTED.value,
    HostStageState.FAILED.value,
    HostStageState.RECOMPUTE_REQUIRED.value,
}

P2D_RELEASE_NATIVE_WON = "native_won"
P2D_RELEASE_HOST_OWNED = "host_owned"
P2D_RELEASE_HOST_TERMINAL = "host_terminal"

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
        HostStageState.ABORTING.value,
        HostStageState.EVICTING.value,
        HostStageState.SPILLING.value,
        HostStageState.CONSUMED.value,
        HostStageState.FAILED.value,
    },
    HostStageState.H2D_LOADING.value: {
        HostStageState.ABORTING.value,
        HostStageState.HOST_READY.value,
        HostStageState.HBM_READY.value,
        HostStageState.CONSUMED.value,
        HostStageState.RETRY_PENDING.value,
        HostStageState.FAILED.value,
    },
    HostStageState.HBM_READY.value: {
        HostStageState.CONSUMED.value,
        HostStageState.RETRY_PENDING.value,
        HostStageState.FAILED.value,
    },
    HostStageState.RETRY_PENDING.value: {
        HostStageState.HOST_READY.value,
        HostStageState.FAILED.value,
    },
    HostStageState.EVICTING.value: {
        HostStageState.RECOMPUTE_REQUIRED.value,
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
        self.path = os.path.abspath(path)
        self.event_directory = f"{self.path}.events"
        os.makedirs(self.event_directory, exist_ok=True)
        self.lock_directory = os.path.join(self.event_directory, ".locks")
        os.makedirs(self.lock_directory, exist_ok=True)
        self.relay_marker_directory = os.path.join(
            self.event_directory, ".relay"
        )
        os.makedirs(self.relay_marker_directory, exist_ok=True)
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
                # A fresh authoritative ledger starts a fresh event epoch.
                # Stale per-snapshot deltas from an earlier server lifetime
                # must never be replayed after the new initial resync.
                for item in os.scandir(self.event_directory):
                    if item.is_file() and item.name.endswith(".json"):
                        try:
                            os.unlink(item.path)
                        except FileNotFoundError:
                            pass
                self._write_locked(
                    file_obj,
                    {"version": self.VERSION, "entries": {}, "relays": {}},
                )
            else:
                data = json.loads(raw)
                if data.get("version") != self.VERSION:
                    raise ValueError("unsupported host staging ledger version")
                # One-time compatibility migration from the original
                # monolithic entries document. Normal operation thereafter
                # keeps only the small relay registry in this file; every
                # request-generation is authoritative in its own manifest.
                legacy_entries = data.get("entries", {})
                if legacy_entries:
                    retained_relay_entries = {}
                    for snapshot_id, entry in legacy_entries.items():
                        value = dict(entry)
                        try:
                            manifest = self.read_entry_event(
                                self._event_path(snapshot_id)
                            )
                            manifest_revision = int(manifest.get("revision", 0))
                        except FileNotFoundError:
                            manifest_revision = 0
                        # The old global document was authoritative. Always
                        # republish it and advance beyond either observed
                        # revision; a stale mirror must never win migration.
                        value["_event_revision"] = max(
                            manifest_revision,
                            int(value.get("_event_revision", 0)),
                        ) + 1
                        self._publish_entry_event_locked(snapshot_id, value)
                        if value.get("relay_id"):
                            self._publish_relay_marker(snapshot_id)
                            retained_relay_entries[snapshot_id] = value
                    data["entries"] = retained_relay_entries
                    self._write_locked(file_obj, data)
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_locked(file_obj, value: dict[str, Any]) -> None:
        file_obj.seek(0)
        json.dump(value, file_obj, separators=(",", ":"), sort_keys=True)
        file_obj.truncate()
        file_obj.flush()
        # /dev/shm is memory-backed; fsync is intentionally omitted from the hot path.

    def _event_path(self, snapshot_id: str) -> str:
        digest = hashlib.sha256(str(snapshot_id).encode("utf-8")).hexdigest()
        return os.path.join(self.event_directory, f"{digest}.json")

    def _entry_lock_path(self, snapshot_id: str) -> str:
        digest = hashlib.sha256(str(snapshot_id).encode("utf-8")).hexdigest()
        # Stable striped locks avoid one permanent inode per generation while
        # never unlinking a flock inode that another process may already have
        # opened. Hash collisions only serialize unrelated snapshots.
        stripe = int(digest[:8], 16) % 4096
        return os.path.join(self.lock_directory, f"{stripe:04x}.lock")

    def _relay_marker_path(self, snapshot_id: str) -> str:
        digest = hashlib.sha256(str(snapshot_id).encode("utf-8")).hexdigest()
        return os.path.join(self.relay_marker_directory, digest)

    def _publish_relay_marker(self, snapshot_id: str) -> None:
        path = self._relay_marker_path(snapshot_id)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)

    def _is_relay_snapshot(self, snapshot_id: str) -> bool:
        return os.path.exists(self._relay_marker_path(snapshot_id))

    @contextmanager
    def _entry_locked(self, snapshot_id: str):
        lock_path = self._entry_lock_path(snapshot_id)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextmanager
    def _entry_read_locked(self, snapshot_id: str):
        lock_path = self._entry_lock_path(snapshot_id)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _publish_entry_event_locked(
        self, snapshot_id: str, entry: Optional[dict[str, Any]]
    ) -> None:
        """Publish one complete request-generation delta while holding flock.

        Publishing before releasing the ledger lock preserves revision order
        across writers.  Atomic rename means an inotify consumer observes
        either the previous complete event or the new complete event, never a
        partially serialized manifest.
        """

        event_path = self._event_path(snapshot_id)
        temporary_path = (
            f"{event_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        payload = {
            "version": self.VERSION,
            "snapshot_id": str(snapshot_id),
            "revision": int((entry or {}).get("_event_revision", 0)),
            "entry": None if entry is None else dict(entry),
        }
        try:
            with open(temporary_path, "w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, separators=(",", ":"), sort_keys=True)
                file_obj.flush()
            os.replace(temporary_path, event_path)
        finally:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    def read_entry_event(self, path: str | os.PathLike) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file_obj:
            event = json.load(file_obj)
        if event.get("version") != self.VERSION:
            raise ValueError("unsupported host staging event version")
        snapshot_id = str(event.get("snapshot_id", ""))
        if not snapshot_id or self._event_path(snapshot_id) != os.fspath(path):
            raise ValueError("host staging event path does not match snapshot")
        return event

    def _mutate(self, callback, *, event_snapshot_id: Optional[str] = None):
        if event_snapshot_id is None:
            raise ValueError("request-generation mutation requires snapshot id")
        snapshot_id = str(event_snapshot_id)
        while True:
            if self._is_relay_snapshot(snapshot_id):
                result = self._mutate_relay_entry(snapshot_id, callback)
                if result is _LEDGER_OWNERSHIP_CHANGED:
                    continue
                return result
            with self._entry_locked(snapshot_id):
                # assign_transfer_path publishes the relay marker while it
                # holds this stripe. A writer that observed normal ownership
                # before waiting for the stripe must not update only the event
                # mirror after ownership has moved to the global relay
                # transaction.
                if self._is_relay_snapshot(snapshot_id):
                    continue
                try:
                    previous_event = self.read_entry_event(
                        self._event_path(snapshot_id)
                    )
                except FileNotFoundError:
                    previous_event = {"revision": 0, "entry": None}
                previous = previous_event.get("entry")
                entries = {} if previous is None else {snapshot_id: dict(previous)}
                result, changed = callback(entries)
                if changed:
                    current = entries.get(snapshot_id)
                    if current is not None:
                        current["_event_revision"] = max(
                            int(previous_event.get("revision", 0)),
                            int(current.get("_event_revision", 0)),
                        ) + 1
                    self._publish_entry_event_locked(snapshot_id, current)
                    with self._snapshot_cache_lock:
                        self._snapshot_cache_at = 0.0
                        self._snapshot_cache = {}
                return result

    def _mutate_relay_entry(self, snapshot_id: str, callback):
        """Mutate a relay-owned entry in the single atomic relay document."""

        with open(self.path, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            try:
                # Relay prune removes the marker and authoritative entry in
                # this same global transaction. A caller that observed the
                # old marker before waiting for the lock must retry through
                # the normal per-snapshot path instead of resurrecting a
                # deleted relay entry.
                if not self._is_relay_snapshot(snapshot_id):
                    return _LEDGER_OWNERSHIP_CHANGED
                file_obj.seek(0)
                data = json.loads(file_obj.read() or "{}")
                if data.get("version") != self.VERSION:
                    raise ValueError("corrupt host staging relay registry")
                entries = data.setdefault("entries", {})
                previous = entries.get(snapshot_id)
                if previous is None:
                    try:
                        previous_event = self.read_entry_event(
                            self._event_path(snapshot_id)
                        )
                    except FileNotFoundError:
                        previous_event = {"revision": 0, "entry": None}
                    previous = previous_event.get("entry")
                    if previous is not None:
                        entries[snapshot_id] = dict(previous)
                result, changed = callback(entries)
                if changed:
                    current = entries.get(snapshot_id)
                    if current is not None:
                        current["_event_revision"] = (
                            int(current.get("_event_revision", 0)) + 1
                        )
                    self._write_locked(file_obj, data)
                    # Global JSON is authoritative for relay entries. The
                    # per-snapshot file remains its event/read-through mirror.
                    self._publish_entry_event_locked(snapshot_id, current)
                    with self._snapshot_cache_lock:
                        self._snapshot_cache_at = 0.0
                        self._snapshot_cache = {}
                return result
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    def _mutate_document(
        self, callback, *, event_snapshot_id: Any = None
    ):
        """Mutate entries plus the node-local relay registry atomically."""

        if callable(event_snapshot_id):
            raise ValueError("dynamic relay claims require claim_relay_job")

        with open(self.path, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            try:
                file_obj.seek(0)
                data = json.loads(file_obj.read() or "{}")
                if data.get("version") != self.VERSION:
                    raise ValueError("corrupt host staging ledger")
                data.setdefault("relays", {})
                if event_snapshot_id is None:
                    data.setdefault("entries", {})
                    result, changed = callback(data)
                    if changed:
                        self._write_locked(file_obj, data)
                    return result
                snapshot_id = str(event_snapshot_id)
                with self._entry_locked(snapshot_id):
                    try:
                        previous_event = self.read_entry_event(
                            self._event_path(snapshot_id)
                        )
                    except FileNotFoundError:
                        previous_event = {"revision": 0, "entry": None}
                    global_entries = data.setdefault("entries", {})
                    previous = global_entries.get(snapshot_id)
                    if previous is None:
                        previous = previous_event.get("entry")
                    working_entries = (
                        {} if previous is None else {snapshot_id: dict(previous)}
                    )
                    data["entries"] = working_entries
                    result, changed = callback(data)
                    if changed:
                        current = working_entries.get(snapshot_id)
                        if current is not None:
                            current["_event_revision"] = max(
                                int(previous_event.get("revision", 0)),
                                int(current.get("_event_revision", 0)),
                            ) + 1
                        relay_owned = bool(
                            current is not None and current.get("relay_id")
                        )
                        if relay_owned:
                            global_entries[snapshot_id] = current
                        else:
                            global_entries.pop(snapshot_id, None)
                        data["entries"] = global_entries
                        self._write_locked(file_obj, data)
                        if relay_owned:
                            # Publish only after the authoritative global
                            # transaction commits, while the entry stripe is
                            # still held. A racing normal mutation therefore
                            # cannot miss the ownership-mode switch.
                            self._publish_relay_marker(snapshot_id)
                        self._publish_entry_event_locked(snapshot_id, current)
                        with self._snapshot_cache_lock:
                            self._snapshot_cache_at = 0.0
                            self._snapshot_cache = {}
                    else:
                        data["entries"] = global_entries
                    return result
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    def offer(self, entry: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = str(entry["snapshot_id"])
        control_offer = bool(entry.get("control_offer", False))

        def callback(entries):
            current = entries.get(snapshot_id)
            now = time.time()
            tp_size = max(1, int(entry.get("tp_size", 1)))
            tp_rank = int(entry.get("tp_rank", 0))
            if not 0 <= tp_rank < tp_size:
                raise ValueError("invalid host-staging TP rank")
            if current is not None:
                # A P TP group may atomically choose native NIXL before every
                # D rank has published its fallback offer.  Keep that decision
                # as a tombstone so a late rank cannot resurrect Host staging.
                if (
                    current.get("state") == HostStageState.REJECTED.value
                    and current.get("native_won")
                ):
                    return dict(current), False
                if int(current.get("tp_size", 1)) != tp_size:
                    raise ValueError("host-staging TP size changed within snapshot")
                if (
                    int(current.get("token_count", -1))
                    != int(entry.get("token_count", -2))
                    or current.get("token_digest") != entry.get("token_digest")
                ):
                    raise ValueError("host-staging TP ranks disagree on tokens")
                current_is_control = bool(current.get("control_offer", False))
                if current_is_control or control_offer:
                    if not (current_is_control and control_offer):
                        raise ValueError(
                            "host-staging snapshot cannot mix control and rank offers"
                        )
                    # Router publication is an idempotent notification.  Once
                    # any P rank has claimed the snapshot, replaying the same
                    # notification must never recreate rank metadata or move
                    # HOST_RESERVED/HOST_WRITING/HOST_READY back to OFFERED.
                    return dict(current), False
                rank_offers = current.setdefault("rank_offers", {})
                rank_key = str(tp_rank)
                incoming = {
                    "tp_rank": tp_rank,
                    "d_pid": int(entry.get("d_pid", -1)),
                    "byte_size": int(entry.get("byte_size", 0)),
                    "source_numa_node": int(entry.get("source_numa_node", -1)),
                    "arena_numa_node": int(entry.get("arena_numa_node", -1)),
                    "source_bootstrap_addr": entry.get("source_bootstrap_addr"),
                    "source_room": entry.get("source_room"),
                }
                previous = rank_offers.get(rank_key)
                if previous is not None and previous != incoming:
                    raise ValueError("host-staging TP rank offer is not idempotent")
                if previous is not None:
                    return dict(current), False
                rank_offers[rank_key] = incoming
                current["byte_size"] = sum(
                    int(value.get("byte_size", 0))
                    for value in rank_offers.values()
                )
                current["updated_at"] = now
                if (
                    len(rank_offers) == tp_size
                    and current.get("state") == "tp_collecting"
                ):
                    current["state"] = HostStageState.OFFERED.value
                return dict(current), True
            value = dict(entry)
            rank_offer = {
                "tp_rank": tp_rank,
                "d_pid": int(entry.get("d_pid", -1)),
                "byte_size": int(entry.get("byte_size", 0)),
                "source_numa_node": int(entry.get("source_numa_node", -1)),
                "arena_numa_node": int(entry.get("arena_numa_node", -1)),
                "source_bootstrap_addr": entry.get("source_bootstrap_addr"),
                "source_room": entry.get("source_room"),
            }
            value.update(
                state=(
                    HostStageState.OFFERED.value
                    if tp_size == 1 or control_offer
                    else "tp_collecting"
                ),
                created_at=float(value.get("created_at", now)),
                updated_at=now,
                grants=[],
                acked_chunks=[],
                sent_chunks=[],
                tp_size=tp_size,
                rank_offers=({} if control_offer else {str(tp_rank): rank_offer}),
                rank_grants={},
                writer_acks=[],
                loader_acks=[],
                loader_drained_ranks=[],
            )
            entries[snapshot_id] = value
            return dict(value), True

        return self._mutate(callback, event_snapshot_id=snapshot_id)

    def get(self, snapshot_id: str) -> Optional[dict[str, Any]]:
        snapshot_id = str(snapshot_id)
        while True:
            if self._is_relay_snapshot(snapshot_id):
                with open(self.path, "r", encoding="utf-8") as file_obj:
                    fcntl.flock(file_obj.fileno(), fcntl.LOCK_SH)
                    try:
                        # prune may have removed relay ownership while this
                        # reader waited for the global lock.
                        if not self._is_relay_snapshot(snapshot_id):
                            continue
                        data = json.loads(file_obj.read() or "{}")
                        value = data.get("entries", {}).get(snapshot_id)
                        return None if value is None else dict(value)
                    finally:
                        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
            # A normal read participates in the snapshot stripe protocol.
            # If assign_transfer_path switched ownership while this reader
            # waited, release the stripe and retry through the global relay
            # document; never acquire global while holding the stripe.
            with self._entry_read_locked(snapshot_id):
                if self._is_relay_snapshot(snapshot_id):
                    continue
                try:
                    event = self.read_entry_event(self._event_path(snapshot_id))
                except FileNotFoundError:
                    return None
                value = event.get("entry")
                return None if value is None else dict(value)

    def snapshot_entries(
        self, *, force_refresh: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Read all staging states under one shared lock.

        Scheduler callers commonly need the state of tens of snapshots at
        once.  Reading and JSON-decoding the complete ledger separately for
        every snapshot makes control-plane cost quadratic in the number of
        in-flight agent turns and serializes all P/D workers on one flock.
        """

        now = time.monotonic()
        with self._snapshot_cache_lock:
            if (
                not force_refresh
                and self._snapshot_cache_seconds > 0
                and now - self._snapshot_cache_at < self._snapshot_cache_seconds
            ):
                return {
                    key: dict(value) for key, value in self._snapshot_cache.items()
                }
        entries = {}
        try:
            paths = tuple(
                item.path
                for item in os.scandir(self.event_directory)
                if item.is_file() and item.name.endswith(".json")
            )
        except FileNotFoundError:
            paths = ()
        for path in paths:
            try:
                event = self.read_entry_event(path)
            except FileNotFoundError:
                continue
            value = event.get("entry")
            if value is not None:
                entries[str(event["snapshot_id"])] = dict(value)
        # Relay entries remain atomically coupled to the relay registry in the
        # small global document. Overlay them so a crash between global commit
        # and mirror publication cannot expose stale state during resync.
        with open(self.path, "r", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_SH)
            try:
                data = json.loads(file_obj.read() or "{}")
                relay_entries = {
                    str(key): dict(value)
                    for key, value in data.get("entries", {}).items()
                }
                # Keep relay ownership stable until every event mirror has
                # either been overlaid from its authoritative global entry or
                # removed as a pruned relay. An assign/prune transaction can
                # only publish/remove its marker while holding global EX.
                for snapshot_id in tuple(entries):
                    if self._is_relay_snapshot(snapshot_id):
                        if snapshot_id in relay_entries:
                            entries[snapshot_id] = relay_entries.pop(snapshot_id)
                        else:
                            entries.pop(snapshot_id, None)
                entries.update(relay_entries)
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

        return self._mutate_document(callback, event_snapshot_id=snapshot_id)

    def claim_relay_job(self, relay_id: str, pid: int) -> Optional[dict[str, Any]]:
        # Relay mode is exceptional and may search all request-generations.
        # Keep its small global registry lock outside the selected snapshot
        # lock so every relay operation uses the same lock order.
        with open(self.path, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            try:
                file_obj.seek(0)
                data = json.loads(file_obj.read() or "{}")
                if data.get("version") != self.VERSION:
                    raise ValueError("corrupt host staging relay registry")
                relay = data.setdefault("relays", {}).get(str(relay_id))
                if relay is None or int(relay.get("pid", -1)) != int(pid):
                    return None
                active = relay.get("active_snapshot")
                if active:
                    current = data.setdefault("entries", {}).get(str(active))
                    return None if current is None else dict(current)
                queued = [
                    value
                    for value in data.setdefault("entries", {}).values()
                    if value.get("relay_id") == str(relay_id)
                    and value.get("relay_job_state") == "queued"
                    and value.get("state") == HostStageState.HOST_WRITING.value
                ]
                queued.sort(
                    key=lambda item: (
                        item.get("created_at", 0.0),
                        item["snapshot_id"],
                    )
                )
                for candidate in queued:
                    snapshot_id = str(candidate["snapshot_id"])
                    current = data["entries"].get(snapshot_id)
                    if current is None:
                        continue
                    current["relay_job_state"] = "claimed"
                    current["updated_at"] = time.time()
                    current["_event_revision"] = (
                        int(current.get("_event_revision", 0)) + 1
                    )
                    relay["active_snapshot"] = snapshot_id
                    relay["updated_at"] = time.time()
                    self._write_locked(file_obj, data)
                    self._publish_entry_event_locked(snapshot_id, current)
                    with self._snapshot_cache_lock:
                        self._snapshot_cache_at = 0.0
                        self._snapshot_cache = {}
                    return dict(current)
                return None
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

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

        return bool(
            self._mutate_document(callback, event_snapshot_id=snapshot_id)
        )

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

        return bool(
            self._mutate_document(callback, event_snapshot_id=snapshot_id)
        )

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

        return bool(
            self._mutate_document(callback, event_snapshot_id=snapshot_id)
        )

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

        return bool(
            self._mutate_document(callback, event_snapshot_id=snapshot_id)
        )

    def list_state(self, *states: HostStageState) -> list[dict[str, Any]]:
        wanted = {state.value for state in states}
        values = [
            dict(value)
            for value in self.snapshot_entries(force_refresh=True).values()
            if value.get("state") in wanted
        ]
        values.sort(
            key=lambda item: (item.get("created_at", 0.0), item["snapshot_id"])
        )
        return values

    def claim(self, snapshot_id: str, owner: str) -> Optional[dict[str, Any]]:
        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None:
                return None, False
            if (
                current.get("state") == HostStageState.HOST_RESERVED.value
                and current.get("p_owner") == owner
            ):
                return dict(current), False
            if current.get("state") != HostStageState.OFFERED.value:
                return None, False
            current["state"] = HostStageState.HOST_RESERVED.value
            current["p_owner"] = owner
            current["updated_at"] = time.time()
            return dict(current), True

        return self._mutate(callback, event_snapshot_id=snapshot_id)

    def claim_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> Optional[dict[str, Any]]:
        if int(tp_size) == 1:
            return self.claim(snapshot_id, owner)

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or int(current.get("tp_size", 1)) != int(tp_size):
                return None, False
            if current.get("state") not in {
                HostStageState.OFFERED.value,
                HostStageState.HOST_RESERVED.value,
            }:
                return None, False
            if current.get("p_owner") not in {None, owner}:
                return None, False
            claimed = set(int(value) for value in current.get("claimed_ranks", []))
            claimed.add(int(tp_rank))
            current["p_owner"] = owner
            current["claimed_ranks"] = sorted(claimed)
            current["state"] = HostStageState.HOST_RESERVED.value
            current["updated_at"] = time.time()
            return dict(current), True

        return self._mutate(callback, event_snapshot_id=snapshot_id)

    def prepare_p2d_write_rank(
        self,
        snapshot_id: str,
        owner: str,
        grant: dict[str, Any],
        *,
        tp_rank: int,
        tp_size: int,
    ) -> Optional[dict[str, Any]]:
        """Publish one already-reserved Host extent without taking KV ownership.

        Every TP producer first reserves its local NUMA arena extent and
        publishes that immutable grant.  The offer remains rejectable by the
        Router until all ranks are prepared; only then may
        :meth:`claim_p2d_write_rank` atomically move the whole logical
        request-generation under Host ownership.  This prevents one rank from
        claiming P KV while a peer rank has no Host capacity.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state") != HostStageState.OFFERED.value
                or current.get("p_owner") is not None
                or current.get("claimed_ranks")
            ):
                return None, False
            rank_key = str(int(tp_rank))
            normalized = dict(grant, tp_rank=int(tp_rank))
            prepared = current.setdefault("prepared_rank_grants", {})
            previous = prepared.get(rank_key)
            if previous is not None and previous != normalized:
                return None, False
            prepared[rank_key] = normalized
            current["prepared_ranks"] = sorted(
                int(rank) for rank in prepared
            )
            current["updated_at"] = time.time()
            return dict(current), True

        return self._mutate(callback, event_snapshot_id=snapshot_id)

    def claim_p2d_write_rank(
        self,
        snapshot_id: str,
        owner: str,
        grant: dict[str, Any],
        *,
        tp_rank: int,
        tp_size: int,
    ) -> Optional[dict[str, Any]]:
        """Atomically give one P->D shard and its Host extent to staging.

        A P->D writer must not expose an intermediate ``HOST_RESERVED`` state
        without also publishing the immutable physical extent that it owns.
        Native transfer arbitration and Router cleanup run in other processes;
        splitting claim and grant into two ledger mutations lets one of those
        processes terminate the offer between the mutations.  The P worker
        would then own GPU/Host resources for a snapshot whose grant can no
        longer be committed.

        This operation is the ownership boundary for one TP shard.  Before it
        returns, native transfer may still win.  After it returns successfully,
        Host owns the complete request-generation snapshot and all later code
        may only wait for the physical write or fail it closed.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or int(current.get("tp_size", 1)) != int(tp_size):
                return None, False
            if current.get("state") not in {
                HostStageState.OFFERED.value,
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
            }:
                return None, False
            if current.get("p_owner") not in {None, owner}:
                return None, False

            rank_key = str(int(tp_rank))
            normalized = dict(grant, tp_rank=int(tp_rank))
            prepared = current.get("prepared_rank_grants", {})
            if len(prepared) != int(tp_size) or prepared.get(rank_key) != normalized:
                return None, False
            rank_grants = current.setdefault("rank_grants", {})
            previous = rank_grants.get(rank_key)
            if previous is not None and previous != normalized:
                return None, False

            claimed = set(int(value) for value in current.get("claimed_ranks", []))
            claimed.add(int(tp_rank))
            rank_grants[rank_key] = normalized
            current["p_owner"] = owner
            current["claimed_ranks"] = sorted(claimed)
            current["grants"] = [
                dict(rank_grants[str(rank)])
                for rank in range(int(tp_size))
                if str(rank) in rank_grants
            ]
            # HOST_RESERVED means Host already owns every claimed shard, while
            # peer TP ranks may still join the same transaction.  Only the
            # complete grant set advances to HOST_WRITING.  ``p_owner`` and
            # ``claimed_ranks`` make both states non-preemptible by native
            # transfer or Router cleanup.
            current["state"] = (
                HostStageState.HOST_WRITING.value
                if len(rank_grants) == int(tp_size)
                else HostStageState.HOST_RESERVED.value
            )
            current["updated_at"] = time.time()
            return dict(current), True

        return self._mutate(callback, event_snapshot_id=snapshot_id)

    def reject_unclaimed_offer(
        self, snapshot_id: str, *, reason: Optional[str] = None
    ) -> bool:
        """Atomically reject only an offer that no P rank has claimed."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or current.get("state") != HostStageState.OFFERED.value
                or current.get("p_owner") is not None
                or current.get("claimed_ranks")
            ):
                return False, False
            current["state"] = HostStageState.REJECTED.value
            current["updated_at"] = time.time()
            if reason:
                current["reason"] = str(reason)[:256]
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def abort_unsubmitted_p2d(
        self, snapshot_id: str, *, reason: Optional[str] = None
    ) -> str:
        """Cancel P->D Host ownership before a Decode request is submitted.

        The Router owns this boundary. An untouched offer can be rejected
        immediately. A claimed writer must first enter ABORTING so the P-side
        staging worker keeps the source pages and Host extent until its CUDA
        fence drains; that worker then publishes FAILED and releases storage.
        HOST_READY has no live writer and can fail directly. H2D_LOADING means
        responsibility has already crossed to Decode and is never cancelled
        by this pre-submit operation.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None:
                return "missing", False
            state = current.get("state")
            changed = False
            if (
                state == HostStageState.OFFERED.value
                and current.get("p_owner") is None
                and not current.get("claimed_ranks")
            ):
                current["state"] = HostStageState.REJECTED.value
                state = current["state"]
                changed = True
            elif state in {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
            }:
                current["state"] = HostStageState.ABORTING.value
                state = current["state"]
                changed = True
            elif state == HostStageState.HOST_READY.value:
                current["state"] = HostStageState.FAILED.value
                state = current["state"]
                changed = True
            if changed:
                current["updated_at"] = time.time()
                if reason:
                    current["reason"] = str(reason)[:256]
            return str(state), changed

        return str(self._mutate(callback, event_snapshot_id=snapshot_id))

    def arbitrate_p2d_release(self, snapshot_id: str, *, tp_size: int) -> str:
        """Resolve whether one untouched P shard may release its source pages.

        A minimal REJECTED tombstone is created even when D has not published
        its offer yet.  This makes the decision group-wide across independent
        TP processes: a later Host offer/claim cannot split the KV shards.

        ``host_terminal`` is distinct from ``host_owned``.  If another TP rank
        claimed Host but the group subsequently failed/aborted before this rank
        started D2H, this rank's source pages were never exposed to Host I/O and
        are safe to release.  Treating both outcomes as False would retain the
        untouched shard forever with no local worker able to consume failure.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            now = time.time()
            if current is None:
                entries[snapshot_id] = {
                    "snapshot_id": snapshot_id,
                    "state": HostStageState.REJECTED.value,
                    "native_won": True,
                    "tp_size": int(tp_size),
                    "created_at": now,
                    "updated_at": now,
                }
                return P2D_RELEASE_NATIVE_WON, True
            if current.get("state") == HostStageState.REJECTED.value:
                return P2D_RELEASE_NATIVE_WON, False
            if current.get("state") in {
                HostStageState.ABORTING.value,
                HostStageState.FAILED.value,
            }:
                return P2D_RELEASE_HOST_TERMINAL, False
            if current.get("p_owner") is not None or current.get("claimed_ranks"):
                return P2D_RELEASE_HOST_OWNED, False
            if current.get("state") not in {
                HostStageState.OFFERED.value,
                "tp_collecting",
            }:
                return P2D_RELEASE_HOST_OWNED, False
            current["state"] = HostStageState.REJECTED.value
            current["native_won"] = True
            current["updated_at"] = now
            return P2D_RELEASE_NATIVE_WON, True

        return str(self._mutate(callback, event_snapshot_id=snapshot_id))

    def arbitrate_p2d_native(self, snapshot_id: str, *, tp_size: int) -> bool:
        """Compatibility bool for callers that only distinguish native/Host."""

        return (
            self.arbitrate_p2d_release(snapshot_id, tp_size=tp_size)
            == P2D_RELEASE_NATIVE_WON
        )

    def publish_rank_grant(
        self,
        snapshot_id: str,
        owner: str,
        grant: dict[str, Any],
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or current.get("p_owner") != owner
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.HOST_RESERVED.value,
                    HostStageState.HOST_WRITING.value,
                }
            ):
                return False, False
            rank_grants = current.setdefault("rank_grants", {})
            rank_key = str(int(tp_rank))
            normalized = dict(grant, tp_rank=int(tp_rank))
            previous = rank_grants.get(rank_key)
            if previous is not None and previous != normalized:
                return False, False
            rank_grants[rank_key] = normalized
            current["grants"] = [
                dict(rank_grants[str(rank)])
                for rank in range(int(tp_size))
                if str(rank) in rank_grants
            ]
            if len(rank_grants) == int(tp_size):
                writer_acks = {
                    int(value) for value in current.get("writer_acks", [])
                }
                current["state"] = (
                    HostStageState.HOST_READY.value
                    if len(writer_acks) == int(tp_size)
                    else HostStageState.HOST_WRITING.value
                )
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

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

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def complete_host_write(
        self,
        snapshot_id: str,
        d_pid: int,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> bool:
        """Atomically publish a complete D-written shared-Host snapshot."""

        def callback(entries):
            current = entries.get(snapshot_id)
            rank_offer = current.get("rank_offers", {}).get(str(int(tp_rank)), {}) if current else {}
            owner_pid = (
                int(current.get("d_pid", -1))
                if int(tp_size) == 1 and current is not None
                else int(rank_offer.get("d_pid", -1))
            )
            if current is None or owner_pid != int(d_pid):
                return False, False
            if current.get("state") == HostStageState.HOST_READY.value:
                return True, False
            if current.get("state") != HostStageState.HOST_WRITING.value:
                return False, False
            grants = current.get("grants", [])
            if len(grants) != int(tp_size) or any(
                grant.get("kind") != "shared_host_extent" for grant in grants
            ):
                return False, False
            writer_acks = set(int(value) for value in current.get("writer_acks", []))
            writer_acks.add(int(tp_rank))
            current["writer_acks"] = sorted(writer_acks)
            if len(writer_acks) == int(tp_size):
                current["sent_chunks"] = list(range(int(tp_size)))
                current["acked_chunks"] = list(range(int(tp_size)))
                current["state"] = HostStageState.HOST_READY.value
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def complete_host_load_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """Publish CONSUMED after every P2D destination rank loaded Host."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("p_owner") != owner:
                return False, False
            if current.get("state") == HostStageState.CONSUMED.value:
                return True, False
            if current.get("state") not in {
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
            }:
                return False, False
            loader_acks = set(int(value) for value in current.get("loader_acks", []))
            loader_acks.add(int(tp_rank))
            current["loader_acks"] = sorted(loader_acks)
            if len(loader_acks) == int(tp_size):
                current["state"] = HostStageState.CONSUMED.value
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def request_host_load_failure(
        self,
        snapshot_id: str,
        owner: str,
        *,
        reason: str,
        recovery_domain: Optional[int] = None,
    ) -> bool:
        """Begin a TP-wide P2D Host-load abort without releasing Host data.

        This only publishes failure intent.  The source Host extents remain
        authoritative until every destination rank separately confirms that
        no H2D DMA can still read them via :meth:`mark_host_load_rank_drained`.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or not self._host_load_abort_requester_matches(
                current, owner, recovery_domain=recovery_domain
            ):
                return False, False
            state = current.get("state")
            if state == HostStageState.FAILED.value:
                return True, False
            if state == HostStageState.ABORTING.value:
                return (
                    bool(current.get("h2d_abort_started", False))
                    and current.get("h2d_abort_owner") in {None, owner}
                ), False
            if state not in {
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
                HostStageState.HBM_READY.value,
            }:
                return False, False
            if state == HostStageState.HBM_READY.value and any(
                claim.get("phase") == "handed"
                for claim in current.get("recovery_claims", {}).values()
            ):
                return False, False
            current["state"] = HostStageState.ABORTING.value
            current["h2d_abort_started"] = True
            current["h2d_abort_owner"] = owner
            if recovery_domain is not None:
                current["h2d_abort_recovery_domain"] = int(recovery_domain)
            current["loader_failure_reason"] = str(reason)[:256]
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def mark_host_load_rank_drained(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """ACK one destination rank is physically quiescent after H2D abort.

        FAILED is deliberately group-committed only after all TP ranks ACK.
        A missing CUDA completion fence therefore leaks the snapshot closed
        instead of allowing the source extent to be reused under live DMA.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or not self._host_load_abort_owner_matches(current, owner)
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.ABORTING.value,
                    HostStageState.FAILED.value,
                }
                or not current.get("h2d_abort_started", False)
            ):
                return False, False
            if current.get("state") == HostStageState.FAILED.value:
                return True, False
            drained = set(
                int(value) for value in current.get("loader_drained_ranks", [])
            )
            changed = int(tp_rank) not in drained
            drained.add(int(tp_rank))
            current["loader_drained_ranks"] = sorted(drained)
            if len(drained) == int(tp_size):
                if current.get("recovery_claims"):
                    return False, False
                current["state"] = HostStageState.FAILED.value
                current["recovery_claim_id"] = None
                current["recovery_claims"] = {}
                current["recovery_owner"] = None
                current["reason"] = current.get(
                    "loader_failure_reason", "p2d_h2d_failed"
                )
            current["updated_at"] = time.time()
            return True, changed

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def complete_d2p_host_load_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """Publish HBM_READY after every D2P shard has reached P HBM."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or not self._d2p_recovery_owner_matches(
                current, owner
            ):
                return False, False
            if current.get("state") in {
                HostStageState.HBM_READY.value,
                HostStageState.CONSUMED.value,
            }:
                return True, False
            if current.get("state") not in {
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
            }:
                return False, False
            loader_acks = set(int(value) for value in current.get("loader_acks", []))
            loader_acks.add(int(tp_rank))
            current["loader_acks"] = sorted(loader_acks)
            if len(loader_acks) == int(tp_size):
                current["state"] = HostStageState.HBM_READY.value
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def complete_host_bind_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """Publish CONSUMED only after every restored shard is Radix-bound."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or not self._d2p_recovery_owner_matches(
                current, owner
            ):
                return False, False
            if current.get("state") == HostStageState.CONSUMED.value:
                return True, False
            if current.get("state") != HostStageState.HBM_READY.value:
                return False, False
            binder_acks = set(
                int(value) for value in current.get("binder_acks", [])
            )
            binder_acks.add(int(tp_rank))
            current["binder_acks"] = sorted(binder_acks)
            if len(binder_acks) == int(tp_size):
                current["state"] = HostStageState.CONSUMED.value
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def request_d2p_retry(
        self, snapshot_id: str, owner: str, *, reason: str
    ) -> bool:
        """Begin a TP-wide retry while retaining the complete Host snapshot."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or not self._d2p_recovery_owner_matches(
                current, owner
            ):
                return False, False
            if current.get("state") == HostStageState.RETRY_PENDING.value:
                return True, False
            if current.get("state") not in {
                HostStageState.H2D_LOADING.value,
                HostStageState.HBM_READY.value,
                HostStageState.RETRY_PENDING.value,
            }:
                return False, False
            current["state"] = HostStageState.RETRY_PENDING.value
            current["retry_acks"] = []
            current["retry_reason"] = str(reason)[:256]
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def complete_d2p_retry_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """Re-arm Host recovery after every rank has quiesced and rolled back."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or not self._d2p_recovery_owner_matches(current, owner)
                or current.get("state")
                not in {
                    HostStageState.RETRY_PENDING.value,
                    HostStageState.HOST_READY.value,
                }
            ):
                return False, False
            if current.get("state") == HostStageState.HOST_READY.value:
                return True, False
            retry_acks = set(
                int(value) for value in current.get("retry_acks", [])
            )
            retry_acks.add(int(tp_rank))
            current["retry_acks"] = sorted(retry_acks)
            if len(retry_acks) == int(tp_size):
                current["state"] = HostStageState.HOST_READY.value
                current["recovery_claim_id"] = None
                current["recovery_claims"] = {}
                for key in (
                    "loading_ranks",
                    "h2d_prepared_ranks",
                    "loader_acks",
                    "binder_acks",
                    "retry_acks",
                ):
                    current[key] = []
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def begin_host_load_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
        recovery: bool = False,
    ) -> bool:
        """Atomically admit one TP rank into Host->device loading.

        HOST_READY is a group-level state.  The first local rank that starts
        loading advances it to H2D_LOADING; peer ranks must still be allowed
        to join that same load instead of treating the state change as a
        rejection.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or (
                    not self._d2p_recovery_owner_matches(current, owner)
                    if recovery
                    else current.get("p_owner") != owner
                )
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.HOST_READY.value,
                    HostStageState.H2D_LOADING.value,
                }
            ):
                return False, False
            loading_ranks = set(
                int(value) for value in current.get("loading_ranks", [])
            )
            loading_ranks.add(int(tp_rank))
            current["loading_ranks"] = sorted(loading_ranks)
            current["state"] = HostStageState.H2D_LOADING.value
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def assign_d2p_recovery_domain(
        self, snapshot_id: str, domain: int, *, lease_seconds: float = 120.0
    ) -> bool:
        """Lease a Host-ready generation to one independently selected P."""

        domain = int(domain)

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("state") not in {
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
            }:
                return False, False
            previous = current.get("recovery_domain")
            if previous is not None and int(previous) != domain:
                return False, False
            expires_at = time.time() + max(1.0, float(lease_seconds))
            changed = previous is None or float(
                current.get("recovery_assignment_expires_at", 0.0)
            ) < expires_at
            current["recovery_domain"] = domain
            current["recovery_assignment_expires_at"] = expires_at
            if changed:
                current["updated_at"] = time.time()
            return True, changed

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    @staticmethod
    def _d2p_recovery_owner_matches(
        current: dict[str, Any], owner: str
    ) -> bool:
        recovery_owner = current.get("recovery_owner")
        if recovery_owner is not None:
            return recovery_owner == owner
        # A Router-assigned generation belongs to that logical P domain even
        # before its per-process recovery claim is published.  The physical
        # Shared-Arena owner is only the legacy/unassigned fallback owner.
        return current.get("recovery_domain") is None and current.get(
            "p_owner"
        ) == owner

    @staticmethod
    def _host_load_abort_requester_matches(
        current: dict[str, Any],
        owner: str,
        *,
        recovery_domain: Optional[int] = None,
    ) -> bool:
        recovery_owner = current.get("recovery_owner")
        if recovery_owner is not None:
            return recovery_owner == owner
        assigned_domain = current.get("recovery_domain")
        if assigned_domain is not None:
            return recovery_domain is not None and int(assigned_domain) == int(
                recovery_domain
            )
        return current.get("p_owner") == owner

    @staticmethod
    def _host_load_abort_owner_matches(
        current: dict[str, Any], owner: str
    ) -> bool:
        abort_owner = current.get("h2d_abort_owner")
        if abort_owner is not None:
            return abort_owner == owner
        return SharedHostStagingLedger._d2p_recovery_owner_matches(
            current, owner
        )

    def claim_d2p_recovery_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
        claim_id: str,
        recovery_domain: Optional[int] = None,
    ) -> bool:
        """Pin a D->P Host snapshot before allocating any P workset pages.

        This is the request-generation ownership boundary between pressure
        eviction and Slow recovery.  Once any TP rank has published the same
        logical claim, the generation is no longer ``HOST_READY`` and is
        therefore invisible to eviction.  Workset allocation happens only
        after this CAS succeeds.
        """

        claim_id = str(claim_id)

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.HOST_READY.value,
                    HostStageState.H2D_LOADING.value,
                }
            ):
                return False, False
            selected_domain = current.get("recovery_domain")
            if selected_domain is None:
                # Static single-P routing has no Router assignment.  Only
                # the physical Host owner may claim that unassigned record;
                # commit its domain together with the pin, under this CAS.
                if current.get("p_owner") != owner:
                    return False, False
                arena_domain = current.get("arena_domain")
                if (
                    recovery_domain is not None
                    and arena_domain is not None
                    and int(arena_domain) != int(recovery_domain)
                ):
                    return False, False
            elif recovery_domain is None or int(selected_domain) != int(recovery_domain):
                return False, False
            recovery_owner = current.get("recovery_owner")
            if recovery_owner not in {None, owner}:
                return False, False
            recovery_claim_id = current.get("recovery_claim_id")
            if recovery_claim_id not in {None, claim_id}:
                return False, False
            claims = dict(current.get("recovery_claims", {}))
            rank_key = str(int(tp_rank))
            previous = claims.get(rank_key)
            if previous is not None and previous.get("claim_id") != claim_id:
                return False, False
            changed = previous is None
            claims[rank_key] = {
                "claim_id": claim_id,
                "phase": "pinned",
                "lease_id": None,
            }
            current["recovery_claim_id"] = claim_id
            current["recovery_owner"] = owner
            if recovery_domain is not None:
                current["recovery_domain"] = int(recovery_domain)
            current["recovery_claims"] = claims
            current["state"] = HostStageState.H2D_LOADING.value
            if changed:
                current["updated_at"] = time.time()
            return True, changed

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def attach_d2p_recovery_lease_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
        claim_id: str,
        lease_id: int,
    ) -> bool:
        """Attach the exact P workset lease to an already pinned snapshot."""

        claim_id = str(claim_id)
        lease_id = int(lease_id)

        def callback(entries):
            current = entries.get(snapshot_id)
            claims = {} if current is None else dict(
                current.get("recovery_claims", {})
            )
            rank_key = str(int(tp_rank))
            claim = claims.get(rank_key)
            if (
                current is None
                or not self._d2p_recovery_owner_matches(current, owner)
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state") != HostStageState.H2D_LOADING.value
                or current.get("recovery_claim_id") != claim_id
                or claim is None
                or claim.get("claim_id") != claim_id
            ):
                return False, False
            previous_lease_id = claim.get("lease_id")
            if previous_lease_id not in {None, lease_id}:
                return False, False
            changed = previous_lease_id is None or claim.get("phase") != "leased"
            claim = dict(claim)
            claim.update(lease_id=lease_id, phase="leased")
            claims[rank_key] = claim
            current["recovery_claims"] = claims
            if changed:
                current["updated_at"] = time.time()
            return True, changed

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def mark_d2p_recovery_phase_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
        claim_id: str,
        lease_id: int,
        phase: str,
    ) -> bool:
        """Advance the ledger-owned recovery/lease lifecycle monotonically."""

        phases = {"leased": 0, "io_inflight": 1, "handed": 2}
        if phase not in phases:
            raise ValueError(f"invalid D->P recovery phase {phase}")
        claim_id = str(claim_id)
        lease_id = int(lease_id)

        def callback(entries):
            current = entries.get(snapshot_id)
            claims = {} if current is None else dict(
                current.get("recovery_claims", {})
            )
            rank_key = str(int(tp_rank))
            claim = claims.get(rank_key)
            if (
                current is None
                or not self._d2p_recovery_owner_matches(current, owner)
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.H2D_LOADING.value,
                    HostStageState.HBM_READY.value,
                    HostStageState.CONSUMED.value,
                }
                or claim is None
                or claim.get("claim_id") != claim_id
                or int(claim.get("lease_id", -1)) != lease_id
            ):
                return False, False
            old_phase = str(claim.get("phase", "leased"))
            if old_phase not in phases or phases[old_phase] > phases[phase]:
                return False, False
            if old_phase == phase:
                return True, False
            claim = dict(claim)
            claim["phase"] = phase
            claims[rank_key] = claim
            current["recovery_claims"] = claims
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def cancel_d2p_recovery_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
        claim_id: str,
        lease_id: Optional[int] = None,
    ) -> bool:
        """Roll back a pinned recovery after its exact workset was cancelled.

        I/O-owned or handed leases cannot be cancelled through this method.
        The caller must first prove transport quiescence and use the normal
        terminal path.  Once every TP rank has rolled back, Host ownership is
        returned to ``HOST_READY`` and pressure eviction may see it again.
        """

        claim_id = str(claim_id)

        def callback(entries):
            current = entries.get(snapshot_id)
            claims = {} if current is None else dict(
                current.get("recovery_claims", {})
            )
            rank_key = str(int(tp_rank))
            claim = claims.get(rank_key)
            if (
                current is None
                or not self._d2p_recovery_owner_matches(current, owner)
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.H2D_LOADING.value,
                    HostStageState.ABORTING.value,
                }
                or current.get("recovery_claim_id") != claim_id
                or claim is None
                or claim.get("claim_id") != claim_id
                or claim.get("phase") == "handed"
                or (
                    claim.get("phase") == "io_inflight"
                    and current.get("state") != HostStageState.ABORTING.value
                )
            ):
                return False, False
            attached_lease_id = claim.get("lease_id")
            if lease_id is not None and attached_lease_id not in {
                None,
                int(lease_id),
            }:
                return False, False
            claims.pop(rank_key, None)
            current["recovery_claims"] = claims
            if (
                not claims
                and current.get("state") == HostStageState.H2D_LOADING.value
            ):
                current["recovery_claim_id"] = None
                current["recovery_owner"] = None
                current.pop("recovery_domain", None)
                current.pop("recovery_assignment_expires_at", None)
                current["state"] = HostStageState.HOST_READY.value
                for key in (
                    "loading_ranks",
                    "h2d_prepared_ranks",
                    "loader_acks",
                    "binder_acks",
                ):
                    current[key] = []
            elif not claims:
                current["recovery_claim_id"] = None
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def prepare_tp_host_load_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """ACK that one P rank reserved all resources needed for Host H2D.

        The copy itself must not start until every TP rank has validated its
        local snapshot and reserved destination pages.  Otherwise one rank can
        enter Host recovery while a peer continues Direct/Prefill work, which
        eventually diverges their model-forward collectives.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or not self._d2p_recovery_owner_matches(current, owner)
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.HOST_READY.value,
                    HostStageState.H2D_LOADING.value,
                }
            ):
                return False, False
            prepared = set(
                int(value) for value in current.get("h2d_prepared_ranks", [])
            )
            changed = int(tp_rank) not in prepared
            prepared.add(int(tp_rank))
            current["h2d_prepared_ranks"] = sorted(prepared)
            # HOST_READY is the level-trigger that lets every P rank move its
            # local extent from the async admission table into host_ready.  A
            # fast rank must not hide that trigger by publishing H2D_LOADING
            # before its peer has observed it.  Advance the global state only
            # after the complete TP group has prepared destination pages.
            if len(prepared) == int(tp_size):
                current["state"] = HostStageState.H2D_LOADING.value
            if changed:
                current["updated_at"] = time.time()
            return True, changed

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def complete_p2d_host_write_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """Commit P->D Host visibility after every P rank wrote its shard."""

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("p_owner") != owner:
                return False, False
            if current.get("state") == HostStageState.HOST_READY.value:
                return True, False
            if current.get("state") not in {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
            }:
                return False, False
            rank_grants = current.get("rank_grants", {})
            if str(int(tp_rank)) not in rank_grants:
                return False, False
            writer_acks = set(int(value) for value in current.get("writer_acks", []))
            writer_acks.add(int(tp_rank))
            current["writer_acks"] = sorted(writer_acks)
            if (
                len(rank_grants) == int(tp_size)
                and len(writer_acks) == int(tp_size)
            ):
                current["state"] = HostStageState.HOST_READY.value
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def mark_writer_drained(self, snapshot_id: str, d_pid: int) -> bool:
        """ACK one rank has no DMA that can still target an aborting extent."""

        return self.mark_writer_rank_drained(
            snapshot_id, d_pid, tp_rank=0, tp_size=1
        )

    def mark_writer_rank_drained(
        self,
        snapshot_id: str,
        d_pid: int,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """Publish group drain only after every D TP rank is quiescent."""

        def callback(entries):
            current = entries.get(snapshot_id)
            rank_offer = (
                current.get("rank_offers", {}).get(str(int(tp_rank)), {})
                if current
                else {}
            )
            owner_pid = (
                int(current.get("d_pid", -1))
                if int(tp_size) == 1 and current is not None
                else int(rank_offer.get("d_pid", -1))
            )
            if (
                current is None
                or owner_pid != int(d_pid)
                or int(current.get("tp_size", 1)) != int(tp_size)
            ):
                return False, False
            if current.get("state") != HostStageState.ABORTING.value:
                return False, False
            drained = set(
                int(value) for value in current.get("writer_drained_ranks", [])
            )
            changed = int(tp_rank) not in drained
            drained.add(int(tp_rank))
            current["writer_drained_ranks"] = sorted(drained)
            current["writer_drained"] = len(drained) == int(tp_size)
            current["updated_at"] = time.time()
            return True, changed

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def fail_host_write(
        self,
        snapshot_id: str,
        d_pid: int,
        reason: str,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> bool:
        """Fail one D rank closed and wait for all peer DMAs to drain."""

        def callback(entries):
            current = entries.get(snapshot_id)
            rank_offer = (
                current.get("rank_offers", {}).get(str(int(tp_rank)), {})
                if current
                else {}
            )
            owner_pid = (
                int(current.get("d_pid", -1))
                if int(tp_size) == 1 and current is not None
                else int(rank_offer.get("d_pid", -1))
            )
            if (
                current is None
                or owner_pid != int(d_pid)
                or int(current.get("tp_size", 1)) != int(tp_size)
            ):
                return False, False
            if current.get("state") not in {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
                HostStageState.ABORTING.value,
            }:
                return False, False
            current["state"] = HostStageState.ABORTING.value
            drained = set(
                int(value) for value in current.get("writer_drained_ranks", [])
            )
            drained.add(int(tp_rank))
            current["writer_drained_ranks"] = sorted(drained)
            current["writer_drained"] = len(drained) == int(tp_size)
            current["reason"] = str(reason)[:256]
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

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

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

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

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

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

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def begin_host_eviction(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_size: int,
        reason: str,
        terminal_on_release: bool = False,
    ) -> bool:
        """Atomically claim one complete HOST_READY generation for eviction.

        A load and an eviction compete through the same per-generation ledger
        mutation.  Any prepared/loading rank makes the snapshot ineligible, so
        a pressure worker can never remove pages underneath Host->P DMA or a
        scheduler-owned Radix bind.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or current.get("p_owner") != owner
                or int(current.get("tp_size", 1)) != int(tp_size)
            ):
                return False, False
            state = current.get("state")
            if state in {
                HostStageState.EVICTING.value,
                HostStageState.RECOMPUTE_REQUIRED.value,
            }:
                return bool(current.get("terminal_on_release", False)) == terminal_on_release, False
            if state != HostStageState.HOST_READY.value:
                return False, False
            # Recovery assignment is the ownership fence before the selected
            # P has published its process/rank claim.  Evicting in that gap
            # can remove the only Host copy underneath the routed child.
            if any(
                current.get(field)
                for field in (
                    "recovery_owner",
                    "recovery_claims",
                    "loading_ranks",
                    "h2d_prepared_ranks",
                    "loader_acks",
                    "binder_acks",
                )
            ):
                return False, False
            if current.get("recovery_domain") is not None:
                if float(current.get("recovery_assignment_expires_at", 0.0)) > time.time():
                    return False, False
                current.pop("recovery_domain", None)
                current.pop("recovery_assignment_expires_at", None)
            current["state"] = HostStageState.EVICTING.value
            current["eviction_acks"] = []
            current["eviction_reason"] = str(reason)[:256]
            current["eviction_started_at"] = time.time()
            if terminal_on_release:
                current["terminal_on_release"] = True
            current["updated_at"] = time.time()
            return True, True

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def complete_host_eviction_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """ACK a released Host shard; commit recompute or application terminal.

        There is deliberately no timeout that converts a partially ACKed TP
        eviction into recomputation.  If one TP rank dies, the model/NCCL
        group is already fail-stop; advancing this state while a surviving
        process may still own a shard would permit use-after-free.  Normal
        process teardown removes the run-scoped ledger and arena together.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if (
                current is None
                or current.get("p_owner") != owner
                or int(current.get("tp_size", 1)) != int(tp_size)
            ):
                return False, False
            if current.get("state") == HostStageState.RECOMPUTE_REQUIRED.value or (
                current.get("state") == HostStageState.CONSUMED.value
                and current.get("terminal_on_release")
            ):
                return True, False
            if current.get("state") != HostStageState.EVICTING.value:
                return False, False
            acknowledgements = {
                int(value) for value in current.get("eviction_acks", [])
            }
            changed = int(tp_rank) not in acknowledgements
            acknowledgements.add(int(tp_rank))
            current["eviction_acks"] = sorted(acknowledgements)
            if len(acknowledgements) == int(tp_size):
                current["state"] = (
                    HostStageState.CONSUMED.value
                    if current.get("terminal_on_release")
                    else HostStageState.RECOMPUTE_REQUIRED.value
                )
                current["evicted_at"] = time.time()
                current["reason"] = current.get(
                    "eviction_reason", "shared_host_pressure_eviction"
                )
            current["updated_at"] = time.time()
            return True, changed

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

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
            if current is None or (
                owner is not None
                and owner
                not in {current.get("p_owner"), current.get("recovery_owner")}
            ):
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

        return bool(self._mutate(callback, event_snapshot_id=snapshot_id))

    def prune(
        self,
        older_than_seconds: float = 600.0,
        consumed_older_than_seconds: float = 5.0,
    ) -> None:
        cutoff = time.time() - max(0.0, older_than_seconds)
        consumed_cutoff = time.time() - max(0.0, consumed_older_than_seconds)

        candidates = self.snapshot_entries(force_refresh=True)
        for snapshot_id, value in candidates.items():
            doomed = (
                value.get("state") == HostStageState.CONSUMED.value
                and float(value.get("updated_at", 0.0)) < consumed_cutoff
            ) or (
                value.get("state") in _TERMINAL_STATES
                and float(value.get("updated_at", 0.0)) < cutoff
            )
            if not doomed:
                continue
            if self._is_relay_snapshot(snapshot_id):
                with open(self.path, "r+", encoding="utf-8") as file_obj:
                    fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
                    try:
                        # Relay ownership is switched under global->stripe,
                        # the same order used by assign_transfer_path.  Delete
                        # the authoritative entry and its mirror while both
                        # locks are held so no writer can resurrect stale
                        # relay state through the read-through fallback.
                        with self._entry_locked(snapshot_id):
                            file_obj.seek(0)
                            data = json.loads(file_obj.read() or "{}")
                            current = data.setdefault("entries", {}).get(snapshot_id)
                            if current is None:
                                continue
                            still_doomed = (
                                current.get("state")
                                == HostStageState.CONSUMED.value
                                and float(current.get("updated_at", 0.0))
                                < consumed_cutoff
                            ) or (
                                current.get("state") in _TERMINAL_STATES
                                and float(current.get("updated_at", 0.0)) < cutoff
                            )
                            if not still_doomed:
                                continue
                            data["entries"].pop(snapshot_id, None)
                            self._write_locked(file_obj, data)
                            try:
                                os.unlink(self._event_path(snapshot_id))
                            except FileNotFoundError:
                                pass
                            try:
                                os.unlink(self._relay_marker_path(snapshot_id))
                            except FileNotFoundError:
                                pass
                    finally:
                        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
                continue
            with self._entry_locked(snapshot_id):
                try:
                    event = self.read_entry_event(self._event_path(snapshot_id))
                except FileNotFoundError:
                    continue
                current = event.get("entry")
                if current is None:
                    continue
                still_doomed = (
                    current.get("state") == HostStageState.CONSUMED.value
                    and float(current.get("updated_at", 0.0)) < consumed_cutoff
                ) or (
                    current.get("state") in _TERMINAL_STATES
                    and float(current.get("updated_at", 0.0)) < cutoff
                )
                if still_doomed:
                    try:
                        os.unlink(self._event_path(snapshot_id))
                    except FileNotFoundError:
                        pass
