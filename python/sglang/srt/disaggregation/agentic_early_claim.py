"""Node-local early-arrival markers for the agentic D-to-P direct path.

The HTTP router publishes a tiny marker as soon as a later agent turn arrives.
Decode workers use the marker only to distinguish a fast tool return from a
slow one.  It allocates no P HBM and carries no global capacity/credit policy.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import queue
import select
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration


_VERSION = 1

# Linux inotify values from <sys/inotify.h>.  Agentic PD V1 already requires
# P and Router to share the same node-local /dev/shm directory, so using
# inotify here avoids adding another control-plane dependency.
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_INOTIFY_EVENT = struct.Struct("iIII")
_LIBC = ctypes.CDLL(None, use_errno=True)
_INOTIFY_INIT1 = getattr(_LIBC, "inotify_init1", None)
_INOTIFY_ADD_WATCH = getattr(_LIBC, "inotify_add_watch", None)
if _INOTIFY_INIT1 is not None:
    _INOTIFY_INIT1.argtypes = [ctypes.c_int]
    _INOTIFY_INIT1.restype = ctypes.c_int
if _INOTIFY_ADD_WATCH is not None:
    _INOTIFY_ADD_WATCH.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    _INOTIFY_ADD_WATCH.restype = ctypes.c_int


def _inotify_init() -> int:
    if _INOTIFY_INIT1 is None:
        raise RuntimeError("agentic Direct arrival watching requires Linux inotify")
    fd = _INOTIFY_INIT1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _inotify_add_watch(fd: int, path: Path, mask: int) -> int:
    if _INOTIFY_ADD_WATCH is None:
        raise RuntimeError("agentic Direct arrival watching requires Linux inotify")
    descriptor = _INOTIFY_ADD_WATCH(fd, os.fsencode(path), mask)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), path)
    return descriptor


class _SharedControlPoller:
    """Bounded background resync for remote writes invisible to inotify.

    This is a metadata-only prototype bridge, not a KV data transport. A
    filesystem with verified cross-client locking/coherency is mandatory.
    No poll ever grants capacity or transfers snapshot ownership.
    """

    def __init__(self, interval: float):
        self.interval = interval
        self.next_scan = 0.0
        self.stopped = threading.Event()

    def due(self, timeout: float | None) -> bool:
        if self.stopped.is_set():
            return False
        remaining = max(0.0, self.next_scan - time.monotonic())
        wait = remaining if timeout is None else min(remaining, max(0.0, timeout))
        if self.stopped.wait(wait):
            return False
        now = time.monotonic()
        if now < self.next_scan:
            return False
        self.next_scan = now + self.interval
        return True


def _shared_control_poller():
    from sglang.srt.disaggregation.agentic_multinode import control_poll_interval

    interval = control_poll_interval()
    return None if interval is None else _SharedControlPoller(interval)


class AgenticArrivalWatcher:
    """Event-driven reader for Router arrival markers.

    The watch is installed before the one-time startup scan, so a marker
    created concurrently with startup is either found by that scan or remains
    queued in the inotify fd.  Normal operation reads only paths named by
    inotify; a full scan is used again solely after kernel queue overflow.
    """

    _shared_poll = None

    def __init__(self, store: "AgenticEarlyClaimStore", max_age_seconds: float):
        self.store = store
        self.max_age_seconds = float(max_age_seconds)
        self._shared_poll = _shared_control_poller()
        if self._shared_poll is not None:
            self._closed = False
            self._seen_arrivals = {}
            # Capture the append cursor BEFORE the one-time recovery scan:
            # an arrival racing that scan is replayed, never lost. Normal
            # polling reads only new fixed-size event records, not retained
            # TP tombstones or every historical marker on NFS.
            self._journal_cursor = store._arrival_journal_size()
            self._startup = store.iter_arrivals(max_age_seconds=self.max_age_seconds)
            return
        self.fd = _inotify_init()
        try:
            self.watch_descriptor = _inotify_add_watch(
                self.fd,
                store.marker_directory,
                _IN_CLOSE_WRITE
                | _IN_MOVED_TO
                | _IN_CREATE
                | _IN_DELETE_SELF
                | _IN_MOVE_SELF,
            )
        except Exception:
            os.close(self.fd)
            raise
        self.poller = select.poll()
        self.poller.register(self.fd, select.POLLIN | select.POLLERR)
        self._startup = store.iter_arrivals(max_age_seconds=self.max_age_seconds)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._shared_poll is not None:
            self._shared_poll.stopped.set()
            return
        try:
            self.poller.unregister(self.fd)
        except (KeyError, OSError):
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass

    def poll(
        self, timeout_seconds: float = 0.0
    ) -> list[tuple[RequestGeneration, dict[str, Any]]]:
        """Return newly published arrivals without rescanning the directory."""

        if self._closed:
            return []
        if self._shared_poll is not None:
            if not self._shared_poll.due(timeout_seconds):
                return []
            paths, self._journal_cursor = self.store._read_arrival_events(
                self._journal_cursor
            )
            current = self._startup
            self._startup = []
            for path in paths:
                item = self.store.read_arrival_path(path, max_age_seconds=self.max_age_seconds)
                if item is not None:
                    current.append(item)
            # Multiple updates of one marker in a batch resolve to its latest
            # authoritative value; the journal is a notification, not a grant.
            current = list({req.snapshot_id: (req, payload) for req, payload in current}.values())
            arrivals = [(request, payload) for request, payload in current
                        if self._seen_arrivals.get(request.snapshot_id) != payload]
            cutoff = time.time() - self.max_age_seconds
            self._seen_arrivals = {sid: p for sid, p in self._seen_arrivals.items()
                                   if float(p["arrived_at"]) >= cutoff}
            self._seen_arrivals.update({req.snapshot_id: p for req, p in current})
            return sorted(arrivals, key=lambda item: float(item[1]["arrived_at"]))
        arrivals = self._startup
        self._startup = []
        timeout_ms = max(0, int(float(timeout_seconds) * 1000.0))
        try:
            ready = self.poller.poll(timeout_ms)
        except OSError:
            return arrivals
        if not ready:
            return arrivals

        paths: set[Path] = set()
        overflow = False
        while True:
            try:
                data = os.read(self.fd, 256 * 1024)
            except BlockingIOError:
                break
            except OSError:
                return arrivals
            if not data:
                break
            offset = 0
            while offset + _INOTIFY_EVENT.size <= len(data):
                _, mask, _, name_length = _INOTIFY_EVENT.unpack_from(data, offset)
                offset += _INOTIFY_EVENT.size
                raw_name = data[offset : offset + name_length]
                offset += name_length
                if mask & _IN_Q_OVERFLOW:
                    overflow = True
                    continue
                if mask & (_IN_IGNORED | _IN_DELETE_SELF | _IN_MOVE_SELF):
                    continue
                name = raw_name.split(b"\0", 1)[0].decode(errors="surrogateescape")
                if (
                    name
                    and not name.startswith(".")
                    and name.endswith(".json")
                    and mask & (_IN_CLOSE_WRITE | _IN_MOVED_TO | _IN_CREATE)
                ):
                    paths.add(self.store.marker_directory / name)

        if overflow:
            arrivals.extend(
                self.store.iter_arrivals(max_age_seconds=self.max_age_seconds)
            )
        else:
            for path in paths:
                item = self.store.read_arrival_path(
                    path, max_age_seconds=self.max_age_seconds
                )
                if item is not None:
                    arrivals.append(item)
        arrivals.sort(key=lambda item: float(item[1]["arrived_at"]))
        return arrivals

    def __enter__(self) -> "AgenticArrivalWatcher":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class AgenticFileChangeWatcher:
    """Block on changes to one node-local control-plane file.

    The file itself may be rewritten in place or atomically replaced. Watching
    its parent directory covers both forms without periodically scanning that
    directory.  Callers retain a low-rate timeout as an overflow/recovery
    backstop; normal progress is edge-triggered by inotify.
    """

    _shared_poll = None

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._shared_poll = _shared_control_poller()
        if self._shared_poll is not None:
            self._closed = False
            self.healthy = True
            return
        self.fd = _inotify_init()
        try:
            self.watch_descriptor = _inotify_add_watch(
                self.fd,
                self.path.parent,
                _IN_CLOSE_WRITE
                | _IN_MOVED_TO
                | _IN_CREATE
                | _IN_DELETE_SELF
                | _IN_MOVE_SELF,
            )
        except Exception:
            os.close(self.fd)
            raise
        self.poller = select.poll()
        self.poller.register(self.fd, select.POLLIN | select.POLLERR)
        self._closed = False
        self.healthy = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.healthy = False
        if self._shared_poll is not None:
            self._shared_poll.stopped.set()
            return
        try:
            self.poller.unregister(self.fd)
        except (KeyError, OSError):
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass

    def poll(self, timeout_seconds: float | None = None) -> bool:
        """Return whether the watched file changed or events overflowed."""

        if self._closed:
            return False
        if self._shared_poll is not None:
            # Treat each tick as invalidation, not an ownership transition.
            return self._shared_poll.due(timeout_seconds)
        timeout_ms = (
            -1
            if timeout_seconds is None
            else max(0, int(float(timeout_seconds) * 1000.0))
        )
        try:
            ready = self.poller.poll(timeout_ms)
        except OSError:
            self.healthy = False
            return True
        if not ready:
            return False

        changed = False
        while True:
            try:
                data = os.read(self.fd, 256 * 1024)
            except BlockingIOError:
                break
            except OSError:
                self.healthy = False
                return True
            if not data:
                break
            offset = 0
            while offset + _INOTIFY_EVENT.size <= len(data):
                _, mask, _, name_length = _INOTIFY_EVENT.unpack_from(data, offset)
                offset += _INOTIFY_EVENT.size
                raw_name = data[offset : offset + name_length]
                offset += name_length
                if mask & _IN_Q_OVERFLOW:
                    changed = True
                    continue
                if mask & (_IN_IGNORED | _IN_DELETE_SELF | _IN_MOVE_SELF):
                    self.healthy = False
                    changed = True
                    continue
                name = raw_name.split(b"\0", 1)[0].decode(
                    errors="surrogateescape"
                )
                if name == self.path.name and mask & (
                    _IN_CLOSE_WRITE | _IN_MOVED_TO | _IN_CREATE
                ):
                    changed = True
        return changed

    def __enter__(self) -> "AgenticFileChangeWatcher":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class SharedDirectoryEventJournal:
    """Sharded, bounded notifications for atomic JSON manifests on shared FS.

    Eight append streams avoid one global lock across independent snapshots.
    Readers never wait for a writer. Records name manifests, not ownership;
    callers must retain startup/periodic reconciliation for publisher crashes.
    """

    SHARDS = 8
    RECORD_BYTES = 65

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.paths = tuple(self.directory / f".changes-{i}" for i in range(self.SHARDS))
        for path in self.paths:
            with path.open("ab"):
                pass

    def cursors(self) -> tuple[int, ...]:
        return tuple(path.stat().st_size // 65 * 65 for path in self.paths)

    def publish(self, manifest_path: str | Path) -> None:
        name = Path(manifest_path).stem
        if len(name) != 64 or any(c not in "0123456789abcdef" for c in name):
            raise ValueError("invalid Host manifest digest")
        path = self.paths[int(name[:8], 16) % self.SHARDS]
        with path.open("a+b", buffering=0) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                size = os.fstat(stream.fileno()).st_size
                end = size // 65 * 65
                if size != end:
                    os.ftruncate(stream.fileno(), end)
                data = (name + "\n").encode("ascii")
                try:
                    while data:
                        written = os.write(stream.fileno(), data)
                        if written <= 0:
                            raise OSError("short Host notification write")
                        data = data[written:]
                except BaseException:
                    os.ftruncate(stream.fileno(), end)
                    raise
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def read(self, cursors: tuple[int, ...]):
        changed = set()
        next_cursors = list(cursors)
        reset = False
        for i, path in enumerate(self.paths):
            with path.open("rb") as stream:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    cursor = cursors[i]
                    if os.fstat(stream.fileno()).st_size < cursor:
                        cursor = 0
                        reset = True
                    stream.seek(cursor)
                    data = stream.read(65 * 32)
                    data = data[:len(data) // 65 * 65]
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            for offset in range(0, len(data), 65):
                record = data[offset:offset + 65]
                if record[-1:] != b"\n" or any(c not in b"0123456789abcdef" for c in record[:64]):
                    raise ValueError("corrupt Host notification journal")
                changed.add(self.directory / (record[:64].decode("ascii") + ".json"))
            next_cursors[i] = cursor + len(data)
        return tuple(sorted(changed)), tuple(next_cursors), reset


class AgenticDirectoryChangeWatcher:
    """Return the exact JSON paths changed in one node-local directory.

    Unlike :class:`AgenticFileChangeWatcher`, this watcher preserves the file
    name carried by inotify.  Request-generation control planes can therefore
    consume only the changed manifest instead of rescanning a monolithic
    ledger after every edge.  ``overflow`` tells the caller to perform its
    infrequent authoritative resync.
    """

    _shared_poll = None

    def __init__(self, directory: str | Path, *, journal=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._shared_poll = _shared_control_poller()
        if self._shared_poll is not None:
            self._closed = False
            self.healthy = True
            self._journal = journal
            # Captured before the consumer's initial reconciliation scan.
            self._journal_cursors = None if journal is None else journal.cursors()
            return
        self.fd = _inotify_init()
        try:
            self.watch_descriptor = _inotify_add_watch(
                self.fd,
                self.directory,
                _IN_CLOSE_WRITE
                | _IN_MOVED_TO
                | _IN_CREATE
                | _IN_DELETE_SELF
                | _IN_MOVE_SELF,
            )
        except Exception:
            os.close(self.fd)
            raise
        self.poller = select.poll()
        self.poller.register(self.fd, select.POLLIN | select.POLLERR)
        self._closed = False
        self.healthy = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.healthy = False
        if self._shared_poll is not None:
            self._shared_poll.stopped.set()
            return
        try:
            self.poller.unregister(self.fd)
        except (KeyError, OSError):
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass

    def poll(
        self, timeout_seconds: float | None = None
    ) -> tuple[tuple[Path, ...], bool]:
        """Return changed JSON paths and whether a full resync is required."""

        if self._closed:
            return (), False
        if self._shared_poll is not None:
            if not self._shared_poll.due(timeout_seconds):
                return (), False
            if self._journal is None:
                return (), True
            paths, self._journal_cursors, reset = self._journal.read(self._journal_cursors)
            return paths, reset
        timeout_ms = (
            -1
            if timeout_seconds is None
            else max(0, int(float(timeout_seconds) * 1000.0))
        )
        try:
            ready = self.poller.poll(timeout_ms)
        except OSError:
            self.healthy = False
            return (), True
        if not ready:
            return (), False

        paths: set[Path] = set()
        overflow = False
        while True:
            try:
                data = os.read(self.fd, 256 * 1024)
            except BlockingIOError:
                break
            except OSError:
                self.healthy = False
                return (), True
            if not data:
                break
            offset = 0
            while offset + _INOTIFY_EVENT.size <= len(data):
                _, mask, _, name_length = _INOTIFY_EVENT.unpack_from(data, offset)
                offset += _INOTIFY_EVENT.size
                raw_name = data[offset : offset + name_length]
                offset += name_length
                if mask & _IN_Q_OVERFLOW:
                    overflow = True
                    continue
                if mask & (_IN_IGNORED | _IN_DELETE_SELF | _IN_MOVE_SELF):
                    self.healthy = False
                    overflow = True
                    continue
                name = raw_name.split(b"\0", 1)[0].decode(
                    errors="surrogateescape"
                )
                if (
                    name
                    and not name.startswith(".")
                    and name.endswith(".json")
                    and mask & (_IN_CLOSE_WRITE | _IN_MOVED_TO | _IN_CREATE)
                ):
                    paths.add(self.directory / name)
        return tuple(sorted(paths)), overflow

    def __enter__(self) -> "AgenticDirectoryChangeWatcher":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class BrokerPathWatcher:
    """One startup snapshot, then only pushed changes on a private queue."""

    def __init__(self, store, prefix):
        self.store = store
        self.prefix = prefix
        records = store._records
        self.events = records.client.watch(records.namespace)
        self.events.ready.result(timeout=30)  # startup / control worker only
        self._startup = tuple(self.events.initial_snapshot)
        self._closed = False
        self.healthy = True

    def poll(self, timeout_seconds=0.0):
        if self._closed:
            return (), False
        self.events.changed.clear()
        keys, self._startup = list(self._startup), ()
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            keys.append(event["key"])
        self.store._records.client.check_health()
        if not keys and (timeout_seconds is None or timeout_seconds > 0):
            self.events.changed.wait(timeout_seconds)
            return BrokerPathWatcher.poll(self, 0.0)
        return tuple(self.store.directory / key for key in dict.fromkeys(keys)
                     if key.startswith(self.prefix)), False

    def close(self):
        self._closed = True
        self.healthy = False
        self.events.close()
        self.events.changed.set()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BrokerArrivalWatcher(BrokerPathWatcher):
    def __init__(self, store, max_age_seconds):
        self.max_age_seconds = float(max_age_seconds)
        super().__init__(store, "arrivals/")

    def poll(self, timeout_seconds=0.0):
        paths, _ = super().poll(timeout_seconds)
        result = []
        for path in paths:
            item = self.store.read_arrival_path(path, max_age_seconds=self.max_age_seconds)
            if item is not None:
                result.append(item)
        return result


class AgenticEarlyClaimStore:
    def __init__(self, directory: str):
        if not directory:
            raise ValueError("early-claim directory must be non-empty")
        self.directory = Path(directory)
        self.marker_directory = self.directory / "arrivals"
        self.final_directory = self.directory / "finals"
        self.tool_directory = self.directory / "tool-valid"
        self.route_directory = self.directory / "routes"
        # D publishes one exact claim-scoped fence here when the shared
        # Direct setup deadline expires before its sender is submitted.  P
        # must observe this negative-send guarantee before recycling a
        # receiver whose transport still reports WaitingForInput.
        self.direct_abort_directory = self.directory / "direct-aborts"
        from sglang.srt.disaggregation.agentic_control_store import (
            control_enabled, control_kv,
        )
        self._records = (
            control_kv("early-claim", directory) if control_enabled() else None
        )
        if self._records is not None:
            # Paths below are compatibility identity labels, never opened.
            self._journal_enabled = False
            return
        self.marker_directory.mkdir(parents=True, exist_ok=True)
        self.final_directory.mkdir(parents=True, exist_ok=True)
        self.tool_directory.mkdir(parents=True, exist_ok=True)
        self.route_directory.mkdir(parents=True, exist_ok=True)
        self.direct_abort_directory.mkdir(parents=True, exist_ok=True)
        self._journal_enabled = _shared_control_poller() is not None
        self._arrival_events = self.directory / "arrival.events"

    def _arrival_journal_size(self) -> int:
        with self._arrival_events.open("ab") as stream:
            # Incomplete tail can only belong to an interrupted publisher;
            # it is not an event. A later publisher repairs it under EX lock.
            return os.fstat(stream.fileno()).st_size // 65 * 65

    def _read_arrival_events(self, cursor: int):
        """Bounded, nonblocking notification read; never mutates ownership."""
        with self._arrival_events.open("rb") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return (), cursor
            try:
                stream.seek(cursor)
                data = stream.read(65 * 256)
                data = data[:len(data) // 65 * 65]
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        names = []
        for offset in range(0, len(data), 65):
            record = data[offset:offset + 65]
            if record[-1:] != b"\n" or any(c not in b"0123456789abcdef" for c in record[:64]):
                raise RuntimeError("corrupt Direct arrival event journal")
            names.append(self.marker_directory / (record[:64].decode("ascii") + ".json"))
        return tuple(dict.fromkeys(names)), cursor + len(data)

    def _publish_arrival_event(self, request, **kwargs):
        # One short writer transaction across Router processes. Readers never
        # block waiting for it. Retained lifecycle markers remain unchanged.
        with self._arrival_events.open("a+b", buffering=0) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                size = os.fstat(stream.fileno()).st_size
                end = size // 65 * 65
                if end != size:
                    os.ftruncate(stream.fileno(), end)
                payload = self._publish(self.marker_path(request), request, "arrival", **kwargs)
                data = (self._digest(request) + "\n").encode("ascii")
                try:
                    while data:
                        written = os.write(stream.fileno(), data)
                        if written <= 0:
                            raise OSError("short Direct event journal write")
                        data = data[written:]
                except BaseException:
                    os.ftruncate(stream.fileno(), end)
                    raise
                return payload
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _digest(request: RequestGeneration) -> str:
        return hashlib.sha256(request.snapshot_id.encode("utf-8")).hexdigest()

    def marker_path(self, request: RequestGeneration) -> Path:
        return self.marker_directory / f"{self._digest(request)}.json"

    def final_path(self, request: RequestGeneration) -> Path:
        return self.final_directory / f"{self._digest(request)}.json"

    def tool_path(self, request: RequestGeneration) -> Path:
        return self.tool_directory / f"{self._digest(request)}.json"

    def route_path(self, request: RequestGeneration) -> Path:
        return self.route_directory / f"{self._digest(request)}.json"

    def direct_abort_path(self, request: RequestGeneration) -> Path:
        return self.direct_abort_directory / f"{self._digest(request)}.json"

    def producer_path(self, request: RequestGeneration) -> Path:
        # Keep producer tombstones at the top level so the run-script's
        # bounded /dev/shm cleanup removes them without a recursive scan.
        return self.directory / f"producer-{self._digest(request)}"

    def claim_generation_producer(
        self, request: RequestGeneration, producer_id: Optional[str] = None
    ) -> bool:
        """Elect exactly one D producer for a request-generation.

        Long model calls can outlive an HTTP client's retry timeout.  A retry
        may then be routed to a different D and finish concurrently with the
        original.  Retain this tiny O_EXCL tombstone for the run so only the
        first D may publish or mutate the generation's KV lifecycle.
        """

        path = self.producer_path(request)
        owner = str(producer_id or os.getpid())
        if self._records is not None:
            acquired, created = self._records.call("claim", self._record_key(path), owner)
            return bool(acquired and (created or producer_id is not None))
        # Publish a fully-written tombstone atomically.  Creating ``path`` and
        # then writing its owner leaves a short empty-file window in which a
        # sibling TP rank can incorrectly conclude that it belongs to a
        # different producer.  A hard link makes the completed temporary file
        # visible at the final name in one filesystem operation.
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, f"{owner}\n".encode())
            finally:
                os.close(fd)
            os.link(temporary, path)
        except FileExistsError:
            if producer_id is None:
                return False
            try:
                return path.read_text(encoding="utf-8").strip() == owner
            except OSError:
                return False
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return True

    def wait_generation_producer(
        self,
        request: RequestGeneration,
        producer_id: str,
        *,
        timeout_seconds: float = 1.0,
    ) -> bool:
        """Wait for TP rank 0's producer election and mirror its result.

        Only rank 0 is allowed to create the tombstone.  Followers call this
        method after the same generation finishes and therefore normally
        observe the atomically-published owner immediately.
        """

        path = self.producer_path(request)
        owner = str(producer_id)
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        if self._records is not None:
            value = self._records.client.wait_for_record(
                self._records.namespace, self._record_key(path),
                lambda record: record is not None,
                timeout=max(0.0, float(timeout_seconds)),
            )
            return value == owner
        while True:
            try:
                return path.read_text(encoding="utf-8").strip() == owner
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.001)
            except OSError:
                return False

    def _publish(
        self,
        path: Path,
        request: RequestGeneration,
        kind: str,
        *,
        extra: Optional[dict[str, Any]] = None,
        published_at: Optional[float] = None,
    ) -> dict[str, Any]:
        now = time.time() if published_at is None else float(published_at)
        payload = {
            "version": _VERSION,
            "kind": kind,
            "snapshot_id": request.snapshot_id,
            # Keep the structured identity in addition to snapshot_id so the
            # P worker can start a reverse transfer before the tokenized Req
            # reaches its scheduler.  Parsing snapshot_id would be ambiguous
            # when an application request id itself contains a colon.
            "request_id": request.request_id,
            "generation": request.generation,
            "arrived_at": now,
            "publisher_pid": os.getpid(),
        }
        if extra:
            payload.update(extra)
        self._publish_payload(path, payload)
        return payload

    def publish_arrival(
        self,
        request: RequestGeneration,
        *,
        target_prefill_domain: Optional[int] = None,
        prompt_token_count: Optional[int] = None,
        arrived_at: Optional[float] = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if target_prefill_domain is not None:
            extra["target_prefill_domain"] = int(target_prefill_domain)
        if prompt_token_count is not None:
            prompt_token_count = int(prompt_token_count)
            if prompt_token_count <= 0:
                raise ValueError("prompt_token_count must be positive")
            extra["prompt_token_count"] = prompt_token_count
        if self._journal_enabled:
            return self._publish_arrival_event(
                request, extra=extra or None, published_at=arrived_at,
            )
        payload = self._publish(
            self.marker_path(request),
            request,
            "arrival",
            extra=extra or None,
            published_at=arrived_at,
        )
        return payload

    def _record_key(self, path: Path) -> str:
        return str(path.relative_to(self.directory))

    def _read_payload(self, path: Path):
        if self._records is not None:
            value = self._records.get(self._record_key(path))
            if value is None:
                raise FileNotFoundError(str(path))
            return value
        return json.loads(path.read_bytes())

    def _publish_payload(self, path: Path, payload: dict[str, Any]) -> None:
        if self._records is not None:
            # Publication can precede ownership transitions (notably abort
            # fences), so callers must observe the acknowledgement.
            self._records.call("upsert", self._record_key(path), payload)
            return
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as file_obj:
                file_obj.write(data)
                file_obj.flush()
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def publish_route(
        self,
        request: RequestGeneration,
        *,
        route: str,
        prefill_domain: int,
        arena_domain: Optional[int] = None,
        arena_numa_node: Optional[int] = None,
        snapshot_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        if route not in {
            "direct_ready",
            "direct_complete",
            "host_writing",
            "host_ready",
            "recompute",
        }:
            raise ValueError(f"unsupported agentic route {route!r}")
        payload = {
            "version": _VERSION,
            "kind": "route",
            "snapshot_id": request.snapshot_id,
            "request_id": request.request_id,
            "generation": request.generation,
            "route": route,
            "prefill_domain": int(prefill_domain),
            "arena_numa_node": (
                None if arena_numa_node is None else int(arena_numa_node)
            ),
            "snapshot_tokens": (
                None if snapshot_tokens is None else int(snapshot_tokens)
            ),
            "published_at": time.time(),
            "publisher_pid": os.getpid(),
        }
        if arena_domain is not None:
            payload["arena_domain"] = int(arena_domain)
        self._publish_payload(self.route_path(request), payload)
        return payload

    def read_route(
        self,
        request: RequestGeneration,
        *,
        max_age_seconds: float = 3600.0,
    ) -> Optional[dict[str, Any]]:
        try:
            payload = self._read_payload(self.route_path(request))
            published_at = float(payload["published_at"])
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        if (
            payload.get("version") != _VERSION
            or payload.get("kind") != "route"
            or payload.get("snapshot_id") != request.snapshot_id
            or published_at + max_age_seconds < time.time()
        ):
            return None
        return payload

    def publish_final(self, request: RequestGeneration) -> dict[str, Any]:
        """Confirm that the application consumed this output as terminal."""

        return self._publish(self.final_path(request), request, "final")

    def publish_tool(self, request: RequestGeneration) -> dict[str, Any]:
        """Confirm that the application parser accepted a real tool call."""

        return self._publish(self.tool_path(request), request, "tool")

    def publish_direct_abort(
        self,
        request: RequestGeneration,
        *,
        claim_id: str,
        fence_kind: str = "unstarted",
    ) -> dict[str, Any]:
        """Publish D's proof that an exact Direct claim can no longer write.

        ``unstarted`` promises that sender DMA was never submitted and never
        will be. ``terminal`` promises that every submitted local transport
        handle has reached its physical DONE/ERR fence. Neither marker is an
        ownership transition: D remains the authoritative source until P
        returns DIRECT_LOADING to DIRECT_READY and Slow completes normally.
        """

        if not claim_id:
            raise ValueError("direct abort claim_id must be non-empty")
        if fence_kind not in {"unstarted", "terminal"}:
            raise ValueError("direct abort fence_kind must be unstarted or terminal")
        return self._publish(
            self.direct_abort_path(request),
            request,
            "direct-abort",
            extra={
                "claim_id": str(claim_id),
                "fence_kind": fence_kind,
            },
        )

    def _read(
        self,
        path: Path,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        try:
            payload = self._read_payload(path)
            arrived_at = float(payload["arrived_at"])
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        now = time.time()
        if (
            payload.get("version") != _VERSION
            or payload.get("snapshot_id") != request.snapshot_id
            or arrived_at + max_age_seconds < now
            or arrived_at + 0.05 < not_before
        ):
            return None
        return payload

    def read_arrival(
        self,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        return self._read(
            self.marker_path(request),
            request,
            not_before=not_before,
            max_age_seconds=max_age_seconds,
        )

    def iter_arrivals(
        self, *, max_age_seconds: float
    ) -> list[tuple[RequestGeneration, dict[str, Any]]]:
        """Return valid arrival markers without consuming them.

        Decode removes the marker after Direct completion or slow fallback.
        P therefore only observes markers here; consuming one in P could race
        with Decode's fast-tool-window check.
        """

        arrivals: list[tuple[RequestGeneration, dict[str, Any]]] = []
        try:
            paths = (
                tuple(self.directory / key for key in self._records.snapshot()
                      if key.startswith("arrivals/"))
                if self._records is not None
                else tuple(self.marker_directory.glob("*.json"))
            )
        except OSError:
            return arrivals
        for path in paths:
            item = self.read_arrival_path(path, max_age_seconds=max_age_seconds)
            if item is not None:
                arrivals.append(item)
        arrivals.sort(key=lambda item: float(item[1]["arrived_at"]))
        return arrivals

    def read_arrival_path(
        self, path: Path, *, max_age_seconds: float
    ) -> Optional[tuple[RequestGeneration, dict[str, Any]]]:
        """Validate one path delivered by :class:`AgenticArrivalWatcher`."""

        try:
            payload = self._read_payload(path)
            request = RequestGeneration(
                str(payload["request_id"]), int(payload["generation"])
            )
            arrived_at = float(payload["arrived_at"])
        except (
            FileNotFoundError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ):
            return None
        if (
            payload.get("version") != _VERSION
            or payload.get("kind") != "arrival"
            or payload.get("snapshot_id") != request.snapshot_id
            or arrived_at + max_age_seconds < time.time()
        ):
            return None
        return request, payload

    def watch_arrivals(self, *, max_age_seconds: float) -> AgenticArrivalWatcher:
        if self._records is not None:
            return BrokerArrivalWatcher(self, max_age_seconds)
        return AgenticArrivalWatcher(self, max_age_seconds)

    def watch_direct_aborts(self):
        if self._records is not None:
            return BrokerPathWatcher(self, "direct-aborts/")
        return AgenticDirectoryChangeWatcher(self.direct_abort_directory)

    def read_final(
        self,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        return self._read(
            self.final_path(request),
            request,
            not_before=not_before,
            max_age_seconds=max_age_seconds,
        )

    def read_tool(
        self,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        return self._read(
            self.tool_path(request),
            request,
            not_before=not_before,
            max_age_seconds=max_age_seconds,
        )

    def read_direct_abort(
        self,
        request: RequestGeneration,
        *,
        claim_id: str,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        payload = self._read(
            self.direct_abort_path(request),
            request,
            not_before=not_before,
            max_age_seconds=max_age_seconds,
        )
        if (
            payload is None
            or payload.get("kind") != "direct-abort"
            or payload.get("claim_id") != claim_id
            or payload.get("fence_kind", "unstarted")
            not in {"unstarted", "terminal"}
        ):
            return None
        return payload

    def remove_arrival(self, request: RequestGeneration) -> None:
        """Remove only the ingress marker; no capacity ledger is involved."""

        self._remove_payload(self.marker_path(request))

    def remove_final(self, request: RequestGeneration) -> None:
        self._remove_payload(self.final_path(request))

    def remove_tool(self, request: RequestGeneration) -> None:
        self._remove_payload(self.tool_path(request))

    def remove_direct_abort(self, request: RequestGeneration) -> None:
        self._remove_payload(self.direct_abort_path(request))

    def _remove_payload(self, path: Path) -> None:
        if self._records is not None:
            self._records.call("remove", self._record_key(path))
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
