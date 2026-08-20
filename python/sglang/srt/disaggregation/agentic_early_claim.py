"""Node-local early-arrival markers for the agentic D-to-P direct path.

The HTTP router publishes a tiny marker as soon as a later agent turn arrives.
Decode workers use the marker only to distinguish a fast tool return from a
slow one.  It allocates no P HBM and carries no global capacity/credit policy.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import select
import struct
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


class AgenticArrivalWatcher:
    """Event-driven reader for Router arrival markers.

    The watch is installed before the one-time startup scan, so a marker
    created concurrently with startup is either found by that scan or remains
    queued in the inotify fd.  Normal operation reads only paths named by
    inotify; a full scan is used again solely after kernel queue overflow.
    """

    def __init__(self, store: "AgenticEarlyClaimStore", max_age_seconds: float):
        self.store = store
        self.max_age_seconds = float(max_age_seconds)
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


class AgenticEarlyClaimStore:
    def __init__(self, directory: str):
        if not directory:
            raise ValueError("early-claim directory must be non-empty")
        self.directory = Path(directory)
        self.marker_directory = self.directory / "arrivals"
        self.final_directory = self.directory / "finals"
        self.tool_directory = self.directory / "tool-valid"
        self.route_directory = self.directory / "routes"
        self.marker_directory.mkdir(parents=True, exist_ok=True)
        self.final_directory.mkdir(parents=True, exist_ok=True)
        self.tool_directory.mkdir(parents=True, exist_ok=True)
        self.route_directory.mkdir(parents=True, exist_ok=True)

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
        while True:
            try:
                return path.read_text(encoding="utf-8").strip() == owner
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.001)
            except OSError:
                return False

    @staticmethod
    def _publish(
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
        return payload

    def publish_arrival(
        self,
        request: RequestGeneration,
        *,
        target_prefill_domain: Optional[int] = None,
        arrived_at: Optional[float] = None,
    ) -> dict[str, Any]:
        payload = self._publish(
            self.marker_path(request),
            request,
            "arrival",
            extra=(
                None
                if target_prefill_domain is None
                else {"target_prefill_domain": int(target_prefill_domain)}
            ),
            published_at=arrived_at,
        )
        return payload

    @staticmethod
    def _publish_payload(path: Path, payload: dict[str, Any]) -> None:
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
        self._publish_payload(self.route_path(request), payload)
        return payload

    def read_route(
        self,
        request: RequestGeneration,
        *,
        max_age_seconds: float = 3600.0,
    ) -> Optional[dict[str, Any]]:
        try:
            payload = json.loads(self.route_path(request).read_bytes())
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

    @staticmethod
    def _read(
        path: Path,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        try:
            payload = json.loads(path.read_bytes())
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
            paths = tuple(self.marker_directory.glob("*.json"))
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
            payload = json.loads(path.read_bytes())
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
        return AgenticArrivalWatcher(self, max_age_seconds)

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

    def remove_arrival(self, request: RequestGeneration) -> None:
        """Remove only the ingress marker; no capacity ledger is involved."""

        try:
            self.marker_path(request).unlink(missing_ok=True)
        except OSError:
            pass

    def remove_final(self, request: RequestGeneration) -> None:
        try:
            self.final_path(request).unlink(missing_ok=True)
        except OSError:
            pass

    def remove_tool(self, request: RequestGeneration) -> None:
        try:
            self.tool_path(request).unlink(missing_ok=True)
        except OSError:
            pass
