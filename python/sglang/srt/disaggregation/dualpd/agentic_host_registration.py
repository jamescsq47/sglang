from __future__ import annotations

"""Host mapping authentication, registered-window cache and startup prewarm.

No snapshot ownership or transfer decisions live here. Callers retain mapping
and window references until their existing physical DMA fences complete.
"""
import json
import logging
import mmap
import os
import threading
import time
from typing import Any, Optional

import torch

# Keep the historical logger identity as well as the caller import aliases.
logger = logging.getLogger("sglang.srt.disaggregation.agentic_host_staging")


def _is_memfd_backing_path(path: str) -> bool:
    """Return whether ``path`` is the narrow cross-process memfd form."""

    parts = str(path).split("/")
    return (
        len(parts) == 5
        and parts[0] == ""
        and parts[1] == "proc"
        and parts[2].isdigit()
        and parts[3] == "fd"
        and parts[4].isdigit()
    )

def _validate_shared_host_backing_path(path: str, *, create: bool = False) -> None:
    """Limit data-plane mappings to owned tmpfs files or anonymous memfds."""

    if str(path).startswith("/dev/shm/"):
        return
    if not create and _is_memfd_backing_path(path):
        return
    raise ValueError(
        "shared Host snapshot must use /dev/shm or a published memfd descriptor"
    )

def _open_shared_host_backing(path: str, flags: int, mode: int = 0o600) -> int:
    """Open and authenticate a shared Host data-plane backing object."""

    create = bool(flags & os.O_CREAT)
    _validate_shared_host_backing_path(path, create=create)
    fd = os.open(path, flags, mode)
    if _is_memfd_backing_path(path):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
            if not target.startswith("/memfd:sglang-agentic-host-arena-"):
                raise ValueError("published Host descriptor is not an agentic memfd")
        except BaseException:
            os.close(fd)
            raise
    return fd

class _RegisteredHostArenaMapping:
    """Process-local persistent mmap with a bounded registered-window cache."""

    def __init__(self, path: str):
        self.path = str(path)
        fd = _open_shared_host_backing(path, os.O_RDWR)
        try:
            self.byte_size = int(os.fstat(fd).st_size)
            self.mapping = mmap.mmap(fd, self.byte_size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        self.raw = torch.frombuffer(
            self.mapping, dtype=torch.uint8, count=self.byte_size
        )
        self.window_bytes = max(
            mmap.ALLOCATIONGRANULARITY,
            int(float(os.getenv("SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB", "8")) * (1024**3)),
        )
        self.window_bytes = (
            self.window_bytes // mmap.ALLOCATIONGRANULARITY
            * mmap.ALLOCATIONGRANULARITY
        )
        self._windows: dict[int, dict[str, Any]] = {}
        self._users = 0
        self._prewarm_started = False
        self._prewarm_thread: Optional[threading.Thread] = None
        self._prewarm_done = threading.Event()
        self._prewarm_error: Optional[BaseException] = None

    def prewarm(self, device) -> None:
        """Start one background pass that registers every backing window.

        Sparse D->P recovery may not revisit a 256-GiB arena window during a
        finite workload warmup.  Leaving registration lazy then charges a
        one-time page-pin cost to an otherwise fast measurement-period H2D.
        The first transport user therefore starts an eager pass, but does not
        run that pass inline: registering a 256-GiB arena can take tens of
        seconds and must not stall D2H/H2D progress.  The worker registers and
        releases one cache window at a time so request-level DMA can interleave
        at every window boundary.  Later request extents still use the normal
        acquire/release ownership and eviction rules.
        """

        enabled = os.getenv(
            "SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA", "0"
        ).strip().lower() not in {"0", "false", "no", "off"}
        if not enabled:
            return
        with _REGISTERED_HOST_ARENAS_LOCK:
            # Lightweight lifecycle tests and a few compatibility callers
            # construct this object without invoking __init__.
            if not hasattr(self, "_prewarm_done"):
                self._prewarm_done = threading.Event()
            if not hasattr(self, "_prewarm_error"):
                self._prewarm_error = None
            if self._prewarm_started:
                return
            self._prewarm_started = True
            worker = threading.Thread(
                target=self._prewarm_windows,
                args=(device,),
                name="agentic-host-register-prewarm",
                daemon=True,
            )
            self._prewarm_thread = worker
        try:
            worker.start()
        except BaseException as exc:
            # The normal request-level acquire remains fully functional even
            # when a background thread cannot be created.
            self._prewarm_error = exc
            self._prewarm_done.set()
            logger.exception(
                "AgenticKV registered_arena_prewarm_start_failed "
                "path=%s bytes=%d; continuing with request-level registration",
                self.path,
                self.byte_size,
            )

    def wait_prewarm(self, timeout: Optional[float] = None) -> None:
        """Wait until eager registration is complete and surface any failure."""

        if not self._prewarm_started:
            raise RuntimeError(
                f"registered Host prewarm was not started: {self.path}"
            )
        if not self._prewarm_done.wait(timeout):
            raise TimeoutError(f"registered Host prewarm timed out: {self.path}")
        if self._prewarm_error is not None:
            raise RuntimeError(
                f"registered Host prewarm failed: {self.path}"
            ) from self._prewarm_error

    def _prewarm_windows(self, device) -> None:
        started_at = time.perf_counter()
        completed_windows = 0
        try:
            for offset in range(0, self.byte_size, self.window_bytes):
                windows = self.acquire(
                    offset,
                    min(self.window_bytes, self.byte_size - offset),
                    device,
                )
                try:
                    completed_windows += len(windows)
                finally:
                    self.release(windows)
                # Give request-level transport workers waiting for the same
                # registry mutex an explicit scheduling point.
                time.sleep(0)
        except BaseException as exc:
            # Prewarming is only a latency optimization.  Partially registered
            # windows remain valid cache entries and the request-level acquire
            # can still use them or fall back before any DMA submission.
            self._prewarm_error = exc
            logger.exception(
                "AgenticKV registered_arena_prewarm_failed path=%s bytes=%d; "
                "continuing with request-level registration",
                self.path,
                self.byte_size,
            )
        else:
            logger.info(
                "AgenticKV registered_arena_prewarm_complete path=%s bytes=%d "
                "windows=%d elapsed_s=%.3f",
                self.path,
                self.byte_size,
                completed_windows,
                time.perf_counter() - started_at,
            )
        finally:
            self._prewarm_done.set()

    def acquire(self, offset: int, byte_size: int, device) -> tuple[int, ...]:
        first = int(offset) // self.window_bytes * self.window_bytes
        last = int(offset) + int(byte_size)
        offsets = tuple(range(first, last, self.window_bytes))
        acquired: list[int] = []
        with _REGISTERED_HOST_ARENAS_LOCK, torch.cuda.device(device):
            try:
                for window_offset in offsets:
                    entry = self._windows.get(window_offset)
                    if entry is None:
                        window_size = min(
                            self.window_bytes, self.byte_size - window_offset
                        )
                        global_limit = max(
                            self.window_bytes,
                            int(
                                float(
                                    os.getenv(
                                        "SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB", "64"
                                    )
                                )
                                * (1024**3)
                            ),
                        )
                        while (
                            sum(
                                int(item["bytes"])
                                for arena in _REGISTERED_HOST_ARENAS.values()
                                for item in arena._windows.values()
                            )
                            + window_size
                            > global_limit
                        ):
                            victims = [
                                (item["used_at"], arena, current_offset, item)
                                for arena in _REGISTERED_HOST_ARENAS.values()
                                for current_offset, item in arena._windows.items()
                                if int(item["refs"]) == 0
                            ]
                            if not victims:
                                raise MemoryError(
                                    "registered Host window cache is fully referenced"
                                )
                            _, victim_arena, victim_offset, victim = min(
                                victims, key=lambda value: value[0]
                            )
                            cudart = torch.cuda.cudart()
                            result = cudart.cudaHostUnregister(
                                victim_arena.raw.data_ptr() + victim_offset
                            )
                            if result != cudart.cudaError.success:
                                raise RuntimeError(
                                    f"cached cudaHostUnregister returned {result}"
                                )
                            victim_arena._windows.pop(victim_offset)
                        cudart = torch.cuda.cudart()
                        result = cudart.cudaHostRegister(
                            self.raw.data_ptr() + window_offset, window_size, 0
                        )
                        if result != cudart.cudaError.success:
                            raise RuntimeError(
                                f"cached cudaHostRegister returned {result}"
                            )
                        entry = self._windows[window_offset] = {
                            "bytes": window_size,
                            "refs": 0,
                            "used_at": time.monotonic(),
                        }
                    entry["refs"] = int(entry["refs"]) + 1
                    entry["used_at"] = time.monotonic()
                    acquired.append(window_offset)
            except BaseException:
                for window_offset in acquired:
                    self._windows[window_offset]["refs"] -= 1
                raise
        return tuple(acquired)

    def release(self, windows: tuple[int, ...]) -> None:
        with _REGISTERED_HOST_ARENAS_LOCK:
            for window_offset in windows:
                entry = self._windows.get(int(window_offset))
                if entry is None or int(entry["refs"]) <= 0:
                    raise RuntimeError("registered Host window reference underflow")
                entry["refs"] -= 1
                entry["used_at"] = time.monotonic()

_REGISTERED_HOST_ARENAS: dict[tuple[int, int, str], _RegisteredHostArenaMapping] = {}

_REGISTERED_HOST_ARENAS_LOCK = threading.Lock()

_REGISTERED_HOST_PREWARM_PINS: list[_RegisteredHostArenaMapping] = []

def _registered_host_arena(path: str, device) -> _RegisteredHostArenaMapping:
    fd = _open_shared_host_backing(path, os.O_RDWR)
    try:
        stat = os.fstat(fd)
        key = (int(stat.st_dev), int(stat.st_ino), str(path))
    finally:
        os.close(fd)
    with _REGISTERED_HOST_ARENAS_LOCK:
        stale = []
        for current_key, current in _REGISTERED_HOST_ARENAS.items():
            if current_key == key or current._users or any(
                int(item["refs"]) for item in current._windows.values()
            ):
                continue
            try:
                current_stat = os.stat(current_key[2])
                still_same = (
                    int(current_stat.st_dev), int(current_stat.st_ino)
                ) == current_key[:2]
            except OSError:
                still_same = False
            if not still_same:
                stale.append((current_key, current))
        with torch.cuda.device(device):
            for current_key, current in stale:
                cudart = torch.cuda.cudart()
                for window_offset in tuple(current._windows):
                    result = cudart.cudaHostUnregister(
                        current.raw.data_ptr() + window_offset
                    )
                    if result != cudart.cudaError.success:
                        raise RuntimeError(
                            f"stale cached cudaHostUnregister returned {result}"
                        )
                    current._windows.pop(window_offset)
                current.raw = None
                current.mapping.close()
                _REGISTERED_HOST_ARENAS.pop(current_key)
        arena = _REGISTERED_HOST_ARENAS.get(key)
        if arena is None:
            arena = _REGISTERED_HOST_ARENAS[key] = _RegisteredHostArenaMapping(path)
        arena._users += 1
    # This function is reached from D2H/H2D transport workers, never from the
    # model forward stream.  prewarm() only starts a one-shot background pass,
    # so discovering a new arena never synchronously stalls transport progress.
    arena.prewarm(device)
    return arena

def _write_startup_prewarm_record(path: str, payload: dict[str, Any]) -> None:
    """Atomically publish one run-scoped startup-prewarm record."""

    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)

def start_registered_host_arena_startup_prewarm(
    *,
    role: str,
    engine_id: str,
    tp_rank: int,
    device,
    d2p_arena_path: Optional[str] = None,
    p2d_arena_path: Optional[str] = None,
) -> Optional[threading.Thread]:
    """Register every arena reachable by this CUDA context before traffic.

    P owns and publishes the physical memfd paths.  A run-scoped start marker
    lets the launcher wait until all P manifests exist before every P/D rank
    maps and registers its required paths.  The worker is deliberately
    separate from request and Forward progress; the launcher does not send
    traffic until every participant publishes a completion record.
    """

    enabled = os.getenv(
        "SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER", "0"
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None
    if role not in {"prefill", "decode"}:
        raise ValueError(f"invalid registered Host prewarm role: {role}")
    if os.getenv("SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA", "0").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        raise ValueError("startup Host prewarm requires eager arena registration")

    root = os.environ.get(
        "SGLANG_AGENTIC_KV_REGISTER_PREWARM_DIR", ""
    ).rstrip("/")
    if not root:
        raise ValueError("startup Host prewarm requires a run-scoped prewarm directory")
    domain_count = int(
        os.environ.get("SGLANG_AGENTIC_KV_PREFILL_DOMAIN_COUNT", "0")
    )
    tp_size = int(os.environ.get("SGLANG_AGENTIC_KV_TP_SIZE", "1"))
    domain = int(os.environ.get("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "-1"))
    if domain_count <= 0 or not 0 <= int(tp_rank) < tp_size:
        raise ValueError("invalid startup Host prewarm topology")
    if role == "prefill" and (
        domain < 0 or (d2p_arena_path is None and p2d_arena_path is None)
    ):
        raise ValueError("P startup Host prewarm requires an enabled Host arena")

    arenas_dir = os.path.join(root, "arenas")
    complete_dir = os.path.join(root, "complete")
    failed_dir = os.path.join(root, "failed")
    start_path = os.path.join(root, "start")
    os.makedirs(arenas_dir, mode=0o700, exist_ok=True)
    os.makedirs(complete_dir, mode=0o700, exist_ok=True)
    os.makedirs(failed_dir, mode=0o700, exist_ok=True)
    participant = f"{role}-{engine_id}-rank-{int(tp_rank)}"

    if role == "prefill":
        _write_startup_prewarm_record(
            os.path.join(arenas_dir, f"domain-{domain}-rank-{int(tp_rank)}.json"),
            {
                "domain": domain,
                "tp_rank": int(tp_rank),
                "d2p_path": (
                    None if d2p_arena_path is None else str(d2p_arena_path)
                ),
                "p2d_path": (
                    None if p2d_arena_path is None else str(p2d_arena_path)
                ),
                "owner_pid": os.getpid(),
            },
        )

    def worker() -> None:
        worker_started_at = time.monotonic()
        timeout = float(
            os.getenv("SGLANG_AGENTIC_KV_REGISTER_PREWARM_TIMEOUT_SECONDS", "1800")
        )
        start_deadline = worker_started_at + max(1.0, timeout)
        try:
            while not os.path.exists(start_path):
                if time.monotonic() >= start_deadline:
                    raise TimeoutError("startup Host prewarm start marker timed out")
                time.sleep(0.1)

            started_at = time.monotonic()
            deadline = started_at + max(1.0, timeout)

            expected_manifests = domain_count * tp_size
            manifests: list[dict[str, Any]] = []
            while True:
                manifest_paths = sorted(
                    name
                    for name in (
                        os.path.join(arenas_dir, item)
                        for item in os.listdir(arenas_dir)
                    )
                    if name.endswith(".json")
                )
                if len(manifest_paths) == expected_manifests:
                    manifests = []
                    for manifest_path in manifest_paths:
                        with open(manifest_path, encoding="utf-8") as source:
                            manifests.append(json.load(source))
                    break
                if len(manifest_paths) > expected_manifests:
                    raise RuntimeError("too many startup Host arena manifests")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "startup Host arena manifests timed out: "
                        f"found={len(manifest_paths)} expected={expected_manifests}"
                    )
                time.sleep(0.1)

            rank_manifests = [
                item for item in manifests if int(item["tp_rank"]) == int(tp_rank)
            ]
            if len(rank_manifests) != domain_count:
                raise RuntimeError("startup Host arena manifest rank coverage mismatch")
            paths = [
                str(item["d2p_path"])
                for item in rank_manifests
                if item.get("d2p_path")
            ]
            if role == "decode":
                paths.extend(
                    str(item["p2d_path"])
                    for item in rank_manifests
                    if item.get("p2d_path")
                )
            else:
                own = [item for item in rank_manifests if int(item["domain"]) == domain]
                if len(own) != 1:
                    raise RuntimeError("P startup Host arena domain is not unique")
                if own[0].get("p2d_path"):
                    paths.append(str(own[0]["p2d_path"]))
            paths = list(dict.fromkeys(paths))

            mappings = [_registered_host_arena(path, device) for path in paths]
            for mapping in mappings:
                remaining = max(0.0, deadline - time.monotonic())
                mapping.wait_prewarm(remaining)
            # Keep one process-lifetime mapping reference so a later stale-path
            # sweep cannot unregister startup-warmed windows between requests.
            _REGISTERED_HOST_PREWARM_PINS.extend(mappings)
            elapsed = time.monotonic() - started_at
            record = {
                "role": role,
                "engine_id": str(engine_id),
                "tp_rank": int(tp_rank),
                "pid": os.getpid(),
                "arena_count": len(mappings),
                "registered_bytes": sum(item.byte_size for item in mappings),
                "elapsed_seconds": elapsed,
            }
            _write_startup_prewarm_record(
                os.path.join(complete_dir, f"{participant}.json"), record
            )
            logger.info(
                "AgenticKV startup_prewarm_complete role=%s engine=%s rank=%d "
                "arenas=%d bytes=%d elapsed_s=%.3f",
                role,
                engine_id,
                int(tp_rank),
                len(mappings),
                record["registered_bytes"],
                elapsed,
            )
        except BaseException as exc:
            logger.exception(
                "AgenticKV startup_prewarm_failed role=%s engine=%s rank=%d",
                role,
                engine_id,
                int(tp_rank),
            )
            _write_startup_prewarm_record(
                os.path.join(failed_dir, f"{participant}.json"),
                {
                    "role": role,
                    "engine_id": str(engine_id),
                    "tp_rank": int(tp_rank),
                    "pid": os.getpid(),
                    "error": repr(exc),
                    "elapsed_seconds": time.monotonic() - worker_started_at,
                },
            )

    thread = threading.Thread(
        target=worker,
        name=f"agentic-host-startup-prewarm-{participant}",
        daemon=True,
    )
    thread.start()
    return thread
