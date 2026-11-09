from __future__ import annotations

"""Request-generation D-HBM -> shared P-Host arena for agentic PD serving.

The control plane is deliberately tiny and node-local.  KV bytes never pass
through it: D publishes an offer in /dev/shm, P atomically reserves one complete
snapshot extent in a shared pinned Host arena, and D writes that extent with its
own CUDA D2H engine.  P later restores it with its own H2D engine.  D may release
its source snapshot only after its CUDA event completed and HOST_READY was
committed.  The custom path keeps KV bytes exclusively in GPU or Shared Arena;
its node-local metadata store contains manifests and claims only.
"""

import fcntl
import hashlib
import json
import logging
import math
import os
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Optional

import torch

from sglang.srt.disaggregation.agentic_early_claim import (
    AgenticDirectoryChangeWatcher,
    AgenticEarlyClaimStore,
)
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    MooncakeSnapshotStore,
    SharedSnapshotEvictionController,
    SnapshotManifest,
    SnapshotState,
    RequestGeneration,
    page_namespace,
)
from sglang.srt.disaggregation.utils import kv_to_page_indices
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo
from sglang.srt.disaggregation.agentic_host_ledger import (
    HostStageState,
    PermanentHostStageError,
    SharedHostStagingLedger,
    P2D_RELEASE_NATIVE_WON,
    P2D_RELEASE_HOST_OWNED,
    P2D_RELEASE_HOST_TERMINAL,
    _ALLOWED_STAGE_TRANSITIONS,
    _TERMINAL_STATES,
    _LEDGER_OWNERSHIP_CHANGED,
)

logger = logging.getLogger(__name__)
# D role owns its sentinel and classes once; preserve historical import identities.
from sglang.srt.disaggregation.agentic_host_writer import (
    _LEDGER_ENTRY_UNSET, _cleanup_nixl_room,
    AgenticDRelayWorker, AgenticDHostStagingClient,
)
# Physical buffers/views have one canonical implementation. Keep historical
# class/helper imports identical for native, hybrid and compatibility callers.
from sglang.srt.disaggregation.agentic_host_buffers import (
    _HOST_LIBC, _HOST_MEMMOVE, _HOST_MADVISE, _HOST_MEMSET, _HOST_MEMFD_CREATE,
    _MADV_POPULATE_WRITE, _MFD_CLOEXEC,
    _CUDA_MEMCPY_HOST_TO_DEVICE, _CUDA_MEMCPY_DEVICE_TO_HOST,
    _registered_indexed_batch_copy, _device_indices_to_host,
    _create_agentic_host_memfd, _copy_layer_first_host_range,
    HostCopyWorkerPool, H2DLaunchFence, _concrete_kv_pool_device,
    LayerFirstD2HStaging, SharedMHAHostSnapshot, PinnedMHAHostBounce,
    LazySharedMHAHostSnapshot, SharedHostSnapshotArena,
)


# Keep historical helper imports available to transport callers. The ABI and
# mutable driver/runtime handles have one canonical owner in agentic_host_dma.
from sglang.srt.disaggregation.agentic_host_dma import (
    _CUDAMemLocation,
    _CUDAMemcpyAttributes,
    _cuda_runtime,
    _cuda_memcpy_async,
    _cuda_driver_batch_memcpy,
    _cuda_batch_memcpy_async,
)


# Canonical mapping/cache ownership lives in this leaf module. Historical
# imports remain aliases; private-hook patches target the defining module.
from sglang.srt.disaggregation.agentic_host_registration import (
    _is_memfd_backing_path,
    _validate_shared_host_backing_path,
    _open_shared_host_backing,
    _RegisteredHostArenaMapping,
    _REGISTERED_HOST_ARENAS,
    _REGISTERED_HOST_ARENAS_LOCK,
    _REGISTERED_HOST_PREWARM_PINS,
    _registered_host_arena,
    _write_startup_prewarm_record,
    start_registered_host_arena_startup_prewarm,
)


def supports_agentic_kv_spill(storage_backend) -> bool:
    """Return the authoritative Shared-Arena spill capability.

    Existing SGLang storage backends predate this opt-out attribute and are
    spill-capable.  The request-level node-local backend explicitly sets it to
    ``False``.  D metadata, P admission and scheduler construction must all
    use this same compatibility rule.
    """

    return bool(getattr(storage_backend, "supports_kv_spill", True))


class AgenticNodeLocalRawStore:
    """Tiny cross-process metadata store backed by tmpfs files.

    Shared Arena owns the KV bytes.  The request-generation state machine only
    needs create-if-absent, replace, read, and remove for small manifests and
    claim records; using this store keeps the custom path independent of
    native HiCache/Mooncake without putting KV data on the filesystem.
    """

    def __init__(self, directory: str):
        if not directory:
            raise ValueError("agentic metadata directory must be non-empty")
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.directory, digest)

    @contextmanager
    def _locked(self, key: str):
        lock_path = f"{self._path(key)}.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield self._path(key)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _replace(path: str, value: bytes) -> None:
        temp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
        try:
            with open(temp, "wb") as handle:
                handle.write(value)
                handle.flush()
            os.replace(temp, path)
        finally:
            try:
                os.unlink(temp)
            except FileNotFoundError:
                pass

    def put(self, key: str, value: bytes) -> int:
        with self._locked(key) as path:
            if os.path.exists(path):
                return -1
            self._replace(path, bytes(value))
        return 0

    def upsert(self, key: str, value: bytes) -> int:
        with self._locked(key) as path:
            self._replace(path, bytes(value))
        return 0

    def get(self, key: str) -> bytes:
        with self._locked(key) as path:
            try:
                with open(path, "rb") as handle:
                    return handle.read()
            except FileNotFoundError:
                return b""

    def is_exist(self, key: str) -> int:
        with self._locked(key) as path:
            return int(os.path.exists(path))

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [self.is_exist(key) for key in keys]

    def remove(self, key: str, force: bool = False) -> int:
        del force
        with self._locked(key) as path:
            try:
                os.unlink(path)
            except FileNotFoundError:
                return -1
        return 0

    def batch_remove(self, keys: list[str], force: bool = False) -> list[int]:
        return [self.remove(key, force=force) for key in keys]


class AgenticNodeLocalMetadataBackend:
    """Manifest backend for Shared-Arena-only deployments."""

    supports_kv_spill = False

    def __init__(self, directory: str):
        self._raw_store = AgenticNodeLocalRawStore(directory)

    def agentic_snapshot_store(self):
        return MooncakeSnapshotStore(self._raw_store)

    def close(self) -> None:
        pass


def create_agentic_node_local_metadata_backend():
    ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
    directory = os.environ.get("SGLANG_AGENTIC_KV_METADATA_DIR", "")
    if not directory:
        if not ready_dir:
            raise ValueError(
                "Shared-Arena metadata requires SGLANG_PD_P_READY_DIR or "
                "SGLANG_AGENTIC_KV_METADATA_DIR"
            )
        directory = os.path.join(ready_dir, "snapshot-metadata")
    return AgenticNodeLocalMetadataBackend(directory)


class AgenticStorageController:
    """Storage-only resources for request-generation slow-path spill.

    This deliberately is not HiCache: it owns no Radix policy, automatic
    backup/prefetch queue, or device-to-host cache.  The Host pool is only a
    temporary page-addressable staging allocation for explicit Mooncake
    snapshot puts/gets initiated by ``AgenticPHostStagingManager``.
    """

    def __init__(self, mem_pool_host, storage_backend, storage_batch_size=128):
        self.mem_pool_host = mem_pool_host
        self.storage_backend = storage_backend
        self.storage_batch_size = int(storage_batch_size)

    def close(self) -> None:
        close = getattr(self.storage_backend, "close", None)
        if close is not None:
            close()


def _agentic_storage_extra_config(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    if not value.startswith("@"):
        config = json.loads(value)
    else:
        path = value[1:]
        extension = os.path.splitext(path)[1].lower()
        with open(path, "rb" if extension == ".toml" else "r") as handle:
            if extension == ".json":
                config = json.load(handle)
            elif extension == ".toml":
                import tomllib

                config = tomllib.load(handle)
            elif extension in {".yaml", ".yml"}:
                import yaml

                config = yaml.safe_load(handle)
            else:
                raise ValueError(f"unsupported storage config format: {extension}")
    if not isinstance(config, dict):
        raise ValueError("agentic storage backend config must be an object")
    # These are generic HiCache scheduling knobs, not backend constructor
    # options.  The custom request-generation state machine does not use them.
    for name in (
        "prefetch_threshold",
        "prefetch_timeout_base",
        "prefetch_timeout_per_ki_token",
        "hicache_storage_pass_prefix_keys",
    ):
        config.pop(name, None)
    return config


def create_agentic_storage_controller(
    *,
    token_allocator,
    server_args,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    model_name: Optional[str],
) -> AgenticStorageController:
    """Create the custom slow-path storage data plane without HiCache."""

    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
    from sglang.srt.mem_cache.memory_pool import (
        MHATokenToKVPool,
        MLATokenToKVPool,
        NSATokenToKVPool,
    )
    from sglang.srt.mem_cache.memory_pool_host import (
        MHATokenToKVPoolHost,
        MLATokenToKVPoolHost,
        NSATokenToKVPoolHost,
    )
    from sglang.srt.mem_cache.storage import StorageBackendFactory

    backend_name = server_args.hicache_storage_backend
    if backend_name is None:
        backend = create_agentic_node_local_metadata_backend()
        logger.info(
            "Agentic Shared-Arena metadata enabled directory=%s "
            "native_hicache=false mooncake=false",
            backend._raw_store.directory,
        )
        return AgenticStorageController(None, backend)
    device_pool = token_allocator.get_kvcache()
    common = (
        device_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        server_args.page_size,
        server_args.hicache_mem_layout,
    )
    if isinstance(device_pool, MHATokenToKVPool):
        host_pool = MHATokenToKVPoolHost(
            *common, allocator_type=backend_name
        )
    elif isinstance(device_pool, NSATokenToKVPool):
        host_pool = NSATokenToKVPoolHost(
            *common, allocator_type=backend_name
        )
    elif isinstance(device_pool, MLATokenToKVPool):
        host_pool = MLATokenToKVPoolHost(
            *common, allocator_type=backend_name
        )
    else:
        raise ValueError(
            f"unsupported agentic storage KV pool: {type(device_pool).__name__}"
        )

    storage_config = HiCacheStorageConfig(
        tp_rank=int(tp_rank),
        tp_size=int(tp_size),
        pp_rank=int(pp_rank),
        pp_size=int(pp_size),
        is_mla_model=isinstance(device_pool, MLATokenToKVPool),
        enable_storage_metrics=False,
        is_page_first_layout=host_pool.layout == "page_first",
        model_name=model_name,
        extra_config=_agentic_storage_extra_config(
            server_args.hicache_storage_backend_extra_config
        ),
    )
    backend = StorageBackendFactory.create_backend(
        backend_name, storage_config, host_pool
    )
    backend.register_mem_pool_host(host_pool)
    logger.info(
        "Agentic custom storage controller enabled backend=%s host_gib=%.2f "
        "native_hicache=false",
        backend_name,
        host_pool.get_total_size() / (1024**3)
        if hasattr(host_pool, "get_total_size")
        else float(server_args.hicache_size),
    )
    return AgenticStorageController(host_pool, backend)


class AgenticPHostStagingManager:
    """P-side Shared Arena and demand-restore manager."""

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
        workset_broker,
        cache_controller,
        tree_cache,
        page_size: int,
        arena_directory: str,
        arena_capacity_bytes: int,
        arena_backend: str = "tmpfs",
        high_watermark: float = 0.90,
        low_watermark: float = 0.75,
        hard_watermark: float = 0.95,
        arena_numa_node: int = -1,
        arena_domain: int = -1,
        tp_rank: int = 0,
        tp_size: int = 1,
        expected_tool_seconds: Optional[dict[str, float]] = None,
        eviction_controller: Optional[SharedSnapshotEvictionController] = None,
    ):
        if not (0 < low_watermark <= high_watermark < hard_watermark <= 1):
            raise ValueError("host watermarks must satisfy 0 < low <= high < hard <= 1")
        self.ledger = ledger
        self.runtime = runtime
        self.token_allocator = token_allocator
        self.workset_broker = workset_broker
        self.cache_controller = cache_controller
        self.tree_cache = tree_cache
        self.host_pool = cache_controller.mem_pool_host
        self.storage_spill_enabled = bool(
            supports_agentic_kv_spill(cache_controller.storage_backend)
            and self.host_pool is not None
        )
        self.device_pool = token_allocator.get_kvcache()
        self.page_size = int(page_size)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid P Host staging TP rank")
        if self.tp_size == 1:
            self.owner = f"p:{os.getpid()}"
        else:
            self.owner = f"p-group:{os.getenv('SGLANG_AGENTIC_KV_ENGINE_ID', 'prefill')}"
        self.high_watermark = float(high_watermark)
        self.low_watermark = float(low_watermark)
        self.hard_watermark = float(hard_watermark)
        self.arena_numa_node = int(arena_numa_node)
        self.arena_domain = int(arena_domain)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid D Host staging TP rank")
        self.expected_tool_seconds = expected_tool_seconds or {}
        self.eviction_controller = eviction_controller
        self.arena = SharedHostSnapshotArena(
            arena_directory,
            int(arena_capacity_bytes),
            backend=arena_backend,
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
        # Edge-triggered notifications consumed by the P scheduler. The
        # control worker owns ledger/I/O progress; the scheduler receives only
        # snapshot ids whose allocator/Radix boundary may now advance.
        self._scheduler_events: queue.SimpleQueue = queue.SimpleQueue()
        self._ledger_entries_cache: dict[str, dict[str, Any]] = {}
        self._pending_host_offers: dict[str, dict[str, Any]] = {}
        self.max_h2d_inflight = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT", "4"))
        )
        from sglang.srt.disaggregation.agentic_hybrid_transfer import request_owned_mamba_enabled
        self.h2d_decoupled = bool(
            os.getenv("SGLANG_AGENTIC_KV_P_H2D_DECOUPLED", "0").lower()
            in {"1", "true"}
            and self.tp_size == 1
            and tree_cache.supports_mamba()
            and request_owned_mamba_enabled()
        )
        # Experimental scheduler-side refill only; no allocator or Radix
        # mutation moves to an I/O thread. Dense models and TP keep their
        # existing admission path, including when this variable is inherited.
        self.h2d_event_progress = self.h2d_decoupled and os.getenv(
            "SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS", "0"
        ).lower() in {"1", "true"}
        self.h2d_async_prepare = self.h2d_event_progress and os.getenv(
            "SGLANG_AGENTIC_KV_P_HOST_ASYNC_PREPARE", "0"
        ).lower() in {"1", "true"}
        # Only the control worker prepares these scheduler-selected requests.
        # No live Req is shared with that worker; tokens are immutable copies.
        self._host_prepares: dict[str, dict[str, Any]] = {}
        self._host_prepare_results: dict[str, tuple] = {}
        # One bounded stage may await scheduler binding while the other is
        # preparing/copying. This does not reserve a separate HBM pool: every
        # item must still obtain the original complete hybrid workset lease.
        self._h2d_resident_reservations: set[str] = set()
        self.h2d_chunk_tokens = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS", "1024"))
        )
        self._h2d_bounce_depth = max(
            2, int(os.getenv("SGLANG_AGENTIC_KV_P_H2D_BOUNCE_DEPTH", "2"))
        )
        self._h2d_host_copy_pool = HostCopyWorkerPool(
            f"agentic-p-h2d-host-copy-{os.getpid()}",
            max(
                1,
                int(
                    os.getenv(
                        "SGLANG_AGENTIC_KV_P_H2D_HOST_COPY_WORKERS",
                        str(min(2, self.max_h2d_inflight)),
                    )
                ),
            ),
        )
        # Slow ingress is launched by each D on its own CUDA stream.  P owns
        # only the latency-sensitive demand H2D stream.
        current_device = torch.cuda.current_device()
        h2d_priority = -1
        self._h2d_poisoned = False
        # A lane owns every mutable resource touched by one pageable-Host H2D
        # pipeline.  Sharing either the pinned bounce or the GPU staging tensor
        # would let a second snapshot overwrite the first while its CUDA work
        # is still in flight.  Separate lanes therefore provide real bounded
        # concurrency instead of merely queueing several copies on one stream.
        self._h2d_lanes = []
        for lane_id in range(self.max_h2d_inflight):
            self._h2d_lanes.append(
                {
                    "lane_id": lane_id,
                    "stream": torch.cuda.Stream(
                        device=current_device, priority=h2d_priority
                    ),
                    "staging": LayerFirstD2HStaging(
                        self.device_pool, self.h2d_chunk_tokens
                    ),
                    "host_bounces": tuple(
                        PinnedMHAHostBounce(
                            self.device_pool, self.h2d_chunk_tokens
                        )
                        for _ in range(self._h2d_bounce_depth)
                    ),
                }
            )
        for lane in self._h2d_lanes:
            lane["host_bounce"] = lane["host_bounces"][0]
        # Compatibility aliases for focused tests and out-of-tree diagnostics.
        self._h2d_stream = self._h2d_lanes[0]["stream"]
        self._h2d_staging = self._h2d_lanes[0]["staging"]
        self._h2d_host_bounce = self._h2d_lanes[0]["host_bounce"]
        # A reservation owns only an I/O lane, never HBM token capacity.  It is
        # acquired before a Slow workset intent is created and retained through
        # H2D plus Radix bind/handoff.  Consequently the number of Slow intents
        # and physical leases can never exceed the number of independent lanes.
        self._h2d_lane_reservations: dict[str, int] = {}
        self._spill_threads: dict[str, threading.Thread] = {}
        self._spilling_pressure = False
        self._host_eviction_pressure = False
        self._host_eviction_required_bytes = 0
        self._host_eviction_count = 0
        self._host_eviction_tokens = 0
        self._host_eviction_bytes = 0
        self._host_eviction_local_released: set[str] = set()
        self._last_host_eviction_blocked_log = 0.0
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
        # The worker never touches the GPU token allocator or Radix cache.
        # It may only chain DMA for a load whose normal-workspace pages were
        # already allocated by gate_request(); ownership-changing allocation
        # and Radix bind remain on the scheduler thread.
        self._state_lock = threading.RLock()
        self._async_control = os.getenv(
            "SGLANG_AGENTIC_KV_P_ASYNC_CONTROL", "1"
        ).lower() not in {"0", "false", "no", "off"}
        self.h2d_async_prepare = self.h2d_async_prepare and self._async_control
        self._control_interval = max(
            0.001,
            float(
                os.getenv(
                    "SGLANG_AGENTIC_KV_P_ASYNC_CONTROL_INTERVAL_SECONDS",
                    "0.005",
                )
            ),
        )
        self._control_idle_backstop = max(
            self._control_interval,
            float(
                os.getenv(
                    "SGLANG_AGENTIC_KV_P_ASYNC_CONTROL_IDLE_BACKSTOP_SECONDS",
                    "5.0",
                )
            ),
        )
        self._admission_batch = max(
            1,
            int(os.getenv("SGLANG_AGENTIC_KV_P_HOST_ADMISSION_BATCH", "16")),
        )
        # Storage pressure must not be allowed to pin finished generations in
        # D HBM forever.  Keep the historical wait-forever behavior by
        # default, but let deployments bound that wait and fail open to a
        # correct full-Prefill recompute when every lower tier is saturated.
        self._capacity_wait_timeout_seconds = max(
            0.0,
            float(
                os.getenv(
                    "SGLANG_AGENTIC_KV_HOST_CAPACITY_WAIT_TIMEOUT_SECONDS", "0"
                )
            ),
        )
        self._control_wakeup = threading.Event()
        self._ledger_event_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._ledger_event_ready = threading.Event()
        # Set only for startup, overflow, watcher failure, and the low-rate
        # recovery backstop. Normal progress consumes per-snapshot deltas.
        self._ledger_changed = threading.Event()
        self._ledger_changed.set()
        self._last_ledger_refresh = 0.0
        self._ledger_watcher = None
        self._ledger_watcher_thread = None
        # The arena is physically allocated and first-touched once at startup.
        # Snapshot admission therefore needs no per-generation prefault pool.
        self._prefault_worker_count = 0
        self._control_cycles = 0
        self._control_errors = 0
        self._control_total_seconds = 0.0
        self._control_max_seconds = 0.0
        self._control_last_stats = time.monotonic()
        self._control_thread = None
        logger.info(
            "Agentic shared Host arena enabled directory=%s backend=%s "
            "capacity_gib=%.1f "
            "reserved_hbm_mib=0 h2d_priority=%d h2d_max_inflight=%d "
            "h2d_chunk_tokens=%d host_bounce=preallocated_pinned "
            "arena_preallocated=true preallocation_s=%.3f "
            "h2d_bounce_depth=%d host_copy_workers=%d",
            self.arena.directory,
            self.arena.backend,
            self.arena.capacity_bytes / (1024**3),
            h2d_priority,
            self.max_h2d_inflight,
            self.h2d_chunk_tokens,
            self.arena.preallocation_seconds,
            self._h2d_bounce_depth,
            len(self._h2d_host_copy_pool._threads),
        )
        if self._async_control:
            try:
                self._ledger_watcher = AgenticDirectoryChangeWatcher(
                    self.ledger.event_directory
                )
                self._ledger_watcher_thread = threading.Thread(
                    target=self._ledger_watch_worker,
                    name=f"agentic-p-ledger-watch-{os.getpid()}",
                    daemon=True,
                )
                self._ledger_watcher_thread.start()
            except Exception:
                # Keep the previous bounded polling behavior on platforms
                # without inotify. Linux node-local deployments take the
                # event-driven path.
                self._ledger_watcher = None
                logger.exception(
                    "Agentic P ledger inotify unavailable; using polling fallback"
                )
            self._control_thread = threading.Thread(
                target=self._control_worker,
                name=f"agentic-p-control-{os.getpid()}",
                daemon=True,
            )
            self._control_thread.start()
            logger.info(
                "Agentic P async control enabled interval_ms=%.3f "
                "idle_backstop_s=%.3f snapshot_delta_events=%s "
                "admission_batch=%d prefault_workers=%d "
                "capacity_wait_timeout_s=%.3f "
                "pageable_snapshot_mmap=true arena_preallocated=true",
                self._control_interval * 1000.0,
                self._control_idle_backstop,
                self._ledger_watcher is not None,
                self._admission_batch,
                self._prefault_worker_count,
                self._capacity_wait_timeout_seconds,
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

    def _offer_targets_this_arena(self, offer: dict[str, Any]) -> bool:
        """Return whether an offer belongs to this P-owned Host arena.

        NUMA identifies the physical memory locality, while arena_domain
        identifies the P owner within that NUMA node.  Older single-P-per-NUMA
        offers do not contain arena_domain; retain their NUMA-only behavior so
        existing 1P and 2P launch configurations remain compatible.
        """

        tp_rank = int(getattr(self, "tp_rank", 0))
        rank_offer = offer.get("rank_offers", {}).get(str(tp_rank), {})
        offer_numa = rank_offer.get(
            "arena_numa_node", offer.get("arena_numa_node", -1)
        )
        if self.arena_numa_node >= 0 and int(offer_numa) != self.arena_numa_node:
            return False
        offer_domain = int(offer.get("arena_domain", -1))
        configured_domain = int(getattr(self, "arena_domain", -1))
        return not (
            configured_domain >= 0
            and offer_domain >= 0
            and offer_domain != configured_domain
        )

    def _publish_arena_grant(
        self, snapshot_id: str, record: dict[str, Any]
    ) -> None:
        with self._get_state_lock():
            if self.active.get(snapshot_id) is not record:
                return
        claimed = record["offer"]
        snapshot = record["snapshot"]
        grant = {
            "kind": "shared_host_extent",
            "seq": 0,
            "tp_rank": self.tp_rank,
            "arena_path": snapshot.path,
            "arena_offset": int(getattr(snapshot, "file_offset", 0)),
            "byte_size": snapshot.byte_size,
            "token_count": snapshot.token_count,
            "arena_numa_node": self.arena_numa_node,
        }
        publish_error = None
        try:
            published = (
                self.ledger.publish_grants(snapshot_id, self.owner, [grant])
                if self.tp_size == 1
                else self.ledger.publish_rank_grant(
                    snapshot_id,
                    self.owner,
                    grant,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
            )
        except Exception as exc:
            publish_error = exc
            published = False
        if not published:
            # An exception can leave commit visibility uncertain.  A False
            # return instead means an authoritative predicate failed.  Read
            # the entry once and retain this extent only while we still own a
            # publishable request-generation reservation.
            lookup_error = None
            try:
                current = self.ledger.get(snapshot_id)
            except Exception as exc:
                lookup_error = exc
                current = None
                if publish_error is None:
                    raise
            rank_grant = None
            if current is not None:
                rank_grant = current.get("rank_grants", {}).get(
                    str(int(self.tp_rank))
                )
                if self.tp_size == 1:
                    grants = current.get("grants", [])
                    rank_grant = None if not grants else grants[0]
            if rank_grant == grant:
                published = True
            elif lookup_error is not None or (
                current is not None
                and current.get("p_owner") == self.owner
                and current.get("state")
                in {
                    HostStageState.HOST_RESERVED.value,
                    HostStageState.HOST_WRITING.value,
                }
            ):
                record["grant_publish_pending"] = True
                logger.warning(
                    "AgenticKV shared_host_grant_publish_retry snapshot=%s "
                    "error=%s",
                    snapshot_id,
                    publish_error,
                )
                return
            else:
                if (
                    current is not None
                    and current.get("state") == HostStageState.ABORTING.value
                ):
                    record.update(
                        grant_publish_pending=False,
                        failure_reason="shared_host_grant_publish_aborted",
                        free_host_on_abort=True,
                    )
                    with self._get_state_lock():
                        if self.active.get(snapshot_id) is record:
                            self.active.pop(snapshot_id, None)
                        self.aborting[snapshot_id] = record
                    return
                with self._get_state_lock():
                    if self.active.get(snapshot_id) is record:
                        self.active.pop(snapshot_id, None)
                self.arena.release(snapshot)
                logger.warning(
                    "AgenticKV shared_host_grant_publish_cancel snapshot=%s "
                    "authoritative_state=%s",
                    snapshot_id,
                    None if current is None else current.get("state"),
                )
                return
        if published:
            record.pop("grant_publish_pending", None)
        else:
            return
        logger.info(
            "AgenticKV shared_host_extent_ready snapshot=%s bytes=%d "
            "arena_preallocated=true",
            snapshot_id,
            snapshot.byte_size,
        )

    def _progress_arena_grants(self) -> None:
        with self._get_state_lock():
            pending_grants = [
                (snapshot_id, record)
                for snapshot_id, record in self.active.items()
                if record.get("grant_publish_pending")
            ]
        for snapshot_id, record in pending_grants:
            self._publish_arena_grant(snapshot_id, record)

    def _admit_one(self, ledger_entries=None) -> bool:
        if ledger_entries is None:
            offers = self.ledger.list_state(
                HostStageState.OFFERED, HostStageState.HOST_RESERVED
            )
        else:
            offers = [
                dict(value)
                for value in ledger_entries.values()
                if value.get("state")
                in {
                    HostStageState.OFFERED.value,
                    HostStageState.HOST_RESERVED.value,
                }
            ]
            offers.sort(
                key=lambda item: (item.get("created_at", 0.0), item["snapshot_id"])
            )
        offers = [offer for offer in offers if self._offer_targets_this_arena(offer)]
        if not offers:
            return False
        offer = offers[0]
        snapshot_id = offer["snapshot_id"]
        with self._get_state_lock():
            if snapshot_id in getattr(self, "active", {}) or snapshot_id in getattr(
                self, "host_ready", {}
            ):
                return False
        tp_rank = int(getattr(self, "tp_rank", 0))
        tp_size = int(getattr(self, "tp_size", 1))
        rank_offer = offer.get("rank_offers", {}).get(str(tp_rank), {})
        if tp_size > 1 and not rank_offer:
            return False
        # Capacity is a transient condition.  Leave the offer unclaimed so a
        # later control cycle can admit it after older snapshots are consumed.
        # Claiming first would turn ordinary backpressure into a rejection and
        # force the D side to abandon an otherwise valid host fallback.
        offered_byte_size = int(
            rank_offer.get("byte_size", offer.get("byte_size", 0))
        )
        arena = getattr(self, "arena", None)
        hard_capacity_bytes = (
            None
            if arena is None
            else int(arena.capacity_bytes * self.hard_watermark)
        )
        if (
            hard_capacity_bytes is not None
            and offered_byte_size > hard_capacity_bytes
        ):
            self.ledger.reject_unclaimed_offer(
                snapshot_id,
                reason="shared_host_snapshot_exceeds_hard_capacity",
            )
            logger.warning(
                "AgenticKV shared_host_oversize_recompute snapshot=%s "
                "bytes=%d hard_capacity_bytes=%d",
                snapshot_id,
                offered_byte_size,
                hard_capacity_bytes,
            )
            return True
        if offered_byte_size > 0 and not self._can_admit(offered_byte_size):
            # Pressure can occur just below the high watermark when the next
            # complete snapshot would cross the hard limit.  Treat that
            # blocked admission as an eviction trigger instead of waiting for
            # another writer to make the current usage numerically larger.
            self._host_eviction_pressure = True
            self._host_eviction_required_bytes = max(
                int(getattr(self, "_host_eviction_required_bytes", 0)),
                offered_byte_size,
            )
            wakeup = getattr(self, "_control_wakeup", None)
            if wakeup is not None:
                wakeup.set()
            return False
        if tp_size == 1:
            claimed = self.ledger.claim(offer["snapshot_id"], self.owner)
        else:
            claimed = self.ledger.claim_rank(
                offer["snapshot_id"],
                self.owner,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        if claimed is None:
            return False
        token_count = int(claimed["token_count"])
        if token_count <= 0 or token_count % self.page_size:
            self._reject(claimed, "unaligned_token_count")
            return True
        claimed_rank_offer = claimed.get("rank_offers", {}).get(
            str(tp_rank), {}
        )
        byte_size = int(
            claimed_rank_offer.get("byte_size", claimed.get("byte_size", 0))
        )
        if byte_size <= 0:
            self._reject(claimed, "invalid_host_extent_size")
            return True
        if not self._can_admit(byte_size):
            # Capacity can change between the pre-check and ownership claim.
            # Keep HOST_RESERVED and retry after older snapshots leave; D
            # retains the authoritative HBM copy throughout this wait.
            self._host_eviction_pressure = True
            self._host_eviction_required_bytes = max(
                int(getattr(self, "_host_eviction_required_bytes", 0)),
                byte_size,
            )
            wakeup = getattr(self, "_control_wakeup", None)
            if wakeup is not None:
                wakeup.set()
            return False
        try:
            snapshot = self.arena.create(
                claimed["snapshot_id"], token_count, self.device_pool, byte_size
            )
        except Exception:
            logger.exception(
                "Failed to allocate shared Host extent for %s; retrying",
                claimed["snapshot_id"],
            )
            return False
        with self._get_state_lock():
            record = {
                "offer": claimed,
                "snapshot": snapshot,
                "loading": False,
            }
            self.active[claimed["snapshot_id"]] = record
        # The process-lifetime arena was populated at startup, so the grant is
        # immediately safe for D to map and write.
        self._publish_arena_grant(claimed["snapshot_id"], record)
        return True

    def _admit_batch(
        self, ledger_entries=None, *, replace_pending: bool = False
    ) -> int:
        """Admit several complete snapshots per control cycle.

        The old one-offer-per-scheduler-tick rule was visible as tens of D
        requests retaining HBM while P had abundant Host capacity.  Each
        admission still owns one complete request-generation extent; this is
        batching of control operations, not partial KV admission.
        """

        if ledger_entries is None:
            offers = self.ledger.list_state(
                HostStageState.OFFERED, HostStageState.HOST_RESERVED
            )
        else:
            if replace_pending:
                self._pending_host_offers.clear()
            for snapshot_id, value in ledger_entries.items():
                if value.get("state") in {
                    HostStageState.OFFERED.value,
                    HostStageState.HOST_RESERVED.value,
                } and self._offer_targets_this_arena(value):
                    self._pending_host_offers[snapshot_id] = dict(value)
                else:
                    self._pending_host_offers.pop(snapshot_id, None)
            offers = list(self._pending_host_offers.values())
            offers.sort(
                key=lambda item: (item.get("created_at", 0.0), item["snapshot_id"])
            )
        offers = [offer for offer in offers if self._offer_targets_this_arena(offer)]
        admitted = 0
        # Pass a single-entry view so _admit_one preserves its validation and
        # atomic claim/publish behavior without repeatedly sorting the ledger.
        for offer in offers[: self._admission_batch]:
            if self._admit_one({offer["snapshot_id"]: offer}):
                admitted += 1
                self._pending_host_offers.pop(offer["snapshot_id"], None)
        return admitted

    def _release_record(self, record: dict[str, Any]) -> bool:
        snapshot = record.get("snapshot")
        if snapshot is None:
            return True
        if record.get("remote_host"):
            try:
                snapshot.close(unlink=False)
            except Exception:
                logger.exception("Failed to close remote Shared-Host mapping")
                return False
            record.pop("snapshot", None)
            return True
        if not self.arena.release(snapshot):
            return False
        record.pop("snapshot", None)
        return True

    def _release_consumed_owned_host(self, ledger_entries, *, snapshot_ids=None) -> None:
        """Return an arena extent after remote recovery reaches a safe terminal."""

        ids = ledger_entries.keys() if snapshot_ids is None else snapshot_ids
        for snapshot_id in ids:
            entry = ledger_entries.get(snapshot_id)
            if (
                entry is None
                or entry.get("p_owner") != self.owner
                or entry.get("state")
                not in {
                    HostStageState.CONSUMED.value,
                    HostStageState.FAILED.value,
                }
            ):
                continue
            # A remote recovery rank can advance the shared ledger from
            # HOST_READY to H2D_LOADING before this physical Host owner moves
            # its local record from active to host_ready. Terminal cleanup
            # must therefore inspect both containers. Remove by object
            # identity under the manager lock, and restore to the same
            # container if the physical arena release fails.
            releasable = {}
            with self._get_state_lock():
                for container_name in ("host_ready", "active"):
                    container = getattr(self, container_name, None)
                    if container is None:
                        continue
                    record = container.get(snapshot_id)
                    if (
                        record is None
                        or record.get("loading")
                        or container.get(snapshot_id) is not record
                    ):
                        continue
                    container.pop(snapshot_id, None)
                    item = releasable.setdefault(id(record), [record, []])
                    item[1].append(container_name)
            for record, container_names in releasable.values():
                if not self._release_record(record):
                    with self._get_state_lock():
                        for container_name in container_names:
                            getattr(self, container_name).setdefault(
                                snapshot_id, record
                            )
                    continue
                logger.info(
                    "AgenticKV shared_host_remote_recovery_release "
                    "snapshot=%s containers=%s host_domain=%d "
                    "recovery_domain=%s state=%s",
                    snapshot_id,
                    ",".join(container_names),
                    getattr(self, "arena_domain", -1),
                    entry.get("recovery_domain"),
                    entry.get("state"),
                )

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

    def _poll_aborting(self, ledger_entries=None, *, snapshot_ids=None) -> None:
        with self._get_state_lock():
            aborting = (
                list(self.aborting.items())
                if snapshot_ids is None
                else [
                    (snapshot_id, self.aborting[snapshot_id])
                    for snapshot_id in snapshot_ids
                    if snapshot_id in self.aborting
                ]
            )
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

    def _poll_active(self, ledger_entries=None, *, snapshot_ids=None) -> None:
        with self._get_state_lock():
            active = (
                list(self.active.items())
                if snapshot_ids is None
                else [
                    (snapshot_id, self.active[snapshot_id])
                    for snapshot_id in snapshot_ids
                    if snapshot_id in self.active
                ]
            )
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
            if state == HostStageState.HOST_READY.value or (
                state == HostStageState.H2D_LOADING.value
                and self._recovery_targets_this_manager(ledger_entry)
            ):
                # The complete Shared-Host snapshot is now the authoritative
                # parent copy.  A Direct marker may have raced the D timeout:
                # its workset can already be allocated even though no Direct
                # DMA ever started.  Retire only that unstarted Direct owner
                # before Slow recovery asks for the same request-generation
                # workset.  In TP mode the broker turns this into the normal
                # group-synchronous retirement; an in-flight Direct attempt is
                # deliberately untouched and must finish through its own
                # transport terminal path.
                workset_broker = getattr(self, "workset_broker", None)
                if workset_broker is not None:
                    workset_broker.supersede_unstarted(
                        snapshot_id,
                        owner=workset_broker.direct_owner(snapshot_id),
                    )
                # HOST_READY means D has finished writing the complete tmpfs
                # extent; it does *not* mean P needs the snapshot now.  A peer
                # TP rank may already have claimed the same group restore and
                # advanced the shared ledger to H2D_LOADING before this rank's
                # control worker observes HOST_READY.  That transition is
                # monotonic proof of the same complete Host copy, so the
                # selected recovery group must still publish its local record
                # to gate_request().  Otherwise the first claimant reserves
                # its shard while a slower peer leaves its record in active
                # forever, deadlocking the all-rank Host command.
                #
                # Keep the extent lazy and let gate_request() materialize only
                # the selected slow-recovery request; a non-selected physical
                # Host owner must continue to retain its record for the remote
                # recovery group.
                entry["ready_at"] = time.time()
                with self._get_state_lock():
                    # gate_request may observe the ledger before this move,
                    # but it treats HOST_READY as deferred and retries.
                    self.host_ready[snapshot_id] = entry
                    self.active.pop(snapshot_id, None)
                AgenticPHostStagingManager._notify_scheduler(
                    self, "host_ready", snapshot_id
                )
                logger.info(
                    "AgenticKV shared_host_ready snapshot=%s tokens=%d bytes=%d "
                    "materialize=deferred_until_selected",
                    snapshot_id,
                    entry["offer"]["token_count"],
                    entry["offer"]["byte_size"],
                )
            elif state == HostStageState.REJECTED.value:
                if self._release_record(entry):
                    with self._get_state_lock():
                        if self.active.get(snapshot_id) is entry:
                            self.active.pop(snapshot_id, None)
            elif state in {
                HostStageState.CONSUMED.value,
                HostStageState.FAILED.value,
            }:
                # Terminal physical-owner cleanup is identity-safe and keeps
                # the exact record discoverable when arena release must retry.
                self._release_consumed_owned_host(
                    {snapshot_id: ledger_entry},
                    snapshot_ids=(snapshot_id,),
                )
            elif state == HostStageState.ABORTING.value:
                if ledger_entry.get("h2d_abort_started", False):
                    # Remote/selected recovery cancellation uses loader-drain
                    # acknowledgements.  Do not move this physical Host-owner
                    # record into the D2H writer-drain queue; retain it until
                    # the all-rank loader protocol publishes FAILED, then let
                    # the terminal helper above release the arena extent.
                    continue
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
            # Normal Host waiting remains unregistered.  Mooncake spill is an
            # exceptional pressure path that genuinely needs a readable P
            # mapping, so materialize it here rather than on every HOST_READY.
            record["snapshot"].materialize()
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
                state = "missing" if current is None else current.state.value
                record["spill_blocked_reason"] = f"manifest_state:{state}"
                raise RuntimeError(
                    "spill requires an owned SLOW_FALLBACK manifest; "
                    f"observed={state}"
                )
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
            snapshot_store.continue_slow_publish(manifest)
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
                        snapshot_store.rollback_slow_publish(
                            observed, fallback_manifest
                        )
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
                if record.get("spill_blocked_reason"):
                    logger.error(
                        "AgenticKV spill_quarantined snapshot=%s reason=%s; "
                        "Shared-Host recovery remains available",
                        snapshot_id,
                        record["spill_blocked_reason"],
                    )
            with self._get_state_lock():
                self.spills.pop(snapshot_id, None)

    def _maybe_spill(self) -> None:
        if not getattr(self, "storage_spill_enabled", True):
            # Shared-Arena-only mode evicts/fails soft at its own hard capacity;
            # it never creates a hidden native HiCache/Mooncake data path.
            return
        if int(getattr(self, "tp_size", 1)) > 1:
            # A Mooncake generation must publish all TP rank page-key sets as
            # one manifest.  Until that storage transaction is group-aware,
            # retaining the complete Shared-Arena snapshot is safer than
            # exposing a rank-partial spill.  Direct and Shared-Host remain
            # fully functional; pressure fails soft to recomputation.
            if not getattr(self, "_tp_spill_warning_logged", False):
                logger.warning(
                    "AgenticKV Mooncake pressure spill is disabled for TP>1; "
                    "Shared Arena remains authoritative"
                )
                self._tp_spill_warning_logged = True
            return
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
                and not record.get("remote_host")
                and not record.get("spill_blocked_reason")
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

    @staticmethod
    def _host_eviction_key(
        snapshot_id: str, record: dict[str, Any]
    ) -> tuple[int, float, str]:
        """Prefer cheap recomputation, then the oldest equal-size wait."""

        offer = record["offer"]
        return (
            int(offer.get("token_count", 0)),
            float(record.get("ready_at", offer.get("created_at", 0.0))),
            str(snapshot_id),
        )

    @staticmethod
    def _host_record_allocation_bytes(record: dict[str, Any]) -> int:
        snapshot = record.get("snapshot")
        return int(
            getattr(
                snapshot,
                "allocation_bytes",
                record.get("offer", {}).get("byte_size", 0),
            )
        )

    def _release_evicted_host_rank(
        self, snapshot_id: str, entry: dict[str, Any]
    ) -> bool:
        """Release one local shard, then ACK the TP-wide eviction fence."""

        with self._get_state_lock():
            record = self.host_ready.get(snapshot_id)
            container = self.host_ready
            container_name = "host_ready"
            if record is None:
                record = self.active.get(snapshot_id)
                container = self.active
                container_name = "active"
            if record is not None and record.get("loading"):
                # A peer may have reserved a lane just before rank0 won the
                # ledger CAS.  Its prepare call will observe EVICTING, cancel
                # the unstarted lease, and make this record eligible next turn.
                return False
            if record is not None:
                container.pop(snapshot_id, None)
        if record is not None:
            token_count = int(record.get("offer", {}).get("token_count", 0))
            byte_size = self._host_record_allocation_bytes(record)
            if not self._release_record(record):
                with self._get_state_lock():
                    destination = (
                        self.host_ready
                        if container_name == "host_ready"
                        else self.active
                    )
                    destination.setdefault(snapshot_id, record)
                logger.error(
                    "AgenticKV shared_host_evict_release_retry snapshot=%s "
                    "tp_rank=%d/%d",
                    snapshot_id,
                    self.tp_rank,
                    self.tp_size,
                )
                return False
            self._host_eviction_local_released.add(snapshot_id)
            terminal = bool(entry.get("terminal_on_release"))
            if not terminal:
                self._host_eviction_count += 1
                self._host_eviction_tokens += token_count
                self._host_eviction_bytes += byte_size
            logger.warning(
                "AgenticKV %s snapshot=%s tokens=%d "
                "bytes=%d tp_rank=%d/%d reason=%s",
                "shared_host_final_release" if terminal else "shared_host_evict_release",
                snapshot_id,
                token_count,
                byte_size,
                self.tp_rank,
                self.tp_size,
                entry.get("eviction_reason", "shared_host_pressure_eviction"),
            )
        if snapshot_id not in self._host_eviction_local_released:
            return False
        acknowledged = self.ledger.complete_host_eviction_rank(
            snapshot_id,
            self.owner,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        if acknowledged and not entry.get("terminal_on_release"):
            self._notify_scheduler("recompute_required", snapshot_id)
        return acknowledged

    def _progress_final_host_cleanup(self, entries) -> None:
        """Reap late application finals on the existing background resync.

        HOST_READY is already D2H-fenced. The eviction CAS also excludes every
        recovery claim/assignment, so a final marker alone never cancels DMA.
        All Host ranks ACK physical release before publishing CONSUMED; this
        is terminal reclamation, not pressure eviction or recomputation.
        """
        if self.tp_rank != 0 or getattr(self, "storage_spill_enabled", False):
            return
        store = getattr(self, "_final_host_claim_store", None)
        if store is None:
            directory = os.getenv("SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR", "")
            if not directory:
                ready = os.getenv("SGLANG_PD_P_READY_DIR", "")
                if not ready:
                    return
                directory = os.path.join(ready, "early-claims")
            store = self._final_host_claim_store = AgenticEarlyClaimStore(directory)
        candidates = [
            (sid, entry) for sid, entry in entries.items()
            if entry.get("p_owner") == self.owner
            and entry.get("state") == HostStageState.HOST_READY.value
        ]
        if not candidates:
            return
        # Bounded marker reads; no directory scan on Forward or each I/O tick.
        start = getattr(self, "_final_host_cursor", 0) % len(candidates)
        count = min(128, len(candidates))
        self._final_host_cursor = (start + count) % len(candidates)
        for i in range(count):
            snapshot_id, entry = candidates[(start + i) % len(candidates)]
            try:
                request_id, generation = snapshot_id.rsplit(":", 1)
                request = RequestGeneration(request_id, int(generation))
            except (ValueError, TypeError):
                logger.error("Invalid Host generation for final cleanup: %s", snapshot_id)
                continue
            marker = store.read_final(
                # Host offer may be queued after the application final ACK.
                # The generation/tool start, not Host publication, is the fence.
                request, not_before=float(entry.get("tool_started_at", entry.get("created_at", 0))),
                max_age_seconds=float("inf"),
            )
            if not marker or marker.get("kind") != "final":
                continue
            broker = getattr(self, "workset_broker", None)
            if broker is not None and broker.eviction_blocker(snapshot_id) is not None:
                continue
            if not self.ledger.begin_host_eviction(
                snapshot_id, self.owner, tp_size=self.tp_size,
                reason="application_final", terminal_on_release=True,
            ):
                continue
            if broker is not None and broker.eviction_blocker(snapshot_id) is not None:
                raise RuntimeError(f"final Host snapshot retains live workset: {snapshot_id}")
            current = self.ledger.get(snapshot_id)
            if current is not None:
                self._release_evicted_host_rank(snapshot_id, current)

    def _progress_host_evictions(
        self, ledger_entries=None, *, snapshot_ids=None
    ) -> None:
        """Apply group eviction decisions to this rank's local Host arena."""

        if ledger_entries is None:
            entries = self.ledger.snapshot_entries(force_refresh=True)
        else:
            entries = ledger_entries
        ids = entries.keys() if snapshot_ids is None else snapshot_ids
        for snapshot_id in ids:
            entry = entries.get(snapshot_id)
            if entry is None or entry.get("p_owner") != self.owner:
                continue
            state = entry.get("state")
            if state == HostStageState.EVICTING.value:
                self._release_evicted_host_rank(snapshot_id, entry)
            elif state == HostStageState.CONSUMED.value and entry.get("terminal_on_release"):
                self._host_eviction_local_released.discard(snapshot_id)
            elif state == HostStageState.RECOMPUTE_REQUIRED.value:
                # Idempotent cleanup after a final ACK/event race.  Normally
                # the local record was already released in EVICTING.
                with self._get_state_lock():
                    record = self.host_ready.get(snapshot_id)
                    container = self.host_ready
                    if record is None:
                        record = self.active.get(snapshot_id)
                        container = self.active
                    if record is not None and not record.get("loading"):
                        container.pop(snapshot_id, None)
                    else:
                        record = None
                if record is not None:
                    if not self._release_record(record):
                        with self._get_state_lock():
                            container.setdefault(snapshot_id, record)
                        logger.error(
                            "AgenticKV terminal Host eviction cleanup failed "
                            "snapshot=%s tp_rank=%d/%d",
                            snapshot_id,
                            self.tp_rank,
                            self.tp_size,
                        )
                self._host_eviction_local_released.discard(snapshot_id)

    def _maybe_evict_shared_host(self) -> None:
        """Evict complete request-generations down to the configured low mark.

        This is the Shared-Arena-only pressure policy.  Native storage spill
        retains its existing controller and never enters this path.
        """

        if getattr(self, "storage_spill_enabled", False):
            return
        usage = self._host_usage()
        required_bytes = int(
            getattr(self, "_host_eviction_required_bytes", 0)
        )
        required_fits = required_bytes <= 0 or self.arena.can_reserve(
            required_bytes, self.hard_watermark
        )
        if (
            not self._host_eviction_pressure
            and (usage >= self.high_watermark or not required_fits)
        ):
            self._host_eviction_pressure = True
        if not self._host_eviction_pressure or self.tp_rank != 0:
            return

        target_bytes = int(self.arena.capacity_bytes * self.low_watermark)
        with self._get_state_lock():
            candidates = sorted(
                (
                    (snapshot_id, record)
                    for snapshot_id, record in self.host_ready.items()
                    if not record.get("loading")
                    and not record.get("remote_host")
                ),
                key=lambda item: self._host_eviction_key(item[0], item[1]),
            )
        initiated = 0
        for snapshot_id, record in candidates:
            required_fits = required_bytes <= 0 or self.arena.can_reserve(
                required_bytes, self.hard_watermark
            )
            if self.arena.used_bytes <= target_bytes and required_fits:
                break
            broker = getattr(self, "workset_broker", None)
            eviction_blocker = (
                None
                if broker is None
                else broker.eviction_blocker(snapshot_id)
            )
            if eviction_blocker is not None:
                logger.error(
                    "AgenticKV shared_host_evict_invariant_blocked "
                    "snapshot=%s lease=%s",
                    snapshot_id,
                    eviction_blocker,
                )
                continue
            if not self.ledger.begin_host_eviction(
                snapshot_id,
                self.owner,
                tp_size=self.tp_size,
                reason="shared_host_pressure_shortest_first",
            ):
                continue
            # The HOST_READY->EVICTING CAS is the final ownership boundary.
            # Recovery must pin the ledger before allocating a workset, so no
            # lease may appear after this transition.  Check the invariant
            # again after the CAS to turn any future ordering regression into
            # an immediate fail-stop instead of an orphaned P-HBM reservation.
            if broker is not None:
                post_cas_blocker = broker.eviction_blocker(snapshot_id)
                if post_cas_blocker is not None:
                    raise RuntimeError(
                        "evicted Host snapshot retains live P workset: "
                        f"snapshot={snapshot_id} lease={post_cas_blocker}"
                    )
            entry = self.ledger.get(snapshot_id)
            if entry is None:
                continue
            if self._release_evicted_host_rank(snapshot_id, entry):
                initiated += 1

        usage = self._host_usage()
        required_fits = required_bytes <= 0 or self.arena.can_reserve(
            required_bytes, self.hard_watermark
        )
        if usage <= self.low_watermark and required_fits:
            self._host_eviction_pressure = False
            self._host_eviction_required_bytes = 0
        elif initiated == 0:
            now = time.monotonic()
            if now - self._last_host_eviction_blocked_log >= 5.0:
                logger.warning(
                    "AgenticKV shared_host_evict_blocked usage=%.4f "
                    "high=%.4f low=%.4f required_bytes=%d required_fits=%s "
                    "host_ready=%d active=%d loads=%d",
                    usage,
                    self.high_watermark,
                    self.low_watermark,
                    required_bytes,
                    required_fits,
                    len(self.host_ready),
                    len(self.active),
                    len(self.loads),
                )
                self._last_host_eviction_blocked_log = now

    def _has_local_io_progress(self) -> bool:
        """Return whether CUDA/thread completion still needs short polling."""

        with self._get_state_lock():
            pending_grant = any(
                record.get("grant_publish_pending")
                for record in self.active.values()
            )
            return bool(
                self.loads
                or self.spills
                or getattr(self, "_pending_host_abort_requests", {})
                or getattr(self, "_prestart_recovery_aborts", {})
                or getattr(self, "_host_prepares", {})
                or pending_grant
            )

    def _ledger_watch_worker(self) -> None:
        watcher = self._ledger_watcher
        if watcher is None:
            return
        try:
            while watcher.healthy:
                paths, overflow = watcher.poll(timeout_seconds=None)
                if overflow:
                    self._ledger_changed.set()
                for path in paths:
                    try:
                        event = self.ledger.read_entry_event(path)
                    except (FileNotFoundError, json.JSONDecodeError, ValueError):
                        # A later event for the same snapshot may have replaced
                        # this path between inotify delivery and open.  The
                        # authoritative resync closes that rare race.
                        self._ledger_changed.set()
                        continue
                    self._ledger_event_queue.put(event)
                    self._ledger_event_ready.set()
                if paths or overflow:
                    self._control_wakeup.set()
        except Exception:
            logger.exception(
                "Agentic P ledger watcher failed; switching to polling fallback"
            )
        finally:
            # The control worker observes watcher health and resumes bounded
            # polling. Never leave Host progress dependent on a dead daemon.
            watcher.healthy = False
            self._ledger_changed.set()
            self._control_wakeup.set()

    def _apply_ledger_events(self) -> set[str]:
        """Merge changed request-generations into the in-memory ledger view."""

        self._ledger_event_ready.clear()
        changed: set[str] = set()
        while True:
            try:
                event = self._ledger_event_queue.get_nowait()
            except queue.Empty:
                break
            snapshot_id = str(event["snapshot_id"])
            incoming_revision = int(event.get("revision", 0))
            current = self._ledger_entries_cache.get(snapshot_id)
            current_revision = int((current or {}).get("_event_revision", -1))
            if incoming_revision <= current_revision:
                continue
            entry = event.get("entry")
            if entry is None:
                self._ledger_entries_cache.pop(snapshot_id, None)
            else:
                self._ledger_entries_cache[snapshot_id] = dict(entry)
            changed.add(snapshot_id)
        return changed

    def _poll_once(self, *, force_ledger: bool = False) -> None:
        now = time.monotonic()
        if now < self._next_poll_at:
            return
        self._next_poll_at = now + self._poll_interval
        # Publish populated extents before taking the shared ledger snapshot so
        # D can observe every new grant in this same control cycle.
        self._progress_arena_grants()
        self._progress_host_abort_requests()
        self._progress_prestart_aborts()
        self._progress_h2d_loads()
        self._progress_host_prepares()
        self._progress_spills()
        self._maybe_spill()
        watcher_healthy = bool(
            self._ledger_watcher is not None and self._ledger_watcher.healthy
        )
        full_resync = force_ledger or not watcher_healthy
        if not full_resync:
            full_resync = self._ledger_changed.is_set()
        if now - self._last_ledger_refresh >= self._control_idle_backstop:
            full_resync = True
        full_snapshot = None
        if full_resync:
            # Clear before reading. A concurrent overflow/failure will set the
            # edge again and therefore cannot be lost behind this snapshot.
            self._ledger_changed.clear()
            ledger_entries = self.ledger.snapshot_entries(force_refresh=True)
            self._ledger_entries_cache = ledger_entries
            self._last_ledger_refresh = time.monotonic()
            full_snapshot = ledger_entries
        # Apply events after a resync. Revision checks discard edges already
        # represented by that snapshot while retaining a concurrent newer one.
        changed_snapshot_ids = self._apply_ledger_events()
        if full_snapshot is not None:
            self._poll_active(self._ledger_entries_cache)
            self._poll_aborting(self._ledger_entries_cache)
            self._progress_final_host_cleanup(self._ledger_entries_cache)
            self._release_consumed_owned_host(self._ledger_entries_cache)
            self._progress_host_evictions(self._ledger_entries_cache)
            self._maybe_evict_shared_host()
            self._admit_batch(
                self._ledger_entries_cache,
                replace_pending=True,
            )
        elif changed_snapshot_ids:
            changed_entries = {
                snapshot_id: self._ledger_entries_cache[snapshot_id]
                for snapshot_id in changed_snapshot_ids
                if snapshot_id in self._ledger_entries_cache
            }
            self._poll_active(
                self._ledger_entries_cache,
                snapshot_ids=changed_snapshot_ids,
            )
            self._poll_aborting(
                self._ledger_entries_cache,
                snapshot_ids=changed_snapshot_ids,
            )
            self._release_consumed_owned_host(
                self._ledger_entries_cache,
                snapshot_ids=changed_snapshot_ids,
            )
            self._progress_host_evictions(
                self._ledger_entries_cache,
                snapshot_ids=changed_snapshot_ids,
            )
            self._maybe_evict_shared_host()
            self._admit_batch(changed_entries)
        else:
            self._maybe_evict_shared_host()
        if time.monotonic() - self._last_prune > 5.0:
            self.ledger.prune()
            self._last_prune = time.monotonic()

    def _control_worker(self) -> None:
        while True:
            # Clear before doing work so a wakeup produced during this cycle
            # remains set and makes the following wait return immediately.
            self._control_wakeup.clear()
            started = time.monotonic()
            try:
                self._poll_once(force_ledger=False)
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
                    h2d_loads = len(self.loads)
                    h2d_lanes_owned = len(self._h2d_lane_reservations)
                logger.info(
                    "Agentic P async control stats cycles=%d avg_us=%.1f "
                    "max_ms=%.3f active=%d host_ready=%d spills=%d "
                    "h2d_loads=%d h2d_lanes=%d/%d errors=%d "
                    "host_evictions=%d host_eviction_tokens=%d "
                    "host_eviction_gib=%.3f",
                    self._control_cycles,
                    self._control_total_seconds
                    / max(self._control_cycles, 1)
                    * 1e6,
                    self._control_max_seconds * 1e3,
                    active,
                    host_ready,
                    spills,
                    h2d_loads,
                    h2d_lanes_owned,
                    self.max_h2d_inflight,
                    self._control_errors,
                    self._host_eviction_count,
                    self._host_eviction_tokens,
                    self._host_eviction_bytes / (1024**3),
                )
                self._control_cycles = 0
                self._control_total_seconds = 0.0
                self._control_max_seconds = 0.0
                self._control_last_stats = now
            watcher_healthy = bool(
                self._ledger_watcher is not None and self._ledger_watcher.healthy
            )
            timeout = (
                self._control_interval
                if (
                    self._has_local_io_progress()
                    or self._ledger_event_ready.is_set()
                    or self._ledger_changed.is_set()
                    or not watcher_healthy
                )
                else max(
                    self._control_interval,
                    self._control_idle_backstop
                    - (time.monotonic() - self._last_ledger_refresh),
                )
            )
            self._control_wakeup.wait(timeout)

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
        """Discover readable Host metadata before TP selects a restore.

        A foreign arena's local mapping is created only by gate_request after
        selection. Requiring that mapping here can starve TP discovery. Inspect
        only the async ledger cache; the selected path still revalidates the
        route and claims every shard before allocating or starting any I/O.
        """

        with self._get_state_lock():
            if request_generation.snapshot_id in self.host_ready:
                entry = getattr(self, "_ledger_entries_cache", {}).get(
                    request_generation.snapshot_id
                )
                if (
                    int(getattr(self, "tp_size", 1)) > 1
                    and entry is not None
                    and entry.get("recovery_domain") is not None
                    and not self._recovery_targets_this_manager(entry)
                ):
                    return False
                return True
        tp_size = int(getattr(self, "tp_size", 1))
        if tp_size <= 1:
            return False
        entry = getattr(self, "_ledger_entries_cache", {}).get(
            request_generation.snapshot_id
        )
        if (
            entry is None
            or entry.get("state") != HostStageState.HOST_READY.value
            or entry.get("p_owner") in {None, self.owner}
            or entry.get("recovery_domain") is None
            or not self._recovery_targets_this_manager(entry)
            or int(entry.get("tp_size", 1)) != tp_size
            or entry.get("recovery_owner") not in {None, self.owner}
        ):
            return False
        grants = entry.get("rank_grants", {})
        return set(grants) == {str(rank) for rank in range(tp_size)}

    def tp_host_control_quiescent(self, snapshot_id: str, rid: str) -> bool:
        """Read-only fence check for a cancelled P-group Host command.

        This retires no physical resource or ledger entry. Background abort
        workers and the ordinary TP workset retirement protocol do that work.
        """

        with self._get_state_lock():
            if self._find_h2d_load(snapshot_id, rid=rid)[1] is not None:
                return False
            for name in (
                "aborting", "_pending_host_abort_requests",
                "_prestart_recovery_aborts", "_h2d_lane_reservations",
                "_h2d_resident_reservations",
            ):
                if snapshot_id in getattr(self, name, ()):
                    return False
            record = self.host_ready.get(snapshot_id)
            if record is not None and record.get("loading"):
                return False
        broker = self.workset_broker
        return not any(
            broker.owner_has_unretired_work(snapshot_id, owner=owner)
            for owner in (
                broker.slow_owner(snapshot_id, rid),
                broker.direct_owner(snapshot_id),
            )
        )

    def _recovery_targets_this_manager(self, entry: dict[str, Any]) -> bool:
        domain = entry.get("recovery_domain")
        if domain is None:
            # Backward-compatible single-P/non-dynamic route.
            return entry.get("p_owner") == self.owner
        return int(domain) == int(self.arena_domain)

    def _import_remote_host_record(
        self, snapshot_id: str, entry: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Map a Host extent owned by another arena for local recovery."""

        if entry.get("p_owner") == self.owner:
            return None
        rank_grants = entry.get("rank_grants", {})
        grant = rank_grants.get(str(self.tp_rank))
        if grant is None and self.tp_size == 1:
            grants = entry.get("grants", [])
            grant = None if not grants else grants[0]
        if grant is None:
            return None
        snapshot = LazySharedMHAHostSnapshot(
            path=str(grant["arena_path"]),
            token_count=int(grant.get("token_count", entry["token_count"])),
            device_pool=self.device_pool,
            byte_size=int(grant.get("byte_size", entry["byte_size"])),
            allocation_bytes=int(grant.get("byte_size", entry["byte_size"])),
            create=False,
            file_offset=int(grant.get("arena_offset", 0)),
        )
        record = {
            "offer": dict(entry),
            "snapshot": snapshot,
            "loading": False,
            "ready_at": float(entry.get("updated_at", time.time())),
            "remote_host": True,
        }
        with self._get_state_lock():
            previous = self.host_ready.get(snapshot_id)
            if previous is not None:
                snapshot.close(unlink=False)
                return previous
            self.host_ready[snapshot_id] = record
        logger.info(
            "AgenticKV shared_host_remote_map snapshot=%s host_domain=%s "
            "recovery_domain=%d tp_rank=%d",
            snapshot_id,
            entry.get("arena_domain"),
            self.arena_domain,
            self.tp_rank,
        )
        return record

    def drain_scheduler_events(
        self, max_events: Optional[int] = None
    ) -> tuple[tuple[str, str], ...]:
        """Drain completed control edges without reading the file ledger."""

        events = []
        while max_events is None or len(events) < max_events:
            try:
                events.append(self._scheduler_events.get_nowait())
            except queue.Empty:
                return tuple(events)
        return tuple(events)

    def _notify_scheduler(self, kind: str, snapshot_id: str) -> None:
        events = getattr(self, "_scheduler_events", None)
        if events is None:
            events = queue.SimpleQueue()
            self._scheduler_events = events
        events.put((str(kind), str(snapshot_id)))

    def _reserve_h2d_lane(self, snapshot_id: str) -> Optional[int]:
        """Reserve one physical Slow-I/O lane without reserving P KV pages."""

        with AgenticPHostStagingManager._get_state_lock(self):
            reservations = getattr(self, "_h2d_lane_reservations", None)
            if reservations is None:
                reservations = {}
                self._h2d_lane_reservations = reservations
            existing = reservations.get(snapshot_id)
            if existing is not None:
                return int(existing)
            lane_count = max(1, int(getattr(self, "max_h2d_inflight", 1)))
            resident = getattr(self, "_h2d_resident_reservations", None)
            if getattr(self, "h2d_decoupled", False):
                if resident is None:
                    resident = self._h2d_resident_reservations = set()
                if snapshot_id not in resident and len(resident) >= 2 * lane_count:
                    return None
            occupied = set(int(value) for value in reservations.values())
            for lane_id in range(lane_count):
                if lane_id not in occupied:
                    reservations[snapshot_id] = lane_id
                    if getattr(self, "h2d_decoupled", False):
                        resident.add(snapshot_id)
                    return lane_id
            return None

    def _release_h2d_lane(self, snapshot_id: str) -> None:
        """Return a Slow-I/O lane after handoff or a fenced terminal path."""

        with AgenticPHostStagingManager._get_state_lock(self):
            reservations = getattr(self, "_h2d_lane_reservations", None)
            if reservations is not None:
                reservations.pop(snapshot_id, None)
            resident = getattr(self, "_h2d_resident_reservations", None)
            if resident is not None:
                resident.discard(snapshot_id)

    def _release_quiesced_h2d_lane(self, load: dict[str, Any]) -> None:
        """Recycle transport resources, NOT the Host copy or destination lease."""

        if not getattr(self, "h2d_decoupled", False):
            return
        if not (load.get("h2d_copy_complete") and load.get("io_quiesced")):
            return
        if (load.get("event") is not None or load.get("copy_refs") is not None
                or load.get("prefetch_future") is not None):
            return
        snapshot_id = load["request_generation"].snapshot_id
        with self._get_state_lock():
            if self.loads.get(load.get("rid")) is not load:
                return
            if load.get("transport_lane_released"):
                return
            self._h2d_lane_reservations.pop(snapshot_id, None)
            load["transport_lane_released"] = True
            load["copy_done_at"] = time.monotonic()
        # No Host/lease/CAS mutation here. The bounded resident credit stays
        # until the existing handoff/cancel path calls _release_h2d_lane().
        self._control_wakeup.set()

    def h2d_physical_occupancy(self) -> int:
        with self._get_state_lock():
            return len(self._h2d_lane_reservations)

    def h2d_selected_snapshots(self) -> set[str]:
        with self._get_state_lock():
            return set(self._h2d_resident_reservations)

    def _find_h2d_load(
        self, snapshot_id: str, *, rid: Optional[str] = None
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Resolve a Slow load by request-generation, not transient HTTP rid."""

        if rid is not None:
            load = self.loads.get(rid)
            generation = None if load is None else load.get("request_generation")
            if load is not None and (
                generation is None or generation.snapshot_id == snapshot_id
            ):
                return str(rid), load
        for load_rid, load in self.loads.items():
            generation = load.get("request_generation")
            if generation is not None and generation.snapshot_id == snapshot_id:
                return str(load_rid), load
        return None, None

    def _retire_stale_direct_before_slow(self, snapshot_id: str) -> bool:
        """Fence an obsolete Direct owner before Slow consumes an H2D lane.

        Host placement and recovery routing are independent, so the P that
        observed HOST_READY (and retired its local Direct race) need not be
        the P selected for H2D.  Repeat the owner-scoped transition on the
        selected recovery P.  While an old physical Direct lease is waiting
        for the scheduler-owned allocator release, return ``True`` so Slow
        holds neither a recovery claim nor an H2D lane.
        """

        broker = self.workset_broker
        direct_owner_fn = getattr(broker, "direct_owner", None)
        supersede = getattr(broker, "supersede_unstarted", None)
        get_lease = getattr(broker, "get", None)
        if not (
            callable(direct_owner_fn)
            and callable(supersede)
            and callable(get_lease)
        ):
            # Compatibility for focused fixtures using a minimal broker.
            return False
        direct_owner = direct_owner_fn(snapshot_id)
        supersede(snapshot_id, owner=direct_owner)
        owner_has_unretired_work = getattr(
            broker, "owner_has_unretired_work", None
        )
        if callable(owner_has_unretired_work):
            return bool(
                owner_has_unretired_work(snapshot_id, owner=direct_owner)
            )
        return get_lease(snapshot_id, owner=direct_owner) is not None

    def _h2d_lane_resources(self, lane_id: int):
        """Return the isolated stream/bounce/staging resources for one lane."""

        lanes = getattr(self, "_h2d_lanes", None)
        if lanes is None:
            # Compatibility for focused tests constructed with ``__new__``.
            return {
                "lane_id": 0,
                "stream": self._h2d_stream,
                "staging": self._h2d_staging,
                "host_bounce": self._h2d_host_bounce,
            }
        lane_id = int(lane_id)
        if not 0 <= lane_id < len(lanes):
            raise RuntimeError(f"invalid Slow H2D lane {lane_id}")
        return lanes[lane_id]

    def _prepare_h2d_load_ledger(self, load: dict[str, Any]) -> bool:
        """Idempotently publish that this rank's Slow lane is prepared."""

        request_generation = load["request_generation"]
        if int(getattr(self, "tp_size", 1)) > 1:
            return bool(
                self.ledger.prepare_tp_host_load_rank(
                    request_generation.snapshot_id,
                    self.owner,
                    tp_rank=int(getattr(self, "tp_rank", 0)),
                    tp_size=int(getattr(self, "tp_size", 1)),
                )
            )
        return bool(
            self.ledger.begin_host_load_rank(
                request_generation.snapshot_id,
                self.owner,
                tp_rank=0,
                tp_size=1,
                recovery=True,
            )
        )

    def _cancel_unstarted_h2d_load(self, rid: str, load: dict[str, Any]) -> None:
        """Release a prepared lane whose DMA was never published."""

        request_generation = load["request_generation"]
        snapshot_id = request_generation.snapshot_id
        lease = load["workset_lease"]
        attempt = load["io_attempt"]
        cancelled = self.workset_broker.cancel_io_attempt(
            snapshot_id, load["workset_lease"], load["io_attempt"]
        )
        if not cancelled:
            raise RuntimeError(
                f"cannot cancel started Slow recovery {snapshot_id}"
            )
        self.workset_broker.request_release(snapshot_id, lease)
        if not self.ledger.cancel_d2p_recovery_rank(
            snapshot_id,
            self.owner,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            claim_id=load["recovery_claim_id"],
            lease_id=lease.lease_id,
        ):
            raise RuntimeError(
                f"cannot roll back Slow recovery ownership {snapshot_id}"
            )
        with AgenticPHostStagingManager._get_state_lock(self):
            if self.loads.get(rid) is load:
                self.loads.pop(rid, None)
            load["record"]["loading"] = False
            load["record"].pop("recovery_claim_id", None)
        AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)

    def _complete_shared_host_manifest(self, request_generation) -> bool:
        """Release the persistent fallback fence after P owns the full GPU KV."""

        controller = getattr(self, "cache_controller", None)
        backend = getattr(controller, "storage_backend", None)
        factory = getattr(backend, "agentic_snapshot_store", None)
        if factory is None:
            # Focused unit tests may construct the manager without a storage
            # backend. Production lifecycle mode always installs one.
            return True
        snapshot_store = factory()
        if int(getattr(self, "tp_size", 1)) > 1:
            return snapshot_store.complete_slow_fallback_group(request_generation)
        current = snapshot_store.load(request_generation, require_ready=False)
        if current is None:
            raise RuntimeError(
                f"missing slow fallback manifest for {request_generation.snapshot_id}"
            )
        if current.state is SnapshotState.CONSUMED:
            return True
        snapshot_store.complete_slow_fallback(current)
        return True

    def _submit_h2d_prefetch(self, load: dict[str, Any], start: int) -> bool:
        """Stage one pageable Host chunk into an idle pinned bounce."""

        if load.get("prefetch_future") is not None:
            return False
        device_indices = load["device_indices"]
        start = int(start)
        if start >= len(device_indices):
            return False
        end = min(start + self.h2d_chunk_tokens, len(device_indices))
        lane = AgenticPHostStagingManager._h2d_lane_resources(
            self, int(load.get("h2d_lane_id", 0))
        )
        bounces = tuple(lane.get("host_bounces", (lane["host_bounce"],)))
        current_bounce = load.get("dma_bounce_index")
        bounce_index = (start // max(1, self.h2d_chunk_tokens)) % len(bounces)
        if current_bounce is not None and bounce_index == int(current_bounce):
            if len(bounces) == 1:
                return False
            bounce_index = (bounce_index + 1) % len(bounces)
        snapshot = load["record"]["snapshot"]
        function = getattr(snapshot, "prepare_load_range_to_bounce", None)
        args = ()
        kwargs = {
            "source_start": start,
            "token_count": end - start,
            "host_bounce": bounces[bounce_index],
        }
        pool = getattr(self, "_h2d_host_copy_pool", None)
        if function is None:
            # Compatibility for focused mocks and external snapshot adapters.
            future = Future()
            future.set_result(0.0)
            load["prefetch_legacy"] = True
        elif pool is None:
            # Lightweight fixtures do not construct background workers.  Keep
            # their deterministic behavior while production always uses the
            # asynchronous NUMA-local pool initialized above.
            future = Future()
            started_at = time.perf_counter()
            try:
                function(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(time.perf_counter() - started_at)
        else:
            future = pool.submit(function, *args, **kwargs)
        load.update(
            prefetch_future=future,
            prefetch_start=start,
            prefetch_end=end,
            prefetch_bounce_index=bounce_index,
        )
        return True

    @staticmethod
    def _clear_h2d_prefetch(load: dict[str, Any]) -> None:
        for key in (
            "prefetch_future",
            "prefetch_start",
            "prefetch_end",
            "prefetch_bounce_index",
            "prefetch_legacy",
        ):
            load.pop(key, None)

    @staticmethod
    def _configure_hybrid_h2d_state(load: dict[str, Any]) -> None:
        """Configure the immutable destination once, on the H2D worker."""
        state_indices = getattr(load["workset_lease"], "state_device_indices", ())
        if not state_indices or load.get("hybrid_state_configured"):
            return
        snapshot = load["record"]["snapshot"]
        # The lease allocator returns GPU indices. Lazy arena metadata must
        # remain CPU-only; mirror here, never in scheduler admission. This is
        # the same worker-side conversion used for Attention page indices.
        indices = _device_indices_to_host(state_indices[0])
        snapshot.materialize()
        snapshot.set_state_indices(indices)
        load["hybrid_state_configured"] = True

    def _start_h2d_chunk(self, load: dict[str, Any]) -> bool:
        """Launch one asynchronously prefetched H2D chunk."""

        record = load["record"]
        device_indices = load["device_indices"]
        AgenticPHostStagingManager._configure_hybrid_h2d_state(load)
        lane = AgenticPHostStagingManager._h2d_lane_resources(
            self, int(load.get("h2d_lane_id", 0))
        )
        if not load.get("io_inflight"):
            self.workset_broker.mark_io_inflight(
                load["request_generation"].snapshot_id,
                load["workset_lease"],
                load["io_attempt"],
            )
            if not self.ledger.mark_d2p_recovery_phase_rank(
                load["request_generation"].snapshot_id,
                self.owner,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                claim_id=load["recovery_claim_id"],
                lease_id=load["workset_lease"].lease_id,
                phase="io_inflight",
            ):
                self.workset_broker.mark_io_quiesced(
                    load["request_generation"].snapshot_id,
                    load["workset_lease"],
                    load["io_attempt"],
                )
                raise RuntimeError(
                    "Slow recovery lost Host ownership before H2D"
                )
            load["io_inflight"] = True
        if load.get("prefetch_future") is None:
            AgenticPHostStagingManager._submit_h2d_prefetch(
                self, load, int(load.get("offset", 0))
            )
        future = load.get("prefetch_future")
        if future is None or not future.done():
            return False
        cpu_elapsed = float(future.result())
        start = int(load["prefetch_start"])
        end = int(load["prefetch_end"])
        device_indices_host = None
        snapshot = record["snapshot"]
        if (
            bool(getattr(snapshot, "cuda_host_registered", False))
            and _cuda_driver_batch_memcpy() is not None
        ):
            try:
                device_indices_host = _device_indices_to_host(
                    device_indices[start:end]
                )
            except BaseException:
                logger.exception(
                    "AgenticKV H2D index mirror failed snapshot=%s; "
                    "using scatter fallback",
                    load["request_generation"].snapshot_id,
                )
        bounce_index = int(load["prefetch_bounce_index"])
        bounces = tuple(lane.get("host_bounces", (lane["host_bounce"],)))
        host_bounce = bounces[bounce_index]
        legacy_h2d = bool(load.get("prefetch_legacy"))
        AgenticPHostStagingManager._clear_h2d_prefetch(load)
        load["host_copy_elapsed_seconds"] = float(
            load.get("host_copy_elapsed_seconds", 0.0)
        ) + cpu_elapsed
        launch_fence = H2DLaunchFence(
            event=torch.cuda.Event(enable_timing=True)
        )
        # Publish the fence object before submitting CUDA work.  Therefore an
        # exception anywhere in launch still leaves cleanup enough physical
        # state to prove completion or quarantine the complete snapshot.
        load["launch_fence"] = launch_fence
        load["event"] = launch_fence.event
        record["loading"] = "h2d"
        if legacy_h2d:
            event, copy_refs = record["snapshot"].start_load_range_to_device(
                device_indices[start:end],
                lane["stream"],
                source_start=start,
                staging=lane["staging"],
                host_bounce=host_bounce,
                launch_fence=launch_fence,
            )
        else:
            event, copy_refs = (
                record["snapshot"].start_load_range_from_bounce_to_device(
                    device_indices[start:end],
                    lane["stream"],
                    staging=lane["staging"],
                    host_bounce=host_bounce,
                    source_start=start,
                    launch_fence=launch_fence,
                    device_indices_host=device_indices_host,
                )
            )
        load["event"] = event
        load["copy_refs"] = copy_refs
        load["chunk_end"] = end
        load["dma_bounce_index"] = bounce_index
        # CPU-prefetch the next disjoint chunk while this DMA/scatter runs.
        AgenticPHostStagingManager._submit_h2d_prefetch(self, load, end)
        if start == 0:
            logger.info(
                "AgenticKV shared_host_h2d_start snapshot=%s tokens=%d "
                "bytes=%d lane=%d tp_prepared=true async_progress=true",
                load["request_generation"].snapshot_id,
                int(record["offer"]["token_count"]),
                int(record["offer"]["byte_size"]),
                int(load.get("h2d_lane_id", 0)),
            )
        return True

    def _release_completed_h2d_host(self, load: dict[str, Any]) -> bool:
        """Release Host only after every TP shard is Radix-bound."""

        request_generation = load["request_generation"]
        snapshot_id = request_generation.snapshot_id
        entry = self.ledger.get(snapshot_id)
        if entry is None or entry.get("state") != HostStageState.CONSUMED.value:
            return False
        record = load["record"]
        with self._get_state_lock():
            if load.get("host_released"):
                return True
        if not self._release_record(record):
            # Keep both load and host_ready records as retry carriers for a
            # transient cudaHostUnregister/arena-release failure.
            return False
        with self._get_state_lock():
            if load.get("host_released"):
                return True
            load["host_released"] = True
            if self.host_ready.get(snapshot_id) is record:
                self.host_ready.pop(snapshot_id, None)
        logger.info(
            "AgenticKV shared_host_h2d_release snapshot=%s tp_rank=%d "
            "reason=all_tp_shards_radix_bound",
            snapshot_id,
            self.tp_rank,
        )
        return True

    def _discard_failed_h2d_load(self, rid: str, load: dict[str, Any]) -> bool:
        """Quiesce one failed shard and retain Host for a group retry."""
        prefetch = load.get("prefetch_future")
        if prefetch is not None:
            prefetch.cancel()
            if not prefetch.done():
                return False
            try:
                prefetch.result()
            except BaseException:
                pass
            AgenticPHostStagingManager._clear_h2d_prefetch(load)
        launch_fence = load.get("launch_fence")
        if launch_fence is not None and launch_fence.submitted:
            if launch_fence.unavailable or not launch_fence.armed:
                load["dma_quarantined"] = True
                self._h2d_poisoned = True
                logger.error(
                    "AgenticKV Slow H2D has no completion fence snapshot=%s; "
                    "quarantining Host and HBM ownership",
                    load["request_generation"].snapshot_id,
                )
                return False
            event = launch_fence.event
        else:
            event = load.get("event") if launch_fence is None else None
        if event is not None:
            try:
                if not event.query():
                    return False
                event.synchronize()
            except Exception as exc:
                # A CUDA API exception is not a DMA completion fence.  Keep
                # both the Host source and HBM destination owned until engine
                # teardown; otherwise a late H2D could corrupt reused pages.
                load["dma_quarantined"] = True
                self._h2d_poisoned = True
                load.setdefault("io_error", exc)
                logger.error(
                    "AgenticKV Slow H2D fence unavailable snapshot=%s; "
                    "quarantining Host and HBM ownership",
                    load["request_generation"].snapshot_id,
                )
                return False
        drop_host_on_abort = bool(load.get("drop_host_on_abort"))
        if drop_host_on_abort:
            try:
                if not self.ledger.request_host_load_failure(
                    load["request_generation"].snapshot_id,
                    self.owner,
                    reason="request_aborted",
                ):
                    return False
            except Exception:
                logger.exception(
                    "AgenticKV shared_host_abort_publish_retry snapshot=%s",
                    load["request_generation"].snapshot_id,
                )
                return False
        else:
            try:
                if not self.ledger.request_d2p_retry(
                    load["request_generation"].snapshot_id,
                    self.owner,
                    reason=type(load.get("io_error")).__name__,
                ):
                    return False
            except Exception:
                logger.exception(
                    "AgenticKV shared_host_retry_publish_retry snapshot=%s",
                    load["request_generation"].snapshot_id,
                )
                return False
        workset_lease = load.get("workset_lease")
        io_attempt = load.get("io_attempt")
        if not load.get("io_quiesced"):
            if load.get("io_inflight"):
                if not self.workset_broker.mark_io_quiesced(
                    load["request_generation"].snapshot_id,
                    workset_lease,
                    io_attempt,
                ):
                    return False
            else:
                if not self.workset_broker.cancel_io_attempt(
                    load["request_generation"].snapshot_id,
                    workset_lease,
                    io_attempt,
                ):
                    return False
            load["io_quiesced"] = True
        if drop_host_on_abort:
            snapshot_id = load["request_generation"].snapshot_id
            workset_lease = load.get("workset_lease")
            if not load.get("device_released"):
                self.workset_broker.request_release(
                    snapshot_id, workset_lease
                )
                load["device_released"] = True
            entry = self.ledger.get(snapshot_id)
            claims = {} if entry is None else entry.get("recovery_claims", {})
            rank_claim = claims.get(str(int(self.tp_rank)))
            if rank_claim is not None:
                claim_id = str(
                    rank_claim.get(
                        "claim_id",
                        load.get("recovery_claim_id", ""),
                    )
                )
                if not self.ledger.cancel_d2p_recovery_rank(
                    snapshot_id,
                    self.owner,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    claim_id=claim_id,
                    lease_id=(
                        None
                        if workset_lease is None
                        else int(workset_lease.lease_id)
                    ),
                ):
                    return False
            record = load["record"]
            # A remote record is only this rank's mmap of another P's Host
            # extent, so it can be closed after this rank's DMA fence.  A
            # colocated record is the physical TP shard, however, and must
            # remain in host_ready until every rank has drained and the shared
            # ledger reaches FAILED.  Releasing it here would let rank 0 reuse
            # Host pages while a peer rank can still have H2D in flight.
            remote_host = bool(record.get("remote_host"))
            if remote_host and not load.get("host_released"):
                if not self._release_record(record):
                    return False
                load["host_released"] = True
                with self._get_state_lock():
                    if self.host_ready.get(snapshot_id) is record:
                        self.host_ready.pop(snapshot_id, None)
            else:
                with self._get_state_lock():
                    record["loading"] = False
                    self.host_ready[snapshot_id] = record
            try:
                if not self.ledger.mark_host_load_rank_drained(
                    snapshot_id,
                    self.owner,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                ):
                    return False
            except Exception:
                logger.exception(
                    "AgenticKV shared_host_abort_drain_retry snapshot=%s",
                    snapshot_id,
                )
                return False
            with self._get_state_lock():
                if self.loads.get(rid) is load:
                    self.loads.pop(rid, None)
                record["loading"] = False
            terminal_entry = self.ledger.get(snapshot_id)
            if (
                terminal_entry is not None
                and terminal_entry.get("state") == HostStageState.FAILED.value
            ):
                self._release_consumed_owned_host(
                    {snapshot_id: terminal_entry},
                    snapshot_ids=(snapshot_id,),
                )
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
            return True
        # No prefetch or composite DMA can still access the old destination.
        # Keep Host bytes for the group retry but discard the old lease's
        # recurrent-slot binding before publishing this record as ready again.
        with self._get_state_lock():
            if self.loads.get(rid) is not load:
                return True
            if getattr(workset_lease, "state_device_indices", ()):
                load["record"]["snapshot"].reset_state_indices_after_quiesce()
            self.loads.pop(rid, None)
            if not load.get("device_released"):
                self.workset_broker.request_release(
                    load["request_generation"].snapshot_id,
                    workset_lease,
                )
                load["device_released"] = True
            record = load["record"]
            record["loading"] = False
            self.host_ready[load["request_generation"].snapshot_id] = record
            AgenticPHostStagingManager._notify_scheduler(
                self,
                "host_ready",
                load["request_generation"].snapshot_id,
            )
        AgenticPHostStagingManager._release_h2d_lane(
            self, load["request_generation"].snapshot_id
        )
        self.ledger.complete_d2p_retry_rank(
            load["request_generation"].snapshot_id,
            self.owner,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        return True

    def _publish_d2p_hbm_ready(self, load: dict[str, Any]) -> bool:
        """Idempotently ACK a physically complete H2D shard to the ledger."""

        snapshot_id = load["request_generation"].snapshot_id
        try:
            acknowledged = self.ledger.complete_d2p_host_load_rank(
                snapshot_id,
                self.owner,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
        except Exception:
            logger.exception(
                "AgenticKV shared_host_hbm_ack_retry snapshot=%s tp_rank=%d",
                snapshot_id,
                self.tp_rank,
            )
            return False
        if not acknowledged:
            return False
        load["io_complete"] = True
        AgenticPHostStagingManager._notify_scheduler(
            self, "hbm_ready", snapshot_id
        )
        record = load["record"]
        h2d_elapsed_ms = float(load["gpu_elapsed_ms"])
        h2d_wall_seconds = max(
            0.0,
            time.monotonic() - float(load.get("wall_started_at", time.monotonic())),
        )
        logger.info(
            "AgenticKV shared_host_h2d_complete snapshot=%s tokens=%d "
            "elapsed_ms=%.3f host_copy_ms=%.3f wall_ms=%.3f "
            "gib_per_s=%.3f e2e_gib_per_s=%.3f async_progress=true",
            snapshot_id,
            int(record["offer"]["token_count"]),
            h2d_elapsed_ms,
            float(load.get("host_copy_elapsed_seconds", 0.0)) * 1000.0,
            h2d_wall_seconds * 1000.0,
            0.0
            if not math.isfinite(h2d_elapsed_ms)
            else int(record["offer"]["byte_size"])
            / max(h2d_elapsed_ms / 1000.0, 1e-9)
            / (1024**3),
            int(record["offer"]["byte_size"])
            / max(h2d_wall_seconds, 1e-9)
            / (1024**3),
        )
        return True

    def _progress_h2d_loads(self) -> None:
        """Progress Slow Host->P copies independently of model scheduling.

        The scheduler still owns normal-workspace allocation and Radix bind.
        This worker only chains already-authorized DMA chunks, publishes the
        rank completion, and releases Host after the complete TP group is in
        HBM.  Direct ingress has its own worker and stream, so neither queue
        waits for the other queue's transport progress.
        """

        if self._h2d_poisoned:
            return
        with self._get_state_lock():
            loads = list(self.loads.items())
        for rid, load in loads:
            if load.get("abort_requested"):
                self._discard_failed_h2d_load(rid, load)
                continue
            snapshot_id = load["request_generation"].snapshot_id
            entry = self.ledger.get(snapshot_id)
            if entry is not None and entry.get("state") == HostStageState.FAILED.value:
                load.setdefault(
                    "io_error", RuntimeError("another TP rank failed Slow H2D")
                )
                continue
            if (
                entry is not None
                and entry.get("state") == HostStageState.RETRY_PENDING.value
            ):
                load.setdefault(
                    "io_error", RuntimeError("TP peer requested Slow H2D retry")
                )
            if load.get("io_error") is not None:
                self._discard_failed_h2d_load(rid, load)
                continue
            if load.get("io_complete"):
                if self.tp_size == 1:
                    # TP=1 has no peer scheduler that still needs to observe a
                    # group COMMIT.  Release the completed Host load now so the
                    # physical H2D lane can accept the next Slow snapshot.  In
                    # TP mode the source must stay pinned until every rank has
                    # consumed the scheduler-owned handoff below.
                    self._release_completed_h2d_host(load)
                    continue
                # DMA completion (and even the ledger's all-rank bind ACK) is
                # not the ownership handoff boundary.  In TP mode the native
                # scheduler broadcast must first make COMMIT visible to every
                # rank, then each rank hands the same workset to its live Req.
                # Releasing Host here races that broadcast: the rank that
                # publishes the final binder ACK can remove its local record
                # before consuming COMMIT, while a peer completes normally.
                # Keep the complete Host source and load context until
                # gate_request() performs the scheduler-owned handoff.
                continue
            if load.get("h2d_copy_complete"):
                self._publish_d2p_hbm_ready(load)
                continue
            if not load.get("start_allowed"):
                continue
            if self.tp_size > 1:
                entry = self.ledger.get(load["request_generation"].snapshot_id)
                if (
                    entry is None
                    or entry.get("state") != HostStageState.H2D_LOADING.value
                ):
                    continue
            event = load.get("event")
            try:
                if event is None:
                    self._start_h2d_chunk(load)
                    continue
                if not event.query():
                    continue
                event.synchronize()
                record = load["record"]
                start_event = getattr(
                    record.get("snapshot"), "_last_h2d_start_event", None
                )
                elapsed_ms = (
                    float("nan")
                    if start_event is None
                    else start_event.elapsed_time(event)
                )
                if math.isfinite(elapsed_ms):
                    load["gpu_elapsed_ms"] += elapsed_ms
                load["offset"] = int(load["chunk_end"])
                load["event"] = None
                load["copy_refs"] = None
                load.pop("dma_bounce_index", None)
                load.pop("launch_fence", None)
                if load["offset"] < len(load["device_indices"]):
                    self._start_h2d_chunk(load)
                    continue

                request_generation = load["request_generation"]
                snapshot_id = request_generation.snapshot_id
                if not self.workset_broker.mark_io_quiesced(
                    snapshot_id,
                    load["workset_lease"],
                    load["io_attempt"],
                ):
                    raise RuntimeError(
                        f"Slow H2D lost I/O ownership for {snapshot_id}"
                    )
                load["io_quiesced"] = True
                load["h2d_copy_complete"] = True
                AgenticPHostStagingManager._release_quiesced_h2d_lane(self, load)
                self._publish_d2p_hbm_ready(load)
                # H2D completion is not an ownership boundary. Keep the Host
                # extent until the scheduler binds and pins this workset.
            except Exception as exc:
                load["io_error"] = exc
                self.ledger.request_d2p_retry(
                    load["request_generation"].snapshot_id,
                    self.owner,
                    reason=f"slow_h2d_failed:{type(exc).__name__}",
                )
                logger.exception(
                    "Agentic asynchronous Slow H2D failed for %s; retrying",
                    load["request_generation"].snapshot_id,
                )

    def _progress_prestart_aborts(self) -> None:
        """Retire pinned Slow recoveries that never started physical H2D.

        Abort, recovery claim, and workset allocation can race on three
        independent threads.  The shared ledger is frozen in ABORTING first;
        then the exact local workset owner is retired; only after both fences
        are visible may this rank release its Host shard and ACK the group.
        """

        with self._get_state_lock():
            pending = list(
                getattr(self, "_prestart_recovery_aborts", {}).items()
            )
        for snapshot_id, context in pending:
            try:
                entry = self.ledger.get(snapshot_id)
                state = None if entry is None else entry.get("state")
                if state in {
                    HostStageState.EVICTING.value,
                    HostStageState.RECOMPUTE_REQUIRED.value,
                }:
                    # Eviction won the HOST_READY ownership CAS.  Restore the
                    # local record's eligibility and let the eviction state
                    # machine release/ACK this shard; abort must not steal it.
                    with self._get_state_lock():
                        record = self.host_ready.get(snapshot_id)
                        if record is not None:
                            record["loading"] = False
                            record.pop("abort_requested", None)
                        self._prestart_recovery_aborts.pop(snapshot_id, None)
                    AgenticPHostStagingManager._release_h2d_lane(
                        self, snapshot_id
                    )
                    continue
                if state == HostStageState.FAILED.value:
                    with self._get_state_lock():
                        record = self.host_ready.pop(snapshot_id, None)
                    if record is not None and not self._release_record(record):
                        with self._get_state_lock():
                            self.host_ready.setdefault(snapshot_id, record)
                        continue
                    with self._get_state_lock():
                        self._prestart_recovery_aborts.pop(snapshot_id, None)
                    AgenticPHostStagingManager._release_h2d_lane(
                        self, snapshot_id
                    )
                    continue
                if state != HostStageState.ABORTING.value:
                    if not self.ledger.request_host_load_failure(
                        snapshot_id,
                        self.owner,
                        reason="request_aborted_before_h2d",
                    ):
                        continue
                    entry = self.ledger.get(snapshot_id)
                    if (
                        entry is None
                        or entry.get("state")
                        != HostStageState.ABORTING.value
                    ):
                        continue
            except Exception:
                logger.exception(
                    "AgenticKV prestart abort ownership retry snapshot=%s",
                    snapshot_id,
                )
                continue

            rid = str(context["rid"])
            claim_id = str(
                context.get("recovery_claim_id")
                or self.workset_broker.slow_owner(snapshot_id, rid)
            )
            lease = self.workset_broker.get(snapshot_id, owner=claim_id)
            if lease is None:
                self.workset_broker.cancel_unstarted(
                    snapshot_id, owner=claim_id
                )
            else:
                self.workset_broker.request_release(
                    snapshot_id, lease, owner=claim_id
                )
                if self.workset_broker.eviction_blocker(snapshot_id) is not None:
                    continue

            entry = self.ledger.get(snapshot_id)
            claims = {} if entry is None else entry.get("recovery_claims", {})
            rank_claim = claims.get(str(int(self.tp_rank)))
            if rank_claim is not None:
                if not self.ledger.cancel_d2p_recovery_rank(
                    snapshot_id,
                    self.owner,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    claim_id=str(rank_claim.get("claim_id", claim_id)),
                    lease_id=(
                        None if lease is None else int(lease.lease_id)
                    ),
                ):
                    continue

            with self._get_state_lock():
                current = getattr(self, "_prestart_recovery_aborts", {}).get(
                    snapshot_id
                )
                if current is not context:
                    continue
                record = self.host_ready.get(snapshot_id)
            # As in the in-flight abort path, only a rank-local remote mapping
            # may be discarded before the group terminal ACK.  The physical
            # Host-owner shard stays discoverable for the terminal-state
            # release scan.
            if record is not None and record.get("remote_host"):
                if not self._release_record(record):
                    continue
                with self._get_state_lock():
                    if self.host_ready.get(snapshot_id) is record:
                        self.host_ready.pop(snapshot_id, None)
            elif record is not None:
                with self._get_state_lock():
                    record["loading"] = False
            if not self.ledger.mark_host_load_rank_drained(
                snapshot_id,
                self.owner,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            ):
                continue
            with self._get_state_lock():
                current = getattr(self, "_prestart_recovery_aborts", {}).get(
                    snapshot_id
                )
                if current is context:
                    self._prestart_recovery_aborts.pop(snapshot_id, None)
            terminal_entry = self.ledger.get(snapshot_id)
            if (
                terminal_entry is not None
                and terminal_entry.get("state") == HostStageState.FAILED.value
            ):
                self._release_consumed_owned_host(
                    {snapshot_id: terminal_entry},
                    snapshot_ids=(snapshot_id,),
                )
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)

    def _progress_host_abort_requests(self) -> None:
        """Retry abort ownership CAS without publishing local tombstones."""

        with self._get_state_lock():
            pending = tuple(
                getattr(self, "_pending_host_abort_requests", {}).values()
            )
        for context in pending:
            self.abort_request(
                str(context["rid"]), context["request_generation"]
            )

    def abort_request(self, rid: str, request_generation) -> None:
        """Cancel one Slow restore without racing an in-flight H2D."""

        snapshot_id = request_generation.snapshot_id
        # A CPU preparation may still hold the mapping or be publishing a
        # claim. Let its sole worker finish before the existing abort/fence
        # path can close Host or release a granted workset. Never wait here.
        with self._get_state_lock():
            prepare = getattr(self, "_host_prepares", {}).get(str(rid))
            if prepare is None:
                prepare = next(
                    (
                        item
                        for item in getattr(self, "_host_prepares", {}).values()
                        if item["parent"].snapshot_id == snapshot_id
                    ),
                    None,
                )
            if prepare is not None:
                prepare["cancelled"] = True
                self._control_wakeup.set()
                return
            getattr(self, "_host_prepare_results", {}).pop(str(rid), None)
        prestart_abort = False
        with self._get_state_lock():
            pending = getattr(self, "_pending_host_abort_requests", None)
            if pending is None:
                pending = self._pending_host_abort_requests = {}
            pending[snapshot_id] = {
                "rid": str(rid),
                "request_generation": request_generation,
            }
        # The ledger CAS is the ownership boundary.  A storage P must not
        # mutate its local record before learning that a foreign recovery P
        # has already claimed this request-generation.
        try:
            authorized = self.ledger.request_host_load_failure(
                snapshot_id,
                self.owner,
                reason="request_aborted",
                recovery_domain=getattr(self, "arena_domain", None),
            )
        except Exception:
            logger.exception(
                "AgenticKV Host abort CAS failed closed snapshot=%s", snapshot_id
            )
            self._control_wakeup.set()
            return
        if not authorized:
            with self._get_state_lock():
                getattr(self, "_pending_host_abort_requests", {}).pop(
                    snapshot_id, None
                )
            logger.info(
                "AgenticKV stale Host abort ignored snapshot=%s owner=%s",
                snapshot_id,
                self.owner,
            )
            return

        no_local_state = False
        try:
            with self._get_state_lock():
                _load_rid, load = self._find_h2d_load(snapshot_id, rid=rid)
                if load is not None:
                    load["abort_requested"] = True
                    load.setdefault("io_error", RuntimeError("request aborted"))
                    load["drop_host_on_abort"] = True
                else:
                    record = self.host_ready.get(snapshot_id)
                    if record is not None:
                        entry = self.ledger.get(snapshot_id)
                        claims = {} if entry is None else entry.get(
                            "recovery_claims", {}
                        )
                        rank_claim = claims.get(str(int(self.tp_rank)))
                        record_claim_id = record.get("recovery_claim_id")
                        claim_id = (
                            None
                            if rank_claim is None
                            else rank_claim.get("claim_id")
                        ) or record_claim_id or self.workset_broker.slow_owner(
                            snapshot_id, rid
                        )
                        record["abort_requested"] = True
                        record["loading"] = "abort_pending"
                        prestart = getattr(
                            self, "_prestart_recovery_aborts", None
                        )
                        if prestart is None:
                            prestart = self._prestart_recovery_aborts = {}
                        prestart[snapshot_id] = {
                            "rid": str(rid),
                            "request_generation": request_generation,
                            "recovery_claim_id": str(claim_id),
                        }
                        prestart_abort = True
                    else:
                        self.workset_broker.cancel_unstarted(
                            snapshot_id,
                            owner=self.workset_broker.slow_owner(
                                snapshot_id, rid
                            ),
                        )
                        AgenticPHostStagingManager._release_h2d_lane(
                            self, snapshot_id
                        )
                        no_local_state = True
            if prestart_abort:
                self._progress_prestart_aborts()
            elif no_local_state and not self.ledger.mark_host_load_rank_drained(
                snapshot_id,
                self.owner,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            ):
                raise RuntimeError("Host abort drain ownership was not committed")
        except Exception:
            logger.exception(
                "AgenticKV Host abort local completion retry snapshot=%s",
                snapshot_id,
            )
            self._control_wakeup.set()
            return
        with self._get_state_lock():
            getattr(self, "_pending_host_abort_requests", {}).pop(
                snapshot_id, None
            )
        self._control_wakeup.set()

    def rollback_bound_parent(self, req, request_generation) -> None:
        """Undo a locally bound TP parent after a peer restore failure."""

        pin = getattr(req, "_agentic_kv_host_pin_node", None)
        if pin is not None:
            self.tree_cache.dec_lock_ref(pin)
            delattr(req, "_agentic_kv_host_pin_node")
        committed_len = int(getattr(req, "_agentic_host_rank_token_count", 0))
        release = getattr(self.tree_cache, "release_agentic_request_cache", None)
        if committed_len and release is not None:
            release(
                req,
                committed_len=committed_len,
                _defer_if_blocked=False,
            )
        workset_lease = getattr(req, "_agentic_host_workset_lease", None)
        if workset_lease is not None:
            self.workset_broker.abort_bind(
                request_generation.snapshot_id,
                workset_lease,
                parent_bound=True,
            )
            delattr(req, "_agentic_host_workset_lease")
        else:
            self.workset_broker.cancel_unstarted(
                request_generation.snapshot_id,
                owner=self.workset_broker.slow_owner(
                    request_generation.snapshot_id, req.rid
                ),
            )
        AgenticPHostStagingManager._release_h2d_lane(
            self, request_generation.snapshot_id
        )
        for name in (
            "_agentic_host_rank_loaded",
            "_agentic_host_rank_token_count",
        ):
            if hasattr(req, name):
                delattr(req, name)

    def _queue_host_prepare(self, req, request_generation, *, allow_prepare=True) -> Optional[bool]:
        """Register an already FIFO-selected restore without filesystem I/O."""
        sid = request_generation.snapshot_id
        entry = getattr(self, "_ledger_entries_cache", {}).get(sid)
        if entry is None:
            with self._get_state_lock():
                return True if sid in self.active or sid in self.host_ready or sid in self.aborting else None
        if entry.get("state") not in {
            HostStageState.HOST_READY.value, HostStageState.H2D_LOADING.value,
            HostStageState.FAILED.value, HostStageState.RECOMPUTE_REQUIRED.value,
        }:
            return True if entry.get("state") in {
                HostStageState.HOST_RESERVED.value, HostStageState.HOST_WRITING.value,
                HostStageState.ABORTING.value, HostStageState.HBM_READY.value,
                HostStageState.EVICTING.value, HostStageState.SPILLING.value,
                HostStageState.RETRY_PENDING.value,
            } else None
        if not allow_prepare:
            return True
        if entry.get("state") in {
            HostStageState.HOST_READY.value, HostStageState.H2D_LOADING.value,
        } and not self._recovery_targets_this_manager(entry):
            return True
        if self._retire_stale_direct_before_slow(sid):
            return True
        with self._get_state_lock():
            if req.rid in self._host_prepares or req.rid in self.loads:
                return True
            # A retry's new rid must not replace an old CPU preparation.
            if any(x["parent"].snapshot_id == sid for x in self._host_prepares.values()):
                return True
            if self._reserve_h2d_lane(sid) is None:
                return True
            self._host_prepares[req.rid] = {
                "view": SimpleNamespace(
                    rid=str(req.rid), origin_input_ids=tuple(req.origin_input_ids),
                ),
                "parent": request_generation,
                "cancelled": False,
                "queued_at": time.monotonic(),
            }
        self._control_wakeup.set()
        return True

    def _progress_host_prepares(self) -> None:
        """Single-writer CPU preparation; allocator and Radix stay scheduler-owned."""
        if not getattr(self, "h2d_async_prepare", False):
            return
        with self._get_state_lock():
            prepares = tuple(self._host_prepares.items())
        for rid, prepare in prepares:
            with self._get_state_lock():
                if self._host_prepares.get(rid) is not prepare:
                    continue
                cancelled = prepare["cancelled"]
            if cancelled:
                # Runs on the same thread as preparation: all CPU accesses to
                # the mapping have quiesced before the old abort path runs.
                with self._get_state_lock():
                    self._host_prepares.pop(rid, None)
                self.abort_request(rid, prepare["parent"])
                # A foreign-owner CAS refusal can legitimately have no local
                # abort work. Return only our metadata admission credit when
                # no load, pending fence/abort or physical lease exists.
                sid = prepare["parent"].snapshot_id
                with self._get_state_lock():
                    if (
                        self._find_h2d_load(sid)[1] is None
                        and not any(
                            item["parent"].snapshot_id == sid
                            for item in self._host_prepares.values()
                        )
                        and sid not in getattr(self, "_pending_host_abort_requests", {})
                        and sid not in getattr(self, "_prestart_recovery_aborts", {})
                        and not self.workset_broker.owner_has_unretired_work(
                            sid, owner=self.workset_broker.slow_owner(sid, rid)
                        )
                    ):
                        self._release_h2d_lane(sid)
                continue
            result = True
            try:
                with self._get_state_lock():
                    load = self.loads.get(rid)
                if load is None:
                    result = self._prepare_host_restore(
                        prepare["view"], prepare["parent"],
                    )
                    with self._get_state_lock():
                        load = self.loads.get(rid)
                elif load.get("ledger_prepare_pending"):
                    if self._prepare_h2d_load_ledger(load):
                        load["ledger_prepare_pending"] = False
                        load["start_allowed"] = True
                    else:
                        self._cancel_unstarted_h2d_load(rid, load)
                        load = None
                with self._get_state_lock():
                    if prepare["cancelled"]:
                        # Do not publish completion or recycle this descriptor
                        # until its pending abort is handled next cycle.
                        if load is not None:
                            load["start_allowed"] = False
                        self._control_wakeup.set()
                        continue
                    started = load is not None and not load.get("ledger_prepare_pending")
                    if started or result is not True:
                        self._host_prepares.pop(rid, None)
                        if not started:
                            self._host_prepare_results[rid] = (
                                result, getattr(prepare["view"], "_agentic_kv_fallback", None),
                            )
                            self._release_h2d_lane(prepare["parent"].snapshot_id)
                        else:
                            load["prepare_queued_at"] = prepare["queued_at"]
                        self._notify_scheduler("host_prepared", prepare["parent"].snapshot_id)
                # A CPU prepare cannot monopolize progress of other copies.
                self._progress_h2d_loads()
            except Exception:
                # Preserve exact descriptor/claim/lease for retry or abort;
                # never reinterpret an I/O exception as permission to recompute.
                logger.exception("AgenticKV host_prepare_retry snapshot=%s rid=%s",
                                 prepare["parent"].snapshot_id, rid)
            finally:
                # Failed claim/attach may have relinquished its admission
                # resource. Do not retain an unbounded second waiting queue:
                # native FIFO admission must select that request again.
                with self._get_state_lock():
                    sid = prepare["parent"].snapshot_id
                    if (
                        self._host_prepares.get(rid) is prepare
                        and not prepare["cancelled"]
                        and rid not in self.loads
                        and sid not in self._h2d_resident_reservations
                    ):
                        self._host_prepares.pop(rid, None)
                        self._notify_scheduler("host_prepare_retry", sid)

    def gate_request(
        self,
        req,
        request_generation,
        *,
        allow_prepare: bool = True,
        allow_start: bool = True,
        allow_bind: bool = True,
    ) -> Optional[bool]:
        """Return True to defer, False when loaded, None when not owned.

        ``allow_prepare`` permits validating the local Host shard and reserving
        its final HBM pages. ``allow_start`` permits launching H2D, and
        ``allow_bind`` permits attaching a completed shard to Radix. TP>1 uses
        three distinct group phases: every rank prepares, every rank completes
        H2D, then every rank binds before rank0 broadcasts admission. TP=1
        keeps the historical prepare-and-start path.
        """

        snapshot_id = request_generation.snapshot_id
        if getattr(self, "h2d_async_prepare", False):
            with self._get_state_lock():
                if req.rid in self._host_prepares:
                    return True
                prepared_result = self._host_prepare_results.pop(req.rid, None)
            if prepared_result is not None:
                result, fallback = prepared_result
                if result is False:
                    req._agentic_kv_gate_complete = True
                    req._agentic_kv_fallback = fallback
                return result
            # A metadata-only waiter must not read the filesystem on the
            # scheduler thread. Known Host work goes to the preparation worker;
            # None preserves Direct/non-Host discovery in the caller.
            if (req.rid not in self.loads
                    and not getattr(req, "_agentic_host_retry_reason", None)
                    and not getattr(req, "_agentic_host_rank_loaded", False)):
                return self._queue_host_prepare(
                    req, request_generation,
                    allow_prepare=allow_prepare and allow_start,
                )
        pending_retry_reason = getattr(req, "_agentic_host_retry_reason", None)
        if pending_retry_reason is not None:
            try:
                retry_started = self.ledger.request_d2p_retry(
                    snapshot_id,
                    self.owner,
                    reason=pending_retry_reason,
                )
            except Exception:
                logger.exception(
                    "AgenticKV shared_host_retry_publish_retry snapshot=%s",
                    snapshot_id,
                )
                return True
            if not retry_started:
                return True
        ledger = getattr(self, "ledger", None)
        retry_entry = None if ledger is None else ledger.get(snapshot_id)
        if (
            retry_entry is not None
            and retry_entry.get("state") == HostStageState.RETRY_PENDING.value
        ):
            with self._get_state_lock():
                retry_load = self.loads.get(req.rid)
            if retry_load is not None:
                retry_load.setdefault(
                    "io_error", RuntimeError("TP group requested Slow retry")
                )
                self._control_wakeup.set()
                return True
            if getattr(req, "_agentic_host_rank_loaded", False):
                self.rollback_bound_parent(req, request_generation)
            self.ledger.complete_d2p_retry_rank(
                snapshot_id,
                self.owner,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
            if hasattr(req, "_agentic_tp_host_failed"):
                delattr(req, "_agentic_tp_host_failed")
            if hasattr(req, "_agentic_host_retry_reason"):
                delattr(req, "_agentic_host_retry_reason")
            return True
        if getattr(req, "_agentic_host_rank_loaded", False):
            tp_size = int(getattr(self, "tp_size", 1))
            commit_matches = (
                tp_size > 1
                and (
                    snapshot_id
                    in getattr(self, "tp_host_commit_snapshots", set())
                    or getattr(self, "tp_host_commit_snapshot", None)
                    == snapshot_id
                )
            )
            if tp_size > 1 and not commit_matches:
                # Rank-local H2D completion is not a scheduler admission
                # boundary.  TP0 publishes host_commit through the native
                # recv_requests broadcast after every shard reports loaded;
                # only that command may make the request runnable.
                return True
            group_entry = self.ledger.get(snapshot_id)
            if (
                not commit_matches
                and (
                    group_entry is None
                    or group_entry.get("state") != HostStageState.CONSUMED.value
                )
            ):
                return True
            workset_lease = getattr(req, "_agentic_host_workset_lease", None)
            if workset_lease is None:
                raise RuntimeError(
                    f"TP Host workset lease disappeared for {snapshot_id}"
                )
            try:
                self.workset_broker.handoff_to_req(
                    snapshot_id, req, workset_lease
                )
                if not self.ledger.mark_d2p_recovery_phase_rank(
                    snapshot_id,
                    self.owner,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    claim_id=workset_lease.owner,
                    lease_id=workset_lease.lease_id,
                    phase="handed",
                ):
                    raise RuntimeError(
                        f"Slow handoff lost lifecycle ownership for {snapshot_id}"
                    )
            except Exception:
                # Host remains authoritative and every cleanup marker remains
                # intact.  Retry this idempotent ownership commit before
                # releasing the lane or any Host extent.
                logger.exception(
                    "AgenticKV shared_host_handoff_retry snapshot=%s tp_rank=%d",
                    snapshot_id,
                    int(getattr(self, "tp_rank", 0)),
                )
                return True
            tp_rank = int(getattr(self, "tp_rank", 0))
            if tp_size > 1:
                # Keep every physical Host shard until the entire TP group has
                # loaded successfully and TP0 broadcasts COMMIT.  Releasing a
                # fast rank's extent at local H2D completion makes a partially
                # loaded group impossible to retry or diagnose safely.
                host_ready = getattr(self, "host_ready", None)
                record = None
                if host_ready is not None:
                    with AgenticPHostStagingManager._get_state_lock(self):
                        record = host_ready.get(snapshot_id)
                if record is not None:
                    self._release_record(record)
                    with AgenticPHostStagingManager._get_state_lock(self):
                        if host_ready.get(snapshot_id) is record:
                            host_ready.pop(snapshot_id, None)
                    logger.info(
                        "AgenticKV shared_host_group_commit_release "
                        "snapshot=%s tp_rank=%d",
                        snapshot_id,
                        int(getattr(self, "tp_rank", 0)),
                    )
            delattr(req, "_agentic_host_rank_loaded")
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
            delattr(req, "_agentic_host_workset_lease")
            req._agentic_kv_gate_complete = True
            req._agentic_kv_host_hit_tokens = int(
                getattr(req, "_agentic_host_rank_token_count", 0)
            )
            if tp_size > 1:
                req._agentic_tp_bootstrap_snapshot_id = snapshot_id
            if hasattr(req, "_agentic_host_rank_token_count"):
                delattr(req, "_agentic_host_rank_token_count")
            return False
        with self._get_state_lock():
            load = self.loads.get(req.rid)
        if load is not None:
            if load.get("ledger_prepare_pending"):
                try:
                    prepared = AgenticPHostStagingManager._prepare_h2d_load_ledger(
                        self, load
                    )
                except Exception:
                    logger.exception(
                        "AgenticKV shared_host_prepare_retry snapshot=%s",
                        snapshot_id,
                    )
                    return True
                if not prepared:
                    AgenticPHostStagingManager._cancel_unstarted_h2d_load(
                        self, req.rid, load
                    )
                    return True
                load["ledger_prepare_pending"] = False
                load["start_allowed"] = bool(allow_start)
                self._control_wakeup.set()
            if load.get("io_error") is not None:
                if not self._discard_failed_h2d_load(req.rid, load):
                    return True
                return True
            if not load.get("io_complete"):
                if allow_start:
                    load["start_allowed"] = True
                    wakeup = getattr(self, "_control_wakeup", None)
                    if wakeup is not None:
                        wakeup.set()
                return True
            tp_rank = int(getattr(self, "tp_rank", 0))
            tp_size = int(getattr(self, "tp_size", 1))
            if tp_size > 1 and not allow_bind:
                # Local DMA completion is deliberately not enough to mutate
                # Radix or enter a model forward. Rank0 first observes that
                # every shard is complete and broadcasts BIND; a second
                # all-rank barrier then broadcasts COMMIT_AND_PREFILL.
                return True
            record = load["record"]
            device_indices = load["device_indices"]
            keys = req.origin_input_ids[: int(record["offer"]["token_count"])]
            workset_lease = load["workset_lease"]
            if not load.get("radix_bound") and not self.workset_broker.begin_bind(
                snapshot_id, workset_lease
            ):
                logger.error(
                    "Slow workset lost ownership before Radix bind snapshot=%s "
                    "req=%s; retaining Host",
                    snapshot_id,
                    req.rid,
                )
                return True
            inserted = False
            try:
                from sglang.srt.mem_cache.base_prefix_cache import (
                    InsertParams,
                    MatchPrefixParams,
                )
                from sglang.srt.mem_cache.radix_cache import RadixKey

                radix_key = RadixKey(keys, req.extra_key)
                if not load.get("radix_bound"):
                    if workset_lease.state_device_indices:
                        from sglang.srt.disaggregation.agentic_hybrid_transfer import log_mamba_digest
                        log_mamba_digest(self.tree_cache.req_to_token_pool, workset_lease.state_device_indices,
                                         phase="d2p_host_received", snapshot_id=snapshot_id)
                    result = self.tree_cache.insert(
                        InsertParams(
                            key=radix_key,
                            value=device_indices,
                            mamba_value=(workset_lease.state_device_indices[0].clone()
                                         if workset_lease.state_device_indices else None),
                            priority=getattr(req, "priority", 0) or 0,
                        )
                    )
                    inserted = True
                    self.workset_broker.commit_parent_bound(
                        snapshot_id, workset_lease,
                        state_donated_to_radix=(bool(workset_lease.state_device_indices) and not result.mamba_exist),
                        state_duplicate=(bool(workset_lease.state_device_indices) and result.mamba_exist),
                    )
                    if workset_lease.runtime_state_device_indices:
                        self.workset_broker.attach_runtime_state_for_bind(snapshot_id, req, workset_lease)
                    if result.prefix_len and not self.tree_cache.supports_mamba():
                        self.token_allocator.free(
                            device_indices[: result.prefix_len]
                        )
                    matched = self.tree_cache.match_prefix(
                        MatchPrefixParams(key=radix_key, req=req)
                    )
                    if len(matched.device_indices) != len(keys):
                        raise RuntimeError(
                            "P Host H2D Radix insert is incomplete"
                        )
                    self.tree_cache.inc_lock_ref(matched.last_device_node)
                    req._agentic_kv_host_pin_node = matched.last_device_node
                    load["radix_bound"] = True
            except Exception:
                if inserted:
                    release = getattr(
                        self.tree_cache, "release_agentic_request_cache", None
                    )
                    if release is not None:
                        release(
                            req,
                            committed_len=len(keys),
                            _defer_if_blocked=False,
                        )
                self.workset_broker.abort_bind(
                    snapshot_id,
                    workset_lease,
                    parent_bound=inserted,
                )
                record["loading"] = False
                with self._get_state_lock():
                    self.loads.pop(req.rid, None)
                    self.host_ready[snapshot_id] = record
                AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
                req._agentic_host_retry_reason = "slow_radix_bind_failed"
                logger.exception(
                    "Failed to insert P Host snapshot for %s; Host retained",
                    req.rid,
                )
                return None
            try:
                bind_committed = self.ledger.complete_host_bind_rank(
                    snapshot_id,
                    self.owner,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                )
            except Exception:
                # Parent pages are already inserted and pinned, while the
                # complete Host extent remains owned.  Retry only this
                # idempotent metadata ACK; never unwind to recomputation.
                logger.exception(
                    "AgenticKV shared_host_bind_commit_retry snapshot=%s "
                    "tp_rank=%d/%d",
                    snapshot_id,
                    tp_rank,
                    tp_size,
                )
                return True
            if not bind_committed:
                logger.warning(
                    "AgenticKV shared_host_bind_commit_wait snapshot=%s "
                    "tp_rank=%d/%d",
                    snapshot_id,
                    tp_rank,
                    tp_size,
                )
                return True
            # Only a Radix-bound workset is a safe Host release boundary.
            if tp_size > 1:
                with self._get_state_lock():
                    self.loads.pop(req.rid, None)
                req._agentic_host_rank_loaded = True
                req._agentic_host_rank_token_count = int(
                    record["offer"]["token_count"]
                )
                req._agentic_host_workset_lease = load["workset_lease"]
                return True
            try:
                if not self._complete_shared_host_manifest(request_generation):
                    return True
            except Exception:
                logger.exception(
                    "AgenticKV shared_host_manifest_commit_retry snapshot=%s",
                    snapshot_id,
                )
                return True
            try:
                self.workset_broker.handoff_to_req(
                    snapshot_id, req, load["workset_lease"]
                )
                if not self.ledger.mark_d2p_recovery_phase_rank(
                    snapshot_id,
                    self.owner,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    claim_id=load["recovery_claim_id"],
                    lease_id=load["workset_lease"].lease_id,
                    phase="handed",
                ):
                    raise RuntimeError(
                        f"Slow handoff lost lifecycle ownership for {snapshot_id}"
                    )
            except Exception:
                # Do not destroy the only retryable recovery context before
                # the live request has atomically accepted the workset.
                logger.exception(
                    "AgenticKV shared_host_handoff_retry snapshot=%s tp_rank=0",
                    snapshot_id,
                )
                return True
            if not self._release_completed_h2d_host(load):
                return True
            if getattr(self, "h2d_decoupled", False):
                now = time.monotonic()
                logger.info(
                    "AgenticKV h2d_stage_timing snapshot=%s "
                    "selected_to_grant_ms=%.3f io_to_fence_ms=%.3f "
                    "fence_to_handoff_ms=%.3f ts=%.6f",
                    snapshot_id,
                    1000 * (load["wall_started_at"] - record.get("selected_at", load["wall_started_at"])),
                    1000 * (load.get("copy_done_at", now) - load["wall_started_at"]),
                    1000 * (now - load.get("copy_done_at", now)), time.time(),
                )
            with self._get_state_lock():
                if self.loads.get(req.rid) is load:
                    self.loads.pop(req.rid, None)
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
            req._agentic_kv_gate_complete = True
            req._agentic_kv_host_hit_tokens = int(record["offer"]["token_count"])
            return False

        if getattr(self, "h2d_async_prepare", False):
            return self._queue_host_prepare(
                req, request_generation,
                allow_prepare=allow_prepare and allow_start,
            )
        return self._prepare_host_restore(
            req, request_generation, allow_prepare=allow_prepare,
            allow_start=allow_start,
        )

    def _prepare_host_restore(
        self, req, request_generation, *, allow_prepare=True, allow_start=True,
    ) -> Optional[bool]:
        """Prepare only: never allocate, bind Radix, or consume an existing load.

        In async mode ``req`` is a worker-private view, not the scheduler Req.
        Legacy/default execution retains its original terminal flag behavior.
        """
        snapshot_id = request_generation.snapshot_id
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
        if ledger_entry is not None and ledger_entry.get("state") in {
            HostStageState.HOST_READY.value,
            HostStageState.H2D_LOADING.value,
            HostStageState.HBM_READY.value,
        }:
            # Host placement and recovery-P selection are independent. A
            # request that reached the former Host owner before Router
            # redirected it must remain metadata-only there.
            if not self._recovery_targets_this_manager(ledger_entry):
                return True
            if record is None and ledger_entry.get("state") in {
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
            }:
                record = self._import_remote_host_record(
                    snapshot_id, ledger_entry
                )
        if ledger_entry is not None and ledger_entry.get("state") in {
            HostStageState.FAILED.value,
            HostStageState.RECOMPUTE_REQUIRED.value,
        }:
            with self._get_state_lock():
                failed_record = self.host_ready.pop(snapshot_id, None)
            if failed_record is not None:
                self._release_record(failed_record)
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = (
                "shared_host_evicted"
                if ledger_entry.get("state")
                == HostStageState.RECOMPUTE_REQUIRED.value
                else "shared_host_h2d_failed"
            )
            return False
        if record is None:
            if ledger_entry is not None and ledger_entry.get("state") in {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
                HostStageState.ABORTING.value,
                HostStageState.HOST_READY.value,
                HostStageState.EVICTING.value,
                HostStageState.SPILLING.value,
                HostStageState.H2D_LOADING.value,
                HostStageState.HBM_READY.value,
                HostStageState.RETRY_PENDING.value,
            }:
                return True
            return None
        # DIRECT_READY -> Slow is a monotonic lifecycle boundary.  Retire a
        # racing Direct allocation on this selected recovery P before taking
        # the finite Slow admission resource.  In particular, do not pin a
        # Host claim or occupy an H2D lane while the scheduler is still
        # freeing the obsolete Direct workset.
        if AgenticPHostStagingManager._retire_stale_direct_before_slow(
            self, snapshot_id
        ):
            return True
        if getattr(self, "h2d_async_prepare", False) and getattr(
            record["snapshot"], "_materialized", record["snapshot"]
        ) is None:
            # The descriptor remains owned by this single preparation worker;
            # cancellation is deferred until it finishes. mmap/open must not
            # hold the manager lock needed by the Forward scheduler.
            record["snapshot"].materialize()
        with self._get_state_lock():
            # Re-read under the ownership lock: the background spill worker
            # may have claimed the record after the first discovery read.
            record = self.host_ready.get(snapshot_id)
            if record is None:
                return True
            loading = record.get("loading")
            if loading and loading != "h2d_reserving":
                return True
            if not allow_prepare:
                return True
            h2d_lane_id = AgenticPHostStagingManager._reserve_h2d_lane(
                self, snapshot_id
            )
            if h2d_lane_id is None:
                return True
            record.setdefault("selected_at", time.monotonic())
            if getattr(record["snapshot"], "_materialized", record["snapshot"]) is None:
                # Materialization now only mmaps the pageable tmpfs extent;
                # the process-lifetime bounce was pinned during manager init.
                started_at = time.monotonic()
                try:
                    record["snapshot"].materialize()
                except (OSError, MemoryError):
                    # mmap/open can fail transiently under virtual-memory or
                    # fd pressure. Host remains authoritative, so relinquish
                    # only the lane and retry this exact snapshot later; a
                    # scheduler-thread exception must not terminate P.
                    AgenticPHostStagingManager._release_h2d_lane(
                        self, snapshot_id
                    )
                    logger.exception(
                        "AgenticKV shared_host_materialize_retry snapshot=%s",
                        snapshot_id,
                    )
                    return True
                logger.info(
                    "AgenticKV shared_host_materialize_complete snapshot=%s "
                    "tokens=%d bytes=%d reason=selected_slow_recovery",
                    snapshot_id,
                    int(record["offer"]["token_count"]),
                    int(record["offer"]["byte_size"]),
                )
                logger.debug(
                    "Pageable shared Host mmap elapsed_ms=%.3f snapshot=%s",
                    (time.monotonic() - started_at) * 1000.0,
                    snapshot_id,
                )
                # mmap is metadata-only and uses the process-lifetime pinned
                # bounce buffer.  Continue directly to normal-page admission;
                # deferring this harmless step to another scheduler visit
                # creates multi-second gaps whenever the next Prefill batch is
                # long, even though HBM is already available.
            # First prevent another local scheduler visit from selecting the
            # same record.  The shared-ledger claim immediately below is the
            # authoritative eviction fence; no P workset may be requested
            # before that CAS succeeds.
            if not loading:
                record["loading"] = "h2d_claiming"
        # Each admitted snapshot owns one isolated H2D lane.  The lane is
        # reserved before this workset intent is created, so no more than
        # ``max_h2d_inflight`` Slow leases can consume P HBM.  Distinct streams,
        # pinned bounces, and GPU staging tensors let the control worker pipeline
        # several snapshots without cross-request buffer reuse.
        offer = record["offer"]
        parent_tokens = req.origin_input_ids[: int(offer["token_count"])]
        from sglang.srt.disaggregation.agentic_kv_lifecycle import token_ids_digest

        if len(parent_tokens) != int(offer["token_count"]) or token_ids_digest(
            parent_tokens
        ) != offer.get("token_digest"):
            try:
                failed = self.ledger.transition(
                    snapshot_id,
                    HostStageState.FAILED,
                    owner=self.owner,
                    reason="permanent_parent_digest_mismatch",
                )
            except Exception:
                failed = False
                logger.exception(
                    "AgenticKV permanent Host mismatch commit retry "
                    "snapshot=%s",
                    snapshot_id,
                )
            if not failed:
                with self._get_state_lock():
                    record["loading"] = False
                return True
            with self._get_state_lock():
                record["loading"] = False
                owned_record = self.host_ready.pop(snapshot_id, None)
            if owned_record is not None:
                self._release_record(owned_record)
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = "permanent_parent_digest_mismatch"
            logger.error(
                "AgenticKV evicted incompatible Host snapshot=%s req=%s; "
                "falling back to full Prefill",
                snapshot_id,
                req.rid,
            )
            return False
        workset_owner = self.workset_broker.slow_owner(snapshot_id, req.rid)
        recovery_claim_id = workset_owner
        try:
            recovery_claimed = self.ledger.claim_d2p_recovery_rank(
                snapshot_id,
                self.owner,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                claim_id=recovery_claim_id,
                recovery_domain=self.arena_domain,
            )
        except Exception:
            recovery_claimed = False
            logger.exception(
                "AgenticKV shared_host_recovery_claim_retry snapshot=%s",
                snapshot_id,
            )
        if not recovery_claimed:
            with self._get_state_lock():
                if self.host_ready.get(snapshot_id) is record:
                    record["loading"] = False
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
            return True
        record["recovery_claim_id"] = recovery_claim_id
        record["loading"] = "h2d_reserving"
        prompt_tokens = len(req.origin_input_ids)
        # Async preparation is single-writer and its cancellation is queued
        # back to that same worker. Scheduler gate cannot touch this rid until
        # the descriptor is retired. Do not hold the shared manager lock over
        # ledger I/O; retain the original lock scope on the legacy path.
        with (nullcontext() if getattr(self, "h2d_async_prepare", False)
              else self._get_state_lock()):
            # Lane ownership and workset intent creation are one atomic
            # admission operation.  An abort may otherwise release/reassign a
            # lane between the Host-record check and broker.request(), leaving
            # a physical Slow lease with no corresponding I/O lane.
            reservations = getattr(self, "_h2d_lane_reservations", {})
            if (
                self.host_ready.get(snapshot_id) is not record
                or reservations.get(snapshot_id) != h2d_lane_id
            ):
                if self.host_ready.get(snapshot_id) is record:
                    record["loading"] = False
                return True
            requested = self.workset_broker.request(
                snapshot_id,
                int(offer["token_count"]),
                prompt_tokens,
                owner=workset_owner,
            )
            if requested is False:
                # A retiring/conflicting owner is not a grant. Keep the exact
                # Host claim and retry; never start from another owner's pages.
                return True
            workset_lease = self.workset_broker.get(
                snapshot_id, owner=workset_owner
            )
            if workset_lease is None:
                # The Host claim remains pinned while the scheduler services
                # this bounded lane's allocation intent.  It owns no P pages
                # yet and cannot be pressure-evicted underneath the intent.
                record["loading"] = "h2d_reserving"
                return True
            if not self.ledger.attach_d2p_recovery_lease_rank(
                snapshot_id,
                self.owner,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                claim_id=recovery_claim_id,
                lease_id=workset_lease.lease_id,
            ):
                self.workset_broker.request_release(snapshot_id, workset_lease)
                self.ledger.cancel_d2p_recovery_rank(
                    snapshot_id,
                    self.owner,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    claim_id=recovery_claim_id,
                    lease_id=workset_lease.lease_id,
                )
                record["loading"] = False
                record.pop("recovery_claim_id", None)
                AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
                raise RuntimeError(
                    f"Slow workset lease lost Host claim for {snapshot_id}"
                )
            io_attempt = (
                f"slow-h2d:{self.owner}:{req.rid}:{workset_lease.lease_id}"
            )
            if not self.workset_broker.begin_io_attempt(
                snapshot_id, workset_lease, io_attempt
            ):
                record["loading"] = False
                return True
            device_indices = workset_lease.parent_indices[
                : int(offer["token_count"])
            ]
            record["loading"] = "h2d_prepared"
            self.loads[req.rid] = {
                "rid": req.rid,
                "record": record,
                "request_generation": request_generation,
                "device_indices": device_indices,
                "workset_lease": workset_lease,
                "recovery_claim_id": recovery_claim_id,
                "io_attempt": io_attempt,
                "io_inflight": False,
                "io_quiesced": False,
                "event": None,
                "copy_refs": None,
                "offset": 0,
                "chunk_end": 0,
                "gpu_elapsed_ms": 0.0,
                "host_copy_elapsed_seconds": 0.0,
                "wall_started_at": time.monotonic(),
                "start_allowed": False,
                "io_complete": False,
                "host_released": False,
                "h2d_lane_id": h2d_lane_id,
                "ledger_prepare_pending": True,
            }
        try:
            prepared = AgenticPHostStagingManager._prepare_h2d_load_ledger(
                self, self.loads[req.rid]
            )
        except Exception:
            # The ledger mutation is idempotent and may have committed before
            # an I/O error became visible.  Keep the exact lane+lease and retry
            # the same prepare on the next scheduler visit; never launch H2D
            # from an unacknowledged TP shard.
            logger.exception(
                "AgenticKV shared_host_prepare_retry snapshot=%s", snapshot_id
            )
            return True
        if not prepared:
            AgenticPHostStagingManager._cancel_unstarted_h2d_load(
                self, req.rid, self.loads[req.rid]
            )
            return True
        self.loads[req.rid]["ledger_prepare_pending"] = False
        self.loads[req.rid]["start_allowed"] = bool(allow_start)
        # TP ranks still wait for the shared ledger to enter H2D_LOADING; TP1
        # can start immediately.  Both are progressed by the independent worker.
        self._control_wakeup.set()
        return True

    def release_request_pin(self, req) -> None:
        node = getattr(req, "_agentic_kv_host_pin_node", None)
        if node is None:
            return
        self.tree_cache.dec_lock_ref(node)
        delattr(req, "_agentic_kv_host_pin_node")
