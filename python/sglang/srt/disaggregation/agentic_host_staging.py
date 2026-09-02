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

import ctypes
import errno
import fcntl
import hashlib
import json
import logging
import math
import mmap
import os
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional

import torch

from sglang.srt.disaggregation.agentic_early_claim import (
    AgenticDirectoryChangeWatcher,
)
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    MooncakeSnapshotStore,
    SharedSnapshotEvictionController,
    SnapshotManifest,
    SnapshotState,
    page_namespace,
)
from sglang.srt.disaggregation.utils import kv_to_page_indices
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo

logger = logging.getLogger(__name__)
_LEDGER_ENTRY_UNSET = object()
_LEDGER_OWNERSHIP_CHANGED = object()
_HOST_LIBC = ctypes.CDLL(None, use_errno=True)
_HOST_MEMMOVE = _HOST_LIBC.memmove
_HOST_MEMMOVE.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
_HOST_MEMMOVE.restype = ctypes.c_void_p
_HOST_MADVISE = _HOST_LIBC.madvise
_HOST_MADVISE.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
_HOST_MADVISE.restype = ctypes.c_int
_HOST_MEMSET = _HOST_LIBC.memset
_HOST_MEMSET.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
_HOST_MEMSET.restype = ctypes.c_void_p
_MADV_POPULATE_WRITE = 23


def supports_agentic_kv_spill(storage_backend) -> bool:
    """Return the authoritative Shared-Arena spill capability.

    Existing SGLang storage backends predate this opt-out attribute and are
    spill-capable.  The request-level node-local backend explicitly sets it to
    ``False``.  D metadata, P admission and scheduler construction must all
    use this same compatibility rule.
    """

    return bool(getattr(storage_backend, "supports_kv_spill", True))


def _copy_layer_first_host_range(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    destination_start: int,
    source_start: int,
    token_count: int,
) -> None:
    """Copy a token range between CPU layer-first KV buffers.

    A slice across ``[K/V, layer, token, head, dim]`` is strided at every
    layer boundary. PyTorch's generic CPU TensorIterator path is especially
    slow when one side is pinned memory (the Shared-Arena bounce-buffer case).
    Each individual ``[token, head, dim]`` layer range is contiguous, so copy
    those ranges with libc. ``CDLL`` releases the GIL while memmove runs,
    keeping transport progress independent of the model scheduler thread.
    """

    destination_start = int(destination_start)
    source_start = int(source_start)
    token_count = int(token_count)
    if token_count < 0:
        raise ValueError("Host KV copy token_count must be non-negative")
    if token_count == 0:
        return
    if destination.device.type != "cpu" or source.device.type != "cpu":
        raise ValueError("Host KV copy requires CPU tensors")
    if destination.dtype != source.dtype:
        raise ValueError("Host KV copy dtype mismatch")
    if destination.ndim != 5 or source.ndim != 5:
        raise ValueError("Host KV copy requires [K/V, layer, token, head, dim]")
    if (
        destination.shape[:2] != source.shape[:2]
        or destination.shape[3:] != source.shape[3:]
    ):
        raise ValueError("Host KV copy layout mismatch")
    if (
        destination_start < 0
        or destination_start + token_count > destination.shape[2]
    ):
        raise ValueError("Host KV destination range is out of bounds")
    if source_start < 0 or source_start + token_count > source.shape[2]:
        raise ValueError("Host KV source range is out of bounds")

    bytes_per_layer_range = (
        token_count
        * int(destination.shape[3])
        * int(destination.shape[4])
        * int(destination.element_size())
    )
    for kv_index in range(int(destination.shape[0])):
        for layer_index in range(int(destination.shape[1])):
            destination_view = destination[
                kv_index,
                layer_index,
                destination_start : destination_start + token_count,
            ]
            source_view = source[
                kv_index,
                layer_index,
                source_start : source_start + token_count,
            ]
            if (
                not destination_view.is_contiguous()
                or not source_view.is_contiguous()
            ):
                raise ValueError("Host KV per-layer token range must be contiguous")
            _HOST_MEMMOVE(
                destination_view.data_ptr(),
                source_view.data_ptr(),
                bytes_per_layer_range,
            )


@dataclass
class H2DLaunchFence:
    """Physical completion authority for one asynchronously launched H2D.

    ``submitted`` becomes true before the first CUDA operation. ``armed`` is
    true only after the completion event has been recorded behind every
    operation on the stream.  A submitted but unarmed launch is permanently
    quarantined because no allocator-safe completion proof exists.
    """

    event: Any
    submitted: bool = False
    armed: bool = False
    unavailable: bool = False
    copy_refs: list[Any] = field(default_factory=list)


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
    # released its local shard.  Only RECOMPUTE_REQUIRED means no physical
    # Host copy remains and the child may safely run a full Prefill.
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
        self, snapshot_id: str, owner: str, *, reason: str
    ) -> bool:
        """Begin a TP-wide P2D Host-load abort without releasing Host data.

        This only publishes failure intent.  The source Host extents remain
        authoritative until every destination rank separately confirms that
        no H2D DMA can still read them via :meth:`mark_host_load_rank_drained`.
        """

        def callback(entries):
            current = entries.get(snapshot_id)
            if current is None or current.get("p_owner") != owner:
                return False, False
            state = current.get("state")
            if state == HostStageState.FAILED.value:
                return True, False
            if state == HostStageState.ABORTING.value:
                return bool(current.get("h2d_abort_started", False)), False
            if state not in {
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
            }:
                return False, False
            current["state"] = HostStageState.ABORTING.value
            current["h2d_abort_started"] = True
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
                or current.get("p_owner") != owner
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
            if current is None or current.get("p_owner") != owner:
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
            if current is None or current.get("p_owner") != owner:
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
            if current is None or current.get("p_owner") != owner:
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
                or current.get("p_owner") != owner
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
                or current.get("p_owner") != owner
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

    def claim_d2p_recovery_rank(
        self,
        snapshot_id: str,
        owner: str,
        *,
        tp_rank: int,
        tp_size: int,
        claim_id: str,
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
                or current.get("p_owner") != owner
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.HOST_READY.value,
                    HostStageState.H2D_LOADING.value,
                }
            ):
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
                or current.get("p_owner") != owner
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
                or current.get("p_owner") != owner
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
                or current.get("p_owner") != owner
                or int(current.get("tp_size", 1)) != int(tp_size)
                or current.get("state")
                not in {
                    HostStageState.H2D_LOADING.value,
                    HostStageState.ABORTING.value,
                }
                or current.get("recovery_claim_id") != claim_id
                or claim is None
                or claim.get("claim_id") != claim_id
                or claim.get("phase") in {"io_inflight", "handed"}
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
                or current.get("p_owner") != owner
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
                return True, False
            if state != HostStageState.HOST_READY.value:
                return False, False
            if any(
                current.get(field)
                for field in (
                    "loading_ranks",
                    "h2d_prepared_ranks",
                    "loader_acks",
                    "binder_acks",
                )
            ):
                return False, False
            current["state"] = HostStageState.EVICTING.value
            current["eviction_acks"] = []
            current["eviction_reason"] = str(reason)[:256]
            current["eviction_started_at"] = time.time()
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
        """ACK one released Host shard and publish group recomputation.

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
            if current.get("state") == HostStageState.RECOMPUTE_REQUIRED.value:
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
                current["state"] = HostStageState.RECOMPUTE_REQUIRED.value
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
    mapping stays pageable.  D and P move bounded chunks through a reusable
    pinned bounce buffer instead of cudaHostRegister-ing this request-sized
    mapping.  It therefore contains exactly one physical Host copy without
    putting multi-GiB registration work on either model process' CUDA context.
    """

    def __init__(
        self,
        *,
        path: str,
        token_count: int,
        device_pool,
        byte_size: int,
        create: bool,
        file_offset: int = 0,
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
        self.file_offset = int(file_offset)
        if self.file_offset < 0 or self.file_offset % mmap.ALLOCATIONGRANULARITY:
            raise ValueError("shared Host extent offset must be mmap-aligned")
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                if self.file_offset:
                    raise ValueError("new standalone Host snapshots require offset zero")
                os.ftruncate(fd, self.byte_size)
            elif os.fstat(fd).st_size < self.file_offset + self.byte_size:
                raise ValueError("shared Host extent file is smaller than the snapshot")
            self.mapping = mmap.mmap(
                fd,
                self.byte_size,
                access=mmap.ACCESS_WRITE,
                offset=self.file_offset,
            )
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
        self._raw = raw
        self._closed = False

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def start_backup_from_device(
        self, source_indices, stream, *, staging=None, host_bounce=None
    ):
        """Launch D-HBM -> this Host extent on the D GPU's own stream."""

        if len(source_indices) != self.token_count:
            raise ValueError("source token count does not match shared Host extent")
        return self.start_backup_range_from_device(
            source_indices,
            destination_start=0,
            stream=stream,
            staging=staging,
            host_bounce=host_bounce,
        )

    def start_backup_range_from_device(
        self,
        source_indices,
        *,
        destination_start: int,
        stream,
        staging=None,
        host_bounce=None,
        launch_fence: Optional[H2DLaunchFence] = None,
    ):
        """Launch one D2H chunk into a fixed pinned bounce buffer.

        The caller commits the completed bounce into this pageable mapping on
        CPU before reusing it.  Keeping that commit outside the CUDA event
        makes the measured GPU time exactly gather + PCIe DMA.
        """

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
        if host_bounce is None:
            raise ValueError("pageable Host snapshots require a pinned D2H bounce")
        if len(source_indices) > host_bounce.token_capacity:
            raise ValueError("D2H chunk is larger than the pinned Host bounce")
        start_event = torch.cuda.Event(enable_timing=True)
        if launch_fence is None:
            launch_fence = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
        event = launch_fence.event
        copy_refs = [source_indices, original_source_indices, staging, host_bounce]
        launch_fence.copy_refs = copy_refs
        try:
            with torch.cuda.stream(stream):
                # The event is the release authority if CUDA accepts any part
                # of this D2H launch and a later submission raises.
                launch_fence.submitted = True
                if not source_indices.is_cuda or source_indices.dtype != torch.int64:
                    source_indices = source_indices.to(
                        device=self.device_pool.device,
                        dtype=torch.int64,
                        non_blocking=True,
                    )
                    copy_refs.append(source_indices)
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
                    for layer_id in range(self.layer_num):
                        host_bounce.k_buffer[layer_id, start : start + count].copy_(
                            staging.k_buffer[layer_id][:count], non_blocking=True
                        )
                        host_bounce.v_buffer[layer_id, start : start + count].copy_(
                            staging.v_buffer[layer_id][:count], non_blocking=True
                        )
                event.record(stream)
                launch_fence.armed = True
                source_indices.record_stream(stream)
                if bool(getattr(original_source_indices, "is_cuda", False)):
                    original_source_indices.record_stream(stream)
        except Exception:
            if launch_fence.submitted and not launch_fence.armed:
                try:
                    # Recording behind all work already accepted by this
                    # private stream gives the caller a physical quiescence
                    # proof even when the original launch only partly formed.
                    with torch.cuda.stream(stream):
                        event.record(stream)
                    launch_fence.armed = True
                except Exception:
                    launch_fence.unavailable = True
            raise
        self._last_d2h_start_event = start_event
        copy_refs.append(start_event)
        launch_fence.copy_refs = copy_refs
        return event, tuple(copy_refs)

    def commit_backup_range_from_bounce(
        self, host_bounce, *, destination_start: int, token_count: int
    ) -> None:
        """Copy a completed pinned D2H chunk into the tmpfs snapshot."""

        destination_start = int(destination_start)
        token_count = int(token_count)
        destination_end = destination_start + token_count
        if destination_start < 0 or destination_end > self.token_count:
            raise ValueError("D2H bounce commit falls outside shared Host extent")
        _copy_layer_first_host_range(
            self.kv_buffer,
            host_bounce.kv_buffer,
            destination_start=destination_start,
            source_start=0,
            token_count=token_count,
        )

    def start_load_range_to_device(
        self,
        device_indices,
        stream,
        *,
        source_start: int,
        staging=None,
        host_bounce=None,
        launch_fence: Optional[H2DLaunchFence] = None,
    ):
        """Launch one pageable Host -> pinned bounce -> P-HBM chunk.

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

        source_start = int(source_start)
        token_count = len(device_indices)
        if source_start < 0 or source_start + token_count > self.token_count:
            raise ValueError("H2D chunk falls outside shared Host extent")
        original_device_indices = device_indices
        if host_bounce is None:
            raise ValueError("pageable Host snapshots require a pinned H2D bounce")
        if token_count > host_bounce.token_capacity:
            raise ValueError("H2D chunk is larger than the pinned Host bounce")
        if staging is None:
            staging = LayerFirstD2HStaging(self.device_pool, token_count)
        if staging.token_capacity < token_count:
            raise ValueError("H2D staging buffer is smaller than the chunk")
        # CPU copy happens before CUDA submission and touches only a bounded,
        # already-pinned allocation.  No CUDA driver registration is needed.
        _copy_layer_first_host_range(
            host_bounce.kv_buffer,
            self.kv_buffer,
            destination_start=0,
            source_start=source_start,
            token_count=token_count,
        )
        start_event = torch.cuda.Event(enable_timing=True)
        if launch_fence is None:
            launch_fence = H2DLaunchFence(
                event=torch.cuda.Event(enable_timing=True)
            )
        event = launch_fence.event
        copy_refs = [device_indices, staging, host_bounce]
        launch_fence.copy_refs = copy_refs
        try:
            with torch.cuda.stream(stream):
                # From this point onward an exception may follow a partially
                # submitted copy.  The pre-created event is the only release
                # authority for both Host and destination HBM.
                launch_fence.submitted = True
                if not device_indices.is_cuda or device_indices.dtype != torch.int64:
                    device_indices = device_indices.to(
                        device=self.device_pool.device,
                        dtype=torch.int64,
                        non_blocking=True,
                    )
                    copy_refs.append(device_indices)
                start_event.record(stream)
                source_indices = staging.local_indices[:token_count]
                for layer_id in range(self.layer_num):
                    staging.k_buffer[layer_id][:token_count].copy_(
                        host_bounce.k_buffer[layer_id, :token_count], non_blocking=True
                    )
                    staging.v_buffer[layer_id][:token_count].copy_(
                        host_bounce.v_buffer[layer_id, :token_count], non_blocking=True
                    )
                transfer_kv_all_layer(
                    src_k_layers=staging.k_data_ptrs,
                    dst_k_layers=self.device_pool.k_data_ptrs,
                    src_v_layers=staging.v_data_ptrs,
                    dst_v_layers=self.device_pool.v_data_ptrs,
                    src_indices=source_indices,
                    dst_indices=device_indices,
                    item_size=self.item_size,
                    num_layers=self.layer_num,
                    block_quota=4,
                    num_warps_per_block=32,
                )
                source_indices.record_stream(stream)
                copy_refs.append(source_indices)
                event.record(stream)
                launch_fence.armed = True
                device_indices.record_stream(stream)
                if bool(getattr(original_device_indices, "is_cuda", False)):
                    original_device_indices.record_stream(stream)
        except BaseException:
            if launch_fence.submitted and not launch_fence.armed:
                try:
                    # Record behind every operation that was accepted before
                    # the exception.  If even this fails, no physical fence is
                    # available and the caller must quarantine ownership.
                    with torch.cuda.stream(stream):
                        event.record(stream)
                    launch_fence.armed = True
                except BaseException:
                    launch_fence.unavailable = True
            raise
        self._last_h2d_start_event = start_event
        copy_refs.extend((original_device_indices, start_event))
        launch_fence.copy_refs = copy_refs
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
        self.kv_buffer = None
        self._raw = None
        self.mapping.close()
        self._closed = True
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


class PinnedMHAHostBounce:
    """Process-lifetime pinned Host chunk reused by every slow snapshot."""

    def __init__(self, device_pool, token_capacity: int):
        self.token_capacity = int(token_capacity)
        if self.token_capacity <= 0:
            raise ValueError("pinned Host bounce capacity must be positive")
        self.kv_buffer = torch.empty(
            (
                2,
                int(device_pool.layer_num),
                self.token_capacity,
                int(device_pool.head_num),
                int(device_pool.head_dim),
            ),
            dtype=device_pool.store_dtype,
            device="cpu",
            pin_memory=True,
        )

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]


class LazySharedMHAHostSnapshot:
    """A granted tmpfs extent whose P-side pinned mapping is built later.

    Creating and CUDA-registering a multi-GiB mapping can take seconds under
    burst load.  D needs only an existing file of the correct size in order to
    open its own mapping and start D2H, so coupling P registration to grant
    publication unnecessarily pins the complete source KV on D.  This object
    creates the sparse tmpfs extent immediately and materializes P's mapping
    independently after D has begun writing it.
    """

    def __init__(
        self,
        *,
        path: str,
        token_count: int,
        device_pool,
        byte_size: int,
        allocation_bytes: Optional[int] = None,
        create: bool = True,
        file_offset: int = 0,
    ):
        if not path.startswith("/dev/shm/"):
            raise ValueError("shared Host snapshot must reside in /dev/shm")
        self.path = path
        self.token_count = int(token_count)
        self.device_pool = device_pool
        self.byte_size = int(byte_size)
        self.allocation_bytes = max(
            self.byte_size,
            self.byte_size if allocation_bytes is None else int(allocation_bytes),
        )
        self.file_offset = int(file_offset)
        if self.file_offset < 0 or self.file_offset % mmap.ALLOCATIONGRANULARITY:
            raise ValueError("shared Host extent offset must be mmap-aligned")
        self.requires_prefault = bool(create)
        self._materialized = None
        self._closed = False
        self._lock = threading.Lock()
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                if self.file_offset:
                    raise ValueError("new standalone Host snapshots require offset zero")
                os.ftruncate(fd, self.allocation_bytes)
            elif os.fstat(fd).st_size < self.file_offset + self.byte_size:
                raise ValueError("recycled Host extent is smaller than the snapshot")
        finally:
            os.close(fd)

    def prefault_for_write(self) -> None:
        """Populate tmpfs pages before publishing the D2H grant.

        A freshly truncated tmpfs file is sparse.  Letting the D transport
        thread take tens of thousands of 4-KiB write faults while committing
        each KV chunk reduces an otherwise memory-bandwidth copy to roughly
        1 GiB/s.  P owns extent creation, so it prepares the pages on a bounded
        background worker before D can see the grant.  Linux
        MADV_POPULATE_WRITE performs that work in-kernel; the portable memset
        fallback preserves the same lifetime and correctness contract.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot prefault a released Host extent")
            # Recycled extents retain their tmpfs pages.  Re-populating them
            # would reintroduce the extra full-memory write that the arena
            # pool exists to avoid.
            if not self.requires_prefault:
                return
            fd = os.open(self.path, os.O_RDWR)
        try:
            # ftruncate creates a sparse tmpfs file.  Reserve the complete
            # backing store before touching any page so capacity exhaustion is
            # reported as an ordinary admission failure instead of SIGBUS in
            # the background worker.
            os.posix_fallocate(fd, 0, self.byte_size)
            mapping = mmap.mmap(fd, self.byte_size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        anchor = ctypes.c_char.from_buffer(mapping)
        address = ctypes.addressof(anchor)
        try:
            ctypes.set_errno(0)
            if _HOST_MADVISE(address, self.byte_size, _MADV_POPULATE_WRITE) != 0:
                error_number = ctypes.get_errno()
                unsupported = {
                    errno.EINVAL,
                    errno.ENOSYS,
                    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                }
                if error_number not in unsupported:
                    raise OSError(
                        error_number,
                        os.strerror(error_number),
                        self.path,
                    )
                # The extent is already fully reserved, so this compatibility
                # fallback cannot fault into an overcommitted sparse file.
                _HOST_MEMSET(address, 0, self.byte_size)
            self.requires_prefault = False
        finally:
            del anchor
            mapping.close()

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
                    file_offset=self.file_offset,
                )
            return self

    def mark_populated(self) -> None:
        """Publish that every byte in the logical snapshot was written."""

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot populate a released Host extent")
            self.requires_prefault = False

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
    """One process-lifetime tmpfs arena with request-generation suballocation.

    The complete file is physically allocated and first-touched once at P
    startup, under the P process' NUMA memory policy. A snapshot subsequently
    owns only one aligned ``(offset, length)`` extent in that file. D and P map
    the same subrange, so request-level ownership is preserved without
    per-snapshot ``ftruncate``, ``fallocate`` or prefault workers.

    The arena intentionally remains pageable. Transfers use the existing
    bounded pinned bounce buffers; this avoids registering hundreds of GiB
    with every CUDA context while still removing sparse-tmpfs page faults from
    the serving hot path.
    """

    _ALIGNMENT = mmap.ALLOCATIONGRANULARITY

    def __init__(self, directory: str, capacity_bytes: int):
        if not directory.startswith("/dev/shm/"):
            raise ValueError("shared Host arena must reside in /dev/shm")
        self.directory = directory.rstrip("/")
        self.capacity_bytes = self._align_down(int(capacity_bytes))
        if self.capacity_bytes <= 0:
            raise ValueError("shared Host arena capacity must be positive")
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        self.path = os.path.join(
            self.directory, f"preallocated-arena-{os.getpid()}.kv"
        )
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        mapping = None
        try:
            os.ftruncate(fd, self.capacity_bytes)
            # Reserve tmpfs backing before publishing the arena. Capacity
            # failure is therefore a startup error, never a SIGBUS after D
            # has relinquished request ownership.
            os.posix_fallocate(fd, 0, self.capacity_bytes)
            mapping = mmap.mmap(
                fd, self.capacity_bytes, access=mmap.ACCESS_WRITE
            )
        except BaseException:
            if mapping is not None:
                mapping.close()
            os.close(fd)
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(fd)

        # First-touch while this P process is NUMA-bound. tmpfs retains the
        # backing pages after this temporary mapping is closed.
        anchor = ctypes.c_char.from_buffer(mapping)
        address = ctypes.addressof(anchor)
        started_at = time.monotonic()
        try:
            ctypes.set_errno(0)
            if _HOST_MADVISE(address, self.capacity_bytes, _MADV_POPULATE_WRITE) != 0:
                error_number = ctypes.get_errno()
                unsupported = {
                    errno.EINVAL,
                    errno.ENOSYS,
                    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                }
                if error_number not in unsupported:
                    raise OSError(error_number, os.strerror(error_number), self.path)
                _HOST_MEMSET(address, 0, self.capacity_bytes)
        except BaseException:
            del anchor
            mapping.close()
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            raise
        else:
            del anchor
            mapping.close()
        self.preallocation_seconds = time.monotonic() - started_at
        self.used_bytes = 0
        self.committed_bytes = self.capacity_bytes
        self._lock = threading.Lock()
        self._free_extents: list[tuple[int, int]] = [(0, self.capacity_bytes)]
        self._active_extents: dict[
            int, tuple[SharedMHAHostSnapshot, int, int]
        ] = {}
        self._closed = False

    @classmethod
    def _align_up(cls, value: int) -> int:
        return (int(value) + cls._ALIGNMENT - 1) // cls._ALIGNMENT * cls._ALIGNMENT

    @classmethod
    def _align_down(cls, value: int) -> int:
        return int(value) // cls._ALIGNMENT * cls._ALIGNMENT

    def path_for(self, snapshot_id: str) -> str:
        del snapshot_id
        return self.path

    def can_reserve(self, byte_size: int, hard_watermark: float) -> bool:
        requested = self._align_up(byte_size)
        with self._lock:
            if self.used_bytes + requested > int(
                self.capacity_bytes * float(hard_watermark)
            ):
                return False
            return any(length >= requested for _, length in self._free_extents)

    def create(self, snapshot_id: str, token_count: int, device_pool, byte_size: int):
        del snapshot_id
        requested = self._align_up(byte_size)
        with self._lock:
            candidates = [
                (length, offset, index)
                for index, (offset, length) in enumerate(self._free_extents)
                if length >= requested
            ]
            if not candidates:
                raise RuntimeError("shared Host arena has no contiguous capacity")
            _, offset, index = min(candidates)
            free_offset, free_length = self._free_extents.pop(index)
            if free_length > requested:
                self._free_extents.append(
                    (free_offset + requested, free_length - requested)
                )
                self._free_extents.sort()
            try:
                snapshot = LazySharedMHAHostSnapshot(
                    path=self.path,
                    token_count=token_count,
                    device_pool=device_pool,
                    byte_size=int(byte_size),
                    allocation_bytes=requested,
                    create=False,
                    file_offset=offset,
                )
                snapshot.offset = offset
            except BaseException:
                self._insert_free_locked(offset, requested)
                raise
            self.used_bytes += requested
            self._active_extents[id(snapshot)] = (snapshot, offset, requested)
        return snapshot

    def _insert_free_locked(self, offset: int, allocation_bytes: int) -> None:
        self._free_extents.append((int(offset), int(allocation_bytes)))
        self._free_extents.sort()
        merged: list[tuple[int, int]] = []
        for current_offset, current_length in self._free_extents:
            if merged and merged[-1][0] + merged[-1][1] == current_offset:
                previous_offset, previous_length = merged[-1]
                merged[-1] = (
                    previous_offset,
                    previous_length + current_length,
                )
            else:
                merged.append((current_offset, current_length))
        self._free_extents = merged

    def release(self, snapshot: SharedMHAHostSnapshot) -> bool:
        with self._lock:
            active = self._active_extents.get(id(snapshot))
            # Identity is the extent lease.  A stale release from the prior
            # request must not release a newer request that recycled the range.
            if active is None or active[0] is not snapshot:
                return True
            _, offset, allocation_bytes = active
            try:
                snapshot.close(unlink=False)
            except Exception:
                logger.exception(
                    "Failed to close shared Host arena extent offset=%d bytes=%d",
                    offset,
                    allocation_bytes,
                )
                # Keep the extent lease and accounting intact.  The caller
                # must retry and may not publish an ownership ACK.
                return False
            self._active_extents.pop(id(snapshot), None)
            self.used_bytes = max(0, self.used_bytes - int(allocation_bytes))
            self._insert_free_locked(offset, allocation_bytes)
            return True

    def usage(self) -> float:
        with self._lock:
            return self.used_bytes / max(1, self.capacity_bytes)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for snapshot, _, _ in self._active_extents.values():
                snapshot.close(unlink=False)
            self._active_extents.clear()
            self._free_extents = []
            self.used_bytes = 0
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self._closed = True


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
        # Edge-triggered notifications consumed by the P scheduler. The
        # control worker owns ledger/I/O progress; the scheduler receives only
        # snapshot ids whose allocator/Radix boundary may now advance.
        self._scheduler_events: queue.SimpleQueue = queue.SimpleQueue()
        self._ledger_entries_cache: dict[str, dict[str, Any]] = {}
        self._pending_host_offers: dict[str, dict[str, Any]] = {}
        self.max_h2d_inflight = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT", "4"))
        )
        self.h2d_chunk_tokens = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS", "1024"))
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
                    "host_bounce": PinnedMHAHostBounce(
                        self.device_pool, self.h2d_chunk_tokens
                    ),
                }
            )
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
            "Agentic shared Host arena enabled directory=%s capacity_gib=%.1f "
            "reserved_hbm_mib=0 h2d_priority=%d h2d_max_inflight=%d "
            "h2d_chunk_tokens=%d host_bounce=preallocated_pinned "
            "arena_preallocated=true preallocation_s=%.3f",
            self.arena.directory,
            self.arena.capacity_bytes / (1024**3),
            h2d_priority,
            self.max_h2d_inflight,
            self.h2d_chunk_tokens,
            self.arena.preallocation_seconds,
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
        if not self.arena.release(snapshot):
            return False
        record.pop("snapshot", None)
        return True

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
            if state == HostStageState.HOST_READY.value:
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
                # extent; it does *not* mean P needs the snapshot now.  Eagerly
                # cudaHostRegister-ing every completed extent caused bursts of
                # multi-GiB registrations to contend with Prefill even while
                # the corresponding tool was still running.  Keep the extent
                # lazy and let gate_request() materialize only the selected
                # slow-recovery request.
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
            self._host_eviction_count += 1
            self._host_eviction_tokens += token_count
            self._host_eviction_bytes += byte_size
            logger.warning(
                "AgenticKV shared_host_evict_release snapshot=%s tokens=%d "
                "bytes=%d tp_rank=%d/%d reason=%s",
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
        if acknowledged:
            self._notify_scheduler("recompute_required", snapshot_id)
        return acknowledged

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
                or getattr(self, "_prestart_recovery_aborts", {})
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
        self._progress_prestart_aborts()
        self._progress_h2d_loads()
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
        """Return whether P already owns a readable Host snapshot."""

        with self._get_state_lock():
            return request_generation.snapshot_id in self.host_ready

    def drain_scheduler_events(self) -> tuple[tuple[str, str], ...]:
        """Drain completed control edges without reading the file ledger."""

        events = []
        while True:
            try:
                events.append(self._scheduler_events.get_nowait())
            except queue.Empty:
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
            occupied = set(int(value) for value in reservations.values())
            for lane_id in range(lane_count):
                if lane_id not in occupied:
                    reservations[snapshot_id] = lane_id
                    return lane_id
            return None

    def _release_h2d_lane(self, snapshot_id: str) -> None:
        """Return a Slow-I/O lane after handoff or a fenced terminal path."""

        with AgenticPHostStagingManager._get_state_lock(self):
            reservations = getattr(self, "_h2d_lane_reservations", None)
            if reservations is not None:
                reservations.pop(snapshot_id, None)

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
            self.ledger.transition(
                request_generation.snapshot_id,
                HostStageState.H2D_LOADING,
                owner=self.owner,
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

    def _start_h2d_chunk(self, load: dict[str, Any]) -> None:
        """Launch one slow-path H2D chunk on the dedicated I/O stream."""

        record = load["record"]
        device_indices = load["device_indices"]
        lane = AgenticPHostStagingManager._h2d_lane_resources(
            self, int(load.get("h2d_lane_id", 0))
        )
        start = int(load.get("offset", 0))
        end = min(start + self.h2d_chunk_tokens, len(device_indices))
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
        launch_fence = H2DLaunchFence(
            event=torch.cuda.Event(enable_timing=True)
        )
        # Publish the fence object before submitting CUDA work.  Therefore an
        # exception anywhere in launch still leaves cleanup enough physical
        # state to prove completion or quarantine the complete snapshot.
        load["launch_fence"] = launch_fence
        load["event"] = launch_fence.event
        record["loading"] = "h2d"
        event, copy_refs = record["snapshot"].start_load_range_to_device(
            device_indices[start:end],
            lane["stream"],
            source_start=start,
            staging=lane["staging"],
            host_bounce=lane["host_bounce"],
            launch_fence=launch_fence,
        )
        load["event"] = event
        load["copy_refs"] = copy_refs
        load["chunk_end"] = end
        if start == 0:
            logger.info(
                "AgenticKV shared_host_h2d_start snapshot=%s tokens=%d "
                "bytes=%d lane=%d tp_prepared=true async_progress=true",
                load["request_generation"].snapshot_id,
                int(record["offer"]["token_count"]),
                int(record["offer"]["byte_size"]),
                int(load.get("h2d_lane_id", 0)),
            )

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
            load["host_released"] = True
            self.host_ready.pop(snapshot_id, None)
        self._release_record(record)
        logger.info(
            "AgenticKV shared_host_h2d_release snapshot=%s tp_rank=%d "
            "reason=all_tp_shards_radix_bound",
            snapshot_id,
            self.tp_rank,
        )
        return True

    def _discard_failed_h2d_load(self, rid: str, load: dict[str, Any]) -> bool:
        """Quiesce one failed shard and retain Host for a group retry."""
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
        if not load.get("drop_host_on_abort"):
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
        with self._get_state_lock():
            if self.loads.get(rid) is not load:
                return True
            self.loads.pop(rid, None)
            if not load.get("device_released"):
                self.workset_broker.request_release(
                    load["request_generation"].snapshot_id,
                    workset_lease,
                )
                load["device_released"] = True
            record = load["record"]
            record["loading"] = False
            if load.get("drop_host_on_abort") and not load.get("host_released"):
                self._release_record(load["record"])
                load["host_released"] = True
            elif not load.get("drop_host_on_abort"):
                self.host_ready[load["request_generation"].snapshot_id] = record
                AgenticPHostStagingManager._notify_scheduler(
                    self,
                    "host_ready",
                    load["request_generation"].snapshot_id,
                )
        AgenticPHostStagingManager._release_h2d_lane(
            self, load["request_generation"].snapshot_id
        )
        if load.get("drop_host_on_abort"):
            ledger = getattr(self, "ledger", None)
            if ledger is not None:
                ledger.transition(
                    load["request_generation"].snapshot_id,
                    HostStageState.FAILED,
                    owner=self.owner,
                    reason="request_aborted",
                )
        else:
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
        logger.info(
            "AgenticKV shared_host_h2d_complete snapshot=%s tokens=%d "
            "elapsed_ms=%.3f gib_per_s=%.3f async_progress=true",
            snapshot_id,
            int(record["offer"]["token_count"]),
            h2d_elapsed_ms,
            0.0
            if not math.isfinite(h2d_elapsed_ms)
            else int(record["offer"]["byte_size"])
            / max(h2d_elapsed_ms / 1000.0, 1e-9)
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
            claim_id = self.workset_broker.slow_owner(snapshot_id, rid)
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
                record = self.host_ready.pop(snapshot_id, None)
            if record is not None and not self._release_record(record):
                with self._get_state_lock():
                    self.host_ready.setdefault(snapshot_id, record)
                continue
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
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)

    def abort_request(self, rid: str, request_generation) -> None:
        """Cancel one Slow restore without racing an in-flight H2D."""

        snapshot_id = request_generation.snapshot_id
        prestart_abort = False
        with self._get_state_lock():
            load = self.loads.get(rid)
            if load is not None:
                load["abort_requested"] = True
                load.setdefault("io_error", RuntimeError("request aborted"))
                load["drop_host_on_abort"] = True
            else:
                record = self.host_ready.get(snapshot_id)
                if record is not None:
                    record["abort_requested"] = True
                    record["loading"] = "abort_pending"
                    pending = getattr(self, "_prestart_recovery_aborts", None)
                    if pending is None:
                        pending = self._prestart_recovery_aborts = {}
                    pending[snapshot_id] = {
                        "rid": str(rid),
                        "request_generation": request_generation,
                    }
                    prestart_abort = True
                else:
                    self.workset_broker.cancel_unstarted(
                        snapshot_id,
                        owner=self.workset_broker.slow_owner(snapshot_id, rid),
                    )
                    AgenticPHostStagingManager._release_h2d_lane(
                        self, snapshot_id
                    )
        if prestart_abort:
            self._progress_prestart_aborts()
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
                    result = self.tree_cache.insert(
                        InsertParams(
                            key=radix_key,
                            value=device_indices,
                            priority=getattr(req, "priority", 0) or 0,
                        )
                    )
                    inserted = True
                    self.workset_broker.commit_parent_bound(
                        snapshot_id, workset_lease
                    )
                    if result.prefix_len:
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
            with self._get_state_lock():
                if self.loads.get(req.rid) is load:
                    self.loads.pop(req.rid, None)
            AgenticPHostStagingManager._release_h2d_lane(self, snapshot_id)
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
        with self._get_state_lock():
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
            self.workset_broker.request(
                snapshot_id,
                int(offer["token_count"]),
                prompt_tokens,
                owner=workset_owner,
            )
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
        staging_tokens = int(
            os.getenv("SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS", "256")
        )
        self._d2h_lanes = [
            {
                "stream": torch.cuda.Stream(
                    device=torch.cuda.current_device(), priority=d2h_priority
                ),
                "staging": LayerFirstD2HStaging(
                    self.device_pool, staging_tokens
                ),
                "host_bounce": PinnedMHAHostBounce(
                    self.device_pool, self._d2h_chunk_tokens
                ),
                "snapshot_id": None,
            }
            for _ in range(lane_count)
        ]

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
        event = write.get("event")
        if event is not None and not event.query():
            return False
        candidate.pop("arena_write", None)
        lane_id = write.pop("lane_id", None)
        if lane_id is not None:
            lane = self._d2h_lanes[int(lane_id)]
            if lane["snapshot_id"] == candidate["manifest"].snapshot_id:
                lane["snapshot_id"] = None
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
        """Submit one bounded chunk on any free independent D2H lane."""

        snapshot_id = candidate["manifest"].snapshot_id
        write = candidate["arena_write"]
        if write["event"] is not None:
            return False
        lane_id = next(
            (
                index
                for index, lane in enumerate(self._d2h_lanes)
                if lane["snapshot_id"] is None
            ),
            None,
        )
        if lane_id is None:
            return False
        lane = self._d2h_lanes[lane_id]
        start = int(write["offset"])
        if start >= len(source_token_indices):
            return False
        end = min(start + self._d2h_chunk_tokens, len(source_token_indices))
        launch_fence = H2DLaunchFence(
            event=torch.cuda.Event(enable_timing=True)
        )
        write["lane_id"] = lane_id
        lane["snapshot_id"] = snapshot_id
        try:
            event, refs = write["snapshot"].start_backup_range_from_device(
                source_token_indices[start:end],
                destination_start=start,
                stream=lane["stream"],
                staging=lane["staging"],
                host_bounce=lane["host_bounce"],
                launch_fence=launch_fence,
            )
        except Exception as exc:
            if launch_fence.submitted:
                # Preserve the lane, source refs, and completion authority.
                # No pageable Host bytes are committed from this failed
                # bounce. Once the fence drains, progress() reopens the same
                # complete extent and retries from the unchanged offset.
                write["launch_fence"] = launch_fence
                write["launch_error"] = exc
                write["copy_refs"] = launch_fence.copy_refs
                write["event"] = (
                    launch_fence.event if launch_fence.armed else None
                )
                if launch_fence.unavailable:
                    logger.error(
                        "AgenticKV D2H launch has no completion fence "
                        "snapshot=%s; quarantining D source and Host extent",
                        snapshot_id,
                    )
            else:
                lane["snapshot_id"] = None
                write.pop("lane_id", None)
                candidate["arena_write_retry_at"] = time.monotonic() + 0.05
                logger.exception(
                    "AgenticKV D2H launch failed before submission "
                    "snapshot=%s; retrying",
                    snapshot_id,
                )
            return False
        write["event"] = event
        write["copy_refs"] = refs
        write["launch_fence"] = launch_fence
        write["chunk_end"] = end
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
        if write.get("launch_error") is not None:
            launch_fence = write.get("launch_fence")
            if launch_fence is None or launch_fence.unavailable:
                return "waiting"
            event = write.get("event")
            if event is None or not event.query():
                return "waiting"
            event.synchronize()
            if not self._cleanup_write(candidate):
                return "waiting"
            candidate["arena_write_retry_at"] = time.monotonic() + 0.05
            logger.warning(
                "AgenticKV drained partial D2H launch snapshot=%s; retrying "
                "complete chunk from D source",
                snapshot_id,
            )
            return "waiting"
        if write["event"] is None:
            self._start_write_chunk(candidate, source_token_indices)
            return "waiting"
        if not write["event"].query():
            return "waiting"
        chunk_elapsed_ms = write["snapshot"]._last_d2h_start_event.elapsed_time(
            write["event"]
        )
        chunk_start = int(write["offset"])
        chunk_end = int(write["chunk_end"])
        lane_id = int(write.pop("lane_id"))
        lane = self._d2h_lanes[lane_id]
        write["snapshot"].commit_backup_range_from_bounce(
            lane["host_bounce"],
            destination_start=chunk_start,
            token_count=chunk_end - chunk_start,
        )
        write["gpu_elapsed_ms"] += chunk_elapsed_ms
        write["offset"] = int(write["chunk_end"])
        write["event"] = None
        write["copy_refs"] = None
        lane["snapshot_id"] = None
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
            "elapsed_ms=%.3f gib_per_s=%.3f",
            snapshot_id,
            elapsed_ready,
            d2h_elapsed_ms,
            d2h_bytes / max(d2h_elapsed_ms / 1000.0, 1e-9) / (1024**3),
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
