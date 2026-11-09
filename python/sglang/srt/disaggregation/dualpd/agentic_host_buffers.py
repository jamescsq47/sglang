from __future__ import annotations

"""Physical Host snapshot views, copy buffers/fences and extent suballocation.

Request-generation managers retain lifecycle ownership and decide when a
physical completion fence permits publication, release, retry or quarantine.
"""
import ctypes
import errno
import logging
import mmap
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger("sglang.srt.disaggregation.agentic_host_staging")

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
_HOST_MEMFD_CREATE = getattr(_HOST_LIBC, "memfd_create", None)
if _HOST_MEMFD_CREATE is not None:
    _HOST_MEMFD_CREATE.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    _HOST_MEMFD_CREATE.restype = ctypes.c_int
_MADV_POPULATE_WRITE = 23
_MFD_CLOEXEC = 0x0001
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2

from sglang.srt.disaggregation.agentic_host_dma import (
    _cuda_driver_batch_memcpy, _cuda_batch_memcpy_async, _cuda_memcpy_async,
)
from sglang.srt.disaggregation.agentic_host_registration import (
    _validate_shared_host_backing_path, _open_shared_host_backing,
    _registered_host_arena, _REGISTERED_HOST_ARENAS_LOCK,
)


def _registered_indexed_batch_copy(
    snapshot,
    *,
    device_indices: list[int] | tuple[int, ...],
    host_start: int,
    stream,
    host_to_device: bool,
) -> tuple[Any, ...] | bool:
    """Copy layer-first KV between Host and arbitrary device token indices.

    Adjacent device indices are coalesced, normally turning each allocator
    page into one 128-KiB operation for Qwen3-8B.  Host spans are additionally
    split at persistent registration-window boundaries.  The resulting batch
    uses only CUDA copy engines; no gather/scatter kernel occupies an SM.
    """

    if _cuda_driver_batch_memcpy() is None:
        return False
    indices = np.asarray(device_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("registered batch-copy indices must be one-dimensional")
    if not indices.size:
        return True
    device_layers = tuple(snapshot.device_pool.k_buffer) + tuple(
        snapshot.device_pool.v_buffer
    )
    if any(
        int(layer.stride(0)) * int(layer.element_size()) != int(snapshot.item_size)
        for layer in device_layers
    ):
        return False
    device_capacity = min(int(layer.shape[0]) for layer in device_layers)
    if int(indices.min()) < 0 or int(indices.max()) >= device_capacity:
        raise ValueError("registered batch-copy device index is out of bounds")
    run_starts = np.concatenate(
        (
            np.asarray((0,), dtype=np.int64),
            np.flatnonzero(np.diff(indices) != 1).astype(np.int64) + 1,
        )
    )
    run_ends = np.concatenate(
        (run_starts[1:], np.asarray((indices.size,), dtype=np.int64))
    )
    run_lengths = run_ends - run_starts
    run_device_starts = indices[run_starts]

    item_size = int(snapshot.item_size)
    # Derive layer addresses from strides instead of constructing a Torch view
    # per layer on every chunk.  Broadcasting builds the same K/V-major batch
    # without 2*layer_num tensor slices and small NumPy allocations under GIL.
    host = snapshot.kv_buffer
    element_size = int(host.element_size())
    layer_offsets = (
        np.arange(2, dtype=np.uint64)[:, None] * np.uint64(host.stride(0))
        + np.arange(int(snapshot.layer_num), dtype=np.uint64)[None, :]
        * np.uint64(host.stride(1))
        + np.uint64(int(host_start) * host.stride(2))
    ).reshape(-1) * np.uint64(element_size)
    host_bases = np.uint64(host.data_ptr()) + layer_offsets
    device_bases = np.asarray(
        [layer.data_ptr() for layer in device_layers], dtype=np.uint64
    )
    host_pointers = (
        host_bases[:, None]
        + run_starts.astype(np.uint64)[None, :] * np.uint64(item_size)
    ).reshape(-1)
    device_pointers = (
        device_bases[:, None]
        + run_device_starts.astype(np.uint64)[None, :] * np.uint64(item_size)
    ).reshape(-1)
    remaining_sizes = np.tile(
        run_lengths.astype(np.uint64) * np.uint64(item_size),
        int(snapshot.layer_num) * 2,
    )

    # Split the rare operation that crosses a separately registered cache
    # window.  Work is vectorized over every operation, so descriptor creation
    # does not hold the Python GIL for thousands of pointer-by-pointer appends.
    host_rounds = []
    device_rounds = []
    size_rounds = []
    arena = getattr(snapshot, "_arena_mapping", None)
    while remaining_sizes.size:
        if arena is None:
            round_sizes = remaining_sizes
        else:
            relative = host_pointers - np.uint64(arena.raw.data_ptr())
            to_boundary = np.uint64(arena.window_bytes) - (
                relative % np.uint64(arena.window_bytes)
            )
            round_sizes = np.minimum(remaining_sizes, to_boundary)
        host_rounds.append(host_pointers)
        device_rounds.append(device_pointers)
        size_rounds.append(round_sizes)
        has_tail = remaining_sizes > round_sizes
        if not np.any(has_tail):
            break
        host_pointers = host_pointers[has_tail] + round_sizes[has_tail]
        device_pointers = device_pointers[has_tail] + round_sizes[has_tail]
        remaining_sizes = remaining_sizes[has_tail] - round_sizes[has_tail]
    host_pointers = np.concatenate(host_rounds)
    device_pointers = np.concatenate(device_rounds)
    sizes = np.concatenate(size_rounds)
    destinations = device_pointers if host_to_device else host_pointers
    sources = host_pointers if host_to_device else device_pointers
    return _cuda_batch_memcpy_async(
        destinations, sources, sizes, stream=stream
    )

def _device_indices_to_host(indices) -> list[int]:
    """Materialize the tiny token-index vector outside the Forward thread."""

    if torch.is_tensor(indices):
        if indices.is_cuda:
            with torch.cuda.device(indices.device):
                indices = indices.detach().to(device="cpu", dtype=torch.int64)
        else:
            indices = indices.detach().to(dtype=torch.int64)
        return [int(value) for value in indices.tolist()]
    return [int(value) for value in indices]

def _create_agentic_host_memfd() -> int:
    """Create anonymous shared memory outside the /dev/shm mount quota."""

    name = f"sglang-agentic-host-arena-{os.getpid()}"
    create = getattr(os, "memfd_create", None)
    if create is not None:
        return int(create(name, getattr(os, "MFD_CLOEXEC", _MFD_CLOEXEC)))
    if _HOST_MEMFD_CREATE is None:
        raise RuntimeError("memfd Host backend is unavailable on this platform")
    ctypes.set_errno(0)
    fd = int(_HOST_MEMFD_CREATE(name.encode(), _MFD_CLOEXEC))
    if fd < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return fd

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

class HostCopyWorkerPool:
    """Small daemon pool for pageable<->pinned Host copies.

    CUDA progress threads must remain cheap: a large synchronous memmove there
    delays polling every other transfer and competes with the model scheduler's
    Python control path.  Jobs retain their tensor/mmap references through the
    Future, and callers keep the corresponding bounce slot reserved until that
    Future reaches a physical completion boundary.

    Threads inherit the model process' CPU affinity and NUMA memory policy, so
    launch scripts can place them with the same per-engine numactl binding.
    """

    def __init__(self, name: str, workers: int):
        self._jobs: queue.Queue = queue.Queue()
        self._threads = []
        for worker_id in range(max(1, int(workers))):
            thread = threading.Thread(
                target=self._worker,
                name=f"{name}-{worker_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def submit(self, function, /, *args, **kwargs) -> Future:
        future = Future()
        self._jobs.put((future, function, args, kwargs))
        return future

    def _worker(self) -> None:
        while True:
            future, function, args, kwargs = self._jobs.get()
            if not future.set_running_or_notify_cancel():
                continue
            started_at = time.perf_counter()
            try:
                function(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(time.perf_counter() - started_at)

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

def _concrete_kv_pool_device(device_pool) -> torch.device:
    """Return the rank-local CUDA device, never the ambiguous ``cuda`` alias.

    CUDA's current device is thread-local.  Agentic I/O runs outside the
    scheduler thread, so ``device_pool.device == torch.device("cuda")`` may
    resolve to cuda:0 on every TP rank.  The allocated KV tensors are the
    authoritative physical owner and always carry the concrete rank index.
    """

    device_pool = getattr(device_pool, "full_kv_pool", device_pool)
    for buffers_name in ("k_buffer", "v_buffer"):
        buffers = getattr(device_pool, buffers_name, None)
        if buffers is not None and len(buffers):
            return torch.device(buffers[0].device)
    device = torch.device(device_pool.device)
    if device.type == "cuda" and device.index is None:
        raise RuntimeError("KV pool does not expose a concrete rank-local CUDA device")
    return device

class LayerFirstD2HStaging:
    """Small reusable HBM gather buffer feeding contiguous PCIe D2H DMA."""

    def __init__(self, device_pool, token_capacity: int):
        device_pool = getattr(device_pool, "full_kv_pool", device_pool)
        self.token_capacity = max(1, int(token_capacity))
        self.layer_num = int(device_pool.layer_num)
        self.head_num = int(device_pool.head_num)
        self.head_dim = int(device_pool.head_dim)
        self.v_head_dim = int(getattr(device_pool, "v_head_dim", self.head_dim))
        self.dtype = device_pool.store_dtype
        self.device = _concrete_kv_pool_device(device_pool)
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
    def __new__(cls, *args, **kwargs):
        if cls is SharedMHAHostSnapshot and hasattr(kwargs.get("device_pool"), "mamba_pool"):
            from sglang.srt.disaggregation.agentic_hybrid_dma import RegisteredHybridHostSnapshot
            return RegisteredHybridHostSnapshot(*args, **kwargs)
        return super().__new__(cls)

    """One request-generation extent mapped by both P and its owning D.

    The hot on-disk shape is layer-first and snapshot-local:
    ``[K/V, layer, token, head, head_dim]``.  This lets D gather scattered KV
    into a small reusable HBM slot and then use contiguous PCIe DMA into Host.
    The backing object lives either in tmpfs or an anonymous memfd, and the
    mapping starts pageable.  By default the I/O worker registers this exact
    request extent on first use, so CUDA DMA reaches the final Shared-Arena
    bytes directly and no CPU commit copy is required.  Registration is
    deliberately lazy (never done by the model forward thread), and failure
    falls back to the bounded pinned-bounce pipeline while GPU ownership is
    still unchanged.
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
        _validate_shared_host_backing_path(path, create=create)
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
        registered_dma_requested = os.getenv(
            "SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._registered_dma_enabled = bool(
            registered_dma_requested
            and hasattr(device_pool, "device")
            and torch.cuda.get_device_capability(
                _concrete_kv_pool_device(device_pool)
            )[0]
            >= 9
        )
        self._arena_mapping = None
        self._owns_mapping = True
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = _open_shared_host_backing(path, flags, 0o600)
        try:
            if create:
                if self.file_offset:
                    raise ValueError("new standalone Host snapshots require offset zero")
                os.ftruncate(fd, self.byte_size)
            elif os.fstat(fd).st_size < self.file_offset + self.byte_size:
                raise ValueError("shared Host extent file is smaller than the snapshot")
        finally:
            os.close(fd)
        if self._registered_dma_enabled and not create:
            self._arena_mapping = _registered_host_arena(
                path, _concrete_kv_pool_device(self.device_pool)
            )
            self.mapping = self._arena_mapping.mapping
            raw = self._arena_mapping.raw[
                self.file_offset : self.file_offset + self.byte_size
            ]
            self._owns_mapping = False
        else:
            fd = _open_shared_host_backing(path, os.O_RDWR)
            try:
                self.mapping = mmap.mmap(
                    fd,
                    self.byte_size,
                    access=mmap.ACCESS_WRITE,
                    offset=self.file_offset,
                )
            finally:
                os.close(fd)
            raw = torch.frombuffer(
                self.mapping, dtype=torch.uint8, count=self.byte_size
            )
        self.kv_buffer = raw.view(self.dtype).view(
            2,
            self.layer_num,
            self.token_count,
            self.head_num,
            self.head_dim,
        )
        self._raw = raw
        self._closed = False
        self._cuda_host_registered = False
        self._registered_windows: tuple[int, ...] = ()
        self._cuda_host_registration_failed = False
        self._cuda_host_registration_lock = threading.Lock()
        self.cuda_host_registration_seconds = 0.0

    @property
    def cuda_host_registered(self) -> bool:
        return bool(self._cuda_host_registered)

    def prepare_registered_host_dma(self) -> bool:
        """Register the final extent for direct DMA, or select safe fallback.

        Callers invoke this on an existing transport/Host-copy worker.  A
        registration error occurs before any CUDA transfer is submitted, so
        retaining the pageable mapping and using the old bounce path preserves
        the original ownership protocol.
        """

        if not self._registered_dma_enabled or self._cuda_host_registration_failed:
            return False
        if self._cuda_host_registered:
            return True
        with self._cuda_host_registration_lock:
            if self._cuda_host_registered:
                return True
            if self._cuda_host_registration_failed:
                return False
            started_at = time.perf_counter()
            try:
                if self._arena_mapping is not None:
                    self._registered_windows = self._arena_mapping.acquire(
                        self.file_offset,
                        self.byte_size,
                        _concrete_kv_pool_device(self.device_pool),
                    )
                else:
                    with torch.cuda.device(_concrete_kv_pool_device(self.device_pool)):
                        cudart = torch.cuda.cudart()
                        result = cudart.cudaHostRegister(
                            self._raw.data_ptr(), self.byte_size, 0
                        )
                        if result != cudart.cudaError.success:
                            raise RuntimeError(f"cudaHostRegister returned {result}")
            except BaseException:
                self._cuda_host_registration_failed = True
                logger.exception(
                    "AgenticKV registered-extent DMA unavailable path=%s "
                    "offset=%d bytes=%d; using pinned-bounce fallback",
                    self.path,
                    self.file_offset,
                    self.byte_size,
                )
                return False
            self.cuda_host_registration_seconds = (
                time.perf_counter() - started_at
            )
            self._cuda_host_registered = True
            logger.info(
                "AgenticKV registered_host_extent path=%s offset=%d bytes=%d "
                "registration_ms=%.3f",
                self.path,
                self.file_offset,
                self.byte_size,
                self.cuda_host_registration_seconds * 1000.0,
            )
            return True

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def _registered_token_spans(self, host_view, token_count: int):
        """Split a contiguous layer range at CUDA registration boundaries."""

        token_count = int(token_count)
        if self._arena_mapping is None or not self._cuda_host_registered:
            return ((0, token_count),)
        relative = int(host_view.data_ptr()) - int(
            self._arena_mapping.raw.data_ptr()
        )
        spans = []
        token_offset = 0
        while token_offset < token_count:
            current = relative + token_offset * self.item_size
            bytes_to_boundary = self._arena_mapping.window_bytes - (
                current % self._arena_mapping.window_bytes
            )
            span_tokens = min(
                token_count - token_offset,
                bytes_to_boundary // self.item_size,
            )
            if span_tokens <= 0:
                raise RuntimeError(
                    "registered Host window boundary splits one KV token"
                )
            spans.append((token_offset, token_offset + span_tokens))
            token_offset += span_tokens
        return tuple(spans)

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
        source_indices_host=None,
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
        registered = self.prepare_registered_host_dma()
        if not registered:
            if host_bounce is None:
                raise ValueError("pageable Host snapshots require a pinned D2H bounce")
            if len(source_indices) > host_bounce.token_capacity:
                raise ValueError("D2H chunk is larger than the pinned Host bounce")
        start_event = torch.cuda.Event(enable_timing=True)
        if launch_fence is None:
            launch_fence = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
        event = launch_fence.event
        copy_refs = [
            source_indices,
            original_source_indices,
            staging,
            self,
        ]
        if source_indices_host is not None:
            if len(source_indices_host) != len(source_indices):
                raise ValueError("Host source-index mirror has the wrong length")
            copy_refs.append(source_indices_host)
        if not registered:
            copy_refs.append(host_bounce)
        launch_fence.copy_refs = copy_refs
        try:
            with torch.cuda.stream(stream):
                # The event is the release authority if CUDA accepts any part
                # of this D2H launch and a later submission raises.
                launch_fence.submitted = True
                start_event.record(stream)
                batch_submitted = False
                if registered and source_indices_host is not None:
                    batch_submitted = _registered_indexed_batch_copy(
                        self,
                        device_indices=source_indices_host,
                        host_start=destination_start,
                        stream=stream,
                        host_to_device=False,
                    )
                    if batch_submitted is not False:
                        copy_refs.append(batch_submitted)
                if not batch_submitted and (
                    not source_indices.is_cuda
                    or source_indices.dtype != torch.int64
                ):
                    source_indices = source_indices.to(
                        device=_concrete_kv_pool_device(self.device_pool),
                        dtype=torch.int64,
                        non_blocking=True,
                    )
                    copy_refs.append(source_indices)
                if not batch_submitted:
                    safe_indexed_copy = (
                        torch.cuda.get_device_capability(
                            _concrete_kv_pool_device(self.device_pool)
                        )[0]
                        < 9
                    )
                    for start in range(0, len(source_indices), staging.token_capacity):
                        count = min(staging.token_capacity, len(source_indices) - start)
                        source_chunk = source_indices[start : start + count]
                        local_indices = staging.local_indices[:count]
                        if safe_indexed_copy:
                            for layer_id in range(self.layer_num):
                                torch.index_select(
                                    self.device_pool.k_buffer[layer_id], 0,
                                    source_chunk,
                                    out=staging.k_buffer[layer_id][:count],
                                )
                                torch.index_select(
                                    self.device_pool.v_buffer[layer_id], 0,
                                    source_chunk,
                                    out=staging.v_buffer[layer_id][:count],
                                )
                        else:
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
                            destination_k = (
                                self.k_buffer[layer_id, host_start:host_end]
                                if registered
                                else host_bounce.k_buffer[layer_id, start : start + count]
                            )
                            destination_v = (
                                self.v_buffer[layer_id, host_start:host_end]
                                if registered
                                else host_bounce.v_buffer[layer_id, start : start + count]
                            )
                            for span_start, span_end in self._registered_token_spans(
                                destination_k, count
                            ):
                                destination_span = destination_k[span_start:span_end]
                                source_span = staging.k_buffer[layer_id][
                                    span_start:span_end
                                ]
                                if registered:
                                    _cuda_memcpy_async(
                                        destination_span,
                                        source_span,
                                        byte_size=(span_end - span_start) * self.item_size,
                                        kind=_CUDA_MEMCPY_DEVICE_TO_HOST,
                                        stream=stream,
                                    )
                                else:
                                    destination_span.copy_(source_span, non_blocking=True)
                            for span_start, span_end in self._registered_token_spans(
                                destination_v, count
                            ):
                                destination_span = destination_v[span_start:span_end]
                                source_span = staging.v_buffer[layer_id][
                                    span_start:span_end
                                ]
                                if registered:
                                    _cuda_memcpy_async(
                                        destination_span,
                                        source_span,
                                        byte_size=(span_end - span_start) * self.item_size,
                                        kind=_CUDA_MEMCPY_DEVICE_TO_HOST,
                                        stream=stream,
                                    )
                                else:
                                    destination_span.copy_(source_span, non_blocking=True)
                event.record(stream)
                launch_fence.armed = True
                if bool(getattr(source_indices, "is_cuda", False)):
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

        if self._cuda_host_registered:
            # The CUDA completion event is already the durability fence: DMA
            # landed in the final Shared-Arena address.
            return
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
        device_indices_host=None,
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

        source_start = int(source_start)
        token_count = len(device_indices)
        if source_start < 0 or source_start + token_count > self.token_count:
            raise ValueError("H2D chunk falls outside shared Host extent")
        original_device_indices = device_indices
        if host_bounce is None:
            raise ValueError("pageable Host snapshots require a pinned H2D bounce")
        if token_count > host_bounce.token_capacity:
            raise ValueError("H2D chunk is larger than the pinned Host bounce")
        self.prepare_load_range_to_bounce(
            source_start=source_start,
            token_count=token_count,
            host_bounce=host_bounce,
        )
        return self.start_load_range_from_bounce_to_device(
            device_indices,
            stream,
            staging=staging,
            host_bounce=host_bounce,
            source_start=source_start,
            launch_fence=launch_fence,
            device_indices_host=device_indices_host,
        )

    def prepare_load_range_to_bounce(
        self, *, source_start: int, token_count: int, host_bounce
    ) -> None:
        """Copy one pageable Host range into a pinned bounce on CPU.

        This phase intentionally has no CUDA side effect, so production callers
        can run it on a NUMA-local Host-copy worker and retain the Host extent as
        its physical source fence until the returned Future completes.
        """

        source_start = int(source_start)
        token_count = int(token_count)
        if source_start < 0 or source_start + token_count > self.token_count:
            raise ValueError("H2D chunk falls outside shared Host extent")
        if self.prepare_registered_host_dma():
            # Registration itself ran on this Host-copy worker.  The caller's
            # next CUDA operation can now DMA directly from the final extent.
            return
        if host_bounce is None or token_count > host_bounce.token_capacity:
            raise ValueError("H2D chunk is larger than the pinned Host bounce")
        _copy_layer_first_host_range(
            host_bounce.kv_buffer,
            self.kv_buffer,
            destination_start=0,
            source_start=source_start,
            token_count=token_count,
        )

    def start_load_range_from_bounce_to_device(
        self,
        device_indices,
        stream,
        *,
        staging=None,
        host_bounce=None,
        source_start: int = 0,
        launch_fence: Optional[H2DLaunchFence] = None,
        device_indices_host=None,
    ):
        """Launch a previously prepared pinned bounce into destination HBM."""

        from sgl_kernel.kvcacheio import transfer_kv_all_layer

        token_count = len(device_indices)
        original_device_indices = device_indices
        registered = self._cuda_host_registered
        source_start = int(source_start)
        if source_start < 0 or source_start + token_count > self.token_count:
            raise ValueError("H2D chunk falls outside shared Host extent")
        if not registered and (
            host_bounce is None or token_count > host_bounce.token_capacity
        ):
            raise ValueError("H2D chunk is larger than the pinned Host bounce")
        if staging is None:
            staging = LayerFirstD2HStaging(self.device_pool, token_count)
        if staging.token_capacity < token_count:
            raise ValueError("H2D staging buffer is smaller than the chunk")
        start_event = torch.cuda.Event(enable_timing=True)
        if launch_fence is None:
            launch_fence = H2DLaunchFence(
                event=torch.cuda.Event(enable_timing=True)
            )
        event = launch_fence.event
        copy_refs = [device_indices, staging, self]
        if device_indices_host is not None:
            if len(device_indices_host) != len(device_indices):
                raise ValueError("Host destination-index mirror has the wrong length")
            copy_refs.append(device_indices_host)
        if not registered:
            copy_refs.append(host_bounce)
        launch_fence.copy_refs = copy_refs
        try:
            with torch.cuda.stream(stream):
                # From this point onward an exception may follow a partially
                # submitted copy.  The pre-created event is the only release
                # authority for both Host and destination HBM.
                launch_fence.submitted = True
                start_event.record(stream)
                if registered and device_indices_host is not None:
                    batch_submitted = _registered_indexed_batch_copy(
                        self,
                        device_indices=device_indices_host,
                        host_start=source_start,
                        stream=stream,
                        host_to_device=True,
                    )
                    if batch_submitted:
                        copy_refs.append(batch_submitted)
                        event.record(stream)
                        launch_fence.armed = True
                        if bool(getattr(device_indices, "is_cuda", False)):
                            device_indices.record_stream(stream)
                        if bool(
                            getattr(original_device_indices, "is_cuda", False)
                        ):
                            original_device_indices.record_stream(stream)
                        self._last_h2d_start_event = start_event
                        copy_refs.extend((original_device_indices, start_event))
                        launch_fence.copy_refs = copy_refs
                        return event, tuple(copy_refs)
                if not device_indices.is_cuda or device_indices.dtype != torch.int64:
                    device_indices = device_indices.to(
                        device=_concrete_kv_pool_device(self.device_pool),
                        dtype=torch.int64,
                        non_blocking=True,
                    )
                    copy_refs.append(device_indices)
                source_indices = staging.local_indices[:token_count]
                source_end = source_start + token_count
                for layer_id in range(self.layer_num):
                    source_k = (
                        self.k_buffer[layer_id, source_start:source_end]
                        if registered
                        else host_bounce.k_buffer[layer_id, :token_count]
                    )
                    source_v = (
                        self.v_buffer[layer_id, source_start:source_end]
                        if registered
                        else host_bounce.v_buffer[layer_id, :token_count]
                    )
                    for span_start, span_end in self._registered_token_spans(
                        source_k, token_count
                    ):
                        destination_span = staging.k_buffer[layer_id][
                            span_start:span_end
                        ]
                        source_span = source_k[span_start:span_end]
                        if registered:
                            _cuda_memcpy_async(
                                destination_span,
                                source_span,
                                byte_size=(span_end - span_start) * self.item_size,
                                kind=_CUDA_MEMCPY_HOST_TO_DEVICE,
                                stream=stream,
                            )
                        else:
                            destination_span.copy_(source_span, non_blocking=True)
                    for span_start, span_end in self._registered_token_spans(
                        source_v, token_count
                    ):
                        destination_span = staging.v_buffer[layer_id][
                            span_start:span_end
                        ]
                        source_span = source_v[span_start:span_end]
                        if registered:
                            _cuda_memcpy_async(
                                destination_span,
                                source_span,
                                byte_size=(span_end - span_start) * self.item_size,
                                kind=_CUDA_MEMCPY_HOST_TO_DEVICE,
                                stream=stream,
                            )
                        else:
                            destination_span.copy_(source_span, non_blocking=True)
                if torch.cuda.get_device_capability(
                    _concrete_kv_pool_device(self.device_pool)
                )[0] < 9:
                    for layer_id in range(self.layer_num):
                        self.device_pool.k_buffer[layer_id].index_copy_(
                            0, device_indices,
                            staging.k_buffer[layer_id][:token_count],
                        )
                        self.device_pool.v_buffer[layer_id].index_copy_(
                            0, device_indices,
                            staging.v_buffer[layer_id][:token_count],
                        )
                else:
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
        if self._cuda_host_registered:
            if self._arena_mapping is not None:
                self._arena_mapping.release(self._registered_windows)
                self._registered_windows = ()
            else:
                with torch.cuda.device(_concrete_kv_pool_device(self.device_pool)):
                    cudart = torch.cuda.cudart()
                    result = cudart.cudaHostUnregister(self._raw.data_ptr())
                    if result != cudart.cudaError.success:
                        # Closing an mmap still visible to CUDA would turn a late DMA
                        # into use-after-unmap.  Keep the mapping leased and let the
                        # owner retry cleanup instead.
                        raise RuntimeError(f"cudaHostUnregister returned {result}")
            self._cuda_host_registered = False
        self.kv_buffer = None
        self._raw = None
        if self._owns_mapping:
            self.mapping.close()
        elif self._arena_mapping is not None:
            with _REGISTERED_HOST_ARENAS_LOCK:
                if self._arena_mapping._users <= 0:
                    raise RuntimeError("registered Host arena user underflow")
                self._arena_mapping._users -= 1
        self.mapping = None
        self._closed = True
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass

class PinnedMHAHostBounce:
    """Process-lifetime pinned Host chunk reused by every slow snapshot."""

    def __init__(self, device_pool, token_capacity: int):
        device_pool = getattr(device_pool, "full_kv_pool", device_pool)
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
    """A granted shared-Host extent whose P-side mapping is built later.

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
        _validate_shared_host_backing_path(path, create=create)
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
        self._pending_state_indices = None
        self._closed = False
        self._lock = threading.Lock()
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = _open_shared_host_backing(path, flags, 0o600)
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
            fd = _open_shared_host_backing(self.path, os.O_RDWR)
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

    def set_state_indices(self, indices):
        """Freeze CPU state metadata without mapping/registering the extent."""
        if not hasattr(self.device_pool, "mamba_pool"):
            raise ValueError("state indices require a hybrid KV pool")
        if torch.is_tensor(indices) and indices.device.type != "cpu":
            raise ValueError("lazy state indices must already be on CPU")
        values = tuple(int(value) for value in np.asarray(indices).reshape(-1))
        if len(values) not in (1, 2) or min(values) < 0:
            raise ValueError("hybrid snapshot requires one or two valid state slots")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot configure a released Host extent")
            if self._pending_state_indices is not None and self._pending_state_indices != values:
                raise ValueError("snapshot state indices are immutable")
            if self._materialized is not None:
                if self._materialized.state_slots != len(values):
                    raise ValueError("cannot change a materialized hybrid layout")
                self._materialized.set_state_indices(values)
            self._pending_state_indices = values

    def reset_state_indices_after_quiesce(self):
        """Reset a failed H2D attempt, without releasing its durable extent."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot reset a released Host extent")
            if self._pending_state_indices is not None and len(self._pending_state_indices) != 1:
                raise RuntimeError("P2D source state binding cannot be reset")
            if self._materialized is not None:
                self._materialized.reset_state_indices_after_quiesce()
            self._pending_state_indices = None

    def materialize(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot materialize a released Host extent")
            if self._materialized is None:
                state_args = {}
                if self._pending_state_indices is not None:
                    state_args["state_slots"] = len(self._pending_state_indices)
                inner = SharedMHAHostSnapshot(
                    path=self.path,
                    token_count=self.token_count,
                    device_pool=self.device_pool,
                    byte_size=self.byte_size,
                    create=False,
                    file_offset=self.file_offset,
                    **state_args,
                )
                try:
                    if self._pending_state_indices is not None:
                        inner.set_state_indices(self._pending_state_indices)
                except BaseException:
                    inner.close(unlink=False)
                    raise
                self._materialized = inner
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
    """One process-lifetime Host arena with request-generation suballocation.

    The complete file is physically allocated and first-touched once at P
    startup, under the P process' NUMA memory policy. A snapshot subsequently
    owns only one aligned ``(offset, length)`` extent in that file. D and P map
    the same subrange, so request-level ownership is preserved without
    per-snapshot ``ftruncate``, ``fallocate`` or prefault workers.

    The arena itself remains pageable.  A live request acquires only its
    covering windows from a bounded process-local registration cache and
    releases their references before its extent lease is returned.  This keeps
    DMA direct without registering the complete hundreds-of-GiB arena or
    repeating page registration for recycled offsets.
    """

    _ALIGNMENT = mmap.ALLOCATIONGRANULARITY

    def __init__(
        self, directory: str, capacity_bytes: int, *, backend: str = "tmpfs"
    ):
        if not directory.startswith("/dev/shm/"):
            raise ValueError(
                "shared Host arena control directory must reside in /dev/shm"
            )
        self.directory = directory.rstrip("/")
        self.backend = str(backend).strip().lower()
        if self.backend not in {"tmpfs", "memfd"}:
            raise ValueError("shared Host arena backend must be tmpfs or memfd")
        self.capacity_bytes = self._align_down(int(capacity_bytes))
        if self.capacity_bytes <= 0:
            raise ValueError("shared Host arena capacity must be positive")
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        self._backing_fd = -1
        if self.backend == "memfd":
            fd = _create_agentic_host_memfd()
            self.path = f"/proc/{os.getpid()}/fd/{fd}"
        else:
            self.path = os.path.join(
                self.directory, f"preallocated-arena-{os.getpid()}.kv"
            )
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        mapping = None
        try:
            os.ftruncate(fd, self.capacity_bytes)
            # Reserve physical backing before publishing the arena. Capacity
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
            if self.backend == "tmpfs":
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
            raise
        else:
            if self.backend == "memfd":
                self._backing_fd = fd
            else:
                os.close(fd)

        # First-touch while this P process is NUMA-bound. The backing object
        # retains the pages after this temporary mapping is closed.
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
            if self.backend == "memfd":
                os.close(self._backing_fd)
                self._backing_fd = -1
            else:
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
            if self.backend == "memfd":
                if self._backing_fd >= 0:
                    os.close(self._backing_fd)
                    self._backing_fd = -1
            else:
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
            self._closed = True
