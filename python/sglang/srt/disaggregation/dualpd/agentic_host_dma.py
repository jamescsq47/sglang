"""CUDA memcpy ABI and process-local handles for fenced Host DMA.

No snapshot ownership, arena allocation, workset admission or queue policy lives
here. Callers retain returned descriptor storage until their existing CUDA
completion fence; submission errors must not be replayed through a fallback.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import threading
from typing import Any

import numpy as np
import torch


_CUDA_RUNTIME = None
_CUDA_RUNTIME_LOCK = threading.Lock()
_CUDA_DRIVER_BATCH = None
_CUDA_DRIVER_BATCH_LOCK = threading.Lock()


class _CUDAMemLocation(ctypes.Structure):
    _fields_ = (("type", ctypes.c_int), ("id", ctypes.c_int))


class _CUDAMemcpyAttributes(ctypes.Structure):
    _fields_ = (
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _CUDAMemLocation),
        ("dstLocHint", _CUDAMemLocation),
        ("flags", ctypes.c_uint),
    )


def _cuda_runtime():
    """Return libcudart with the narrow async memcpy ABI configured.

    ``torch.cuda.cudart()`` intentionally exposes only a subset of the CUDA
    runtime in current PyTorch builds and does not include cudaMemcpyAsync.
    Load the same process runtime directly so registered subranges do not go
    through PyTorch's whole-storage pinned-memory classification.
    """

    global _CUDA_RUNTIME
    if _CUDA_RUNTIME is not None:
        return _CUDA_RUNTIME
    with _CUDA_RUNTIME_LOCK:
        if _CUDA_RUNTIME is not None:
            return _CUDA_RUNTIME
        candidates = []
        try:
            from importlib.resources import files

            candidates.append(
                str(files("nvidia.cuda_runtime.lib").joinpath("libcudart.so.12"))
            )
        except (ImportError, ModuleNotFoundError, TypeError):
            pass
        discovered = ctypes.util.find_library("cudart")
        if discovered:
            candidates.append(discovered)
        candidates.extend(("libcudart.so.12", "libcudart.so"))
        errors = []
        for candidate in candidates:
            try:
                runtime = ctypes.CDLL(candidate)
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            runtime.cudaMemcpyAsync.argtypes = (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            )
            runtime.cudaMemcpyAsync.restype = ctypes.c_int
            _CUDA_RUNTIME = runtime
            return runtime
        raise RuntimeError(
            "unable to load libcudart for registered Host DMA: " + "; ".join(errors)
        )


def _cuda_memcpy_async(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    byte_size: int,
    kind: int,
    stream,
) -> None:
    """Submit one non-crossing registered-memory DMA to ``stream``."""

    result = _cuda_runtime().cudaMemcpyAsync(
        destination.data_ptr(),
        source.data_ptr(),
        int(byte_size),
        int(kind),
        int(stream.cuda_stream),
    )
    if result != 0:
        raise RuntimeError(f"cudaMemcpyAsync returned cudaError={result}")


def _cuda_driver_batch_memcpy():
    """Return Hopper's SM-free batched pointer-copy entry point if usable.

    The CUDA 13 symbol can be exported by an A100 driver, but the disjoint
    registered-Host batches used here do not execute safely on Ampere. Keep
    the ordinary fenced copy/gather implementation there; this batch path is
    an optimization and never part of the ownership protocol.
    """

    if torch.cuda.get_device_capability()[0] < 9:
        return None

    global _CUDA_DRIVER_BATCH
    if _CUDA_DRIVER_BATCH is False:
        return None
    if _CUDA_DRIVER_BATCH is not None:
        return _CUDA_DRIVER_BATCH
    with _CUDA_DRIVER_BATCH_LOCK:
        if _CUDA_DRIVER_BATCH is False:
            return None
        if _CUDA_DRIVER_BATCH is not None:
            return _CUDA_DRIVER_BATCH
        try:
            driver = ctypes.CDLL("libcuda.so.1")
            # Bind the explicitly versioned ABI.  The unversioned CUDA 13.1
            # driver symbol already has a ninth ``failIdx`` argument while
            # the public CUDA 13 API and ``_v2`` ABI have eight arguments.
            function = driver.cuMemcpyBatchAsync_v2
        except (OSError, AttributeError):
            _CUDA_DRIVER_BATCH = False
            return None
        function.argtypes = (
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(_CUDAMemcpyAttributes),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        function.restype = ctypes.c_int
        # Keep the CDLL alive through the bound function object.
        function._agentic_driver = driver
        _CUDA_DRIVER_BATCH = function
        return function


def _cuda_batch_memcpy_async(
    destinations,
    sources,
    sizes,
    *,
    stream,
) -> tuple[Any, ...] | bool:
    """Submit disjoint pointer copies as one copy-engine batch.

    Returns ``False`` only when the running NVIDIA driver predates the batch
    API, before any work is submitted.  A CUDA error after calling the API is
    raised so the caller's launch fence drains/quarantines a possible partial
    submission instead of unsafely replaying it through another path.
    """

    function = _cuda_driver_batch_memcpy()
    if function is None:
        return False
    if len(destinations) == 0:
        return True
    if not (len(destinations) == len(sources) == len(sizes)):
        raise ValueError("CUDA batch memcpy arrays have different lengths")
    destination_array = np.ascontiguousarray(destinations, dtype=np.uint64)
    source_array = np.ascontiguousarray(sources, dtype=np.uint64)
    size_array = np.ascontiguousarray(sizes, dtype=np.uintp)
    count = int(destination_array.size)
    # STREAM is the conservative ordering contract for both GPU and registered
    # Host sources.  The overlap flag is intentionally zero: the H100 driver
    # already overlaps copy-engine traffic with compute, while the optional
    # hint rejects large mixed Host/device batches on current CUDA 13.1.
    attributes = _CUDAMemcpyAttributes(
        1,
        _CUDAMemLocation(0, 0),
        _CUDAMemLocation(0, 0),
        0,
    )
    attribute_indices = (ctypes.c_size_t * 1)(0)
    result = function(
        destination_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        source_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        size_array.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        count,
        ctypes.byref(attributes),
        attribute_indices,
        1,
        int(stream.cuda_stream),
    )
    if result != 0:
        raise RuntimeError(f"cuMemcpyBatchAsync_v2 returned CUresult={result}")
    # Retain descriptor storage through the CUDA completion event.  Current
    # drivers consume it during the API call, but the stronger lifetime costs
    # almost nothing and avoids depending on an undocumented implementation
    # detail of this new API.
    return (
        destination_array,
        source_array,
        size_array,
        attributes,
        attribute_indices,
    )

