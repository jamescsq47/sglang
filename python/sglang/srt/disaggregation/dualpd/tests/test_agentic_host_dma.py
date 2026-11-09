"""CPU fault/ABI checks; fake drivers never submit CUDA work."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from sglang.srt.disaggregation import agentic_host_dma as dma


def test_historical_transport_helpers_are_canonical_aliases():
    from sglang.srt.disaggregation import agentic_host_staging as staging

    for name in ("_CUDAMemLocation", "_CUDAMemcpyAttributes", "_cuda_runtime",
                 "_cuda_memcpy_async", "_cuda_driver_batch_memcpy", "_cuda_batch_memcpy_async"):
        assert getattr(staging, name) is getattr(dma, name)


def test_ampere_never_opens_batch_driver(monkeypatch):
    monkeypatch.setattr(dma.torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(dma.ctypes, "CDLL", lambda *_: pytest.fail("Ampere must use existing fallback"))
    assert dma._cuda_driver_batch_memcpy() is None


def test_unavailable_batch_driver_is_cached_without_submission(monkeypatch):
    calls = []
    def missing(name):
        calls.append(name)
        raise OSError("no batch driver")
    monkeypatch.setattr(dma.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(dma, "_CUDA_DRIVER_BATCH", None)
    monkeypatch.setattr(dma.ctypes, "CDLL", missing)
    assert dma._cuda_driver_batch_memcpy() is None
    assert dma._cuda_driver_batch_memcpy() is None
    assert calls == ["libcuda.so.1"]


def test_versioned_eight_argument_driver_abi_keeps_library_alive(monkeypatch):
    def batch(*_):
        pytest.fail("binding must not submit a transfer")
    driver = SimpleNamespace(cuMemcpyBatchAsync_v2=batch)
    monkeypatch.setattr(dma.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(dma, "_CUDA_DRIVER_BATCH", None)
    monkeypatch.setattr(dma.ctypes, "CDLL", lambda _: driver)
    assert dma._cuda_driver_batch_memcpy() is batch
    assert len(batch.argtypes) == 8
    assert batch.restype is ctypes.c_int
    assert batch._agentic_driver is driver
    assert dma._cuda_driver_batch_memcpy() is batch


def test_batch_unavailable_returns_false_before_submission(monkeypatch):
    monkeypatch.setattr(dma, "_cuda_driver_batch_memcpy", lambda: None)
    assert dma._cuda_batch_memcpy_async([1], [2], [8], stream=None) is False


def test_empty_or_mismatched_descriptors_do_not_submit(monkeypatch):
    monkeypatch.setattr(dma, "_cuda_driver_batch_memcpy", lambda: lambda *_: pytest.fail("unexpected DMA"))
    assert dma._cuda_batch_memcpy_async([], [], [], stream=None) is True
    with pytest.raises(ValueError, match="different lengths"):
        dma._cuda_batch_memcpy_async([1], [2, 3], [8], stream=None)


def test_batch_descriptors_and_storage_lifetime(monkeypatch):
    calls = []
    def batch(dest, source, sizes, count, attributes, indices, attribute_count, stream):
        attrs = ctypes.cast(attributes, ctypes.POINTER(dma._CUDAMemcpyAttributes)).contents
        calls.append((list(dest[:count]), list(source[:count]), list(sizes[:count]),
                      attrs.srcAccessOrder, attrs.flags, indices[0], attribute_count, stream))
        return 0
    monkeypatch.setattr(dma, "_cuda_driver_batch_memcpy", lambda: batch)
    keepalive = dma._cuda_batch_memcpy_async([4096, 8192], [32, 64], [8, 16],
                                           stream=SimpleNamespace(cuda_stream=1234))
    assert calls == [([4096, 8192], [32, 64], [8, 16], 1, 0, 0, 1, 1234)]
    assert len(keepalive) == 5
    assert [array.dtype for array in keepalive[:3]] == [np.dtype("uint64"), np.dtype("uint64"), np.dtype(np.uintp)]
    assert keepalive[0].tolist() == [4096, 8192]
    assert isinstance(keepalive[3], dma._CUDAMemcpyAttributes)


def test_batch_submission_error_raises_without_fallback_or_retry(monkeypatch):
    calls = []
    def fail(*args):
        calls.append(args)
        return 7
    monkeypatch.setattr(dma, "_cuda_driver_batch_memcpy", lambda: fail)
    with pytest.raises(RuntimeError, match="cuMemcpyBatchAsync_v2 returned CUresult=7"):
        dma._cuda_batch_memcpy_async([1], [2], [8], stream=SimpleNamespace(cuda_stream=3))
    assert len(calls) == 1


@pytest.mark.parametrize("result", [0, 13])
def test_runtime_memcpy_arguments_and_error_fence(monkeypatch, result):
    calls = []
    def copy(*args):
        calls.append(args)
        return result
    monkeypatch.setattr(dma, "_cuda_runtime", lambda: SimpleNamespace(cudaMemcpyAsync=copy))
    args = (SimpleNamespace(data_ptr=lambda: 44), SimpleNamespace(data_ptr=lambda: 88))
    kw = dict(byte_size=16, kind=2, stream=SimpleNamespace(cuda_stream=17))
    if result:
        with pytest.raises(RuntimeError, match="cudaError=13"):
            dma._cuda_memcpy_async(*args, **kw)
    else:
        dma._cuda_memcpy_async(*args, **kw)
    assert calls == [(44, 88, 16, 2, 17)]


def test_runtime_binding_is_narrow_and_cached(monkeypatch):
    def copy(*_):
        pytest.fail("binding must not submit a transfer")
    runtime = SimpleNamespace(cudaMemcpyAsync=copy)
    opened = []
    def load(name):
        opened.append(name)
        return runtime
    monkeypatch.setattr(dma, "_CUDA_RUNTIME", None)
    monkeypatch.setattr(dma.ctypes, "CDLL", load)
    monkeypatch.setattr(dma.ctypes.util, "find_library", lambda _: None)
    assert dma._cuda_runtime() is runtime
    assert dma._cuda_runtime() is runtime
    assert len(opened) == 1
    assert len(copy.argtypes) == 5
    assert copy.restype is ctypes.c_int
