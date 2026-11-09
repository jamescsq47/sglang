"""Physical buffer aliases, CPU copies and extent/failure lifetimes."""
import pickle
import tempfile
import threading
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation import agentic_host_buffers as buffers


def test_historical_physical_aliases_and_logger_are_identical():
    from sglang.srt.disaggregation import agentic_host_staging as staging

    for name in ("_HOST_LIBC", "_HOST_MEMMOVE", "_HOST_MADVISE", "_HOST_MEMSET",
                 "_HOST_MEMFD_CREATE", "_MADV_POPULATE_WRITE", "_MFD_CLOEXEC",
                 "_CUDA_MEMCPY_HOST_TO_DEVICE", "_CUDA_MEMCPY_DEVICE_TO_HOST",
                 "_registered_indexed_batch_copy", "_device_indices_to_host",
                 "_create_agentic_host_memfd", "_copy_layer_first_host_range",
                 "HostCopyWorkerPool", "H2DLaunchFence", "_concrete_kv_pool_device",
                 "LayerFirstD2HStaging", "SharedMHAHostSnapshot", "PinnedMHAHostBounce",
                 "LazySharedMHAHostSnapshot", "SharedHostSnapshotArena"):
        assert getattr(staging, name) is getattr(buffers, name)
    assert buffers.logger is staging.logger


def test_launch_fence_defaults_keepalive_lists_and_old_pickle_path():
    first, second = buffers.H2DLaunchFence(1), buffers.H2DLaunchFence(2)
    assert not first.submitted and not first.armed and not first.unavailable
    first.copy_refs.append("held")
    assert second.copy_refs == []
    legacy = pickle.dumps(first, protocol=0).replace(
        b"sglang.srt.disaggregation.agentic_host_buffers\n",
        b"sglang.srt.disaggregation.agentic_host_staging\n")
    restored = pickle.loads(legacy)
    assert type(restored) is buffers.H2DLaunchFence
    assert restored.event == 1 and restored.copy_refs == ["held"]


def test_cpu_copy_handles_layer_padding_and_nonzero_offsets():
    source = torch.arange(2 * 3 * 11 * 2 * 4, dtype=torch.int32).view(2, 3, 11, 2, 4)
    destination_backing = torch.full((2, 3, 14, 2, 4), -1, dtype=torch.int32)
    destination = destination_backing[:, :, 1:13]
    buffers._copy_layer_first_host_range(destination, source,
        destination_start=4, source_start=2, token_count=5)
    assert torch.equal(destination[:, :, 4:9], source[:, :, 2:7])
    assert (destination_backing[:, :, :5] == -1).all()
    assert (destination_backing[:, :, 10:] == -1).all()


@pytest.mark.parametrize("start,count", [(-1, 1), (3, 2), (0, -1)])
def test_cpu_copy_rejects_invalid_ranges_without_writes(start, count):
    source = torch.ones((2, 1, 4, 1, 4))
    destination = torch.zeros_like(source)
    with pytest.raises(ValueError):
        buffers._copy_layer_first_host_range(destination, source,
            destination_start=start, source_start=0, token_count=count)
    assert not destination.any()


def test_copy_worker_retains_job_through_physical_completion_and_survives_error():
    pool = buffers.HostCopyWorkerPool("buffer-cpu-test", 1)
    entered, release = threading.Event(), threading.Event()
    held = object()

    def work(value):
        assert value is held
        entered.set()
        assert release.wait(3)

    future = pool.submit(work, held)
    assert entered.wait(3) and not future.done()
    release.set()
    assert future.result(timeout=3) >= 0

    def fail():
        raise RuntimeError("copy failure")

    with pytest.raises(RuntimeError, match="copy failure"):
        pool.submit(fail).result(timeout=3)
    assert pool.submit(lambda: None).result(timeout=3) >= 0


def test_copy_worker_does_not_execute_cancelled_queued_job():
    pool = buffers.HostCopyWorkerPool("buffer-cancel-test", 1)
    entered, release = threading.Event(), threading.Event()

    def wait():
        entered.set()
        assert release.wait(3)

    first = pool.submit(wait)
    assert entered.wait(3)
    calls = []
    cancelled = pool.submit(lambda: calls.append(True))
    assert cancelled.cancel()
    release.set()
    first.result(timeout=3)
    pool.submit(lambda: None).result(timeout=3)
    assert cancelled.cancelled() and calls == []


@pytest.fixture
def arena():
    with tempfile.TemporaryDirectory(prefix="dualpd-buffer-test-", dir="/dev/shm") as directory:
        value = buffers.SharedHostSnapshotArena(directory, 32768, backend="memfd")
        try:
            yield value
        finally:
            value.close()


def test_extent_recycle_and_stale_release_preserve_new_owner(arena):
    first = arena.create("first", 64, NS(), 4096)
    second = arena.create("second", 64, NS(), 4096)
    assert arena.used_bytes == 8192
    assert arena.release(first)
    third = arena.create("third", 64, NS(), 4096)
    assert third.offset == first.offset
    assert arena.release(first) and arena.used_bytes == 8192
    assert id(third) in arena._active_extents
    assert arena.release(second) and arena.release(third)
    assert arena._free_extents == [(0, arena.capacity_bytes)]


def test_extent_constructor_failure_returns_complete_reservation(arena, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("view construction failed")

    monkeypatch.setattr(buffers, "LazySharedMHAHostSnapshot", fail)
    with pytest.raises(RuntimeError, match="view construction failed"):
        arena.create("failed", 64, NS(), 4096)
    assert arena.used_bytes == 0 and arena._active_extents == {}
    assert arena._free_extents == [(0, arena.capacity_bytes)]


def test_extent_close_failure_keeps_lease_and_capacity_until_retry(arena, monkeypatch):
    snapshot = arena.create("held", 64, NS(), 4096)
    original = snapshot.close

    def fail(**kwargs):
        raise RuntimeError("unregister not complete")

    monkeypatch.setattr(snapshot, "close", fail)
    assert arena.release(snapshot) is False
    assert arena.used_bytes == 4096 and id(snapshot) in arena._active_extents
    monkeypatch.setattr(snapshot, "close", original)
    assert arena.release(snapshot) and arena.used_bytes == 0
    assert arena.release(snapshot) and arena.used_bytes == 0


def test_lazy_failed_state_binding_closes_unpublished_view(arena, monkeypatch):
    snapshot = arena.create("hybrid", 64, NS(mamba_pool=object()), 4096)
    snapshot.set_state_indices(torch.tensor([1]))
    closed = []

    def fail(indices):
        raise RuntimeError("state installation failed")

    monkeypatch.setattr(buffers, "SharedMHAHostSnapshot", lambda **kwargs: NS(
        set_state_indices=fail, close=lambda **kwargs: closed.append(kwargs)))
    with pytest.raises(RuntimeError, match="state installation failed"):
        snapshot.materialize()
    assert snapshot._materialized is None and snapshot._pending_state_indices == (1,)
    assert closed == [{"unlink": False}]
    assert arena.used_bytes == 4096
    assert arena.release(snapshot) and arena.used_bytes == 0


def test_snapshot_unregister_failure_keeps_mapping_for_retry(monkeypatch):
    snapshot = object.__new__(buffers.SharedMHAHostSnapshot)
    raw, kv = NS(data_ptr=lambda: 42), object()
    closed = []
    snapshot.__dict__.update(_closed=False, _cuda_host_registered=True,
        _arena_mapping=None, _raw=raw, kv_buffer=kv, device_pool=object(),
        _owns_mapping=True, mapping=NS(close=lambda: closed.append(True)))
    runtime = NS(cudaError=NS(success=0), result=7)
    runtime.cudaHostUnregister = lambda _: runtime.result
    monkeypatch.setattr(buffers, "_concrete_kv_pool_device", lambda _: "cuda:0")
    monkeypatch.setattr(buffers.torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(buffers.torch.cuda, "cudart", lambda: runtime)
    with pytest.raises(RuntimeError, match="cudaHostUnregister returned 7"):
        snapshot.close()
    assert not snapshot._closed and snapshot._cuda_host_registered
    assert snapshot._raw is raw and snapshot.kv_buffer is kv and not closed
    runtime.result = 0
    snapshot.close()
    snapshot.close()
    assert snapshot._closed and not snapshot._cuda_host_registered
    assert snapshot._raw is None and snapshot.kv_buffer is None and closed == [True]


def test_registration_failure_selects_fallback_before_any_dma(monkeypatch):
    snapshot = object.__new__(buffers.SharedMHAHostSnapshot)
    calls = []

    def fail(*args):
        calls.append(args)
        raise MemoryError("pin capacity")

    snapshot.__dict__.update(_registered_dma_enabled=True,
        _cuda_host_registered=False, _cuda_host_registration_failed=False,
        _cuda_host_registration_lock=threading.Lock(), _registered_windows=(),
        _arena_mapping=NS(acquire=fail), file_offset=0, byte_size=4096,
        device_pool=object(), path="unmapped-test-extent")
    monkeypatch.setattr(buffers, "_concrete_kv_pool_device", lambda _: "cuda:0")
    assert snapshot.prepare_registered_host_dma() is False
    assert snapshot.prepare_registered_host_dma() is False
    assert len(calls) == 1 and snapshot._cuda_host_registration_failed
    assert not snapshot._cuda_host_registered and snapshot._registered_windows == ()
