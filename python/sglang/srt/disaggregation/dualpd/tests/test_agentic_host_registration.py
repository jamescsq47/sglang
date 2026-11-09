"""Window lifetime/capacity faults without CUDA submissions or model services."""
import ctypes
import json
import mmap
import os
import threading
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation import agentic_host_registration as registration


def test_historical_aliases_share_one_registry_and_lock():
    from sglang.srt.disaggregation import agentic_host_staging as staging

    names = ("_RegisteredHostArenaMapping", "_REGISTERED_HOST_ARENAS",
             "_REGISTERED_HOST_ARENAS_LOCK", "_REGISTERED_HOST_PREWARM_PINS",
             "_registered_host_arena", "_write_startup_prewarm_record",
             "start_registered_host_arena_startup_prewarm",
             "_is_memfd_backing_path", "_validate_shared_host_backing_path",
             "_open_shared_host_backing")
    for name in names:
        assert getattr(staging, name) is getattr(registration, name)
    assert registration.logger is staging.logger


@pytest.fixture
def cache(monkeypatch):
    calls = []
    runtime = NS(cudaError=NS(success=0), register_error=0, unregister_error=0)

    def register(address, size, flags):
        calls.append(("register", address, size, flags))
        return runtime.register_error

    def unregister(address):
        calls.append(("unregister", address))
        return runtime.unregister_error

    runtime.cudaHostRegister = register
    runtime.cudaHostUnregister = unregister
    monkeypatch.setattr(registration, "_REGISTERED_HOST_ARENAS", {})
    monkeypatch.setattr(registration, "_REGISTERED_HOST_ARENAS_LOCK", threading.Lock())
    monkeypatch.setattr(registration.torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(registration.torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB", str(128 / 1024**3))
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA", "0")

    def mapping(base=4096):
        arena = registration._RegisteredHostArenaMapping.__new__(registration._RegisteredHostArenaMapping)
        arena.raw = NS(data_ptr=lambda: base)
        arena.byte_size = 256
        arena.window_bytes = 64
        arena._windows = {}
        arena._users = 0
        arena._prewarm_started = False
        arena._prewarm_thread = None
        arena.path = f"/proc/1/fd/{base}"
        registration._REGISTERED_HOST_ARENAS[(0, base, arena.path)] = arena
        return arena

    return NS(mapping=mapping, calls=calls, runtime=runtime)


def test_overlapping_users_pin_windows_until_each_reference_returns(cache):
    arena = cache.mapping()
    first = arena.acquire(32, 64, "cuda:0")
    second = arena.acquire(64, 1, "cuda:0")
    assert first == (0, 64) and second == (64,)
    assert [arena._windows[x]["refs"] for x in (0, 64)] == [1, 2]
    arena.release(first)
    assert arena._windows[64]["refs"] == 1
    arena.release(second)
    assert all(x["refs"] == 0 for x in arena._windows.values())
    assert cache.calls == [("register", 4096, 64, 0), ("register", 4160, 64, 0)]


def test_capacity_failure_rolls_back_only_newly_acquired_refs(cache):
    arena = cache.mapping()
    held = arena.acquire(0, 64, "cuda:0")
    with pytest.raises(MemoryError, match="fully referenced"):
        arena.acquire(64, 128, "cuda:0")
    assert arena._windows[0]["refs"] == 1
    assert arena._windows[64]["refs"] == 0
    assert 128 not in arena._windows
    assert not any(call[0] == "unregister" for call in cache.calls)
    arena.release(held)


def test_only_zero_ref_lru_window_is_evicted_across_arenas(cache):
    first, second, third = (cache.mapping(base) for base in (4096, 8192, 12288))
    held = first.acquire(0, 64, "cuda:0")
    second.release(second.acquire(0, 64, "cuda:0"))
    first._windows[0]["used_at"] = 0
    third.acquire(0, 64, "cuda:0")
    assert ("unregister", 8192) in cache.calls
    assert ("unregister", 4096) not in cache.calls
    assert second._windows == {} and first._windows[0]["refs"] == 1
    first.release(held)


def test_unregister_failure_does_not_drop_cached_mapping(cache):
    arena = cache.mapping()
    arena.release(arena.acquire(0, 128, "cuda:0"))
    cache.runtime.unregister_error = 7
    with pytest.raises(RuntimeError, match="cudaHostUnregister returned 7"):
        arena.acquire(128, 64, "cuda:0")
    assert set(arena._windows) == {0, 64}
    assert all(item["refs"] == 0 for item in arena._windows.values())
    assert 128 not in arena._windows


def test_registration_failure_unwinds_prior_window_refs(cache, monkeypatch):
    arena = cache.mapping()
    calls = []

    def register(address, *_):
        calls.append(address)
        return 0 if len(calls) == 1 else 7

    monkeypatch.setattr(cache.runtime, "cudaHostRegister", register)
    with pytest.raises(RuntimeError, match="cudaHostRegister returned 7"):
        arena.acquire(0, 128, "cuda:0")
    assert calls == [4096, 4160]
    assert set(arena._windows) == {0} and arena._windows[0]["refs"] == 0


@pytest.mark.parametrize("windows", [(0,), (64,)])
def test_reference_underflow_is_rejected(cache, windows):
    arena = cache.mapping()
    arena.release(arena.acquire(0, 64, "cuda:0"))
    with pytest.raises(RuntimeError, match="reference underflow"):
        arena.release(windows)
    assert arena._windows[0]["refs"] == 0


def _memfd(name):
    # This deployed Python omits os.memfd_create; use the real libc/kernel
    # interface, just as the supported production fallback does. Never stub it.
    create = ctypes.CDLL(None, use_errno=True).memfd_create
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    fd = create(name.encode(), 1)
    if fd < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return fd


def test_memfd_mapping_authentication_and_owner_fd_lifetime():
    fd = _memfd("sglang-agentic-host-arena-registration-test")
    try:
        os.ftruncate(fd, mmap.ALLOCATIONGRANULARITY)
        path = f"/proc/{os.getpid()}/fd/{fd}"
        arena = registration._RegisteredHostArenaMapping(path)
        assert arena.byte_size == mmap.ALLOCATIONGRANULARITY
        arena.raw[0] = 73
        assert os.pread(fd, 1, 0) == b"I"
        arena.raw = None
        arena.mapping.close()
        assert os.fstat(fd).st_size == mmap.ALLOCATIONGRANULARITY
    finally:
        os.close(fd)


def test_non_agentic_memfd_rejected_without_closing_owner_fd():
    fd = _memfd("unrelated-object")
    try:
        with pytest.raises(ValueError, match="not an agentic memfd"):
            registration._open_shared_host_backing(f"/proc/{os.getpid()}/fd/{fd}", os.O_RDWR)
        os.fstat(fd)
    finally:
        os.close(fd)


def test_unsupported_path_rejected_before_open(monkeypatch):
    monkeypatch.setattr(registration.os, "open", lambda *_: pytest.fail("must validate before opening"))
    with pytest.raises(ValueError, match="shared Host snapshot"):
        registration._open_shared_host_backing("/etc/passwd", os.O_RDWR)


@pytest.mark.parametrize("users,refs", [(1, 0), (0, 1), (0, 0)])
def test_stale_inode_sweep_respects_mapping_and_dma_references(cache, monkeypatch, users, refs):
    old = cache.mapping()
    old._users = users
    old._windows = {0: {"refs": refs, "bytes": 64, "used_at": 0}}
    closed = []
    old.mapping = NS(close=lambda: closed.append(True))
    fresh = NS(_users=0, prewarm=lambda device: None)
    monkeypatch.setattr(registration, "_open_shared_host_backing", lambda *_: 100)
    monkeypatch.setattr(registration.os, "fstat", lambda _: NS(st_dev=2, st_ino=3))
    monkeypatch.setattr(registration.os, "close", lambda _: None)
    monkeypatch.setattr(registration.os, "stat", lambda _: NS(st_dev=2, st_ino=3))
    monkeypatch.setattr(registration, "_RegisteredHostArenaMapping", lambda _: fresh)
    assert registration._registered_host_arena("/proc/1/fd/new", "cuda:0") is fresh
    assert fresh._users == 1
    if users or refs:
        assert not closed and old.raw is not None and old._windows
        assert not any(x[0] == "unregister" for x in cache.calls)
    else:
        assert closed == [True] and old.raw is None and old._windows == {}
        assert cache.calls == [("unregister", 4096)]


def test_background_prewarm_failure_surfaces_without_hiding_cached_windows(cache, monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA", "1")
    arena = cache.mapping()
    cache.runtime.register_error = 7
    arena.prewarm("cuda:0")
    arena._prewarm_thread.join(timeout=2)
    assert not arena._prewarm_thread.is_alive()
    with pytest.raises(RuntimeError, match="prewarm failed"):
        arena.wait_prewarm(0)
    assert arena._windows == {}
    cache.runtime.register_error = 0
    arena.release(arena.acquire(0, 64, "cuda:0"))
    assert arena._windows[0]["refs"] == 0


def test_startup_failure_publishes_failed_not_complete(monkeypatch, tmp_path):
    values = {"REGISTER_STARTUP_BARRIER": "1", "REGISTER_EAGER_ARENA": "1",
              "REGISTER_PREWARM_DIR": str(tmp_path), "PREFILL_DOMAIN_COUNT": "1",
              "PREFILL_DOMAIN": "0", "TP_SIZE": "1"}
    for key, value in values.items():
        monkeypatch.setenv("SGLANG_AGENTIC_KV_" + key, value)
    monkeypatch.setattr(registration, "_REGISTERED_HOST_PREWARM_PINS", [])

    def fail(*_):
        raise MemoryError("injected pin capacity failure")

    monkeypatch.setattr(registration, "_registered_host_arena", fail)
    (tmp_path / "start").touch()
    thread = registration.start_registered_host_arena_startup_prewarm(
        role="prefill", engine_id="p0", tp_rank=0, device="cuda:0", d2p_arena_path="host0"
    )
    thread.join(timeout=2)
    assert not thread.is_alive()
    failed = json.loads(next((tmp_path / "failed").glob("*.json")).read_text())
    assert "injected pin capacity failure" in failed["error"]
    assert not list((tmp_path / "complete").glob("*.json"))
    assert registration._REGISTERED_HOST_PREWARM_PINS == []
