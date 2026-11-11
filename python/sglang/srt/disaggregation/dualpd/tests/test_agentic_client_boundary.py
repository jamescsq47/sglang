"""CPU-only public application ACK and read-only observation contracts."""
import gc
import json
import os
from pathlib import Path
import select
import signal
import threading

import pytest

from sglang.srt.disaggregation import agentic_host_observation as observation
from sglang.srt.disaggregation.agentic_client import publish_local_ack
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration


@pytest.mark.parametrize("event", ["tool", "final"])
def test_ack_uses_existing_marker_protocol(tmp_path, event):
    request = RequestGeneration("request:unicode-测试", 3)
    payload = publish_local_ack(str(tmp_path), request.request_id, 3, event=event)
    store = AgenticEarlyClaimStore(str(tmp_path / "early-claims"))
    reader = store.read_tool if event == "tool" else store.read_final
    assert reader(request, not_before=0, max_age_seconds=3600) == payload
    assert payload["kind"] == event
    assert payload["snapshot_id"] == request.snapshot_id
    assert len(list(tmp_path.rglob("*.json"))) == 1


@pytest.mark.parametrize("event,directory", [("invalid", "/unused"), ("tool", "")])
def test_invalid_ack_does_not_publish(event, directory):
    with pytest.raises(ValueError):
        publish_local_ack(directory, "r", 0, event=event)


@pytest.fixture
def observer(monkeypatch, tmp_path):
    monkeypatch.setenv("PD_DRAM_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(observation, "_arenas", {})
    monkeypatch.setattr(observation, "_thread_pid", None)
    monkeypatch.setattr(observation, "_registration_lock", threading.Lock())
    starts = []
    class Thread:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def start(self): starts.append(self.kwargs)
    monkeypatch.setattr(observation.threading, "Thread", Thread)
    return observation, starts


class Arena:
    used_bytes = 4
    capacity_bytes = 16
    committed_bytes = 16
    _closed = False
    path = "memory-only"


def test_observation_disabled_is_noop(observer, monkeypatch):
    module, starts = observer
    monkeypatch.delenv("PD_DRAM_TRACE_DIR")
    module.register_arena(Arena(), "d2p")
    assert not starts and not module._arenas


def test_direction_dedup_and_weak_reference(observer):
    module, starts = observer
    d2p, p2d = Arena(), Arena()
    module.register_arena(d2p, "d2p")
    module.register_arena(d2p, "d2p")
    module.register_arena(p2d, "p2d")
    rows = module.snapshot()["pools"]
    assert {row["direction"] for row in rows} == {"d2p", "p2d"}
    assert len(rows) == 2 and len(starts) == 1
    del d2p
    gc.collect()
    assert len(module.snapshot()["pools"]) == 1


def test_closed_arena_remains_diagnostic_and_state_is_untouched(observer):
    module, _ = observer
    arena = Arena()
    module.register_arena(arena, "p2d")
    arena._closed = True
    before = dict(arena.__dict__)
    row = module.snapshot()["pools"][0]
    assert row["closed"] and row["capacity_bytes"] == 16
    assert arena.__dict__ == before


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "bad"])
def test_observation_error_does_not_escape_or_start_writer(observer, monkeypatch, interval):
    module, starts = observer
    monkeypatch.setenv("PD_DRAM_SAMPLE_SECONDS", interval)
    module.register_arena(Arena(), "d2p")
    assert not starts and not module._arenas


def test_inconsistent_counter_is_error_not_zero(observer):
    module, _ = observer
    arena = Arena()
    arena.used_bytes = 17
    module.register_arena(arena, "d2p")
    row = module.snapshot()
    assert not row["pools"] and row["errors"]


def test_writer_retains_existing_record_schema(observer, monkeypatch, tmp_path):
    module, _ = observer
    arena = Arena()
    module.register_arena(arena, "d2p")
    class Stop(BaseException): pass
    monkeypatch.setattr(module.time, "sleep", lambda _: (_ for _ in ()).throw(Stop()))
    with pytest.raises(Stop):
        module._write_samples(str(tmp_path), 2)
    latest = json.loads((tmp_path / f"{os.getpid()}.latest.json").read_text())
    assert latest == json.loads((tmp_path / f"{os.getpid()}.jsonl").read_text())
    assert {"ts", "monotonic", "pid", "role", "port", "engine", "pools", "errors"} == set(latest)
    assert latest["pools"][0]["used_bytes"] == 4


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Linux fork isolation")
def test_real_fork_clears_inherited_registration_and_locked_observer_lock(observer):
    module, _ = observer
    arena = Arena()
    module.register_arena(arena, "d2p")
    original_lock = module._registration_lock
    original_lock.acquire()
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            empty = not module.snapshot()["pools"] and module._thread_pid is None
            child_arena = Arena()
            module.register_arena(child_arena, "p2d")
            os.write(write_fd, json.dumps(dict(empty=empty, count=len(module.snapshot()["pools"]))).encode())
        finally:
            os._exit(0)
    original_lock.release()
    os.close(write_fd)
    try:
        if not select.select([read_fd], [], [], 5)[0]:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("forked observer blocked on inherited parent lock")
        result = json.loads(os.read(read_fd, 4096))
        assert result == dict(empty=True, count=1)
    finally:
        os.close(read_fd)
        os.waitpid(pid, 0)
    assert len(module.snapshot()["pools"]) == 1
