"""CPU-only regression tests: shared-control discovery, never GPU ownership."""

import threading
import time

import pytest

from sglang.srt.disaggregation import agentic_early_claim as early
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration


@pytest.fixture
def remote_poll(monkeypatch):
    monkeypatch.setattr(early, "_shared_control_poller", lambda: early._SharedControlPoller(0.05))
    def no_local_events():
        raise AssertionError("remote control must not depend on local inotify")
    monkeypatch.setattr(early, "_inotify_init", no_local_events)


def test_remote_arrival_after_start_and_retarget(tmp_path, remote_poll):
    store = early.AgenticEarlyClaimStore(str(tmp_path))
    request = RequestGeneration("remote-trajectory", 1)
    with store.watch_arrivals(max_age_seconds=60) as watcher:
        assert watcher.poll() == []
        store.publish_arrival(request, target_prefill_domain=0, prompt_token_count=128)
        assert watcher.poll() == []  # bounded: no hot full-directory scan
        arrivals = watcher.poll(0.2)
        assert len(arrivals) == 1
        assert arrivals[0][0] == request
        assert watcher.poll(0.2) == []  # an unchanged marker is not a new arrival
        store.publish_arrival(request, target_prefill_domain=1, prompt_token_count=256)
        changed = watcher.poll(0.2)
        assert len(changed) == 1
        assert changed[0][1]["prompt_token_count"] == 256
    assert watcher.poll() == []


def test_remote_arrival_deleted_and_recreated(tmp_path, remote_poll):
    store = early.AgenticEarlyClaimStore(str(tmp_path))
    request = RequestGeneration("recreated", 2)
    with store.watch_arrivals(max_age_seconds=60) as watcher:
        store.publish_arrival(request, prompt_token_count=128)
        assert len(watcher.poll()) == 1
        store.remove_arrival(request)
        assert watcher.poll(0.2) == []
        store.publish_arrival(request, prompt_token_count=128)
        assert len(watcher.poll(0.2)) == 1


def test_remote_file_and_directory_use_authoritative_resync(tmp_path, remote_poll):
    with early.AgenticFileChangeWatcher(tmp_path / "ledger.json") as watcher:
        assert watcher.poll(0) is True
        assert watcher.poll(0) is False
        assert watcher.poll(0.2) is True
    assert watcher.poll(0) is False
    with early.AgenticDirectoryChangeWatcher(tmp_path / "events") as watcher:
        assert watcher.poll(0) == ((), True)
        assert watcher.poll(0) == ((), False)
        assert watcher.poll(0.2) == ((), True)
    assert watcher.poll(0) == ((), False)


def test_close_wakes_remote_poll_without_waiting_for_tick(tmp_path, remote_poll):
    watcher = early.AgenticFileChangeWatcher(tmp_path / "ledger.json")
    watcher.poll(0)
    watcher._shared_poll.next_scan = time.monotonic() + 60
    result = []
    thread = threading.Thread(target=lambda: result.append(watcher.poll(None)))
    thread.start()
    watcher.close()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert result == [False]


def test_default_single_node_still_uses_inotify(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_MULTINODE_ENABLED", raising=False)
    store = early.AgenticEarlyClaimStore(str(tmp_path))
    with store.watch_arrivals(max_age_seconds=60) as watcher:
        assert watcher._shared_poll is None
        assert watcher.fd >= 0
        request = RequestGeneration("local", 1)
        store.publish_arrival(request, prompt_token_count=128)
        assert any(item[0] == request for item in watcher.poll(0.2))
