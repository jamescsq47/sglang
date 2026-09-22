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
        assert not store._arrival_events.exists()


def test_remote_idle_poll_never_rereads_retained_history(tmp_path, remote_poll, monkeypatch):
    store = early.AgenticEarlyClaimStore(str(tmp_path))
    for i in range(1000):
        store.publish_arrival(RequestGeneration("old", i), arrived_at=time.time() - 600)
    watchers = [store.watch_arrivals(max_age_seconds=5) for _ in range(8)]
    reads = []
    original = store.read_arrival_path
    monkeypatch.setattr(store, "read_arrival_path", lambda p, **kw: (reads.append(p), original(p, **kw))[1])
    monkeypatch.setattr(store, "iter_arrivals", lambda **kw: pytest.fail("historical rescan"))
    request = RequestGeneration("fresh", 0)
    try:
        for watcher in watchers:
            assert watcher.poll() == []
        assert reads == []
        store.publish_arrival(request, prompt_token_count=128)
        for watcher in watchers:
            watcher._shared_poll.next_scan = 0
            assert [r for r, _ in watcher.poll()] == [request]
            watcher._shared_poll.next_scan = 0
            assert watcher.poll() == []
        assert len(reads) == 8  # One NEW marker per rank, not 1000 old files.
    finally:
        for watcher in watchers:
            watcher.close()


def test_remote_concurrent_publishers_and_bounded_batches(tmp_path, remote_poll):
    from concurrent.futures import ThreadPoolExecutor
    store = early.AgenticEarlyClaimStore(str(tmp_path))
    with store.watch_arrivals(max_age_seconds=60) as watcher:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: store.publish_arrival(RequestGeneration("burst", i)), range(300)))
        first = watcher.poll()
        assert len(first) == 256
        watcher._shared_poll.next_scan = 0
        second = watcher.poll()
        assert len(second) == 44
        assert len({r for r, _ in first + second}) == 300


def test_remote_partial_event_is_not_consumed_and_writer_repairs_tail(tmp_path, remote_poll):
    store = early.AgenticEarlyClaimStore(str(tmp_path))
    with store.watch_arrivals(max_age_seconds=60) as watcher:
        with store._arrival_events.open("ab") as f:
            f.write(b"partial")
        assert watcher.poll() == []
        assert watcher._journal_cursor == 0
        request = RequestGeneration("after-crashed-publisher", 0)
        store.publish_arrival(request)
        watcher._shared_poll.next_scan = 0
        assert [r for r, _ in watcher.poll()] == [request]


def test_remote_startup_scan_race_is_replayed_once(tmp_path, remote_poll, monkeypatch):
    store = early.AgenticEarlyClaimStore(str(tmp_path))
    request = RequestGeneration("startup-race", 0)
    original = store.iter_arrivals
    def scan(**kw):
        store.publish_arrival(request)
        return original(**kw)
    monkeypatch.setattr(store, "iter_arrivals", scan)
    with store.watch_arrivals(max_age_seconds=60) as watcher:
        assert [r for r, _ in watcher.poll()] == [request]
        watcher._shared_poll.next_scan = 0
        assert watcher.poll() == []


def test_host_journal_only_returns_changes_and_coalesces(tmp_path, remote_poll):
    import hashlib
    journal = early.SharedDirectoryEventJournal(tmp_path)
    path = tmp_path / (hashlib.sha256(b"snapshot").hexdigest() + ".json")
    journal.publish(path)  # historical event covered by startup resync
    with early.AgenticDirectoryChangeWatcher(tmp_path, journal=journal) as watcher:
        assert watcher.poll(0) == ((), False)
        journal.publish(path)
        journal.publish(path)
        assert watcher.poll(0.2) == ((path,), False)
        assert watcher.poll(0.2) == ((), False)


def test_host_journal_concurrent_writers_partial_tail_and_reset(tmp_path):
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    journal = early.SharedDirectoryEventJournal(tmp_path)
    cursors = journal.cursors()
    paths = [tmp_path / (hashlib.sha256(str(i).encode()).hexdigest() + ".json") for i in range(500)]
    with journal.paths[0].open("ab") as stream:
        stream.write(b"partial")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(journal.publish, paths))
    seen = set()
    while True:
        changed, cursors, reset = journal.read(cursors)
        assert not reset
        seen.update(changed)
        if not changed:
            break
    assert seen == set(paths)
    with journal.paths[0].open("wb"):
        pass
    _, _, reset = journal.read(cursors)
    assert reset  # consumer must reconcile a truncated/rotated epoch


def test_host_journal_reader_does_not_wait_for_writer(tmp_path):
    import fcntl
    journal = early.SharedDirectoryEventJournal(tmp_path)
    cursors = journal.cursors()
    with journal.paths[0].open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        assert journal.read(cursors) == ((), cursors, False)
