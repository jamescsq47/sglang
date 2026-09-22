"""CPU-only ownership barriers for asynchronous P->D source retirement."""

import threading
from collections import deque
from types import SimpleNamespace

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin as Prefill


def _owner(group_status, host=None):
    lock = threading.RLock()
    stop = threading.Event()
    request = SimpleNamespace(rid="request", bootstrap_room=11)
    bridge = SimpleNamespace(allow_cleanup_owner=lambda owner, state_lock: None)
    broker = SimpleNamespace(
        native_bridge=bridge,
        allow_native_cleanup_owner=lambda owner, state_lock: None,
    )
    scheduler = Prefill.__new__(Prefill)
    scheduler.tp_size = 2 if group_status is not None else 1
    scheduler.tp_rank = 0
    scheduler.agentic_p_workset_broker = broker
    scheduler.agentic_p2d_host_staging_manager = host
    scheduler.agentic_tp_p2d_cleanup_mailbox = SimpleNamespace(
        transfer_group_status=group_status or (lambda key: (None, False)),
        publish_local=lambda key, status: None,
    )
    scheduler._prefill_native_state_lock = lock
    scheduler._prefill_native_release_condition = threading.Condition()
    scheduler._prefill_native_release_queue = deque()
    scheduler._prefill_transfer_poll_lock = threading.Lock()
    scheduler._prefill_transfer_terminal_queue = deque()
    scheduler._prefill_transfer_interval = 0.001
    scheduler._prefill_transfer_stop = stop
    return scheduler, request, lock, stop


def test_tp_release_waits_for_all_rank_terminal(monkeypatch):
    polls = iter((None, int(KVPoll.Success)))
    scheduler, request, lock, stop = _owner(
        lambda key: (next(polls, int(KVPoll.Success)), False)
    )
    released = []

    def release(self, req):
        assert lock._is_owned()
        released.append(req)
        stop.set()

    monkeypatch.setattr(Prefill, "_release_prefill_native_success", release)
    Prefill._enqueue_prefill_native_release(scheduler, request, int(KVPoll.Success))
    Prefill._enqueue_prefill_native_release(scheduler, request, int(KVPoll.Success))
    assert len(scheduler._prefill_native_release_queue) == 1
    Prefill._prefill_native_release_worker(scheduler)
    assert released == [request]
    assert request._agentic_p2d_native_release_done == int(KVPoll.Success)


def test_single_rank_host_claim_must_authorize_before_native_free(monkeypatch):
    calls = []

    def authorize(req):
        calls.append("authorize")
        return len(calls) > 1

    host = SimpleNamespace(prepare_scheduler_release=authorize)
    scheduler, request, lock, stop = _owner(None, host)

    def release(self, req):
        assert lock._is_owned()
        calls.append("release")
        stop.set()

    monkeypatch.setattr(Prefill, "_release_prefill_native_success", release)
    Prefill._enqueue_prefill_native_release(scheduler, request, int(KVPoll.Success))
    Prefill._prefill_native_release_worker(scheduler)
    assert calls == ["authorize", "authorize", "release"]
    assert list(scheduler._prefill_transfer_terminal_queue) == [
        Prefill._prefill_transfer_key(request)
    ]


def test_native_failure_keeps_source_quarantined(monkeypatch):
    scheduler, request, _, _ = _owner(lambda key: (int(KVPoll.Success), False))

    def failed(self, req):
        raise RuntimeError("native free failed")

    monkeypatch.setattr(Prefill, "_release_prefill_native_success", failed)
    Prefill._enqueue_prefill_native_release(scheduler, request, int(KVPoll.Success))
    Prefill._prefill_native_release_worker(scheduler)
    assert not hasattr(request, "_agentic_p2d_native_release_done")
    assert isinstance(request._agentic_p2d_native_release_error, RuntimeError)
    assert scheduler._prefill_native_release_fatal is request._agentic_p2d_native_release_error


def test_tp_group_failure_overrides_local_sender_success(monkeypatch):
    scheduler, request, lock, stop = _owner(
        lambda key: (int(KVPoll.Failed), False)
    )
    scheduler._prefill_transfer_async_release_enabled = True
    request._agentic_p2d_release_authorized = int(KVPoll.Success)
    events = []

    def failed(self, req, host, metadata, **kwargs):
        assert lock._is_owned()
        assert kwargs["clear_mailboxes"] is False
        events.append("failed")
        stop.set()
        return True

    def success(self, req):
        events.append("unsafe-success")

    monkeypatch.setattr(Prefill, "_cleanup_failed_prefill_transfer", failed)
    monkeypatch.setattr(Prefill, "_release_prefill_native_success", success)
    Prefill._enqueue_prefill_native_release(scheduler, request, int(KVPoll.Success))
    Prefill._prefill_native_release_worker(scheduler)
    assert events == ["failed"]
    assert request._agentic_p2d_native_release_done == int(KVPoll.Failed)


def test_host_copy_in_flight_and_shutdown_never_free_source(monkeypatch):
    attempts = []
    host = SimpleNamespace()
    scheduler, request, _, stop = _owner(None, host)

    def pending(req):
        attempts.append(req)
        stop.set()
        return False

    host.prepare_scheduler_release = pending
    released = []
    monkeypatch.setattr(
        Prefill, "_release_prefill_native_success",
        lambda self, req: released.append(req),
    )
    Prefill._enqueue_prefill_native_release(scheduler, request, int(KVPoll.Success))
    Prefill._prefill_native_release_worker(scheduler)
    assert attempts == [request]
    assert not released
    assert not hasattr(request, "_agentic_p2d_native_release_done")
    assert list(scheduler._prefill_native_release_queue) == [
        (request, int(KVPoll.Success))
    ]


def test_tp_completion_becomes_scheduler_visible_only_after_native_group_release():
    key = "request@11"
    cleared = []
    scheduler, _, _, _ = _owner(lambda key: (None, False))
    scheduler._prefill_native_release_enabled = True
    scheduler._prefill_native_group_pending = {key}
    scheduler._prefill_native_completed = {}
    scheduler._prefill_transfer_cleanup_lock = threading.Lock()
    scheduler._prefill_transfer_cleanup_pending = set()
    scheduler.agentic_tp_p2d_cleanup_mailbox.group_status = (
        lambda name: int(KVPoll.Success)
        if name == "native-released:" + key else None
    )
    scheduler.agentic_tp_p2d_cleanup_mailbox.clear_group = cleared.append
    assert Prefill._prefill_transfer_cleanup_once(scheduler) == 0
    assert scheduler._prefill_native_completed == {key: int(KVPoll.Success)}
    assert not scheduler._prefill_native_group_pending
    assert not cleared
