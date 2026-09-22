"""Real CPU ledger/controller checks; no scheduler ticks, GPU or transport."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time

import pytest

from sglang.srt.disaggregation.agentic_workset_controller import (
    ControllerQueueFull, ControllerStopped, PendingWorksetCancelled,
    WorksetController, WorksetIntent,
)
from sglang.srt.disaggregation.agentic_workset_ledger import PageRun, WorksetLedger


def intent(sid="r:1", attempt="a1", owner="fresh", tokens=8, **kwargs):
    return WorksetIntent(sid, attempt, owner, 0, tokens, **kwargs)


class CountingLedger(WorksetLedger):
    def __init__(self, **kwargs):
        super().__init__(incarnation="run", page_count=4, page_size=4, **kwargs)
        self.grants = []

    def grant(self, *args, **kwargs):
        self.grants.append((threading.current_thread(), args, kwargs))
        return super().grant(*args, **kwargs)


@pytest.fixture
def controller():
    ledger = CountingLedger(tp_size=2)
    result = WorksetController(ledger)
    yield result
    result.shutdown()


def close_and_free(controller, plan):
    assert controller.cancel(plan.key).result(2).operation == "cancel"
    for rank in range(controller.ledger.tp_size):
        controller.report_fence(plan.key, rank, quiet=True, unreferenced=True).result(2)
    return controller.free(plan.key).result(2)


def barrier(controller):
    # An unknown exact cancellation is a read-equivalent command queue barrier;
    # it does not create any ownership or resource-return event.
    from sglang.srt.disaggregation.agentic_workset_ledger import LeaseKey
    assert controller.cancel(LeaseKey("run", "absent", "absent", 999)).result(2) is None


def test_grants_without_scheduler_ticks_single_worker_and_immutable_inputs(controller):
    future = controller.request(intent())
    assert not future.cancel()  # Future cancellation cannot lose an address grant.
    plan = future.result(2)
    assert plan.allocated_tokens == 8
    assert controller.ledger.counts.free_pages == 2
    assert controller.ledger.grants[0][0] is controller._thread
    with pytest.raises(RuntimeError, match="different controller"):
        controller.ledger.cancel(plan.key)
    with pytest.raises(TypeError):
        controller.request(dict(snapshot_id="r"))


def test_capacity_wait_has_no_poll_and_requires_real_return(controller):
    first = controller.request(intent(tokens=16)).result(2)
    waiting = controller.request(intent("r:2"))
    barrier(controller)
    attempts = len(controller.ledger.grants)
    time.sleep(0.025)
    assert not waiting.done() and len(controller.ledger.grants) == attempts
    controller.cancel(first.key).result(2)
    controller.report_fence(first.key, 0, quiet=True, unreferenced=True).result(2)
    assert controller.free(first.key).result(2) is None
    assert not waiting.done() and len(controller.ledger.grants) == attempts
    controller.report_fence(first.key, 1, quiet=True, unreferenced=True).result(2)
    assert controller.free(first.key).result(2).operation == "free"
    assert waiting.result(2).key.snapshot_id == "r:2"
    assert len(controller.ledger.grants) == attempts + 1


def test_pending_cancel_is_distinct_from_granted_cancel_and_rejects_late_intent(controller):
    first_intent = intent(tokens=16)
    first = controller.request(first_intent).result(2)
    wanted = intent("later")
    pending, duplicate = controller.request(wanted), controller.request(wanted)
    assert controller.cancel_pending(wanted).result(2)
    for future in (pending, duplicate):
        with pytest.raises(PendingWorksetCancelled):
            future.result(2)
    with pytest.raises(PendingWorksetCancelled):
        controller.request(wanted).result(2)
    assert controller.cancel_pending(first_intent).result(2) is False
    assert controller.ledger.may_start(first.key)
    assert controller.ledger.counts.free_pages == 0
    close_and_free(controller, first)
    assert controller.ledger.counts.free_pages == 4


def test_cancel_before_request_prevents_resurrection_without_touching_new_attempt(controller):
    previous = intent()
    assert controller.cancel_pending(previous).result(2)
    with pytest.raises(PendingWorksetCancelled):
        controller.request(previous).result(2)
    successor = controller.request(replace(previous, attempt_id="new")).result(2)
    before = controller.ledger.counts
    assert controller.cancel_pending(previous).result(2)
    assert controller.ledger.counts == before and controller.ledger.may_start(successor.key)
    assert controller.ledger.current_view(previous.snapshot_id).plan.key == successor.key
    assert controller.ledger.current_view("unknown") is None


def test_stale_return_cannot_free_successor_and_stale_grant_rejected(controller):
    previous = intent()
    first = controller.request(previous).result(2)
    close_and_free(controller, first)
    current = controller.request(replace(previous, attempt_id="a2")).result(2)
    before = controller.ledger.counts
    assert controller.free(first.key).result(2).key == first.key
    assert controller.cancel(first.key).result(2) is None
    assert controller.ledger.counts == before
    assert controller.ledger.may_start(current.key)
    with pytest.raises(ValueError):
        controller.request(previous).result(2)


@pytest.mark.parametrize("size", [1, 2, 8])
def test_partial_fenced_return_wakes_waiter_not_entire_workset(size):
    ledger = CountingLedger(tp_size=size)
    c = WorksetController(ledger)
    try:
        first = c.request(intent(tokens=16)).result(2)
        waiting = c.request(intent("next", tokens=4))
        pages = [PageRun(first.suffix_pages[0].start, 1)]
        returned = c.begin_return(first.key, "suffix-done", pages=pages).result(2)
        pages.clear()  # Caller list cannot change the queued/frozen subset.
        assert returned.pages[0].count == 1
        for rank in range(size):
            c.report_return_fence(returned, rank, quiet=True, unreferenced=False).result(2)
        assert c.commit_return(returned).result(2) is None and not waiting.done()
        for rank in range(size):
            c.report_return_fence(returned, rank, quiet=True, unreferenced=True).result(2)
        assert c.commit_return(returned).result(2).operation == "return"
        second = waiting.result(2)
        assert second.suffix_pages == returned.pages
        assert sum(run.count for run in ledger.view(first.key).remaining_pages) == 3
        before = ledger.counts
        c.commit_return(returned).result(2)  # Duplicate is not another capacity event.
        assert ledger.counts == before
    finally:
        c.shutdown()


def test_multithread_enqueue_has_one_writer_and_no_double_allocation():
    ledger = WorksetLedger(incarnation="run", page_count=64, page_size=4)
    c = WorksetController(ledger)
    try:
        with ThreadPoolExecutor(8) as threads:
            submitted = list(threads.map(lambda i: c.request(intent(str(i), tokens=4)), range(64)))
        plans = [future.result(2) for future in submitted]
        addresses = [plan.suffix_pages[0].start for plan in plans]
        assert len(set(addresses)) == 64 and ledger.counts.free_pages == 0
    finally:
        c.shutdown()


def test_no_type_priority_pending_insertion_order():
    ledger = CountingLedger()
    c = WorksetController(ledger)
    try:
        first = c.request(intent(tokens=16)).result(2)
        slow = c.request(intent("slow", owner="slow", tokens=12))
        direct = c.request(intent("direct", owner="direct", tokens=4))
        fresh = c.request(intent("fresh", owner="fresh", tokens=4))
        close_and_free(c, first)
        assert slow.result(2).sequence < direct.result(2).sequence
        assert not fresh.done()
        close_and_free(c, direct.result())
        assert fresh.result(2).owner == "fresh"
    finally:
        c.shutdown()


def test_pending_overflow_is_explicit_and_shutdown_retains_live_ownership():
    ledger = CountingLedger()
    c = WorksetController(ledger, max_pending=1)
    first = c.request(intent(tokens=16)).result(2)
    pending = c.request(intent("waiting"))
    with pytest.raises(ControllerQueueFull):
        c.request(intent("overflow")).result(2)
    before = ledger.counts
    c.shutdown()
    with pytest.raises(ControllerStopped):
        c.check_health()
    with pytest.raises(ControllerStopped):
        pending.result(2)
    with pytest.raises(ControllerStopped):
        c.free(first.key)
    assert ledger.counts == before and ledger.view(first.key) is not None


def test_command_queue_overflow_does_not_drop_accepted_operations():
    entered, proceed = threading.Event(), threading.Event()
    class PausedLedger(CountingLedger):
        def grant(self, *args, **kwargs):
            entered.set()
            assert proceed.wait(2)
            return super().grant(*args, **kwargs)
    ledger = PausedLedger()
    c = WorksetController(ledger, max_queue=1)
    try:
        first = c.request(intent(tokens=4))
        assert entered.wait(2)
        second = c.request(intent("two", tokens=4))
        with pytest.raises(ControllerQueueFull):
            c.request(intent("three", tokens=4))
        proceed.set()
        assert first.result(2).key.snapshot_id == "r:1"
        assert second.result(2).key.snapshot_id == "two"
        assert ledger.counts.live_leases == 2
    finally:
        proceed.set()
        c.shutdown()


def test_unexpected_post_mutation_failure_stops_without_implicit_free():
    class FailingLedger(CountingLedger):
        def grant(self, *args, **kwargs):
            super().grant(*args, **kwargs)
            raise RuntimeError("injected after ownership mutation")
    ledger = FailingLedger()
    c = WorksetController(ledger)
    future = c.request(intent())
    with pytest.raises(RuntimeError, match="injected"):
        future.result(2)
    c.shutdown()
    assert ledger.counts.live_leases == 1 and ledger.counts.free_pages == 2
    with pytest.raises(ControllerStopped):
        c.request(intent("new"))


def test_duplicate_waiter_shapes_and_cancel_receipts_are_bounded():
    c = WorksetController(CountingLedger(), max_cancelled=1)
    try:
        c.request(intent(tokens=16)).result(2)
        first = intent("first")
        a = c.request(first)
        with pytest.raises(ValueError, match="changed"):
            c.request(replace(first, prompt_tokens=12)).result(2)
        assert c.cancel_pending(first).result(2)
        with pytest.raises(PendingWorksetCancelled):
            a.result(2)
        second = intent("second")
        b = c.request(second)
        with pytest.raises(ControllerQueueFull, match="receipt"):
            c.cancel_pending(second).result(2)
        assert not b.done()  # Failed cancel did not silently discard the intent.
    finally:
        c.shutdown()


def test_ledger_previously_written_on_another_thread_fails_closed():
    ledger = CountingLedger()
    existing = ledger.grant("old", "attempt", owner="old", parent_tokens=0, prompt_tokens=4)
    c = WorksetController(ledger)
    try:
        with pytest.raises(RuntimeError, match="different controller"):
            c.request(intent()).result(2)
    finally:
        c.shutdown()
    assert ledger.view(existing.key) is not None and ledger.counts.live_leases == 1


def test_shutdown_from_return_callback_does_not_grant_pending():
    c = WorksetController(CountingLedger())
    first = c.request(intent(tokens=16)).result(2)
    waiting = c.request(intent("waiting"))
    c.cancel(first.key).result(2)
    c.report_fence(first.key, 0, quiet=True, unreferenced=True).result(2)
    # Install a deterministic callback before releasing the worker operation.
    entered, proceed = threading.Event(), threading.Event()
    original = c.ledger.free
    def blocked_free(key):
        entered.set()
        assert proceed.wait(2)
        return original(key)
    c.ledger.free = blocked_free
    returned = c.free(first.key)
    assert entered.wait(2)
    returned.add_done_callback(lambda _: c.shutdown(wait=False))
    proceed.set()
    assert returned.result(2).operation == "free"
    c.shutdown()
    with pytest.raises(ControllerStopped):
        waiting.result(2)
    assert c.ledger.counts.live_leases == 0
