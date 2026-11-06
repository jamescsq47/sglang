"""Regressions for full-state-pool progress after TP1 Direct ownership commit."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from sglang.srt.managers.scheduler import Scheduler


def make_scheduler(requests):
    scheduler = object.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_kv_waiting_queue = list(enumerate(requests))
    scheduler.agentic_kv_waiting_queue = [
        (req, float(index)) for index, req in scheduler.agentic_kv_waiting_queue
    ]
    scheduler.agentic_early_direct_completion_queue = deque()
    scheduler.agentic_early_direct_poll_lock = nullcontext()
    scheduler.agentic_host_staging_manager = None
    scheduler.waiting_queue = []
    scheduler._agentic_publish_p_scheduled = lambda req: None
    scheduler._add_request_to_queue = scheduler.waiting_queue.append
    # Blocked metadata waiters require free state slots. The ready waiters
    # already own their entire workset and therefore require no allocation.
    scheduler._agentic_should_defer = lambda req, *_args, **_kwargs: not getattr(
        req, "_agentic_kv_gate_complete", False
    )
    return scheduler


def request(index, *, resident=False, kind="fast"):
    req = SimpleNamespace(rid=str(index), _agentic_kv_queue_class=kind)
    if resident:
        req._agentic_kv_gate_complete = True
        req._agentic_workset_backed = True
        req._agentic_mamba_runtime_reserved = True
        req._agentic_p_workset_lease = SimpleNamespace(state="handed")
    return req


@pytest.mark.parametrize("kind", ["fast", "slow"])
def test_78_resident_worksets_progress_behind_16_capacity_waiters(monkeypatch, kind):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_SCAN_LIMIT", "16")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "8")
    blocked = [request(i) for i in range(16)]
    resident = [request(i + 16, resident=True, kind=kind) for i in range(78)]
    scheduler = make_scheduler(blocked + resident)
    Scheduler._drain_agentic_kv_waiting_queue(scheduler)
    assert scheduler.waiting_queue == resident
    assert [r for r, _ in scheduler.agentic_kv_waiting_queue] == blocked
    # No duplicate bootstrap handoff on a subsequent scheduler tick.
    Scheduler._drain_agentic_kv_waiting_queue(scheduler)
    assert scheduler.waiting_queue == resident


def test_resident_handoff_does_not_consume_new_io_budget(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "1")
    resident = [request(i, resident=True) for i in range(3)]
    fresh = request(3, kind="new")
    fresh._agentic_kv_gate_complete = True
    scheduler = make_scheduler(resident + [fresh])
    Scheduler._drain_agentic_kv_waiting_queue(scheduler)
    assert scheduler.waiting_queue == resident + [fresh]


@pytest.mark.parametrize("missing", ["gate", "workset", "runtime", "handed"])
def test_partial_or_non_mamba_worksets_keep_existing_admission(monkeypatch, missing):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_SCAN_LIMIT", "1")
    blocked = request(0)
    candidate = request(1, resident=True)
    if missing == "handed":
        candidate._agentic_p_workset_lease.state = "io_inflight"
    else:
        setattr(candidate, {
            "gate": "_agentic_kv_gate_complete",
            "workset": "_agentic_workset_backed",
            "runtime": "_agentic_mamba_runtime_reserved",
        }[missing], False)
    scheduler = make_scheduler([blocked, candidate])
    Scheduler._drain_agentic_kv_waiting_queue(scheduler)
    assert scheduler.waiting_queue == []
    assert len(scheduler.agentic_kv_waiting_queue) == 2


def test_tp_local_residency_never_bypasses_group_admission(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_SCAN_LIMIT", "1")
    scheduler = make_scheduler([request(0), request(1, resident=True)])
    scheduler.tp_size = 2
    Scheduler._drain_agentic_kv_waiting_queue(scheduler)
    assert scheduler.waiting_queue == []
    assert len(scheduler.agentic_kv_waiting_queue) == 2
