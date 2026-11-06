"""CPU regressions for scheduler-selected, worker-prepared Host recovery.

Preparation/claim/grant is not physical ownership. These tests keep allocation
on the caller thread and require cancellation to wait for CPU preparation,
without making the scheduler wait for that preparation.
"""

import ast
import inspect
import queue
import textwrap
import threading
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticPHostStagingManager as Manager,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    RequestGeneration,
    token_ids_digest,
)
from sglang.srt.disaggregation.test_agentic_h2d_decoupling import manager
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker


class MainThreadAllocator:
    def __init__(self):
        self.owner = threading.get_ident()
        self.allocated = []
        self.freed = []

    def alloc(self, count):
        assert threading.get_ident() == self.owner
        self.allocated.append(count)
        return torch.arange(count)

    def free(self, indices):
        assert threading.get_ident() == self.owner
        self.freed.append(len(indices))


def fixture(lanes=2, count=1):
    m = manager(lanes=lanes)
    m.h2d_event_progress = m.h2d_async_prepare = True
    m._host_prepares, m._host_prepare_results = {}, {}
    m._scheduler_events = queue.SimpleQueue()
    m.active, m.aborting, m.host_ready, m.spills = {}, {}, {}, {}
    m._ledger_entries_cache = {}
    m.tp_rank, m.tp_size, m.arena_domain, m.owner = 0, 1, 0, "p:async-test"
    m.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
    m._progress_h2d_loads = lambda: None  # Never submit CUDA in this CPU suite.
    m.released_records = []
    m._release_record = lambda record: m.released_records.append(record) or True
    m._release_consumed_owned_host = lambda *args, **kwargs: None
    m.tree_cache = NS(insert=lambda *args, **kwargs: pytest.fail("worker bound Radix"))

    def claim(sid, owner, **kwargs):
        m._ledger_entries_cache[sid]["recovery_claims"] = {
            "0": {"claim_id": kwargs["claim_id"]}
        }
        return True

    def abort(sid, owner, **kwargs):
        m._ledger_entries_cache[sid]["state"] = "aborting"
        return True

    def cancel(sid, owner, **kwargs):
        m._ledger_entries_cache[sid].pop("recovery_claims", None)
        return True

    def drained(sid, owner, **kwargs):
        m._ledger_entries_cache[sid]["state"] = "failed"
        return True

    m.ledger = NS(
        get=lambda sid: m._ledger_entries_cache.get(sid),
        claim_d2p_recovery_rank=claim,
        attach_d2p_recovery_lease_rank=lambda *args, **kwargs: True,
        begin_host_load_rank=lambda *args, **kwargs: True,
        request_host_load_failure=abort,
        cancel_d2p_recovery_rank=cancel,
        mark_host_load_rank_drained=drained,
    )
    reqs = []
    for i in range(count):
        parent = RequestGeneration(f"async-{i}", 0)
        req = NS(rid=f"rid-{i}", parent=parent, origin_input_ids=[11, 22])
        reqs.append(req)
        m._ledger_entries_cache[parent.snapshot_id] = {
            "state": "host_ready", "p_owner": m.owner,
        }
        m.host_ready[parent.snapshot_id] = {
            "snapshot": NS(_materialized=object()), "loading": False,
            "offer": {"token_count": 1, "byte_size": 128,
                      "token_digest": token_ids_digest([11])},
        }
    return m, MainThreadAllocator(), reqs


def worker_once(m):
    errors = []

    def run():
        try:
            m._progress_host_prepares()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(3)
    assert not thread.is_alive(), "preparation worker failed to return"
    assert not errors, errors


def test_real_broker_grant_starts_without_second_scheduler_gate():
    m, alloc, (req,) = fixture()
    assert m.gate_request(req, req.parent) is True
    assert not m.workset_broker._intents and not alloc.allocated
    worker_once(m)
    assert req.parent.snapshot_id in m.workset_broker._intents
    assert not m.loads and not alloc.allocated
    m.workset_broker.service(alloc)
    assert alloc.allocated == [8]  # Parent and suffix independently page rounded.
    worker_once(m)
    load = m.loads[req.rid]
    assert load["start_allowed"] and not load["ledger_prepare_pending"]
    assert load["workset_lease"].state == "io_reserved"
    assert not m._host_prepares
    assert not hasattr(req, "_agentic_kv_gate_complete")
    assert not hasattr(req, "_agentic_p_workset_lease")
    assert not m.released_records and not alloc.freed


def test_queue_freezes_prompt_and_preserves_original_request():
    m, alloc, (req,) = fixture()
    m.gate_request(req, req.parent)
    view = m._host_prepares[req.rid]["view"]
    assert view is not req and view.origin_input_ids == (11, 22)
    req.origin_input_ids[0] = 99
    worker_once(m)
    assert req.parent.snapshot_id in m.workset_broker._intents
    assert req.origin_input_ids == [99, 22]


def test_first_scheduler_gate_does_not_read_file_ledger():
    m, alloc, (req,) = fixture()
    m.ledger.get = lambda *args: pytest.fail("scheduler read ledger during async queue")
    assert m.gate_request(req, req.parent) is True
    assert req.rid in m._host_prepares


@pytest.mark.parametrize("state", [None, "direct_ready", "consumed"])
def test_non_host_parent_remains_available_to_direct_discovery(state):
    m, alloc, (req,) = fixture()
    m.host_ready.clear()
    m._ledger_entries_cache.clear()
    if state is not None:
        m._ledger_entries_cache[req.parent.snapshot_id] = {"state": state}
    assert m.gate_request(req, req.parent) is None
    assert not m._host_prepares and not m.h2d_selected_snapshots()


@pytest.mark.parametrize("kwargs", [{"allow_prepare": False}, {"allow_start": False}])
def test_no_background_selection_without_native_admission(kwargs):
    m, alloc, (req,) = fixture()
    assert m.gate_request(req, req.parent, **kwargs) is True
    assert not m._host_prepares and not m.h2d_selected_snapshots()


def test_lane_cap_and_duplicate_generation_bound_cpu_prepares():
    m, alloc, reqs = fixture(lanes=2, count=5)
    for req in reqs:
        m.gate_request(req, req.parent)
    assert len(m._host_prepares) == m.h2d_physical_occupancy() == 2
    original = reqs[0]
    duplicate = NS(rid="retry-rid", parent=original.parent, origin_input_ids=[11, 22])
    m.gate_request(duplicate, duplicate.parent)
    assert len(m._host_prepares) == 2 and duplicate.rid not in m._host_prepares
    assert not alloc.allocated


def test_abort_before_worker_cannot_create_late_intent():
    m, alloc, (req,) = fixture()
    m.gate_request(req, req.parent)
    m.abort_request(req.rid, req.parent)
    assert m._host_prepares[req.rid]["cancelled"]
    worker_once(m)
    assert not m._host_prepares and not m.workset_broker._intents
    assert not m.loads and not m.h2d_selected_snapshots()
    assert not alloc.allocated


@pytest.mark.parametrize("phase", ["materialize", "claim", "attach"])
@pytest.mark.parametrize("retry_rid", [False, True])
def test_cancel_during_cpu_phase_is_nonblocking_and_prevents_launch(phase, retry_rid):
    m, alloc, (req,) = fixture()
    m.gate_request(req, req.parent)
    if phase == "attach":
        worker_once(m)
        m.workset_broker.service(alloc)
    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(3), "test did not release CPU preparation"

    if phase == "materialize":
        snapshot = m.host_ready[req.parent.snapshot_id]["snapshot"]
        snapshot._materialized = None

        def materialize():
            block()
            snapshot._materialized = object()

        snapshot.materialize = materialize
    else:
        method = "claim_d2p_recovery_rank" if phase == "claim" else "attach_d2p_recovery_lease_rank"
        original = getattr(m.ledger, method)

        def delayed(*args, **kwargs):
            result = original(*args, **kwargs)
            block()
            return result

        setattr(m.ledger, method, delayed)
    errors = []

    def run_prepare():
        try:
            m._progress_host_prepares()
        except BaseException as exc:
            errors.append(exc)

    preparer = threading.Thread(target=run_prepare)
    preparer.start()
    assert entered.wait(2)
    abort_rid = "retry-rid" if retry_rid else req.rid
    aborter = threading.Thread(target=lambda: m.abort_request(abort_rid, req.parent))
    aborter.start()
    try:
        aborter.join(0.5)
        assert not aborter.is_alive(), "scheduler cancellation blocked on CPU/ledger work"
        assert not m.released_records and not alloc.freed
    finally:
        release.set()
        aborter.join(3)
        preparer.join(3)
    assert not errors and not preparer.is_alive()
    assert m._host_prepares[req.rid]["cancelled"]
    assert not m.loads or not m.loads[req.rid]["start_allowed"]
    worker_once(m)
    assert not m._host_prepares
    if phase == "attach":
        # Published io_reserved load goes through the existing fenced abort,
        # not prestart freeing. No CUDA work may start from this descriptor.
        assert m.loads[req.rid]["abort_requested"]
        assert m.loads[req.rid]["io_error"] is not None
        assert not alloc.freed
    else:
        assert not m.workset_broker._intents and not m.loads
        assert not m.h2d_selected_snapshots()


def test_prepare_exception_retains_owner_and_does_not_stop_other_request():
    m, alloc, reqs = fixture(count=2)
    for req in reqs:
        m.gate_request(req, req.parent)
    original = m._prepare_host_restore

    def prepare(view, parent, **kwargs):
        if view.rid == reqs[0].rid:
            raise OSError("transient mapping failure")
        return original(view, parent, **kwargs)

    m._prepare_host_restore = prepare
    worker_once(m)
    assert set(m._host_prepares) == {r.rid for r in reqs}
    assert reqs[1].parent.snapshot_id in m.workset_broker._intents
    assert not alloc.freed and not m.released_records


def test_no_capacity_keeps_host_and_full_intent_without_recompute():
    m, alloc, (req,) = fixture()
    m.gate_request(req, req.parent)
    worker_once(m)
    m.workset_broker.service(NS(alloc=lambda count: None))
    worker_once(m)
    assert req.rid in m._host_prepares and not m.loads
    assert req.parent.snapshot_id in m.workset_broker._intents
    assert not hasattr(req, "_agentic_kv_fallback")
    assert not m.released_records


def test_failed_prepare_ack_retries_exact_grant_without_scheduler_ledger_io():
    m, alloc, (req,) = fixture()
    m.gate_request(req, req.parent)
    worker_once(m)
    m.workset_broker.service(alloc)
    calls = []

    def acknowledge(*args, **kwargs):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            raise OSError("ACK lost after prepare")
        return True

    m.ledger.begin_host_load_rank = acknowledge
    worker_once(m)
    load = m.loads[req.rid]
    lease = load["workset_lease"]
    assert load["ledger_prepare_pending"] and not load["start_allowed"]
    assert req.rid in m._host_prepares
    m.ledger.get = lambda *args: pytest.fail("pending scheduler gate touched ledger")
    assert m.gate_request(req, req.parent) is True
    worker_once(m)
    assert m.loads[req.rid] is load and load["workset_lease"] is lease
    assert load["start_allowed"] and not load["ledger_prepare_pending"]
    assert all(tid != alloc.owner for tid in calls)
    assert alloc.allocated == [8] and not alloc.freed


def test_terminal_result_changes_live_request_only_when_scheduler_consumes():
    m, alloc, (req,) = fixture()
    m._ledger_entries_cache[req.parent.snapshot_id]["state"] = "recompute_required"
    assert m.gate_request(req, req.parent) is True
    worker_once(m)
    assert not hasattr(req, "_agentic_kv_gate_complete")
    assert not m._host_prepares and not m.h2d_selected_snapshots()
    assert m.gate_request(req, req.parent) is False
    assert req._agentic_kv_gate_complete
    assert req._agentic_kv_fallback == "shared_host_evicted"
    assert not alloc.allocated


def test_refused_broker_intent_does_not_consume_another_owner():
    m, alloc, (req,) = fixture()
    sid = req.parent.snapshot_id
    owner = m.workset_broker.slow_owner(sid, req.rid)
    m.workset_broker._superseded_owners.add((sid, owner))
    m.gate_request(req, req.parent)
    worker_once(m)
    assert not m.workset_broker._intents and not m.loads
    assert req.rid in m._host_prepares
    assert not alloc.allocated and not m.released_records


def test_foreign_recovery_route_does_not_take_local_lane_or_claim():
    m, alloc, (req,) = fixture()
    m._ledger_entries_cache[req.parent.snapshot_id]["recovery_domain"] = 1
    assert m.gate_request(req, req.parent) is True
    assert not m._host_prepares and not m.h2d_selected_snapshots()
    assert not m.workset_broker._intents


def test_repeated_failed_claims_cannot_accumulate_unbounded_prepares():
    m, alloc, reqs = fixture(lanes=1, count=6)
    m.ledger.claim_d2p_recovery_rank = lambda *args, **kwargs: False
    for req in reqs:
        assert m.gate_request(req, req.parent) is True
        worker_once(m)
        # Releasing a failed claim's lane must not let an old CPU descriptor
        # survive while arbitrarily many freshly selected descriptors enter.
        assert len(m._host_prepares) <= 2 * m.max_h2d_inflight
    assert not m.workset_broker._intents and not m.loads
    assert not alloc.allocated


def test_pending_cpu_prepare_keeps_control_worker_on_short_poll():
    m, alloc, (req,) = fixture()
    assert not m._has_local_io_progress()
    m.gate_request(req, req.parent)
    assert m._has_local_io_progress()


def test_foreign_abort_rejection_retires_only_unclaimed_local_prepare_slot():
    m, alloc, (req,) = fixture()
    record = m.host_ready[req.parent.snapshot_id]
    m.gate_request(req, req.parent)
    # The Router has moved recovery to another P before this CPU descriptor
    # acquires a claim. A stale abort cannot mutate that foreign ownership.
    m.ledger.request_host_load_failure = lambda *args, **kwargs: False
    m.abort_request(req.rid, req.parent)
    worker_once(m)
    assert not m._host_prepares and not m.loads
    assert not m.workset_broker._intents
    assert not m.h2d_selected_snapshots(), "unclaimed metadata lane leaked"
    assert m.host_ready[req.parent.snapshot_id] is record
    assert not record["loading"] and not m.released_records
    assert not alloc.allocated


def test_cancelled_old_descriptor_cannot_release_new_retry_metadata_credit():
    m, alloc, (req,) = fixture()
    m.ledger.request_host_load_failure = lambda *args, **kwargs: False
    m.gate_request(req, req.parent)
    m.abort_request(req.rid, req.parent)
    new = NS(rid="new-retry", parent=req.parent, origin_input_ids=[11, 22])
    original_abort = m.abort_request

    def abort_then_scheduler_reselect(rid, parent):
        original_abort(rid, parent)
        # Deterministically model scheduler selection after old abort returns
        # but before the worker performs old-descriptor metadata cleanup.
        assert m.gate_request(new, parent) is True

    m.abort_request = abort_then_scheduler_reselect
    worker_once(m)
    assert new.rid in m._host_prepares
    assert req.parent.snapshot_id in m.h2d_selected_snapshots()
    assert m.h2d_physical_occupancy() == 1
    assert not m.released_records and not alloc.allocated


def test_async_switch_requires_prior_tp1_hybrid_event_mode(monkeypatch):
    init = ast.parse(textwrap.dedent(inspect.getsource(Manager.__init__)))
    assignment = next(node for node in ast.walk(init)
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Attribute) and target.attr == "h2d_async_prepare"
                          for target in node.targets))
    expression = compile(ast.Expression(assignment.value), "<mode>", "eval")
    import os
    for previous in (False, True):
        for enabled in ("0", "true"):
            monkeypatch.setenv("SGLANG_AGENTIC_KV_P_HOST_ASYNC_PREPARE", enabled)
            assert eval(expression, {"os": os, "self": NS(h2d_event_progress=previous)}) is (
                previous and enabled == "true"
            )


def test_disabled_worker_is_noop_without_async_fields():
    m = manager(enabled=False)
    m._progress_host_prepares()
