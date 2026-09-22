"""Terminal ownership is metadata progress, never an H2D admission."""
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.test_agentic_host_async_prepare import fixture, worker_once
from sglang.srt.disaggregation.test_agentic_tp_host_pipeline import leader
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager as Manager


@pytest.mark.parametrize("state,reason", [
    ("failed", "shared_host_h2d_failed"),
    ("recompute_required", "shared_host_evicted"),
])
def test_tp1_terminal_ignores_full_io_admission(state, reason):
    m, allocator, (req,) = fixture(lanes=1)
    m._ledger_entries_cache[req.parent.snapshot_id]["state"] = state
    m._reserve_h2d_lane = lambda *a: pytest.fail("terminal acquired lane")
    m.ledger.get = lambda *a: pytest.fail("terminal did a scheduler RPC")
    assert m.gate_request(req, req.parent, allow_prepare=False, allow_start=False)
    worker_once(m)
    assert not m.gate_request(req, req.parent, allow_prepare=False, allow_start=False)
    assert req._agentic_kv_fallback == reason
    assert not allocator.allocated
    assert len(m.released_records) == 1


@pytest.mark.parametrize("block", ["copy", "record", "lease", "release"])
def test_terminal_waits_for_real_local_ownership(block):
    m, _, (req,) = fixture()
    sid = req.parent.snapshot_id
    if block == "copy":
        m.loads[req.rid] = {"request_generation": req.parent,
                           "event": NS(query=lambda: False)}
    elif block == "record":
        m.host_ready[sid]["loading"] = True
    elif block == "lease":
        m.workset_broker.get = lambda *a, **kw: object()
    else:
        m._release_record = lambda record: False
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    worker_once(m)
    assert req.rid not in m._host_prepare_results
    assert req.rid in m._host_prepares
    if block == "copy":
        m.loads.clear()
    elif block == "record":
        m.host_ready[sid]["loading"] = False
    elif block == "lease":
        m.workset_broker.get = lambda *a, **kw: None
    else:
        m._release_record = lambda record: True
    worker_once(m)
    assert m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")


def test_existing_cpu_prepare_cannot_be_replaced_by_terminal():
    m, _, (req,) = fixture()
    old = {"parent": req.parent, "cancelled": False}
    m._host_prepares["old-rid"] = old
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert m._host_prepares == {"old-rid": old}


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_same_attempt_prepare_switches_to_terminal_without_waiting_for_host(tp_size):
    m, allocator, (req,) = fixture(tp_size=tp_size)
    old = {"parent": req.parent, "cancelled": False, "view": req}
    m._host_prepares[req.rid] = old
    m._prepare_host_restore = lambda *a, **kw: pytest.fail("tried to load evicted Host")
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert m._host_prepares[req.rid] is old
    assert old["terminal_reason"] == "shared_host_evicted"
    worker_once(m)
    assert m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert not allocator.allocated


def test_terminal_does_not_revive_cancelled_prepare():
    m, _, (req,) = fixture(tp_size=8)
    old = {"parent": req.parent, "cancelled": True}
    m._host_prepares[req.rid] = old
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert old == {"parent": req.parent, "cancelled": True}


@pytest.mark.parametrize("marker", ["h2d_claiming", "h2d_reserving"])
@pytest.mark.parametrize("granted", [False, True])
def test_terminal_prestart_marker_cancels_workset_before_releasing_record(marker, granted):
    m, allocator, (req,) = fixture(tp_size=8)
    sid = req.parent.snapshot_id
    broker = m.workset_broker
    owner = broker.slow_owner(sid, req.rid)
    broker.request(sid, 1, 2, owner=owner)
    if granted:
        broker.service(allocator)
    record = m.host_ready[sid]
    record["loading"] = marker
    m._host_prepares[req.rid] = {
        "parent": req.parent, "cancelled": False, "view": req,
    }
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    worker_once(m)
    if granted:
        # Closing an active lease is not its physical retirement receipt.
        assert not m.released_records
        assert req.rid in m._host_prepares
        broker.service(allocator)
        worker_once(m)
        assert allocator.freed == [8]
    assert not broker.owner_has_unretired_work(sid, owner=owner)
    assert m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert not record["loading"]
    assert m.released_records == [record]
    assert sid not in m.host_ready


@pytest.mark.parametrize("state", ["aborting", "retry_pending"])
def test_follower_stale_nonterminal_cache_cannot_override_leader(state):
    m, _, (req,) = fixture(tp_size=8)
    m._ledger_entries_cache[req.parent.snapshot_id]["state"] = state
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    worker_once(m)
    assert m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")


def test_ordinary_fallback_result_is_not_a_terminal_cleanup_receipt():
    m, _, (req,) = fixture(tp_size=8)
    m._host_prepare_results[req.rid] = (False, "shared_host_evicted")
    m.workset_broker.get = lambda *a, **kw: object()
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    worker_once(m)
    assert not getattr(m, "_host_terminal_results", {})
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")


def test_tp_ordinary_prepare_racing_terminal_cannot_admit_locally():
    m, _, (req,) = fixture(tp_size=8)
    m._host_prepare_results[req.rid] = (False, "shared_host_evicted")
    assert m.gate_request(req, req.parent)
    assert not getattr(req, "_agentic_kv_gate_complete", False)


def test_legacy_synchronous_terminal_progress():
    m, _, (req,) = fixture(tp_size=8)
    m.h2d_async_prepare = False
    assert m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert not m._host_prepares


@pytest.mark.parametrize("partial", [False, True])
def test_terminal_bound_parent_rollback_is_scheduler_owned(partial):
    import threading
    m, _, (req,) = fixture(tp_size=8)
    sid = req.parent.snapshot_id
    record = m.host_ready[sid]
    record["loading"] = True
    scheduler_thread = threading.get_ident()
    calls = []
    def rollback(actual_req, parent):
        assert threading.get_ident() == scheduler_thread
        assert actual_req is req and parent == req.parent
        calls.append(True)
        if hasattr(req, "_agentic_host_rank_loaded"):
            del req._agentic_host_rank_loaded
    m.rollback_bound_parent = rollback
    if partial:
        m.loads[req.rid] = {"request_generation": req.parent, "record": record,
                           "radix_bound": True, "workset_lease": object()}
    else:
        req._agentic_host_rank_loaded = True
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert calls == [True]
    if partial:
        assert not m.loads[req.rid]["radix_bound"]
    else:
        assert not record["loading"]


def test_terminal_copy_never_restarts_from_stale_mirror():
    m, _, (req,) = fixture(tp_size=8)
    load = {"request_generation": req.parent, "start_allowed": True}
    m.loads[req.rid] = load
    calls = []
    m._discard_failed_h2d_load = lambda rid, item, **kw: calls.append(kw) or False
    m._start_h2d_chunk = lambda *a: pytest.fail("terminal copy restarted")
    m._h2d_poisoned = False
    assert not m._progress_terminal_restore(req.rid, {"parent": req.parent})
    assert load["terminal_cleanup_requested"] and not load["start_allowed"]
    # Repeat normal progress with a stale, apparently ready ledger entry.
    Manager._progress_h2d_loads(m)
    Manager._progress_h2d_loads(m)
    assert calls == [{"terminal": True}] * 3


def test_retry_cannot_rollback_another_rids_bound_parent():
    m, _, (req,) = fixture(tp_size=8)
    m.loads["old-rid"] = {
        "request_generation": req.parent, "radix_bound": True,
        "workset_lease": object(), "record": m.host_ready[req.parent.snapshot_id],
    }
    m.rollback_bound_parent = lambda *a: pytest.fail("rolled back another Req")
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert m.loads["old-rid"]["radix_bound"]
    assert not hasattr(req, "_agentic_host_workset_lease")


def test_cancelled_terminal_preparation_and_receipt_are_retired():
    m, _, (req,) = fixture(tp_size=8)
    m.ledger.request_host_load_failure = lambda *a, **kw: False
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    assert not m.tp_host_control_quiescent(req.parent.snapshot_id, req.rid)
    m.abort_request(req.rid, req.parent)
    worker_once(m)
    assert not m._host_prepares
    assert not getattr(m, "_host_terminal_results", {})
    assert not m.prepare_terminal_restore(req, req.parent, "shared_host_evicted")
    worker_once(m)
    assert req.rid in m._host_terminal_results
    m.abort_request(req.rid, req.parent)
    assert req.rid not in m._host_terminal_results


@pytest.mark.parametrize("tp_size", [2, 8])
def test_terminal_progress_with_all_physical_lanes_busy(tp_size):
    state = leader(tp_size=tp_size)
    initial = Scheduler._agentic_tp_prepare_admission_control(state)
    assert len(initial["host_commands"]) == 4
    state.agentic_host_staging_manager.terminal_restore_reason = (
        lambda p: "shared_host_evicted" if p.snapshot_id == "r4:1" else None
    )
    result = Scheduler._agentic_tp_prepare_admission_control(state)
    command = next(c for c in result["host_commands"] if c["snapshot"] == "r4:1")
    assert command["action"] == "terminal_prepare"
    assert command["terminal"] == {"rid": "r4", "reason": "shared_host_evicted"}
    assert len(result["host_commands"]) == 5


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("direct_only", [False, True])
def test_terminal_all_rank_handshake_and_stale_retry(tp_size, direct_only, monkeypatch):
    import threading
    from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotState
    if direct_only:
        monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "unit-test:1")
        monkeypatch.setenv("SGLANG_AGENTIC_KV_HOST_STAGING", "false")
    states = [leader(tp_size=tp_size) for _ in range(tp_size)]
    reports = {}
    for rank, s in enumerate(states):
        s.tp_rank = rank
        s.agentic_kv_waiting_queue = s.agentic_kv_waiting_queue[:1]
        s.agentic_tp_host_local_admitted = set()
        s.agentic_host_staging_manager.loads = {}
        s.agentic_host_staging_manager.ledger = NS(is_event_control=True)
        # Followers deliberately never receive the terminal ledger update.
        s.agentic_host_staging_manager.terminal_restore_reason = (
            (lambda p: "shared_host_evicted") if rank == 0 else (lambda p: None)
        )
        s.agentic_tp_host_mailbox = NS(
            publish_local=lambda key, value, rank=rank: reports.__setitem__((rank, key), value),
            group_status=lambda key: min(reports[(r, key)] for r in range(tp_size))
                if all((r, key) in reports for r in range(tp_size)) else None,
            clear_local=lambda key: None, clear_group=lambda key: None,
        )
        s.agentic_host_staging_manager.prepare_terminal_restore = lambda *a: False
        if direct_only:
            s.agentic_host_staging_manager = None
            s.agentic_early_direct_poll_lock = threading.RLock()
            s.agentic_early_direct_receives = {}
            s._agentic_bind_early_direct_receive = lambda *a: None
            manifest = NS(state=SnapshotState.FAILED if rank == 0 else SnapshotState.DIRECT_READY,
                          failure_reason="shared_host_staging_unavailable")
            s._agentic_snapshot_store = lambda manifest=manifest: NS(load=lambda *a, **kw: manifest)
            s.agentic_p_workset_broker = NS(
                install_tp_plan=lambda *a, **kw: None,
                direct_owner=lambda sid: 'direct:' + sid,
                supersede_unstarted=lambda *a, **kw: None,
                owner_has_unretired_work=lambda *a, **kw: True,
            )
            # Same child, intentionally different pushed lifecycle views.
            assert Scheduler._agentic_should_defer(s, s.agentic_kv_waiting_queue[0][0], 0)
            assert not getattr(s.agentic_kv_waiting_queue[0][0], '_agentic_kv_gate_complete', False)
    def broadcast(control):
        for s in states:
            if not direct_only:
                s.agentic_p_workset_broker = NS(install_tp_plan=lambda *a, **kw: None)
            s.agentic_tp_direct_group_status = {}
            Scheduler._agentic_tp_consume_admission_control(s, [control])
            if not direct_only:
                del s.agentic_p_workset_broker
    control = Scheduler._agentic_tp_prepare_admission_control(states[0])
    broadcast(control)
    for s in states:
        req = s.agentic_kv_waiting_queue[0][0]
        assert Scheduler._agentic_should_defer(s, req, 0, allow_start_io=False)
    for s in states[:-1]:
        s.agentic_kv_waiting_queue[0][0]._agentic_host_terminal_ready = True
    for s in reversed(states):
        Scheduler._agentic_tp_reduce_host_status(s)
    control = Scheduler._agentic_tp_prepare_admission_control(states[0])
    assert control["host_commands"][0]["action"] == "terminal_prepare"
    states[-1].agentic_kv_waiting_queue[0][0]._agentic_host_terminal_ready = True
    for s in reversed(states):
        Scheduler._agentic_tp_reduce_host_status(s)
    control = Scheduler._agentic_tp_prepare_admission_control(states[0])
    assert control["host_commands"][0]["action"] == "terminal_admit"
    broadcast(control)
    for s in states:
        req = s.agentic_kv_waiting_queue[0][0]
        retry = NS(**vars(req))
        retry.rid = "retry-rid"
        assert Scheduler._agentic_should_defer(s, retry, 0, allow_start_io=False)
        assert not Scheduler._agentic_should_defer(s, req, 0, allow_start_io=False)
        assert req._agentic_kv_fallback == ("shared_host_staging_unavailable" if direct_only else "shared_host_evicted")
        s.agentic_kv_waiting_queue.clear()
    for s in reversed(states):
        Scheduler._agentic_tp_reduce_host_status(s)
    control = Scheduler._agentic_tp_prepare_admission_control(states[0])
    assert control["host_commands"][0]["action"] == "clear"
    broadcast(control)
    assert all(not s._agentic_tp_host_terminals for s in states)


def test_no_host_terminal_keeps_receiver_and_retiring_pages():
    import threading
    parent = RequestGeneration('no-host', 1)
    calls = []
    s = NS(agentic_early_direct_poll_lock=threading.RLock(),
           agentic_early_direct_receives={parent.snapshot_id: object()},
           agentic_p_workset_broker=NS(
               direct_owner=lambda sid: 'direct:' + sid,
               supersede_unstarted=lambda *a, **kw: calls.append((a, kw)),
               owner_has_unretired_work=lambda *a, **kw: True))
    assert not Scheduler._agentic_no_host_terminal_quiescent(s, parent)
    assert not calls
    s.agentic_early_direct_receives.clear()
    assert not Scheduler._agentic_no_host_terminal_quiescent(s, parent)
    s.agentic_p_workset_broker.owner_has_unretired_work = lambda *a, **kw: False
    assert Scheduler._agentic_no_host_terminal_quiescent(s, parent)


def test_no_host_cancelled_terminal_retires_only_after_direct_fence():
    import threading
    s = leader()
    s.agentic_host_staging_manager = None
    s.tree_cache = NS()
    req = s.agentic_kv_waiting_queue[0][0]
    parent = RequestGeneration('r0', 1)
    sid = parent.snapshot_id
    s.agentic_tp_host_active_requests = {sid: parent}
    s.agentic_tp_host_command_visible = True
    s._agentic_tp_host_actions = {sid: 'terminal_prepare'}
    s.agentic_early_direct_poll_lock = threading.RLock()
    s.agentic_early_direct_receives = {sid: object()}
    s.agentic_p_workset_broker = NS(
        direct_owner=lambda sid: 'direct:' + sid,
        supersede_unstarted=lambda *a, **kw: None,
        owner_has_unretired_work=lambda *a, **kw: False)
    reports = {}
    s.agentic_tp_host_mailbox = NS(
        publish_local=lambda key, value: reports.__setitem__(key, value),
        group_status=lambda key: reports.get(key))
    Scheduler._agentic_abort_cleanup(s, req)
    assert s.agentic_tp_host_cancelled_requests == {sid: req.rid}
    s.agentic_kv_waiting_queue.clear()
    Scheduler._agentic_tp_reduce_host_status(s)
    assert reports[sid] == 0
    s.agentic_early_direct_receives.clear()
    Scheduler._agentic_tp_reduce_host_status(s)
    assert reports[sid] == 4
