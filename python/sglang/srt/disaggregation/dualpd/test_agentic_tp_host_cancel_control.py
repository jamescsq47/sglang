"""Cancelled Host commands cannot hold an otherwise empty P hostage."""
import ast
import inspect
import os
import threading
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    CUSTOM_GENERATION, CUSTOM_PARENT_GENERATION, CUSTOM_REQUEST_ID, RequestGeneration,
)
from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler


def host_manager():
    manager = object.__new__(AgenticPHostStagingManager)
    manager.loads = {}
    manager.host_ready = {}
    manager.workset_broker = SimpleNamespace(
        slow_owner=lambda sid, rid: f"slow:{sid}:{rid}",
        direct_owner=lambda sid: f"direct:{sid}",
        owner_has_unretired_work=lambda *a, **kw: False,
    )
    return manager


@pytest.mark.parametrize("blocker", [
    "loads", "aborting", "_pending_host_abort_requests",
    "_prestart_recovery_aborts", "_h2d_lane_reservations",
    "_h2d_resident_reservations", "host_ready", "slow_work", "direct_work",
])
def test_control_retirement_waits_for_all_physical_and_planned_work(blocker):
    value = host_manager()
    assert value.tp_host_control_quiescent("req:1", "child")
    if blocker == "loads":
        # Even a changed HTTP rid cannot hide the same generation's transfer.
        value.loads["old-child"] = {"request_generation": RequestGeneration("req", 1)}
    elif blocker == "host_ready":
        value.host_ready["req:1"] = {"loading": "abort_pending"}
    elif blocker.endswith("_work"):
        expected = "slow:req:1:child" if blocker == "slow_work" else "direct:req:1"
        value.workset_broker.owner_has_unretired_work = (
            lambda sid, *, owner: sid == "req:1" and owner == expected
        )
    else:
        setattr(value, blocker, {"req:1": object()})
    assert not value.tp_host_control_quiescent("req:1", "child")


def test_storage_only_copy_does_not_hold_cancelled_control():
    value = host_manager()
    # The other P owns recovery; retaining its source storage mapping is not
    # an active local H2D nor permission to destroy its ledger or payload.
    value.host_ready["req:1"] = {"loading": None}
    assert value.tp_host_control_quiescent("req:1", "child")
    assert "req:1" in value.host_ready


def test_abort_remembers_control_until_group_fences_drain():
    parent = RequestGeneration("req", 1)
    calls = []
    scheduler = SimpleNamespace(
        tp_size=2,
        tree_cache=SimpleNamespace(),
        agentic_tp_host_active_requests={parent.snapshot_id: parent},
        agentic_host_staging_manager=SimpleNamespace(
            abort_request=lambda rid, rg: calls.append((rid, rg))
        ),
    )
    req = SimpleNamespace(rid="child", sampling_params=SimpleNamespace(custom_params={
        CUSTOM_REQUEST_ID: "req", CUSTOM_GENERATION: 2, CUSTOM_PARENT_GENERATION: 1,
    }))
    Scheduler._agentic_abort_cleanup(scheduler, req)
    assert calls == [("child", parent)]
    assert scheduler.agentic_tp_host_cancelled_requests == {"req:1": "child"}
    assert scheduler.agentic_tp_host_active_requests == {"req:1": parent}
    req.rid = "cancelled-retry"
    Scheduler._agentic_abort_cleanup(scheduler, req)
    # Do not lose the original owner's pending intent/frozen TP allocation.
    assert scheduler.agentic_tp_host_cancelled_requests == {"req:1": "child"}


def test_abort_cas_failure_holds_control_until_retry_resolves():
    value = host_manager()
    value.owner = "p-group:prefill-0"
    value._control_wakeup = threading.Event()
    def fail(*args, **kwargs):
        raise OSError("injected ledger failure")
    value.ledger = SimpleNamespace(request_host_load_failure=fail)
    value.abort_request("child", RequestGeneration("req", 1))
    assert not value.tp_host_control_quiescent("req:1", "child")
    # Retry discovers foreign ownership: leave its payload/ledger untouched,
    # but the old P's purely local command may now retire.
    value.ledger.request_host_load_failure = lambda *args, **kw: False
    value.abort_request("child", RequestGeneration("req", 1))
    assert value.tp_host_control_quiescent("req:1", "child")


def test_cancelled_control_requires_both_ranks_and_not_just_missing_http(tmp_path):
    parent = RequestGeneration("req", 1)
    ranks = []
    for rank in range(2):
        value = SimpleNamespace(
            tp_rank=rank,
            agentic_tp_host_command_visible=True,
            agentic_tp_host_active_requests={"req:1": parent},
            _agentic_tp_host_actions={"req:1": "prepare"},
            agentic_tp_host_cancelled_requests={},
            agentic_tp_host_local_admitted=set(),
            agentic_tp_host_group_statuses={},
            agentic_kv_waiting_queue=[],
            agentic_tp_host_mailbox=TPGroupMailbox(
                "d2p-host:0", tp_rank=rank, tp_size=2, directory=str(tmp_path)
            ),
            agentic_host_staging_manager=host_manager(),
        )
        ranks.append(value)
    for value in ranks:
        Scheduler._agentic_tp_reduce_host_status(value)
        # Late HTTP arrival is NOT cancellation; never retire it on absence.
        assert value.agentic_tp_host_mailbox.local_status("req:1") == 0
        value.agentic_tp_host_cancelled_requests["req:1"] = "child"
    ranks[1].agentic_host_staging_manager._h2d_lane_reservations = {"req:1": 0}
    for value in ranks:
        Scheduler._agentic_tp_reduce_host_status(value)
    Scheduler._agentic_tp_reduce_host_status(ranks[0])
    assert ranks[0].agentic_tp_host_group_statuses["req:1"] == 0
    ranks[1].agentic_host_staging_manager._h2d_lane_reservations.clear()
    Scheduler._agentic_tp_reduce_host_status(ranks[1])
    Scheduler._agentic_tp_reduce_host_status(ranks[0])
    assert Scheduler._agentic_tp_host_next_action(
        ranks[0].agentic_tp_host_group_statuses["req:1"]
    ) == "clear"


def test_actual_scheduler_host_namespaces_isolate_redirected_p_groups(tmp_path, monkeypatch):
    # Evaluate the production namespace expression, not a separately invented
    # test naming convention; this catches reversion to the old shared name.
    tree = ast.parse(inspect.getsource(Scheduler))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == "agentic_tp_host_mailbox"
                           for t in node.targets)
                   and isinstance(node.value, ast.Call)]
    assert len(assignments) == 1
    expression = ast.Expression(assignments[0].value.args[0])
    code = compile(ast.fix_missing_locations(expression), "namespace", "eval")
    groups = []
    for domain in (0, 1):
        monkeypatch.setenv("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", str(domain))
        namespace = eval(code, {"os": os})
        groups.append(TPGroupMailbox(namespace, tp_rank=0, tp_size=2, directory=str(tmp_path)))
    assert groups[0].directory != groups[1].directory
    groups[1].publish_local("req:1", 3)
    groups[0].publish_local("req:1", 0)
    groups[0].clear_local("req:1")
    groups[0].clear_group("req:1")
    assert groups[1].local_status("req:1") == 3


def test_same_parent_retry_cannot_start_io_before_cancel_clear():
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError("new retry must not touch the cancelling I/O")
    for rank in range(2):
        value = SimpleNamespace(
            tp_size=2, tp_rank=rank,
            agentic_tp_host_cancelled_requests={"req:1": "old-http"},
            _agentic_bind_early_direct_receive=forbidden,
            agentic_host_staging_manager=SimpleNamespace(gate_request=forbidden),
        )
        req = SimpleNamespace(rid="retry-http", sampling_params=SimpleNamespace(custom_params={
            CUSTOM_REQUEST_ID: "req", CUSTOM_GENERATION: 2, CUSTOM_PARENT_GENERATION: 1,
        }))
        assert Scheduler._agentic_should_defer(value, req, 0.0)
        assert not getattr(req, "_agentic_kv_gate_complete", False)
        value.agentic_tp_host_cancelled_requests.clear()
        value._agentic_bind_early_direct_receive = lambda *args, **kw: calls.append(rank) or True
        assert Scheduler._agentic_should_defer(value, req, 0.0)
    assert calls == [0, 1]


@pytest.mark.parametrize("status", [0, 1, 2, 3, 4])
def test_cancelled_producer_never_commits_stale_peer_progress(status):
    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled Host attempt must not commit its manifest")
    parent = RequestGeneration("req", 1)
    owner = SimpleNamespace(
        tp_size=2, tp_rank=0,
        disaggregation_mode=DisaggregationMode.PREFILL,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
        agentic_tp_direct_admission_active={},
        disagg_prefill_inflight_queue=[],
        agentic_tp_p2d_sender_mailbox=None,
        agentic_tp_p2d_receiver_mailbox=None,
        agentic_tp_host_active_requests={"req:1": parent},
        agentic_tp_host_active_since_by_snapshot={"req:1": 0.0},
        agentic_tp_host_group_statuses={"req:1": status},
        agentic_tp_host_cancelled_requests={"req:1": "old-http"},
        agentic_kv_waiting_queue=[],
        agentic_host_staging_manager=SimpleNamespace(
            max_h2d_inflight=1, _complete_shared_host_manifest=forbidden,
        ),
    )
    control = Scheduler._agentic_tp_prepare_admission_control(owner)
    assert control["host_commands"][0]["action"] == ("clear" if status == 4 else "abort")
