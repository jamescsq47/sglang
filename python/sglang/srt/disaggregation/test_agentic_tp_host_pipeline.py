"""TP Host admission must use physical lanes, with one leader-ordered plan."""
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    CUSTOM_GENERATION, CUSTOM_PARENT_GENERATION, CUSTOM_REQUEST_ID,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler


@pytest.mark.parametrize("tp_size", [2, 8])
def test_delayed_handed_ack_cannot_split_native_prefill_admission(tp_size):
    from concurrent.futures import Future
    from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
    parent = RequestGeneration("r0", 1)
    reports = {}
    completions = [Future() for _ in range(tp_size)]
    ranks = []
    for rank in range(tp_size):
        state = leader(tp_size=tp_size)
        state.tp_rank = rank
        req = state.agentic_kv_waiting_queue[0][0]
        req._agentic_host_rank_loaded = True
        state.agentic_kv_waiting_queue = [(req, 0)]
        state.agentic_tp_host_active_requests = {parent.snapshot_id: parent}
        state.agentic_tp_host_command_visible = True
        state._agentic_tp_host_actions = {parent.snapshot_id: "commit"}
        state.agentic_tp_host_local_admitted = set()
        state.agentic_host_staging_manager.ledger = SimpleNamespace(is_event_control=True)
        state.agentic_host_staging_manager.loads = {}
        state.agentic_host_staging_manager.tp_host_control_quiescent = lambda *args: True
        state.agentic_tp_host_mailbox = SimpleNamespace(
            publish_local=lambda key, value, rank=rank: reports.__setitem__(rank, value),
            group_status=lambda key: min(reports.values()) if len(reports) == tp_size else None)
        ranks.append(state)
    def apply_and_batch(control):
        batches = []
        for state in ranks:
            state.agentic_p_workset_broker = SimpleNamespace(install_tp_plan=lambda *args, **kwargs: None)
            state.agentic_tp_direct_group_status = {}
            Scheduler._agentic_tp_consume_admission_control(state, [control])
            del state.agentic_p_workset_broker
            # Exercise the real waiting-queue consumer after native command
            # installation. A local ACK alone cannot change its batch.
            drain = SimpleNamespace(**vars(state))
            queued = []
            drain.agentic_kv_waiting_queue = list(state.agentic_kv_waiting_queue)
            drain._agentic_bind_completed_waiters = lambda: None
            drain._agentic_io_active = lambda req: True
            drain._agentic_io_kind = lambda req: "slow"
            drain._agentic_queue_class = lambda req: "slow"
            drain._agentic_publish_p_scheduled = lambda req: None
            drain._add_request_to_queue = lambda req: queued.append(req.rid)
            drain._agentic_should_defer = lambda req, *args, **kwargs: not (
                req._agentic_host_handoff_ready
                and parent.snapshot_id in drain.agentic_host_staging_manager.tp_host_admit_snapshots)
            Scheduler._drain_agentic_kv_waiting_queue(drain)
            batches.append(queued)
        return batches
    # Half the HTTP/RPC completions return first, exactly like the live stall.
    for future in completions[:tp_size // 2]:
        future.set_result(True)
    for rank, state in enumerate(ranks):
        state.agentic_kv_waiting_queue[0][0]._agentic_host_handoff_ready = completions[rank].done()
        Scheduler._agentic_tp_reduce_host_status(state)
    Scheduler._agentic_tp_reduce_host_status(ranks[0])
    control = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert [command["action"] for command in control["host_commands"]] == ["commit"]
    assert min(reports.values()) == 3  # no rank can enter Forward yet
    assert apply_and_batch(control) == [[] for _ in ranks]
    for rank in range(tp_size // 2, tp_size):
        completions[rank].set_result(True)
        ranks[rank].agentic_kv_waiting_queue[0][0]._agentic_host_handoff_ready = True
        Scheduler._agentic_tp_reduce_host_status(ranks[rank])
    Scheduler._agentic_tp_reduce_host_status(ranks[0])
    control = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert [command["action"] for command in control["host_commands"]] == ["admit"]
    assert apply_and_batch(control) == [["r0"] for _ in ranks]
    # A cancellation after readiness wins over admission; 4 no longer means
    # quiescent cleanup and cannot clear a possibly live workset.
    ranks[0].agentic_tp_host_cancelled_requests = {parent.snapshot_id: "r0"}
    control = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert control["host_commands"][0]["action"] == "abort"
    assert apply_and_batch(control) == [[] for _ in ranks]
    for state in ranks:
        state.agentic_tp_host_cancelled_requests = {parent.snapshot_id: "r0"}
        Scheduler._agentic_tp_reduce_host_status(state)
    Scheduler._agentic_tp_reduce_host_status(ranks[0])
    control = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert control["host_commands"][0]["action"] == "clear"


def leader(lanes=4, tp_size=8):
    reqs = [SimpleNamespace(rid=f"r{i}", sampling_params=SimpleNamespace(
        custom_params={CUSTOM_REQUEST_ID: f"r{i}", CUSTOM_GENERATION: 2,
                       CUSTOM_PARENT_GENERATION: 1})) for i in range(6)]
    return SimpleNamespace(
        tp_size=tp_size, tp_rank=0, disaggregation_mode=DisaggregationMode.PREFILL,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
        agentic_tp_direct_admission_active={}, disagg_prefill_inflight_queue=[],
        agentic_tp_p2d_sender_mailbox=None, agentic_tp_p2d_receiver_mailbox=None,
        agentic_tp_host_active_requests={}, agentic_tp_host_active_since_by_snapshot={},
        agentic_tp_host_group_statuses={},
        agentic_host_staging_manager=SimpleNamespace(
            max_h2d_inflight=lanes, snapshot_ready=lambda parent: True,
            _complete_shared_host_manifest=lambda parent: True),
        _agentic_tp_host_next_action=Scheduler._agentic_tp_host_next_action,
        agentic_kv_waiting_queue=[(req, float(i)) for i, req in enumerate(reqs)],
    )


@pytest.mark.parametrize("tp_size", [2, 8])
def test_background_handed_receipt_survives_stale_scheduler_observation(tp_size):
    """Only a native group command admits; old Req flags cannot erase ACKs."""
    from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager
    from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration

    parent = RequestGeneration("r0", 1)
    reports, ranks = {}, []
    for rank in range(tp_size):
        state = leader(tp_size=tp_size)
        state.tp_rank = rank
        req = state.agentic_kv_waiting_queue[0][0]
        req._agentic_host_rank_loaded = True
        state.agentic_kv_waiting_queue = [(req, 0)]
        state.agentic_tp_host_command_visible = True
        state.agentic_tp_host_active_requests = {parent.snapshot_id: parent}
        state._agentic_tp_host_actions = {parent.snapshot_id: "commit"}
        state.agentic_tp_host_local_admitted = set()
        state.agentic_tp_host_mailbox = SimpleNamespace(
            publish_local=lambda key, status, rank=rank: reports.__setitem__(rank, status),
            group_status=lambda key: min(reports.values()) if len(reports) == tp_size else None,
        )
        manager = object.__new__(AgenticPHostStagingManager)
        manager.ledger = SimpleNamespace(is_event_control=True)
        manager.loads = {}
        manager.max_h2d_inflight = 4
        manager._complete_shared_host_manifest = lambda parent: True
        manager.register_tp_host_progress(parent.snapshot_id, parent.snapshot_id,
                                          state.agentic_tp_host_mailbox.publish_local)
        state.agentic_host_staging_manager = manager
        Scheduler._agentic_tp_reduce_host_status(state)
        ranks.append(state)

    for state in ranks[:-1]:
        manager = state.agentic_host_staging_manager
        manager._tp_host_progress[parent.snapshot_id]["ready"] = True
        manager.publish_tp_host_status(parent.snapshot_id, parent.snapshot_id, 4)
        # No scheduler has changed _agentic_host_handoff_ready on live Req.
        Scheduler._agentic_tp_reduce_host_status(state)
    Scheduler._agentic_tp_reduce_host_status(ranks[0])
    control = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert control["host_commands"][0]["action"] == "commit"
    manager = ranks[-1].agentic_host_staging_manager
    manager._tp_host_progress[parent.snapshot_id]["ready"] = True
    manager.publish_tp_host_status(parent.snapshot_id, parent.snapshot_id, 4)
    Scheduler._agentic_tp_reduce_host_status(ranks[0])
    control = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert control["host_commands"][0]["action"] == "admit"


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("override,expected", [(None, 4), ("1", 1), ("20", 4)])
def test_default_uses_all_lanes_and_override_never_exceeds_physical(monkeypatch, tp_size, override, expected):
    monkeypatch.delenv("SGLANG_AGENTIC_KV_TP_HOST_PIPELINE_DEPTH", raising=False)
    monkeypatch.delenv("SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT", raising=False)
    if override is not None:
        monkeypatch.setenv("SGLANG_AGENTIC_KV_TP_HOST_PIPELINE_DEPTH", override)
    value = leader(tp_size=tp_size)
    control = Scheduler._agentic_tp_prepare_admission_control(value)
    assert [c["snapshot"] for c in control["host_commands"]] == [f"r{i}:1" for i in range(expected)]
    assert all(c["action"] == "prepare" for c in control["host_commands"])
    # Pending phases cannot admit a fifth request or skip all-rank barriers.
    value.agentic_tp_host_group_statuses.update({"r0:1": 1})
    again = Scheduler._agentic_tp_prepare_admission_control(value)
    assert len(again["host_commands"]) == expected
    assert again["host_commands"][0]["action"] == "start"
    assert all(c["action"] == "prepare" for c in again["host_commands"][1:])


def test_tp1_and_follower_never_make_group_selection():
    value = leader(tp_size=1)
    assert Scheduler._agentic_tp_prepare_admission_control(value) is None
    value.tp_size, value.tp_rank = 8, 1
    assert Scheduler._agentic_tp_prepare_admission_control(value) is None
    assert not value.agentic_tp_host_active_requests


def initial_request(rid):
    return SimpleNamespace(
        rid=rid, _agentic_kv_queue_class="new",
        sampling_params=SimpleNamespace(custom_params={
            CUSTOM_REQUEST_ID: rid, CUSTOM_GENERATION: 0,
        }),
    )


def test_leader_includes_bounded_initial_admission_alongside_host(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "2")
    value = leader()
    value.agentic_kv_waiting_queue.extend(
        (initial_request(f"new{i}"), float(i + 10)) for i in range(5)
    )
    control = Scheduler._agentic_tp_prepare_admission_control(value)
    assert len(control["host_commands"]) == 4
    assert control["ordinary_prefill_rids"] == ["new0", "new1"]


def drain_state(tp_size, ordinary, *, forced_present=True, restore_ready=False):
    host = leader().agentic_kv_waiting_queue[0][0]
    queued, visited = [], []
    value = SimpleNamespace(
        tp_size=tp_size,
        _agentic_tp_host_actions={"r0:1": "prepare"},
        _agentic_tp_ordinary_prefill_rids=("new0", "new1"),
        agentic_host_staging_manager=SimpleNamespace(),
        agentic_kv_waiting_queue=(
            ([(host, 0.0)] if forced_present else [])
            + [(req, float(i + 1)) for i, req in enumerate(ordinary)]
        ),
        _agentic_bind_completed_waiters=lambda: None,
        _agentic_io_active=lambda req: False,
        _agentic_io_kind=lambda req: None,
        _agentic_queue_class=lambda req: req._agentic_kv_queue_class,
        _agentic_publish_p_scheduled=lambda req: None,
        _add_request_to_queue=lambda req: queued.append(req.rid),
    )
    host._agentic_kv_queue_class = "slow"

    def defer(req, *args, **kwargs):
        visited.append(req.rid)
        return req is host and not restore_ready

    value._agentic_should_defer = defer
    return value, queued, visited


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("forced_present", [False, True])
@pytest.mark.parametrize("restore_ready", [False, True])
def test_tp_restore_cannot_starve_initial_requests(
    monkeypatch, tp_size, forced_present, restore_ready,
):
    # Even a ready restore spending the entire old budget cannot exclude New.
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "1")
    orders = []
    for rank in range(tp_size):
        ordinary = [initial_request("new0"), initial_request("new1")]
        if rank % 2:
            ordinary.reverse()
        state, queued, visited = drain_state(
            tp_size, ordinary, forced_present=forced_present,
            restore_ready=restore_ready,
        )
        # Leader only authorizes one request for this tick.
        state._agentic_tp_ordinary_prefill_rids = ("new0",)
        Scheduler._drain_agentic_kv_waiting_queue(state)
        assert "new0" in queued
        assert "new1" not in queued
        orders.append(tuple(queued))
        # Continuous restores must not prevent the next ordinary request.
        state._agentic_tp_ordinary_prefill_rids = ("new1",)
        Scheduler._drain_agentic_kv_waiting_queue(state)
        assert queued.count("new0") == queued.count("new1") == 1
        for _ in range(10):
            Scheduler._drain_agentic_kv_waiting_queue(state)
        assert queued.count("new1") == 1
    assert len(set(orders)) == 1


def test_cancelled_initial_command_cannot_admit_an_unselected_retry():
    state, queued, visited = drain_state(8, [initial_request("new0-retry")])
    state._agentic_tp_ordinary_prefill_rids = ("new0",)
    Scheduler._drain_agentic_kv_waiting_queue(state)
    assert not queued
    assert any(req.rid == "new0-retry" for req, _ in state.agentic_kv_waiting_queue)


def test_tp1_ignores_tp_ordinary_plan_and_preserves_arrival_order():
    state, queued, visited = drain_state(
        1, [initial_request("new1"), initial_request("new0")],
        forced_present=False,
    )
    Scheduler._drain_agentic_kv_waiting_queue(state)
    assert queued == ["new1", "new0"]


@pytest.mark.parametrize("tp_size", [2, 8])
def test_all_rank_copy_complete_reuses_lanes_but_keeps_commit_context(tp_size):
    value = leader(tp_size=tp_size)
    value.agentic_host_staging_manager.h2d_lane_overlap = True
    Scheduler._agentic_tp_prepare_admission_control(value)
    value.agentic_tp_host_group_statuses.update({f"r{i}:1": 2 for i in range(4)})
    plan = Scheduler._agentic_tp_prepare_admission_control(value)
    assert [c["action"] for c in plan["host_commands"]] == ["bind"] * 4 + ["prepare"] * 2
    assert len(value.agentic_tp_host_active_requests) == 6


def test_tp_overlap_does_not_treat_failure_as_free_lane():
    value = leader()
    value.agentic_host_staging_manager.h2d_lane_overlap = True
    Scheduler._agentic_tp_prepare_admission_control(value)
    value.agentic_tp_host_group_statuses.update({f"r{i}:1": -1 for i in range(4)})
    plan = Scheduler._agentic_tp_prepare_admission_control(value)
    assert len(plan["host_commands"]) == 4
    assert all(c["action"] == "abort" for c in plan["host_commands"])


def test_tp_overlap_resident_limit_even_if_commit_stalls():
    value = leader(lanes=2)
    value.agentic_host_staging_manager.h2d_lane_overlap = True
    Scheduler._agentic_tp_prepare_admission_control(value)
    value.agentic_tp_host_group_statuses.update({"r0:1": 2, "r1:1": 2})
    Scheduler._agentic_tp_prepare_admission_control(value)
    value.agentic_tp_host_group_statuses.update({"r2:1": 2, "r3:1": 2})
    plan = Scheduler._agentic_tp_prepare_admission_control(value)
    assert len(plan["host_commands"]) == 4


@pytest.mark.parametrize("tp_size", [2, 8])
def test_tp_physical_recycle_preserves_lease_and_host_until_group_handoff(tp_size):
    from sglang.srt.disaggregation.test_agentic_h2d_decoupling import manager, completed_load
    m = manager(enabled=False, lanes=1)
    m.tp_size = tp_size
    m.h2d_lane_overlap = True
    old = completed_load(m, "old")
    old["io_quiesced"] = False
    m._release_quiesced_h2d_lane(old)
    assert m._reserve_h2d_lane("new:0") is None
    old["io_quiesced"] = True
    m._release_quiesced_h2d_lane(old)
    assert m._reserve_h2d_lane("new:0") == 0
    assert old["record"] and old["workset_lease"]
    assert m.h2d_selected_snapshots() == {"old:0", "new:0"}
    m._release_h2d_lane("old:0")
    assert m._h2d_lane_reservations == {"new:0": 0}


@pytest.mark.parametrize("tp_size", [2, 8])
def test_tp_granted_prestart_intents_progress_when_all_physical_lanes_owned(tp_size):
    # Reproduce R17: allocator grants arrived on the next scheduler boundary,
    # before a load object existed. Own lane/workset != requesting a NEW lane.
    from sglang.srt.disaggregation.test_agentic_h2d_decoupling import manager
    m = manager(enabled=False, lanes=4)
    m.h2d_lane_overlap = True
    reqs = [entry[0] for entry in leader(tp_size=tp_size).agentic_kv_waiting_queue[:4]]
    state, _, _ = drain_state(tp_size, [], forced_present=False)
    state.agentic_host_staging_manager = m
    state._agentic_tp_ordinary_prefill_rids = ()
    state._agentic_tp_host_actions = {f"r{i}:1": "prepare" for i in range(4)}
    state.agentic_kv_waiting_queue = [(req, float(i)) for i, req in enumerate(reqs)]
    state._agentic_io_kind = lambda req: Scheduler._agentic_io_kind(state, req)
    state._agentic_io_active = lambda req: Scheduler._agentic_io_active(state, req)
    for i, req in enumerate(reqs):
        req._agentic_kv_queue_class = "slow"
        assert m._reserve_h2d_lane(f"r{i}:1") is not None
    assert m.h2d_physical_occupancy() == 4
    assert not m.loads
    started = []
    def gate(req, *args, allow_start_io=True):
        if allow_start_io:
            started.append(req.rid)
            m.loads[req.rid] = {}
        return True
    state._agentic_should_defer = gate
    Scheduler._drain_agentic_kv_waiting_queue(state)
    assert started == [f"r{i}" for i in range(4)]
    assert m.h2d_physical_occupancy() == 4
