"""Actual Scheduler entry points: controller owns pages, native header owns compute."""
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    CUSTOM_REQUEST_ID, CUSTOM_GENERATION, CUSTOM_PARENT_GENERATION,
)
from sglang.srt.disaggregation.test_agentic_workset_admission import group
from sglang.srt.disaggregation.test_agentic_tp_host_pipeline import leader
from sglang.srt.managers.scheduler import Scheduler


def metadata(req, parent=False):
    fields = {CUSTOM_REQUEST_ID: "session", CUSTOM_GENERATION: 0}
    if parent:
        fields[CUSTOM_PARENT_GENERATION] = 0
        fields[CUSTOM_GENERATION] = 1
    req.sampling_params = NS(custom_params=fields)


@pytest.mark.parametrize("size", [2, 8])
def test_native_header_prepares_then_admits_fresh_without_parent_commands(size):
    _, _, reports, members = group(size, registered_ranks={0})
    ranks = []
    for rank, member in enumerate(members):
        metadata(member.req)
        s = leader(tp_size=size)
        s.tp_rank = rank
        s.agentic_host_staging_manager = None
        s.agentic_kv_waiting_queue = [(member.req, 0)]
        s.agentic_fresh_workset_admission = member.a
        member.b.controller_mode = True
        member.b.install_tp_plan = lambda *a, **kw: pytest.fail("native allocation plan")
        member.b.prepare_tp_control = lambda *a, **kw: pytest.fail("native allocation decision")
        s.agentic_p_workset_broker = member.b
        s.agentic_tp_direct_group_status = {}
        ranks.append(s)
    reports.hold.add(size - 1)
    header = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert not header["direct_commands"] and not header["host_commands"]
    assert not header["workset_allocation_plan"]
    assert header["fresh_workset_control"]["commands"][0]["action"] == "prepare"
    for s in ranks:
        assert Scheduler._agentic_tp_consume_admission_control(s, [header]) == []
    assert all(Scheduler._agentic_should_defer(s, member.req, 0) for s, member in zip(ranks, members))
    header = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert not header["fresh_workset_control"]["commands"]
    for s in ranks:
        Scheduler._agentic_tp_consume_admission_control(s, [header])
    reports.flush()
    header = Scheduler._agentic_tp_prepare_admission_control(ranks[0])
    assert header["fresh_workset_control"]["commands"][0]["action"] == "commit"
    for s, member in zip(ranks, members):
        Scheduler._agentic_tp_consume_admission_control(s, [header])
        assert not Scheduler._agentic_should_defer(s, member.req, 0)


def test_fresh_and_recompute_cannot_bypass_complete_workset(monkeypatch):
    monkeypatch.setattr(Scheduler, "_agentic_should_defer_parent", lambda *a, **kw: False)
    registered = []
    admission = NS(selected=lambda req: False, register=lambda *a, **kw: registered.append((a, kw)))
    s = NS(agentic_fresh_workset_admission=admission, tp_rank=0)
    req = NS(rid="r", extra_key="agentic-v1:session:g0")
    metadata(req)
    assert Scheduler._agentic_should_defer(s, req, 0)
    assert registered[-1][1] == {"owner": "fresh"}
    metadata(req, parent=True)
    assert Scheduler._agentic_should_defer(s, req, 0)
    assert registered[-1][1] == {"owner": "recompute"}
    req._agentic_workset_backed = True
    assert not Scheduler._agentic_should_defer(s, req, 0)
    assert len(registered) == 2


def test_device_ready_is_not_parent_dma_ready(monkeypatch):
    monkeypatch.setattr(Scheduler, "_agentic_should_defer_parent", lambda *a, **kw: True)
    admission = NS(selected=lambda req: False, register=lambda *a, **kw: pytest.fail("recomputed pending parent"))
    req = NS(_agentic_workset_backed=False)
    assert Scheduler._agentic_should_defer(NS(agentic_fresh_workset_admission=admission), req, 0)


def test_legacy_wrapper_keeps_parent_result(monkeypatch):
    for value in (True, False):
        monkeypatch.setattr(Scheduler, "_agentic_should_defer_parent", lambda *a, **kw: value)
        assert Scheduler._agentic_should_defer(NS(), NS(), 0) is value


def test_pending_abort_does_not_free_provisional_native_reference():
    calls = []
    req = NS()
    s = NS(agentic_fresh_workset_admission=NS(pending=lambda r: True, cancel=lambda r: calls.append(r)))
    Scheduler._agentic_abort_cleanup(s, req)
    assert calls == [req]


@pytest.mark.parametrize("size", [1, 2, 8])
def test_common_ingress_observes_all_ranks_but_only_leader_scan_requests(size):
    from sglang.srt.disaggregation.utils import DisaggregationMode
    _, _, _, members = group(size, registered_ranks=set())
    for rank, member in enumerate(members):
        # Use a genuinely empty native admission table, as at HTTP ingress.
        admission = type(member.a)(member.b, member.a.mailbox, rank=rank,
            tp_size=size, on_abort=lambda *args: True)
        metadata(member.req)
        calls = []
        request = member.b.request
        member.b.request = lambda *args, request=request, **kwargs: (
            calls.append(args), request(*args, **kwargs))[1]
        s = NS(agentic_fresh_workset_admission=admission, tp_rank=rank,
            disaggregation_mode=DisaggregationMode.PREFILL,
            agentic_tp_prefill_sequence=0, agentic_kv_waiting_queue=[],
            _agentic_publish_p_accepted=lambda req: None)
        Scheduler._add_request_to_queue(s, member.req)
        assert admission.contains(member.req) and not admission.selected(member.req)
        assert not calls
        assert s.agentic_kv_waiting_queue[0][0] is member.req
        assert Scheduler._agentic_should_defer(s, member.req, 0)
        assert bool(calls) is (rank == 0)
        assert admission.selected(member.req) is (rank == 0)
