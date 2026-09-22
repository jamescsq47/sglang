"""Native admission headers with real CPU leases and controllable TP receipts."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_workset_admission import (
    DuplicateGenerationRequest, FreshWorksetAdmission,
)
from sglang.srt.disaggregation.agentic_workset_device import prepare_workset
from sglang.srt.disaggregation.agentic_workset_ledger import WorksetLedger
from sglang.srt.disaggregation.agentic_workset_tp import ReadyCut


class Reports:
    def __init__(self, size):
        self.size, self.rows, self.delayed = size, {}, {}
        self.hold = set()

    def mailbox(self, rank):
        def publish(key, status):
            target = self.delayed if rank in self.hold and key.attempt_id.startswith("fresh-prepare:") else self.rows
            target.setdefault(key, {})[rank] = status
        def group(key):
            assert rank == 0, "only the leader can reduce TP reports"
            reports = self.rows.get(key, {})
            return min(reports.values()) if len(reports) == self.size else None
        def any_negative(key):
            assert rank == 0, "only the leader can reduce TP reports"
            return any(status < 0 for status in self.rows.get(key, {}).values())
        return NS(publish_local=publish, group_status=group, any_negative_report=any_negative)

    def flush(self):
        for key, ranks in self.delayed.items():
            self.rows.setdefault(key, {}).update(ranks)
        self.delayed.clear()
        self.hold.clear()


def group(size, *, owner="fresh", grant=True, mailboxes=None, registered_ranks=None):
    ledger = WorksetLedger(incarnation="run", page_count=8, page_size=4, tp_size=size)
    plan = ledger.grant("fresh:session:0:r", "grant1", owner=owner, parent_tokens=0, prompt_tokens=9)
    reports, members = Reports(size), []
    for rank in range(size):
        broker = AgenticPWorksetLeaseBroker(4)
        broker.runtime = NS(ready_cut=lambda: ReadyCut("run", 999, (plan.key,)))
        req = NS(rid="r", origin_input_ids=list(range(9)), extra_key="agentic-v1:session:g0",
                 req_pool_idx=None, prefix_indices=torch.empty(0, dtype=torch.int64),
                 mamba_pool_idx=None, mamba_ping_pong_track_buffer=None)
        aborts = []
        def abort(req, lease, broker=broker, aborts=aborts):
            aborts.append(req)
            if lease is not None and lease.state == "handed":
                broker.release_handed(lease.snapshot_id, lease, req=req)
            else:
                broker.cancel_unstarted("fresh:session:0:r")
            return True
        admission = FreshWorksetAdmission(broker, reports.mailbox(rank) if mailboxes is None else mailboxes[rank], rank=rank,
                                         tp_size=size, on_abort=abort)
        admission.observe(req, "session:0", req.extra_key)
        if registered_ranks is None or rank in registered_ranks:
            admission.register(req, "session:0", req.extra_key, owner=owner)
        if grant:
            broker.install_prepared(prepare_workset(plan, device="cpu", page_capacity=8))
        members.append(NS(a=admission, req=req, b=broker, aborts=aborts))
    return ledger, plan, reports, members


def build(members):
    return members[0].a.build_control([members[0].req])


def apply(members, header):
    for member in members:
        member.a.apply_control(header, {member.req.rid: member.req})


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("owner", ["fresh", "recompute"])
def test_native_prepare_registers_unscanned_followers(size, owner):
    _, plan, _, members = group(size, owner=owner, registered_ranks={0})
    for member in members[1:]:
        assert member.a.contains(member.req)
        assert not member.a.selected(member.req)
        assert not member.b._intents  # Observation never queues allocation.
    apply(members, build(members))
    assert all(m.a.selected(m.req) and m.req._agentic_p_workset_lease.controller_plan == plan
               for m in members)
    apply(members, build(members))
    assert all(m.a.is_committed(m.req) for m in members)


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("grant", [False, True])
def test_cancelled_unscanned_req_is_retained_for_exact_native_abort(size, grant):
    _, _, _, members = group(size, grant=grant, registered_ranks={0})
    # HTTP removed this Req from the ordinary waiting queue on all ranks.
    for member in members:
        member.a.terminal(member.req)
    header = members[0].a.build_control([])
    assert header["commands"][0]["action"] == "abort"
    for member in members:
        member.a.apply_control(header, member.a.req_by_rid())
        assert not member.b._intents
        assert member.aborts == [member.req]
    header = members[0].a.build_control([])
    assert header["commands"][0]["action"] == "forget"
    for member in members:
        member.a.apply_control(header, member.a.req_by_rid())
        assert not member.a.contains(member.req)


def test_late_prepare_does_not_revive_cancelled_unselected_follower():
    _, _, _, members = group(2, registered_ranks={0})
    header = build(members)
    follower = members[1]
    follower.a.cancel(follower.req)
    follower.b.request = lambda *a, **kw: pytest.fail("cancelled header created intent")
    apply(members, header)
    assert follower.a._requests[follower.req.rid].failed
    assert not follower.a.is_committed(follower.req)
    apply(members, build(members))
    assert all(m.aborts == [m.req] for m in members)


def test_observed_parent_terminal_does_not_create_fresh_allocation():
    _, _, _, members = group(2, grant=False, registered_ranks=set())
    for member in members:
        member.b.request = lambda *a, **kw: pytest.fail("parent observation created allocation")
        member.a.on_abort = lambda *a: pytest.fail("metadata retirement touched parent references")
        member.req.req_pool_idx = 3  # Parent/P->D still owns native references.
        assert not member.a.pending(member.req)
        assert not member.a.terminal(member.req)
    apply(members, build(members))
    apply(members, build(members))
    assert all(not m.a.contains(m.req) for m in members)
    assert all(m.req.req_pool_idx == 3 for m in members)


@pytest.mark.parametrize("size", [1, 2, 8])
@pytest.mark.parametrize("owner", ["fresh", "recompute"])
def test_prepare_then_group_commit_never_allocates_or_waits_for_global_frontier(size, owner):
    ledger, plan, reports, members = group(size, owner=owner)
    before = ledger.counts
    header = build(members)
    assert [item["action"] for item in header["commands"]] == ["prepare"]
    apply(members, header)
    assert all(member.b.get("fresh:session:0:r").state == "handed" for member in members)
    assert all(member.a.defer_fresh(member.req) == (size > 1) for member in members)
    if size > 1:
        header = build(members)
        assert header["commands"][0]["action"] == "commit"
        apply(members, header)
    assert all(not member.a.defer_fresh(member.req) for member in members)
    assert ledger.counts == before
    # A later ready cut need not retain this chunked, already-owned request.
    members[0].b.runtime.ready_cut = lambda: ReadyCut("run", 1000, ())
    apply(members, build(members))
    assert all(not member.a.defer_fresh(member.req) for member in members)


@pytest.mark.parametrize("size", [2, 8])
def test_one_delayed_prepare_ack_keeps_all_rank_batches_empty(size):
    _, _, reports, members = group(size)
    reports.hold.add(size - 1)
    apply(members, build(members))
    empty = build(members)
    assert not empty["commands"]
    apply(members, empty)
    assert all(member.a.defer_fresh(member.req) for member in members)
    reports.flush()
    commit = build(members)
    apply(members, commit)
    assert [[m.req.rid] if not m.a.defer_fresh(m.req) else [] for m in members] == [["r"]] * size


@pytest.mark.parametrize("size", [2, 8])
def test_cancel_after_commit_header_freeze_is_next_ordered_abort_not_rank_veto(size):
    ledger, _, _, members = group(size)
    apply(members, build(members))
    commit = build(members)
    members[-1].a.cancel(members[-1].req)
    apply(members, commit)
    assert all(not member.a.defer_fresh(member.req) for member in members)
    assert all(member.b.get("fresh:session:0:r").state == "handed" for member in members)
    before = ledger.counts
    abort = build(members)
    assert abort["commands"][0]["action"] == "abort"
    apply(members, abort)
    assert all(member.a.defer_fresh(member.req) for member in members)
    assert all(len(member.aborts) == 1 for member in members)
    assert ledger.counts == before  # logical abort does not fabricate physical fences
    apply(members, abort)  # duplicate native header is side-effect-free
    assert all(len(member.aborts) == 1 for member in members)


@pytest.mark.parametrize("size", [2, 8])
def test_cancel_before_prepare_prevents_any_commit_and_partial_handoff_is_aborted(size):
    _, _, _, members = group(size)
    prepare = build(members)
    members[-1].a.cancel(members[-1].req)
    apply(members, prepare)
    assert members[-1].b.get("fresh:session:0:r").state == "active"
    assert all(member.a.defer_fresh(member.req) for member in members)
    abort = build(members)
    assert abort["commands"][0]["action"] == "abort"
    apply(members, abort)
    assert all(len(member.aborts) == 1 for member in members)


def test_cancel_pending_without_grant_and_async_native_abort_retry():
    _, _, _, members = group(2, grant=False)
    members[-1].a.cancel(members[-1].req)
    original = members[-1].a.on_abort
    members[-1].a.on_abort = lambda req, lease: False
    apply(members, build(members))
    assert members[0].a._requests["r"].aborted
    with pytest.raises(RuntimeError, match="forget"):
        members[0].a.forget(members[0].req)
    retry = build(members)
    assert retry["commands"][0]["action"] == "abort"
    members[-1].a.on_abort = original
    apply(members, retry)
    assert all(len(member.aborts) == 1 for member in members)
    apply(members, build(members))  # common retirement, no follower group read
    for member in members:
        member.a.forget(member.req)
        assert not member.a._requests


@pytest.mark.parametrize("size", [2, 8])
def test_last_rank_handoff_exception_is_failed_preparation_not_forward(size):
    _, _, _, members = group(size)
    members[-1].b.handoff_fresh_to_req = lambda *args: (_ for _ in ()).throw(RuntimeError("handoff failed"))
    apply(members, build(members))
    assert all(member.a.defer_fresh(member.req) for member in members)
    header = build(members)
    assert header["commands"][0]["action"] == "abort"
    apply(members, header)


@pytest.mark.parametrize("size", [2, 8])
def test_prepare_failure_requests_abort_before_other_ranks_report(size):
    _, _, reports, members = group(size)
    reports.hold.add(0)
    members[-1].b.handoff_fresh_to_req = lambda *a: (_ for _ in ()).throw(RuntimeError("failed"))
    apply(members, build(members))
    key = members[0].a._prepare_key(members[0].a._requests["r"])
    assert members[0].a.mailbox.group_status(key) is None
    header = build(members)
    assert header["commands"][0]["action"] == "abort"
    apply(members, header)
    assert all(member.a.defer_fresh(member.req) for member in members)


def test_exact_generation_single_live_request_and_header_attempt_validation():
    _, _, _, members = group(2)
    member = members[0]
    duplicate = NS(**vars(member.req))
    duplicate.rid = "other"
    with pytest.raises(DuplicateGenerationRequest, match="one live") as error:
        member.a.register(duplicate, "session:0", duplicate.extra_key)
    assert error.value.existing_rid == member.req.rid
    assert error.value.incoming_rid == duplicate.rid
    assert error.value.snapshot_id == "session:0"
    assert member.a.contains(member.req) and not member.a.contains(duplicate)
    assert not member.a._requests[member.req.rid].cancelled
    with pytest.raises(ValueError, match="namespace"):
        member.a.register(duplicate, "session:1", duplicate.extra_key)
    header = build(members)
    apply(members, header)
    corrupt = deepcopy(header)
    corrupt["commands"][0]["nonce"] += 1
    with pytest.raises(RuntimeError, match="replay"):
        member.a.apply_control(corrupt, {"r": member.req})
    next_header = build(members)
    next_header["commands"][0]["key"][3] += 1
    with pytest.raises(RuntimeError, match="attempt"):
        member.a.apply_control(next_header, {"r": member.req})


def test_cancelled_request_removed_from_waiting_still_gets_native_abort():
    _, _, _, members = group(2)
    apply(members, build(members))
    members[-1].a.cancel(members[-1].req)
    header = members[0].a.build_control([])
    assert header["commands"][0]["action"] == "abort"
    for member in members:
        member.a.apply_control(header, member.a.req_by_rid())
    apply(members, members[0].a.build_control([]))
    assert all(not member.a.contains(member.req) for member in members)


def test_fresh_allocator_owner_does_not_collide_with_next_turn_parent_snapshot():
    ledger, plan, _, members = group(1)
    assert members[0].a.workset_id(members[0].req) == "fresh:session:0:r"
    parent = ledger.grant("session:0", "parent-direct", owner="direct", parent_tokens=4, prompt_tokens=8)
    assert parent.key != plan.key and parent.key.snapshot_id == "session:0"


def test_committed_order_comes_from_common_header_not_local_registration_order():
    ledger, plan, _, members = group(2)
    second_plan = ledger.grant("fresh:other:0:s", "grant2", owner="fresh", parent_tokens=0, prompt_tokens=9)
    for member in members:
        req = NS(**vars(member.req))
        req.rid, req.extra_key = "s", "agentic-v1:other:g0"
        member.a.register(req, "other:0", req.extra_key)
        member.b.install_prepared(prepare_workset(second_plan, device="cpu", page_capacity=8))
        member.b.runtime.ready_cut = lambda: ReadyCut("run", 2, (plan.key, second_plan.key))
    follower = members[1].a
    follower._requests = dict(reversed(tuple(follower._requests.items())))
    waiting = list(members[0].a.req_by_rid().values())
    for _ in range(2):
        header = members[0].a.build_control(waiting)
        for member in members:
            member.a.apply_control(header, member.a.req_by_rid())
    assert [member.a.committed_rids() for member in members] == [("r", "s")] * 2


@pytest.mark.parametrize("size", [2, 8])
def test_real_socket_receipts_and_single_follower_cancel_without_file_io(size, monkeypatch):
    from sglang.srt.disaggregation.agentic_tp_events import TPEventClient, TPEventServer
    from sglang.srt.disaggregation.agentic_tp_socket_mailbox import SocketTPGroupMailbox
    server = TPEventServer("admission", "secret")
    clients = [TPEventClient(server.address, run_id="admission", token="secret",
                            group="P", rank=rank, size=size) for rank in range(size)]
    try:
        for client in clients:
            client.wait_ready()
        mailboxes = [SocketTPGroupMailbox("native-workset-admission", tp_rank=rank,
                     tp_size=size, client=client) for rank, client in enumerate(clients)]
        monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("control file access"))
        _, _, _, members = group(size, mailboxes=mailboxes)
        apply(members, build(members))
        leader = members[0]
        prepare_key = leader.a._prepare_key(leader.a._requests["r"])
        with clients[0]._condition:
            assert clients[0]._condition.wait_for(lambda: mailboxes[0].group_status(prepare_key) == 1, 5)
        commit = build(members)
        assert commit["commands"][0]["action"] == "commit"
        members[-1].a.cancel(members[-1].req)
        apply(members, commit)
        cancel_key = leader.a._cancel_key(leader.a._requests["r"])
        with clients[0]._condition:
            assert clients[0]._condition.wait_for(lambda: mailboxes[0].any_negative_report(cancel_key), 5)
        apply(members, build(members))
        abort_key = leader.a._abort_key(leader.a._requests["r"])
        with clients[0]._condition:
            assert clients[0]._condition.wait_for(lambda: mailboxes[0].group_status(abort_key) == 1, 5)
        apply(members, build(members))
        assert all(not member.a.contains(member.req) for member in members)
    finally:
        for client in clients:
            client.close()
        server.close()


def test_tp8_c128_ingress_is_silent_and_sparse_cancellation_still_retires():
    from sglang.srt.disaggregation.agentic_tp_events import TPEventClient, TPEventServer
    from sglang.srt.disaggregation.agentic_tp_socket_mailbox import SocketTPGroupMailbox
    server = TPEventServer("burst", "secret")  # unchanged bounded queue=1024
    clients = [TPEventClient(server.address, run_id="burst", token="secret",
                            group="P", rank=rank, size=8) for rank in range(8)]
    try:
        members = []
        for rank, client in enumerate(clients):
            client.wait_ready()
            mailbox = SocketTPGroupMailbox("native-workset-admission", tp_rank=rank,
                                            tp_size=8, client=client)
            broker = AgenticPWorksetLeaseBroker(4)
            broker.runtime = NS(ready_cut=lambda: ReadyCut("run", 0, ()))
            members.append(FreshWorksetAdmission(broker, mailbox, rank=rank, tp_size=8,
                           on_abort=lambda *a: pytest.fail("unallocated metadata has no physical ownership")))
        sequences = [client._seq for client in clients]
        for member in members:
            for i in range(128):
                req = NS(rid=f"r{i}", origin_input_ids=list(range(9)),
                         extra_key=f"agentic-v1:s{i}:g0")
                member.observe(req, f"s{i}:0", req.extra_key)
        assert [client._seq for client in clients] == sequences
        assert server._entry_count == 0
        leader, follower = members[0], members[-1]
        # Actual cancels/cleanup, in bounded work waves. No neutral peer reports.
        for start in range(0, 128, 16):
            records = [follower._requests[f"r{i}"] for i in range(start, start + 16)]
            for record in records:
                follower.cancel(record.req)
            with clients[0]._condition:
                assert clients[0]._condition.wait_for(lambda: all(
                    leader.mailbox.any_negative_report(leader._cancel_key(r)) for r in records), 5)
            header = leader.build_control([])
            assert len(header["commands"]) == 16
            for member in members[:-1]:
                member.apply_control(header, member.req_by_rid())
            # One rank has not performed cleanup: no forget may be issued.
            retry = leader.build_control([])
            assert all(c["action"] == "abort" for c in retry["commands"])
            follower.apply_control(header, follower.req_by_rid())
            for member in members:
                member.apply_control(retry, member.req_by_rid())
            with clients[0]._condition:
                assert clients[0]._condition.wait_for(lambda: all(
                    leader.mailbox.group_status(leader._abort_key(r)) == 1 for r in records), 5)
            header = leader.build_control([])
            assert all(c["action"] == "forget" for c in header["commands"])
            for member in members:
                member.apply_control(header, member.req_by_rid())
        assert all(not member._requests for member in members)
        assert not server.errors
    finally:
        for client in clients:
            client.close()
        server.close()
