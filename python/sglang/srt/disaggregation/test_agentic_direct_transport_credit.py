"""Transport credit is not the lifetime of a native-owned Direct workset."""

from dataclasses import replace
from types import SimpleNamespace as NS
import threading
import time

import pytest

from sglang.srt.disaggregation import agentic_direct_control as control
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, SnapshotState
from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.agentic_tp_events import ControlUnavailable, EventKey
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.test_agentic_direct_control import setup, claim  # noqa: F401
from sglang.srt.disaggregation.test_agentic_direct_early_authorization import group, admit
from sglang.srt.managers.scheduler import AgenticEarlyDirectReceive


@pytest.fixture(autouse=True)
def socket(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "4")
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    monkeypatch.delenv("SGLANG_AGENTIC_TP_EVENT_ENDPOINT", raising=False)


def configured_group(size, tmp_path, *, allocated=True, request=None):
    g = group(size)
    if request is not None:
        old_sid = g.request.snapshot_id
        g.request = request
        g.manifest = replace(g.manifest, request=request)
        g.store.current = g.manifest
        g.owners[0].agentic_early_direct_admission_queue.clear()
        g.owners[0].agentic_early_direct_admission_queue.append((request, g.payload, g.manifest))
        g.owners[0].agentic_early_direct_admission_ids.discard(old_sid)
        g.owners[0].agentic_early_direct_admission_ids.add(request.snapshot_id)
    sid = g.request.snapshot_id
    for rank, owner in enumerate(g.owners):
        client = owner.agentic_tp_direct_mailbox.client
        def status(namespace, key, _client=client):
            if g.shared.get("disconnected"):
                raise ControlUnavailable("test connection lost")
            values = [g.shared.get("reports", {}).get((namespace, key, r)) for r in range(size)]
            return None if None in values else min(values)
        client.group_status = status
        client.clear = lambda *_a: None
        owner.agentic_tp_workset_retire_group_statuses = {}
        owner.agentic_tp_workset_retire_mailbox = TPGroupMailbox(
            "credit-retire", tp_rank=rank, tp_size=size, directory=str(tmp_path))
        owner.agentic_early_direct_progress_thread = threading.current_thread()
        owner._agentic_clear_direct_receiver = lambda *_a: None
    admit(g)
    command = g.owners[0]._agentic_tp_prepare_admission_control()
    for owner, allocator in zip(g.owners, g.allocators):
        owner._agentic_tp_consume_admission_control([command])
        if allocated:
            owner.agentic_p_workset_broker.service(allocator)
            owner._agentic_tp_consume_admission_control([command])
    g.key = EventKey(sid, "direct-room:12")
    return g


def cancel(g):
    g.store.current = replace(g.manifest, state=SnapshotState.SLOW_FALLBACK)
    g.shared["receipts"][g.key] = -1


def progress(g):
    for owner in g.owners:
        owner._agentic_progress_tp_direct_grants(g.store)
    for owner in g.owners[1:] + g.owners[:1]:
        owner._agentic_commit_tp_direct_groups(g.store)


def receive(g, rank, *, posted=False, terminal=False):
    owner = g.owners[rank]
    sid = g.request.snapshot_id
    broker = owner.agentic_p_workset_broker
    lease = broker.get(sid)
    assert broker.begin_io_attempt(sid, lease, "physical")
    broker.mark_io_inflight(sid, lease, "physical")
    if terminal:
        assert broker.mark_io_quiesced(sid, lease, "physical")
    entry = AgenticEarlyDirectReceive(
        g.request, g.manifest, "physical", NS(clear=lambda: None), lease.parent_indices,
        time.monotonic(), time.time(), workset_lease=lease, io_attempt="physical",
        transport_poll=KVPoll.Failed if terminal else KVPoll.Transferring)
    owner.agentic_early_direct_receives[sid] = entry
    return entry


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("allocated", [False, True])
def test_unposted_group_returns_credit_before_native_clear_without_freeing_hbm(tmp_path, size, allocated):
    g = configured_group(size, tmp_path, allocated=allocated)
    root, sid = g.owners[0], g.request.snapshot_id
    cancel(g)
    # One local no-I/O proof is not the group proof.
    root._agentic_progress_tp_direct_grants(g.store)
    assert sid in root.agentic_tp_direct_local_rolled_back
    assert root._agentic_early_direct_slots_used() == 1
    progress(g)
    assert all(sid in o.agentic_tp_direct_local_rolled_back for o in g.owners)
    assert root._agentic_early_direct_slots_used() == 0
    assert sid in root.agentic_tp_direct_admission_active
    assert all(a.available_size() == (52 if allocated else 64) for a in g.allocators)
    # Frozen allocations may still arrive after the no-I/O ACK, but can never
    # start this obsolete owner. They do not create a fresh transport credit.
    for owner, allocator in zip(g.owners, g.allocators):
        broker = owner.agentic_p_workset_broker
        broker.service(allocator)
        lease = broker.get(sid)
        if lease is not None:
            assert not broker.begin_io_attempt(sid, lease, "late")
            assert not broker.begin_bind(sid, lease)
    assert root._agentic_early_direct_slots_used() == 0
    # Original native retire reduction/commit is still required for free.
    for _ in range(3):
        command = root._agentic_tp_prepare_admission_control()
        for owner, allocator in zip(g.owners, g.allocators):
            owner._agentic_tp_consume_admission_control([command])
            owner.agentic_p_workset_broker.service(allocator)
        for owner in g.owners[1:] + g.owners[:1]:
            owner._agentic_tp_reduce_workset_retire_status()
    assert all(a.available_size() == 64 for a in g.allocators)
    assert root._agentic_early_direct_slots_used() == 0


@pytest.mark.parametrize("size", [2, 8])
def test_drop_does_not_recreate_returned_lane_and_old_hbm_remains_real_capacity(tmp_path, size):
    g = configured_group(size, tmp_path)
    root, sid = g.owners[0], g.request.snapshot_id
    entries = [receive(g, rank, terminal=True) for rank in range(size)]
    for owner, entry in zip(g.owners, entries):
        entry.completed_at = time.monotonic()
        entry.group_committed = True  # Avoid another lifecycle CAS in this physical-credit test.
        owner.agentic_tp_direct_mailbox.publish_local_progress(sid, 3)
    root._agentic_commit_tp_direct_groups(g.store)
    assert root._agentic_early_direct_slots_used() == 0
    broker = root.agentic_p_workset_broker
    for i in range(3):
        broker.request(f"next-{i}:1", 8, 12, owner=broker.direct_owner(f"next-{i}:1"))
    assert root._agentic_early_direct_slots_used() == 3
    broker.request("next-3:1", 8, 12, owner=broker.direct_owner("next-3:1"))
    assert root._agentic_early_direct_slots_used() == 4
    cancel(g)
    progress(g)  # Real drop of the old receiver must not turn 4 into 5.
    assert not root.agentic_early_direct_receives
    assert root._agentic_early_direct_slots_used() == 4
    assert g.allocators[0].available_size() == 52
    # New worksets use the same native allocator while old HBM is retained.
    broker.prepare_tp_control(2)
    broker.service(g.allocators[0])
    assert g.allocators[0].available_size() == 4
    assert broker.get(sid) is entries[0].workset_lease
    assert not broker.request(sid, 8, 12, owner="slow-successor")


@pytest.mark.parametrize("size", [2, 8])
def test_posted_follower_holds_group_credit_until_actual_terminal(tmp_path, size):
    g = configured_group(size, tmp_path)
    root, follower, sid = g.owners[0], g.owners[-1], g.request.snapshot_id
    entry = receive(g, size - 1, posted=True)
    cancel(g)
    progress(g)
    assert sid not in follower.agentic_tp_direct_local_rolled_back
    assert root._agentic_early_direct_slots_used() == 1
    assert entry.workset_lease.state == "release_pending"
    assert all(a.available_size() == 52 for a in g.allocators)
    assert follower.agentic_p_workset_broker.mark_io_quiesced(sid, entry.workset_lease, "physical")
    entry.transport_poll = KVPoll.Failed
    progress(g)
    assert root._agentic_early_direct_slots_used() == 0
    assert all(a.available_size() == 52 for a in g.allocators)


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("unknown", [False, True])
def test_pending_claim_must_drain_before_worker_rollback(setup, tmp_path, size, unknown):
    g = configured_group(size, tmp_path, request=setup.request)
    root, sid = g.owners[0], g.request.snapshot_id
    lease = root.agentic_p_workset_broker.get(sid)
    assert root.agentic_p_workset_broker.begin_io_attempt(sid, lease, setup.claim)
    call = control.DirectControlCall((sid, 12, lease.lease_id, setup.claim),
                                    control.claim_direct, setup.store, setup.request, setup.claim, 12)
    call.poll()
    root._agentic_direct_claim_calls = {sid: (lease, g.request, call)}
    cancel(g)
    progress(g)
    assert not call.settled and root._agentic_early_direct_slots_used() == 1
    assert sid not in root.agentic_tp_direct_local_rolled_back
    if unknown:
        future, _, _ = setup.client.queue.pop()
        future.set_exception(ControlUnavailable("unknown claim ACK"))
        progress(g)
        assert not call.settled and sid in root._agentic_direct_claim_calls
        assert root._agentic_early_direct_slots_used() == 1
    else:
        setup.client.execute()
        progress(g)
        assert call.settled and sid not in root._agentic_direct_claim_calls
        assert root._agentic_early_direct_slots_used() == 0
    assert all(a.available_size() == 52 for a in g.allocators)


@pytest.mark.parametrize("state", ["binding", "handed", "consumed"])
def test_native_bind_winner_is_not_worker_rollback_even_without_prepared_req(tmp_path, state):
    g = configured_group(2, tmp_path)
    root, sid = g.owners[0], g.request.snapshot_id
    broker, lease = root.agentic_p_workset_broker, root.agentic_p_workset_broker.get(sid)
    assert broker.begin_bind(sid, lease)
    if state != "binding":
        broker.commit_parent_bound(sid, lease)
        req = NS(origin_input_ids=list(range(12)))
        broker.handoff_to_req(sid, req, lease)
        if state == "consumed":
            broker.consume_suffix(lease, 4, final_prompt_chunk=True)
    cancel(g)
    progress(g)
    assert sid not in root.agentic_tp_direct_local_rolled_back
    assert root._agentic_early_direct_slots_used() == 1
    assert not broker.owner_is_superseded(sid, owner=lease.owner)


def test_prepared_radix_rollback_remains_native_owned(tmp_path):
    g = configured_group(2, tmp_path)
    root, sid = g.owners[0], g.request.snapshot_id
    entry = receive(g, 0, terminal=True)
    broker = root.agentic_p_workset_broker
    assert broker.begin_bind(sid, entry.workset_lease)
    broker.commit_parent_bound(sid, entry.workset_lease)
    req = entry.prepared_req = NS(_agentic_direct_parent_pin_node=object(),
                                 _agentic_direct_parent_token_count=8)
    entry.radix_prepared = True
    released = []
    root.tree_cache = NS(dec_lock_ref=lambda pin: released.append("unpin"),
                        release_agentic_request_cache=lambda *_a, **_kw: released.append("radix"))
    cancel(g)
    progress(g)
    assert not released and entry.prepared_req is req
    assert sid not in root.agentic_tp_direct_local_rolled_back
    command = root._agentic_tp_prepare_admission_control()
    root._agentic_tp_consume_admission_control([command])
    assert released == ["unpin", "radix"]
    assert entry.prepared_req is None and not entry.radix_prepared
    assert sid in root.agentic_tp_direct_local_rolled_back


def test_receiver_clear_failure_is_sticky_and_cannot_ack_rollback(tmp_path):
    g = configured_group(2, tmp_path)
    root, sid = g.owners[0], g.request.snapshot_id
    entry = receive(g, 0, terminal=True)
    def fail():
        raise RuntimeError("clear did not acknowledge")
    entry.receiver.clear = fail
    cancel(g)
    progress(g)
    assert sid in root.agentic_early_direct_receives
    assert sid not in root.agentic_tp_direct_local_rolled_back
    # A later ordinary poll's drop must preserve the strict teardown proof.
    root._agentic_drop_early_direct_receive(entry, g.store, release_claim=False, reason="retry")
    assert sid in root.agentic_early_direct_receives
    assert root._agentic_early_direct_slots_used() == 1
    entry.receiver.clear = lambda: None
    progress(g)
    assert root._agentic_early_direct_slots_used() == 0


def test_posted_then_regular_poll_clear_failure_cannot_ack(tmp_path):
    g = configured_group(2, tmp_path)
    root, sid = g.owners[0], g.request.snapshot_id
    entry = receive(g, 0, posted=True)
    def fail():
        raise RuntimeError("terminal received, clear ACK missing")
    entry.receiver.clear = fail
    entry.receiver.poll = lambda: KVPoll.Failed
    cancel(g)
    root._agentic_progress_tp_direct_grants(g.store)
    assert entry.abort_requested and entry.abort_require_clear_success
    assert entry.workset_lease.state == "release_pending"
    root.agentic_direct_runtime = object()
    root._agentic_collect_direct_arrivals = lambda *_a: None
    root._agentic_collect_direct_abort_events = lambda: None
    root._agentic_handle_unstarted_direct_abort = lambda *_a: False
    root._agentic_poll_early_direct_receives_once()
    assert entry.io_quiesced and entry.transport_poll == KVPoll.Failed
    assert sid in root.agentic_early_direct_receives
    progress(g)
    assert sid not in root.agentic_tp_direct_local_rolled_back
    assert root._agentic_early_direct_slots_used() == 1
    entry.receiver.clear = lambda: None
    progress(g)
    assert root._agentic_early_direct_slots_used() == 0


@pytest.mark.parametrize("unknown", [False, True])
def test_completion_future_must_settle_before_original_rollback_ack(setup, tmp_path, unknown):
    g = configured_group(2, tmp_path, request=setup.request)
    root, sid = g.owners[0], g.request.snapshot_id
    claimed = claim(setup)
    entry = receive(g, 0, terminal=True)
    entry.manifest, entry.claim_id = claimed, setup.claim
    call = control.DirectControlCall(
        (sid, 12, entry.workset_lease.lease_id, setup.claim),
        control.complete_direct, setup.store, claimed, setup.claim)
    call.poll()
    entry._direct_control_calls = {"complete": call}
    cancel(g)
    progress(g)
    assert not call.settled and sid in root.agentic_early_direct_receives
    assert sid not in root.agentic_tp_direct_local_rolled_back
    if unknown:
        future, _, _ = setup.client.queue.pop()
        future.set_exception(ControlUnavailable("unknown completion ACK"))
    else:
        setup.client.execute()
    progress(g)
    assert root._agentic_early_direct_slots_used() == int(unknown)
    assert (sid in root.agentic_early_direct_receives) == unknown


@pytest.mark.parametrize("bind_first", [False, True])
def test_bind_and_worker_cancel_share_one_atomic_owner_boundary(tmp_path, bind_first):
    g = configured_group(2, tmp_path)
    root, sid = g.owners[0], g.request.snapshot_id
    broker, lease = root.agentic_p_workset_broker, root.agentic_p_workset_broker.get(sid)
    arrived, release = threading.Event(), threading.Event()
    result = []
    def contender():
        arrived.set()
        release.wait(2)
        result.append(broker.begin_bind(sid, lease))
    thread = threading.Thread(target=contender)
    thread.start()
    assert arrived.wait(2)
    if bind_first:
        release.set()
        thread.join(2)
        assert result == [True]
    cancel(g)
    root._agentic_progress_tp_direct_grants(g.store)
    if not bind_first:
        release.set()
        thread.join(2)
        assert result == [False]
    assert (sid in root.agentic_tp_direct_local_rolled_back) == (not bind_first)


def test_old_attempt_and_disconnect_cannot_return_successor_credit(tmp_path):
    g = configured_group(2, tmp_path)
    root, sid = g.owners[0], g.request.snapshot_id
    old_active = root.agentic_tp_direct_admission_active[sid]
    root.agentic_tp_direct_admission_active[sid] = tuple(list(old_active))
    root._agentic_return_direct_transport_credit(sid, old_active)
    assert root._agentic_early_direct_slots_used() == 1
    root.agentic_tp_direct_transport_returned = {sid: EventKey(sid, "direct-room:older")}
    assert root._agentic_early_direct_slots_used() == 1
    # All local ACKs do not substitute for an unavailable authoritative group view.
    cancel(g)
    for owner in g.owners:
        owner._agentic_progress_tp_direct_grants(g.store)
    g.shared["disconnected"] = True
    with pytest.raises(ControlUnavailable):
        root._agentic_commit_tp_direct_groups(g.store)
    assert root._agentic_early_direct_slots_used() == 1


def test_broker_exact_old_lease_cannot_tombstone_successor(tmp_path):
    g = configured_group(2, tmp_path)
    broker, sid = g.owners[0].agentic_p_workset_broker, g.request.snapshot_id
    old = broker.get(sid)
    newer = replace(old, lease_id=old.lease_id + 1, owner="slow:new")
    broker._leases[sid] = newer
    assert not broker.cancel_direct_before_bind(sid, old)
    assert not broker.cancel_direct_before_bind(sid, None)
    assert not broker.tp_retire_candidates
    assert not broker._superseded_owners
    assert broker.get(sid) is newer
