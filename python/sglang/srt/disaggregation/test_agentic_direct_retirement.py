"""Real broker/worker cancellation races; no CUDA or control server required."""

from types import SimpleNamespace as NS
import threading

import pytest

from sglang.srt.disaggregation import agentic_direct_control as control
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, SnapshotState
from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager
from sglang.srt.disaggregation.agentic_tp_events import ControlUnavailable
from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.test_agentic_direct_control import finish, setup  # noqa: F401
from sglang.srt.disaggregation.test_agentic_direct_descriptors import Allocator
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker, Scheduler


def workset(request, tp_size=8, *, owner=None):
    broker, allocator = AgenticPWorksetLeaseBroker(4), Allocator(64)
    sid = request.snapshot_id
    broker.request(sid, 8, 12, owner=owner or broker.direct_owner(sid))
    if tp_size > 1:
        broker.prepare_tp_control(1)
    broker.service(allocator)
    return broker, allocator, broker.get(sid)


def scheduler(request, broker, lease, tp_size=8):
    owner = Scheduler.__new__(Scheduler)
    owner.tp_size, owner.tp_rank = tp_size, 0
    owner.disaggregation_mode = DisaggregationMode.PREFILL
    owner.agentic_p_workset_broker = broker
    owner.agentic_tp_direct_admission_active = {
        request.snapshot_id: (request, 0, None, 12, lease)
    }
    owner.agentic_early_direct_receives = {}
    owner.agentic_early_direct_terminal = {}
    owner.agentic_tp_direct_local_failed = set()
    owner.agentic_tp_direct_local_admitted = set()
    owner.agentic_tp_direct_local_rolled_back = set()
    owner.agentic_tp_direct_group_status = {}
    owner.agentic_early_direct_poll_lock = threading.RLock()
    owner.rollback_acks = []
    owner.agentic_tp_direct_mailbox = NS(
        receipt=lambda _sid: -1,
        publish_local_rollback_complete=owner.rollback_acks.append,
    )
    return owner


def abort_command(broker, sid):
    return {
        Scheduler._AGENTIC_TP_CONTROL_KEY: True,
        "workset_plan_epoch": 2,
        "workset_allocation_plan": broker._tp_plan,
        "workset_retire_commands": [{"snapshot": sid, "action": "prepare"}],
        "direct_commands": [{"snapshot": sid, "action": "abort"}],
        "host_commands": [],
    }


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("retire", ["native", "release"])
@pytest.mark.parametrize("posted", [False, True])
def test_tp_retirement_preserves_unposted_vs_physical_fence(tp_size, retire, posted):
    request = RequestGeneration("retirement-fence", 6)
    sid = request.snapshot_id
    broker, allocator, lease = workset(request, tp_size)
    assert broker.begin_io_attempt(sid, lease, "exact")
    if posted:
        broker.mark_io_inflight(sid, lease, "exact")
    if retire == "native":
        assert not broker.prepare_tp_retire(sid)
    else:
        assert not broker.request_release(sid, lease, io_attempt="exact")
    assert not broker.tp_retire_ready(sid)
    assert not broker.cancel_io_attempt(sid, lease, "stale-attempt")
    broker.service(allocator)
    assert allocator.available_size() == 52
    if posted:
        assert lease.state == "release_pending"
        assert not broker.cancel_io_attempt(sid, lease, "exact")
        assert not broker.commit_tp_retire(sid)
        assert broker.mark_io_quiesced(sid, lease, "exact")
    else:
        assert lease.state == "io_reserved"
        with pytest.raises(RuntimeError, match="retiring workset"):
            broker.mark_io_inflight(sid, lease, "exact")
        assert lease.state == "io_reserved" and lease.io_attempt == "exact"
        assert broker.cancel_io_attempt(sid, lease, "exact")
        broker.request_release(sid, lease)
    assert broker.tp_retire_ready(sid)
    broker.service(allocator)
    assert allocator.available_size() == 52  # Local ACK is not all-rank commit.
    assert broker.commit_tp_retire(sid)
    broker.service(allocator)
    assert allocator.available_size() == 64
    assert broker.get(sid) is None


def test_tp1_unplanned_release_keeps_original_semantics():
    request = RequestGeneration("tp1-compatibility", 1)
    broker, allocator, lease = workset(request, 1)
    sid = request.snapshot_id
    assert broker.begin_io_attempt(sid, lease, "exact")
    assert not broker.request_release(sid, lease, io_attempt="exact")
    assert lease.state == "release_pending"
    assert not broker.cancel_io_attempt(sid, lease, "exact")
    assert broker.mark_io_quiesced(sid, lease, "exact")
    broker.service(allocator)
    assert allocator.available_size() == 64


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("unknown", [False, True])
def test_real_pending_claim_native_abort_waits_for_ack_then_retires(setup, tp_size, unknown):
    sid = setup.request.snapshot_id
    broker, allocator, lease = workset(setup.request, tp_size)
    assert broker.begin_io_attempt(sid, lease, setup.claim)
    call = control.DirectControlCall(
        (sid, 12, lease.lease_id, setup.claim), control.claim_direct,
        setup.store, setup.request, setup.claim, 12,
    )
    assert call.poll() == (False, None)
    owner = scheduler(setup.request, broker, None, tp_size)  # Stale native tuple.
    owner._agentic_direct_claim_calls = {sid: (lease, setup.request, call)}
    command = abort_command(broker, sid)
    owner._agentic_tp_consume_admission_control([command])
    assert owner.rollback_acks == []
    assert lease.state == "io_reserved"
    owner._agentic_progress_cancelled_direct_claims()
    assert not call.settled and sid in owner._agentic_direct_claim_calls
    if unknown:
        future, _, _ = setup.client.queue.pop()
        future.set_exception(ControlUnavailable("ACK lost"))
        owner._agentic_progress_cancelled_direct_claims()
        owner._agentic_tp_consume_admission_control([command])
        assert not call.settled and sid in owner._agentic_direct_claim_calls
        assert owner.rollback_acks == [] and lease.state == "io_reserved"
        broker.service(allocator)
        assert allocator.available_size() == 52
        return
    setup.client.execute()
    owner._agentic_progress_cancelled_direct_claims()
    assert call.settled and sid not in owner._agentic_direct_claim_calls
    assert lease.state == "active" and lease.io_attempt is None
    owner._agentic_tp_consume_admission_control([command])
    assert owner.rollback_acks == [sid]
    assert allocator.available_size() == 52
    assert broker.commit_tp_retire(sid)
    broker.service(allocator)
    assert allocator.available_size() == 64


def test_settled_claim_context_cannot_disappear_before_physical_quiescence(setup):
    sid = setup.request.snapshot_id
    broker, _, lease = workset(setup.request)
    assert broker.begin_io_attempt(sid, lease, setup.claim)
    broker.mark_io_inflight(sid, lease, setup.claim)
    broker.prepare_tp_retire(sid)
    call = NS(identity=(sid, 12, lease.lease_id, setup.claim), settled=True,
              poll=lambda **_kw: (True, None))
    owner = scheduler(setup.request, broker, lease)
    owner._agentic_direct_claim_calls = {sid: (lease, setup.request, call)}
    owner._agentic_progress_cancelled_direct_claims()
    assert sid in owner._agentic_direct_claim_calls
    assert lease.state == "release_pending"
    assert broker.mark_io_quiesced(sid, lease, setup.claim)
    owner._agentic_progress_cancelled_direct_claims()
    assert not owner._agentic_direct_claim_calls
    assert lease.state == "retire_ready"


@pytest.mark.parametrize("replacement", [False, True])
def test_stale_none_grant_cancels_only_actual_direct_owner(monkeypatch, replacement):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test")
    request = RequestGeneration("stale-tuple", 6)
    sid = request.snapshot_id
    broker, allocator, lease = workset(request, owner="slow:new-owner" if replacement else None)
    owner = scheduler(request, broker, None)
    owner._agentic_snapshot_store = lambda: NS(
        load=lambda *_a, **_kw: NS(state=SnapshotState.SLOW_FALLBACK)
    )
    assert not owner._agentic_tp_start_direct_shard(request, arrived_at=0, prefill_domain=None)
    assert (sid in broker.tp_retire_candidates) is not replacement
    command = abort_command(broker, sid)
    if replacement:
        command["workset_retire_commands"] = []
    owner._agentic_tp_consume_admission_control([command])
    assert owner.rollback_acks == [sid]
    if replacement:
        assert sid not in broker.tp_retire_candidates
        assert lease.state == "active"
    else:
        assert broker.commit_tp_retire(sid)
        broker.service(allocator)
        assert allocator.available_size() == 64


def test_old_claim_context_does_not_cancel_replacement_slow(setup):
    sid = setup.request.snapshot_id
    broker, allocator, old = workset(setup.request)
    broker.prepare_tp_retire(sid)
    broker.commit_tp_retire(sid)
    broker.service(allocator)
    broker.install_tp_plan(2, ())
    broker.request(sid, 8, 12, owner="slow:replacement")
    broker.prepare_tp_control(3)
    broker.service(allocator)
    current = broker.get(sid)
    assert current is not None and current.lease_id != old.lease_id
    assert broker.begin_io_attempt(sid, current, "new-attempt")
    call = NS(identity=(sid, 12, old.lease_id, setup.claim), settled=True,
              poll=lambda **_kw: (True, None))
    owner = scheduler(setup.request, broker, None)
    owner._agentic_direct_claim_calls = {sid: (old, setup.request, call)}
    owner._agentic_progress_cancelled_direct_claims()
    assert not owner._agentic_direct_claim_calls
    assert current.state == "io_reserved" and current.io_attempt == "new-attempt"
    assert not broker.cancel_io_attempt(sid, old, "new-attempt")


@pytest.mark.parametrize("tp_size", [2, 8])
def test_retirement_between_claim_and_metadata_is_normal_cancellation(setup, monkeypatch, caplog, tp_size):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "p")
    sid = setup.request.snapshot_id
    broker, allocator, lease = workset(setup.request)
    assert broker.begin_io_attempt(sid, lease, setup.claim)
    call = control.DirectControlCall(
        (sid, 12, lease.lease_id, setup.claim), control.claim_direct,
        setup.store, setup.request, setup.claim, 12,
    )
    claimed = finish(setup, call)
    owner = scheduler(setup.request, broker, lease, tp_size)
    from dataclasses import replace
    claimed = replace(claimed, tp_size=tp_size)
    owner._agentic_direct_claim_calls = {sid: (lease, setup.request, call)}
    owner.agentic_tp_direct_mailbox.receipt = lambda _sid: 1
    owner.tree_cache = NS(is_eagle=False)
    cleared = []
    receiver = NS(
        init=lambda **_kw: broker.prepare_tp_retire(sid),
        poll=lambda: KVPoll.WaitingForInput,
        send_metadata=lambda *_a, **_kw: pytest.fail("retired destination published"),
        started_transfer=False, clear=lambda: cleared.append(True),
    )
    owner.agentic_direct_runtime = NS(
        layout_hash="", manager=NS(try_ensure_parallel_info=lambda _addr: True),
        receiver_class=lambda **_kw: receiver,
    )
    owner._agentic_clear_direct_receiver = lambda *_a: None
    assert not owner._agentic_start_early_direct_receive(
        setup.request, claimed, setup.store, arrived_at=0, workset_lease=lease
    )
    assert cleared == [True]
    assert lease.state == "active" and lease.io_attempt is None
    assert sid in owner.agentic_tp_direct_local_failed
    assert not owner.agentic_early_direct_receives
    assert not owner._agentic_direct_claim_calls
    assert broker.tp_retire_ready(sid)
    assert allocator.available_size() == 52  # Still waits for all-rank retirement.
    assert "Could not start early Direct" not in caplog.text
    assert "retiring workset" not in caplog.text


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("cancel_first", [False, True])
def test_submission_and_cancellation_share_one_atomic_boundary(tp_size, cancel_first):
    request = RequestGeneration("atomic-submit", 1)
    broker, allocator, lease = workset(request, tp_size)
    sid = request.snapshot_id
    assert broker.begin_io_attempt(sid, lease, "exact")
    ready, proceed = threading.Event(), threading.Event()
    results, errors = [], []

    def submit():
        try:
            ready.set()
            assert proceed.wait(3)
            results.append(broker.try_mark_io_inflight(sid, lease, "exact"))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=submit)
    worker.start()
    assert ready.wait(3)
    if cancel_first:
        assert broker.cancel_direct_before_bind(sid, lease)
    proceed.set()
    worker.join(3)
    assert not worker.is_alive() and not errors
    assert results == [not cancel_first]
    if not cancel_first:
        assert broker.cancel_direct_before_bind(sid, lease)
        assert lease.state == "release_pending"
        assert not broker.cancel_io_attempt(sid, lease, "exact")
        assert not broker.commit_tp_retire(sid)
        assert broker.mark_io_quiesced(sid, lease, "exact")
    else:
        assert lease.state == "io_reserved"
        assert broker.cancel_io_attempt(sid, lease, "exact")
    assert broker.commit_tp_retire(sid)
    broker.service(allocator)
    assert allocator.available_size() == 64


def test_abort_phases_do_not_repeat_while_waiting_for_socket_echo(monkeypatch):
    request = RequestGeneration("abort-once", 1)
    broker, _, lease = workset(request)
    owner = scheduler(request, broker, lease)
    decisions, releases = [], []
    owner.agentic_tp_direct_mailbox.publish_receipt = lambda *a: decisions.append(a)
    # A stale cached receipt must not cause a second finalization or re-publish -1.
    owner.agentic_tp_direct_mailbox.receipt = lambda _sid: -1
    broker.request_release = lambda *a, **kw: releases.append(a) or True
    reads = []
    store = NS(load=lambda *a, **kw: reads.append(a) or NS(state=SnapshotState.SLOW_FALLBACK))
    for _ in range(10):
        assert owner._agentic_abort_tp_direct_grant(request, store, reason="rank_failure")
    assert decisions == [(request.snapshot_id, -1)]
    for _ in range(10):
        assert owner._agentic_abort_tp_direct_grant(
            request, store, reason="all_ranks_rolled_back", rolled_back=True)
        assert owner._agentic_abort_tp_direct_grant(request, store, reason="late_failure")
    assert decisions == [(request.snapshot_id, -1), (request.snapshot_id, -2)]
    assert len(reads) == len(releases) == 1
    # A genuinely different active grant is not hidden by an old completion.
    owner.agentic_tp_direct_admission_active[request.snapshot_id] = (request, 1, None, 12, lease)
    assert owner._agentic_abort_tp_direct_grant(request, store, reason="new_grant")
    assert len(decisions) == 3


def test_abort_finalization_failure_is_retryable():
    request = RequestGeneration("abort-retry", 1)
    broker, _, lease = workset(request)
    owner = scheduler(request, broker, lease)
    decisions, releases = [], []
    owner.agentic_tp_direct_mailbox.publish_receipt = lambda *a: decisions.append(a)
    broker.request_release = lambda *a, **kw: releases.append(a) or True
    reads = []

    def load(*a, **kw):
        reads.append(a)
        if len(reads) == 1:
            raise OSError("temporary read failure")
        return NS(state=SnapshotState.SLOW_FALLBACK)

    store = NS(load=load)
    assert not owner._agentic_abort_tp_direct_grant(request, store, reason="retry", rolled_back=True)
    assert not decisions and not releases
    assert owner._agentic_abort_tp_direct_grant(request, store, reason="retry", rolled_back=True)
    assert owner._agentic_abort_tp_direct_grant(request, store, reason="retry", rolled_back=True)
    assert len(reads) == 2 and len(releases) == len(decisions) == 1


def test_stale_abort_cannot_finalize_a_replaced_grant(monkeypatch):
    request = RequestGeneration("abort-replaced", 1)
    broker, _, lease = workset(request)
    owner = scheduler(request, broker, lease)
    original = Scheduler._agentic_abort_tp_direct_grant_once
    replacement = (request, 2, None, 12, lease)

    def replace_before_execute(self, *args, **kwargs):
        self.agentic_tp_direct_admission_active[request.snapshot_id] = replacement
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Scheduler, "_agentic_abort_tp_direct_grant_once", replace_before_execute)
    owner.agentic_tp_direct_mailbox.publish_receipt = lambda *a: pytest.fail("stale decision")
    store = NS(load=lambda *a, **kw: pytest.fail("stale lifecycle read"))
    assert not owner._agentic_abort_tp_direct_grant(request, store, reason="stale", rolled_back=True)
    assert owner.agentic_tp_direct_admission_active[request.snapshot_id] is replacement
    assert not owner._agentic_tp_direct_abort_phases
    assert lease.state == "active"


def test_abort_control_wait_does_not_hold_scheduler_lock_or_duplicate_finish():
    request = RequestGeneration("abort-nonblocking", 1)
    broker, _, lease = workset(request)
    owner = scheduler(request, broker, lease)
    entered, proceed = threading.Event(), threading.Event()
    decisions, releases, errors = [], [], []
    owner.agentic_tp_direct_mailbox.publish_receipt = lambda *a: decisions.append(a)
    broker.request_release = lambda *a, **kw: releases.append(a) or True

    def load(*a, **kw):
        entered.set()
        assert proceed.wait(3)
        return NS(state=SnapshotState.SLOW_FALLBACK)

    store = NS(load=load)

    def finish_abort():
        try:
            assert owner._agentic_abort_tp_direct_grant(request, store, reason="done", rolled_back=True)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=finish_abort)
    worker.start()
    try:
        assert entered.wait(3)
        assert owner.agentic_early_direct_poll_lock.acquire(blocking=False)
        owner.agentic_early_direct_poll_lock.release()
        assert not owner._agentic_abort_tp_direct_grant(request, store, reason="duplicate", rolled_back=True)
        assert not decisions and not releases
    finally:
        proceed.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert len(decisions) == len(releases) == 1


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("posted", [False, True])
@pytest.mark.parametrize("retry", [False, True])
def test_slow_cancel_and_retry_share_reserved_vs_posted_boundary(tp_size, posted, retry):
    request = RequestGeneration("slow-retirement", 6)
    sid = request.snapshot_id
    broker, allocator, lease = workset(request, tp_size, owner="slow:claim")
    assert broker.begin_io_attempt(sid, lease, "slow-attempt")
    if posted:
        broker.mark_io_inflight(sid, lease, "slow-attempt")
    broker.prepare_tp_retire(sid)
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.owner, manager.tp_rank, manager.tp_size = "p", 0, tp_size
    manager.workset_broker = broker
    acknowledgements = []
    manager.ledger = NS(
        cancel_d2p_recovery_rank=lambda *_a, **_kw: acknowledgements.append("cancel") or True,
        request_d2p_retry=lambda *_a, **_kw: True,
        complete_d2p_retry_rank=lambda *_a, **_kw: acknowledgements.append("retry") or True,
    )
    ready = [False]
    load = {"request_generation": request, "workset_lease": lease,
            "io_attempt": "slow-attempt", "recovery_claim_id": "slow:claim",
            "io_inflight": posted, "record": {"loading": True}}
    if posted:
        load["event"] = NS(query=lambda: ready[0], synchronize=lambda: None)
    manager.loads, manager.host_ready = {"rid": load}, {}
    manager._h2d_lane_reservations = {sid: 0}
    if not retry and posted:
        with pytest.raises(RuntimeError, match="cannot cancel started"):
            manager._cancel_unstarted_h2d_load("rid", load)
        assert not acknowledgements
        assert lease.state == "release_pending"
        assert manager.loads["rid"] is load
        assert allocator.available_size() == 52
        return
    if retry:
        if posted:
            assert not manager._discard_failed_h2d_load("rid", load)
            assert lease.state == "release_pending" and not acknowledgements
            ready[0] = True
        assert manager._discard_failed_h2d_load("rid", load)
        assert acknowledgements == ["retry"]
    else:
        manager._cancel_unstarted_h2d_load("rid", load)
        assert acknowledgements == ["cancel"]
    assert not manager.loads and not manager._h2d_lane_reservations
    assert lease.io_attempt is None and broker.tp_retire_ready(sid)
    assert allocator.available_size() == 52
    assert broker.commit_tp_retire(sid)
    broker.service(allocator)
    assert allocator.available_size() == 64


@pytest.mark.parametrize("tp_size", [2, 8])
def test_all_ranks_retire_only_after_delayed_claim_and_dma_then_allow_slow(setup, tmp_path, monkeypatch, tp_size):
    # Real per-rank brokers, native command producer/consumer, and all-rank
    # mailbox reduction. The test-only mailbox directory is not runtime NFS.
    monkeypatch.delenv("SGLANG_AGENTIC_TP_EVENT_ENDPOINT", raising=False)
    sid = setup.request.snapshot_id
    brokers = [AgenticPWorksetLeaseBroker(4) for _ in range(tp_size)]
    allocators = [Allocator(64) for _ in brokers]
    brokers[0].request(sid, 8, 12, owner=brokers[0].direct_owner(sid))
    plan = brokers[0].prepare_tp_plan(1)
    for broker in brokers[1:]:
        broker.install_tp_plan(1, plan)
    owners, leases = [], []
    for rank, (broker, allocator) in enumerate(zip(brokers, allocators)):
        broker.service(allocator)
        lease = broker.get(sid)
        leases.append(lease)
        owner = scheduler(setup.request, broker, None, tp_size)
        owner.tp_rank = rank
        owner.agentic_tp_workset_retire_group_statuses = {}
        owner.agentic_tp_workset_retire_mailbox = TPGroupMailbox(
            "retire-test", tp_rank=rank, tp_size=tp_size, directory=str(tmp_path)
        )
        owners.append(owner)
    assert brokers[0].begin_io_attempt(sid, leases[0], setup.claim)
    call = control.DirectControlCall(
        (sid, 12, leases[0].lease_id, setup.claim), control.claim_direct,
        setup.store, setup.request, setup.claim, 12,
    )
    call.poll()
    root = owners[0]
    root._agentic_direct_claim_calls = {sid: (leases[0], setup.request, call)}
    assert brokers[1].begin_io_attempt(sid, leases[1], "posted")
    brokers[1].mark_io_inflight(sid, leases[1], "posted")
    brokers[0].prepare_tp_retire(sid)
    root._agentic_tp_workset_plan_epoch = 1
    root.agentic_tp_p2d_sender_mailbox = None
    root.agentic_tp_p2d_receiver_mailbox = None
    root.agentic_host_staging_manager = None
    root.agentic_tp_host_active_requests = {}
    root.agentic_tp_host_active_since_by_snapshot = {}
    root.agentic_tp_host_group_statuses = {}
    root.agentic_kv_waiting_queue = []

    def broadcast():
        command = root._agentic_tp_prepare_admission_control()
        committed = command["workset_retire_commands"][0]["action"] == "commit"
        for owner, broker, allocator in zip(owners, brokers, allocators):
            owner._agentic_tp_consume_admission_control([command])
            if committed:
                assert broker.get(sid).state == "releasing"
                assert not broker.cancel_unstarted(sid, owner=broker.direct_owner(sid))
                assert sid not in broker.tp_retire_candidates
                assert sid not in broker._tp_cancel_pending
            broker.service(allocator)
            if committed:
                assert broker.get(sid) is None
                assert not broker.cancel_unstarted(sid, owner=broker.direct_owner(sid))
                assert sid not in broker.tp_retire_candidates
                assert sid not in broker._tp_cancel_pending
        return command

    def reduce():
        for owner in owners[1:] + owners[:1]:
            owner._agentic_tp_reduce_workset_retire_status()
        return root.agentic_tp_workset_retire_group_statuses[sid]

    command = broadcast()
    assert command["workset_retire_commands"][0]["action"] == "prepare"
    assert reduce() == 0
    assert not owners[0].rollback_acks and not owners[1].rollback_acks
    root._agentic_progress_cancelled_direct_claims()
    assert not call.settled and sid in root._agentic_direct_claim_calls
    setup.client.execute()
    root._agentic_progress_cancelled_direct_claims()
    assert call.settled and leases[0].state == "active"
    command = broadcast()
    assert command["workset_retire_commands"][0]["action"] == "prepare"
    assert reduce() == 0  # Claim ACK is not rank1's physical DMA fence.
    assert owners[0].rollback_acks and not owners[1].rollback_acks
    assert all(a.available_size() == 52 for a in allocators)
    assert brokers[1].mark_io_quiesced(sid, leases[1], "posted")
    command = broadcast()
    assert command["workset_retire_commands"][0]["action"] == "prepare"
    assert all(owner.rollback_acks for owner in owners)
    assert reduce() == 1
    assert all(a.available_size() == 52 for a in allocators)
    command = broadcast()
    assert command["workset_retire_commands"][0]["action"] == "commit"
    assert all(broker.get(sid) is None for broker in brokers)
    assert all(a.available_size() == 64 for a in allocators)

    # A later Slow incarnation of the same generation is allowed only after
    # every shard dropped the old Direct destination at that shared boundary.
    epoch = root._agentic_tp_workset_plan_epoch + 1
    empty = brokers[0].prepare_tp_plan(epoch)
    assert empty == ()
    for broker in brokers[1:]:
        broker.install_tp_plan(epoch, empty)
    assert brokers[0].request(sid, 8, 12, owner="slow:next-attempt")
    plan = brokers[0].prepare_tp_plan(epoch + 1)
    for broker in brokers[1:]:
        broker.install_tp_plan(epoch + 1, plan)
    for broker, allocator, old in zip(brokers, allocators, leases):
        broker.service(allocator)
        current = broker.get(sid, owner="slow:next-attempt")
        assert current is not None and current.lease_id != old.lease_id
        assert current.state == "active" and allocator.available_size() == 52
        assert not broker.cancel_io_attempt(sid, old, setup.claim)
