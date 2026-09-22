"""CPU-only real ownership tests for fused Slow BIND/handoff (no real DMA)."""
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_host_control import InMemoryHostStagingLedger
from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.test_agentic_direct_descriptors import Allocator
from sglang.srt.disaggregation.test_agentic_host_control import SID, OWNER, ready, age
from sglang.srt.disaggregation.test_agentic_tp_host_worker_handoff import manager, mock_manifest_future
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker, Scheduler
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


def prepared_group(size):
    """Real CPU leases/ledger; only the physical transport is a controlled Future."""
    ledger = InMemoryHostStagingLedger()
    ledger.is_event_control = True
    ready(ledger, size)
    items = []
    for rank in range(size):
        allocator = Allocator(64)
        broker = AgenticPWorksetLeaseBroker(4)
        broker.request(SID, 16, 20, owner="attempt1")
        broker.prepare_tp_control(1)
        broker.service(allocator)
        lease = broker.get(SID)
        assert broker.begin_io_attempt(SID, lease, "read1")
        assert ledger.claim_d2p_recovery_rank(
            SID, OWNER, tp_rank=rank, tp_size=size, claim_id="attempt1", recovery_domain=0)
        assert ledger.attach_d2p_recovery_lease_rank(
            SID, OWNER, tp_rank=rank, tp_size=size, claim_id="attempt1", lease_id=lease.lease_id)
        m, _ = manager(rank, size)
        m.owner, m.ledger, m.workset_broker = OWNER, ledger, broker
        m._h2d_poisoned, m.tp_h2d_async_prepare = False, True
        m.host_ready, m.loads, m.posts, m.cancel_receipts = {}, {}, [], []
        m._remote_host_bridge = NS(load=lambda *a, **k: None,
                                  cancel_unstarted=lambda *a, m=m, **k:m.cancel_receipts.append((a,k)))
        future = Future()
        def submit(*a, m=m, future=future, **kw):
            m.posts.append((a, kw))
            assert future.set_running_or_notify_cancel()
            return future
        m._h2d_host_copy_pool = NS(submit=submit)
        m.register_tp_host_progress(SID, "prepare1", lambda k,v,m=m:m.reports.append((k,v)))
        record = dict(network_host=True, loading=True, snapshot=NS(grant={}))
        load = dict(request_generation=RequestGeneration("request",0), workset_lease=lease,
                    recovery_claim_id="attempt1", remote_h2d_attempt="attempt1:epoch:1",
                    io_attempt="read1", io_inflight=False, io_quiesced=False, record=record,
                    device_indices=lease.parent_indices, start_allowed=False,
                    ledger_prepare_pending=False, io_complete=False)
        m.loads["child"], m.host_ready[SID] = load, record
        m._h2d_lane_reservations = {SID:0}
        m._capture_tp_host_context(load)
        items.append(NS(m=m, load=load, lease=lease, broker=broker,
                        allocator=allocator, future=future))
    return ledger, items


@pytest.mark.parametrize("size", [2,8])
def test_prepared_group_starts_on_worker_without_native_start(size):
    ledger, items = prepared_group(size)
    for rank in range(size-1):
        assert ledger.prepare_tp_host_load_rank(SID,OWNER,tp_rank=rank,tp_size=size)
    # First claim already makes H2D_LOADING; it is not proof of all preparation.
    assert ledger.get(SID)["state"] == "h2d_loading"
    for item in items:
        item.m._progress_h2d_loads()
        assert not item.m.posts and item.lease.state == "io_reserved"
    assert ledger.prepare_tp_host_load_rank(SID,OWNER,tp_rank=size-1,tp_size=size)
    for item in items:
        item.m._progress_h2d_loads()  # No gate_request / START / scheduler visit.
        item.m._progress_h2d_loads()
        assert len(item.m.posts)==1 and item.lease.state=="io_inflight"
        assert item.load["start_allowed"]
        assert item.allocator.available_size()==44
        assert not item.m.releases
        assert all(status not in {2,3,4} for _,status in item.m.reports)


@pytest.mark.parametrize("size", [2,8])
@pytest.mark.parametrize("reason", ["pending_prepare", "cancelled", "old_context", "wrong_epoch", "wrong_lease", "missing_peer_claim"])
def test_prepared_worker_rejects_uncommitted_or_stale_authorization(size, reason):
    ledger, items = prepared_group(size)
    for rank in range(size):
        assert ledger.prepare_tp_host_load_rank(SID,OWNER,tp_rank=rank,tp_size=size)
    item=items[0]
    entry=ledger.get(SID)
    if reason=="pending_prepare":
        item.load["ledger_prepare_pending"]=True
    elif reason=="cancelled":
        item.m._cancel_tp_host_handoff(SID)
    elif reason=="old_context":
        replacement=dict(item.load)
        item.m._capture_tp_host_context(replacement)
    elif reason=="wrong_epoch":
        entry["remote_read_epoch"]+=1
    elif reason=="wrong_lease":
        entry["recovery_claims"]["0"]["lease_id"]+=1
    else:
        entry["recovery_claims"].pop(str(size-1))
    assert not item.m._authorize_prepared_tp_host_load(item.load,entry)
    assert not item.load["start_allowed"] and not item.m.posts
    assert item.lease.state=="io_reserved" and item.allocator.available_size()==44


@pytest.mark.parametrize("size", [2,8])
@pytest.mark.parametrize("posted", [False,True])
def test_worker_start_peer_failure_retains_pages_until_physical_fence_and_native_retire(size, posted):
    ledger, items = prepared_group(size)
    for rank in range(size if posted else size-1):
        assert ledger.prepare_tp_host_load_rank(SID,OWNER,tp_rank=rank,tp_size=size)
    if posted:
        items[0].m._progress_h2d_loads()
        assert items[0].m.posts
    assert ledger.request_d2p_retry(SID,OWNER,reason="last_rank_prepare_or_read_failure")
    for item in items:
        assert not item.m._authorize_prepared_tp_host_load(item.load,ledger.get(SID))
        item.load["io_error"]=RuntimeError("peer failed")
        assert not item.m._discard_failed_h2d_load("child",item.load)
        assert item.allocator.available_size()==44
        assert not item.m.releases
    if posted:
        first=items[0]
        assert first.lease.state=="io_inflight" and not first.broker.tp_retire_ready(SID)
        first.future.set_exception(RuntimeError("transport returned physical cancellation fence"))
        assert not first.m._discard_failed_h2d_load("child",first.load)
    for item in items:
        assert item.broker.tp_retire_ready(SID)
        assert item.broker.commit_tp_retire(SID)
        item.broker.service(item.allocator)
        assert item.allocator.available_size()==64
        assert item.m._discard_failed_h2d_load("child",item.load)
        assert not item.m.loads and not item.m._h2d_lane_reservations
        assert not item.m.releases  # Original durable Host survives group retry.
    assert ledger.get(SID)["state"]=="host_ready"
    assert not ledger.get(SID)["h2d_prepared_ranks"]
    # Delayed old prepare ACK may set a bit, but cannot supply the old claim/epoch.
    ledger.prepare_tp_host_load_rank(SID,OWNER,tp_rank=size-1,tp_size=size)
    assert not items[0].m._authorize_prepared_tp_host_load(items[0].load,ledger.get(SID))


@pytest.mark.parametrize("size", [2,8])
def test_control_disconnect_never_starts_prepared_workset(size):
    ledger, items=prepared_group(size)
    def disconnected(sid):
        raise ConnectionError("pushed mirror is no longer authoritative")
    items[0].m.ledger=NS(is_event_control=True,get=disconnected)
    with pytest.raises(ConnectionError):
        items[0].m._progress_h2d_loads()
    assert not items[0].m.posts
    assert items[0].lease.state=="io_reserved"
    assert items[0].allocator.available_size()==44


@pytest.mark.parametrize("size", [2,8])
def test_prepared_worker_waits_lagged_mirror_and_accepts_faster_peer(size):
    ledger, items=prepared_group(size)
    for rank in range(size-1):
        assert ledger.prepare_tp_host_load_rank(SID,OWNER,tp_rank=rank,tp_size=size)
    lagged=ledger.get(SID)
    assert ledger.prepare_tp_host_load_rank(SID,OWNER,tp_rank=size-1,tp_size=size)
    items[0].m._progress_h2d_loads()
    assert items[0].lease.state=="io_inflight"
    # A different worker can see a later peer phase, but must not invent the
    # missing final prepare receipt from an earlier pushed mirror revision.
    last=items[-1]
    assert not last.m._authorize_prepared_tp_host_load(last.load,lagged)
    assert last.m._authorize_prepared_tp_host_load(last.load,ledger.get(SID))
    last.m._progress_h2d_loads()
    assert len(last.m.posts)==1


@pytest.mark.parametrize("size", [2,8])
def test_cancel_between_authorization_and_physical_post_uses_existing_broker_fence(size):
    ledger, items=prepared_group(size)
    for rank in range(size):
        assert ledger.prepare_tp_host_load_rank(SID,OWNER,tp_rank=rank,tp_size=size)
    item=items[0]
    assert item.m._authorize_prepared_tp_host_load(item.load,ledger.get(SID))
    item.broker.request_release(SID,item.lease,io_attempt=item.load["io_attempt"])
    with pytest.raises(RuntimeError,match="retiring workset"):
        item.m._start_h2d_chunk(item.load)
    assert not item.m.posts and item.allocator.available_size()==44
    assert ledger.get(SID)["recovery_claims"]["0"]["phase"]=="leased"


@pytest.mark.parametrize("mode", ["tp1", "legacy", "async_disabled"])
def test_non_endpoint_async_tp_keeps_existing_start_gate(mode):
    ledger, items=prepared_group(2)
    item=items[0]
    if mode=="tp1":
        item.m.tp_size=1
    elif mode=="legacy":
        ledger.is_event_control=False
    else:
        item.m.tp_h2d_async_prepare=False
    def forbidden(*a):
        raise AssertionError("new authorization changed a legacy/TP1 path")
    item.m._authorize_prepared_tp_host_load=forbidden
    item.m._progress_h2d_loads()
    assert not item.m.posts and not item.load["start_allowed"]


def group(size, *, checkpoint=False):
    ledger = InMemoryHostStagingLedger()
    ready(ledger, size)
    parent = RequestGeneration("request", 0)
    items, pending = [], {}
    for rank in range(size):
        allocator, state_pool = Allocator(64), Allocator(32)
        allocator.device = torch.device("cpu")
        broker = AgenticPWorksetLeaseBroker(4)
        broker.request(SID, 16, 20, owner="attempt1")
        broker.prepare_tp_control(1)
        broker.service(allocator)
        lease = broker.get(SID)
        assert broker.begin_io_attempt(SID, lease, "physical-copy")
        broker.mark_io_inflight(SID, lease, "physical-copy")
        broker.mark_io_quiesced(SID, lease, "physical-copy")
        assert broker.begin_bind(SID, lease)
        lease.parent_bound = True
        # Real Radix parent: another request keeps four shared prefix pages.
        tree = RadixCache.create_simulated(mock_allocator=allocator, page_size=4)
        tree.insert(InsertParams(key=RadixKey(list(range(16)), "conversation"), value=lease.parent_indices))
        other = tree.match_prefix(MatchPrefixParams(key=RadixKey(list(range(4)), "conversation")))
        tree.inc_lock_ref(other.last_device_node)
        restored = tree.match_prefix(MatchPrefixParams(key=RadixKey(list(range(16)), "conversation")))
        tree.inc_lock_ref(restored.last_device_node)
        req = NS(rid="child", origin_input_ids=list(range(20)), output_ids=[], extra_key="conversation",
                 req_pool_idx=None, mamba_pool_idx=None, _agentic_kv_host_pin_node=restored.last_device_node)
        # Use real broker runtime donation and native no-reqslot Mamba cleanup.
        broker._reserve_mamba_checkpoint = checkpoint
        lease.runtime_state_device_indices = (state_pool.alloc(4 if checkpoint else 3),)
        broker._state_allocators = (state_pool,)
        broker._mamba_req_to_token_pool = NS(enable_mamba_extra_buffer=True,
                                            mamba_ping_pong_track_buffer_size=2)
        broker.attach_runtime_state_for_bind(SID, req, lease)
        tree.supports_mamba = lambda: True
        tree.req_to_token_pool = NS(mamba_pool=state_pool)
        identity = dict(tp_rank=rank, tp_size=size, claim_id="attempt1", lease_id=lease.lease_id,
                        remote_read_epoch=1)
        assert ledger.claim_d2p_recovery_rank(
            SID, OWNER, tp_rank=rank, tp_size=size, claim_id="attempt1", recovery_domain=0)
        assert ledger.attach_d2p_recovery_lease_rank(
            SID, OWNER, **{k:v for k,v in identity.items() if k != "remote_read_epoch"})
        assert ledger.mark_d2p_recovery_phase_rank(
            SID, OWNER, phase="io_inflight",
            **{k:v for k,v in identity.items() if k != "remote_read_epoch"})
        assert ledger.apply_recovery_event(SID, OWNER, event="loaded", **identity)
        m, _ = manager(rank, size)
        m.owner, m.workset_broker, m.tree_cache = OWNER, broker, tree
        m.host_ready, m.loads = {}, {}
        m.tp_host_commit_snapshots, m.tp_host_admit_snapshots = [], []
        m._h2d_lane_reservations = {SID:0}
        m.register_tp_host_progress(SID, "native-bind", lambda k,v,m=m:m.reports.append((k,v)))
        def submit(method, *args, rank=rank, **kw):
            key = (rank, method)
            assert key not in pending, "duplicate receipt RPC"
            future = Future()
            pending[key] = (future, kw)
            return future
        m.ledger = NS(is_event_control=True, get=ledger.get, submit=submit)
        record = {"offer":{"token_count":16}, "network_host":True, "loading":True}
        m.host_ready[SID] = record
        load = dict(request_generation=parent, workset_lease=lease, recovery_claim_id="attempt1",
                    remote_h2d_attempt="attempt1:epoch:1", record=record, device_indices=lease.parent_indices,
                    radix_bound=True, io_complete=True)
        m.loads[req.rid] = load
        m._capture_tp_host_context(load)
        m._report_tp_host_completion(SID, load["tp_progress_context"], 2)
        items.append(NS(m=m, req=req, lease=lease, broker=broker, allocator=allocator,
                        state_pool=state_pool, tree=tree, other=other, load=load, identity=identity))
    def ack(rank, method):
        future, identity = pending[(rank,method)]
        result = ledger.apply_recovery_event(
            SID, OWNER, event="bound" if method=="complete_host_bind_rank" else "handed",
            **{k:v for k,v in identity.items() if k != "phase"})
        future.set_result(result)
        return result
    return parent, ledger, items, pending, ack


@pytest.mark.parametrize("size", [2,8])
def test_real_group_fused_bind_never_needs_commit_but_requires_all_handed_admit(size, monkeypatch):
    parent, ledger, items, pending, ack = group(size)
    manifest = Future(); manifest.set_result(True)
    mock_manifest_future(monkeypatch, manifest)
    for item in items:
        assert item.m.gate_request(item.req, parent, allow_bind=True)
        assert item.lease.state == "handed"
        assert not getattr(item.req,"_agentic_kv_gate_complete",False)
        item.m._progress_tp_host_handoffs()
    for rank in range(size-1):
        assert ack(rank,"complete_host_bind_rank")
    for item in items:
        item.m._progress_tp_host_handoffs()
        assert not item.m.releases
        assert item.m.publish_tp_host_status(SID,"native-bind",3)==2
    assert ledger.get(SID)["state"] == "hbm_ready"
    assert ack(size-1,"complete_host_bind_rank")
    for item in items:
        item.m._progress_tp_host_handoffs()
        assert len(item.m.releases)==1
    for rank in range(size-1):
        assert ack(rank,"mark_d2p_recovery_phase_rank")
    for item in items:
        item.m._progress_tp_host_handoffs()
        assert item.m.gate_request(item.req,parent)
    # One delayed exact handed ACK prevents the logical group's ADMIT.
    assert items[-1].m.reports[-1][1]==2
    assert ack(size-1,"mark_d2p_recovery_phase_rank")
    for rank in range(size):
        assert ledger.complete_source_host_release_rank(SID,OWNER,tp_rank=rank,tp_size=size)
    age(ledger)
    ledger.prune(0,0)
    assert ledger.get(SID) is None
    for item in items:
        item.m._progress_tp_host_handoffs()
        assert item.m.reports[-1][1]==4
        assert all(status!=3 for _,status in item.m.reports)
        assert Scheduler._agentic_tp_host_next_action(4,handoff_barrier=True)=="admit"
    for item in items:
        item.m.tp_host_commit_snapshots=item.m.tp_host_admit_snapshots=[SID]
        assert item.m.gate_request(item.req,parent) is False
        assert item.req._agentic_kv_gate_complete
        assert len(item.m.releases)==1


@pytest.mark.parametrize("size", [2,8])
@pytest.mark.parametrize("failure_after_handoff", [False, True])
def test_partial_handoff_failure_keeps_host_and_retires_parent_suffix_runtime_once(
    size, failure_after_handoff, monkeypatch,
):
    parent, ledger, items, pending, ack = group(size)
    last = items[-1]
    original_handoff = last.broker.handoff_to_req
    def fail(*a):
        if failure_after_handoff:
            original_handoff(*a)
        raise RuntimeError("injected last-rank handoff failure")
    monkeypatch.setattr(last.broker,"handoff_to_req",fail)
    for item in items:
        assert item.m.gate_request(item.req,parent,allow_bind=True)
        item.m._progress_tp_host_handoffs()
    assert (size-1,"complete_host_bind_rank") not in pending
    for rank in range(size-1):
        assert ack(rank,"complete_host_bind_rank")
    assert ledger.get(SID)["state"]=="hbm_ready"
    assert ledger.request_d2p_retry(SID,OWNER,reason="peer_handoff_failed")
    for item in items:
        # Original scheduler rollback, selecting binding vs handed ownership.
        item.m.rollback_bound_parent(item.req,parent)
        item.m.rollback_bound_parent(item.req,parent)  # Idempotent repeat.
        item.m._progress_tp_host_handoffs()
        assert not item.m.releases  # Complete Host source is retained for retry.
        assert not getattr(item.req,"_agentic_kv_gate_complete",False)
        assert not hasattr(item.req,"_agentic_p_workset_lease")
        assert item.req.mamba_pool_idx is None
        assert item.state_pool.available_size()==(29 if item is last and not failure_after_handoff else 32)
        assert item.broker.tp_retire_ready(SID)
        assert item.allocator.available_size()==56  # Shared prefix4 + suffix4.
    for item in items:
        assert item.broker.commit_tp_retire(SID)
        item.broker.service(item.allocator)
        assert item.allocator.available_size()==60
        assert len(item.allocator.free_indices.unique())==60
        assert len(item.state_pool.free_indices.unique())==32
        prefix=item.tree.match_prefix(MatchPrefixParams(key=RadixKey(list(range(4)), "conversation")))
        assert len(prefix.device_indices)==4 and prefix.last_device_node.lock_ref==1
    for rank in range(size):
        assert ledger.complete_d2p_retry_rank(SID,OWNER,tp_rank=rank,tp_size=size,remote_read_epoch=1)
    assert ledger.get(SID)["state"]=="host_ready"
    # A delayed old bound receipt cannot consume the recovered Host generation.
    assert not ledger.apply_recovery_event(SID,OWNER,event="bound",**items[0].identity)


@pytest.mark.parametrize("size", [2,8])
def test_cancel_before_delayed_bound_reply_never_closes_or_admits(size):
    parent, ledger, items, pending, ack = group(size)
    for item in items:
        assert item.m.gate_request(item.req,parent,allow_bind=True)
        item.m._progress_tp_host_handoffs()
    assert ledger.request_d2p_retry(SID,OWNER,reason="cancel_during_receipt")
    for item in items:
        item.m.rollback_bound_parent(item.req,parent)
        item.m._progress_tp_host_handoffs()
        assert SID in item.m._tp_host_handoff_jobs  # Pending reply still fenced.
    for rank,item in enumerate(items):
        assert not ack(rank,"complete_host_bind_rank")
        item.m._progress_tp_host_handoffs()
        assert not item.m._tp_host_handoff_jobs
        assert not item.m.releases
        assert not getattr(item.req,"_agentic_kv_gate_complete",False)
        assert all(status not in {3,4} for _,status in item.m.reports)
        assert item.broker.commit_tp_retire(SID)
        item.broker.service(item.allocator)
        assert item.allocator.available_size()==60  # Shared prefix survives.
        assert item.state_pool.available_size()==32


@pytest.mark.parametrize("size", [2,8])
def test_handed_runtime_checkpoint_cleanup_preserves_other_radix_owned_slot(size):
    parent, ledger, items, pending, ack = group(size,checkpoint=True)
    for item in items:
        # This separate checkpoint is not donated to this Req. The fixture
        # uses real attention Radix + CPU state allocator, not MambaRadix DMA.
        other_radix_checkpoint = item.state_pool.alloc(1)
        assert item.m.gate_request(item.req,parent,allow_bind=True)
        assert item.req._agentic_mamba_prefill_checkpoint.numel()==1
        assert item.state_pool.available_size()==27
        item.m.rollback_bound_parent(item.req,parent)
        item.m.rollback_bound_parent(item.req,parent)
        assert item.req._agentic_mamba_prefill_checkpoint is None
        assert item.req.mamba_pool_idx is None
        assert item.req.mamba_ping_pong_track_buffer is None
        assert not item.req._agentic_mamba_runtime_reserved
        assert item.state_pool.available_size()==31
        assert other_radix_checkpoint.item() not in item.state_pool.free_indices.tolist()
        assert len(item.state_pool.free_indices.unique())==31
        assert item.broker.commit_tp_retire(SID)
        item.broker.service(item.allocator)
        assert item.state_pool.available_size()==31  # Broker cannot free Req slots again.
        assert item.allocator.available_size()==60  # Other request's shared prefix4 remains.


@pytest.mark.parametrize("size", [2,8])
def test_cancelled_context_at_bound_queue_entry_rolls_back_real_gate(size):
    parent, ledger, items, pending, ack = group(size,checkpoint=True)
    for item in items:
        # Inject precisely after real Radix insert/pin, before notification
        # enqueue. No pre-existing Req rollback metadata may be assumed.
        assert item.lease.state=="binding"
        assert not hasattr(item.req,"_agentic_host_workset_lease")
        item.m._cancel_tp_host_handoff(SID)
        assert item.m.gate_request(item.req,parent,allow_bind=True)
        assert item.req._agentic_host_retry_reason=="slow_local_handoff_failed"
        assert not item.m.loads
        assert not hasattr(item.req,"_agentic_kv_host_pin_node")
        assert not hasattr(item.req,"_agentic_host_workset_lease")
        assert item.broker.tp_retire_ready(SID)
        assert item.broker.commit_tp_retire(SID)
        item.broker.service(item.allocator)
        assert item.allocator.available_size()==60
        assert item.state_pool.available_size()==32
    assert not pending  # No rejected/cancelled context published bound ACK.
    assert ledger.get(SID)["state"]=="hbm_ready"
