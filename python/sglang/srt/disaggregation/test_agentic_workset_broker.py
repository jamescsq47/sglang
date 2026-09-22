"""Real facade/authority/TP executor, with CPU descriptor and producer events."""
from concurrent.futures import Future
from types import SimpleNamespace
import threading

import pytest
import torch

from sglang.srt.disaggregation.agentic_tp_events import TPEventClient, TPEventServer
from sglang.srt.disaggregation.agentic_workset_broker import ControllerWorksetBroker
from sglang.srt.disaggregation.agentic_workset_controller import WorksetController
from sglang.srt.disaggregation.agentic_workset_ledger import WorksetLedger
from sglang.srt.disaggregation.agentic_workset_native import NativeLastRefFreeAdapter
from sglang.srt.disaggregation.agentic_workset_runtime import PWorksetRuntime
from sglang.srt.disaggregation.test_agentic_workset_runtime import eventually


@pytest.fixture
def cluster():
    created = []

    def make(size=2, pages=16, hybrid=False, defer_last_bridge=False):
        server = TPEventServer("facade-test", "secret")
        clients = [TPEventClient(server.address, run_id="facade-test", token="secret",
                    group="P", rank=i, size=size) for i in range(size)]
        for client in clients:
            client.wait_ready()
        ledger = WorksetLedger(incarnation="run", page_count=pages, page_size=4, tp_size=size,
                              mamba_slots=8 if hybrid else 0)
        controller = WorksetController(ledger)
        brokers, runtimes, bridges = [], [], []
        for rank, client in enumerate(clients):
            from sglang.srt.disaggregation.test_agentic_workset_device import pool
            mamba = pool() if hybrid else None
            req_pool = SimpleNamespace(enable_mamba_extra_buffer=True,
                mamba_ping_pong_track_buffer_size=2) if hybrid else None
            broker = ControllerWorksetBroker(4, rank=rank,
                state_allocators=(mamba,) if hybrid else (),
                mamba_req_to_token_pool=req_pool, reserve_mamba_checkpoint=hybrid)
            runtime = PWorksetRuntime(client, broker, controller=controller if rank == 0 else None,
                incarnation="run", page_size=4, page_capacity=pages,
                reference_fence=broker.reference_fence, dedicated_client=True,
                mamba_slots=8 if hybrid else 0, mamba_pool=mamba)
            bridge = NativeLastRefFreeAdapter(incarnation="run", rank=rank, page_size=4,
                counts=runtime.counts, on_ready=runtime.native_free)
            if not (defer_last_bridge and rank == size - 1):
                broker.attach_runtime(runtime, native_bridge=bridge)
            brokers.append(broker)
            runtimes.append(runtime)
            bridges.append(bridge)
        created.append((server, clients, runtimes, bridges))
        return ledger, brokers, runtimes, bridges

    yield make
    for server, clients, runtimes, bridges in created:
        for bridge in bridges:
            bridge.shutdown()
        for runtime in runtimes:
            runtime.shutdown()
        for client in clients:
            client.close()
        server.close()


def admit(brokers, runtimes, sid="r:0", parent=4, prompt=12, owner="direct:r:0"):
    for broker in brokers:
        assert broker.request(sid, parent, prompt, owner=owner)
    plan = brokers[0]._requests[(sid, owner)]["future"].result(5)
    return [runtime.wait_ready(plan.key, 5) for runtime in runtimes]


@pytest.mark.parametrize("size", [1, 2, 8])
def test_private_cancel_frees_only_after_every_real_transport_fence(cluster, size):
    ledger, brokers, runtimes, _ = cluster(size)
    leases = admit(brokers, runtimes)
    for broker, lease in zip(brokers, leases):
        assert broker.begin_io_attempt("r:0", lease, "dma")
        broker.mark_io_inflight("r:0", lease, "dma")
        assert not broker.request_release("r:0", lease, io_attempt="dma")
    assert ledger.counts.free_pages == 13
    for broker, lease in zip(brokers[:-1], leases[:-1]):
        assert broker.mark_io_quiesced("r:0", lease, "dma")
    assert ledger.counts.free_pages == 13
    assert brokers[-1].mark_io_quiesced("r:0", leases[-1], "dma")
    eventually(lambda: all(runtime.counts().live_leases == 0 for runtime in runtimes))
    assert ledger.counts.free_pages == 16
    assert all(not broker._leases for broker in brokers)
    admit(brokers, runtimes, owner="slow:r:0:req2")


@pytest.mark.parametrize("size", [2, 8])
def test_reserved_claim_must_settle_before_retirement(cluster, size):
    ledger, brokers, runtimes, _ = cluster(size)
    leases = admit(brokers, runtimes)
    for broker, lease in zip(brokers, leases):
        assert broker.begin_io_attempt("r:0", lease, "claim")
        assert not broker.prepare_tp_retire("r:0")
        assert lease.state == "io_reserved"
        with pytest.raises(RuntimeError, match="retiring"):
            broker.mark_io_inflight("r:0", lease, "claim")
    assert ledger.counts.free_pages == 13
    for broker, lease in zip(brokers, leases):
        assert broker.cancel_io_attempt("r:0", lease, "claim")
    eventually(lambda: ledger.counts.free_pages == 16)


def test_pending_capacity_cancel_has_no_late_grant(cluster):
    ledger, brokers, runtimes, _ = cluster(2, pages=3)
    leases = admit(brokers, runtimes)
    for broker in brokers:
        assert broker.request("waiting", 0, 12, owner="fresh:waiting")
    pending = brokers[0]._requests[("waiting", "fresh:waiting")]["future"]
    assert not pending.done()
    for broker in brokers:
        assert broker.cancel_unstarted("waiting", owner="fresh:waiting")
    eventually(pending.done)
    for broker in brokers:
        assert broker.cancel_unstarted("r:0", owner="direct:r:0")
    eventually(lambda: ledger.counts.free_pages == 3)
    assert all(broker.get("waiting") is None for broker in brokers)
    admit(brokers, runtimes, sid="waiting", parent=0, owner="fresh:new")


@pytest.mark.parametrize("size", [2, 8])
def test_handed_private_suffix_and_native_lastref_are_distinct(cluster, size):
    ledger, brokers, runtimes, bridges = cluster(size)
    leases = admit(brokers, runtimes, parent=0, owner="fresh:r:0")
    reqs = []
    for broker, lease in zip(brokers, leases):
        req = SimpleNamespace(origin_input_ids=list(range(12)), req_pool_idx=None,
            prefix_indices=(), mamba_pool_idx=None, mamba_ping_pong_track_buffer=None)
        broker.handoff_fresh_to_req("r:0", req, lease)
        used = broker.consume_suffix(lease, 4, final_prompt_chunk=False)
        assert used.tolist() == [4, 5, 6, 7]
        assert broker.release_handed("r:0", lease, req=req)
        assert not broker.release_handed("r:0", lease, req=req)
        reqs.append(req)
    eventually(lambda: ledger.counts.free_pages == 15)
    # Used Req pages stay owned despite all private suffix returns.
    assert ledger.counts.live_leases == 1
    for bridge in bridges:
            bridge.free(torch.tensor([4, 5, 6, 7], dtype=torch.int64), resource="attention")
    eventually(lambda: all(runtime.counts().live_leases == 0 for runtime in runtimes))
    assert ledger.counts.free_pages == 16


def test_producer_event_blocks_private_reuse_and_service_never_allocates(cluster):
    ledger, brokers, runtimes, _ = cluster(2)
    leases = admit(brokers, runtimes)
    completed, resume = threading.Event(), threading.Event()
    class Event:
        def synchronize(self):
            completed.set()
            assert resume.wait(5)
    try:
        for broker, lease in zip(brokers, leases):
            broker.service(object())
            assert broker.begin_bind("r:0", lease)
            # Fake only the CUDA event, not the physical ownership protocol.
            original = broker._producer_done
            broker._producer_done = lambda event=None, original=original: original(Event())
            assert broker.abort_bind("r:0", lease, parent_bound=False)
        assert completed.wait(5)
        assert ledger.counts.free_pages == 13
    finally:
        resume.set()
    eventually(lambda: ledger.counts.free_pages == 16)


def test_rank0_get_waits_for_follower_descriptor_install(cluster):
    _, brokers, runtimes, _ = cluster(2)
    entered, resume = threading.Event(), threading.Event()
    original = runtimes[1]._prepare
    def delayed(plan):
        entered.set()
        assert resume.wait(5)
        return original(plan)
    runtimes[1]._prepare = delayed
    try:
        assert brokers[0].request("r:0", 4, 12, owner="direct:r:0")
        assert entered.wait(5)
        eventually(lambda: "r:0" in brokers[0]._leases)
        assert brokers[0].get("r:0") is None
    finally:
        resume.set()
    eventually(lambda: brokers[0].get("r:0") is not None)


@pytest.mark.parametrize("parent", [0, 4])
def test_hybrid_every_source_has_two_exact_output_checkpoints(cluster, parent):
    ledger, brokers, runtimes, _ = cluster(2, hybrid=True)
    leases = admit(brokers, runtimes, parent=parent)
    for broker, lease in zip(brokers, leases):
        plan = lease.controller_plan
        assert sum(run.count for run in plan.runtime_slots) == 5
        assert sum(run.count for run in plan.checkpoint_slots) == bool(parent)
        req = SimpleNamespace(origin_input_ids=list(range(12)), req_pool_idx=None,
            prefix_indices=(), mamba_pool_idx=None, mamba_ping_pong_track_buffer=None)
        if parent:
            assert broker.begin_bind("r:0", lease)
            broker.commit_parent_bound("r:0", lease, state_donated_to_radix=True)
            broker.attach_runtime_state_for_bind("r:0", req, lease)
            broker.handoff_to_req("r:0", req, lease)
        else:
            broker.handoff_fresh_to_req("r:0", req, lease)
        checkpoint = req._agentic_mamba_prefill_checkpoint
        assert checkpoint.numel() == 2
        broker.prepare_req_checkpoints(req, lease)
        rotation = req._agentic_checkpoint_rotation
        assert rotation.key == plan.key and rotation.can_take()
        broker.prepare_req_checkpoints(req, lease)
        assert req._agentic_checkpoint_rotation is rotation
    assert ledger.counts.free_mamba_slots == (2 if parent else 3)


def test_native_prefix_adoption_returns_only_redundant_private_prefix(cluster):
    ledger, brokers, runtimes, bridges = cluster(2)
    leases = admit(brokers, runtimes, parent=0, owner="fresh:r:0")
    for broker, lease in zip(brokers, leases):
        req = SimpleNamespace(origin_input_ids=list(range(12)), req_pool_idx=None,
            prefix_indices=(), mamba_pool_idx=None, mamba_ping_pong_track_buffer=None)
        broker.handoff_fresh_to_req("r:0", req, lease)
        req.prefix_indices = torch.tensor([80, 81, 82, 83])  # Other pinned Radix owner.
        broker.adopt_fresh_prefix(req, lease, prefix_tokens=4, pinned=True,
                                 return_prefix=broker.return_fresh_prefix)
        assert lease.suffix_cursor == 4
        assert req.prefix_indices.tolist() == [80, 81, 82, 83]
        assert broker.consume_suffix(lease, 8, final_prompt_chunk=True).tolist() == list(range(8, 16))
    eventually(lambda: ledger.counts.free_pages == 14)
    for bridge in bridges:
        bridge.free(torch.arange(8, 16), resource="attention")
    eventually(lambda: all(runtime.counts().live_leases == 0 for runtime in runtimes))
    assert ledger.counts.free_pages == 16


def test_cancel_before_prepared_descriptor_does_not_revive_start(cluster):
    ledger, brokers, runtimes, _ = cluster(2)
    entered, resume = threading.Event(), threading.Event()
    original = runtimes[1]._prepare
    def delayed(plan):
        entered.set()
        assert resume.wait(5)
        return original(plan)
    runtimes[1]._prepare = delayed
    try:
        for broker in brokers:
            assert broker.request("r:0", 4, 12, owner="direct:r:0")
        assert entered.wait(5)
        assert brokers[1].supersede_unstarted("r:0", owner="direct:r:0")
        assert brokers[0].cancel_unstarted("r:0", owner="direct:r:0")
        assert ledger.counts.free_pages == 13
    finally:
        resume.set()
    eventually(lambda: ledger.counts.free_pages == 16)
    eventually(lambda: all(runtime.counts().live_leases == 0 for runtime in runtimes))
    assert ledger.counts.free_pages == 16


def test_wait_ready_does_not_hold_runtime_lock_while_waiting_for_broker(cluster):
    _, brokers, runtimes, _ = cluster(1)
    lease = admit(brokers, runtimes)[0]
    started, done = threading.Event(), threading.Event()
    def waiter():
        started.set()
        assert runtimes[0].wait_ready(lease.controller_plan.key, 5) is lease
        done.set()
    with brokers[0]._lock:
        thread = threading.Thread(target=waiter)
        thread.start()
        assert started.wait(5)
        # Same lock order as native adopt_fresh_prefix -> begin_return.
        assert runtimes[0]._lock.acquire(timeout=1)
        runtimes[0]._lock.release()
    thread.join(5)
    assert done.is_set()


def test_conservation_snapshot_has_one_atomic_address_cut(cluster):
    _, brokers, runtimes, _ = cluster(2)
    admit(brokers, runtimes)
    for runtime in runtimes:
        counts, views = runtime.conservation_snapshot()
        assert counts.free_pages + sum(run.count for view in views
            for run in view.remaining_pages) == 16
        assert counts.live_leases == len(views) == 1


def test_bound_abort_returns_private_ranges_not_donated_radix_parent(cluster):
    ledger, brokers, runtimes, bridges = cluster(2, hybrid=True)
    leases = admit(brokers, runtimes)
    for broker, lease in zip(brokers, leases):
        req = SimpleNamespace(mamba_pool_idx=None, mamba_ping_pong_track_buffer=None)
        assert broker.begin_bind("r:0", lease)
        broker.commit_parent_bound("r:0", lease, state_donated_to_radix=True)
        broker.attach_runtime_state_for_bind("r:0", req, lease)
        assert broker.abort_bind("r:0", lease, parent_bound=True)
        assert req.mamba_pool_idx is None
        assert broker.prepare_tp_retire("r:0")
        assert broker.commit_tp_retire("r:0")
    eventually(lambda: ledger.counts.free_pages == 15 and ledger.counts.free_mamba_slots == 7)
    assert ledger.counts.live_leases == 1
    # The actual native last-reference path owns both donated resources.
    for bridge in bridges:
        bridge.free(torch.arange(4, 8), resource="attention")
        bridge.free(torch.tensor([1]), resource="mamba")
    eventually(lambda: ledger.counts.free_pages == 16 and ledger.counts.free_mamba_slots == 8)
    eventually(lambda: all(runtime.counts().live_leases == 0 for runtime in runtimes))


def test_repeated_old_native_retirement_cannot_poison_new_owner(cluster):
    ledger, brokers, runtimes, _ = cluster(2)
    admit(brokers, runtimes)
    for broker in brokers:
        assert broker.cancel_unstarted("r:0", owner="direct:r:0")
    eventually(lambda: ledger.counts.free_pages == 16)
    eventually(lambda: all(runtime.counts().live_leases == 0 for runtime in runtimes))
    for broker in brokers:
        assert broker.prepare_tp_retire("r:0")
        assert broker.commit_tp_retire("r:0")
        assert "r:0" not in broker._tp_retire_requested
    admit(brokers, runtimes, owner="slow:r:0:next")


def test_checkpoint_return_cannot_reclaim_active_or_tracking_slots(cluster):
    ledger, brokers, runtimes, _ = cluster(2, hybrid=True)
    leases = admit(brokers, runtimes)
    before = ledger.counts
    for broker, lease in zip(brokers, leases):
        plan = lease.controller_plan
        with pytest.raises(ValueError, match="output slots"):
            broker.return_state_slots(plan, (plan.runtime_slots[0].start,))
        with pytest.raises(ValueError, match="fresh"):
            broker.return_fresh_prefix(plan, 4)
        assert not broker._permits
    assert ledger.counts == before


@pytest.mark.parametrize("size", [2, 8])
def test_ungranted_slow_retry_requires_strictly_new_authoritative_epoch(cluster, size):
    ledger, brokers, runtimes, _ = cluster(size, pages=3)
    admit(brokers, runtimes)
    owner = "slow:waiting:rid"
    for broker in brokers:
        assert broker.bind_restore_epoch("waiting", owner, 1)
        assert broker.request("waiting", 4, 12, owner=owner)
    pending = brokers[0]._requests[("waiting", owner)]["future"]
    for broker in brokers:
        assert broker.cancel_unstarted("waiting", owner=owner)
    eventually(pending.done)
    # The existing Host all-rank retry barrier is the caller's authority for 2.
    assert not brokers[-1].bind_restore_epoch("waiting", owner, 1)
    assert not brokers[-1].request("waiting", 4, 12, owner=owner)
    for broker in brokers:
        assert broker.bind_restore_epoch("waiting", owner, 2)
        assert not broker.bind_restore_epoch("waiting", owner, 1)
        assert broker.request("waiting", 4, 12, owner=owner)
    for broker in brokers:
        assert broker.cancel_unstarted("r:0", owner="direct:r:0")
    plan = brokers[0]._requests[("waiting", owner)]["future"].result(5)
    for runtime in runtimes:
        runtime.wait_ready(plan.key, 5)
    assert ledger.counts.free_pages == 0
    for broker in brokers:
        assert not broker.bind_restore_epoch("waiting", owner, 3)


def test_peer_grant_and_cancel_before_local_bridge_attachment_keep_observer_bound(cluster):
    ledger, brokers, runtimes, bridges = cluster(2, defer_last_bridge=True)
    assert brokers[1].runtime is runtimes[1]
    with pytest.raises(RuntimeError, match="native return bridge"):
        brokers[1].request("not-started", 0, 12)
    assert brokers[0].request("r:0", 4, 12, owner="direct:r:0")
    plan = brokers[0]._requests[("r:0", "direct:r:0")]["future"].result(5)
    runtimes[0].wait_ready(plan.key, 5)
    assert brokers[0].cancel_unstarted("r:0", owner="direct:r:0")
    eventually(lambda: ledger.counts.free_pages == 16)
    eventually(lambda: all(runtime.counts().live_leases == 0 for runtime in runtimes))
    brokers[1].attach_runtime(runtimes[1], native_bridge=bridges[1])
    admit(brokers, runtimes, sid="next")
