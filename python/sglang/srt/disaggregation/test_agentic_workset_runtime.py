"""Real actor/controller/TCP/broker/device wiring; CPU physical buffers only."""
from concurrent.futures import Future
from types import SimpleNamespace
import threading
import time

import pytest
import torch

from sglang.srt.disaggregation.agentic_tp_events import TPEventClient, TPEventServer
from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_workset_controller import WorksetController, WorksetIntent
from sglang.srt.disaggregation.agentic_workset_ledger import PageRun, WorksetLedger
from sglang.srt.disaggregation.agentic_workset_native import NativeFreeReceipt
from sglang.srt.disaggregation.agentic_workset_runtime import PWorksetRuntime, WorksetRuntimeStopped
from sglang.srt.disaggregation.agentic_workset_tp import FenceProof


def eventually(predicate, timeout=5):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError("condition did not progress")
        time.sleep(.005)  # Driver only; runtime has no polling timer.


@pytest.fixture
def cluster():
    created = []
    def make(size=2, page_count=16, server=None):
        server = server or TPEventServer("runtime-test", "secret")
        clients = [TPEventClient(server.address, run_id="runtime-test", token="secret",
                    group="P", rank=i, size=size) for i in range(size)]
        for client in clients:
            client.wait_ready()
        ledger = WorksetLedger(incarnation="run", page_count=page_count, page_size=4, tp_size=size)
        controller = WorksetController(ledger)
        proofs = [{} for _ in clients]
        runtimes = []
        for rank, client in enumerate(clients):
            def fence(plan, scope, rank=rank):
                future = Future()
                proofs[rank][scope.sequence] = (plan, scope, future)
                return future
            runtime = PWorksetRuntime(client, AgenticPWorksetLeaseBroker(4),
                controller=controller if rank == 0 else None,
                incarnation="run", page_size=4, page_capacity=page_count,
                reference_fence=fence, dedicated_client=True)
            runtimes.append(runtime)
        created.append((server, clients, runtimes))
        return ledger, runtimes, proofs
    yield make
    for server, clients, runtimes in created:
        for runtime in runtimes:
            runtime.shutdown()
        for client in clients:
            client.close()
        server.close()


def request(runtime, sid="r:0", attempt="a"):
    return runtime.request(WorksetIntent(sid, attempt, "fresh", 0, 12)).result(5)


def test_unqualified_cuda_device_captures_native_owner_before_worker(monkeypatch):
    server = TPEventServer("capture-device", "secret")
    client = TPEventClient(server.address, run_id="capture-device", token="secret",
                          group="P", rank=0, size=1)
    client.wait_ready()
    owner = threading.current_thread()
    def current_device():
        assert threading.current_thread() is owner
        return 3
    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    controller = WorksetController(WorksetLedger(incarnation="run", page_count=16, page_size=4))
    runtime = None
    try:
        runtime = PWorksetRuntime(client, AgenticPWorksetLeaseBroker(4),
            controller=controller, device="cuda", incarnation="run", page_size=4,
            page_capacity=16, reference_fence=lambda *_: Future(), dedicated_client=True)
        assert runtime.device == torch.device("cuda:3")
    finally:
        if runtime is not None:
            runtime.shutdown()
        controller.shutdown()
        client.close()
        server.close()


def finish(proofs, scope, ranks=None):
    ranks = range(len(proofs)) if ranks is None else ranks
    for rank in ranks:
        eventually(lambda: scope.sequence in proofs[rank])
        _, _, future = proofs[rank][scope.sequence]
        future.set_result(FenceProof(scope.key, scope.sequence, True, True))


@pytest.mark.parametrize("size", [1, 2, 8])
def test_real_runtime_grant_prepare_cancel_fence_and_reuse(cluster, size):
    ledger, runtimes, proofs = cluster(size)
    plan = request(runtimes[0])
    leases = [runtime.wait_ready(plan.key, 5) for runtime in runtimes]
    assert all(lease.device_indices.tolist() == list(range(4, 16)) for lease in leases)
    assert runtimes[0].ready_cut().ready == (plan.key,)
    assert ledger.counts.free_pages == 13
    scope = runtimes[0].cancel(plan.key).result(5)
    finish(proofs, scope, range(size - 1))
    assert ledger.counts.free_pages == 13
    finish(proofs, scope, [size - 1])
    eventually(lambda: ledger.counts.free_pages == 16)
    eventually(lambda: all(runtime.broker.get("r:0") is None for runtime in runtimes))
    successor = request(runtimes[0], attempt="b")
    assert successor.suffix_pages == plan.suffix_pages
    for runtime in runtimes:
        runtime.wait_ready(successor.key, 5)
        assert plan.key not in runtime._descriptors


def test_tp8_c128_actual_grant_prepare_free_and_reuse_burst(cluster):
    ledger, runtimes, proofs = cluster(8, page_count=512)
    # Real controller/actor/TCP/preparation, not fabricated status reports.
    for wave in range(2):
        futures = [runtimes[0].request(WorksetIntent(f"r{i}", f"wave{wave}", "fresh", 0, 12))
                   for i in range(128)]
        plans = [f.result(15) for f in futures]
        for runtime in runtimes:
            for plan in plans:
                runtime.wait_ready(plan.key, 15)
        assert ledger.counts.free_pages == 128
        scopes = [runtimes[0].cancel(plan.key) for plan in plans]
        scopes = [f.result(15) for f in scopes]
        for scope in scopes:
            finish(proofs, scope, range(7))
        # The last shard has NOT fenced: even under burst no address is reusable.
        assert ledger.counts.free_pages == 128
        for scope in scopes:
            finish(proofs, scope, [7])
        eventually(lambda: all(r.counts().free_pages == 512 and r.counts().live_leases == 0
                               for r in runtimes), timeout=15)


@pytest.mark.parametrize("size", [1, 2, 8])
def test_runtime_reclaims_history_with_small_event_capacity(cluster, size):
    # More lifetime operations than the quota, while only one workset is live.
    # Exercise actual CPU allocation, preparation, physical proof and TCP ACKs.
    server = TPEventServer("runtime-test", "secret", max_entries=24)
    ledger, runtimes, proofs = cluster(size, server=server)
    for cycle in range(32):
        plan = request(runtimes[0], sid=f"history:{cycle}")
        for runtime in runtimes:
            runtime.wait_ready(plan.key, 5)
        scope = runtimes[0].cancel(plan.key).result(5)
        if size > 1:
            finish(proofs, scope, range(size - 1))
            assert ledger.counts.free_pages == 13
        finish(proofs, scope, [size - 1])
        eventually(lambda: all(r.counts().free_pages == 16 for r in runtimes))
        eventually(lambda: server._entry_count == 0)
        assert not server.errors
        for runtime in runtimes:
            runtime.check_health()


@pytest.mark.parametrize("size", [2, 8])
def test_partial_return_preserves_rest_and_allrank_lastref(cluster, size):
    ledger, runtimes, proofs = cluster(size)
    plan = request(runtimes[0])
    for runtime in runtimes:
        runtime.wait_ready(plan.key, 5)
    scope = runtimes[0].begin_return(plan.key, "private-prefix", pages=(PageRun(1, 1),)).result(5)
    finish(proofs, scope, range(size - 1))
    assert ledger.counts.free_pages == 13
    finish(proofs, scope, [size - 1])
    eventually(lambda: ledger.counts.free_pages == 14)
    assert ledger.view(plan.key).remaining_pages == (PageRun(2, 2),)
    assert runtimes[0].broker.get("r:0") is not None
    whole = runtimes[0].cancel(plan.key).result(5)
    finish(proofs, whole)
    eventually(lambda: ledger.counts.free_pages == 16)


@pytest.mark.parametrize("size", [2, 8])
def test_native_free_pairs_exact_addresses_not_rank_ticket_order(cluster, size):
    ledger, runtimes, proofs = cluster(size)
    plan = request(runtimes[0])
    for runtime in runtimes:
        runtime.wait_ready(plan.key, 5)
    # The same physical subset, intentionally different per-rank ticket ids.
    for rank in range(size - 1):
        runtimes[rank].native_free(NativeFreeReceipt(f"rank{rank}-ticket{100-rank}", rank,
            "attention", plan.sequence, (4, 5, 6, 7))).result(5)
    assert ledger.counts.free_pages == 13
    runtimes[-1].native_free(NativeFreeReceipt("last-ticket-independent", size-1,
        "attention", plan.sequence, (4, 5, 6, 7))).result(5)
    eventually(lambda: ledger.counts.free_pages == 14)
    assert all(not p for p in proofs)  # Uses actual native mirror receipts.
    assert ledger.view(plan.key).remaining_pages == (PageRun(2, 2),)


def test_shutdown_retains_unfenced_ownership_and_refuses_old_ready_cut(cluster):
    ledger, runtimes, proofs = cluster(2)
    plan = request(runtimes[0])
    runtimes[0].wait_ready(plan.key, 5)
    scope = runtimes[0].cancel(plan.key).result(5)
    finish(proofs, scope, [0])
    before = ledger.counts
    for runtime in runtimes:
        assert runtime.shutdown(timeout=2)
    assert ledger.counts == before
    with pytest.raises(WorksetRuntimeStopped):
        runtimes[0].ready_cut()


def test_invalid_request_failure_completes_public_future_and_keeps_existing_lease(cluster):
    ledger, runtimes, _ = cluster(1)
    plan = request(runtimes[0])
    runtimes[0].wait_ready(plan.key, 5)
    # An incompatible live retry is not a second physical grant.
    bad = runtimes[0].request(WorksetIntent("r:0", "a", "foreign", 0, 12))
    with pytest.raises(ValueError):
        bad.result(5)
    assert ledger.counts.free_pages == 13
    good = request(runtimes[0], sid="healthy-after-rejected-intent")
    runtimes[0].wait_ready(good.key, 5)


def test_all_range_partial_return_then_empty_whole_free_counts_once(cluster):
    ledger, runtimes, proofs = cluster(2)
    plan = request(runtimes[0])
    runtimes[0].wait_ready(plan.key, 5)
    part = runtimes[0].begin_return(plan.key, "all-private", pages=plan.suffix_pages).result(5)
    finish(proofs, part)
    eventually(lambda: all(r.counts().free_pages == 16 for r in runtimes))
    assert all(r.counts().live_leases == 1 for r in runtimes)
    whole = runtimes[0].cancel(plan.key).result(5)
    finish(proofs, whole)
    eventually(lambda: all(r.counts().live_leases == 0 for r in runtimes))
    assert all(r.counts().free_pages == 16 for r in runtimes)


def test_consumed_broker_lease_still_clears_exact_retire_marker(cluster):
    ledger, runtimes, proofs = cluster(2)
    plan = request(runtimes[0])
    for runtime in runtimes:
        lease = runtime.wait_ready(plan.key, 5)
        req = SimpleNamespace(origin_input_ids=list(range(12)), req_pool_idx=None,
            prefix_indices=(), mamba_pool_idx=None, mamba_ping_pong_track_buffer=None)
        runtime.broker.handoff_fresh_to_req("r:0", req, lease)
        runtime.broker.consume_suffix(lease, 12, final_prompt_chunk=True)
        assert runtime.broker.get("r:0") is None
    whole = runtimes[0].cancel(plan.key).result(5)
    finish(proofs, whole)
    eventually(lambda: all(r.counts().live_leases == 0 for r in runtimes))
    assert all("r:0" not in r.broker._tp_retire_requested for r in runtimes)
    successor = request(runtimes[0], attempt="next")
    for runtime in runtimes:
        runtime.wait_ready(successor.key, 5)


def test_prepare_failure_isolated_and_unknown_fence_never_released(cluster):
    ledger, runtimes, proofs = cluster(2)
    original = runtimes[1]._prepare
    def prepare(plan):
        if plan.key.snapshot_id == "bad":
            raise RuntimeError("unknown device submission")
        return original(plan)
    runtimes[1]._prepare = prepare
    bad = request(runtimes[0], sid="bad")
    good = request(runtimes[0], sid="good")
    runtimes[0].wait_ready(good.key, 5)
    eventually(lambda: bad.key in runtimes[1]._prepare_errors)
    assert bad.key not in runtimes[0].ready_cut().ready
    close = runtimes[0].cancel(bad.key).result(5)
    finish(proofs, close)  # Premature external proof cannot override unknown prep.
    eventually(lambda: close.sequence in runtimes[1]._native_closes or
        any(item[0] == close for item in runtimes[1]._fence_joins.get(bad.key, ())))
    assert ledger.view(bad.key) is not None and ledger.counts.free_pages == 10
    assert runtimes[0].ready_cut().ready == (good.key,)


def test_raw_prepare_done_cannot_publish_ready_before_broker_install(cluster):
    _, runtimes, _ = cluster(2)
    entered, resume = threading.Event(), threading.Event()
    original = runtimes[1]._prepared
    def delayed(descriptor):
        entered.set()
        assert resume.wait(5)
        return original(descriptor)
    runtimes[1]._prepared = delayed
    try:
        plan = request(runtimes[0])
        assert entered.wait(5)
        assert runtimes[1]._prepare_jobs[plan.key].done()
        assert runtimes[1].broker.get("r:0") is None
        assert plan.key not in runtimes[0].ready_cut().ready
    finally:
        resume.set()
    runtimes[0].wait_ready(plan.key, 5)
    assert runtimes[1].broker.get("r:0") is not None


def test_cancel_pending_prepare_retains_until_actual_worker_settles(cluster):
    ledger, runtimes, proofs = cluster(2)
    entered, resume = threading.Event(), threading.Event()
    original = runtimes[1]._prepare
    def paused(plan):
        entered.set()
        assert resume.wait(5)
        return original(plan)
    runtimes[1]._prepare = paused
    try:
        plan = request(runtimes[0])
        assert entered.wait(5)
        close = runtimes[0].cancel(plan.key).result(5)
        finish(proofs, close)
        assert ledger.counts.free_pages == 13
        assert runtimes[1].broker.get("r:0") is None
    finally:
        resume.set()
    eventually(lambda: ledger.counts.free_pages == 16)
    eventually(lambda: all(r.counts().live_leases == 0 for r in runtimes))
    assert runtimes[1].broker.get("r:0") is None
