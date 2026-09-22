"""Real CPU TCP tests of grant/preparation only, not runtime batch admission."""

from concurrent.futures import Future
from dataclasses import replace

import pytest

from sglang.srt.disaggregation.agentic_tp_events import (
    ControlUnavailable, TPEventClient, TPEventServer,
)
from sglang.srt.disaggregation.agentic_workset_ledger import (
    PageRun, WorksetLedger, LedgerDecision, ReturnPlan,
)
from sglang.srt.disaggregation.agentic_workset_tp import (
    TPWorksetExecutor, WorksetProtocolError, event_key, plan_from_wire,
    FenceProof, decision_key, decision_from_wire, decision_to_wire,
)


class CheckedFuture(Future):
    def result(self, *args, **kwargs):
        assert self.done(), "progress must never block waiting for preparation"
        return super().result(*args, **kwargs)


class Descriptor:
    def __init__(self, plan, ready=False):
        self.plan, self.ready, self.queries = plan, ready, 0

    def is_ready(self):
        self.queries += 1
        return self.ready


class Adapter:
    def __init__(self):
        self.installed, self.prepared, self.cancelled = [], [], []
        self.futures = {}
        self.fail_install = 0
        self.fail_prepare = False
        self.closed, self.close_futures, self.released = [], {}, []
        self.release_decisions = []
        self.fail_close = False
        self.fail_release = 0

    def install(self, plan):
        if self.fail_install:
            self.fail_install -= 1
            raise RuntimeError("retry idempotent CPU installation")
        self.installed.append(plan)

    def prepare(self, plan):
        self.prepared.append(plan.key)
        if self.fail_prepare:
            raise RuntimeError("possibly partially submitted CUDA initialization")
        future = self.futures[plan.key] = CheckedFuture()
        return future

    def cancel(self, plan):
        self.cancelled.append(plan.key)

    def close(self, plan, scope):
        self.closed.append(scope)
        if self.fail_close:
            raise RuntimeError("unknown partially installed close")
        future = self.close_futures[scope.sequence] = CheckedFuture()
        return future

    def release(self, plan, pages, slots, *, decision):
        if self.fail_release:
            self.fail_release -= 1
            raise RuntimeError("retry exact CPU release installation")
        self.released.append((plan.key, pages, slots))
        self.release_decisions.append(decision)


def wait(client, predicate):
    with client._condition:
        assert client._condition.wait_for(predicate, timeout=5)


@pytest.fixture
def cluster():
    resources = []

    def create(size=2, **kwargs):
        server = TPEventServer("workset-test", "secret")
        clients = [TPEventClient(server.address, run_id="workset-test", token="secret",
                                 group="P", rank=rank, size=size) for rank in range(size)]
        resources.append((server, clients))
        for client in clients:
            client.wait_ready()
        adapters = [Adapter() for _ in clients]
        executors = [TPWorksetExecutor(client, adapter, incarnation="run/P",
                                      page_count=32, page_size=4, mamba_slots=16, **kwargs)
                     for client, adapter in zip(clients, adapters)]
        ledger = WorksetLedger(incarnation="run/P", page_count=32, page_size=4,
                              mamba_slots=16, tp_size=size)
        return ledger, clients, executors, adapters

    yield create
    for server, clients in resources:
        for client in clients:
            client.close()
        server.close()


def plan(ledger, index=1):
    return ledger.grant(f"request-{index}:1", f"room:{index}", owner="direct",
                        parent_tokens=8, prompt_tokens=12,
                        checkpoint_slots=1, runtime_slots=1)


def delivered(clients, grant):
    clients[0].flush()
    for client in clients:
        wait(client, lambda: client.entry("workset-grant", event_key(grant.key)) is not None)


def progress(clients, executors):
    for executor in executors:
        executor.progress()
    for client in clients:
        client.flush()


def prepared(clients, grant, expected=1):
    wait(clients[0], lambda: clients[0].group_status(
        "workset-grant:prepared", event_key(grant.key)) == expected)


@pytest.mark.parametrize("size", [1, 2, 8])
def test_cpu_ack_future_done_and_device_event_are_distinct(cluster, size):
    ledger, clients, executors, adapters = cluster(size)
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    wait(clients[0], lambda: clients[0].command_complete("workset-grant", event_key(grant.key)))
    assert all(adapter.installed == [grant] for adapter in adapters)
    assert executors[0].ready_snapshot().ready == ()
    descriptors = [Descriptor(grant) for _ in clients]
    for adapter, descriptor in zip(adapters, descriptors):
        adapter.futures[grant.key].set_result(descriptor)
    progress(clients, executors)
    assert executors[0].ready_snapshot().ready == ()
    for descriptor in descriptors[:-1]:
        descriptor.ready = True
    progress(clients, executors)
    assert executors[0].ready_snapshot().ready == ()
    descriptors[-1].ready = True
    progress(clients, executors)
    prepared(clients, grant)
    cut = executors[0].ready_snapshot()
    assert cut.through_sequence == 1 and cut.ready == (grant.key,)
    if size > 1:
        with pytest.raises(ValueError, match="rank zero"):
            executors[-1].ready_snapshot()
    assert all(not executor._active for executor in executors)
    queries = [descriptor.queries for descriptor in descriptors]
    for _ in range(5):
        progress(clients, executors)
        assert executors[0].ready_snapshot() == cut
    assert [descriptor.queries for descriptor in descriptors] == queries


@pytest.mark.parametrize("size", [1, 2, 8])
def test_slow_future_does_not_block_later_exact_grant(cluster, size):
    ledger, clients, executors, adapters = cluster(size)
    slow, fast = plan(ledger), plan(ledger, 2)
    for grant in (slow, fast):
        executors[0].publish_grant(grant)
        delivered(clients, grant)
    progress(clients, executors)
    for adapter in adapters:
        adapter.futures[fast.key].set_result(Descriptor(fast, True))
    progress(clients, executors)
    prepared(clients, fast)
    cut = executors[0].ready_snapshot()
    assert cut.through_sequence == 2 and cut.ready == (fast.key,)
    assert all(executor.status(slow.key)["future_pending"] for executor in executors)


def test_drained_command_survives_failed_install_and_preserves_order(cluster):
    ledger, clients, executors, adapters = cluster()
    first, second = plan(ledger), plan(ledger, 2)
    for grant in (first, second):
        executors[0].publish_grant(grant)
        delivered(clients, grant)
    adapters[1].fail_install = 1
    progress(clients, executors)
    assert executors[1].installed_sequence == 0
    assert clients[1].drain_commands("workset-grant") == []
    assert executors[1].status(first.key)["install_error"] is not None
    progress(clients, executors)
    assert adapters[1].installed == [first, second]
    assert adapters[1].prepared == [first.key, second.key]


def test_out_of_order_delivery_does_not_install_out_of_sequence(cluster):
    ledger, clients, executors, adapters = cluster()
    first, second = plan(ledger), plan(ledger, 2)
    clients[0].publish_command("workset-grant", event_key(second.key), second.to_wire(), command_id=1)
    delivered(clients, second)
    progress(clients, executors)
    assert all(executor.installed_sequence == 0 for executor in executors)
    clients[0].publish_command("workset-grant", event_key(first.key), first.to_wire(), command_id=1)
    delivered(clients, first)
    progress(clients, executors)
    assert all(adapter.installed == [first, second] for adapter in adapters)


def test_duplicate_grant_does_not_reinstall_or_reprepare(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    for _ in range(2):
        executors[0].publish_grant(grant)
        delivered(clients, grant)
        progress(clients, executors)
    assert all(adapter.installed == [grant] and adapter.prepared == [grant.key] for adapter in adapters)
    with pytest.raises(WorksetProtocolError, match="different plan"):
        executors[0].publish_grant(replace(grant, owner="foreign"))


def test_publication_normalizes_mutable_input_before_retaining_authority(cluster):
    ledger, clients, executors, adapters = cluster()
    original = plan(ledger)
    mutable = list(original.parent_pages)
    executors[0].publish_grant(replace(original, parent_pages=mutable))
    mutable.clear()
    delivered(clients, original)
    progress(clients, executors)
    assert executors[0]._published[1] == original
    assert all(adapter.installed == [original] for adapter in adapters)


def test_completed_history_not_scanned_and_cancel_is_incremental(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    for adapter in adapters:
        adapter.futures[grant.key].set_result(Descriptor(grant, True))
    progress(clients, executors)
    prepared(clients, grant)
    assert executors[0].ready_snapshot().ready == (grant.key,)

    class NoHistoryTraversal(dict):
        def __iter__(self):
            raise AssertionError("history traversal")

        def values(self):
            raise AssertionError("history traversal")

        def items(self):
            raise AssertionError("history traversal")

    for executor in executors:
        executor._pending = NoHistoryTraversal(executor._pending)
        executor._keys = NoHistoryTraversal(executor._keys)
        executor._published = NoHistoryTraversal(executor._published)
    progress(clients, executors)
    assert executors[0].ready_snapshot().ready == (grant.key,)
    executors[0].cancel(grant.key)
    clients[0].flush()
    for client in clients:
        wait(client, lambda: client.receipt("workset-grant", event_key(grant.key)) == -1)
    progress(clients, executors)
    prepared(clients, grant, -1)
    assert executors[0].ready_snapshot().ready == ()
    assert all(adapter.cancelled == [grant.key] for adapter in adapters)
    assert all(not executor._active for executor in executors)
    assert ledger.counts.free_pages == 29


def test_active_budget_rotates_without_head_of_line_wait(cluster):
    ledger, clients, executors, adapters = cluster(progress_budget=1)
    first, second = plan(ledger), plan(ledger, 2)
    for grant in (first, second):
        executors[0].publish_grant(grant)
        delivered(clients, grant)
    for _ in range(3):
        progress(clients, executors)
    for adapter in adapters:
        adapter.futures[second.key].set_result(Descriptor(second, True))
    for _ in range(2):
        progress(clients, executors)
    prepared(clients, second)
    assert executors[0].ready_snapshot().ready == (second.key,)
    assert all(executor.status(first.key)["future_pending"] for executor in executors)


@pytest.mark.parametrize("started", [False, True])
def test_cancel_never_frees_or_forges_physical_fence(cluster, started):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    if started:
        progress(clients, executors)
    executors[0].cancel(grant.key)
    clients[0].flush()
    for client in clients:
        wait(client, lambda: client.receipt("workset-grant", event_key(grant.key)) == -1)
    progress(clients, executors)
    assert all(adapter.cancelled == [grant.key] for adapter in adapters)
    if started:
        assert all(not adapter.futures[grant.key].cancelled() for adapter in adapters)
        for adapter in adapters:
            adapter.futures[grant.key].set_result(Descriptor(grant, True))
        progress(clients, executors)
    else:
        assert all(adapter.prepared == [] for adapter in adapters)
    prepared(clients, grant, -1)
    assert executors[0].ready_snapshot().ready == ()
    assert ledger.counts.free_pages == 29 and ledger.counts.free_mamba_slots == 14
    assert ledger.free(grant.key) is None


@pytest.mark.parametrize("submit_failure", [False, True])
def test_failed_preparation_is_retained_without_stalling_healthy_plan(cluster, submit_failure):
    ledger, clients, executors, adapters = cluster()
    bad, good = plan(ledger), plan(ledger, 2)
    executors[0].publish_grant(bad)
    delivered(clients, bad)
    adapters[1].fail_prepare = submit_failure
    progress(clients, executors)
    adapters[1].fail_prepare = False
    if not submit_failure:
        adapters[1].futures[bad.key].set_exception(RuntimeError("unknown CUDA submission"))
    executors[0].publish_grant(good)
    delivered(clients, good)
    progress(clients, executors)
    for adapter in adapters:
        adapter.futures[good.key].set_result(Descriptor(good, True))
    progress(clients, executors)
    prepared(clients, good)
    assert executors[0].ready_snapshot().ready == (good.key,)
    assert executors[1].status(bad.key)["prepare_error"] is not None
    assert adapters[1].prepared.count(bad.key) == 1
    assert ledger.counts.free_pages == 26


@pytest.mark.parametrize("wrong", ["missing", "foreign"])
def test_future_result_requires_exact_descriptor_plan_and_physical_event(cluster, wrong):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    for adapter in adapters:
        descriptor = None if wrong == "missing" else Descriptor(replace(grant, owner="foreign"), True)
        adapter.futures[grant.key].set_result(descriptor)
    progress(clients, executors)
    prepared(clients, grant, -1)
    assert executors[0].ready_snapshot().ready == ()


@pytest.mark.parametrize("mutation", [
    {"suffix_pages": []}, {"suffix_pages": [[1, 1]]}, {"version": True},
    {"extra": 1}, {"owner": ""}, {"parent_pages": [[1, -1]]},
])
def test_wire_rejects_incomplete_malformed_or_overlapping_plan(cluster, mutation):
    ledger, _, _, _ = cluster()
    wire = plan(ledger).to_wire()
    wire.update(mutation)
    with pytest.raises(WorksetProtocolError):
        plan_from_wire(wire)


def test_foreign_pool_bounds_and_follower_authority_rejected(cluster):
    ledger, _, executors, _ = cluster()
    grant = plan(ledger)
    with pytest.raises(ValueError, match="rank zero"):
        executors[1].publish_grant(grant)
    with pytest.raises(WorksetProtocolError, match="incarnation"):
        executors[0].publish_grant(replace(grant, key=replace(grant.key, incarnation="old")))
    with pytest.raises(WorksetProtocolError, match="physical pool"):
        executors[0].publish_grant(replace(grant, parent_pages=(PageRun(99, 2),)))


def test_overlapping_grants_fail_closed_without_releasing_first(cluster):
    ledger, clients, executors, adapters = cluster()
    first, second = plan(ledger), plan(ledger, 2)
    second = replace(second, parent_pages=first.parent_pages)
    for grant in (first, second):
        executors[0].publish_grant(grant)
        delivered(clients, grant)
    for executor, adapter in zip(executors, adapters):
        with pytest.raises(WorksetProtocolError, match="already owned"):
            executor.progress()
        assert adapter.installed == [first]
        with pytest.raises(WorksetProtocolError):
            executor.progress()
    assert ledger.counts.free_pages == 26


def test_disconnect_retains_unknown_future_and_all_addresses(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    clients[1].close()
    wait(clients[0], lambda: clients[0]._error is not None)
    with pytest.raises(ControlUnavailable):
        executors[0].progress()
    with pytest.raises(ControlUnavailable):
        executors[0].ready_snapshot()
    assert not adapters[0].futures[grant.key].done()
    assert ledger.counts.free_pages == 29


def deliver_decision(clients, scope):
    clients[0].flush()
    for client in clients:
        wait(client, lambda: client.entry("workset-grant:decisions", decision_key(scope)) is not None)


def close_done(clients, scope):
    wait(clients[0], lambda: clients[0].group_status("workset-grant:fenced", decision_key(scope)) == 1)


def complete_preparation(clients, executors, adapters, grant):
    for adapter in adapters:
        future = adapter.futures.get(grant.key)
        if future is not None and not future.done():
            future.set_result(Descriptor(grant, True))
    progress(clients, executors)


def complete_close(adapters, scope, ranks=None):
    for rank in range(len(adapters)) if ranks is None else ranks:
        adapters[rank].close_futures[scope.sequence].set_result(
            FenceProof(scope.key, scope.sequence, True, True))


def commit_fences(ledger, executor):
    decisions = []
    for scope in executor.drain_fenced():
        for rank in range(ledger.tp_size):
            if isinstance(scope, ReturnPlan):
                ledger.report_return_fence(scope, rank, quiet=True, unreferenced=True)
            else:
                ledger.report_fence(scope.key, rank, quiet=True, unreferenced=True)
        decision = ledger.commit_return(scope) if isinstance(scope, ReturnPlan) else ledger.free(scope.key)
        assert decision is not None
        executor.publish_decision(decision)
        decisions.append(decision)
    return decisions


@pytest.mark.parametrize("size", [1, 2, 8])
def test_exact_cancel_all_rank_fence_free_and_successor_address_reuse(cluster, size):
    ledger, clients, executors, adapters = cluster(size)
    first = plan(ledger)
    executors[0].publish_grant(first)
    delivered(clients, first)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, first)
    scope = ledger.cancel(first.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    assert all(not executor.may_use(first.key, pages=first.parent_pages) for executor in executors)
    assert ledger.counts.free_pages == 29
    complete_close(adapters, scope, range(size - 1))
    progress(clients, executors)
    assert executors[0].drain_fenced() == []
    with pytest.raises(WorksetProtocolError, match="all-rank"):
        executors[0].publish_decision(LedgerDecision(scope.sequence + 1, "free", first.key))
    complete_close(adapters, scope, [size - 1])
    progress(clients, executors)
    close_done(clients, scope)
    free, = commit_fences(ledger, executors[0])
    deliver_decision(clients, free)
    successor = ledger.grant(first.key.snapshot_id, "new-attempt", owner="slow",
                             parent_tokens=8, prompt_tokens=12, checkpoint_slots=1, runtime_slots=1)
    executors[0].publish_grant(successor)
    delivered(clients, successor)
    progress(clients, executors)
    assert successor.parent_pages == first.parent_pages
    for executor, adapter in zip(executors, adapters):
        assert executor.installed_sequence == successor.sequence
        assert executor.status(first.key) is None
        assert executor.status(successor.key)["installed"]
        assert adapter.released == [(first.key, (PageRun(1, 3),), (PageRun(1, 2),))]
        assert not executor._closes and first.key not in executor._active
    # Replayed old cancel/free do not close or recycle the newer exact owner.
    for old in (scope, free):
        executors[0].publish_decision(old)
        progress(clients, executors)
    assert all(executor.may_use(successor.key, pages=successor.parent_pages) for executor in executors)
    assert all(len(adapter.released) == 1 for adapter in adapters)


@pytest.mark.parametrize("size", [2, 8])
def test_terminal_event_retirement_waits_for_last_rank_free_ack(cluster, size, monkeypatch):
    ledger, clients, executors, adapters = cluster(size)
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    scope = ledger.cancel(grant.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    complete_close(adapters, scope)
    progress(clients, executors)
    close_done(clients, scope)
    free, = commit_fences(ledger, executors[0])
    deliver_decision(clients, free)

    retired = []
    retire = clients[0].retire_workset
    def record(*args):
        retired.append(args)
        return retire(*args)
    monkeypatch.setattr(clients[0], "retire_workset", record)
    for executor in executors[:-1]:
        executor.progress()
    for client in clients:
        client.flush()
    executors[0].progress()
    assert retired == []
    assert executors[0].status(grant.key) is None
    assert executors[-1].status(grant.key) is not None
    assert clients[0].entry("workset-grant", event_key(grant.key)) is not None
    assert decision_key(free) in executors[0]._retire_pending

    executors[-1].progress()
    clients[-1].flush()
    wait(clients[0], lambda: clients[0].command_complete(
        "workset-grant:decisions", decision_key(free)))
    executors[0].progress()
    clients[0].flush()
    assert retired == [("workset-grant", event_key(grant.key), decision_key(free))]
    assert not executors[0]._retire_pending
    for client in clients:
        wait(client, lambda: client.entry("workset-grant", event_key(grant.key)) is None)
    for _ in range(3):
        progress(clients, executors)
    assert len(retired) == 1


@pytest.mark.parametrize("size", [2, 8])
def test_partial_return_retains_shared_parent_and_whole_free_never_double_returns(cluster, size):
    ledger, clients, executors, adapters = cluster(size)
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    subset = ledger.begin_return(grant.key, "req-private-lastref", pages=grant.suffix_pages,
                                 slots=grant.runtime_slots)
    executors[0].publish_decision(subset)
    deliver_decision(clients, subset)
    progress(clients, executors)
    assert all(executor.may_use(grant.key, pages=grant.parent_pages) for executor in executors)
    assert all(not executor.may_use(grant.key, pages=grant.suffix_pages) for executor in executors)
    complete_close(adapters, subset)
    progress(clients, executors)
    close_done(clients, subset)
    returned, = commit_fences(ledger, executors[0])
    deliver_decision(clients, returned)
    progress(clients, executors)
    assert ledger.counts.free_pages == 30 and ledger.counts.free_mamba_slots == 15
    assert all(executor.status(grant.key)["remaining_pages"] == grant.parent_pages for executor in executors)
    whole = ledger.cancel(grant.key)
    executors[0].publish_decision(whole)
    deliver_decision(clients, whole)
    progress(clients, executors)
    complete_close(adapters, whole)
    progress(clients, executors)
    close_done(clients, whole)
    free, = commit_fences(ledger, executors[0])
    deliver_decision(clients, free)
    progress(clients, executors)
    assert ledger.counts.free_pages == 32 and ledger.counts.free_mamba_slots == 16
    for adapter in adapters:
        assert adapter.released == [(grant.key, grant.suffix_pages, grant.runtime_slots),
                                    (grant.key, grant.parent_pages, grant.checkpoint_slots)]


def test_close_proof_cannot_overtake_unknown_preparation_future(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    scope = ledger.cancel(grant.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    complete_close(adapters, scope)
    progress(clients, executors)
    assert executors[0].drain_fenced() == []
    assert all(not executor.close_status(scope)["reported"] for executor in executors)
    complete_preparation(clients, executors, adapters, grant)
    close_done(clients, scope)
    assert executors[0].drain_fenced() == [scope]
    assert ledger.counts.free_pages == 29


@pytest.mark.parametrize("failure", ["submit", "future", "wrong_sequence", "wrong_lease", "not_quiet", "referenced"])
def test_close_failure_or_invalid_proof_never_recycles(cluster, failure):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    scope = ledger.cancel(grant.key)
    adapters[-1].fail_close = failure == "submit"
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    complete_close(adapters, scope, [0])
    if failure != "submit":
        future = adapters[-1].close_futures[scope.sequence]
        if failure == "future":
            future.set_exception(RuntimeError("unknown physical completion"))
        else:
            proof = FenceProof(scope.key, scope.sequence, True, True)
            proof = {"wrong_sequence": replace(proof, close_sequence=scope.sequence + 1),
                     "wrong_lease": replace(proof, key=replace(scope.key, version=99)),
                     "not_quiet": replace(proof, quiet=False),
                     "referenced": replace(proof, unreferenced=False)}[failure]
            future.set_result(proof)
    progress(clients, executors)
    assert executors[-1].close_status(scope)["error"] is not None
    assert executors[0].drain_fenced() == []
    assert ledger.free(grant.key) is None and ledger.counts.free_pages == 29
    assert all(adapter.released == [] for adapter in adapters)


def test_partial_preparation_exception_requires_explicit_real_close_proof(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    adapters[-1].fail_prepare = True
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    scope = ledger.cancel(grant.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    assert executors[0].drain_fenced() == []
    complete_close(adapters, scope)
    progress(clients, executors)
    close_done(clients, scope)
    assert executors[0].drain_fenced() == [scope]


def test_unknown_release_installation_holds_later_reuse_but_retries_exactly(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    scope = ledger.cancel(grant.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    complete_close(adapters, scope)
    progress(clients, executors)
    close_done(clients, scope)
    free, = commit_fences(ledger, executors[0])
    deliver_decision(clients, free)
    successor = plan(ledger, 2)
    executors[0].publish_grant(successor)
    delivered(clients, successor)
    adapters[-1].fail_release = 1
    progress(clients, executors)
    assert executors[-1].installed_sequence == scope.sequence
    assert executors[-1].status(grant.key) is not None
    assert not executors[-1].status(successor.key)["installed"]
    progress(clients, executors)
    assert executors[-1].installed_sequence == successor.sequence
    assert len(adapters[-1].released) == 1


def test_whole_close_can_subsume_unfinished_subset_fence_without_double_release(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    subset = ledger.begin_return(grant.key, "unfinished-private", pages=grant.suffix_pages)
    whole = ledger.cancel(grant.key)
    for scope in (subset, whole):
        executors[0].publish_decision(scope)
        deliver_decision(clients, scope)
    progress(clients, executors)
    # Whole proof includes remaining references and outstanding close observers.
    complete_close(adapters, whole)
    progress(clients, executors)
    close_done(clients, whole)
    free, = commit_fences(ledger, executors[0])
    deliver_decision(clients, free)
    progress(clients, executors)
    complete_close(adapters, subset)
    progress(clients, executors)
    assert executors[0].drain_fenced() == []
    assert all(len(adapter.released) == 1 for adapter in adapters)
    assert all(not executor._closes and not executor._closing_active for executor in executors)
    assert ledger.counts.free_pages == 32


def test_future_done_but_device_event_pending_cannot_ack_close(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    descriptors = [Descriptor(grant, False) for _ in clients]
    for adapter, descriptor in zip(adapters, descriptors):
        adapter.futures[grant.key].set_result(descriptor)
    progress(clients, executors)
    scope = ledger.cancel(grant.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    complete_close(adapters, scope)
    progress(clients, executors)
    assert executors[0].drain_fenced() == []
    assert all(not executor.close_status(scope)["reported"] for executor in executors)
    for descriptor in descriptors:
        descriptor.ready = True
    progress(clients, executors)
    close_done(clients, scope)
    assert executors[0].drain_fenced() == [scope]


def test_slow_close_future_does_not_block_other_grant_preparation(cluster):
    ledger, clients, executors, adapters = cluster()
    old = plan(ledger)
    executors[0].publish_grant(old)
    delivered(clients, old)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, old)
    closing = ledger.cancel(old.key)
    executors[0].publish_decision(closing)
    deliver_decision(clients, closing)
    fresh = plan(ledger, 2)
    executors[0].publish_grant(fresh)
    delivered(clients, fresh)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, fresh)
    prepared(clients, fresh)
    assert executors[0].ready_snapshot().ready == (fresh.key,)
    assert all(executor.close_status(closing)["future_pending"] for executor in executors)
    assert ledger.counts.free_pages == 26


def test_cancel_installed_in_same_cut_as_grant_starts_no_preparation(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    scope = ledger.cancel(grant.key)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    assert all(adapter.prepared == [] for adapter in adapters)
    complete_close(adapters, scope)
    progress(clients, executors)
    close_done(clients, scope)
    assert executors[0].drain_fenced() == [scope]


def test_whole_and_subset_ready_notifications_commit_only_whole(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    subset = ledger.begin_return(grant.key, "same-batch", pages=grant.suffix_pages)
    whole = ledger.cancel(grant.key)
    for scope in (subset, whole):
        executors[0].publish_decision(scope)
        deliver_decision(clients, scope)
    progress(clients, executors)
    for scope in (subset, whole):
        complete_close(adapters, scope)
    progress(clients, executors)
    for scope in (subset, whole):
        close_done(clients, scope)
    assert executors[0].drain_fenced() == [whole]


def test_free_cpu_install_is_not_repeated_after_ack_submission_failure(cluster, monkeypatch):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    scope = ledger.cancel(grant.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    complete_close(adapters, scope)
    progress(clients, executors)
    close_done(clients, scope)
    free, = commit_fences(ledger, executors[0])
    deliver_decision(clients, free)
    original = clients[-1].ack_command
    failures = [1]

    def fail_once(namespace, key, command_id):
        if namespace == "workset-grant:decisions" and key == decision_key(free) and failures:
            failures.pop()
            raise ControlUnavailable("unknown ACK enqueue")
        return original(namespace, key, command_id)

    monkeypatch.setattr(clients[-1], "ack_command", fail_once)
    with pytest.raises(ControlUnavailable):
        progress(clients, executors)
    assert len(adapters[-1].released) == 1
    progress(clients, executors)
    assert executors[-1].installed_sequence == free.sequence
    assert len(adapters[-1].released) == 1


def test_disconnect_during_close_keeps_exact_ownership_and_future(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    scope = ledger.cancel(grant.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    clients[-1].close()
    wait(clients[0], lambda: clients[0]._error is not None)
    with pytest.raises(ControlUnavailable):
        executors[0].drain_fenced()
    assert not adapters[0].close_futures[scope.sequence].done()
    assert ledger.free(grant.key) is None and ledger.counts.free_pages == 29


@pytest.mark.parametrize("mutation", [
    {"sequence": True}, {"version": False}, {"operation": "force_free"},
    {"return_id": "wrong-whole-subset"}, {"extra": 1},
])
def test_decision_wire_strict_identity_and_operation_validation(cluster, mutation):
    ledger, _, _, _ = cluster()
    grant = plan(ledger)
    scope = ledger.cancel(grant.key)
    wire = decision_to_wire(scope)
    wire.update(mutation)
    with pytest.raises(WorksetProtocolError):
        decision_from_wire(wire)


def test_retirement_and_reuse_progress_never_scans_historical_maps(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)

    class NoHistoryTraversal(dict):
        def __iter__(self):
            raise AssertionError("history traversal")

        def values(self):
            raise AssertionError("history traversal")

        def items(self):
            raise AssertionError("history traversal")

    for executor in executors:
        for name in ("_pending", "_keys", "_published", "_seen", "_publication_log", "_closes"):
            setattr(executor, name, NoHistoryTraversal(getattr(executor, name)))
    scope = ledger.cancel(grant.key)
    executors[0].publish_decision(scope)
    deliver_decision(clients, scope)
    progress(clients, executors)
    complete_close(adapters, scope)
    progress(clients, executors)
    close_done(clients, scope)
    free, = commit_fences(ledger, executors[0])
    deliver_decision(clients, free)
    successor = plan(ledger, 2)
    executors[0].publish_grant(successor)
    delivered(clients, successor)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, successor)
    prepared(clients, successor)
    assert executors[0].ready_snapshot().ready == (successor.key,)
    assert all(executor.status(grant.key) is None for executor in executors)


def test_foreign_subset_range_cannot_close_another_lease_addresses(cluster):
    ledger, clients, executors, adapters = cluster()
    first, second = plan(ledger), plan(ledger, 2)
    for grant in (first, second):
        executors[0].publish_grant(grant)
        delivered(clients, grant)
    progress(clients, executors)
    forged = ReturnPlan(first.key, "cross-owner", 3, second.parent_pages, ())
    executors[0].publish_decision(forged)
    deliver_decision(clients, forged)
    for executor in executors:
        with pytest.raises(WorksetProtocolError, match="exactly owned"):
            executor.progress()
    assert all(not adapter.closed and not adapter.released for adapter in adapters)
    assert ledger.counts.free_pages == 26


def test_all_range_partial_return_and_empty_free_have_distinct_exact_decisions(cluster):
    ledger, clients, executors, adapters = cluster()
    grant = plan(ledger)
    executors[0].publish_grant(grant)
    delivered(clients, grant)
    progress(clients, executors)
    complete_preparation(clients, executors, adapters, grant)
    subset = ledger.begin_return(grant.key, "all-native-lastrefs",
                                 pages=grant.parent_pages + grant.suffix_pages,
                                 slots=grant.checkpoint_slots + grant.runtime_slots)
    executors[0].publish_decision(subset)
    deliver_decision(clients, subset)
    progress(clients, executors)
    complete_close(adapters, subset)
    progress(clients, executors)
    close_done(clients, subset)
    returned, = commit_fences(ledger, executors[0])
    deliver_decision(clients, returned)
    progress(clients, executors)
    assert ledger.counts.live_leases == 1 and ledger.counts.free_pages == 32
    assert all(executor.local_view(grant.key) is not None for executor in executors)
    whole = ledger.cancel(grant.key)
    executors[0].publish_decision(whole)
    deliver_decision(clients, whole)
    progress(clients, executors)
    complete_close(adapters, whole)
    progress(clients, executors)
    close_done(clients, whole)
    free, = commit_fences(ledger, executors[0])
    deliver_decision(clients, free)
    progress(clients, executors)
    for adapter in adapters:
        assert adapter.release_decisions == [returned, free]
        assert adapter.released[-1] == (grant.key, (), ())
    assert ledger.counts.live_leases == 0
