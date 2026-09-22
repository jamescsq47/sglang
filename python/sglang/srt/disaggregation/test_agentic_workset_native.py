"""CPU boundary tests for opt-in native last-reference ownership bridging."""
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.agentic_workset_ledger import WorksetLedger, PageRun
from sglang.srt.disaggregation.agentic_workset_native import (
    NativeLastRefFreeAdapter, NativeFreeFailure, install_checkpoint_rotation,
)
from sglang.srt.disaggregation.agentic_mamba_prefill import (
    fork_prefill_checkpoint, release_prefill_checkpoint, reserve_prefill_state,
)


def completed(value=None):
    result = Future()
    result.set_result(value)
    return result


def ledger(**kwargs):
    return WorksetLedger(incarnation="p-group", page_count=16, page_size=4,
                         mamba_slots=16, **kwargs)


def grant(owner, sid="a", **kwargs):
    args = dict(owner="fresh", parent_tokens=0, prompt_tokens=8,
                checkpoint_slots=0, runtime_slots=2)
    args.update(kwargs)
    return owner.grant(sid, "attempt", **args)


def commit(owner, plan):
    for rank in range(owner.tp_size):
        owner.report_return_fence(plan, rank, quiet=True, unreferenced=True)
    return owner.commit_return(plan)


@pytest.mark.parametrize("tp", [1, 2, 8])
def test_native_return_deduplicates_pages_groups_owners_and_requires_fences(tp):
    owner = ledger(tp_size=tp)
    first, second = grant(owner), grant(owner, "b")
    plans = owner.native_return("ticket", "attention", (4, 5, 7, 12, 15), owner.counts.sequence)
    assert [plan.key for plan in plans] == [first.key, second.key]
    assert [plan.pages for plan in plans] == [(PageRun(1, 1),), (PageRun(3, 1),)]
    assert owner.counts.free_pages == 12
    for plan in plans:
        assert owner.commit_return(plan) is None
        commit(owner, plan)
    assert owner.counts.free_pages == 14


def test_native_return_validates_every_address_before_any_owner_mutation():
    owner = ledger()
    first, second = grant(owner), grant(owner, "b")
    before = owner.counts
    with pytest.raises(ValueError):
        owner.native_return("bad", "attention", (4, 12, 900), before.sequence)
    assert owner.counts == before
    plans = owner.native_return("valid", "attention", (4, 12), before.sequence)
    assert len(plans) == 2
    with pytest.raises(ValueError):
        owner.native_return("overlap", "attention", (4, 8), owner.counts.sequence)
    # Failed mixed pending/new request did not claim page 2.
    assert owner.native_return("still-free", "attention", (8,), owner.counts.sequence)


def test_native_return_replay_and_capture_epoch_after_reuse():
    owner = ledger()
    old = grant(owner)
    epoch = owner.counts.sequence
    original = owner.native_return("old", "attention", (4,), epoch)
    commit(owner, original[0])
    new = grant(owner, "b", prompt_tokens=4)
    assert new.suffix_pages == (PageRun(1, 1),)
    assert owner.native_return("old", "attention", (4,), epoch) is original
    with pytest.raises(ValueError, match="newer"):
        owner.native_return("late", "attention", (4,), epoch)
    with pytest.raises(ValueError, match="changed"):
        owner.native_return("old", "attention", (8,), epoch)
    assert owner.current_view("b").plan.key == new.key


def test_native_return_mamba_and_receipt_bound_fail_closed():
    owner = ledger(max_native_returns=1)
    plan = grant(owner)
    result = owner.native_return("state", "mamba", (1, 1), owner.counts.sequence)
    assert result[0].slots == (PageRun(1, 1),)
    before = owner.counts
    with pytest.raises(RuntimeError, match="capacity"):
        owner.native_return("other", "mamba", (2,), owner.counts.sequence)
    assert owner.counts == before


def test_committed_native_receipts_compact_without_releasing_live_returns():
    owner = ledger(max_native_returns=2)
    grant(owner)
    source = "run:rank0:native-free"
    first = owner.native_return(source + "1", "attention", (4,), owner.counts.sequence)
    second = owner.native_return(source + "2", "attention", (8,), owner.counts.sequence)
    # A pending return cannot be forgotten just to make room for a new one.
    with pytest.raises(RuntimeError, match="capacity"):
        owner.native_return(source + "3", "mamba", (1,), owner.counts.sequence)
    commit(owner, first[0])
    third = owner.native_return(source + "3", "mamba", (1,), owner.counts.sequence)
    assert len(third) == 1
    assert len(owner._native_returns) == 2
    with pytest.raises(ValueError, match="retired"):
        owner.native_return(source + "1", "attention", (4,), owner.counts.sequence)
    # Still-live receipts preserve exact idempotent retry semantics.
    assert owner.native_return(source + "2", "attention", (8,), second[0].sequence - 1) is second
    commit(owner, second[0])
    commit(owner, third[0])


def test_native_receipt_capacity_is_not_a_lifetime_generation_limit():
    owner = ledger(max_native_returns=2)
    for ordinal in range(1, 257):
        plan = grant(owner, f"generation-{ordinal}", prompt_tokens=4)
        ticket = f"p-group:rank0:native-free{ordinal}"
        returned = owner.native_return(
            ticket, "attention", (plan.suffix_pages[0].start * 4,),
            owner.counts.sequence,
        )
        commit(owner, returned[0])
        owner.cancel(plan.key)
        owner.report_fence(plan.key, 0, quiet=True, unreferenced=True)
        owner.free(plan.key)
    assert owner.counts.free_pages == 16
    assert len(owner._native_returns) <= 2


def bridge(owner, sink, **kwargs):
    return NativeLastRefFreeAdapter(incarnation="run", rank=0, page_size=4,
                                   counts=lambda: owner.counts, on_ready=sink, **kwargs)


def test_cleanup_thread_needs_native_state_lock_for_real_bridge():
    owner = ledger()
    grant(owner)
    receipts = []
    adapter = bridge(owner, lambda receipt: (receipts.append(receipt), completed())[1])
    native_lock = threading.RLock()
    result = []

    def release():
        adapter.allow_cleanup_owner(threading.current_thread(), native_lock)
        with pytest.raises(NativeFreeFailure, match="state lock"):
            adapter.free(torch.tensor([4]), resource="attention")
        with native_lock:
            result.append(adapter.free(torch.tensor([4]), resource="attention"))

    try:
        thread = threading.Thread(target=release)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert result[0].result(timeout=5).indices == (4,)
        assert len(receipts) == 1
        assert receipts[0].indices == (4,)
    finally:
        adapter.shutdown()


def test_native_bridge_private_clone_and_cpu_readback_only_on_worker(monkeypatch):
    owner = ledger()
    grant(owner)
    native = threading.current_thread()
    cpu = torch.Tensor.cpu
    def guarded_cpu(tensor, *args, **kwargs):
        assert threading.current_thread() is not native
        return cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", guarded_cpu)
    receipts = []
    adapter = bridge(owner, lambda receipt: (receipts.append(receipt), completed())[1])
    try:
        adapter.free_group_begin()
        source = torch.tensor([4, 5])
        result = adapter.free(source, resource="attention")
        source.fill_(900)
        assert not result.done()
        adapter.free_group_end()
        receipt = result.result(2)
        assert receipt.indices == (4, 5)
        assert owner.counts.free_pages == 14  # acceptance is not physical free
        assert not adapter._pending
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(NativeFreeFailure, match="owner"):
                pool.submit(adapter.free, source, resource="attention").result()
    finally:
        adapter.shutdown()


def test_bridge_overflow_and_shutdown_do_not_drop_or_free_pending():
    owner = ledger()
    grant(owner)
    ack = Future()
    adapter = bridge(owner, lambda receipt: ack, max_pending=1)
    result = adapter.free(torch.tensor([4]), resource="attention")
    with pytest.raises(NativeFreeFailure, match="full"):
        adapter.free(torch.tensor([8]), resource="attention")
    before = owner.counts
    adapter.shutdown()
    with pytest.raises(NativeFreeFailure):
        result.result()
    assert adapter._pending and owner.counts == before


def test_bridge_failed_receipt_sink_retains_ownership():
    owner = ledger()
    grant(owner)
    def fail(receipt):
        result = Future()
        result.set_exception(RuntimeError("actor unavailable"))
        return result
    adapter = bridge(owner, fail)
    try:
        before = owner.counts
        result = adapter.free(torch.tensor([4]), resource="attention")
        with pytest.raises(RuntimeError, match="actor unavailable"):
            result.result(2)
        assert owner.counts == before and adapter._pending
        with pytest.raises(NativeFreeFailure):
            adapter.check_health()
    finally:
        adapter.shutdown()


def test_exact_event_observer_uses_existing_worker_without_readback():
    owner = ledger()
    native = threading.current_thread()
    entered, release = threading.Event(), threading.Event()
    class Event:
        def synchronize(self):
            assert threading.current_thread() is not native
            entered.set()
            assert release.wait(2)
    adapter = bridge(owner, lambda receipt: pytest.fail("exact event must not enter raw sink"))
    try:
        before = owner.counts
        result = adapter.observe_event(Event())
        assert entered.wait(2) and not result.done()
        release.set()
        assert result.result(2) is True
        assert adapter.observe_event(None).result(2) is True
        assert owner.counts == before and not adapter._pending
    finally:
        release.set()
        adapter.shutdown()


def test_exact_event_failure_is_not_quiet_or_free():
    owner = ledger()
    class Event:
        def synchronize(self):
            raise RuntimeError("unknown CUDA fence")
    adapter = bridge(owner, lambda _: completed())
    try:
        with pytest.raises(RuntimeError, match="unknown CUDA"):
            adapter.observe_event(Event()).result(2)
        assert adapter._pending
        with pytest.raises(NativeFreeFailure):
            adapter.observe_event(None)
    finally:
        adapter.shutdown()


def test_controller_native_return_wakes_capacity_wait_only_after_all_rank_commit():
    from sglang.srt.disaggregation.agentic_workset_controller import WorksetController, WorksetIntent
    owner = WorksetLedger(incarnation="run", page_count=2, page_size=4, tp_size=2)
    controller = WorksetController(owner)
    try:
        first = controller.request(WorksetIntent("a", "a", "fresh", 0, 8)).result(2)
        waiting = controller.request(WorksetIntent("b", "a", "slow", 0, 4))
        plans = controller.native_return("free", "attention", (4,), owner.counts.sequence).result(2)
        assert not waiting.done()
        controller.report_return_fence(plans[0], 0, quiet=True, unreferenced=True).result(2)
        assert controller.commit_return(plans[0]).result(2) is None
        assert not waiting.done()
        controller.report_return_fence(plans[0], 1, quiet=True, unreferenced=True).result(2)
        assert controller.commit_return(plans[0]).result(2).operation == "return"
        assert waiting.result(2).suffix_pages == (PageRun(1, 1),)
    finally:
        controller.shutdown()


@pytest.mark.parametrize("paged", [False, True])
def test_native_allocator_guards_are_opt_in_and_no_second_free_list(paged):
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator
    args = dict(size=64, dtype=torch.int64, device="cpu", kvcache=None, need_sort=False)
    allocator = (PagedTokenToKVPoolAllocator(page_size=4, **args) if paged
                 else TokenToKVPoolAllocator(**args))
    sample = allocator.alloc(4)
    allocator.free(sample)
    owner = ledger() if paged else WorksetLedger(incarnation="run", page_count=64, page_size=1)
    adapter = NativeLastRefFreeAdapter(incarnation="run", rank=0,
        page_size=4 if paged else 1, counts=lambda: owner.counts, on_ready=lambda r: completed())
    try:
        allocator.install_workset_adapter(adapter)
        assert allocator.free_pages is None and allocator.release_pages is None
        assert allocator.available_size() == 64
        for call in (lambda: allocator.alloc(4), allocator.clear, allocator.backup_state):
            with pytest.raises(RuntimeError):
                call()
        if paged:
            with pytest.raises(RuntimeError):
                allocator.alloc_extend(None, None, None, None, None, 4)
    finally:
        adapter.shutdown()


def rotation_fixture():
    owner = ledger()
    plan = grant(owner)
    indices = torch.tensor([1, 2])
    req = SimpleNamespace(_agentic_mamba_prefill_checkpoint=indices,
                          mamba_pool_idx=torch.tensor(3))
    receipts = []
    rotation = install_checkpoint_rotation(req, plan.key, (1, 2), indices,
        lambda receipt: (receipts.append(receipt), completed())[1])
    return req, rotation, receipts


def test_checkpoint_rotation_multiple_chunks_never_global_allocates():
    req, rotation, receipts = rotation_fixture()
    pool = SimpleNamespace(copy_from=lambda source, target: None,
                           alloc=lambda n: pytest.fail("unreserved allocation"))
    req_pool = SimpleNamespace(mamba_pool=pool, enable_mamba_extra_buffer=False)
    previous = None
    for chunk in range(8):
        assert reserve_prefill_state(req, req_pool, None) is not None
        checkpoint = fork_prefill_checkpoint(req, pool, torch.tensor([3]))
        if previous is not None:
            rotation.release(previous)
        previous = checkpoint
    assert receipts == []
    release_prefill_checkpoint(req, pool)
    assert len(receipts) == 1  # donated checkpoint remains referenced
    rotation.release(previous)
    assert {r.slot for r in receipts} == {1, 2}
    release_prefill_checkpoint(req, pool)
    assert len(receipts) == 2


def test_checkpoint_shared_reference_blocks_reuse_and_late_duplicate_is_rejected():
    req, rotation, receipts = rotation_fixture()
    first, second = rotation.take(), rotation.take()
    assert not rotation.can_take()
    with pytest.raises(NativeFreeFailure, match="still referenced"):
        rotation.take()
    rotation.release(first)
    replacement = rotation.take()
    with pytest.raises(NativeFreeFailure, match="duplicate"):
        rotation.release(first)
    rotation.close()
    assert not receipts
    rotation.release(replacement)
    rotation.release(second)
    assert len(receipts) == 2


def test_checkpoint_copy_failure_returns_slot_to_same_lease():
    req, rotation, receipts = rotation_fixture()
    def fail(source, target):
        raise RuntimeError("copy failed")
    with pytest.raises(RuntimeError, match="copy failed"):
        fork_prefill_checkpoint(req, SimpleNamespace(copy_from=fail), torch.tensor([3]))
    assert receipts == []
    rotation.close()
    assert len(receipts) == 2


def test_checkpoint_failed_sink_retains_original_exact_owner():
    req, rotation, receipts = rotation_fixture()
    failed = Future()
    failed.set_exception(RuntimeError("control disconnected"))
    rotation._on_return = lambda receipt: failed
    rotation.close()
    with pytest.raises(NativeFreeFailure):
        rotation.take()
    assert rotation.key is not None and rotation._returns


def test_request_cleanup_closes_rotation_without_anonymous_checkpoint():
    from sglang.srt.mem_cache.common import release_kv_cache
    req, rotation, receipts = rotation_fixture()
    req.req_pool_idx = None
    req.mamba_pool_idx = None
    req.mamba_ping_pong_track_buffer = None
    pool = SimpleNamespace(free=lambda indices: pytest.fail("unexpected anonymous free"))
    tree = SimpleNamespace(supports_mamba=lambda: True, req_to_token_pool=SimpleNamespace(mamba_pool=pool))
    release_kv_cache(req, tree, is_insert=False)
    release_kv_cache(req, tree, is_insert=False)
    assert len(receipts) == 2 and req._agentic_checkpoint_rotation is None


def test_mamba_pool_rotation_free_keeps_original_lease_and_guards_alloc():
    from sglang.srt.mem_cache.memory_pool import MambaPool
    owner = ledger()
    pool = MambaPool.__new__(MambaPool)
    pool.size, pool.free_slots = 16, torch.arange(1, 17)
    adapter = bridge(owner, lambda receipt: completed())
    try:
        pool.install_workset_adapter(adapter)
        assert pool.free_slots is None and pool.available_size() == 16
        with pytest.raises(RuntimeError):
            pool.alloc(1)
        with pytest.raises(RuntimeError):
            pool.clear()
        req, rotation, receipts = rotation_fixture()
        value = rotation.take()
        pool.free(value)
        assert not adapter._pending and not receipts
        rotation.close()
        assert len(receipts) == 2
    finally:
        adapter.shutdown()


def test_adapter_capacity_mismatch_rejects_before_replacing_native_free_lists():
    from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MambaPool
    adapter = bridge(ledger(), lambda receipt: completed())
    try:
        allocator = PagedTokenToKVPoolAllocator(32, 4, torch.int64, "cpu", None, False)
        with pytest.raises(RuntimeError, match="capacity mismatch"):
            allocator.install_workset_adapter(adapter)
        assert allocator.available_size() == 32
        assert allocator._agentic_workset_adapter is None
        pool = MambaPool.__new__(MambaPool)
        pool.size, pool.free_slots = 8, torch.arange(1, 9)
        with pytest.raises(RuntimeError, match="capacity mismatch"):
            pool.install_workset_adapter(adapter)
        assert pool.available_size() == 8
    finally:
        adapter.shutdown()
