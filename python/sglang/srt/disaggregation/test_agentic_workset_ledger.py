"""CPU-only ownership core tests; no runtime allocator or GPU integration."""

from dataclasses import FrozenInstanceError, replace
import json
import random
import threading

import pytest

from sglang.srt.disaggregation.agentic_workset_ledger import (
    LeaseKey, PageRun, WorksetLedger,
)


def ledger(**kwargs):
    return WorksetLedger(incarnation="run-1/p-group", page_count=16, page_size=4,
                         mamba_slots=8, **kwargs)


def grant(controller, sid="request:1", attempt="direct-room:12", **kwargs):
    args = dict(owner="direct", parent_tokens=8, prompt_tokens=12,
                checkpoint_slots=1, runtime_slots=2)
    args.update(kwargs)
    return controller.grant(sid, attempt, **args)


def fence_all(owner, key):
    for rank in range(owner.tp_size):
        owner.report_fence(key, rank, quiet=True, unreferenced=True)


def addresses(runs):
    return {index for run in runs for index in range(run.start, run.end)}


def test_complete_workset_exact_compressed_immutable_plan():
    owner = ledger()
    plan = grant(owner, parent_tokens=5, prompt_tokens=11)
    assert plan.parent_pages == (PageRun(1, 2),)
    assert plan.suffix_pages == (PageRun(3, 2),)
    assert plan.checkpoint_slots == (PageRun(1, 1),)
    assert plan.runtime_slots == (PageRun(2, 2),)
    assert plan.allocated_tokens == 16
    assert owner.counts.free_pages == 12 and owner.counts.free_mamba_slots == 5
    assert plan.sequence == plan.key.version == 1
    with pytest.raises(FrozenInstanceError):
        plan.owner = "other"
    wire = json.loads(json.dumps(plan.to_wire()))
    assert wire["parent_pages"] == [[1, 2]]
    wire["parent_pages"][0][0] = 99
    assert plan.parent_pages[0].start == 1


@pytest.mark.parametrize("resource", ["attention", "checkpoint", "runtime"])
def test_composite_capacity_failure_mutates_no_pool_or_version(resource):
    owner = ledger()
    before = owner.counts
    args = {"attention": dict(prompt_tokens=100),
            "checkpoint": dict(checkpoint_slots=9),
            "runtime": dict(runtime_slots=9)}[resource]
    assert grant(owner, **args) is None
    assert owner.counts == before
    plan = grant(owner)
    assert plan.sequence == plan.key.version == 1
    assert plan.parent_pages[0].start == plan.checkpoint_slots[0].start == 1


def test_same_grant_is_idempotent_changed_or_competing_owner_rejected():
    owner = ledger()
    first = grant(owner)
    before = owner.counts
    assert grant(owner) is first
    for args in (dict(owner="slow"), dict(prompt_tokens=13), dict(runtime_slots=1),
                 dict(attempt="replacement")):
        with pytest.raises(ValueError, match="different or closing"):
            grant(owner, **args)
        assert owner.counts == before


@pytest.mark.parametrize("size", [1, 2, 8])
def test_close_requires_all_rank_quiet_and_reference_fence(size):
    owner = ledger(tp_size=size)
    plan = grant(owner)
    assert owner.may_start(plan.key)
    assert not owner.report_fence(plan.key, 0, quiet=True, unreferenced=True)
    assert owner.free(plan.key) is None  # Pre-close completion cannot authorize reuse.
    close = owner.cancel(plan.key)
    retained = owner.counts
    assert close.operation == "cancel" and close.sequence == 2
    assert owner.cancel(plan.key) is close and owner.counts == retained
    assert not owner.may_start(plan.key)
    for rank in range(size):
        assert not owner.report_fence(plan.key, rank, quiet=True, unreferenced=False)
    assert owner.free(plan.key) is None
    for rank in range(size - 1):
        assert not owner.report_fence(plan.key, rank, quiet=False, unreferenced=True)
    assert owner.free(plan.key) is None
    assert owner.report_fence(plan.key, size - 1, quiet=False, unreferenced=True)
    assert owner.counts == retained  # A report is not a free decision.
    freed = owner.free(plan.key)
    assert freed.operation == "free" and freed.sequence == 3
    assert owner.free(plan.key) is freed
    assert owner.counts.free_pages == 16 and owner.counts.free_mamba_slots == 8
    assert owner.counts.live_leases == 0 and owner.counts.sequence == 3


@pytest.mark.parametrize("fact", ["quiet", "unreferenced"])
def test_unknown_or_missing_fence_never_expires_or_reclaims(fact):
    owner = ledger(tp_size=8)
    plan = grant(owner)
    owner.cancel(plan.key)
    retained = owner.counts
    for rank in range(7):
        owner.report_fence(plan.key, rank, quiet=True, unreferenced=True)
    flags = dict(quiet=True, unreferenced=True)
    flags[fact] = False
    owner.report_fence(plan.key, 7, **flags)
    for _ in range(20):
        assert owner.free(plan.key) is None
    assert owner.counts == retained


def test_old_version_attempt_and_incarnation_do_not_touch_successor():
    owner = ledger(tp_size=2)
    old = grant(owner)
    owner.cancel(old.key)
    fence_all(owner, old.key)
    old_free = owner.free(old.key)
    new = grant(owner, attempt="slow-recovery:2", owner="slow")
    assert new.key.version == 2 and new.sequence == 4
    before = owner.counts
    assert owner.free(old.key) is old_free  # ACK retry, not another free.
    keys = (old.key, replace(new.key, version=1), replace(new.key, attempt_id="old"),
            replace(new.key, incarnation="old-run"))
    for key in keys:
        assert owner.cancel(key) is None
        assert not owner.report_fence(key, 0, quiet=True, unreferenced=True)
        assert owner.may_start(new.key)
        assert owner.counts == before
    owner.cancel(new.key)
    fence_all(owner, new.key)
    owner.free(new.key)
    with pytest.raises(ValueError, match="retired attempt"):
        grant(owner)


def test_closed_owner_cannot_reopen_or_replace_before_real_free():
    owner = ledger()
    plan = grant(owner)
    owner.cancel(plan.key)
    before = owner.counts
    for attempt in (plan.key.attempt_id, "slow-successor"):
        with pytest.raises(ValueError, match="closing"):
            grant(owner, attempt=attempt)
    assert owner.counts == before and not owner.may_start(plan.key)


def test_fragmented_full_workset_reassembles_exact_pages_without_overlap():
    owner = ledger()
    plans = [grant(owner, sid=f"request-{i}:1", parent_tokens=4, prompt_tokens=8,
                   checkpoint_slots=0, runtime_slots=0) for i in range(8)]
    assert owner.counts.free_pages == 0
    for i in (0, 2, 4):
        owner.cancel(plans[i].key)
        fence_all(owner, plans[i].key)
        owner.free(plans[i].key)
    merged = grant(owner, sid="fragmented:1", parent_tokens=12, prompt_tokens=24,
                   checkpoint_slots=0, runtime_slots=0)
    assert merged.parent_pages == (PageRun(1, 2), PageRun(5, 1))
    assert merged.suffix_pages == (PageRun(6, 1), PageRun(9, 2))
    new_pages = addresses(merged.parent_pages + merged.suffix_pages)
    for i in (1, 3, 5, 6, 7):
        assert new_pages.isdisjoint(addresses(plans[i].parent_pages + plans[i].suffix_pages))


def test_native_parent_zero_suffix_and_attention_only_model():
    owner = WorksetLedger(incarnation="plain", page_count=4, page_size=64)
    plan = owner.grant("fresh:1", "native:1", owner="native", parent_tokens=0, prompt_tokens=193)
    assert plan.parent_pages == plan.checkpoint_slots == plan.runtime_slots == ()
    assert plan.suffix_pages == (PageRun(1, 4),)
    assert plan.allocated_tokens == 256
    assert owner.counts.free_pages == 0


def test_large_pool_has_run_metadata_not_per_page_or_token_objects():
    owner = WorksetLedger(incarnation="large", page_count=10**9, page_size=64)
    plan = owner.grant("large:1", "native:1", owner="native", parent_tokens=64 * 10**8,
                       prompt_tokens=64 * 5 * 10**8)
    assert len(owner._pages) == len(plan.parent_pages) == len(plan.suffix_pages) == 1
    assert len(json.dumps(plan.to_wire())) < 600


@pytest.mark.parametrize("kwargs", [dict(rank=1), dict(rank=True), dict(page_size=0),
                                    dict(page_count=-1), dict(tp_size=0), dict(mamba_slots=True)])
def test_invalid_authority_or_capacity_rejected(kwargs):
    args = dict(incarnation="run", page_count=4, page_size=4)
    args.update(kwargs)
    with pytest.raises(ValueError):
        WorksetLedger(**args)


@pytest.mark.parametrize("kwargs", [dict(parent_tokens=-1), dict(prompt_tokens=0),
                                    dict(parent_tokens=16, prompt_tokens=12),
                                    dict(runtime_slots=True), dict(owner="")])
def test_invalid_request_has_no_ownership_effect(kwargs):
    owner = ledger()
    before = owner.counts
    with pytest.raises(ValueError):
        grant(owner, **kwargs)
    assert owner.counts == before


def test_only_controller_thread_may_mutate_but_views_are_readable():
    owner = ledger()
    plan = grant(owner)
    errors, views = [], []
    def other_thread():
        views.append(owner.view(plan.key))
        try:
            owner.cancel(plan.key)
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    assert views[0].plan is plan and not views[0].closing
    assert errors == ["ownership ledger has a different controller writer"]
    assert owner.may_start(plan.key)


def test_randomized_conservation_and_sequence_with_retry_and_fragmentation():
    rng, owner, live = random.Random(4837), ledger(tp_size=2), {}
    last_sequence = 0
    for step in range(500):
        if live and rng.random() < .45:
            sid = rng.choice(tuple(live))
            plan = live.pop(sid)
            owner.cancel(plan.key)
            assert owner.free(plan.key) is None
            fence_all(owner, plan.key)
            decision = owner.free(plan.key)
            assert owner.free(plan.key) == decision
        else:
            sid = f"random-{step}:1"
            plan = grant(owner, sid=sid, parent_tokens=4 * rng.randrange(4),
                         prompt_tokens=4 * rng.randrange(4, 8),
                         checkpoint_slots=1, runtime_slots=rng.randrange(3))
            if plan is not None:
                live[sid] = plan
        pages, slots = set(), set()
        for plan in live.values():
            p = addresses(plan.parent_pages + plan.suffix_pages)
            s = addresses(plan.checkpoint_slots + plan.runtime_slots)
            assert pages.isdisjoint(p) and slots.isdisjoint(s)
            pages |= p
            slots |= s
        assert len(pages) + owner.counts.free_pages == 16
        assert len(slots) + owner.counts.free_mamba_slots == 8
        assert owner.counts.live_leases == len(live)
        assert owner.counts.sequence >= last_sequence
        last_sequence = owner.counts.sequence


@pytest.mark.parametrize("size", [1, 2, 8])
def test_partial_radix_return_needs_same_subset_all_rank_fences(size):
    owner = ledger(tp_size=size)
    plan = grant(owner)
    subset = owner.begin_return(plan.key, "radix-duplicate", pages=(PageRun(1, 1),),
                                slots=plan.checkpoint_slots)
    assert subset.sequence == 2
    assert owner.begin_return(plan.key, "radix-duplicate", pages=(PageRun(1, 1),),
                              slots=plan.checkpoint_slots) is subset
    assert not owner.may_start(plan.key)
    assert not owner.may_use(plan.key, pages=subset.pages)
    assert owner.may_use(plan.key, pages=plan.suffix_pages, slots=plan.runtime_slots)
    before = owner.counts
    wrong = replace(subset, pages=plan.suffix_pages)
    for rank in range(size):
        assert not owner.report_return_fence(wrong, rank, quiet=True, unreferenced=True)
        assert not owner.report_return_fence(subset, rank, quiet=True, unreferenced=False)
    assert owner.commit_return(subset) is None
    for rank in range(size):
        owner.report_return_fence(subset, rank, quiet=False, unreferenced=True)
    assert owner.counts == before
    returned = owner.commit_return(subset)
    assert returned.operation == "return" and returned.sequence == 3
    assert owner.commit_return(subset) is returned
    assert owner.counts.free_pages == 14 and owner.counts.free_mamba_slots == 6
    assert owner.view(plan.key).remaining_pages == (PageRun(2, 2),)
    assert not owner.may_use(plan.key, pages=subset.pages)
    assert owner.may_use(plan.key, pages=plan.suffix_pages)


def test_whole_free_after_partial_return_does_not_reclaim_other_lease():
    owner = ledger()
    old = grant(owner)
    part = owner.begin_return(old.key, "parent-donated-last-reference", pages=old.parent_pages,
                              slots=old.checkpoint_slots)
    owner.report_return_fence(part, 0, quiet=True, unreferenced=True)
    owner.commit_return(part)
    successor = grant(owner, sid="unrelated:1", parent_tokens=4, prompt_tokens=8,
                      checkpoint_slots=1, runtime_slots=0)
    assert successor.parent_pages == (PageRun(1, 1),)
    assert successor.checkpoint_slots == (PageRun(1, 1),)
    owner.cancel(old.key)
    fence_all(owner, old.key)
    owner.free(old.key)
    assert owner.counts.free_pages == 14 and owner.counts.free_mamba_slots == 7
    assert owner.may_start(successor.key)
    before = owner.counts
    assert owner.commit_return(part) is None  # Retired owner, never touch successor.
    assert not owner.report_return_fence(part, 0, quiet=True, unreferenced=True)
    assert owner.counts == before


def test_partial_return_rejects_cross_owner_overlap_and_changed_retry_atomically():
    owner = ledger()
    first, other = grant(owner), grant(owner, sid="other:1")
    before = owner.counts
    for pages, slots in ((other.parent_pages, ()), ((), other.runtime_slots),
                         ((PageRun(1, 1), PageRun(1, 1)), ()),
                         ((PageRun(0, 1),), ()), ((), ())):
        with pytest.raises(ValueError):
            owner.begin_return(first.key, "bad", pages=pages, slots=slots)
        assert owner.counts == before
    partial = owner.begin_return(first.key, "one", pages=(PageRun(1, 1),))
    before = owner.counts
    for name, pages in (("overlap", (PageRun(1, 2),)), ("one", (PageRun(2, 1),))):
        with pytest.raises(ValueError):
            owner.begin_return(first.key, name, pages=pages)
        assert owner.counts == before
    assert owner.begin_return(replace(first.key, version=99), "old", pages=partial.pages) is None
    assert owner.counts == before


def test_disjoint_parent_suffix_runtime_returns_preserve_shared_reference_lifetime():
    owner = ledger(tp_size=2)
    plan = grant(owner)
    parent = owner.begin_return(plan.key, "radix", pages=plan.parent_pages,
                                slots=plan.checkpoint_slots)
    request = owner.begin_return(plan.key, "req", pages=plan.suffix_pages,
                                 slots=plan.runtime_slots)
    # Parent remains shared; finished request's private suffix/runtime can be
    # returned promptly without waiting for the unrelated shared reference.
    for rank in range(2):
        owner.report_return_fence(parent, rank, quiet=True, unreferenced=False)
        owner.report_return_fence(request, rank, quiet=True, unreferenced=True)
    assert owner.commit_return(parent) is None
    assert owner.commit_return(request) is not None
    assert owner.counts.free_pages == 14 and owner.counts.free_mamba_slots == 7
    assert owner.view(plan.key).remaining_pages == plan.parent_pages
    for rank in range(2):
        owner.report_return_fence(parent, rank, quiet=False, unreferenced=True)
    owner.commit_return(parent)
    assert owner.counts.free_pages == 16 and owner.counts.free_mamba_slots == 8
    # Zero remaining addresses does not manufacture a whole-lease fence.
    assert owner.free(plan.key) is None
    owner.cancel(plan.key)
    fence_all(owner, plan.key)
    assert owner.free(plan.key) is not None
    assert owner.counts.free_pages == 16 and owner.counts.free_mamba_slots == 8


def test_whole_fence_can_retire_pending_subset_without_double_release():
    owner = ledger()
    plan = grant(owner)
    pending = owner.begin_return(plan.key, "pending", pages=plan.parent_pages)
    owner.cancel(plan.key)
    fence_all(owner, plan.key)
    assert owner.free(plan.key) is not None
    assert owner.commit_return(pending) is None
    assert owner.counts.free_pages == 16


def test_partial_report_wrong_sequence_version_or_incarnation_is_not_ack():
    owner = ledger()
    plan = grant(owner)
    returning = owner.begin_return(plan.key, "subset", pages=plan.parent_pages)
    for stale in (replace(returning, sequence=returning.sequence + 1),
                  replace(returning, key=replace(plan.key, version=99)),
                  replace(returning, key=replace(plan.key, incarnation="old"))):
        assert not owner.report_return_fence(stale, 0, quiet=True, unreferenced=True)
        assert owner.commit_return(stale) is None
    assert owner.commit_return(returning) is None
