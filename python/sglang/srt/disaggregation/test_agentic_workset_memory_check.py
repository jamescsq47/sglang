"""Controller conservation is address ownership, not native reference totals."""
from dataclasses import replace
from types import SimpleNamespace as NS
import threading

import pytest

from sglang.srt.disaggregation.agentic_workset_ledger import (
    LedgerCounts, LeaseKey, PageRun, WorksetPlan,
)
from sglang.srt.disaggregation.agentic_workset_tp import LocalWorksetView
from sglang.srt.managers.scheduler_runtime_checker_mixin import SchedulerRuntimeCheckerMixin


def scheduler(*, hybrid=True, closing=False, prepared=False):
    plan = WorksetPlan(LeaseKey("run", "s", "a", 1), 1, "fresh:s", 4,
                       4, 12, (PageRun(1, 1),), (PageRun(2, 2),),
                       (PageRun(1, 1),) if hybrid else (),
                       (PageRun(2, 2),) if hybrid else ())
    view = LocalWorksetView(plan, True, None, prepared, closing,
                            plan.parent_pages + plan.suffix_pages,
                            plan.checkpoint_slots + plan.runtime_slots)
    check = SchedulerRuntimeCheckerMixin()
    check.token_to_kv_pool_allocator = NS(size=40, page_size=4)
    check.req_to_token_pool = NS(mamba_pool=NS(size=8)) if hybrid else NS()
    state = NS(counts=LedgerCounts(1, 7, 5 if hybrid else 0, 1), views=(view,))
    check.agentic_p_workset_broker = NS(controller_mode=True, check_health=lambda: None,
        runtime=NS(conservation_snapshot=lambda: (state.counts, state.views)))
    # No native free list, Radix tensors or transport counter is available:
    # the check must not double-count donated/shared ranges or read GPU state.
    return check, state


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("closing,prepared", [(False, False), (False, True), (True, True)])
def test_prepared_donated_or_pending_return_remain_owned(hybrid, closing, prepared):
    check, _ = scheduler(hybrid=hybrid, closing=closing, prepared=prepared)
    method = check._check_mamba_memory if hybrid else check._check_radix_cache_memory
    leaked, message = method()
    assert not leaked, message
    check.self_check_during_busy()


def test_partial_and_whole_return_reduce_remaining_exactly_once():
    check, state = scheduler(closing=True, prepared=True)
    state.views = (replace(state.views[0], remaining_pages=(PageRun(3, 1),),
                           remaining_slots=(PageRun(3, 1),)),)
    state.counts = LedgerCounts(3, 9, 7, 1)
    assert not check._check_controller_memory()[0]
    state.views = ()
    state.counts = LedgerCounts(4, 10, 8, 0)
    assert not check._check_controller_memory()[0]


@pytest.mark.parametrize("corruption,expected", [
    ("missing_page", "attention free plus owned"),
    ("missing_slot", "mamba free plus owned"),
    ("live_count", "live lease count"),
    ("duplicate", "duplicate physical ownership"),
    ("foreign", "outside exact lease grant"),
    ("outside", "outside pool"),
    ("future", "outside installed allocation cut"),
])
def test_real_conservation_failures_not_suppressed(corruption, expected):
    check, state = scheduler()
    if corruption == "missing_page":
        state.counts = replace(state.counts, free_pages=6)
    elif corruption == "missing_slot":
        state.counts = replace(state.counts, free_mamba_slots=4)
    elif corruption == "live_count":
        state.counts = replace(state.counts, live_leases=0)
    elif corruption == "duplicate":
        # Equal scalar total is insufficient: two owners claim the same page.
        first = replace(state.views[0], remaining_pages=(PageRun(1, 1),), remaining_slots=())
        other = replace(first, plan=replace(first.plan, key=replace(first.plan.key, snapshot_id="other")))
        state.views = (first, other)
        state.counts = LedgerCounts(1, 8, 8, 2)
    elif corruption in {"foreign", "outside"}:
        state.views = (replace(state.views[0], remaining_pages=(PageRun(8 if corruption == "foreign" else 10, 3),)),)
    else:
        state.counts = replace(state.counts, sequence=0)
    leaked, message = check._check_controller_memory()
    assert leaked and expected in message
    with pytest.raises(AssertionError, match="Mem Leak Detected"):
        check.self_check_during_busy()


def test_legacy_accounting_unchanged():
    check = SchedulerRuntimeCheckerMixin()
    check._get_token_info = lambda: (0, 0, 4, 2)
    check.tree_cache = NS(protected_size=lambda: 1)
    check._session_held_tokens = lambda: 1
    check._agentic_reserved_tokens = lambda: 2
    check.max_total_num_tokens = 10
    assert check._check_controller_memory() is None
    assert not check._check_radix_cache_memory()[0]
    check.max_total_num_tokens = 11
    assert check._check_radix_cache_memory()[0]


def test_runtime_failure_not_misreported_as_empty_or_freed():
    check, _ = scheduler()
    def broken():
        raise RuntimeError("ownership retained")
    check.agentic_p_workset_broker.runtime.conservation_snapshot = broken
    with pytest.raises(RuntimeError, match="ownership retained"):
        check._check_controller_memory()


@pytest.mark.parametrize("corrupt", [False, True])
def test_idle_prepared_lease_checks_ownership_without_req_pool_false_alarm(monkeypatch, corrupt):
    from sglang.srt.disaggregation.utils import DisaggregationMode
    import sglang.srt.managers.scheduler_runtime_checker_mixin as module
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", "true")
    check, state = scheduler(prepared=True)
    check.enable_hisparse = False
    check.is_hybrid_ssm = True
    check.disaggregation_mode = DisaggregationMode.PREFILL
    check.agentic_p_workset_broker._lock = threading.RLock()
    check.agentic_p_workset_broker._leases = {"s": object()}
    warnings = []
    monkeypatch.setattr(module, "raise_error_or_warn", lambda *args: warnings.append(args[-1]))
    if corrupt:
        state.counts = replace(state.counts, free_pages=6)
    check.self_check_during_idle()
    assert bool(warnings) is corrupt
