from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.agentic_workset_device import (
    WorksetPreparationFailure, prepare_workset,
)
from sglang.srt.disaggregation.agentic_workset_ledger import PageRun, WorksetLedger
from sglang.srt.mem_cache.memory_pool import MambaPool


def plan(**kwargs):
    ledger = WorksetLedger(incarnation="run", page_count=32, page_size=4, mamba_slots=8)
    return ledger, ledger.grant("g", "a", owner="p", parent_tokens=5,
                                prompt_tokens=10, **kwargs)


def pool():
    result = MambaPool.__new__(MambaPool)
    result.size, result.device = 8, "cpu"
    result.free_slots = torch.arange(1, 9)
    result.mamba_cache = SimpleNamespace(
        conv=[torch.ones(2, 9, 3)], temporal=torch.ones(2, 9, 4))
    return result


def test_exact_cpu_descriptors_no_allocator_mutation():
    ledger, grant = plan(checkpoint_slots=1, runtime_slots=3)
    mamba = pool()
    before = ledger.counts
    view = prepare_workset(grant, device="cpu", page_capacity=32, mamba_pool=mamba)
    assert ledger.counts == before
    assert mamba.free_slots.tolist() == list(range(1, 9))
    assert view.device_indices.tolist() == list(range(4, 20))
    assert view.parent_page_indices.tolist() == [1, 2]
    assert view.checkpoint_indices.tolist() == [1]
    assert view.runtime_indices.tolist() == [2, 3, 4]
    assert not view.parent_page_indices.flags.writeable
    assert view.is_ready()
    assert torch.all(mamba.mamba_cache.temporal[:, 1:5] == 0)
    assert torch.all(mamba.mamba_cache.temporal[:, 5:] == 1)


def test_native_mamba_alloc_keeps_original_zeroing_and_free_semantics():
    mamba = pool()
    selected = mamba.alloc(3)
    assert selected.tolist() == [1, 2, 3]
    assert mamba.available_size() == 5
    assert torch.all(mamba.mamba_cache.conv[0][:, 1:4] == 0)
    assert torch.all(mamba.mamba_cache.temporal[:, 1:4] == 0)
    assert torch.all(mamba.mamba_cache.temporal[:, 4:] == 1)
    mamba.free(selected)
    assert mamba.available_size() == 8


def test_split_pages_and_fresh_zero_parent():
    _, grant = plan()
    grant = replace(grant, parent_tokens=0, prompt_tokens=9, parent_pages=(),
                    suffix_pages=(PageRun(3, 1), PageRun(8, 2)))
    view = prepare_workset(grant, device="cpu", page_capacity=32)
    assert view.parent_page_indices.size == 0
    assert view.device_indices.tolist() == list(range(12, 16)) + list(range(32, 40))


@pytest.mark.parametrize("change", [
    {"suffix_pages": (PageRun(1, 2),)},  # overlaps parent
    {"suffix_pages": (PageRun(32, 2),)},  # out of bounds
    {"suffix_pages": (PageRun(3, 1),)},  # incomplete workset
    {"parent_tokens": -1},
])
def test_invalid_plan_rejected_before_initialization(change):
    _, grant = plan()
    with pytest.raises(ValueError):
        prepare_workset(replace(grant, **change), device="cpu", page_capacity=32)


def test_missing_mamba_pool_or_cuda_stream_rejected():
    _, grant = plan(checkpoint_slots=1)
    with pytest.raises(ValueError, match="state pool"):
        prepare_workset(grant, device="cpu", page_capacity=32)
    _, grant = plan()
    with pytest.raises(ValueError, match="explicit preparation stream"):
        prepare_workset(grant, device="cuda:0", page_capacity=32)


def test_prepare_failure_never_frees_or_publishes_grant():
    ledger, grant = plan(checkpoint_slots=1)
    mamba = pool()
    before = ledger.counts
    def fail(_):
        raise RuntimeError("injected preparation failure")
    mamba.initialize_slots = fail
    with pytest.raises(WorksetPreparationFailure, match="injected") as failure:
        prepare_workset(grant, device="cpu", page_capacity=32, mamba_pool=mamba)
    assert failure.value.is_quiescent()  # CPU test, not permission to free.
    assert ledger.counts == before
    assert ledger.view(grant.key) is not None


def test_preparation_error_without_a_real_fence_stays_quarantined():
    _, grant = plan()
    error = WorksetPreparationFailure(grant, RuntimeError("CUDA failed"))
    assert not error.is_quiescent()
    fence = SimpleNamespace(query=lambda: False)
    error = WorksetPreparationFailure(grant, RuntimeError("after enqueue"), ready_event=fence)
    assert not error.is_quiescent()
    fence.query = lambda: True
    assert error.is_quiescent()
    def unavailable():
        raise RuntimeError("context failed")
    fence.query = unavailable
    assert not error.is_quiescent()
