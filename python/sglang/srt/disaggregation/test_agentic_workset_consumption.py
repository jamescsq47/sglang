"""Native batch consumption of prepared controller descriptors, CPU only."""
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_workset_device import prepare_workset
from sglang.srt.disaggregation.agentic_workset_ledger import WorksetLedger
from sglang.srt.mem_cache import common


def setup_batch(monkeypatch, tokens=9):
    authority = WorksetLedger(incarnation="group", page_count=16, page_size=4)
    plan = authority.grant("r", "fresh", owner="fresh", parent_tokens=0, prompt_tokens=tokens)
    broker = AgenticPWorksetLeaseBroker(4)
    lease = broker.install_prepared(prepare_workset(plan, device="cpu", page_capacity=16))
    req = NS(rid="r", origin_input_ids=list(range(tokens)), req_pool_idx=None,
             prefix_indices=torch.empty(0, dtype=torch.int64), mamba_pool_idx=None,
             mamba_ping_pong_track_buffer=None)
    broker.handoff_fresh_to_req("r", req, lease)
    effects = []
    req_pool = NS(alloc=lambda reqs: (effects.append("req-slot"), list(range(len(reqs))))[1])
    allocator = NS(_agentic_workset_adapter=object(), page_size=4)
    batch = NS(reqs=[req], req_to_token_pool=req_pool, device="cpu",
               tree_cache=NS(token_to_kv_pool_allocator=allocator, page_size=4),
               maybe_evict_swa=lambda: effects.append("swa"))
    monkeypatch.setattr(common, "write_cache_indices", lambda *args: effects.append("mapping"))
    monkeypatch.setattr(common, "alloc_paged_token_slots_extend", lambda **kwargs: pytest.fail("native fallback"))
    set_chunk(batch, 0, 4)
    return authority, broker, lease, req, batch, effects


def set_chunk(batch, prefix, extend):
    batch.prefix_lens, batch.extend_lens = [prefix], [extend]
    batch.seq_lens_cpu = batch.seq_lens = torch.tensor([prefix + extend])
    batch.extend_num_tokens = extend


def test_complete_fresh_workset_consumes_three_native_chunks_without_alloc(monkeypatch):
    authority, broker, lease, req, batch, effects = setup_batch(monkeypatch)
    before = authority.counts
    chunks = []
    for prefix, size in ((0, 4), (4, 4), (8, 1)):
        set_chunk(batch, prefix, size)
        chunks.append(common.alloc_for_extend(batch)[0])
        req.prefix_indices = torch.cat(chunks)
    assert torch.cat(chunks).tolist() == list(range(4, 13))
    assert lease.state == "consumed" and broker.get("r") is None
    assert authority.counts == before  # Req donation is not a global free.
    assert effects == ["swa", "req-slot", "mapping"] * 3


def test_fresh_shared_prefix_consumes_private_suffix_and_returns_exact_redundant_scope(monkeypatch):
    authority, broker, lease, req, batch, effects = setup_batch(monkeypatch)
    req.prefix_indices = torch.tensor([80, 81, 82, 83])
    returns = []
    broker.adopt_fresh_prefix(req, lease, prefix_tokens=4, pinned=True,
                             return_prefix=lambda plan, tokens: returns.append((plan, tokens)))
    set_chunk(batch, 4, 5)
    result = common.alloc_for_extend(batch)[0]
    assert result.tolist() == list(range(8, 13))
    assert returns == [(lease.controller_plan, 4)]
    assert req.prefix_indices.tolist() == [80, 81, 82, 83]
    assert authority.counts.free_pages == 13  # async scope ACK not yet granted


@pytest.mark.parametrize("fault", ["missing", "wrong-owner", "wrong-view", "short", "prefix", "unaligned", "mixed"])
def test_invalid_batch_is_rejected_before_reqslot_or_any_native_side_effect(monkeypatch, fault):
    authority, broker, lease, req, batch, effects = setup_batch(monkeypatch)
    if fault == "missing":
        req._agentic_workset_backed = False
    elif fault == "wrong-owner":
        req._agentic_p_workset_lease = NS(controller_plan=lease.controller_plan, state="handed", snapshot_id="other")
    elif fault == "wrong-view":
        req._agentic_workset_suffix_indices = req._agentic_workset_suffix_indices.clone()
    elif fault == "short":
        set_chunk(batch, 0, 20)
    elif fault == "prefix":
        set_chunk(batch, 4, 4)
    elif fault == "unaligned":
        set_chunk(batch, 0, 3)
    else:
        batch.reqs.append(NS(rid="unreserved"))
        batch.prefix_lens.append(0)
        batch.extend_lens.append(4)
    before = authority.counts
    with pytest.raises(RuntimeError):
        common.alloc_for_extend(batch)
    assert not effects and lease.suffix_cursor == 0 and authority.counts == before


def test_legacy_preflight_is_noop():
    common.preflight_controller_worksets(NS(tree_cache=NS(token_to_kv_pool_allocator=object())))


def test_mamba_preflight_installs_exact_rotation_and_never_allocates(monkeypatch):
    from sglang.srt.disaggregation.test_agentic_workset_device import pool
    from sglang.srt.disaggregation.agentic_workset_native import install_checkpoint_rotation
    from concurrent.futures import Future
    mamba = pool()
    req_pool = NS(mamba_pool=mamba, enable_mamba_extra_buffer=True, mamba_ping_pong_track_buffer_size=2)
    authority = WorksetLedger(incarnation="group", page_count=16, page_size=4, mamba_slots=8)
    plan = authority.grant("r", "fresh", owner="fresh", parent_tokens=0, prompt_tokens=9, runtime_slots=5)
    broker = AgenticPWorksetLeaseBroker(4, state_allocators=(mamba,), mamba_req_to_token_pool=req_pool)
    lease = broker.install_prepared(prepare_workset(plan, device="cpu", page_capacity=16, mamba_pool=mamba))
    req = NS(rid="r", origin_input_ids=list(range(9)), req_pool_idx=None,
             prefix_indices=torch.empty(0, dtype=torch.int64), mamba_pool_idx=None,
             mamba_ping_pong_track_buffer=None)
    prepared_runtime = lease.runtime_state_device_indices[0]
    monkeypatch.setattr(torch, "full", lambda *a, **k: pytest.fail("native descriptor materialization"))
    broker.handoff_fresh_to_req("r", req, lease)
    assert req.mamba_ping_pong_track_buffer.data_ptr() == prepared_runtime[1:3].data_ptr()
    calls = []
    def prepare(req, lease):
        if getattr(req, "_agentic_checkpoint_rotation", None) is None:
            calls.append(lease.controller_plan.key)
            install_checkpoint_rotation(req, plan.key, (4, 5), req._agentic_mamba_prefill_checkpoint, lambda receipt: Future())
    broker.prepare_req_checkpoints = prepare
    batch = NS(reqs=[req], prefix_lens=[0], extend_lens=[4], req_to_token_pool=req_pool,
               tree_cache=NS(token_to_kv_pool_allocator=NS(_agentic_workset_adapter=object(), page_size=4)))
    monkeypatch.setattr(mamba, "alloc", lambda _: pytest.fail("native state allocation"))
    common.preflight_controller_worksets(batch)
    common.preflight_controller_worksets(batch)
    assert calls == [plan.key]
    assert req._agentic_checkpoint_rotation.key == plan.key
    req.mamba_ping_pong_track_buffer = None
    with pytest.raises(RuntimeError, match="runtime"):
        common.preflight_controller_worksets(batch)
