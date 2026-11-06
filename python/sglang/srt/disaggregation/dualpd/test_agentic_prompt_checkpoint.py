"""Stable-prefix mode leaves serving/harness tokens unchanged and fails closed."""
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_hybrid_transfer import StateType
from sglang.srt.disaggregation.agentic_hybrid_transfer import (
    p2d_mamba_checkpoint_tokens, p2d_mamba_source_indices,
    p2d_mamba_destination_indices, freeze_p2d_mamba_checkpoint_after_cache,
    snapshot_token_count_for_req, state_indices_for_req,
    validate_agentic_mamba_tracking,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
import sglang.srt.managers.schedule_batch as sb
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin as SchedulerBatchResultProcessor,
)


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", "true")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT", "true")


def request(n=390, thinking=True):
    ids = [5] * n
    if thinking:
        ids[-2:] = [91, 92]
    return NS(origin_input_ids=ids, tokenizer=NS(encode=lambda *a, **k: [91, 92]))


@pytest.mark.parametrize("n,expected", [(390,384),(384,320),(385,320),(383,320),(64,0),(65,0)])
def test_stable_boundary_before_removed_opener(n, expected):
    req = request(n)
    before = list(req.origin_input_ids)
    assert p2d_mamba_checkpoint_tokens(req,64) == expected
    assert req.origin_input_ids == before


def test_opt_in_only_and_non_thinking(monkeypatch):
    assert p2d_mamba_checkpoint_tokens(request(384,False),64) == 384
    monkeypatch.setenv("SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT", "false")
    assert p2d_mamba_checkpoint_tokens(request(384),64) == 384


@pytest.mark.parametrize("n,start,extend,expected,mask", [
    (390,0,390,384,True), (384,0,384,320,True),
    (8193,0,8192,8128,True), (8193,8192,1,None,False),
    (384,320,64,None,False), (65,0,65,None,False),
    (17000,0,8192,8192,True),
])
def test_real_prefill_tracker_targets_stable_or_current_chunk(monkeypatch,n,start,extend,expected,mask):
    monkeypatch.setattr(sb,"get_global_server_args",lambda: NS(
        mamba_cache_chunk_size=64,disaggregation_mode="prefill",
        enable_mamba_extra_buffer_lazy=lambda: False))
    req = request(n)
    req.extend_input_len = extend
    req.prefix_indices = [0] * start
    req.mamba_ping_pong_track_buffer = torch.tensor([7,8])
    req.mamba_next_track_idx = 0
    req.mamba_branching_seqlen = None
    req.mamba_last_track_seqlen = None
    batch = ScheduleBatch(reqs=[req],req_to_token_pool=NS(
        get_mamba_ping_pong_other_idx=lambda x: 1-x),
        token_to_kv_pool_allocator=NS(page_size=64))
    masks, indices, seqlens = [], [], []
    batch._mamba_radix_cache_v2_req_prepare_for_extend(req, masks, indices, seqlens)
    assert masks == [mask]
    assert req.mamba_last_track_seqlen == expected
    if mask and expected < start+extend:
        assert seqlens == [expected+1]
    if not mask:
        assert req.mamba_next_track_idx == 0


def test_p2d_frozen_boundary_and_long_decode_does_not_rotate():
    p=request(384)
    p.mamba_pool_idx=torch.tensor(11)
    p.last_node=NS(mamba_value=torch.tensor([13]))
    p.cache_protected_len=320
    freeze_p2d_mamba_checkpoint_after_cache(p,320,64)
    assert p2d_mamba_source_indices(p,64)[0].tolist() == [11,13]
    d=request(384)
    d.mamba_pool_idx=torch.tensor(21)
    d.mamba_ping_pong_track_buffer=torch.tensor([22,23])
    d.mamba_next_track_idx=0
    dest=p2d_mamba_destination_indices(d,NS(get_mamba_ping_pong_other_idx=lambda x:1-x),64)
    assert dest[0].tolist() == [21,23]
    for _ in range(1000):
        SchedulerBatchResultProcessor._mamba_prefix_cache_update(None,d,None,None,0)
    assert d.mamba_last_track_seqlen == 320
    assert d.mamba_next_track_idx == 0
    assert snapshot_token_count_for_req(d,10000,[StateType.MAMBA],64)==320
    assert int(state_indices_for_req(d,[StateType.MAMBA],checkpoint_tokens=320,page_size=64)[0][0])==23
    d._agentic_mamba_frozen_prompt_valid=False
    with pytest.raises(RuntimeError,match="retraction"):
        snapshot_token_count_for_req(d,10000,[StateType.MAMBA],64)


def test_wrong_locked_boundary_and_overwritten_state_fail_closed():
    p=request(384);p.cache_protected_len=384;p.last_node=NS(mamba_value=torch.tensor([13]))
    with pytest.raises(RuntimeError,match="locked prefix"):
        freeze_p2d_mamba_checkpoint_after_cache(p,320,64)
    d=NS(_agentic_mamba_frozen_prompt_tokens=320,_agentic_mamba_frozen_prompt_valid=True,
        mamba_last_track_seqlen=384)
    with pytest.raises(RuntimeError,match="overwritten"):
        snapshot_token_count_for_req(d,1000,[StateType.MAMBA],64)


def test_retraction_marks_checkpoint_invalid():
    req=NS(retraction_count=0,_agentic_mamba_frozen_prompt_tokens=320,
        _agentic_mamba_frozen_prompt_valid=True,input_embeds=None)
    Req.reset_for_retract(req)
    assert not req._agentic_mamba_frozen_prompt_valid
    assert req._agentic_mamba_frozen_prompt_tokens==320


@pytest.mark.parametrize("spec,lazy", [("EAGLE",False),(None,True)])
def test_unsupported_tracking_variants_rejected(spec,lazy):
    with pytest.raises(ValueError):
        validate_agentic_mamba_tracking(NS(page_size=64,state_types=[StateType.MAMBA]),NS(
            mamba_track_interval=64,enable_int8_mamba_checkpoint=False,
            speculative_algorithm=spec,enable_mamba_extra_buffer_lazy=lambda:lazy))
