"""Request-owned state: shared refs, replacement, backup and native gating."""
from functools import partial
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_hybrid_transfer import (
    request_owned_mamba_enabled, frozen_mamba_checkpoint,
    p2d_mamba_destination_indices, snapshot_token_count_for_req, StateType,
    offload_request_mamba, restore_request_mamba,
    freeze_p2d_mamba_checkpoint_after_cache,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, _key_match_paged, get_child_key
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin


@pytest.fixture
def enabled(monkeypatch):
    for name in ('LIFECYCLE','MAMBA_PROMPT_CHECKPOINT','CUSTOM_STORAGE_ONLY','MAMBA_REQUEST_OWNED'):
        monkeypatch.setenv('SGLANG_AGENTIC_KV_'+name,'true')


def test_opt_in_requires_all_flags(enabled, monkeypatch):
    assert request_owned_mamba_enabled()
    for name in ('LIFECYCLE','MAMBA_PROMPT_CHECKPOINT','CUSTOM_STORAGE_ONLY','MAMBA_REQUEST_OWNED'):
        monkeypatch.setenv('SGLANG_AGENTIC_KV_'+name,'false')
        assert not request_owned_mamba_enabled()
        monkeypatch.setenv('SGLANG_AGENTIC_KV_'+name,'true')


def tree_fixture():
    tree = MambaRadixCache.__new__(MambaRadixCache)
    tree.disable=False; tree.page_size=64; tree.device='cpu'
    tree.enable_metrics=False; tree.enable_kv_cache_events=False
    tree.key_match_fn=partial(_key_match_paged,page_size=64)
    tree.get_child_key_fn=partial(get_child_key,page_size=64)
    states=[]; pages=[]
    tree.req_to_token_pool=NS(mamba_pool=NS(free=lambda x:states.extend(x.tolist())))
    tree.token_to_kv_pool_allocator=NS(free=lambda x:pages.extend(x.tolist()))
    tree.reset()
    return tree,states,pages


def insert(tree, tokens, slot):
    tree.insert(InsertParams(key=RadixKey(tokens,'shared'),value=torch.tensor(tokens),mamba_value=torch.tensor([slot])))
    return tree.match_prefix(MatchPrefixParams(key=RadixKey(tokens,'shared'))).last_device_node


def test_new_checkpoint_locked_before_old_release_and_shared_kv_retained(enabled):
    tree,states,pages=tree_fixture()
    parent=insert(tree,list(range(64)),1)
    tree.inc_lock_ref(parent)
    a=insert(tree,list(range(128)),2)
    tree.inc_lock_ref(a)
    b=insert(tree,list(range(64))+list(range(256,320)),3)
    tree.inc_lock_ref(b)
    pages.clear()  # insert already reclaimed duplicate incoming prefix pages.
    # Old checkpoint is still referenced by a transfer: cannot be freed.
    assert tree._release_unowned_mamba_checkpoints()==0
    tree.dec_lock_ref(parent)
    assert tree._release_unowned_mamba_checkpoints()==1
    assert states==[1] and pages==[]
    assert parent.mamba_value is None and parent.full_lock_ref==2
    assert a.mamba_value.item()==2 and b.mamba_value.item()==3
    # One request retires; the other still owns shared prefix KV and its state.
    tree.dec_lock_ref(a)
    tree._release_unowned_mamba_checkpoints()
    assert 2 in states and 3 not in states
    assert not set(range(64)) & set(pages)
    tree.dec_lock_ref(b)
    tree._release_unowned_mamba_checkpoints()
    assert sorted(states)==[1,2,3]
    assert tree.mamba_evictable_size()==0 and tree.full_evictable_size()==0


def test_disabled_keeps_native_cached_state(monkeypatch):
    monkeypatch.setenv('SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED','false')
    tree,states,pages=tree_fixture()
    insert(tree,list(range(64)),1)
    assert tree._release_unowned_mamba_checkpoints()==0
    assert not states and not pages and tree.mamba_evictable_size()==1


@pytest.mark.parametrize('prompt_len,boundary', [(309,256),(8193,8128),(8220,8192)])
def test_final_short_chunk_reuses_exact_locked_radix_checkpoint(enabled,prompt_len,boundary):
    tree,_,_=tree_fixture()
    node=insert(tree,list(range(boundary)),13)
    tree.inc_lock_ref(node)
    tree.enable_mamba_extra_buffer=True
    tree.req_to_token_pool.req_to_token=torch.arange(prompt_len).reshape(1,-1)
    req=NS(origin_input_ids=list(range(prompt_len)),fill_ids=list(range(prompt_len)),
        tokenizer=NS(encode=lambda *a,**k:list(range(prompt_len-2,prompt_len))),
        mamba_last_track_seqlen=None,req_pool_idx=0,cache_protected_len=boundary,
        last_node=node)
    # Native no-new-checkpoint path updates prefix_indices but retains the
    # checkpoint copied into Radix and locked by the preceding chunk.
    tree.cache_unfinished_req(req)
    assert len(req.prefix_indices)==prompt_len
    freeze_p2d_mamba_checkpoint_after_cache(req,None,64)
    assert req._agentic_p2d_mamba_checkpoint_tokens==boundary
    assert req._agentic_p2d_mamba_checkpoint_index==13


@pytest.mark.parametrize('fault', ['protected','mamba_lock','full_lock','missing',
                                  'multiple','boundary','wrong_tracked'])
def test_retained_checkpoint_rejects_stale_or_unowned_state(enabled,fault):
    tree,_,_=tree_fixture()
    node=insert(tree,list(range(256)),13)
    tree.inc_lock_ref(node)
    req=NS(origin_input_ids=list(range(309)),cache_protected_len=256,last_node=node,
        tokenizer=NS(encode=lambda *a,**k:[307,308]))
    tracked=None
    if fault=='protected': req.cache_protected_len=192
    elif fault=='mamba_lock': node.mamba_lock_ref=0
    elif fault=='full_lock': node.full_lock_ref=0
    elif fault=='missing': node.mamba_value=None
    elif fault=='multiple': node.mamba_value=torch.tensor([13,14])
    elif fault=='boundary': node.key=RadixKey(list(range(192)),'shared')
    elif fault=='wrong_tracked': tracked=192
    with pytest.raises(RuntimeError):
        freeze_p2d_mamba_checkpoint_after_cache(req,tracked,64)


def test_single_decode_checkpoint_and_retraction_roundtrip(enabled):
    conv=[torch.arange(48,dtype=torch.float32).reshape(2,8,3)]
    temporal=torch.arange(64,dtype=torch.float32).reshape(2,8,4)
    pool=NS(mamba_pool=NS(mamba_cache=NS(conv=conv,temporal=temporal)),
            get_mamba_ping_pong_other_idx=lambda idx:0)
    req=NS(origin_input_ids=[5]*1024,tokenizer=NS(encode=lambda *a,**k:[91,92]),
           mamba_pool_idx=torch.tensor(1),mamba_ping_pong_track_buffer=torch.tensor([2]),
           mamba_next_track_idx=0)
    assert p2d_mamba_destination_indices(req,pool,64)[0].tolist()==[1,2]
    backup=offload_request_mamba(req,pool)
    assert backup[2]==1024
    req._agentic_mamba_frozen_prompt_valid=False
    assert frozen_mamba_checkpoint(req)  # Never resume writes to the single slot.
    with pytest.raises(RuntimeError,match='lost during retraction'):
        snapshot_token_count_for_req(req,2048,[StateType.MAMBA],64)
    conv[0].zero_(); temporal.zero_()
    req.mamba_pool_idx=torch.tensor(4);req.mamba_ping_pong_track_buffer=torch.tensor([5])
    restore_request_mamba(req,pool,backup)
    assert torch.equal(conv[0][:,[4,5]],backup[0][0])
    assert torch.equal(temporal[:,[4,5]],backup[1])
    assert snapshot_token_count_for_req(req,2048,[StateType.MAMBA],64)==1024


def test_capacity_budget_preserves_p_scratch(enabled):
    for mode,expected in [('prefill',5),('decode',3)]:
        runner=NS(server_args=NS(disaggregation_mode=mode,disable_radix_cache=False,
                  enable_mamba_extra_buffer=lambda:True,disable_overlap_schedule=False))
        assert ModelRunnerKVCacheMixin._calculate_mamba_ratio(runner)==expected


def test_actual_decode_pool_one_tracking_slot(enabled, monkeypatch):
    from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool, DecodeReqToTokenPool
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    monkeypatch.setattr(DecodeReqToTokenPool,'__init__',lambda self,**kwargs:None)
    monkeypatch.setattr(HybridReqToTokenPool,'_init_mamba_pool',lambda self,**kwargs:None)
    kwargs=dict(size=10,max_context_len=4096,device='cpu',enable_memory_saver=False,
                cache_params=None,mamba_layer_ids=[],speculative_num_draft_tokens=None,
                enable_mamba_extra_buffer=True,pre_alloc_size=10,enable_overlap_schedule=True,mamba_size=64)
    pool=HybridMambaDecodeReqToTokenPool(**kwargs)
    assert pool.mamba_ping_pong_track_buffer_size==1
    assert pool.get_mamba_ping_pong_other_idx(0)==0
    monkeypatch.setenv('SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED','false')
    assert HybridMambaDecodeReqToTokenPool(**kwargs).mamba_ping_pong_track_buffer_size==2


def test_invalid_retracted_state_never_rotates_single_slot(enabled):
    from sglang.srt.managers.scheduler_output_processor_mixin import SchedulerOutputProcessorMixin
    req=NS(_agentic_mamba_frozen_prompt_tokens=960,_agentic_mamba_frozen_prompt_valid=False,
           mamba_next_track_idx=0,mamba_last_track_seqlen=None)
    SchedulerOutputProcessorMixin._mamba_prefix_cache_update(None,req,None,None,0)
    assert req.mamba_next_track_idx==0 and req.mamba_last_track_seqlen is None


def test_retraction_fences_forward_before_reading_active(enabled,monkeypatch):
    events=[]
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:events.append('fence'))
    def select(dim,indices):
        events.append('read')
        assert events[0]=='fence'
        return torch.zeros(1,2,3)
    state=NS(temporal=NS(is_cuda=True,device='cuda:0',index_select=select),
             conv=[NS(index_select=select)])
    pool=NS(mamba_pool=NS(mamba_cache=state),get_mamba_ping_pong_other_idx=lambda _:0)
    req=NS(_agentic_mamba_frozen_prompt_valid=True,_agentic_mamba_frozen_prompt_tokens=960,
           mamba_pool_idx=torch.tensor(1),mamba_ping_pong_track_buffer=torch.tensor([2]),mamba_next_track_idx=0)
    offload_request_mamba(req,pool)
    assert events==['fence','read','read']


def test_req_retraction_integration_keeps_backup_until_restore(enabled,monkeypatch):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    req=Req('test',None,[1]*64,SamplingParams(max_new_tokens=10))
    req.output_ids=[2,3]; req.req_pool_idx=0
    req.mamba_pool_idx=torch.tensor(1);req.mamba_ping_pong_track_buffer=torch.tensor([2]);req.mamba_next_track_idx=0
    req._agentic_mamba_frozen_prompt_tokens=64;req._agentic_mamba_frozen_prompt_valid=True
    state=NS(conv=[torch.arange(48.).reshape(2,8,3)],temporal=torch.arange(64.).reshape(2,8,4))
    pool=NS(mamba_pool=NS(mamba_cache=state),get_mamba_ping_pong_other_idx=lambda _:0,
            req_to_token=torch.arange(128).reshape(1,128))
    copies=[]
    full=NS(get_cpu_copy=lambda indices:indices.clone(),load_cpu_copy=lambda saved,indices:copies.append((saved,indices)))
    allocator=NS(get_kvcache=lambda:NS(full_kv_pool=full))
    req.offload_kv_cache(pool,allocator)
    expected=req._agentic_retracted_mamba
    req.reset_for_retract()
    assert not req._agentic_mamba_frozen_prompt_valid and req._agentic_retracted_mamba is expected
    req.req_pool_idx=0;req.mamba_pool_idx=torch.tensor(4);req.mamba_ping_pong_track_buffer=torch.tensor([5]);req.mamba_next_track_idx=0
    def failed_restore(*args):raise RuntimeError('injected copy failure')
    full.load_cpu_copy=failed_restore
    with pytest.raises(RuntimeError,match='injected'):req.load_kv_cache(pool,allocator)
    assert req._agentic_retracted_mamba is expected and not req._agentic_mamba_frozen_prompt_valid
    full.load_cpu_copy=lambda saved,indices:copies.append((saved,indices))
    req.load_kv_cache(pool,allocator)
    assert copies and not hasattr(req,'_agentic_retracted_mamba') and not hasattr(req,'kv_cache_cpu')
    assert req._agentic_mamba_frozen_prompt_valid and req.mamba_last_track_seqlen==64
    assert torch.equal(state.temporal[:,[4,5]],expected[1])
