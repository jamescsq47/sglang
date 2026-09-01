from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.agentic_direct_transfer import _make_kv_args
from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.utils import TransferBackend


class _FakeKVPool:
    start_layer = 0
    head_num = 4
    page_size = 64

    @staticmethod
    def get_contiguous_buf_infos():
        return [1000, 2000], [4096, 4096], [128, 128]


class _FakeHybridReqPool:
    @staticmethod
    def get_state_buf_infos():
        # Temporal state and convolution state, one item of each per request.
        return [3000, 4000], [8192, 2048], [512, 128]

    @staticmethod
    def get_state_dim_per_tensor():
        return [4, 3]


def _server_args():
    return SimpleNamespace(
        disaggregation_ib_device="",
        disaggregation_ib_traffic_class="",
    )


def test_reverse_direct_runtime_describes_complete_qwen35_state():
    args, _aux = _make_kv_args(
        transfer_backend=TransferBackend.NIXL,
        kv_pool=_FakeKVPool(),
        server_args=_server_args(),
        engine_rank=0,
        pp_rank=0,
        gpu_id=0,
        total_kv_heads=8,
        req_to_token_pool=_FakeHybridReqPool(),
    )

    assert args.state_types == [StateType.MAMBA]
    assert args.state_data_ptrs == [[3000, 4000]]
    assert args.state_data_lens == [[8192, 2048]]
    assert args.state_item_lens == [[512, 128]]
    assert args.state_dim_per_tensor == [[4, 3]]


def test_dense_reverse_runtime_has_no_auxiliary_state_component():
    args, _aux = _make_kv_args(
        transfer_backend=TransferBackend.NIXL,
        kv_pool=_FakeKVPool(),
        server_args=_server_args(),
        engine_rank=0,
        pp_rank=0,
        gpu_id=0,
        total_kv_heads=8,
    )

    assert args.state_types == []
    assert args.state_data_ptrs == []


class _Allocator:
    def __init__(self, available=256):
        self.available = available
        self.cursor = 0
        self.freed = []

    def available_size(self):
        return self.available

    def alloc(self, count):
        if count > self.available:
            return None
        result = torch.arange(self.cursor, self.cursor + count, dtype=torch.int64)
        self.cursor += count
        self.available -= count
        return result

    def free(self, indices):
        self.available += int(indices.numel())
        self.freed.append(indices.clone())


def test_hybrid_workset_grant_and_handoff_are_one_composite_ownership_unit():
    kv_allocator = _Allocator()
    mamba_allocator = _Allocator(available=2)
    broker = AgenticPWorksetLeaseBroker(
        page_size=64, state_allocators=[mamba_allocator]
    )
    broker.request("req:1", parent_tokens=65, prompt_tokens=70)
    broker.service(kv_allocator)
    lease = broker.get("req:1")

    assert lease is not None
    assert lease.parent_allocated_tokens == 128
    assert len(lease.state_device_indices) == 1
    assert lease.state_device_indices[0].numel() == 1

    assert broker.begin_bind("req:1", lease)
    broker.commit_parent_bound("req:1", lease)
    req = SimpleNamespace(origin_input_ids=list(range(70)), mamba_pool_idx=None)
    broker.handoff_to_req("req:1", req, lease)
    assert req.mamba_pool_idx.item() == lease.state_device_indices[0][0].item()
    assert req.mamba_last_track_seqlen == 65
    assert mamba_allocator.freed == []


def test_hybrid_workset_rolls_back_attention_when_mamba_slot_is_unavailable():
    kv_allocator = _Allocator()
    mamba_allocator = _Allocator(available=0)
    broker = AgenticPWorksetLeaseBroker(
        page_size=64, state_allocators=[mamba_allocator]
    )
    broker.request("req:2", parent_tokens=64, prompt_tokens=65)
    broker.service(kv_allocator)

    assert broker.get("req:2") is None
    assert len(kv_allocator.freed) == 1
    assert kv_allocator.available == 256
