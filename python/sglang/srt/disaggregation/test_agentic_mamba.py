from types import SimpleNamespace
import os
import tempfile

import torch

from sglang.srt.disaggregation.agentic_direct_transfer import _make_kv_args
from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_hybrid_snapshot import (
    HybridSnapshotLayout,
    SharedHybridHostSnapshot,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    RequestGeneration,
    SnapshotManifest,
    SnapshotState,
)
from sglang.srt.disaggregation.agentic_hybrid_transfer import (
    state_indices_for_req,
    state_indices_for_workset,
    submit_reverse_receive,
    submit_reverse_send,
)
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


class _HostKVPool:
    layer_num = 2
    head_num = 2
    head_dim = 4
    v_head_dim = 4
    store_dtype = torch.float16

    def __init__(self):
        self.k_buffer = [torch.empty(8, 2, 4) for _ in range(self.layer_num)]
        self.v_buffer = [torch.empty(8, 2, 4) for _ in range(self.layer_num)]


class _HostMambaPool:
    def __init__(self):
        self.mamba_cache = SimpleNamespace(
            conv=[torch.zeros(2, 3, 2, dtype=torch.float32)],
            temporal=torch.zeros(2, 3, 2, 2, dtype=torch.float32),
        )
        self.loaded = None

    def get_cpu_copy(self, indices):
        return (
            [tensor[:, indices].clone() for tensor in self.mamba_cache.conv],
            self.mamba_cache.temporal[:, indices].clone(),
        )

    def load_cpu_copy(self, state, indices):
        conv, temporal = state
        for destination, source in zip(self.mamba_cache.conv, conv):
            destination[:, indices] = source
        self.mamba_cache.temporal[:, indices] = temporal
        self.loaded = indices.clone()


def test_shared_hybrid_snapshot_round_trips_complete_mamba_slot():
    kv_pool = _HostKVPool()
    source_pool = _HostMambaPool()
    source_pool.mamba_cache.conv[0][:, 1].fill_(3)
    source_pool.mamba_cache.temporal[:, 1].fill_(7)
    layout = HybridSnapshotLayout.from_pools(5, kv_pool, source_pool)
    assert layout.state_offset % 4096 == 0
    assert layout.total_bytes > layout.attention_bytes

    directory = tempfile.mkdtemp(prefix="sglang-agentic-mamba-", dir="/dev/shm")
    path = os.path.join(directory, "snapshot.bin")
    snapshot = SharedHybridHostSnapshot(
        path=path,
        token_count=5,
        kv_pool=kv_pool,
        mamba_pool=source_pool,
        create=True,
        layout=layout,
    )
    try:
        snapshot.mamba.backup_from_device(1)
        destination_pool = _HostMambaPool()
        snapshot.mamba.mamba_pool = destination_pool
        snapshot.mamba.load_to_device(2)
        assert torch.all(destination_pool.mamba_cache.conv[0][:, 2] == 3)
        assert torch.all(destination_pool.mamba_cache.temporal[:, 2] == 7)
        assert destination_pool.loaded.tolist() == [2]
    finally:
        snapshot.close(unlink=True)
        os.rmdir(directory)


def test_manifest_commits_attention_and_mamba_as_one_generation():
    manifest = SnapshotManifest(
        request=RequestGeneration("hybrid", 2),
        page_keys=("kv-page-0", "mamba-state"),
        token_count=129,
        byte_size=8192,
        state=SnapshotState.MOONCAKE_READY,
        cache_components=("attention", "mamba"),
        state_byte_size=2048,
        state_checkpoint_tokens=128,
    )
    restored = SnapshotManifest.from_bytes(manifest.to_bytes())

    assert restored.cache_components == ("attention", "mamba")
    assert restored.state_byte_size == 2048
    assert restored.state_checkpoint_tokens == 128


def test_reverse_wire_submission_includes_mamba_source_and_destination():
    mamba_type = [StateType.MAMBA]
    req = SimpleNamespace(mamba_pool_idx=torch.tensor(7))
    lease = SimpleNamespace(
        parent_page_indices=torch.tensor([10, 11]).numpy(),
        state_device_indices=(torch.tensor([9]),),
    )
    sent = []
    metadata = []
    sender = SimpleNamespace(
        send=lambda pages, state_indices=None: sent.append((pages, state_indices))
    )
    receiver = SimpleNamespace(
        send_metadata=lambda pages, aux_index=None, state_indices=None: metadata.append(
            (pages, aux_index, state_indices)
        )
    )

    submit_reverse_send(sender, torch.tensor([1, 2]).numpy(), req, mamba_type)
    submit_reverse_receive(receiver, lease, mamba_type)

    assert int(state_indices_for_req(req, mamba_type)[0][0]) == 7
    assert int(state_indices_for_workset(lease, mamba_type)[0][0]) == 9
    assert int(sent[0][1][0][0]) == 7
    assert metadata[0][1] == 0
    assert int(metadata[0][2][0][0]) == 9
