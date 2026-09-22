"""Real CPU byte copies through the remote descriptor ABI, not RDMA claims."""
import ctypes
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.agentic_remote_hybrid import HybridWireLayout
from sglang.srt.disaggregation.agentic_remote_host import RemoteHostTransport
from sglang.srt.disaggregation.agentic_remote_host_engine import RemoteHostEngineBridge
from sglang.srt.disaggregation.test_agentic_remote_host import FakeNixl


class CPUBuffer:
    device = SimpleNamespace(index=0)
    def __init__(self):
        self.tensor = torch.zeros((32, 1, 2), dtype=torch.bfloat16)
        self.shape = self.tensor.shape
    def data_ptr(self):
        return self.tensor.data_ptr()


def test_multinode_hybrid_guard_requires_complete_opt_in(monkeypatch, tmp_path):
    from sglang.srt.disaggregation.test_agentic_multinode import environment, m
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool
    config = m.load_multinode_config(environment())
    bridge, _ = make_bridge(tmp_path, 'd2p', 0, 8, 'source')
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.full_kv_pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.full_kv_pool.store_dtype = torch.bfloat16
    pool.mamba_pool = bridge.hybrid.pool.mamba_pool
    flags = ['SGLANG_AGENTIC_MULTINODE_QWEN35_HYBRID',
             'SGLANG_AGENTIC_KV_LIFECYCLE', 'SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY',
             'SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT', 'SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED']
    for flag in flags:
        monkeypatch.setenv(flag, 'true')
    config.validate_kv_pool(pool)
    for flag in flags:
        monkeypatch.setenv(flag, 'false')
        with pytest.raises(ValueError):
            config.validate_kv_pool(pool)
        monkeypatch.setenv(flag, 'true')
    pool.full_kv_pool.store_dtype = torch.int8
    with pytest.raises(ValueError):
        config.validate_kv_pool(pool)


def test_hybrid_guard_requires_checkpoint_page_alignment(monkeypatch):
    from sglang.srt.disaggregation.test_agentic_multinode import environment, m
    monkeypatch.setenv('SGLANG_AGENTIC_MULTINODE_QWEN35_HYBRID', 'true')
    args = SimpleNamespace(tp_size=8, dp_size=1, pp_size=1, nnodes=1,
        disaggregation_mode='prefill', disaggregation_transfer_backend='nixl',
        mamba_scheduler_strategy='extra_buffer', mamba_track_interval=64, page_size=64)
    config = m.load_multinode_config(environment())
    config.validate_server_args(args)
    args.mamba_track_interval = 128
    with pytest.raises(ValueError, match='page-aligned'):
        config.validate_server_args(args)


class CopyNixl(FakeNixl):
    def transfer(self, handle):
        result = super().transfer(handle)
        local, remote = self.reads[-1]
        for (target, size, _), (source, source_size, _) in zip(local[1], remote[1]):
            assert size == source_size
            ctypes.memmove(target, source, size)
        return result


def make_bridge(root, direction, rank, size, node):
    attention = SimpleNamespace(k_buffer=[CPUBuffer()], v_buffer=[CPUBuffer()],
        head_num=1, head_dim=2, store_dtype=torch.bfloat16, layer_num=1)
    cache = SimpleNamespace(conv=[torch.zeros((2, 8, 4, 3), dtype=torch.bfloat16)],
                            temporal=torch.zeros((2, 8, 2, 2, 2), dtype=torch.float32))
    pool = SimpleNamespace(full_kv_pool=attention, mamba_pool=SimpleNamespace(mamba_cache=cache))
    cfg = SimpleNamespace(tp_size=size, control_directory=str(root), node_id=node, engine_id=node)
    agent = CopyNixl()
    agent.status = 'DONE'
    transport = RemoteHostTransport(agent)
    bridge = RemoteHostEngineBridge(pool, 64, rank, size, direction, cfg,
                                   transport_factory=lambda: transport)
    return bridge, agent


def publish(root, source, rank):
    layout = source.hybrid.layout(3)
    raw = bytes((i + rank) % 251 for i in range(layout.total_bytes))
    fd = os.memfd_create('sglang-agentic-host-arena-unit-test')
    try:
        os.write(fd, raw)
        snapshot = SimpleNamespace(path=f'/proc/{os.getpid()}/fd/{fd}', file_offset=0,
                                   token_count=3, byte_size=layout.total_bytes)
        source.export_snapshot('snapshot', snapshot)
    finally:
        os.close(fd)
    return {'token_count': 3, 'byte_size': len(raw), 'remote_host_node': 'source'}, raw


@pytest.mark.parametrize('direction', ['p2d', 'd2p'])
@pytest.mark.parametrize('size', [1, 2, 8])
def test_composite_bytes_all_ranks_and_atomic_release(tmp_path, direction, size):
    sources = [make_bridge(tmp_path, direction, r, size, 'source') for r in range(size)]
    targets = [make_bridge(tmp_path, direction, r, size, 'target') for r in range(size)]
    payloads = [publish(tmp_path, source, r) for r, (source, _) in enumerate(sources)]
    slots = [2, 5] if direction == 'p2d' else [2]
    for rank, (target, agent) in enumerate(targets):
        grant, raw = payloads[rank]
        target.load('snapshot', grant, [7, 8, 13], attempt_id='one', state_indices=slots)
        # Check every descriptor byte: replicated Attention and distinct state
        # both land at the destination indices, padding is not interpreted as KV.
        local, remote = agent.reads[0]
        base = sources[rank][0]._exports['snapshot'].shard.address
        copied = 0
        for (dst, length, _), (src, _, _) in zip(local[1], remote[1]):
            offset = src - base
            assert ctypes.string_at(dst, length) == raw[offset:offset + length]
            copied += length
        assert copied == sum(n for _, n in target.hybrid.ranges(3))
        if rank < size - 1:
            assert not sources[rank][0].cleanup_source('snapshot', {'state': 'consumed'})
    for source, _ in sources:
        assert source.cleanup_source('snapshot', {'state': 'consumed'})
        assert not source._source_views


def test_hybrid_rejects_missing_state_and_wrong_schema(tmp_path):
    source, _ = make_bridge(tmp_path, 'p2d', 0, 1, 'source')
    target, agent = make_bridge(tmp_path, 'p2d', 0, 1, 'target')
    grant, _ = publish(tmp_path, source, 0)
    with pytest.raises(ValueError, match='state slots'):
        target.load('snapshot', grant, [1, 2, 3], attempt_id='bad')
    assert not agent.reads
    for slots in ([1], [1, 1], [1, 8]):
        with pytest.raises(ValueError):
            target.hybrid.spans(3, slots)
    other, _ = make_bridge(tmp_path, 'd2p', 0, 1, 'target')
    assert other.layout != target.layout
    assert source.cleanup_source('snapshot', {'state': 'evicting'})


def test_one_failed_rank_retains_every_source_until_fenced(tmp_path):
    sources = [make_bridge(tmp_path, 'd2p', r, 8, 'source') for r in range(8)]
    targets = [make_bridge(tmp_path, 'd2p', r, 8, 'target') for r in range(8)]
    grants = [publish(tmp_path, source, r)[0] for r, (source, _) in enumerate(sources)]
    for r in range(7):
        targets[r][0].load('snapshot', grants[r], [1, 2, 3], attempt_id='one', state_indices=[2])
    assert not sources[0][0].cleanup_source('snapshot', {'state': 'failed'})
    targets[7][1].status = 'ERR'
    with pytest.raises(RuntimeError, match='READ failed'):
        targets[7][0].load('snapshot', grants[7], [1, 2, 3], attempt_id='one', state_indices=[2])
    for source, _ in sources:
        assert source.cleanup_source('snapshot', {'state': 'failed'})


def test_padding_cannot_hide_missing_state(tmp_path):
    source, _ = make_bridge(tmp_path, 'd2p', 0, 1, 'source')
    target, agent = make_bridge(tmp_path, 'd2p', 0, 1, 'target')
    publish(tmp_path, source, 0)
    shard = source._exports['snapshot'].shard
    with pytest.raises(ValueError, match='complete shard'):
        target._transport_for_worker().prepare_read(shard, read_id='bad', tp_rank=0,
            tp_size=1, layout=target.layout, gpu_id=0, spans=[(0, 12345, 24)],
            payload_ranges=target.hybrid.ranges(3))
    assert not agent.reads
    assert source.cleanup_source('snapshot', {'state': 'evicting'})


def test_rank_validation_failure_can_retry_entire_group(tmp_path):
    sources = [make_bridge(tmp_path, 'd2p', r, 8, 'source') for r in range(8)]
    targets = [make_bridge(tmp_path, 'd2p', r, 8, 'target') for r in range(8)]
    grants = [publish(tmp_path, source, r)[0] for r, (source, _) in enumerate(sources)]
    for rank in range(7):
        targets[rank][0].load('snapshot', grants[rank], [1, 2, 3], attempt_id='old', state_indices=[2])
    with pytest.raises(ValueError):
        targets[7][0].load('snapshot', grants[7], [1, 2, 3], attempt_id='old', state_indices=[99])
    for rank in range(8):
        targets[rank][0].load('snapshot', grants[rank], [8, 9, 10], attempt_id='new', state_indices=[3])
    for source, _ in sources:
        assert source.cleanup_source('snapshot', {'state': 'consumed'})


def test_network_hybrid_retry_does_not_touch_foreign_mapping(monkeypatch):
    import threading
    from concurrent.futures import Future
    from sglang.srt.disaggregation import agentic_host_staging as host
    from sglang.srt.disaggregation.agentic_multinode_d2p import RemoteHostDescriptor
    record = {'network_host': True, 'loading': True,
              'snapshot': RemoteHostDescriptor({'token_count': 3, 'byte_size': 4096})}
    future = Future()
    future.set_running_or_notify_cancel()
    load = dict(record=record, remote_h2d_future=future,
        workset_lease=SimpleNamespace(state_device_indices=(torch.tensor([2]),)),
        request_generation=SimpleNamespace(snapshot_id='s'), io_attempt='one', io_inflight=True,
        remote_h2d_attempt='child:epoch:1')
    calls = []
    manager = SimpleNamespace(loads={'r': load}, host_ready={}, owner='p', tp_rank=0, tp_size=8,
        _remote_host_bridge=object(), _get_state_lock=lambda: threading.RLock(),
        ledger=SimpleNamespace(request_d2p_retry=lambda *a, **k: True,
                               complete_d2p_retry_rank=lambda *a, **k: True),
        workset_broker=SimpleNamespace(mark_io_quiesced=lambda *a: calls.append('quiesce') or True,
                                       request_release=lambda *a: calls.append('release')))
    monkeypatch.setattr(host.AgenticPHostStagingManager, '_notify_scheduler', lambda *a: None)
    monkeypatch.setattr(host.AgenticPHostStagingManager, '_release_h2d_lane', lambda *a: None)
    host.AgenticPHostStagingManager._configure_hybrid_h2d_state(load)
    assert not host.AgenticPHostStagingManager._discard_failed_h2d_load(manager, 'r', load)
    assert calls == []
    future.set_exception(RuntimeError('READ failed after physical drain'))
    assert host.AgenticPHostStagingManager._discard_failed_h2d_load(manager, 'r', load)
    assert calls == ['quiesce', 'release']
    assert manager.host_ready['s'] is record and not record['loading']
