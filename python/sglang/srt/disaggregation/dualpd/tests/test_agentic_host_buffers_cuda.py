"""Explicit H100-only real data roundtrip; excluded from the CPU gate.

This creates tiny private GPU tensors/Host extents, not model servers. The
registered copy-engine and legacy pinned-bounce paths must both preserve data.
"""
import tempfile
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation import agentic_host_buffers as buffers


@pytest.mark.parametrize("registered", [True, False])
def test_actual_h100_d2h_h2d_roundtrip(registered, monkeypatch):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA", "1" if registered else "0")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA", "0")
    device = torch.device("cuda:0")
    shape = (128, 2, 64)
    k = [torch.arange(128 * 2 * 64, device=device).remainder(257).reshape(shape).to(torch.float16)
         + i * 512 for i in range(2)]
    v = [value + 1024 for value in k]  # Disjoint, exactly representable layer/K/V values.
    if registered:
        assert buffers._cuda_driver_batch_memcpy() is not None
    pool = NS(k_buffer=k, v_buffer=v, layer_num=2, head_num=2, head_dim=64,
              store_dtype=torch.float16, device=device,
              k_data_ptrs=torch.tensor([t.data_ptr() for t in k], dtype=torch.uint64, device=device),
              v_data_ptrs=torch.tensor([t.data_ptr() for t in v], dtype=torch.uint64, device=device))
    source = torch.arange(63, -1, -1, dtype=torch.int64, device=device)
    destination = torch.arange(127, 63, -1, dtype=torch.int64, device=device)
    expected = torch.stack([torch.stack([t.index_select(0, source).cpu() for t in layers])
                            for layers in (k, v)])
    byte_size = expected.numel() * expected.element_size()
    staging = buffers.LayerFirstD2HStaging(pool, 64)
    bounce = buffers.PinnedMHAHostBounce(pool, 64)
    stream = torch.cuda.Stream(device=device)
    # Source initialization/indices must finish before the private transfer
    # stream reads them; the production caller has its equivalent ready fence.
    stream.wait_stream(torch.cuda.current_stream(device))
    with tempfile.TemporaryDirectory(prefix="dualpd-real-dma-", dir="/dev/shm") as directory:
        arena = buffers.SharedHostSnapshotArena(directory, byte_size * 2, backend="memfd")
        snapshot = arena.create("roundtrip", 64, pool, byte_size)
        try:
            snapshot.materialize()
            event, backup_refs = snapshot.start_backup_range_from_device(
                source, destination_start=0, stream=stream, staging=staging,
                host_bounce=bounce, source_indices_host=source.cpu().tolist())
            event.synchronize()
            assert snapshot.cuda_host_registered is registered
            snapshot.commit_backup_range_from_bounce(bounce, destination_start=0, token_count=64)
            snapshot.mark_populated()
            assert torch.equal(snapshot.kv_buffer, expected)
            with torch.cuda.stream(stream):
                for t in (*k, *v):
                    t.zero_()
            snapshot.prepare_load_range_to_bounce(source_start=0, token_count=64, host_bounce=bounce)
            event, load_refs = snapshot.start_load_range_from_bounce_to_device(
                destination, stream, staging=staging, host_bounce=bounce,
                source_start=0, device_indices_host=destination.cpu().tolist())
            event.synchronize()
            for kind, layers in enumerate((k, v)):
                for layer, tensor in enumerate(layers):
                    assert torch.equal(tensor.index_select(0, destination).cpu(), expected[kind, layer])
                    assert not tensor[:64].any()
            assert backup_refs and load_refs  # Keep descriptor/tensor storage until physical completion.
        finally:
            # If synchronization cannot establish quiescence, do not release
            # the Host extent; the failing test process retains ownership.
            stream.synchronize()
            assert arena.release(snapshot)
            assert arena.used_bytes == 0
            arena.close()
