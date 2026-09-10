"""Exact GPU->memfd->GPU Attention + recurrent state transport check."""
import os
from types import SimpleNamespace as NS

import torch

from sglang.srt.disaggregation.agentic_hybrid_dma import RegisteredHybridHostSnapshot
from sglang.srt.disaggregation.agentic_hybrid_snapshot import HybridSnapshotLayout
from sglang.srt.disaggregation.agentic_host_staging import LayerFirstD2HStaging, PinnedMHAHostBounce, _create_agentic_host_memfd
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


def check(slots, registered):
    os.environ["SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA"] = str(int(registered))
    os.environ["SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB"] = "1"
    os.environ["SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB"] = "1"
    device = "cuda:0"
    attention = MHATokenToKVPool(size=384, page_size=64, dtype=torch.bfloat16,
        head_num=1, head_dim=32, layer_num=2, device=device, enable_memory_saver=False)
    state = NS(mamba_cache=NS(conv=[torch.randn(2, 8, 4, 8, device=device)],
                              temporal=torch.randn(2, 8, 8, 8, device=device)))
    pool = NS(full_kv_pool=attention, mamba_pool=state, device=device)
    for tensor in attention.k_buffer + attention.v_buffer:
        tensor.copy_(torch.randn_like(tensor))
    source_ids = torch.arange(64, 192, device=device)
    destination_ids = torch.arange(192, 320, device=device)
    source_state = list(range(1, 1 + slots))
    destination_state = list(range(4, 4 + slots))
    layout = HybridSnapshotLayout.from_pools(128, attention, state, state_slots=slots)
    fd = _create_agentic_host_memfd()
    os.ftruncate(fd, layout.total_bytes)
    path = f"/proc/{os.getpid()}/fd/{fd}"
    writer = reader = None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    staging = LayerFirstD2HStaging(attention, 64)
    bounce = PinnedMHAHostBounce(attention, 64)
    try:
        writer = RegisteredHybridHostSnapshot(path=path, token_count=128, device_pool=pool,
            byte_size=layout.total_bytes, state_slots=slots)
        writer.set_state_indices(source_state)
        for offset in (0, 64):
            event, refs = writer.start_backup_range_from_device(source_ids[offset:offset+64],
                destination_start=offset, stream=stream, staging=staging, host_bounce=bounce,
                source_indices_host=source_ids[offset:offset+64].cpu())
            event.synchronize()
            writer.commit_backup_range_from_bounce(bounce, destination_start=offset, token_count=64)
        writer.close()
        writer = None
        reader = RegisteredHybridHostSnapshot(path=path, token_count=128, device_pool=pool,
            byte_size=layout.total_bytes, state_slots=slots)
        reader.set_state_indices(destination_state)
        for offset in (0, 64):
            event, refs = reader.start_load_range_to_device(destination_ids[offset:offset+64], stream,
                source_start=offset, staging=staging, host_bounce=bounce,
                device_indices_host=destination_ids[offset:offset+64].cpu())
            event.synchronize()
        for tensor in attention.k_buffer + attention.v_buffer:
            assert torch.equal(tensor[source_ids], tensor[destination_ids]), "Attention mismatch"
        for tensor in (*state.mamba_cache.conv, state.mamba_cache.temporal):
            assert torch.equal(tensor[:, source_state], tensor[:, destination_state]), "Mamba mismatch"
        print(f"PASS slots={slots} registered_requested={registered} bytes={layout.total_bytes}", flush=True)
    finally:
        stream.synchronize()
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
        os.close(fd)


if __name__ == "__main__":
    for registered in (True, False):
        for slots in (1, 2):
            check(slots, registered)
