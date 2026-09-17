"""M2.7 TP8 ordinary GQA wire layout; CPU only, not an RDMA/GPU test."""
import ctypes

import pytest
import torch

from sglang.srt.disaggregation.agentic_remote_host import mha_read_spans


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_minimax_gqa_all_rank_byte_mapping(tp):
    # Official config:62 layers,8 KV heads,head_dim128. No hybrid state.
    layers, heads, dim = 62, 8 // tp, 128
    indices = [2, 3, 7]
    item_bytes = heads * dim * 2
    logical_bytes = 0
    for rank in range(tp):
        source = torch.arange(2 * layers * len(indices) * heads * dim,
                              dtype=torch.int32).add_(rank).to(torch.int16)
        target = [torch.zeros((8, heads, dim), dtype=torch.int16) for _ in range(2 * layers)]
        spans = mha_read_spans(layer_bases=[t.data_ptr() for t in target],
                               token_indices=indices, bytes_per_token=item_bytes)
        assert len(spans) == 2 * layers * 2  # contiguous [2,3] plus token7
        assert sum(size for _, _, size in spans) == source.nbytes
        for offset, address, size in spans:
            # CPU byte copies emulate the READ descriptor destinations only.
            ctypes.memmove(address, source.data_ptr() + offset, size)
        restored = torch.stack([t[indices] for t in target]).flatten()
        assert torch.equal(restored, source)
        assert all(torch.count_nonzero(t[0]) == 0 for t in target)
        logical_bytes += source.nbytes
    assert logical_bytes == len(indices) * 253952
