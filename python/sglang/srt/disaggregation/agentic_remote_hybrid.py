"""Composite Host wire layout for existing Attention + GDN snapshots.

No scheduling or ownership policy lives here. The caller owns a complete
workset and publishes a receipt only after the one composite READ is fenced.
"""
from __future__ import annotations

import mmap
import os
import torch

from sglang.srt.disaggregation.agentic_hybrid_snapshot import HybridSnapshotLayout


class HybridWireLayout:
    def __init__(self, pool, state_slots):
        self.pool = pool
        self.attention = pool.full_kv_pool
        self.slots = int(state_slots)
        if self.slots not in (1, 2):
            raise ValueError("hybrid wire requires one checkpoint or active+checkpoint")
        cache = pool.mamba_pool.mamba_cache
        self.tensors = (*cache.conv, cache.temporal)
        if not self.tensors:
            raise ValueError("hybrid wire missing recurrent state")
        for tensor in self.tensors:
            if tensor.ndim < 3 or not tensor.is_contiguous():
                raise ValueError("hybrid wire requires contiguous layer/slot state pools")

    def schema(self):
        # Capacity/address is local; per-slot layout must agree across peers.
        return {"kind": "qwen35-attention-gdn-v1", "state_slots": self.slots,
                "state_order": "conv_then_temporal/layer/slot",
                "states": [{"dtype": str(t.dtype), "layers": t.shape[0],
                            "shape": list(t.shape[2:])} for t in self.tensors]}

    def layout(self, tokens):
        return HybridSnapshotLayout.from_pools(tokens, self.attention,
                                               self.pool.mamba_pool, state_slots=self.slots)

    def ranges(self, tokens):
        layout = self.layout(tokens)
        return ((0, layout.attention_bytes), (layout.state_offset, layout.state_bytes))

    def spans(self, tokens, indices):
        if indices is None:
            raise ValueError("remote hybrid load requires pinned state slots")
        if hasattr(indices, 'detach'):
            indices = indices.detach().cpu().reshape(-1).tolist()
        indices = tuple(int(i) for i in indices)
        if len(indices) != self.slots or len(set(indices)) != self.slots or min(indices) < 0:
            raise ValueError("invalid hybrid destination state slot vector")
        layout = self.layout(tokens)
        offset, spans = layout.state_offset, []
        for tensor in self.tensors:
            if max(indices) >= tensor.shape[1]:
                raise ValueError("hybrid destination state outside registered pool")
            for layer in range(tensor.shape[0]):
                for slot in indices:
                    piece = tensor[layer, slot]
                    size = piece.numel() * piece.element_size()
                    spans.append((offset, piece.data_ptr(), size))
                    offset += size
        if offset != layout.total_bytes:
            raise ValueError("incomplete hybrid state payload")
        return spans


class CompleteHostMapping:
    """Map the full composite extent, not only its Attention sub-mapping."""
    def __init__(self, snapshot):
        from sglang.srt.disaggregation.agentic_host_staging import _open_shared_host_backing
        self.snapshot = snapshot
        fd = _open_shared_host_backing(snapshot.path, os.O_RDWR)
        try:
            offset, size = int(snapshot.file_offset), int(snapshot.byte_size)
            if offset < 0 or offset % mmap.ALLOCATIONGRANULARITY or size <= 0:
                raise ValueError("invalid composite Host extent")
            if os.fstat(fd).st_size < offset + size:
                raise ValueError("composite Host extent exceeds source backing")
            self.mapping = mmap.mmap(fd, size, offset=offset)
        finally:
            os.close(fd)
        self.raw = torch.frombuffer(self.mapping, dtype=torch.uint8)

    @property
    def address(self):
        return self.raw.data_ptr()

    def close(self):
        self.raw = None
        self.mapping.close()
        self.snapshot = None
