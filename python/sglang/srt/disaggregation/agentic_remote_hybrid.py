"""Composite Host wire layout for Attention plus request-owned Mamba state.

This module contains layout and address calculations only.  Scheduling,
ownership and TP group decisions belong to the group controller.
"""
from __future__ import annotations

import mmap
import os

import torch

from sglang.srt.disaggregation.agentic_hybrid_snapshot import HybridSnapshotLayout


class HybridWireLayout:
    """Describe one complete Qwen3.5 Attention + GDN wire snapshot."""

    def __init__(self, pool, state_slots: int):
        self.pool = pool
        self.attention = pool.full_kv_pool
        self.slots = int(state_slots)
        if self.slots not in (1, 2):
            raise ValueError("hybrid wire requires one checkpoint or active+checkpoint")
        cache = pool.mamba_pool.mamba_cache
        self.tensors = (*cache.conv, cache.temporal)
        if not self.tensors:
            raise ValueError("hybrid wire is missing recurrent state")
        for tensor in self.tensors:
            if tensor.ndim < 3 or not tensor.is_contiguous():
                raise ValueError(
                    "hybrid wire requires contiguous layer/slot state pools"
                )

    def schema(self) -> dict:
        # Capacity and addresses are rank-local.  Per-slot shape must match.
        return {
            "kind": "qwen35-attention-gdn-v1",
            "state_slots": self.slots,
            "state_order": "conv_then_temporal/layer/slot",
            "states": [
                {
                    "dtype": str(tensor.dtype),
                    "layers": tensor.shape[0],
                    "shape": list(tensor.shape[2:]),
                }
                for tensor in self.tensors
            ],
        }

    def layout(self, tokens: int) -> HybridSnapshotLayout:
        return HybridSnapshotLayout.from_pools(
            tokens,
            self.attention,
            self.pool.mamba_pool,
            state_slots=self.slots,
        )

    def payload_ranges(self, tokens: int) -> tuple[tuple[int, int], ...]:
        layout = self.layout(tokens)
        return (
            (0, layout.attention_bytes),
            (layout.state_offset, layout.state_bytes),
        )

    def state_spans(self, tokens: int, indices) -> list[tuple[int, int, int]]:
        if indices is None:
            raise ValueError("remote hybrid load requires reserved state slots")
        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().reshape(-1).tolist()
        indices = tuple(int(index) for index in indices)
        if (
            len(indices) != self.slots
            or len(set(indices)) != self.slots
            or min(indices) < 0
        ):
            raise ValueError("invalid hybrid destination state slot vector")

        layout = self.layout(tokens)
        offset = layout.state_offset
        spans = []
        for tensor in self.tensors:
            if max(indices) >= tensor.shape[1]:
                raise ValueError("hybrid destination state is outside the pool")
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
    """Keep the complete composite Host extent mapped for NIXL registration."""

    def __init__(self, snapshot):
        from sglang.srt.disaggregation.agentic_host_staging import (
            _open_shared_host_backing,
        )

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
    def address(self) -> int:
        return self.raw.data_ptr()

    def close(self) -> None:
        self.raw = None
        self.mapping.close()
        self.snapshot = None
