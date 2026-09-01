"""Wire helpers for complete reverse Attention+Mamba snapshots."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sglang.srt.disaggregation.base.conn import StateType


def _one_mamba_component(state_types: Sequence[StateType]) -> bool:
    return sum(item == StateType.MAMBA for item in state_types) == 1


def state_indices_for_req(req, state_types: Sequence[StateType]) -> list:
    """Build the state-index vector expected by native NIXL/Mooncake.

    The list is parallel to ``kv_args.state_types``.  Dense models return an
    empty list; Qwen3.5 returns one nested one-slot index for its Mamba state.
    """

    if not state_types:
        return []
    result = []
    for state_type in state_types:
        if state_type == StateType.MAMBA:
            index = getattr(req, "mamba_pool_idx", None)
            if index is None:
                raise RuntimeError("complete hybrid snapshot is missing Mamba state")
            result.append([np.asarray(int(index), dtype=np.int32)])
        else:
            raise RuntimeError(
                f"agentic reverse transfer does not support state type {state_type}"
            )
    return result


def state_indices_for_workset(lease, state_types: Sequence[StateType]) -> list:
    if not state_types:
        return []
    if not _one_mamba_component(state_types):
        raise RuntimeError("agentic workset supports exactly one Mamba component")
    indices = getattr(lease, "state_device_indices", ())
    if len(indices) != 1 or indices[0].numel() != 1:
        raise RuntimeError("hybrid destination workset is missing its Mamba slot")
    return [[indices[0].detach().cpu().numpy().astype(np.int32)[0]]]


def submit_reverse_receive(receiver, lease, state_types: Sequence[StateType]) -> None:
    receiver.send_metadata(
        lease.parent_page_indices,
        aux_index=0,
        state_indices=state_indices_for_workset(lease, state_types),
    )


def submit_reverse_send(
    sender,
    page_indices,
    req,
    state_types: Sequence[StateType],
) -> None:
    sender.send(
        page_indices,
        state_indices=state_indices_for_req(req, state_types),
    )


def state_payload_bytes(kv_args) -> int:
    """Physical bytes for one request slot across all registered state tensors."""

    return sum(
        int(item_len)
        for component in getattr(kv_args, "state_item_lens", ())
        for item_len in component
    )


def complete_snapshot_bytes(kv_args, token_count: int) -> int:
    attention = int(token_count) * sum(
        int(item_len) for item_len in getattr(kv_args, "kv_item_lens", ())
    )
    return attention + state_payload_bytes(kv_args)
