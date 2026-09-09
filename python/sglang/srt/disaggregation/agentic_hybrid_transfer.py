"""Wire helpers for complete reverse Attention+Mamba snapshots."""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

import mmap

import numpy as np
import torch
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.environ import envs


def prompt_checkpoint_enabled() -> bool:
    return (
        envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
        and envs.SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT.get()
    )


def p2d_mamba_checkpoint_tokens(req, page_size: int) -> int:
    """Select a conservative prompt checkpoint without rewriting any input.

    Qwen's next-user template can remove the last assistant's thinking opener
    and reasoning. Preserve a page checkpoint before that opener. The receiver
    still checks the actual next prompt digest; this is not permission to reuse
    an arbitrary prefix after a history edit.
    """
    length = len(req.origin_input_ids)
    if prompt_checkpoint_enabled():
        tokenizer = getattr(req, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("stable Mamba prompt checkpoint requires a tokenizer")
        opener = list(tokenizer.encode("<think>\n", add_special_tokens=False))
        if opener and list(req.origin_input_ids[-len(opener):]) == opener:
            length -= len(opener)
    return length // int(page_size) * int(page_size)


def _one_mamba_component(state_types: Sequence[StateType]) -> bool:
    return sum(item == StateType.MAMBA for item in state_types) == 1


def debug_mamba_digest(req_to_token_pool, state_indices) -> str | None:
    """Hash exact conv/temporal slots for opt-in D->P diagnostics."""

    if os.getenv("SGLANG_AGENTIC_KV_DEBUG_DIGEST", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    if not state_indices or len(state_indices) != 1:
        return None
    raw_indices = state_indices[0]
    if torch.is_tensor(raw_indices):
        indices = (
            raw_indices.detach().to(device="cpu", dtype=torch.int64).numpy().reshape(-1)
        )
    else:
        indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return None
    mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
    cache = None if mamba_pool is None else getattr(mamba_pool, "mamba_cache", None)
    if cache is None:
        return None
    device = cache.temporal.device
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)

    def digest_tensors(tensors) -> str:
        digest = hashlib.sha256()
        for tensor in tensors:
            selected = tensor.index_select(1, index_tensor).contiguous().view(torch.uint8)
            digest.update(selected.cpu().numpy().tobytes())
        return digest.hexdigest()

    return "conv=" + digest_tensors(cache.conv) + ",temporal=" + digest_tensors(
        [cache.temporal]
    )


def validate_agentic_mamba_tracking(kv_args, server_args) -> None:
    """Require one reverse-safe state checkpoint per Attention page."""

    if StateType.MAMBA not in (getattr(kv_args, "state_types", ()) or ()):
        return
    page_size = int(getattr(kv_args, "page_size", 1) or 1)
    interval = int(getattr(server_args, "mamba_track_interval", 0) or 0)
    if interval != page_size:
        raise ValueError(
            "agentic reverse KV for Mamba requires "
            f"--mamba-track-interval={page_size}; got {interval}"
        )
    if bool(getattr(server_args, "enable_int8_mamba_checkpoint", False)):
        raise ValueError(
            "agentic reverse KV V1 does not support " "--enable-int8-mamba-checkpoint"
        )
    if prompt_checkpoint_enabled():
        if getattr(server_args, "speculative_algorithm", None):
            raise ValueError("stable Mamba prompt checkpoints do not support speculation")
        lazy = getattr(server_args, "enable_mamba_extra_buffer_lazy", None)
        if callable(lazy) and lazy():
            raise ValueError("stable Mamba prompt checkpoints require non-lazy buffers")


def state_indices_for_req(
    req,
    state_types: Sequence[StateType],
    *,
    checkpoint_tokens: int | None = None,
    page_size: int | None = None,
) -> list:
    """Build the state-index vector expected by native NIXL/Mooncake.

    The list is parallel to ``kv_args.state_types``.  Dense models return an
    empty list; Qwen3.5 returns one nested one-slot index for its Mamba state.
    """

    if not state_types:
        return []
    result = []
    for state_type in state_types:
        if state_type == StateType.MAMBA:
            index = mamba_checkpoint_index_for_req(
                req,
                checkpoint_tokens=checkpoint_tokens,
                page_size=page_size,
            )
            if index is None:
                raise RuntimeError("complete hybrid snapshot is missing Mamba state")
            result.append([np.asarray(int(index), dtype=np.int32)])
        else:
            raise RuntimeError(
                f"agentic reverse transfer does not support state type {state_type}"
            )
    return result


def mamba_checkpoint_index_for_req(
    req, *, checkpoint_tokens: int | None = None, page_size: int | None = None
):
    """Return the slot matching ``req.mamba_last_track_seqlen``.

    Mamba Cache V2 keeps page-aligned checkpoints in a ping-pong buffer.  In
    normal mode both slots exist and the most recently written checkpoint is
    opposite ``mamba_next_track_idx``.  Lazy mode frees the old slot after a
    boundary, making the sole non-negative slot (the next index) authoritative.
    Falling back to ``mamba_pool_idx`` is valid for requests without the V2
    extra buffer and for an exact, currently active checkpoint.
    """

    if getattr(req, "mamba_lazy_is_insert", None) is False:
        raise RuntimeError(
            "lazy Mamba checkpoint is invalid after boundary allocation failure"
        )
    if getattr(req, "mamba_last_track_seqlen", None) is None:
        # No page checkpoint has been established on this D generation.  The
        # only provable state is the exact active state (accepted only for an
        # aligned token count by snapshot_token_count_for_req()).
        return getattr(req, "mamba_pool_idx", None)
    tracked = int(req.mamba_last_track_seqlen)
    buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
    next_index = getattr(req, "mamba_next_track_idx", None)
    if buffer is not None and next_index is not None and len(buffer):
        next_index = int(next_index)
        other_index = 1 - next_index if len(buffer) == 2 else next_index
        other_slot = int(buffer[other_index])
        keep_index = next_index if other_slot == -1 else other_index
        if checkpoint_tokens is not None and int(checkpoint_tokens) != tracked:
            if page_size is None:
                raise RuntimeError(
                    "selecting an older Mamba checkpoint requires page_size"
                )
            if other_slot == -1:
                raise RuntimeError(
                    "requested Mamba checkpoint has no retained physical slot: "
                    f"tracked={tracked} requested={checkpoint_tokens}"
                )
            if (
                len(buffer) != 2
                or int(checkpoint_tokens) != tracked - int(page_size)
            ):
                raise RuntimeError(
                    "requested Mamba checkpoint is not retained: "
                    f"tracked={tracked} requested={checkpoint_tokens} "
                    f"page_size={page_size}"
                )
            # In normal two-slot mode the latest checkpoint is opposite the
            # next write index, so the next index still holds the immediately
            # preceding page checkpoint.  Lazy mode has no retained previous
            # slot and therefore fails closed below.
            keep_index = next_index
        slot = int(buffer[keep_index])
        if slot >= 0:
            return slot
        if checkpoint_tokens is not None:
            raise RuntimeError(
                "requested Mamba checkpoint has no retained physical slot: "
                f"tracked={tracked} requested={checkpoint_tokens}"
            )
    if checkpoint_tokens is not None and int(checkpoint_tokens) != tracked:
        raise RuntimeError(
            "requested Mamba checkpoint has no retained ping-pong slot: "
            f"tracked={tracked} requested={checkpoint_tokens}"
        )
    return getattr(req, "mamba_pool_idx", None)


def p2d_mamba_source_indices(
    req, page_size: int, *, preserve_checkpoint: bool = True
) -> list[np.ndarray]:
    """Return active plus the latest page checkpoint for native P->D.

    Decode needs the active state to continue generation and the checkpoint to
    preserve the page-tail recompute bound if a short Decode ends before the
    next tracking boundary.  For a sub-page prompt, the second slot is an
    intentionally unused duplicate; Decode keeps ``mamba_last_track_seqlen``
    unset until it crosses its first page boundary.
    """

    active = getattr(req, "mamba_pool_idx", None)
    if active is None:
        raise RuntimeError("P->D Mamba transfer is missing active state")
    active = int(active)
    if not preserve_checkpoint:
        return [np.asarray([active], dtype=np.int32)]
    expected_checkpoint = p2d_mamba_checkpoint_tokens(req, page_size)
    if expected_checkpoint == 0:
        checkpoint = active
    else:
        frozen_checkpoint = getattr(req, "_agentic_p2d_mamba_checkpoint_index", None)
        frozen_tokens = getattr(req, "_agentic_p2d_mamba_checkpoint_tokens", None)
        if frozen_checkpoint is not None:
            if frozen_tokens is None or int(frozen_tokens) != expected_checkpoint:
                raise RuntimeError(
                    "P->D frozen Mamba checkpoint boundary mismatch: "
                    f"frozen={frozen_tokens} expected={expected_checkpoint}"
                )
            checkpoint = int(frozen_checkpoint)
            return [np.asarray([active, checkpoint], dtype=np.int32)]
        tracked = getattr(req, "mamba_last_track_seqlen", None)
        if tracked is None or int(tracked) != expected_checkpoint:
            raise RuntimeError(
                "P->D Mamba transfer is missing the latest page checkpoint: "
                f"tracked={tracked} expected={expected_checkpoint}"
            )
        checkpoint = mamba_checkpoint_index_for_req(req)
        if checkpoint is None:
            raise RuntimeError("P->D Mamba checkpoint has no physical state slot")
        checkpoint = int(checkpoint)
    return [np.asarray([active, checkpoint], dtype=np.int32)]


def freeze_p2d_mamba_checkpoint_after_cache(
    req, tracked_tokens, page_size: int
) -> None:
    """Pin the post-``cache_unfinished`` Radix checkpoint for P->D.

    Native Mamba caching donates the ping-pong checkpoint to Radix and clears
    ``mamba_last_track_seqlen``.  The request's locked ``last_node`` is then
    the authoritative physical owner until P->D completion.  Freeze that
    index and its logical boundary instead of reading the replacement,
    uninitialized ping-pong slot later.
    """

    expected = p2d_mamba_checkpoint_tokens(req, page_size)
    if expected == 0:
        req._agentic_p2d_mamba_checkpoint_index = None
        req._agentic_p2d_mamba_checkpoint_tokens = None
        return
    if tracked_tokens is None or int(tracked_tokens) != expected:
        raise RuntimeError(
            "Prefill did not produce the expected page Mamba checkpoint: "
            f"tracked={tracked_tokens} expected={expected} "
            f"origin={len(req.origin_input_ids)} "
            f"fill={getattr(req, 'fill_len', None)} "
            f"prefix={len(getattr(req, 'prefix_indices', ())) } "
            f"protected={getattr(req, 'cache_protected_len', None)} "
            f"committed={getattr(req, 'kv_committed_len', None)} "
            f"allocated={getattr(req, 'kv_allocated_len', None)} "
            f"extend={getattr(req, 'extend_input_len', None)} "
            f"inflight_chunks={getattr(req, 'inflight_middle_chunks', None)}"
        )
    if prompt_checkpoint_enabled() and int(getattr(req, "cache_protected_len", -1)) != expected:
        raise RuntimeError("stable Prefill checkpoint is not the locked prefix boundary")
    node = getattr(req, "last_node", None)
    value = None if node is None else getattr(node, "mamba_value", None)
    if value is None or torch.as_tensor(value).numel() != 1:
        raise RuntimeError("cached Prefill prefix has no unique Mamba checkpoint")
    req._agentic_p2d_mamba_checkpoint_index = int(torch.as_tensor(value).item())
    req._agentic_p2d_mamba_checkpoint_tokens = expected


def p2d_mamba_destination_indices(
    req, req_pool, page_size: int, *, preserve_checkpoint: bool = True
) -> list[np.ndarray]:
    """Reserve active/checkpoint destinations and restore D tracking metadata."""

    active = getattr(req, "mamba_pool_idx", None)
    if not preserve_checkpoint:
        if active is None:
            raise RuntimeError("D P->D workset is missing active Mamba state")
        return [np.asarray([int(active)], dtype=np.int32)]
    buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
    next_index = getattr(req, "mamba_next_track_idx", None)
    if active is None or buffer is None or next_index is None or not len(buffer):
        raise RuntimeError("D P->D workset is missing active/checkpoint Mamba slots")
    keep_index = req_pool.get_mamba_ping_pong_keep_idx(req)
    checkpoint = int(buffer[int(keep_index)])
    if checkpoint < 0:
        raise RuntimeError("D P->D workset has no allocated checkpoint slot")
    tracked = p2d_mamba_checkpoint_tokens(req, page_size)
    req.mamba_last_track_seqlen = tracked if tracked > 0 else None
    if prompt_checkpoint_enabled():
        # Existing imported tracking slot becomes immutable. No new allocation
        # or ownership class: native request free still owns both buffers.
        req._agentic_mamba_frozen_prompt_tokens = tracked
        req._agentic_mamba_frozen_prompt_valid = True
    return [np.asarray([int(active), checkpoint], dtype=np.int32)]


def snapshot_token_count_for_req(
    req, requested_tokens: int, state_types: Sequence[StateType], page_size: int
) -> int:
    """Choose the largest Attention prefix with a matching hybrid state.

    A reverse snapshot may omit at most one page tail; P recomputes that tail.
    Pairing newer Mamba state with older page-aligned Attention KV is invalid.
    """

    requested_tokens = int(requested_tokens)
    if StateType.MAMBA not in state_types:
        return requested_tokens
    frozen = getattr(req, "_agentic_mamba_frozen_prompt_tokens", None)
    if frozen is not None:
        if not getattr(req, "_agentic_mamba_frozen_prompt_valid", False):
            raise RuntimeError("stable Mamba checkpoint lost during retraction")
        if int(frozen) > requested_tokens:
            raise RuntimeError("stable Mamba checkpoint exceeds committed Attention")
        if int(frozen) == 0:
            return 0
        if getattr(req, "mamba_last_track_seqlen", None) != int(frozen):
            raise RuntimeError("stable Mamba checkpoint boundary was overwritten")
        mamba_checkpoint_index_for_req(
            req, checkpoint_tokens=int(frozen), page_size=page_size
        )
        return int(frozen)
    if getattr(req, "mamba_lazy_is_insert", None) is False:
        raise RuntimeError(
            "lazy Mamba checkpoint is invalid after boundary allocation failure"
        )
    tracked = getattr(req, "mamba_last_track_seqlen", None)
    if tracked is None:
        # Native P->D imports only the current active state.  It is a valid
        # reverse checkpoint only when Attention ends at that exact page
        # boundary.  Never pair it with floor(requested/page) Attention.
        if requested_tokens % page_size:
            raise RuntimeError(
                "no page-boundary Mamba checkpoint for an unaligned snapshot: "
                f"requested={requested_tokens} page_size={page_size}"
            )
        if getattr(req, "mamba_pool_idx", None) is None:
            raise RuntimeError("aligned Mamba snapshot has no active state")
        return requested_tokens
    tracked = int(tracked)
    if tracked > requested_tokens:
        # Decode accounts a sampled stop token in its committed recurrent
        # state, while the reusable logical prefix intentionally excludes that
        # token.  If the stop lands exactly on a tracking boundary, retain the
        # preceding ping-pong checkpoint and let P recompute at most one page.
        previous = tracked - int(page_size)
        if tracked == requested_tokens + 1 and previous >= 0:
            if previous > 0:
                # Validate now, before publishing a Direct/Host candidate.
                # Lazy mode does not retain this older checkpoint and must
                # fall back to full Prefill instead of failing asynchronously.
                mamba_checkpoint_index_for_req(
                    req,
                    checkpoint_tokens=previous,
                    page_size=page_size,
                )
            return previous
        raise RuntimeError(
            "invalid Mamba checkpoint boundary: "
            f"tracked={tracked} requested={requested_tokens} page_size={page_size}"
        )
    if tracked <= 0 or tracked % page_size:
        raise RuntimeError(
            "invalid Mamba checkpoint boundary: "
            f"tracked={tracked} requested={requested_tokens} page_size={page_size}"
        )
    return tracked


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
    page_indices = np.asarray(lease.parent_page_indices, dtype=np.int32)
    receiver.send_metadata(
        page_indices,
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
    page_size = int(getattr(kv_args, "page_size", 1) or 1)
    if int(token_count) % page_size:
        raise ValueError(
            f"snapshot token count must be page aligned: "
            f"tokens={token_count} page_size={page_size}"
        )
    # v0.5.14 NIXL descriptors expose kv_item_lens per physical page, not
    # per token.  Reverse snapshots name page indices on the wire.
    attention = (int(token_count) // page_size) * sum(
        int(item_len) for item_len in getattr(kv_args, "kv_item_lens", ())
    )
    state = state_payload_bytes(kv_args)
    if not state:
        return attention
    attention = (
        (attention + mmap.ALLOCATIONGRANULARITY - 1)
        // mmap.ALLOCATIONGRANULARITY
        * mmap.ALLOCATIONGRANULARITY
    )
    return attention + state
