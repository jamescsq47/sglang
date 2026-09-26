"""Short native Req/Radix transactions for multi-node V2.

The data-plane controller owns physical leases through
``AgenticMemoryAuthority``.  This module is the only model-facing adapter that
turns a committed lease into SGLang request metadata.  It contains no routing,
transport polling, filesystem control, or free list.

Two ownership rules are intentionally explicit:

* Radix insertion may deduplicate and free part of the newly allocated tensor.
  After insertion, cleanup therefore goes through the live Req/tree API and
  never frees the original lease indices again.
* Qwen3.5 recurrent slots are split into an imported checkpoint and
  request-owned runtime slots.  The adapter records that split once and the
  ordinary request cleanup remains the sole owner after handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Optional, Sequence

import numpy as np

from sglang.srt.disaggregation.agentic_memory_authority import PhysicalMemoryLease
from sglang.srt.disaggregation.agentic_hybrid_transfer import (
    request_owned_mamba_enabled,
)


@dataclass(frozen=True, slots=True)
class NativeRequestBinding:
    req_pool_idx: Optional[int]
    committed_tokens: int
    radix_parent_tokens: int
    temporary_pin: Any = None
    state_runtime_indices: Any = None


@dataclass(frozen=True, slots=True)
class NativeSourceSnapshot:
    token_count: int
    token_indices: Any
    page_indices: tuple[int, ...]
    state_indices: tuple[int, ...]
    page_chain_hashes: tuple[str, ...]


def token_page_chain_hashes(
    token_ids: Sequence[int], page_size: int
) -> tuple[str, ...]:
    """Cumulative logical-token proof at every complete page boundary."""

    page_size = int(page_size)
    if page_size <= 0 or len(token_ids) % page_size:
        raise ValueError("token chain requires complete pages")
    digest = hashlib.sha256()
    result = []
    for offset in range(0, len(token_ids), page_size):
        page = token_ids[offset : offset + page_size]
        for token in page:
            digest.update(int(token).to_bytes(8, "little", signed=True))
        result.append(digest.hexdigest())
    return tuple(result)


def common_page_prefix_tokens(
    source_hashes: Sequence[str], child_token_ids: Sequence[int], page_size: int
) -> int:
    child_pages = len(child_token_ids) // int(page_size) * int(page_size)
    child_hashes = token_page_chain_hashes(
        child_token_ids[:child_pages], int(page_size)
    )
    common = 0
    for source, child in zip(source_hashes, child_hashes):
        if source != child:
            break
        common += 1
    return common * int(page_size)


def hybrid_state_allocator(scheduler: Any) -> tuple[Any, ...]:
    pool = getattr(getattr(scheduler, "req_to_token_pool", None), "mamba_pool", None)
    return () if pool is None else (pool,)


def prefill_state_slot_count(scheduler: Any, *, imported_parent: bool) -> int:
    """Complete Qwen3.5 P workset state footprint.

    Runtime owns one active slot and, when enabled, the native ping-pong
    tracking slots.  Request-owned Prefill additionally needs the output
    checkpoint slot.  A restored parent contributes one imported checkpoint.
    """

    req_pool = getattr(scheduler, "req_to_token_pool", None)
    if getattr(req_pool, "mamba_pool", None) is None:
        return 0
    count = 1
    if bool(getattr(req_pool, "enable_mamba_extra_buffer", False)):
        count += int(req_pool.mamba_ping_pong_track_buffer_size)
    from sglang.srt.disaggregation.agentic_mamba_prefill import (
        prefill_state_admission_enabled,
    )

    count += int(prefill_state_admission_enabled(scheduler.server_args, req_pool))
    return count + int(imported_parent)


def decode_state_slot_count(scheduler: Any) -> int:
    req_pool = getattr(scheduler, "req_to_token_pool", None)
    if getattr(req_pool, "mamba_pool", None) is None:
        return 0
    count = 1
    if bool(getattr(req_pool, "enable_mamba_extra_buffer", False)):
        count += int(req_pool.mamba_ping_pong_track_buffer_size)
    return count


class NativeRequestMemoryAdapter:
    """Bind/release one local shard while authority.native_guard is held."""

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler

    @property
    def hybrid(self) -> bool:
        return bool(hybrid_state_allocator(self.scheduler))

    @staticmethod
    def _flat_state(lease: PhysicalMemoryLease):
        if not lease.state_indices:
            return None
        if len(lease.state_indices) != 1:
            raise RuntimeError("V2 supports exactly one recurrent state allocator")
        return lease.state_indices[0]

    def source_snapshot(self, req: Any, *, direction: str) -> NativeSourceSnapshot:
        """Freeze the current live Req mapping after compute completion.

        This deliberately does not use ``lease.device_indices``.  Decode may
        have materialized growth pages after import, while a P-side Radix bind
        may have deduplicated and freed some originally reserved parent pages.
        The request-to-token row is the authoritative current snapshot.
        """

        if direction not in {"d2p", "p2d"}:
            raise ValueError("direction must be d2p or p2d")
        page_size = int(self.scheduler.token_to_kv_pool_allocator.page_size)
        committed = int(
            getattr(req, "kv_committed_len", len(getattr(req, "fill_ids", ())))
        )
        if direction == "d2p":
            logical = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            committed = min(committed, logical)
            if self.hybrid:
                from sglang.srt.disaggregation.agentic_hybrid_transfer import (
                    StateType,
                    snapshot_token_count_for_req,
                )

                committed = snapshot_token_count_for_req(
                    req, committed, (StateType.MAMBA,), page_size
                )
            else:
                committed = committed // page_size * page_size
        else:
            committed = committed // page_size * page_size
        if committed <= 0:
            raise RuntimeError("computed request has no page-aligned KV snapshot")
        if getattr(req, "req_pool_idx", None) is None:
            raise RuntimeError("computed request has no request-to-token row")
        token_indices = self.scheduler.req_to_token_pool.req_to_token[
            req.req_pool_idx, :committed
        ].clone()
        from sglang.srt.disaggregation.utils import kv_to_page_indices

        values = token_indices.detach().cpu().numpy()
        pages = tuple(
            int(value) for value in kv_to_page_indices(values, page_size).tolist()
        )
        states: tuple[int, ...] = ()
        if self.hybrid:
            if direction == "d2p":
                from sglang.srt.disaggregation.agentic_hybrid_transfer import (
                    StateType,
                    state_indices_for_req,
                )

                raw = state_indices_for_req(
                    req,
                    (StateType.MAMBA,),
                    checkpoint_tokens=committed,
                    page_size=page_size,
                )[0]
            else:
                from sglang.srt.disaggregation.agentic_hybrid_transfer import (
                    p2d_mamba_source_indices,
                )

                raw = p2d_mamba_source_indices(req, page_size)[0]
            # Native hybrid helpers use a state-type-parallel nested layout
            # (for Mamba: ``[[np.asarray(slot)]]``), while the P2D helper
            # already returns an ndarray.  Normalize both representations at
            # this adapter boundary before serializing the immutable shard.
            states = tuple(
                int(value)
                for value in np.asarray(raw, dtype=np.int32).reshape(-1).tolist()
            )
        if direction == "d2p":
            logical_tokens = (
                list(req.origin_input_ids) + list(req.output_ids[:-1])
            )[:committed]
        else:
            logical_tokens = list(getattr(req, "fill_ids", ()))[:committed]
        return NativeSourceSnapshot(
            committed,
            token_indices,
            pages,
            states,
            token_page_chain_hashes(logical_tokens, page_size),
        )

    def _attach_prefill_runtime_state(
        self, req: Any, lease: PhysicalMemoryLease, *, imported_parent: bool
    ) -> tuple[Any, Any]:
        state = self._flat_state(lease)
        if state is None:
            return None, None
        expected = prefill_state_slot_count(
            self.scheduler, imported_parent=imported_parent
        )
        if len(state) != expected:
            raise RuntimeError(
                "Prefill state lease has the wrong number of slots: "
                f"reserved={len(state)} expected={expected}"
            )
        if getattr(req, "mamba_pool_idx", None) is not None:
            raise RuntimeError("Prefill request already owns recurrent state")
        cursor = 0
        parent_checkpoint = None
        if imported_parent:
            parent_checkpoint = state[cursor : cursor + 1]
            cursor += 1
        req.mamba_pool_idx = state[cursor]
        cursor += 1
        req.mamba_needs_clear = False
        req_pool = self.scheduler.req_to_token_pool
        if bool(getattr(req_pool, "enable_mamba_extra_buffer", False)):
            size = int(req_pool.mamba_ping_pong_track_buffer_size)
            req.mamba_ping_pong_track_buffer = state[cursor : cursor + size]
            req.mamba_next_track_idx = 0
            cursor += size
        remaining = state[cursor:]
        if len(remaining):
            req._agentic_mamba_prefill_checkpoint = remaining
        req._agentic_mamba_runtime_reserved = True
        if imported_parent:
            req.mamba_last_track_seqlen = int(lease.parent_tokens)
        return parent_checkpoint, state[cursor:]

    def bind_prefill_parent(
        self, lease: PhysicalMemoryLease, req: Any
    ) -> NativeRequestBinding:
        """Bind a restored parent, or initialize an all-new P workset.

        The method must execute inside ``authority.native_guard``.  It may
        mutate Radix/request metadata but does no I/O and launches no Forward.
        """

        parent_count = int(lease.parent_tokens)
        imported_parent = parent_count > 0
        # Validate every condition that can be checked without mutating native
        # ownership before attaching recurrent slots or inserting Radix KV.
        if imported_parent and parent_count > len(req.origin_input_ids):
            raise ValueError("restored parent exceeds child prompt")
        if imported_parent and bool(
            getattr(self.scheduler.tree_cache, "disable", False)
        ):
            raise RuntimeError("restored parent requires native Radix cache")
        parent_checkpoint, runtime_state = self._attach_prefill_runtime_state(
            req, lease, imported_parent=imported_parent
        )
        req._agentic_v2_prefill_state_attached = bool(lease.state_indices)
        if parent_checkpoint is not None:
            # The imported checkpoint is deliberately not attached to the
            # live Req.  Until Radix insert commits, it remains a private part
            # of the authority lease and needs its own rollback identity.
            req._agentic_v2_prefill_parent_checkpoint = parent_checkpoint
        pin = None
        if imported_parent:
            from sglang.srt.mem_cache.base_prefix_cache import (
                InsertParams,
                MatchPrefixParams,
            )
            from sglang.srt.mem_cache.radix_cache import RadixKey

            tokens = list(req.origin_input_ids[:parent_count])
            # Radix insertion may deduplicate/free pages while walking the
            # tree.  Mark that narrow operation explicitly: if native insert
            # itself raises, ownership is unknowable and release must fail
            # closed (quarantine the lease) rather than guess and double-free.
            req._agentic_v2_prefill_parent_phase = "insert_inflight"
            result = self.scheduler.tree_cache.insert(
                InsertParams(
                    key=RadixKey(tokens, req.extra_key),
                    value=lease.parent_indices[:parent_count],
                    mamba_value=(
                        None
                        if parent_checkpoint is None
                        else parent_checkpoint.clone()
                    ),
                    priority=getattr(req, "priority", 0) or 0,
                )
            )
            # From this point the parent is owned by native Radix (possibly
            # deduplicated against an existing branch), not by the raw lease.
            req._agentic_v2_prefill_parent_inserted = parent_count
            req._agentic_v2_prefill_parent_phase = "radix"
            # Plain Radix insertion does not consume/free the duplicate input
            # prefix; MambaRadix does so internally because checkpoint
            # ownership is coupled to the same walk.
            if not self.hybrid and int(result.prefix_len) > 0:
                self.scheduler.token_to_kv_pool_allocator.free(
                    lease.parent_indices[: int(result.prefix_len)]
                )
            # Complete checkpoint ownership before match/pin, which can fail
            # independently.  A new checkpoint is now Radix-owned; a
            # duplicate remains private and is returned immediately.
            if parent_checkpoint is not None and bool(result.mamba_exist):
                self.scheduler.req_to_token_pool.mamba_pool.free(parent_checkpoint)
            if hasattr(req, "_agentic_v2_prefill_parent_checkpoint"):
                delattr(req, "_agentic_v2_prefill_parent_checkpoint")
            match = self.scheduler.tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(tokens, req.extra_key),
                    req=req,
                )
            )
            if len(match.device_indices) != parent_count:
                raise RuntimeError("restored parent disappeared during bind")
            pin = match.last_device_node
            self.scheduler.tree_cache.inc_lock_ref(pin)
            req._agentic_direct_parent_pin_node = pin
            req._agentic_direct_parent_token_count = parent_count
        return NativeRequestBinding(
            req_pool_idx=None,
            committed_tokens=parent_count,
            radix_parent_tokens=parent_count,
            temporary_pin=pin,
            state_runtime_indices=runtime_state,
        )

    def bind_decode_prompt(
        self,
        lease: PhysicalMemoryLease,
        req: Any,
        *,
        mamba_checkpoint_tokens: Optional[int] = None,
    ) -> NativeRequestBinding:
        """Attach an imported complete prompt to the native Decode request."""

        if getattr(req, "req_pool_idx", None) is not None:
            raise RuntimeError("Decode request already owns a request-pool row")
        state = self._flat_state(lease)
        if state is not None:
            expected = decode_state_slot_count(self.scheduler)
            if len(state) != expected:
                raise RuntimeError(
                    "Decode state lease has the wrong number of slots: "
                    f"reserved={len(state)} expected={expected}"
                )
            # HybridReqToTokenPool.alloc() writes Req state indices into its
            # row mappings.  Attach the authority-owned state first so alloc
            # only allocates the Req row and records these exact slots; doing
            # this afterwards leaks a second implicit state allocation and
            # leaves Forward reading the wrong mapping.
            cursor = 0
            req.mamba_pool_idx = state[cursor]
            cursor += 1
            req.mamba_needs_clear = False
            req_pool = self.scheduler.req_to_token_pool
            if bool(getattr(req_pool, "enable_mamba_extra_buffer", False)):
                size = int(req_pool.mamba_ping_pong_track_buffer_size)
                req.mamba_ping_pong_track_buffer = state[cursor : cursor + size]
                req.mamba_next_track_idx = 0
                cursor += size
            if cursor != len(state):
                raise RuntimeError("Decode state lease has unused slots")
            req._agentic_v2_decode_state_attached = True
            req._agentic_mamba_runtime_reserved = True
            if request_owned_mamba_enabled():
                if mamba_checkpoint_tokens is None:
                    raise RuntimeError(
                        "request-owned Mamba decode bind is missing its checkpoint boundary"
                    )
                boundary = int(mamba_checkpoint_tokens)
                if boundary < 0 or boundary > int(lease.prompt_tokens):
                    raise RuntimeError(
                        "request-owned Mamba checkpoint is outside the imported prompt: "
                        f"checkpoint={boundary} prompt={lease.prompt_tokens}"
                    )
                page_size = int(self.scheduler.page_size)
                if boundary % page_size:
                    raise RuntimeError(
                        "request-owned Mamba checkpoint is not page aligned: "
                        f"checkpoint={boundary} page_size={page_size}"
                    )
                req._agentic_mamba_frozen_prompt_tokens = boundary
                req._agentic_mamba_frozen_prompt_valid = True
                req.mamba_last_track_seqlen = boundary if boundary else None
        indices = self.scheduler.req_to_token_pool.alloc([req])
        if indices is None or len(indices) != 1:
            raise MemoryError("request-to-token pool is full")
        logical = int(lease.prompt_tokens)
        try:
            self.scheduler.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, logical)),
                lease.device_indices[:logical],
            )
            req.kv_allocated_len = logical
            req.kv_committed_len = logical
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.set_extend_input_len(len(req.fill_ids))
        except BaseException:
            self.scheduler.req_to_token_pool.free(req)
            req.req_pool_idx = None
            raise
        return NativeRequestBinding(
            req_pool_idx=int(req.req_pool_idx),
            committed_tokens=logical,
            radix_parent_tokens=0,
            state_runtime_indices=state,
        )

    def _free_raw_state(self, lease: PhysicalMemoryLease) -> None:
        pools = hybrid_state_allocator(self.scheduler)
        if len(pools) != len(lease.state_indices):
            raise RuntimeError("lease state layout does not match native pools")
        for pool, indices in zip(pools, lease.state_indices):
            pool.free(indices)

    @staticmethod
    def _cleanup_once(req: Any, step: str, action) -> None:
        """Commit one synchronous cleanup step at most once per Req.

        Authority intentionally retains a lease when native cleanup raises so
        the same terminal fence can be retried.  Receipts prevent a later
        step's failure from replaying earlier successful frees.
        """

        receipts = getattr(req, "_agentic_v2_cleanup_receipts", None)
        if receipts is None:
            receipts = set()
            req._agentic_v2_cleanup_receipts = receipts
        if step in receipts:
            return
        action()
        receipts.add(step)

    @staticmethod
    def _clear_transfer_attrs(req: Any) -> None:
        for name in (
            "_agentic_direct_parent_pin_node",
            "_agentic_direct_parent_token_count",
            "_agentic_v2_prefill_parent_inserted",
            "_agentic_v2_prefill_parent_phase",
            "_agentic_v2_prefill_parent_checkpoint",
            "_agentic_v2_prefill_state_attached",
            "_agentic_v2_decode_state_attached",
            "_agentic_v2_cleanup_receipts",
        ):
            if hasattr(req, name):
                delattr(req, name)

    def release_prefill_unadopted(
        self,
        lease: PhysicalMemoryLease,
        req: Any,
        _binding: Optional[NativeRequestBinding],
    ) -> None:
        """Rollback a P workset that never entered native scheduling."""

        from sglang.srt.mem_cache.common import release_kv_cache

        inserted = int(
            getattr(req, "_agentic_v2_prefill_parent_inserted", 0)
        )
        state_attached = bool(
            getattr(req, "_agentic_v2_prefill_state_attached", False)
        )
        parent_phase = getattr(req, "_agentic_v2_prefill_parent_phase", None)
        if parent_phase == "insert_inflight":
            raise RuntimeError(
                "native Radix insert failed with ambiguous ownership; "
                "the workset remains quarantined"
            )
        pin = getattr(req, "_agentic_direct_parent_pin_node", None)
        if pin is not None:
            self._cleanup_once(
                req,
                "prefill_unadopted_unpin",
                lambda: self.scheduler.tree_cache.dec_lock_ref(pin),
            )
            delattr(req, "_agentic_direct_parent_pin_node")

        if inserted:
            release = getattr(
                self.scheduler.tree_cache,
                "release_agentic_request_cache",
                None,
            )
            if not callable(release):
                raise RuntimeError("native Radix rollback API is unavailable")
            self._cleanup_once(
                req,
                "prefill_unadopted_radix",
                lambda: release(
                    req,
                    committed_len=inserted,
                    _defer_if_blocked=False,
                ),
            )
            # Parent pages moved into/deduplicated through Radix.  Only the
            # never-scheduled suffix and page padding remain private.
            if len(lease.suffix_indices):
                self._cleanup_once(
                    req,
                    "prefill_unadopted_suffix",
                    lambda: self.scheduler.token_to_kv_pool_allocator.free(
                        lease.suffix_indices
                    ),
                )
            if lease.parent_allocated_tokens > inserted:
                self._cleanup_once(
                    req,
                    "prefill_unadopted_parent_padding",
                    lambda: self.scheduler.token_to_kv_pool_allocator.free(
                        lease.parent_indices[inserted:]
                    ),
                )
        else:
            self._cleanup_once(
                req,
                "prefill_unadopted_device",
                lambda: self.scheduler.token_to_kv_pool_allocator.free(
                    lease.device_indices
                ),
            )

        if state_attached:
            parent_checkpoint = getattr(
                req, "_agentic_v2_prefill_parent_checkpoint", None
            )
            if parent_checkpoint is not None:
                self._cleanup_once(
                    req,
                    "prefill_unadopted_parent_checkpoint",
                    lambda: self.scheduler.req_to_token_pool.mamba_pool.free(
                        parent_checkpoint
                    ),
                )
                delattr(req, "_agentic_v2_prefill_parent_checkpoint")
            # req_pool_idx is intentionally absent; this path only retires
            # request-owned recurrent aliases and the Prefill checkpoint.
            self._cleanup_once(
                req,
                "prefill_unadopted_runtime_state",
                lambda: release_kv_cache(
                    req, self.scheduler.tree_cache, is_insert=False
                ),
            )
        elif lease.state_indices:
            self._cleanup_once(
                req,
                "prefill_unadopted_raw_state",
                lambda: self._free_raw_state(lease),
            )
        self._clear_transfer_attrs(req)

    def release_decode_unadopted(
        self,
        lease: PhysicalMemoryLease,
        req: Any,
        _binding: Optional[NativeRequestBinding],
    ) -> None:
        """Rollback a D lease when native prompt binding did not complete."""

        from sglang.srt.mem_cache.common import release_kv_cache

        if getattr(req, "req_pool_idx", None) is not None:
            self._cleanup_once(
                req,
                "decode_unadopted_native",
                lambda: release_kv_cache(
                    req, self.scheduler.tree_cache, is_insert=False
                ),
            )
        else:
            self._cleanup_once(
                req,
                "decode_unadopted_device",
                lambda: self.scheduler.token_to_kv_pool_allocator.free(
                    lease.device_indices
                ),
            )
            if bool(getattr(req, "_agentic_v2_decode_state_attached", False)):
                self._cleanup_once(
                    req,
                    "decode_unadopted_attached_state",
                    lambda: release_kv_cache(
                        req, self.scheduler.tree_cache, is_insert=False
                    ),
                )
            elif lease.state_indices:
                self._cleanup_once(
                    req,
                    "decode_unadopted_raw_state",
                    lambda: self._free_raw_state(lease),
                )
        self._clear_transfer_attrs(req)

    def release_bound(
        self,
        lease: PhysicalMemoryLease,
        req: Any,
        binding: NativeRequestBinding,
    ) -> None:
        """Release through native ownership; never raw-free donated indices."""

        from sglang.srt.mem_cache.common import release_kv_cache

        committed_len = int(
            getattr(
                req,
                "kv_committed_len",
                lease.prompt_tokens
                if binding is None
                else binding.committed_tokens,
            )
        )
        pin = getattr(req, "_agentic_direct_parent_pin_node", None)
        if pin is not None:
            self.scheduler.tree_cache.dec_lock_ref(pin)
            delattr(req, "_agentic_direct_parent_pin_node")
        # A scheduler-adopted workset can still abort before consuming all of
        # its pre-reserved suffix.  Those pages never entered the Req row and
        # therefore are not visible to native release_kv_cache().
        remaining_suffix = getattr(
            req, "_agentic_workset_suffix_indices", None
        )
        if remaining_suffix is not None and len(remaining_suffix):
            self.scheduler.token_to_kv_pool_allocator.free(remaining_suffix)
            delattr(req, "_agentic_workset_suffix_indices")
        # Native cleanup releases the Req row, request-owned recurrent state,
        # private suffix/over-allocation and the now-unlocked Radix branch in
        # the order required by both RadixCache and MambaRadixCache.
        release_kv_cache(req, self.scheduler.tree_cache, is_insert=False)
        release = getattr(
            self.scheduler.tree_cache, "release_agentic_request_cache", None
        )
        if callable(release) and committed_len:
            release(
                req,
                committed_len=committed_len,
                _defer_if_blocked=False,
            )
        self._clear_transfer_attrs(req)


__all__ = [
    "NativeRequestBinding",
    "NativeRequestMemoryAdapter",
    "NativeSourceSnapshot",
    "decode_state_slot_count",
    "common_page_prefix_tokens",
    "hybrid_state_allocator",
    "prefill_state_slot_count",
    "token_page_chain_hashes",
]
