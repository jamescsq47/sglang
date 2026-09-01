"""Scheduler-owned P workset leases for agentic reverse-KV restore.

Kept outside Scheduler so the ownership state machine can be tested and
ported across SGLang releases without copying an older scheduler.
"""

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.common import kv_to_page_indices




@dataclass
class AgenticPWorksetLease:
    """Physical P-HBM ownership for one complete next-turn prompt.

    The parent slice is the destination of Direct or Slow restore.  The suffix
    slice stays unavailable to every other request until this exact request is
    admitted for incremental Prefill.  All allocator mutations are performed
    by the model scheduler; I/O workers consume only immutable page ids.
    """

    snapshot_id: str
    lease_id: int
    owner: str
    parent_tokens: int
    parent_allocated_tokens: int
    prompt_tokens: int
    allocated_tokens: int
    device_indices: torch.Tensor
    parent_page_indices: np.ndarray
    # One one-slot allocation for each non-attention state component.  Qwen3.5
    # currently contributes exactly one MAMBA component (temporal + conv
    # tensors share the same slot index on the wire).
    state_device_indices: Tuple[torch.Tensor, ...] = ()
    parent_bound: bool = False
    state: str = "active"
    suffix_cursor: int = 0
    io_attempt: Optional[str] = None

    @property
    def parent_indices(self) -> torch.Tensor:
        return self.device_indices[: self.parent_allocated_tokens]

    @property
    def suffix_indices(self) -> torch.Tensor:
        return self.device_indices[self.parent_allocated_tokens :]

    @property
    def suffix_allocated_tokens(self) -> int:
        return self.allocated_tokens - self.parent_allocated_tokens

    @property
    def remaining_suffix_indices(self) -> torch.Tensor:
        if self.state == "consumed":
            return self.suffix_indices[:0]
        return self.suffix_indices[self.suffix_cursor :]


class AgenticPWorksetLeaseBroker:
    """Thread-safe intent queue with scheduler-owned physical allocation."""

    def __init__(self, page_size: int, *, state_allocators: Sequence = ()):
        self.page_size = int(page_size)
        self._state_allocators = tuple(state_allocators)
        self._intents: Dict[str, Tuple[str, int, int]] = {}
        self._leases: Dict[str, AgenticPWorksetLease] = {}
        self._release_requested: Dict[str, int] = {}
        self._grant_events: Deque[str] = deque()
        self._next_lease_id = 1
        self._grants = 0
        self._allocation_failures = 0
        # TP ranks must allocate the same request-generation worksets in the
        # same order.  A scheduler-broadcast plan is frozen for one native TP
        # control epoch; background Direct/Slow workers may publish intents or
        # request cancellation, but cannot mutate pages named by that epoch.
        # TP=1 never installs a plan and retains the original eager behavior.
        self._tp_plan_epoch = -1
        self._tp_plan: Tuple[Tuple[str, str, int, int], ...] = ()
        self._tp_authoritative_retirements: frozenset[str] = frozenset()
        self._tp_cancel_pending: Dict[str, Optional[str]] = {}
        self._tp_release_pending: Dict[str, int] = {}
        self._tp_retire_requested: set[str] = set()
        self._tp_retired_in_epoch: set[str] = set()
        # A final suffix consume transfers page ownership from this broker to
        # the native Req; it is not an allocator release.  TP0 publishes that
        # ownership commit so follower ranks can drop only their broker
        # metadata without freeing the KV pages now owned by the same Req.
        self._tp_handoff_committed: set[str] = set()
        # Monotonic lifecycle boundaries can make one asynchronous producer
        # obsolete before its marker is observed.  Remember that fact so a
        # late marker cannot recreate the cancelled owner.
        self._superseded_owners: set[tuple[str, str]] = set()
        self._lock = threading.RLock()

    @staticmethod
    def direct_owner(snapshot_id: str) -> str:
        return f"direct:{snapshot_id}"

    @staticmethod
    def slow_owner(snapshot_id: str, rid: str) -> str:
        return f"slow:{snapshot_id}:{rid}"

    def request(
        self,
        snapshot_id: str,
        parent_tokens: int,
        prompt_tokens: int,
        *,
        owner: str = "legacy",
    ) -> bool:
        parent_tokens = int(parent_tokens)
        prompt_tokens = int(prompt_tokens)
        if parent_tokens <= 0 or prompt_tokens < parent_tokens:
            raise ValueError(
                f"invalid workset shape parent={parent_tokens} prompt={prompt_tokens}"
            )
        with self._lock:
            if (snapshot_id, owner) in self._superseded_owners:
                return False
            current = self._leases.get(snapshot_id)
            if current is not None:
                return (
                    current.owner == owner
                    and current.parent_tokens == parent_tokens
                    and current.prompt_tokens == prompt_tokens
                    and current.state not in {"releasing", "consumed"}
                )
            pending = self._intents.get(snapshot_id)
            if pending is not None:
                return pending == (owner, parent_tokens, prompt_tokens)
            if snapshot_id in self._release_requested:
                return False
            self._intents[snapshot_id] = (owner, parent_tokens, prompt_tokens)
            return True

    def get(
        self, snapshot_id: str, *, owner: Optional[str] = None
    ) -> Optional[AgenticPWorksetLease]:
        with self._lock:
            lease = self._leases.get(snapshot_id)
            if lease is not None and (owner is None or lease.owner == owner):
                return lease
            return None

    def request_release(
        self,
        snapshot_id: str,
        lease: Optional[AgenticPWorksetLease] = None,
        *,
        owner: Optional[str] = None,
        io_attempt: Optional[str] = None,
    ) -> bool:
        """Schedule release of the exact lease owned by the caller.

        A caller without a lease identity cannot release anything; intent
        cancellation is a separate owner-scoped operation.  This prevents a
        delayed Direct callback from cancelling a newer Slow owner for the
        same request-generation.
        """

        with self._lock:
            if lease is None:
                return False
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if owner is not None and current.owner != owner:
                return False
            if (
                current.state == "active"
                and any(entry[0] == snapshot_id for entry in self._tp_plan)
            ):
                # The current TP allocation epoch is immutable.  Record the
                # exact lease terminal now; TP0 omits it from the next epoch,
                # and every rank releases at the same scheduler-safe boundary.
                # Releasing eagerly here would let the still-frozen plan
                # recreate the snapshot with a new lease id.
                self._tp_release_pending[snapshot_id] = current.lease_id
                self._tp_retire_requested.add(snapshot_id)
                return True
            pending = self._intents.get(snapshot_id)
            if pending is not None and pending[0] == current.owner:
                self._intents.pop(snapshot_id, None)
            if current.state in {"io_reserved", "io_inflight"}:
                if current.io_attempt != io_attempt:
                    return False
                current.state = "release_pending"
                if any(entry[0] == snapshot_id for entry in self._tp_plan):
                    self._tp_release_pending[snapshot_id] = current.lease_id
                    self._tp_retire_requested.add(snapshot_id)
                return False
            if current.state in {
                "release_pending",
                "binding",
                "handed",
                "consumed",
                "releasing",
            }:
                return False
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def release_handed(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        *,
        req,
    ) -> bool:
        """Release a workset after scheduler ownership was handed to ``req``.

        Transport callbacks deliberately cannot release a handed lease.  Only
        the live request that owns the suffix may return it on cancellation.
        This separates request lifetime from stale Direct/Slow completions.
        """

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None:
                # The complete suffix may already belong to the native Req.
                # Native request cleanup owns those pages; treating this as a
                # broker retirement would double-free them on a TP follower.
                return False
            if current.lease_id != lease.lease_id:
                return False
            if current.state != "handed":
                return False
            if getattr(req, "_agentic_p_workset_lease", None) is not current:
                return False
            if self._tp_plan_epoch >= 0:
                # A handed lease has deliberately left the allocator plan,
                # but cancellation still has to retire its remaining local
                # suffix on every TP rank as one group transaction.
                current.state = "retire_ready"
                self._tp_release_pending[snapshot_id] = current.lease_id
                self._tp_retire_requested.add(snapshot_id)
                return True
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def service(
        self,
        allocator,
        *,
        reserve_tokens: int = 0,
    ) -> None:
        """Allocate/free only at a scheduler-safe boundary.

        ``reserve_tokens`` protects the unfinished suffix of the one native
        chunked-Prefill request.  That request was admitted only after the
        scheduler proved that its complete prompt fitted, but its later
        chunks are allocated lazily.  Background Direct/Slow intents must not
        consume that already-promised capacity between chunks.
        """

        reserve_tokens = max(0, int(reserve_tokens))

        with self._lock:
            releases = tuple(self._release_requested.items())
            self._release_requested.clear()
            for snapshot_id, expected_id in releases:
                lease = self._leases.get(snapshot_id)
                if (
                    lease is not None
                    and lease.lease_id == expected_id
                    and lease.state == "releasing"
                ):
                    self._leases.pop(snapshot_id, None)
                    allocator.free(
                        lease.remaining_suffix_indices
                        if lease.parent_bound
                        else lease.device_indices
                    )
                    for state_allocator, state_indices in zip(
                        self._state_allocators, lease.state_device_indices
                    ):
                        state_allocator.free(state_indices)
                    if self._tp_plan_epoch >= 0:
                        self._tp_retired_in_epoch.add(snapshot_id)

            allocation_plan = self._tp_plan if self._tp_plan_epoch >= 0 else None
            if allocation_plan is None:
                candidates = tuple(
                    (snapshot_id, *intent)
                    for snapshot_id, intent in self._intents.items()
                )
            else:
                planned_ids = {entry[0] for entry in allocation_plan}
                # Rank 0 omits a cancelled/released active lease from the next
                # epoch.  Followers mirror that removal here.  I/O-owned or
                # scheduler-owned leases may never disappear by reconciliation:
                # reaching this branch would indicate an earlier TP split and
                # must fail closed before another model collective.
                for snapshot_id, lease in tuple(self._leases.items()):
                    if snapshot_id in planned_ids:
                        continue
                    if lease.state != "active":
                        raise RuntimeError(
                            "TP workset plan removed a non-cancellable lease "
                            f"{snapshot_id} state={lease.state}"
                        )
                    self._leases.pop(snapshot_id, None)
                    allocator.free(lease.device_indices)
                    for state_allocator, state_indices in zip(
                        self._state_allocators, lease.state_device_indices
                    ):
                        state_allocator.free(state_indices)
                    self._tp_release_pending.pop(snapshot_id, None)
                for snapshot_id, cancel_owner in tuple(
                    self._tp_cancel_pending.items()
                ):
                    if snapshot_id not in planned_ids:
                        pending = self._intents.get(snapshot_id)
                        if pending is not None and (
                            cancel_owner is None or pending[0] == cancel_owner
                        ):
                            self._intents.pop(snapshot_id, None)
                        self._tp_cancel_pending.pop(snapshot_id, None)
                candidates = tuple(
                    (
                        str(snapshot_id),
                        str(owner),
                        int(parent_tokens),
                        int(prompt_tokens),
                    )
                    for snapshot_id, owner, parent_tokens, prompt_tokens in (
                        allocation_plan
                    )
                )

            for snapshot_id, owner, parent_tokens, prompt_tokens in candidates:
                if snapshot_id in self._tp_retired_in_epoch:
                    continue
                existing = self._leases.get(snapshot_id)
                if existing is not None:
                    if (
                        existing.owner != owner
                        or existing.parent_tokens != parent_tokens
                        or existing.prompt_tokens != prompt_tokens
                    ):
                        raise RuntimeError(
                            "TP workset allocation plan disagrees with an existing "
                            f"lease for {snapshot_id}"
                        )
                    continue
                local_intent = self._intents.get(snapshot_id)
                if local_intent is None:
                    # TP0 is the sole logical decision maker.  A follower can
                    # observe the filesystem/HTTP marker one tick later, so
                    # install the immutable group intent from TP0 rather than
                    # allowing rank-local observation order to choose pages.
                    if allocation_plan is None:
                        continue
                    self._intents[snapshot_id] = (
                        owner,
                        parent_tokens,
                        prompt_tokens,
                    )
                    local_intent = self._intents[snapshot_id]
                if local_intent != (owner, parent_tokens, prompt_tokens):
                    local_owner = local_intent[0]
                    direct_to_slow_race = (
                        allocation_plan is not None
                        and owner == self.direct_owner(snapshot_id)
                        and (snapshot_id, owner) in self._superseded_owners
                        and local_owner.startswith(f"slow:{snapshot_id}:")
                        and local_intent[1:] == (parent_tokens, prompt_tokens)
                    )
                    if not direct_to_slow_race:
                        raise RuntimeError(
                            "TP workset allocation plan disagrees with local intent "
                            f"for {snapshot_id}: "
                            f"plan={(owner, parent_tokens, prompt_tokens)} "
                            f"local={local_intent}"
                        )
                    # The plan is already a TP0-broadcast group decision, but
                    # one rank may observe HOST_READY and enqueue Slow before
                    # another.  Both ranks must still allocate the old Direct
                    # entry in this epoch so every following page allocation
                    # remains identical.  TP0's persistent tombstone retires
                    # it group-wide in the next control epoch; Host recovery
                    # then retries the Slow intent.
                    local_intent = (owner, parent_tokens, prompt_tokens)
                # Parent and suffix have distinct ownership transitions: the
                # parent is filled by Direct/Slow I/O, while the suffix is
                # filled by incremental Prefill.  Round each slice
                # independently so an unaligned parent can never steal the
                # first page required by the suffix.
                parent_allocated = (
                    (parent_tokens + self.page_size - 1) // self.page_size
                ) * self.page_size
                suffix_tokens = prompt_tokens - parent_tokens
                suffix_allocated = (
                    (suffix_tokens + self.page_size - 1) // self.page_size
                ) * self.page_size
                allocated_tokens = parent_allocated + suffix_allocated
                if reserve_tokens and (
                    allocator.available_size() - reserve_tokens < allocated_tokens
                ):
                    self._allocation_failures += 1
                    if allocation_plan is not None:
                        break
                    continue
                device_indices = allocator.alloc(allocated_tokens)
                if device_indices is None:
                    self._allocation_failures += 1
                    if allocation_plan is not None:
                        break
                    continue
                state_device_indices = []
                for state_allocator in self._state_allocators:
                    state_indices = state_allocator.alloc(1)
                    if state_indices is None:
                        for rollback_allocator, rollback_indices in zip(
                            self._state_allocators, state_device_indices
                        ):
                            rollback_allocator.free(rollback_indices)
                        allocator.free(device_indices)
                        device_indices = None
                        break
                    state_device_indices.append(state_indices)
                if device_indices is None:
                    self._allocation_failures += 1
                    if allocation_plan is not None:
                        break
                    continue
                parent_indices = device_indices[:parent_allocated]
                page_indices = kv_to_page_indices(
                    parent_indices.cpu().numpy(), self.page_size
                )
                self._leases[snapshot_id] = AgenticPWorksetLease(
                    snapshot_id=snapshot_id,
                    lease_id=self._next_lease_id,
                    owner=owner,
                    parent_tokens=parent_tokens,
                    parent_allocated_tokens=parent_allocated,
                    prompt_tokens=prompt_tokens,
                    allocated_tokens=allocated_tokens,
                    device_indices=device_indices,
                    parent_page_indices=page_indices,
                    state_device_indices=tuple(state_device_indices),
                )
                self._next_lease_id += 1
                self._grants += 1
                self._intents.pop(snapshot_id, None)
                self._grant_events.append(snapshot_id)

    def install_tp_plan(
        self,
        epoch: int,
        plan: Sequence[Tuple[str, str, int, int]],
        *,
        retiring_ids: Sequence[str] = (),
    ) -> None:
        """Freeze one TP0-authored allocator transaction until the next epoch."""

        normalized = tuple(
            (
                str(snapshot_id),
                str(owner),
                int(parent_tokens),
                int(prompt_tokens),
            )
            for snapshot_id, owner, parent_tokens, prompt_tokens in plan
        )
        if len({entry[0] for entry in normalized}) != len(normalized):
            raise RuntimeError(f"duplicate snapshot in TP workset epoch {epoch}")
        authoritative_retirements = frozenset(str(item) for item in retiring_ids)
        with self._lock:
            epoch = int(epoch)
            if epoch == self._tp_plan_epoch:
                if (
                    normalized == self._tp_plan
                    and authoritative_retirements
                    == self._tp_authoritative_retirements
                ):
                    return
                raise RuntimeError(
                    f"TP workset epoch {epoch} was replayed with new content"
                )
            if epoch < self._tp_plan_epoch:
                raise RuntimeError(
                    f"stale TP workset epoch {epoch} < {self._tp_plan_epoch}"
                )
            for snapshot_id, owner, parent_tokens, prompt_tokens in normalized:
                current = self._leases.get(snapshot_id)
                if current is not None and (
                    current.owner != owner
                    or current.parent_tokens != parent_tokens
                    or current.prompt_tokens != prompt_tokens
                ):
                    raise RuntimeError(
                        "TP workset epoch changes a live lease shape for "
                        f"{snapshot_id}"
                    )
                pending = self._intents.get(snapshot_id)
                if pending is not None and pending != (
                    owner,
                    parent_tokens,
                    prompt_tokens,
                ):
                    pending_owner = pending[0]
                    direct_to_slow_race = (
                        owner == self.direct_owner(snapshot_id)
                        and (snapshot_id, owner) in self._superseded_owners
                        and pending_owner.startswith(f"slow:{snapshot_id}:")
                        and pending[1:] == (parent_tokens, prompt_tokens)
                    )
                    if not direct_to_slow_race:
                        raise RuntimeError(
                            "TP workset epoch disagrees with local intent for "
                            f"{snapshot_id}: "
                            f"plan={(owner, parent_tokens, prompt_tokens)} "
                            f"local={pending}"
                        )
                if snapshot_id not in authoritative_retirements:
                    # Rank-local marker expiry is not a TP group decision.
                    # TP0's live plan clears only unstarted/active local
                    # tombstones.  A shard with a posted or quiesced failed
                    # DMA keeps its terminal state until the ordinary group
                    # failure path publishes an authoritative retire command.
                    self._tp_cancel_pending.pop(snapshot_id, None)
                    if current is None or current.state == "active":
                        self._tp_release_pending.pop(snapshot_id, None)
                        self._tp_retire_requested.discard(snapshot_id)
            self._tp_plan_epoch = epoch
            self._tp_plan = normalized
            self._tp_authoritative_retirements = authoritative_retirements
            self._tp_retired_in_epoch.clear()
            for snapshot_id in authoritative_retirements:
                self._prepare_tp_retire_locked(snapshot_id)

    def prepare_tp_plan(
        self,
        epoch: int,
        *,
        retiring_ids: Optional[Sequence[str]] = None,
    ) -> tuple[tuple[str, str, int, int], ...]:
        """Build and freeze TP0's plan before it enters the broadcast.

        Freezing at construction, rather than when TP0 later consumes its own
        broadcast, closes the interval in which a background timeout could
        delete a newly planned intent without leaving a deferred-cancel record.
        """

        with self._lock:
            plan = []
            seen = set()
            for snapshot_id, lease in self._leases.items():
                if (
                    lease.state in {"consumed", "releasing"}
                    or snapshot_id in self._release_requested
                ):
                    # A consumed lease has already moved to the native Req on
                    # this rank.  Native TP scheduling must perform that same
                    # ownership move on every rank before TP0 publishes the
                    # separate group handoff acknowledgement.
                    continue
                plan.append(
                    (
                        snapshot_id,
                        lease.owner,
                        int(lease.parent_tokens),
                        int(lease.prompt_tokens),
                    )
                )
                seen.add(snapshot_id)
            for snapshot_id, (
                owner,
                parent_tokens,
                prompt_tokens,
            ) in self._intents.items():
                if (
                    snapshot_id not in seen
                ):
                    plan.append(
                        (
                            snapshot_id,
                            owner,
                            int(parent_tokens),
                            int(prompt_tokens),
                        )
                    )
            frozen = tuple(plan)
            self.install_tp_plan(
                epoch,
                frozen,
                retiring_ids=(
                    self.tp_retire_candidates
                    if retiring_ids is None
                    else retiring_ids
                ),
            )
            return frozen

    def prepare_tp_control(
        self,
        epoch: int,
        *,
        retiring_ids: Sequence[str] = (),
    ) -> tuple[
        tuple[tuple[str, str, int, int], ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        """Atomically freeze one TP allocation-and-retirement transaction.

        TP0 must not sample retire candidates and freeze its allocation plan
        under separate broker locks: an asynchronous I/O completion can ask
        to retire a lease in that interval.  Folding every candidate visible
        at the freeze boundary into the authoritative retirement set ensures
        that the native TP broadcast carries exactly the terminal decisions
        used when the plan was installed.
        """

        with self._lock:
            authoritative_retirements = set(str(item) for item in retiring_ids)
            authoritative_retirements.update(self._tp_retire_requested)
            # HOST_READY can supersede Direct after TP0 has already frozen the
            # current allocation epoch but before that epoch is installed on
            # every rank.  Such an old plan is still installed verbatim for
            # rank consistency; the next TP0 epoch must then derive a group
            # retirement from the persistent owner tombstone.  Relying only
            # on the one-shot cancel flag loses this race when no local intent
            # or lease existed at the instant HOST_READY was observed.
            for snapshot_id, owner in self._superseded_owners:
                if owner != self.direct_owner(snapshot_id):
                    continue
                lease = self._leases.get(snapshot_id)
                intent = self._intents.get(snapshot_id)
                planned = any(
                    entry[0] == snapshot_id and entry[1] == owner
                    for entry in self._tp_plan
                )
                # Retirement is snapshot-scoped, while supersession is
                # owner-scoped.  Once Slow has taken over the same snapshot,
                # an older Direct plan must not retire the new owner.  The old
                # plan is evidence only while there is no newer live owner.
                live_owner = (
                    lease.owner
                    if lease is not None
                    else (intent[0] if intent is not None else None)
                )
                if live_owner == owner or (live_owner is None and planned):
                    authoritative_retirements.add(snapshot_id)
                    self._tp_retire_requested.add(snapshot_id)
            frozen_retirements = tuple(sorted(authoritative_retirements))
            plan = self.prepare_tp_plan(
                epoch,
                retiring_ids=frozen_retirements,
            )
            return (
                plan,
                frozen_retirements,
                tuple(sorted(self._tp_handoff_committed)),
            )

    def drain_grant_events(self) -> tuple[str, ...]:
        """Return newly allocated worksets without scanning broker state."""

        with self._lock:
            events = tuple(self._grant_events)
            self._grant_events.clear()
            return events

    def handoff_to_req(
        self, snapshot_id: str, req, lease: AgenticPWorksetLease
    ) -> None:
        """Move the complete lease from broker ownership to one live Req."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            # Handoff is the ownership commit boundary.  Cleanup after that
            # boundary (Host extent release, TP bookkeeping, lane return) may
            # itself need a retry, so accepting the exact same Req/lease again
            # makes the commit safely idempotent.
            if current.state == "handed":
                if getattr(req, "_agentic_p_workset_lease", None) is current:
                    return
                raise RuntimeError(
                    f"workset lease was handed to another request for {snapshot_id}"
                )
            if current.state != "binding":
                raise RuntimeError(
                    f"workset lease is {current.state} for {snapshot_id}"
                )
            if not current.parent_bound:
                raise RuntimeError(f"workset parent is not bound for {snapshot_id}")
            actual_prompt_tokens = len(req.origin_input_ids)
            if actual_prompt_tokens != current.prompt_tokens:
                raise RuntimeError(
                    f"workset prompt changed for {snapshot_id}: "
                    f"reserved={current.prompt_tokens} actual={actual_prompt_tokens}"
                )
            if len(current.state_device_indices) > 1:
                raise RuntimeError(
                    "agentic workset currently supports one Mamba state component"
                )
            if current.state_device_indices and getattr(
                req, "mamba_pool_idx", None
            ) is not None:
                raise RuntimeError(
                    f"request already owns Mamba state for {snapshot_id}"
                )
            current.state = "handed"
            self._intents.pop(snapshot_id, None)
            # This marker is the scheduler-visible ownership contract.  The
            # request already owns parent+suffix KV, so ordinary free-KV
            # admission must not reject it.  Keep the marker until the whole
            # P-side request-generation leaves HBM (P->D or P->D Host).
            req._agentic_workset_backed = True
            req._agentic_p_workset_lease = current
            req._agentic_p_workset_broker = self
            req._agentic_workset_suffix_indices = current.remaining_suffix_indices
            if current.state_device_indices:
                req.mamba_pool_idx = current.state_device_indices[0][0]
                req.mamba_needs_clear = False
                # The reverse snapshot represents the complete parent at this
                # request-generation boundary.  Any page/chunk tail that the
                # native cache cannot bind is recomputed by normal Prefill.
                req.mamba_last_track_seqlen = current.parent_tokens

    def consume_suffix(
        self,
        lease: AgenticPWorksetLease,
        extend_tokens: int,
        *,
        final_prompt_chunk: bool,
    ) -> torch.Tensor:
        """Transfer suffix slots to one Prefill chunk.

        The returned tensor has the length expected by SGLang's extend
        batch.  On the final logical prompt chunk, ownership of the whole
        final KV page (including unused padding slots) moves to the request,
        so the broker drops the lease without freeing that padding.
        """

        extend_tokens = int(extend_tokens)
        if extend_tokens < 0:
            raise ValueError("extend_tokens must be non-negative")
        with self._lock:
            current = self._leases.get(lease.snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {lease.snapshot_id}")
            if current.state != "handed":
                raise RuntimeError(
                    f"workset lease is {current.state} for {lease.snapshot_id}"
                )
            start = current.suffix_cursor
            end = start + extend_tokens
            logical_suffix_tokens = current.prompt_tokens - current.parent_tokens
            physical_suffix_tokens = current.suffix_allocated_tokens
            if end > physical_suffix_tokens:
                raise RuntimeError(
                    f"workset suffix over-consumed for {lease.snapshot_id}: "
                    f"end={end} physical={physical_suffix_tokens}"
                )
            if final_prompt_chunk and end < logical_suffix_tokens:
                raise RuntimeError(
                    f"final workset chunk is incomplete for {lease.snapshot_id}: "
                    f"end={end} logical={logical_suffix_tokens}"
                )
            if (
                not final_prompt_chunk
                and end < physical_suffix_tokens
                and end % self.page_size
            ):
                raise RuntimeError(
                    "non-final chunked Prefill must end on a KV page boundary"
                )
            indices = current.suffix_indices[start:end]
            current.suffix_cursor = end
            if final_prompt_chunk or end == physical_suffix_tokens:
                current.state = "consumed"
                self._leases.pop(lease.snapshot_id, None)
                if self._tp_plan_epoch >= 0:
                    self._tp_handoff_committed.add(lease.snapshot_id)
            return indices

    def commit_tp_handoff(self, snapshot_id: str) -> bool:
        """Finish a TP-wide broker-to-Req ownership transfer.

        This intentionally does not call the allocator.  Native TP scheduling
        must consume the same final suffix on every rank before this command;
        therefore a remaining local lease is a rank split and fails closed.
        """

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is not None:
                return False
            self._intents.pop(snapshot_id, None)
            self._tp_cancel_pending.pop(snapshot_id, None)
            self._tp_release_pending.pop(snapshot_id, None)
            self._tp_handoff_committed.discard(snapshot_id)
            return True

    def begin_bind(self, snapshot_id: str, lease: AgenticPWorksetLease) -> bool:
        """Atomically transfer a completed I/O lease to scheduler binding."""

        with self._lock:
            if (
                snapshot_id in self._tp_cancel_pending
                or snapshot_id in self._tp_release_pending
                or snapshot_id in self._tp_retire_requested
            ):
                return False
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "active":
                return current.state == "binding"
            current.state = "binding"
            return True

    def begin_io_attempt(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Exclusively reserve one lease for one concrete Direct session."""

        with self._lock:
            if (
                snapshot_id in self._tp_cancel_pending
                or snapshot_id in self._tp_release_pending
                or snapshot_id in self._tp_retire_requested
            ):
                return False
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "active":
                return False
            current.state = "io_reserved"
            current.io_attempt = str(attempt)
            return True

    def mark_io_inflight(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> None:
        """Fence allocator reuse after this attempt publishes destinations."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.state != "io_reserved" or current.io_attempt != attempt:
                raise RuntimeError(
                    f"cannot start attempt={attempt} on {current.state} "
                    f"workset {snapshot_id} owned_by={current.io_attempt}"
                )
            current.state = "io_inflight"

    def cancel_io_attempt(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Drop an exclusive attempt before any remote write can begin."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "io_reserved" or current.io_attempt != attempt:
                return False
            current.state = "active"
            current.io_attempt = None
            return True

    def mark_io_quiesced(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Publish a definitive transport terminal state to the allocator."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.io_attempt != attempt:
                return False
            if current.state == "io_inflight":
                current.state = "active"
                current.io_attempt = None
                return True
            if current.state == "release_pending":
                current.io_attempt = None
                if snapshot_id in self._tp_retire_requested:
                    current.state = "retire_ready"
                    self._tp_release_pending[snapshot_id] = current.lease_id
                else:
                    current.state = "releasing"
                    self._release_requested[snapshot_id] = current.lease_id
                return True
            return False

    def commit_parent_bound(
        self, snapshot_id: str, lease: AgenticPWorksetLease
    ) -> None:
        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.state != "binding":
                raise RuntimeError(
                    f"cannot bind {current.state} workset lease for {snapshot_id}"
                )
            current.parent_bound = True

    def abort_bind(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        *,
        parent_bound: bool,
    ) -> bool:
        """Release a scheduler-owned bind after its Radix mutation is undone."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "binding":
                return False
            if self._tp_plan_epoch >= 0:
                # A bind failure is a TP-group retirement.  It must be
                # broadcast and committed at one scheduler-safe boundary;
                # no rank rolls back pages independently.
                current.parent_bound = bool(parent_bound)
                current.state = "retire_ready"
                self._tp_release_pending[snapshot_id] = current.lease_id
                self._tp_retire_requested.add(snapshot_id)
                return True
            current.parent_bound = bool(parent_bound)
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def cancel_unstarted(
        self, snapshot_id: str, *, owner: Optional[str] = None
    ) -> bool:
        """Atomically cancel work that has not started physical I/O.

        The scheduler can grant an intent between two progress-worker passes.
        Cancellation must therefore cover both representations: a pending
        intent and an ``active`` lease.  Once I/O, binding, or request handoff
        begins, the corresponding owner-specific terminal path is solely
        responsible for release.
        """

        with self._lock:
            cancelled = False
            pending = self._intents.get(snapshot_id)
            lease = self._leases.get(snapshot_id)
            planned = any(entry[0] == snapshot_id for entry in self._tp_plan)
            if planned and (
                (pending is not None and (owner is None or pending[0] == owner))
                or (
                    lease is not None
                    and lease.state == "active"
                    and (owner is None or lease.owner == owner)
                )
            ):
                # The current TP epoch is immutable.  Rank0 will omit this
                # entry from its next plan; followers learn the same removal
                # from the native broadcast.  Until then all ranks retain the
                # exact same physical ownership.
                self._tp_cancel_pending[snapshot_id] = owner
                self._tp_retire_requested.add(snapshot_id)
                return True
            if pending is not None and (owner is None or pending[0] == owner):
                self._intents.pop(snapshot_id, None)
                cancelled = True
            if (
                lease is not None
                and lease.state == "active"
                and (owner is None or lease.owner == owner)
            ):
                lease.state = "releasing"
                self._release_requested[snapshot_id] = lease.lease_id
                cancelled = True
            return cancelled

    def supersede_unstarted(self, snapshot_id: str, *, owner: str) -> bool:
        """Permanently supersede an unstarted owner for this generation.

        Unlike a one-shot cancellation, this also rejects a late intent from
        the same owner.  Posted I/O is deliberately left to its transport
        terminal path.
        """

        with self._lock:
            self._superseded_owners.add((snapshot_id, owner))
            return self.cancel_unstarted(snapshot_id, owner=owner)

    def owner_is_superseded(self, snapshot_id: str, *, owner: str) -> bool:
        with self._lock:
            return (snapshot_id, owner) in self._superseded_owners

    @property
    def tp_retire_candidates(self) -> tuple[str, ...]:
        """Return TP0's locally requested group terminal transitions."""

        with self._lock:
            return tuple(self._tp_retire_requested)

    def prepare_tp_retire(self, snapshot_id: str) -> bool:
        """Install a group tombstone and report whether local pages are safe."""

        with self._lock:
            return self._prepare_tp_retire_locked(snapshot_id)

    def _prepare_tp_retire_locked(self, snapshot_id: str) -> bool:
        """Materialize one TP tombstone while ``self._lock`` is held."""

        self._tp_retire_requested.add(snapshot_id)
        lease = self._leases.get(snapshot_id)
        if lease is None:
            return True
        if lease.state in {"io_reserved", "io_inflight"}:
            lease.state = "release_pending"
            self._tp_release_pending[snapshot_id] = lease.lease_id
            return False
        return lease.state in {"active", "retire_ready", "releasing"}

    def tp_retire_ready(self, snapshot_id: str) -> bool:
        with self._lock:
            lease = self._leases.get(snapshot_id)
            return lease is None or lease.state in {
                "active",
                "retire_ready",
                "releasing",
            }

    def commit_tp_retire(self, snapshot_id: str) -> bool:
        """Commit one all-rank-safe terminal at the scheduler boundary."""

        with self._lock:
            if snapshot_id not in self._tp_retire_requested:
                return False
            lease = self._leases.get(snapshot_id)
            if lease is not None:
                if lease.state not in {"active", "retire_ready", "releasing"}:
                    return False
                if lease.state != "releasing":
                    lease.state = "releasing"
                    lease.io_attempt = None
                    self._release_requested[snapshot_id] = lease.lease_id
            self._intents.pop(snapshot_id, None)
            self._tp_cancel_pending.pop(snapshot_id, None)
            self._tp_release_pending.pop(snapshot_id, None)
            self._tp_retire_requested.discard(snapshot_id)
            return True

    @property
    def leased_tokens(self) -> int:
        with self._lock:
            return sum(lease.allocated_tokens for lease in self._leases.values())

    @property
    def stats(self) -> Tuple[int, int, int]:
        """Return pending intents, grants, and allocation misses."""

        with self._lock:
            return len(self._intents), self._grants, self._allocation_failures

    @property
    def lease_state_summary(self) -> str:
        """Compact count/token ownership summary for progress diagnostics."""

        with self._lock:
            counts: Dict[str, int] = {}
            tokens: Dict[str, int] = {}
            for lease in self._leases.values():
                counts[lease.state] = counts.get(lease.state, 0) + 1
                tokens[lease.state] = tokens.get(lease.state, 0) + int(
                    lease.allocated_tokens
                )
            return ",".join(
                f"{state}:{counts[state]}/{tokens[state]}"
                for state in sorted(counts)
            ) or "empty"

    @property
    def active_lease_summary(self) -> str:
        """Identify unstarted leases when progress stops making forward progress."""

        with self._lock:
            active = [
                f"{lease.snapshot_id}@{lease.owner}/{lease.allocated_tokens}"
                for lease in self._leases.values()
                if lease.state == "active"
            ]
            return ";".join(active) or "empty"

    @property
    def unaccounted_tokens(self) -> int:
        """Lease pages not already represented by a bound Radix parent."""

        with self._lock:
            return sum(
                (
                    lease.remaining_suffix_indices.numel()
                    if lease.parent_bound
                    else lease.allocated_tokens
                )
                for lease in self._leases.values()
            )
