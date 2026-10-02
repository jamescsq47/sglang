"""Event-driven rank-zero D→P path policy.

This actor joins a completed D parent with the next P child without polling a
filesystem or dropping an early intent.  The only timer is the configured
Direct arrival window.  Capacity rejection retains the generation and retries
only on an explicit memory-available edge.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from sglang.srt.disaggregation.agentic_group_protocol import GenerationKey
from sglang.srt.disaggregation.agentic_group_transfer import GroupTransferPlan


class ParentPolicyPhase(str, Enum):
    WAIT_CHILD = "wait_child"
    DIRECT_SUBMITTED = "direct_submitted"
    HOST_STORE_SUBMITTED = "host_store_submitted"
    HOST_CAPACITY_WAIT = "host_capacity_wait"
    HOST_DURABLE = "host_durable"
    HOST_RESTORE_SUBMITTED = "host_restore_submitted"
    HOST_EVICT_SUBMITTED = "host_evict_submitted"
    COMPLETE = "complete"


@dataclass(slots=True)
class _ParentState:
    key: GenerationKey
    candidate: GroupTransferPlan
    deadline: float
    child: Any = None
    host_descriptors: Any = None
    phase: ParentPolicyPhase = ParentPolicyPhase.WAIT_CHILD
    waiting_capacity: bool = False
    deadline_epoch: int = 0
    host_submit_epoch: int = 0
    restore_submit_epoch: int = 0


class D2PPolicyActor:
    """One coordinator-local actor; callbacks must be fast and nonblocking."""

    def __init__(
        self,
        *,
        direct_window_seconds: float,
        submit: Callable[[GroupTransferPlan], Any],
        make_direct: Callable[[GroupTransferPlan, Any], GroupTransferPlan],
        make_host_store: Callable[[GroupTransferPlan], GroupTransferPlan],
        make_host_restore: Callable[
            [GroupTransferPlan, Any, Any], GroupTransferPlan
        ],
        make_recompute: Optional[
            Callable[[GenerationKey, Any], GroupTransferPlan]
        ] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window = max(0.0, float(direct_window_seconds))
        self._submit = submit
        self._make_direct = make_direct
        self._make_host_store = make_host_store
        self._make_host_restore = make_host_restore
        self._make_recompute = make_recompute
        self._clock = clock
        self._states: dict[GenerationKey, _ParentState] = {}
        self._deadlines: list[tuple[float, int, GenerationKey]] = []
        self._submissions: list[
            tuple[Optional[GenerationKey], GroupTransferPlan]
        ] = []
        self._evicted: set[GenerationKey] = set()
        self._next_epoch = 1
        self._host_capacity_epoch = 0
        self._restore_capacity_epoch = 0
        self._closed = False
        self._fatal: Optional[BaseException] = None
        self._changed = threading.Condition()
        self._thread = threading.Thread(
            target=self._deadline_loop,
            name="agentic-d2p-policy",
            daemon=True,
        )
        self._thread.start()

    def _submit_locked(self, state: _ParentState, plan: GroupTransferPlan) -> None:
        # Public callbacks can run on the low-level coordinator thread.  They
        # must never recursively call submit(), which needs that same thread.
        # Serialize the decision here and let the actor thread submit later.
        self._submissions.append((state.key, plan))
        self._changed.notify_all()

    def _submit_recompute_locked(self, key: GenerationKey, child: Any) -> None:
        if self._make_recompute is None:
            raise RuntimeError("Host eviction has no recompute plan factory")
        self._submissions.append((None, self._make_recompute(key, child)))
        self._changed.notify_all()

    def _submit_host_store_locked(self, state: _ParentState) -> None:
        state.phase = ParentPolicyPhase.HOST_STORE_SUBMITTED
        state.host_submit_epoch = self._host_capacity_epoch
        self._submit_locked(state, self._make_host_store(state.candidate))

    def _check_locked(self) -> None:
        if self._closed:
            raise RuntimeError("D2P policy actor is closed")
        if self._fatal is not None:
            raise RuntimeError("D2P policy actor failed") from self._fatal

    def offer_parent(self, candidate: GroupTransferPlan) -> None:
        with self._changed:
            self._check_locked()
            old = self._states.get(candidate.key)
            if old is not None:
                if old.candidate != candidate:
                    raise RuntimeError("parent generation was offered twice")
                return
            deadline = self._clock() + self._window
            epoch = self._next_epoch
            self._next_epoch += 1
            state = _ParentState(candidate.key, candidate, deadline)
            state.deadline_epoch = epoch
            self._states[candidate.key] = state
            heapq.heappush(self._deadlines, (deadline, epoch, candidate.key))
            self._join_early_child_locked(state)
            self._changed.notify_all()

    def child_arrived(self, parent_key: GenerationKey, child: Any) -> None:
        with self._changed:
            self._check_locked()
            if parent_key in self._evicted:
                self._submit_recompute_locked(parent_key, child)
                return
            state = self._states.get(parent_key)
            if state is None:
                # The child may beat the D intent.  Retain it in a separate
                # placeholder-free table so offer_parent can join it without
                # inventing a generation state.
                pending = getattr(self, "_early_children", None)
                if pending is None:
                    pending = self._early_children = {}
                existing = pending.get(parent_key)
                if existing is not None and existing is not child:
                    raise RuntimeError("parent generation has two child objects")
                pending[parent_key] = child
                return
            if state.child is not None and state.child is not child:
                raise RuntimeError("parent generation has two child objects")
            state.child = child
            if state.phase is ParentPolicyPhase.WAIT_CHILD:
                state.phase = ParentPolicyPhase.DIRECT_SUBMITTED
                self._submit_locked(
                    state, self._make_direct(state.candidate, state.child)
                )
            elif state.phase is ParentPolicyPhase.HOST_DURABLE:
                state.phase = ParentPolicyPhase.HOST_RESTORE_SUBMITTED
                self._submit_locked(
                    state,
                    self._make_host_restore(
                        state.candidate, state.child, state.host_descriptors
                    ),
                )

    def phase_counts(self) -> dict[str, int]:
        with self._changed:
            counts: dict[str, int] = {}
            for state in self._states.values():
                name = state.phase.value
                counts[name] = counts.get(name, 0) + 1
            if self._evicted:
                counts["evicted_wait_child"] = len(self._evicted)
            return counts

    def begin_eviction(self, key: GenerationKey) -> bool:
        """Move one durable generation behind the TP-wide eviction fence."""

        with self._changed:
            self._check_locked()
            state = self._states.get(key)
            if state is None or state.phase is not ParentPolicyPhase.HOST_DURABLE:
                return False
            state.phase = ParentPolicyPhase.HOST_EVICT_SUBMITTED
            return True

    def eviction_rejected(self, key: GenerationKey) -> None:
        with self._changed:
            state = self._states.get(key)
            if state is None or state.phase is not ParentPolicyPhase.HOST_EVICT_SUBMITTED:
                return
            state.phase = ParentPolicyPhase.HOST_DURABLE
            if state.child is not None and not state.waiting_capacity:
                state.phase = ParentPolicyPhase.HOST_RESTORE_SUBMITTED
                self._submit_locked(
                    state,
                    self._make_host_restore(
                        state.candidate, state.child, state.host_descriptors
                    ),
                )

    def evicted(self, key: GenerationKey) -> None:
        """Commit RECOMPUTE_REQUIRED after every TP Host shard is gone."""

        with self._changed:
            state = self._states.get(key)
            if state is None or state.phase is not ParentPolicyPhase.HOST_EVICT_SUBMITTED:
                raise RuntimeError("Host eviction committed outside its policy phase")
            self._states.pop(key, None)
            self._evicted.add(key)
            if state.child is not None:
                self._submit_recompute_locked(key, state.child)

    def application_final(self, key: GenerationKey) -> bool:
        """Retire finality only when no physical policy state remains.

        False leaves the terminal edge pending behind the normal Direct/Host
        fence. True removes an already-committed eviction tombstone or an
        already-complete generation.
        """

        with self._changed:
            early = getattr(self, "_early_children", {})
            if key in self._states or key in early:
                return False
            # A completely unknown key may be an application-final edge that
            # overtook the asynchronous parent intent. Keep it pending in the
            # provider until offer_parent and its physical fence arrive. Only
            # an explicit committed eviction proves there is no live owner.
            if key not in self._evicted:
                return False
            self._evicted.remove(key)
            self._submissions = [
                (origin, plan)
                for origin, plan in self._submissions
                if origin != key and plan.key != key
            ]
            self._changed.notify_all()
            return True

    def _join_early_child_locked(self, state: _ParentState) -> None:
        pending = getattr(self, "_early_children", None)
        if not pending:
            return
        child = pending.pop(state.key, None)
        if child is not None:
            state.child = child
            state.phase = ParentPolicyPhase.DIRECT_SUBMITTED
            self._submit_locked(
                state, self._make_direct(state.candidate, state.child)
            )

    def direct_rejected(self, key: GenerationKey) -> None:
        """A real Direct PREPARE/capacity failure falls back without loss."""

        with self._changed:
            self._check_locked()
            # Abort/timeout delivery is asynchronous and may race a terminal
            # commit or refer to an initial-prefill allocation, which is not
            # owned by this parent policy.  Cleanup must therefore be
            # idempotent for an unknown generation.
            state = self._states.get(key)
            if state is None:
                return
            if state.phase is not ParentPolicyPhase.DIRECT_SUBMITTED:
                return
            self._submit_host_store_locked(state)

    def host_durable(self, key: GenerationKey, descriptors: Any) -> None:
        with self._changed:
            self._check_locked()
            state = self._states[key]
            if state.phase is not ParentPolicyPhase.HOST_STORE_SUBMITTED:
                raise RuntimeError("Host durability arrived in an invalid phase")
            state.host_descriptors = descriptors
            state.phase = ParentPolicyPhase.HOST_DURABLE
            if state.child is not None:
                state.phase = ParentPolicyPhase.HOST_RESTORE_SUBMITTED
                self._submit_locked(
                    state,
                    self._make_host_restore(
                        state.candidate, state.child, descriptors
                    ),
                )

    def host_store_rejected(self, key: GenerationKey) -> None:
        """Retain source HBM until a source-Host extent becomes available."""

        with self._changed:
            self._check_locked()
            state = self._states.get(key)
            if state is None or state.phase is not ParentPolicyPhase.HOST_STORE_SUBMITTED:
                return
            state.phase = ParentPolicyPhase.HOST_CAPACITY_WAIT
            if state.host_submit_epoch < self._host_capacity_epoch:
                self._submit_host_store_locked(state)

    def host_memory_available(self) -> None:
        """Retry Host stores only on a real source-arena release edge."""

        with self._changed:
            self._check_locked()
            self._host_capacity_epoch += 1
            for state in tuple(self._states.values()):
                if (
                    state.phase is ParentPolicyPhase.HOST_CAPACITY_WAIT
                    and state.host_submit_epoch < self._host_capacity_epoch
                ):
                    self._submit_host_store_locked(state)

    def restore_capacity_rejected(self, key: GenerationKey) -> None:
        with self._changed:
            self._check_locked()
            state = self._states[key]
            if state.phase is ParentPolicyPhase.HOST_RESTORE_SUBMITTED:
                state.phase = ParentPolicyPhase.HOST_DURABLE
                state.waiting_capacity = True

    @staticmethod
    def _required_tokens(state: _ParentState) -> int:
        child = state.child
        req = getattr(child, "req", None)
        if req is not None:
            return max(0, len(getattr(req, "origin_input_ids", ()) or ()))
        return max(0, int(state.candidate.payload.get("token_count", 0)))

    def memory_available(
        self,
        available_tokens: Optional[int] = None,
        *,
        endpoint_group: str = "",
    ) -> None:
        """Admit feasible Host restores on one causal allocator edge.

        Capacity notification is a hint; the physical authority remains the
        final arbiter on every TP rank.  The epoch prevents a rejected request
        from immediately retrying against the same unchanged capacity.  A
        large head item does not block smaller feasible work behind it.
        """

        with self._changed:
            self._check_locked()
            self._restore_capacity_epoch += 1
            epoch = self._restore_capacity_epoch
            remaining = None if available_tokens is None else max(0, int(available_tokens))
            admitted = 0
            for state in tuple(self._states.values()):
                if not (
                    state.phase is ParentPolicyPhase.HOST_DURABLE
                    and state.waiting_capacity
                    and state.child is not None
                    and state.restore_submit_epoch < epoch
                ):
                    continue
                child_group = (
                    str(state.child.get("target_group", ""))
                    if isinstance(state.child, Mapping)
                    else state.candidate.target_group
                )
                if endpoint_group and child_group != str(endpoint_group):
                    continue
                required = self._required_tokens(state)
                if remaining is not None and required > remaining:
                    continue
                state.restore_submit_epoch = epoch
                state.waiting_capacity = False
                state.phase = ParentPolicyPhase.HOST_RESTORE_SUBMITTED
                self._submit_locked(
                    state,
                    self._make_host_restore(
                        state.candidate,
                        state.child,
                        state.host_descriptors,
                    ),
                )
                admitted += 1
                if remaining is not None:
                    remaining -= required
                elif admitted >= 1:
                    # Compatibility for callers without a capacity snapshot.
                    break

    def committed(self, key: GenerationKey) -> None:
        with self._changed:
            state = self._states.get(key)
            if state is not None:
                state.phase = ParentPolicyPhase.COMPLETE
                self._states.pop(key, None)

    def cancelled(self, key: GenerationKey) -> None:
        """Retire policy-only state during runtime shutdown without fallback."""

        with self._changed:
            self._states.pop(key, None)
            pending = getattr(self, "_early_children", None)
            if pending is not None:
                pending.pop(key, None)
            self._evicted.discard(key)
            self._submissions = [
                (origin, plan)
                for origin, plan in self._submissions
                if origin != key and plan.key != key
            ]
            self._changed.notify_all()

    def _deadline_loop(self) -> None:
        while True:
            plan = None
            origin_key = None
            with self._changed:
                while not self._closed:
                    if self._submissions:
                        origin_key, plan = self._submissions.pop(0)
                        break
                    if not self._deadlines:
                        self._changed.wait()
                        continue
                    deadline, epoch, key = self._deadlines[0]
                    delay = deadline - self._clock()
                    if delay > 0:
                        self._changed.wait(delay)
                        continue
                    heapq.heappop(self._deadlines)
                    state = self._states.get(key)
                    if (
                        state is None
                        or state.deadline_epoch != epoch
                        or state.phase is not ParentPolicyPhase.WAIT_CHILD
                    ):
                        continue
                    self._join_early_child_locked(state)
                    if state.phase is ParentPolicyPhase.WAIT_CHILD:
                        self._submit_host_store_locked(state)
                if self._closed:
                    return
            try:
                with self._changed:
                    if self._closed:
                        return
                    if origin_key is not None and origin_key not in self._states:
                        continue
                self._submit(plan)
            except BaseException as error:
                with self._changed:
                    # Shutdown may cancel this generation in the narrow gap
                    # between the state check and low-level submit.  The
                    # low-level runtime then correctly rejects CLOSING work;
                    # do not turn that expected cancellation into a policy
                    # fatal error.
                    if self._closed:
                        return
                    if origin_key is not None and origin_key not in self._states:
                        continue
                    self._fatal = error
                    self._changed.notify_all()
                return

    def close(self) -> None:
        with self._changed:
            self._closed = True
            self._changed.notify_all()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("D2P policy actor did not stop")


class P2DPolicyPhase(str, Enum):
    DIRECT_SUBMITTED = "direct_submitted"
    HOST_STORE_SUBMITTED = "host_store_submitted"
    HOST_CAPACITY_WAIT = "host_capacity_wait"
    HOST_DURABLE = "host_durable"
    HOST_RESTORE_SUBMITTED = "host_restore_submitted"
    COMPLETE = "complete"


@dataclass(slots=True)
class _P2DState:
    direct: GroupTransferPlan
    target_bound: bool = True
    spilled_unbound: bool = False
    phase: P2DPolicyPhase = P2DPolicyPhase.DIRECT_SUBMITTED
    host_descriptors: Any = None
    waiting_capacity: bool = False
    host_submit_epoch: int = 0
    restore_submit_epoch: int = 0


class P2DPolicyActor:
    """Event-driven P→D fallback; it never polls D capacity.

    A Prefill-complete Direct plan is registered before submission.  Direct
    PREPARE failure stores the immutable source snapshot in source-local Host
    DRAM.  A failed Host restore remains durable and is retried only after a
    remote D memory-available edge.
    """

    def __init__(
        self,
        *,
        submit: Callable[[GroupTransferPlan], Any],
        make_host_store: Callable[[GroupTransferPlan], GroupTransferPlan],
        make_host_restore: Callable[
            [GroupTransferPlan, Any], GroupTransferPlan
        ],
    ) -> None:
        self._submit = submit
        self._make_host_store = make_host_store
        self._make_host_restore = make_host_restore
        self._states: dict[GenerationKey, _P2DState] = {}
        self._host_capacity_epoch = 0
        self._restore_capacity_epoch = 0
        self._lock = threading.RLock()

    def _make_host_store_locked(self, state: _P2DState) -> GroupTransferPlan:
        state.phase = P2DPolicyPhase.HOST_STORE_SUBMITTED
        state.host_submit_epoch = self._host_capacity_epoch
        return self._make_host_store(state.direct)

    def direct_submitted(self, plan: GroupTransferPlan) -> None:
        with self._lock:
            old = self._states.get(plan.key)
            if old is not None:
                if old.direct != plan:
                    raise RuntimeError("P2D generation changed its Direct plan")
                return
            self._states[plan.key] = _P2DState(
                plan,
                target_bound=not bool(plan.payload.get("late_bind_pending", False)),
            )

    def phase_counts(self) -> dict[str, int]:
        """Read-only progress summary for low-frequency diagnostics."""

        with self._lock:
            counts: dict[str, int] = {}
            for state in self._states.values():
                name = state.phase.value
                counts[name] = counts.get(name, 0) + 1
            return counts

    def bind_target(self, plan: GroupTransferPlan) -> Optional[GroupTransferPlan]:
        """Attach the D selected after Prefill to an existing P result.

        The return value is offered by the coordinator's current control
        turn.  A Host store already in flight keeps progressing independently;
        its durability callback will start restore after this binding.
        """

        if not plan.target_group:
            raise ValueError("late-bound P2D plan has no target group")
        with self._lock:
            state = self._states.get(plan.key)
            if state is None:
                self._states[plan.key] = _P2DState(plan)
                return plan
            state.direct = plan
            state.target_bound = True
            if state.phase is P2DPolicyPhase.DIRECT_SUBMITTED:
                return plan
            if (
                state.phase is P2DPolicyPhase.HOST_CAPACITY_WAIT
                and state.spilled_unbound
            ):
                # The failed Host store retained source P HBM, so a newly
                # available D may consume it directly without waiting for an
                # unrelated Host extent release.
                state.phase = P2DPolicyPhase.DIRECT_SUBMITTED
                state.spilled_unbound = False
                return plan
            if state.phase is P2DPolicyPhase.HOST_DURABLE:
                state.phase = P2DPolicyPhase.HOST_RESTORE_SUBMITTED
                return self._make_host_restore(plan, state.host_descriptors)
            return None

    def direct_rejected(self, key: GenerationKey) -> None:
        with self._lock:
            # Initial-prefill allocation and duplicate terminal callbacks do
            # not have a P2D policy entry.  They require no Host fallback.
            state = self._states.get(key)
            if state is None:
                return
            if state.phase is not P2DPolicyPhase.DIRECT_SUBMITTED:
                return
            plan = self._make_host_store_locked(state)
        self._submit(plan)

    def spill_if_unbound(self, key: GenerationKey) -> bool:
        """Atomically spill only while no D has won late binding."""

        with self._lock:
            state = self._states.get(key)
            if (
                state is None
                or state.phase is not P2DPolicyPhase.DIRECT_SUBMITTED
                or state.target_bound
            ):
                return False
            plan = self._make_host_store_locked(state)
            state.spilled_unbound = True
        self._submit(plan)
        return True

    def host_durable(self, key: GenerationKey, descriptors: Any) -> None:
        with self._lock:
            state = self._states[key]
            if state.phase is not P2DPolicyPhase.HOST_STORE_SUBMITTED:
                raise RuntimeError("P2D Host durability arrived in an invalid phase")
            state.host_descriptors = descriptors
            if state.target_bound:
                state.phase = P2DPolicyPhase.HOST_RESTORE_SUBMITTED
                plan = self._make_host_restore(state.direct, descriptors)
            else:
                state.phase = P2DPolicyPhase.HOST_DURABLE
                plan = None
        if plan is not None:
            self._submit(plan)

    def host_store_rejected(self, key: GenerationKey) -> None:
        with self._lock:
            state = self._states.get(key)
            if state is None or state.phase is not P2DPolicyPhase.HOST_STORE_SUBMITTED:
                return
            if state.target_bound and state.spilled_unbound:
                state.phase = P2DPolicyPhase.DIRECT_SUBMITTED
                state.spilled_unbound = False
                plan = state.direct
            else:
                state.phase = P2DPolicyPhase.HOST_CAPACITY_WAIT
                plan = (
                    self._make_host_store_locked(state)
                    if state.host_submit_epoch < self._host_capacity_epoch
                    else None
                )
        if plan is not None:
            self._submit(plan)

    def host_memory_available(self) -> None:
        plans = []
        with self._lock:
            self._host_capacity_epoch += 1
            for state in self._states.values():
                if (
                    state.phase is P2DPolicyPhase.HOST_CAPACITY_WAIT
                    and state.host_submit_epoch < self._host_capacity_epoch
                ):
                    plans.append(self._make_host_store_locked(state))
        for plan in plans:
            self._submit(plan)

    def restore_capacity_rejected(self, key: GenerationKey) -> None:
        with self._lock:
            state = self._states[key]
            if state.phase is P2DPolicyPhase.HOST_RESTORE_SUBMITTED:
                state.phase = P2DPolicyPhase.HOST_DURABLE
                state.waiting_capacity = True

    @staticmethod
    def _required_tokens(state: _P2DState) -> int:
        values = state.direct.payload
        # Endpoint capacity already excludes the one allocator-wide Decode
        # growth reserve.  Charging max growth again per request recreates the
        # old N * growth over-reservation and starves otherwise empty D GPUs.
        return max(0, int(values.get("prompt_tokens", 0)))

    def memory_available(
        self,
        available_tokens: Optional[int] = None,
        *,
        endpoint_group: str = "",
    ) -> None:
        plans = []
        with self._lock:
            self._restore_capacity_epoch += 1
            epoch = self._restore_capacity_epoch
            remaining = None if available_tokens is None else max(0, int(available_tokens))
            for state in self._states.values():
                if not (
                    state.phase is P2DPolicyPhase.HOST_DURABLE
                    and state.waiting_capacity
                    and state.target_bound
                    and state.restore_submit_epoch < epoch
                ):
                    continue
                if endpoint_group and state.direct.target_group != str(endpoint_group):
                    continue
                required = self._required_tokens(state)
                if remaining is not None and required > remaining:
                    continue
                state.restore_submit_epoch = epoch
                state.waiting_capacity = False
                state.phase = P2DPolicyPhase.HOST_RESTORE_SUBMITTED
                plans.append(
                    self._make_host_restore(state.direct, state.host_descriptors)
                )
                if remaining is not None:
                    remaining -= required
                else:
                    break
        for plan in plans:
            self._submit(plan)

    def committed(self, key: GenerationKey) -> None:
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.phase = P2DPolicyPhase.COMPLETE
                self._states.pop(key, None)

    def cancelled(self, key: GenerationKey) -> None:
        """Retire policy-only state during runtime shutdown without fallback."""

        with self._lock:
            self._states.pop(key, None)


__all__ = [
    "D2PPolicyActor",
    "P2DPolicyActor",
    "P2DPolicyPhase",
    "ParentPolicyPhase",
]
