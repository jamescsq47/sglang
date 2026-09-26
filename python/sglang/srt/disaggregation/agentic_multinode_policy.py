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
from typing import Any, Callable, Optional

from sglang.srt.disaggregation.agentic_group_protocol import GenerationKey
from sglang.srt.disaggregation.agentic_group_transfer import GroupTransferPlan


class ParentPolicyPhase(str, Enum):
    WAIT_CHILD = "wait_child"
    DIRECT_SUBMITTED = "direct_submitted"
    HOST_STORE_SUBMITTED = "host_store_submitted"
    HOST_CAPACITY_WAIT = "host_capacity_wait"
    HOST_DURABLE = "host_durable"
    HOST_RESTORE_SUBMITTED = "host_restore_submitted"
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window = max(0.0, float(direct_window_seconds))
        self._submit = submit
        self._make_direct = make_direct
        self._make_host_store = make_host_store
        self._make_host_restore = make_host_restore
        self._clock = clock
        self._states: dict[GenerationKey, _ParentState] = {}
        self._deadlines: list[tuple[float, int, GenerationKey]] = []
        self._submissions: list[GroupTransferPlan] = []
        self._next_epoch = 1
        self._host_capacity_epoch = 0
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
        self._submissions.append(plan)
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

    def memory_available(self) -> None:
        """Retry retained Host restores only on an allocator-release edge."""

        with self._changed:
            self._check_locked()
            for state in tuple(self._states.values()):
                if (
                    state.phase is ParentPolicyPhase.HOST_DURABLE
                    and state.waiting_capacity
                    and state.child is not None
                ):
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

    def committed(self, key: GenerationKey) -> None:
        with self._changed:
            state = self._states.get(key)
            if state is not None:
                state.phase = ParentPolicyPhase.COMPLETE
                self._states.pop(key, None)

    def _deadline_loop(self) -> None:
        while True:
            plan = None
            with self._changed:
                while not self._closed:
                    if self._submissions:
                        plan = self._submissions.pop(0)
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
                self._submit(plan)
            except BaseException as error:
                with self._changed:
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
    phase: P2DPolicyPhase = P2DPolicyPhase.DIRECT_SUBMITTED
    host_descriptors: Any = None
    waiting_capacity: bool = False
    host_submit_epoch: int = 0


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
            self._states[plan.key] = _P2DState(plan)

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

    def host_durable(self, key: GenerationKey, descriptors: Any) -> None:
        with self._lock:
            state = self._states[key]
            if state.phase is not P2DPolicyPhase.HOST_STORE_SUBMITTED:
                raise RuntimeError("P2D Host durability arrived in an invalid phase")
            state.host_descriptors = descriptors
            state.phase = P2DPolicyPhase.HOST_RESTORE_SUBMITTED
            plan = self._make_host_restore(state.direct, descriptors)
        self._submit(plan)

    def host_store_rejected(self, key: GenerationKey) -> None:
        with self._lock:
            state = self._states.get(key)
            if state is None or state.phase is not P2DPolicyPhase.HOST_STORE_SUBMITTED:
                return
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

    def memory_available(self) -> None:
        plans = []
        with self._lock:
            for state in self._states.values():
                if (
                    state.phase is P2DPolicyPhase.HOST_DURABLE
                    and state.waiting_capacity
                ):
                    state.waiting_capacity = False
                    state.phase = P2DPolicyPhase.HOST_RESTORE_SUBMITTED
                    plans.append(
                        self._make_host_restore(
                            state.direct, state.host_descriptors
                        )
                    )
        for plan in plans:
            self._submit(plan)

    def committed(self, key: GenerationKey) -> None:
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.phase = P2DPolicyPhase.COMPLETE
                self._states.pop(key, None)


__all__ = [
    "D2PPolicyActor",
    "P2DPolicyActor",
    "P2DPolicyPhase",
    "ParentPolicyPhase",
]
