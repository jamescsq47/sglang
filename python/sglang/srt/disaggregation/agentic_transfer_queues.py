"""Bounded, event-driven queues for agentic KV data-plane work.

This module deliberately owns no lifecycle policy, routing, page allocation,
or transport implementation.  A controller submits an immutable physical
attempt after those decisions have already been made.  An injected executor
starts the rank-local NIXL/Host operation and notifies the queue whenever its
physical state may have changed.

The four directions use separate queue instances and lane budgets.  A blocked
Host operation therefore cannot consume a Direct lane or stop the opposite
direction.  Completion is reported only with a physical fence proof.  In
particular, requesting cancellation never makes an in-flight operation safe to
reclaim; the executor must later report a drained fence.

There is no filesystem access and no periodic status polling here.  Executors
may internally adapt transports that require polling, but must turn that into
an edge notification before calling the supplied ``notify`` callback.
"""

from __future__ import annotations

import logging
import copy
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol

logger = logging.getLogger(__name__)


class TransferPath(str, Enum):
    D2P_DIRECT = "d2p_direct"
    D2P_HOST = "d2p_host"
    P2D_DIRECT = "p2d_direct"
    P2D_HOST = "p2d_host"


class PhysicalState(str, Enum):
    INFLIGHT = "inflight"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FenceKind(str, Enum):
    """Proof that a source/destination lease can leave transport ownership."""

    NOT_POSTED = "not_posted"
    NO_IO_REQUIRED = "no_io_required"
    DMA_COMPLETE = "dma_complete"
    CANCEL_DRAINED = "cancel_drained"
    ERROR_DRAINED = "error_drained"


@dataclass(frozen=True, slots=True)
class TransferAttempt:
    """One immutable, rank-local shard operation.

    ``payload`` is an opaque transport descriptor.  The controller must keep
    objects referenced by it alive and immutable through completion.  The
    queue intentionally does not inspect it or infer routing/capacity from it.
    """

    snapshot_id: str
    attempt_id: str
    lease_id: str
    path: TransferPath
    payload: Any = None

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "attempt_id", "lease_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.path, TransferPath):
            raise TypeError("path must be a TransferPath")

    @property
    def key(self) -> tuple[str, str, str]:
        return self.snapshot_id, self.attempt_id, self.lease_id


@dataclass(frozen=True, slots=True)
class PhysicalProgress:
    state: PhysicalState
    fence: Optional[FenceKind] = None
    error: Optional[str] = None
    result: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, PhysicalState):
            raise TypeError("state must be a PhysicalState")
        value = {} if self.result is None else copy.deepcopy(dict(self.result))
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "result", MappingProxyType(value))
        if self.state is PhysicalState.INFLIGHT:
            if self.fence is not None:
                raise ValueError("an in-flight operation cannot carry a fence")
            return
        if self.fence is None:
            raise ValueError("a terminal physical state requires a fence")
        if self.state is PhysicalState.SUCCEEDED and self.fence not in {
            FenceKind.DMA_COMPLETE,
            FenceKind.NO_IO_REQUIRED,
        }:
            raise ValueError("successful transfer requires a completion proof")
        if self.state is PhysicalState.CANCELLED and self.fence not in {
            FenceKind.NOT_POSTED,
            FenceKind.CANCEL_DRAINED,
        }:
            raise ValueError("cancelled transfer requires not-posted or drained proof")


@dataclass(frozen=True, slots=True)
class TransferCompletion:
    attempt: TransferAttempt
    state: PhysicalState
    fence: FenceKind
    error: Optional[str]
    submitted_at: float
    completed_at: float
    result: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        value = {} if self.result is None else copy.deepcopy(dict(self.result))
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "result", MappingProxyType(value))

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.completed_at - self.submitted_at)


class TransferExecutor(Protocol):
    """Rank-local physical engine injected into one path queue.

    ``submit`` and ``request_cancel`` may schedule asynchronous work but must
    return quickly.  They receive an edge callback and invoke it whenever a
    subsequent ``progress`` call can observe new state.  ``progress`` itself
    must be non-blocking.  A cancellation is complete only when ``progress``
    returns a terminal state with a real fence.
    """

    def submit(self, attempt: TransferAttempt, notify: Callable[[], None]) -> Any:
        ...

    def progress(self, handle: Any) -> PhysicalProgress:
        ...

    def request_cancel(self, handle: Any, notify: Callable[[], None]) -> None:
        ...


class TransferNotPosted(RuntimeError):
    """Executor proves that submit failed before any DMA could be posted."""


class TransferQueueError(RuntimeError):
    pass


class TransferIdentityConflict(TransferQueueError):
    pass


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    pending: int
    active: int
    lanes: int
    completed: int
    cancelled: int
    failed: int
    fatal_error: Optional[str]


@dataclass(slots=True)
class _Item:
    attempt: TransferAttempt
    callback: Callable[[TransferCompletion], None]
    submitted_at: float
    handle: Any = None
    starting: bool = False
    cancel_requested: bool = False
    cancel_sent: bool = False


class _LiveRegistry:
    """Prevent one generation from entering two physical queues at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owners: dict[str, tuple[TransferPath, str, str]] = {}

    def acquire(self, attempt: TransferAttempt) -> bool:
        identity = (attempt.path, attempt.attempt_id, attempt.lease_id)
        with self._lock:
            current = self._owners.get(attempt.snapshot_id)
            if current is None:
                self._owners[attempt.snapshot_id] = identity
                return True
            if current == identity:
                return False
            raise TransferIdentityConflict(
                f"snapshot {attempt.snapshot_id} is live as {current}, "
                f"cannot submit {identity}"
            )

    def release(self, attempt: TransferAttempt) -> None:
        identity = (attempt.path, attempt.attempt_id, attempt.lease_id)
        with self._lock:
            if self._owners.get(attempt.snapshot_id) != identity:
                raise TransferQueueError("live transfer identity changed before completion")
            self._owners.pop(attempt.snapshot_id)


class EventDrivenTransferQueue:
    """One path's bounded pending queue and independent physical lanes."""

    def __init__(
        self,
        path: TransferPath,
        executor: TransferExecutor,
        *,
        lanes: int,
        pending_capacity: int,
        registry: Optional[_LiveRegistry] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(path, TransferPath):
            raise TypeError("path must be a TransferPath")
        if int(lanes) < 1 or int(pending_capacity) < 1:
            raise ValueError("lanes and pending_capacity must be positive")
        self.path = path
        self.executor = executor
        self.lanes = int(lanes)
        self.pending_capacity = int(pending_capacity)
        self._registry = registry or _LiveRegistry()
        self._condition = threading.Condition()
        self._pending: deque[_Item] = deque()
        self._active: dict[tuple[str, str, str], _Item] = {}
        self._events: deque[tuple[str, str, str]] = deque()
        self._event_keys: set[tuple[str, str, str]] = set()
        self._stop = False
        self._fatal: Optional[BaseException] = None
        self._completed = 0
        self._cancelled = 0
        self._failed = 0
        self._thread = threading.Thread(
            target=self._run,
            name=name or f"agentic-transfer-{path.value}",
            daemon=True,
        )
        self._thread.start()

    def _check_health_locked(self) -> None:
        if self._fatal is not None:
            raise TransferQueueError(f"{self.path.value} queue failed") from self._fatal
        if self._stop:
            raise TransferQueueError(f"{self.path.value} queue is closed")

    def check_health(self) -> None:
        with self._condition:
            self._check_health_locked()

    def submit(
        self,
        attempt: TransferAttempt,
        callback: Callable[[TransferCompletion], None],
    ) -> bool:
        if attempt.path is not self.path:
            raise ValueError(f"attempt path {attempt.path.value} does not match queue {self.path.value}")
        if not callable(callback):
            raise TypeError("completion callback must be callable")
        with self._condition:
            self._check_health_locked()
            if len(self._pending) >= self.pending_capacity:
                raise queue.Full(self.path.value)
            acquired = self._registry.acquire(attempt)
            if not acquired:
                return False
            self._pending.append(_Item(attempt, callback, time.monotonic()))
            self._condition.notify()
            return True

    def cancel(self, snapshot_id: str, attempt_id: str, lease_id: str) -> bool:
        key = (str(snapshot_id), str(attempt_id), str(lease_id))
        completion = None
        item = None
        with self._condition:
            self._check_health_locked()
            for pending in tuple(self._pending):
                if pending.attempt.key == key:
                    self._pending.remove(pending)
                    item = pending
                    completion = self._make_completion(
                        pending,
                        PhysicalProgress(
                            PhysicalState.CANCELLED, FenceKind.NOT_POSTED
                        ),
                    )
                    break
            if completion is None:
                item = self._active.get(key)
                if item is None:
                    return False
                item.cancel_requested = True
                self._signal_locked(key)
                return True
        self._finish(item, completion)
        return True

    def _signal_locked(self, key: tuple[str, str, str]) -> None:
        if key not in self._event_keys:
            self._event_keys.add(key)
            self._events.append(key)
        self._condition.notify()

    def _notify(self, key: tuple[str, str, str]) -> None:
        with self._condition:
            if key in self._active:
                self._signal_locked(key)

    def _next_action(self):
        with self._condition:
            self._condition.wait_for(
                lambda: self._stop
                or self._fatal is not None
                or bool(self._events)
                or (bool(self._pending) and len(self._active) < self.lanes)
            )
            if self._fatal is not None:
                return "stop", None
            if self._events:
                key = self._events.popleft()
                self._event_keys.discard(key)
                item = self._active.get(key)
                if item is not None:
                    return "progress", item
            if self._pending and len(self._active) < self.lanes:
                item = self._pending.popleft()
                item.starting = True
                self._active[item.attempt.key] = item
                return "start", item
            if self._stop and not self._active and not self._pending:
                return "stop", None
            return "again", None

    def _run(self) -> None:
        while True:
            action, item = self._next_action()
            if action == "stop":
                return
            if action == "again":
                continue
            try:
                if action == "start":
                    self._start(item)
                else:
                    self._progress(item)
            except BaseException as exc:
                # An unexpected executor/control exception has no physical
                # fence proof.  Keep the active item and lane quarantined.
                with self._condition:
                    self._fatal = exc
                    self._condition.notify_all()
                logger.exception(
                    "Agentic %s queue retained an unfenced transfer",
                    self.path.value,
                )
                return

    def _start(self, item: _Item) -> None:
        key = item.attempt.key
        try:
            handle = self.executor.submit(
                item.attempt, lambda key=key: self._notify(key)
            )
        except TransferNotPosted as exc:
            completion = self._make_completion(
                item,
                PhysicalProgress(
                    PhysicalState.FAILED,
                    FenceKind.NOT_POSTED,
                    str(exc) or type(exc).__name__,
                ),
            )
            self._finish(item, completion)
            return
        if handle is None:
            raise RuntimeError("executor returned no physical transfer handle")
        with self._condition:
            current = self._active.get(key)
            if current is not item:
                raise RuntimeError("active transfer changed during submit")
            item.handle = handle
            item.starting = False
            self._signal_locked(key)

    def _progress(self, item: _Item) -> None:
        key = item.attempt.key
        if item.handle is None:
            # A notify may race executor.submit(); submit completion will wake
            # the queue again after installing the handle.
            return
        if item.cancel_requested and not item.cancel_sent:
            self.executor.request_cancel(
                item.handle, lambda key=key: self._notify(key)
            )
            item.cancel_sent = True
        progress = self.executor.progress(item.handle)
        if not isinstance(progress, PhysicalProgress):
            raise TypeError("executor.progress must return PhysicalProgress")
        if progress.state is PhysicalState.INFLIGHT:
            return
        completion = self._make_completion(item, progress)
        self._finish(item, completion)

    @staticmethod
    def _make_completion(
        item: _Item, progress: PhysicalProgress
    ) -> TransferCompletion:
        if progress.state is PhysicalState.INFLIGHT or progress.fence is None:
            raise ValueError("cannot complete an unfenced transfer")
        return TransferCompletion(
            attempt=item.attempt,
            state=progress.state,
            fence=progress.fence,
            error=progress.error,
            submitted_at=item.submitted_at,
            completed_at=time.monotonic(),
            result=progress.result,
        )

    def _finish(self, item: _Item, completion: TransferCompletion) -> None:
        with self._condition:
            self._active.pop(item.attempt.key, None)
            self._registry.release(item.attempt)
            if completion.state is PhysicalState.SUCCEEDED:
                self._completed += 1
            elif completion.state is PhysicalState.CANCELLED:
                self._cancelled += 1
            else:
                self._failed += 1
            self._condition.notify_all()
        try:
            item.callback(completion)
        except BaseException as exc:
            # The DMA is fenced but the lifecycle owner did not consume the
            # completion.  Stop admitting work so that it can recover/abort
            # explicitly rather than silently losing the owner transition.
            with self._condition:
                self._fatal = exc
                self._condition.notify_all()
            logger.exception("Agentic %s completion callback failed", self.path.value)

    def snapshot(self) -> QueueSnapshot:
        with self._condition:
            return QueueSnapshot(
                pending=len(self._pending),
                active=len(self._active),
                lanes=self.lanes,
                completed=self._completed,
                cancelled=self._cancelled,
                failed=self._failed,
                fatal_error=None if self._fatal is None else repr(self._fatal),
            )

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._pending or self._active:
                if self._fatal is not None:
                    self._check_health_locked()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, require_idle: bool = True, timeout: float = 5.0) -> None:
        if require_idle and not self.wait_idle(timeout):
            raise TransferQueueError(
                f"{self.path.value} queue still owns pending or in-flight work"
            )
        with self._condition:
            if not require_idle and (self._pending or self._active):
                raise TransferQueueError(
                    "closing live transfers would discard physical ownership"
                )
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            raise TransferQueueError(f"{self.path.value} queue worker did not stop")


class AgenticTransferQueues:
    """Four independent path queues sharing only a live-snapshot guard."""

    def __init__(
        self,
        executors: Mapping[TransferPath, TransferExecutor],
        *,
        lanes: Mapping[TransferPath, int],
        pending_capacity: Mapping[TransferPath, int],
    ) -> None:
        paths = set(TransferPath)
        if set(executors) != paths or set(lanes) != paths or set(pending_capacity) != paths:
            raise ValueError("executors, lanes and capacities must cover all four paths")
        registry = _LiveRegistry()
        self.queues = {
            path: EventDrivenTransferQueue(
                path,
                executors[path],
                lanes=lanes[path],
                pending_capacity=pending_capacity[path],
                registry=registry,
            )
            for path in TransferPath
        }

    def submit(
        self,
        attempt: TransferAttempt,
        callback: Callable[[TransferCompletion], None],
    ) -> bool:
        return self.queues[attempt.path].submit(attempt, callback)

    def cancel(self, attempt: TransferAttempt) -> bool:
        return self.queues[attempt.path].cancel(*attempt.key)

    def snapshot(self) -> dict[TransferPath, QueueSnapshot]:
        return {path: transfer_queue.snapshot() for path, transfer_queue in self.queues.items()}

    def check_health(self) -> None:
        for transfer_queue in self.queues.values():
            transfer_queue.check_health()

    def close(self, *, timeout: float = 5.0) -> None:
        for transfer_queue in self.queues.values():
            transfer_queue.close(timeout=timeout)
