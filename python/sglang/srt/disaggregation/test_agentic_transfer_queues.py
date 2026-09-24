from __future__ import annotations

import dataclasses
import queue
import threading
import time

import pytest

from sglang.srt.disaggregation.agentic_transfer_queues import (
    AgenticTransferQueues,
    EventDrivenTransferQueue,
    FenceKind,
    PhysicalProgress,
    PhysicalState,
    TransferAttempt,
    TransferIdentityConflict,
    TransferNotPosted,
    TransferPath,
)


class ManualExecutor:
    """A transport whose physical fences are advanced explicitly by tests."""

    def __init__(self):
        self._condition = threading.Condition()
        self._next = 0
        self.records = {}
        self.submit_error = None

    def submit(self, attempt, notify):
        if self.submit_error is not None:
            raise self.submit_error
        with self._condition:
            self._next += 1
            handle = self._next
            self.records[handle] = {
                "attempt": attempt,
                "notify": notify,
                "progress": PhysicalProgress(PhysicalState.INFLIGHT),
                "cancel_requested": False,
            }
            self._condition.notify_all()
            return handle

    def progress(self, handle):
        with self._condition:
            return self.records[handle]["progress"]

    def request_cancel(self, handle, notify):
        del notify
        with self._condition:
            self.records[handle]["cancel_requested"] = True
            self._condition.notify_all()

    def wait_handles(self, count, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self.records) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("executor did not receive expected submissions")
                self._condition.wait(remaining)
            return tuple(sorted(self.records))

    def wait_cancel(self, handle, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self.records[handle]["cancel_requested"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("executor did not receive cancellation")
                self._condition.wait(remaining)

    def set_progress(self, handle, progress):
        with self._condition:
            record = self.records[handle]
            record["progress"] = progress
            notify = record["notify"]
        notify()


def attempt(path, suffix="one"):
    return TransferAttempt(
        snapshot_id=f"request:{suffix}",
        attempt_id=f"attempt:{suffix}",
        lease_id=f"lease:{suffix}",
        path=path,
        payload=("rank-local-descriptor", suffix),
    )


def make_queue(path=TransferPath.D2P_DIRECT, *, lanes=1, capacity=8):
    executor = ManualExecutor()
    transfer_queue = EventDrivenTransferQueue(
        path, executor, lanes=lanes, pending_capacity=capacity
    )
    return transfer_queue, executor


def test_attempt_identity_is_immutable():
    value = attempt(TransferPath.D2P_DIRECT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.attempt_id = "changed"


def test_success_requires_real_dma_fence_and_calls_completion():
    transfer_queue, executor = make_queue()
    completions = []
    item = attempt(TransferPath.D2P_DIRECT)
    assert transfer_queue.submit(item, completions.append)
    (handle,) = executor.wait_handles(1)

    # A wakeup with an in-flight observation neither frees the lane nor calls
    # the lifecycle completion callback.
    executor.set_progress(handle, PhysicalProgress(PhysicalState.INFLIGHT))
    time.sleep(0.02)
    assert transfer_queue.snapshot().active == 1
    assert completions == []

    executor.set_progress(
        handle,
        PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE),
    )
    assert transfer_queue.wait_idle(2.0)
    assert len(completions) == 1
    assert completions[0].attempt is item
    assert completions[0].fence is FenceKind.DMA_COMPLETE
    transfer_queue.close()


def test_active_cancel_retains_lane_until_executor_reports_drain_fence():
    transfer_queue, executor = make_queue()
    completions = []
    item = attempt(TransferPath.D2P_HOST)
    # Use a matching path queue for this test.
    transfer_queue.close()
    transfer_queue = EventDrivenTransferQueue(
        TransferPath.D2P_HOST, executor, lanes=1, pending_capacity=8
    )
    transfer_queue.submit(item, completions.append)
    (handle,) = executor.wait_handles(1)

    assert transfer_queue.cancel(*item.key)
    executor.wait_cancel(handle)
    assert transfer_queue.snapshot().active == 1
    assert completions == []

    executor.set_progress(
        handle,
        PhysicalProgress(PhysicalState.CANCELLED, FenceKind.CANCEL_DRAINED),
    )
    assert transfer_queue.wait_idle(2.0)
    assert completions[0].state is PhysicalState.CANCELLED
    assert completions[0].fence is FenceKind.CANCEL_DRAINED
    transfer_queue.close()


def test_queued_cancel_has_explicit_not_posted_proof():
    transfer_queue, executor = make_queue(lanes=1)
    completions = []
    first = attempt(TransferPath.D2P_DIRECT, "first")
    second = attempt(TransferPath.D2P_DIRECT, "second")
    transfer_queue.submit(first, completions.append)
    (handle,) = executor.wait_handles(1)
    transfer_queue.submit(second, completions.append)
    assert transfer_queue.cancel(*second.key)

    assert len(completions) == 1
    assert completions[0].attempt is second
    assert completions[0].fence is FenceKind.NOT_POSTED
    executor.set_progress(
        handle,
        PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE),
    )
    assert transfer_queue.wait_idle(2.0)
    transfer_queue.close()


def test_submit_failure_can_only_complete_with_not_posted_proof():
    transfer_queue, executor = make_queue()
    executor.submit_error = TransferNotPosted("descriptor validation failed")
    completions = []
    transfer_queue.submit(attempt(TransferPath.D2P_DIRECT), completions.append)
    assert transfer_queue.wait_idle(2.0)
    assert completions[0].state is PhysicalState.FAILED
    assert completions[0].fence is FenceKind.NOT_POSTED
    transfer_queue.close()


def test_pending_capacity_is_bounded():
    transfer_queue, executor = make_queue(lanes=1, capacity=1)
    first = attempt(TransferPath.D2P_DIRECT, "first")
    second = attempt(TransferPath.D2P_DIRECT, "second")
    third = attempt(TransferPath.D2P_DIRECT, "third")
    transfer_queue.submit(first, lambda _: None)
    (handle,) = executor.wait_handles(1)
    transfer_queue.submit(second, lambda _: None)
    with pytest.raises(queue.Full):
        transfer_queue.submit(third, lambda _: None)

    executor.set_progress(
        handle,
        PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE),
    )
    handles = executor.wait_handles(2)
    executor.set_progress(
        handles[-1],
        PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE),
    )
    assert transfer_queue.wait_idle(2.0)
    transfer_queue.close()


def test_four_paths_have_independent_lanes_and_global_live_guard():
    executors = {path: ManualExecutor() for path in TransferPath}
    queues = AgenticTransferQueues(
        executors,
        lanes={path: 1 for path in TransferPath},
        pending_capacity={path: 4 for path in TransferPath},
    )
    completions = []
    slow = attempt(TransferPath.D2P_HOST, "blocked")
    direct = attempt(TransferPath.P2D_DIRECT, "fast")
    queues.submit(slow, completions.append)
    queues.submit(direct, completions.append)
    executors[TransferPath.D2P_HOST].wait_handles(1)
    (direct_handle,) = executors[TransferPath.P2D_DIRECT].wait_handles(1)

    # The blocked D->P Host lane does not prevent P->D Direct completion.
    executors[TransferPath.P2D_DIRECT].set_progress(
        direct_handle,
        PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE),
    )
    deadline = time.monotonic() + 2.0
    while not completions and time.monotonic() < deadline:
        time.sleep(0.005)
    assert [entry.attempt for entry in completions] == [direct]
    assert queues.snapshot()[TransferPath.D2P_HOST].active == 1

    conflict = TransferAttempt(
        slow.snapshot_id,
        "new-attempt",
        "new-lease",
        TransferPath.D2P_DIRECT,
    )
    with pytest.raises(TransferIdentityConflict):
        queues.submit(conflict, completions.append)

    slow_handle = next(iter(executors[TransferPath.D2P_HOST].records))
    executors[TransferPath.D2P_HOST].set_progress(
        slow_handle,
        PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE),
    )
    for transfer_queue in queues.queues.values():
        assert transfer_queue.wait_idle(2.0)
    queues.close()


def test_terminal_progress_without_fence_is_rejected():
    with pytest.raises(ValueError):
        PhysicalProgress(PhysicalState.FAILED)
