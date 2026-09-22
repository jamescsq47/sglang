"""Opt-in native last-reference bridge to the sole CPU workset authority.

No free list exists here. Only the native scheduler/Radix owner thread may
submit raw-index frees. It must obey the existing exactly-once last-reference
contract; this is NOT a generic transport/cancellation cleanup interface.
Those callers must retain exact LeaseKey/range provenance. Capture epoch only
detects address reuse AFTER submission, not an invalid stale raw free first
submitted after reuse. Such a call was already invalid for the native allocator.

free() snapshots indices on the producer stream without CPU readback/wait.
A bounded separate mirror worker waits that event and produces immutable CPU
receipts. The injected nonblocking sink routes them to the existing controller
and all-rank return protocol. Sink completion means receipt acceptance, NOT
physical page reuse; only the authoritative all-rank commit returns capacity.
Rank-local ticket ordinals must never be used to pair different TP ranks.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading

import torch

from sglang.srt.disaggregation.agentic_workset_ledger import LeaseKey


class NativeFreeFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeFreeReceipt:
    ticket_id: str
    rank: int
    resource: str
    allocation_sequence: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointReturn:
    key: LeaseKey
    slot: int
    device_indices: torch.Tensor
    event: object


class LeaseCheckpointRotation:
    """Two output slots remain owned by one exact workset through all chunks.

    This is a per-lease consumption cursor, not another global allocator.
    Radix temporarily owns a borrowed view; its native last-reference free
    returns that view to this lease. Closing the Req returns only spare slots;
    donated views retain their lease provenance until Radix eventually frees
    them. Unique active request-generation enforcement is an integration
    requirement: a still-shared borrowed checkpoint cannot be overwritten.
    """

    def __init__(self, key, slots, device_indices, on_return):
        if (not isinstance(key, LeaseKey) or len(slots) != 2
                or any(type(slot) is not int or slot < 1 for slot in slots)
                or len(set(slots)) != 2 or device_indices.numel() != 2):
            raise ValueError("rotation requires an exact lease and two distinct prepared slots")
        self.key, self.slots = key, tuple(slots)
        self._owner, self._on_return = threading.current_thread(), on_return
        self._cleanup_owner = None
        self._views = tuple(device_indices[i:i + 1] for i in range(2))
        self._borrowed = [None, None]
        self._states, self._events, self._returns = ["spare"] * 2, [None] * 2, {}
        self._cursor, self._closed, self._failure = 0, False, None

    def _native(self):
        if (threading.current_thread() not in (self._owner, self._cleanup_owner)
                or self._failure is not None):
            raise NativeFreeFailure("checkpoint rotation unavailable; exact lease retained") from self._failure

    def allow_cleanup_owner(self, owner):
        """Transfer final close permission after the request left Prefill."""
        if threading.current_thread() is not owner:
            raise NativeFreeFailure("checkpoint cleanup owner must register itself")
        if self._cleanup_owner not in (None, owner):
            raise NativeFreeFailure("checkpoint cleanup owner changed")
        self._cleanup_owner = owner

    def can_take(self):
        self._native()
        return not self._closed and self._states[self._cursor % 2] == "spare"

    def take(self):
        self._native()
        if not self.can_take():
            raise NativeFreeFailure("promised checkpoint is still referenced; no unreserved fallback")
        index = self._cursor % 2
        view, event = self._views[index][:], self._events[index]
        if event is not None:
            torch.cuda.current_stream(view.device).wait_event(event)
        self._states[index] = "borrowed"
        view._agentic_checkpoint_rotation = self
        view._agentic_checkpoint_slot = index
        self._borrowed[index] = view
        self._cursor += 1
        return view

    def release(self, view):
        self._native()
        index = getattr(view, "_agentic_checkpoint_slot", None)
        if (type(index) is not int or index not in (0, 1)
                or self._borrowed[index] is not view or self._states[index] != "borrowed"):
            raise NativeFreeFailure("duplicate or foreign checkpoint return")
        try:
            if view.is_cuda:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(view.device))
                self._events[index] = event
            self._states[index] = "spare"
            self._borrowed[index] = None
            if self._closed:
                self._retire(index)
        except BaseException as error:
            self._failure = error
            raise

    def _retire(self, index):
        # on_return merely enqueues the original key/slot and real local fence;
        # it cannot recycle an address without the ordinary all-rank protocol.
        receipt = CheckpointReturn(self.key, self.slots[index], self._views[index], self._events[index])
        future = self._on_return(receipt)
        if not isinstance(future, Future):
            raise NativeFreeFailure("checkpoint return sink did not return a Future")
        self._returns[index] = future
        self._states[index] = "retiring"
        def accepted(done):
            try:
                done.result()
            except BaseException as error:
                self._failure = error
        future.add_done_callback(accepted)

    def close(self):
        self._native()
        if self._closed:
            return
        self._closed = True
        try:
            for index, state in enumerate(self._states):
                if state == "spare":
                    self._retire(index)
        except BaseException as error:
            self._failure = error
            raise


def install_checkpoint_rotation(req, key, slots, device_indices, on_return):
    """Native ownership handoff; descriptors must already be controller-ready."""
    if getattr(req, "_agentic_checkpoint_rotation", None) is not None:
        raise NativeFreeFailure("checkpoint rotation already installed")
    previous = getattr(req, "_agentic_mamba_prefill_checkpoint", None)
    if previous is not None and previous is not device_indices:
        raise NativeFreeFailure("rotation must adopt the exact existing output reservation")
    rotation = LeaseCheckpointRotation(key, slots, device_indices, on_return)
    req._agentic_checkpoint_rotation = rotation
    # The same indices must not also be freed as an anonymous checkpoint list.
    req._agentic_mamba_prefill_checkpoint = None
    return rotation


class NativeLastRefFreeAdapter:
    """Construct on the native owner thread; install only into an empty pool.

    counts() is an in-process immutable CPU authority view. on_ready(receipt)
    must enqueue nonblocking work and return a Future for acceptance. It must
    retain/route every receipt, and cannot assume one rank proves another's
    last reference. No CUDA wait is ever performed on the controller writer.
    """

    def __init__(self, *, incarnation, rank, page_size, counts, on_ready,
                 max_pending=256):
        if (not isinstance(incarnation, str) or not incarnation
                or type(rank) is not int or rank < 0
                or type(page_size) is not int or page_size < 1
                or type(max_pending) is not int or max_pending < 1):
            raise ValueError("invalid native free adapter identity/capacity")
        self.incarnation, self.rank, self.page_size = incarnation, rank, page_size
        self._counts, self._on_ready = counts, on_ready
        self._max_pending = max_pending
        self._owner = threading.current_thread()
        self._cleanup_owner = None
        self._cleanup_state_lock = None
        self._lock = threading.RLock()
        self._pending, self._next = {}, 0
        self._group = None
        self._failure, self._closed = None, False
        self._mirror_pool = ThreadPoolExecutor(1, thread_name_prefix="workset-free-mirror")

    def check_health(self):
        with self._lock:
            if self._closed or self._failure is not None:
                raise NativeFreeFailure("native free bridge unavailable; ownership retained") from self._failure

    def _native(self):
        self.check_health()
        current = threading.current_thread()
        if current is self._cleanup_owner:
            if not self._cleanup_state_lock._is_owned():
                raise NativeFreeFailure("native cleanup requires the scheduler state lock")
        elif current is not self._owner:
            raise NativeFreeFailure("raw free is restricted to the native last-reference owner")

    def allow_cleanup_owner(self, owner, state_lock):
        """Permit one P->D retirement worker under the native state lock."""
        if threading.current_thread() is not owner:
            raise NativeFreeFailure("native cleanup owner must register itself")
        with self._lock:
            if self._cleanup_owner not in (None, owner):
                raise NativeFreeFailure("native cleanup owner changed")
            self._cleanup_owner = owner
            self._cleanup_state_lock = state_lock

    def available_size(self, resource):
        self.check_health()
        counts = self._counts()
        if resource == "attention":
            return counts.free_pages * self.page_size
        if resource == "mamba":
            return counts.free_mamba_slots
        raise ValueError("unknown native resource")

    def free_group_begin(self):
        self._native()
        if self._group is not None:
            raise NativeFreeFailure("nested native free groups are unsupported")
        self._group = []

    def free_group_end(self):
        self._native()
        if self._group is None:
            raise NativeFreeFailure("native free group was not begun")
        group, self._group = self._group, None
        for record in group:
            self._submit_mirror(record)

    def free(self, indices, *, resource):
        self._native()
        if resource not in {"attention", "mamba"}:
            raise ValueError("unknown native resource")
        if not torch.is_tensor(indices) or indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("native free requires a one-dimensional integer index tensor")
        if indices.numel() == 0:
            return None
        future = Future()
        future.set_running_or_notify_cancel()
        with self._lock:
            if len(self._pending) >= self._max_pending:
                raise NativeFreeFailure("native free receipt queue full; no return accepted")
            self._next += 1
            identity = f"{self.incarnation}:rank{self.rank}:native-free{self._next}"
            record = dict(ticket_id=identity, resource=resource, future=future,
                          allocation_sequence=self._counts().sequence,
                          original=indices, snapshot=None, event=None)
            self._pending[identity] = record
        try:
            record["snapshot"] = indices.detach().clone()
            if indices.is_cuda:
                event = torch.cuda.Event()
                record["event"] = event
                event.record(torch.cuda.current_stream(indices.device))
            if self._group is not None:
                self._group.append(record)
            else:
                self._submit_mirror(record)
        except BaseException as error:
            self._fail(record, error)
            raise
        return future

    def observe_event(self, event):
        """Observe an exact owner's producer fence on the existing worker.

        The caller already retains LeaseKey/range provenance. This proves only
        local device quiescence, never last-reference or a TP-wide free. None
        is permitted only for a CPU/no-device-operation scope. Future callbacks
        must enqueue nonblocking control work, not run CUDA or wait for RPC.
        """
        self._native()
        future = Future()
        future.set_running_or_notify_cancel()
        with self._lock:
            if len(self._pending) >= self._max_pending:
                raise NativeFreeFailure("native fence queue full; no observation accepted")
            self._next += 1
            identity = f"{self.incarnation}:rank{self.rank}:exact-fence{self._next}"
            record = dict(ticket_id=identity, future=future, event=event,
                          exact_event=True, receipt=True)
            self._pending[identity] = record
        self._submit_mirror(record)
        return future

    def _submit_mirror(self, record):
        try:
            exact = record.get("exact_event", False)
            job = self._mirror_pool.submit(self._observe if exact else self._mirror, record)
            record["mirror"] = job
            job.add_done_callback(lambda done: self._accepted(record, done)
                                  if exact else self._mirrored(record, done))
        except BaseException as error:
            self._fail(record, error)
            raise

    @staticmethod
    def _observe(record):
        if record["event"] is not None:
            record["event"].synchronize()
        return True

    def _mirror(self, record):
        # Dedicated mirror thread only. This wait must never migrate to the
        # native Forward thread or the ledger's single writer.
        event, snapshot = record["event"], record["snapshot"]
        if event is not None:
            event.synchronize()
        values = tuple(int(value) for value in snapshot.cpu().tolist())
        return NativeFreeReceipt(record["ticket_id"], self.rank, record["resource"],
                                 record["allocation_sequence"], values)

    def _mirrored(self, record, done):
        try:
            receipt = done.result()
            self.check_health()
            accepted = self._on_ready(receipt)
            if not isinstance(accepted, Future):
                raise NativeFreeFailure("native receipt sink did not return a Future")
            record["receipt"], record["accepted"] = receipt, accepted
            accepted.add_done_callback(lambda result: self._accepted(record, result))
        except BaseException as error:
            self._fail(record, error)

    def _accepted(self, record, result):
        try:
            result.result()
            with self._lock:
                if self._failure is not None or self._closed:
                    raise NativeFreeFailure("native receipt acceptance after bridge stopped")
                self._pending.pop(record["ticket_id"])
            record["future"].set_result(record["receipt"])
        except BaseException as error:
            self._fail(record, error)

    def _fail(self, record, error):
        # Retain the record/index snapshots on every ambiguous failure. No
        # local error, lost ACK or shutdown is a physical reuse permission.
        with self._lock:
            self._failure = self._failure or error
            record["error"] = error
        if not record["future"].done():
            record["future"].set_exception(error)

    def shutdown(self, *, wait=True):
        with self._lock:
            self._closed = True
            for record in self._pending.values():
                if not record["future"].done():
                    record["future"].set_exception(NativeFreeFailure("bridge stopped; ownership retained"))
        self._mirror_pool.shutdown(wait=wait, cancel_futures=False)
