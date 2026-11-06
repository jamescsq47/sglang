"""Rank-local, scheduler-independent NIXL Direct adapter for V2.

The group controller decides the route and owns the cross-TP transaction.  This
module only executes one rank's physical D->P or P->D shard.  Stock SGLang
``NixlKVSender``/``NixlKVReceiver`` objects remain the wire implementation;
their polling is isolated in a dedicated I/O pool and never runs on a model
scheduler thread.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from sglang.srt.disaggregation.agentic_direct_transfer import (
    AgenticDirectRuntime,
    create_agentic_direct_runtime,
)
from sglang.srt.disaggregation.agentic_group_protocol import GroupCommand
from sglang.srt.disaggregation.agentic_group_transfer import (
    CallbackPathHandler,
    PreparedRankTransfer,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    FenceKind,
    PhysicalProgress,
    PhysicalState,
    TransferAttempt,
    TransferExecutor,
    TransferPath,
)
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode


class DirectDirection(str, Enum):
    D2P = "d2p"
    P2D = "p2d"


class DirectEndpoint(str, Enum):
    SOURCE = "source"
    TARGET = "target"


class UnfencedNixlDirect(RuntimeError):
    """NIXL may still access the lease; the queue must quarantine it."""


@dataclass(frozen=True, slots=True)
class NixlDirectShard:
    """Immutable, rank-local page mapping resolved from a workset lease."""

    bootstrap_addr: str
    room: int
    page_indices: tuple[int, ...]
    state_indices: Optional[tuple[int, ...]] = None
    aux_index: int = 0
    prefill_dp_rank: int = 0
    destination_tp_ranks: tuple[int, ...] = (0,)
    pp_rank: int = 0

    def __post_init__(self) -> None:
        if not self.bootstrap_addr or int(self.room) < 0:
            raise ValueError("Direct shard requires bootstrap address and room")
        if not self.page_indices:
            raise ValueError("Direct shard must cover at least one page")
        if not self.destination_tp_ranks:
            raise ValueError("Direct source requires a destination TP rank")


def _cleanup_room(manager: Any, room: int, bootstrap_addr: str) -> None:
    for name in (
        "request_status",
        "failure_records",
        "required_prefill_response_num_table",
        "prefill_response_tracker",
        "transfer_infos",
        "transfer_statuses",
    ):
        table = getattr(manager, name, None)
        if table is not None:
            table.pop(int(room), None)
    tracker = getattr(manager, "addr_to_rooms_tracker", None)
    rooms = None if tracker is None else tracker.get(bootstrap_addr)
    if rooms is not None:
        rooms.discard(int(room))


class NixlDirectOperation:
    """One source or target shard, advanced only by the Direct I/O pool."""

    def __init__(
        self,
        runtime: AgenticDirectRuntime,
        endpoint: DirectEndpoint,
        shard: Optional[NixlDirectShard],
        *,
        ready_event: Any = None,
        shard_factory: Optional[Callable[[], NixlDirectShard]] = None,
    ) -> None:
        if shard is None and shard_factory is None:
            raise ValueError("Direct operation requires a shard or shard factory")
        self.runtime = runtime
        self.endpoint = DirectEndpoint(endpoint)
        self.shard = shard
        self.sender = None
        self.receiver = None
        self.sent = False
        self.posted = False
        self._launch_error: Optional[BaseException] = None
        self._cleaned = False
        self._ready_event = ready_event
        self._ready_waited = False
        self._shard_factory = shard_factory
        self._lock = threading.Lock()

    def wait_ready(
        self,
        cancel: Optional[threading.Event] = None,
        poll_interval: float = 0.001,
    ) -> bool:
        """Fence source Forward without making pre-post cancellation block.

        CUDA events expose ``query``; poll that edge from the I/O worker so a
        Direct admission timeout can cancel an operation that has not posted
        any transport access yet.  Once the event is ready, the immutable
        source descriptor may be materialized and ordinary NIXL fencing takes
        over.
        """

        if self._ready_waited:
            return True
        event = self._ready_event
        if event is not None:
            query = getattr(event, "query", None)
            if callable(query):
                while not bool(query()):
                    if cancel is not None and cancel.wait(poll_interval):
                        return False
            else:
                if cancel is not None and cancel.is_set():
                    return False
                event.synchronize()
        if cancel is not None and cancel.is_set():
            return False
        if self.shard is None:
            self.shard = self._shard_factory()
            self._shard_factory = None
        self._ready_waited = True
        return True

    @staticmethod
    def _pages(values: tuple[int, ...]):
        return np.asarray(values, dtype=np.int32)

    @staticmethod
    def _states(values: Optional[tuple[int, ...]]):
        return None if values is None else [int(value) for value in values]

    def _source_step(self) -> int:
        if self.sender is None:
            runtime_addr = self.runtime.bootstrap_addr
            if runtime_addr and runtime_addr != self.shard.bootstrap_addr:
                raise ValueError("source shard does not match its Direct runtime")
            self.sender = self.runtime.sender_class(
                mgr=self.runtime.manager,
                bootstrap_addr=self.shard.bootstrap_addr,
                bootstrap_room=int(self.shard.room),
                dest_tp_ranks=list(self.shard.destination_tp_ranks),
                pp_rank=int(self.shard.pp_rank),
            )
        status = self.sender.poll()
        if self.sent or status != KVPoll.WaitingForInput:
            return status
        self.sender.init(
            len(self.shard.page_indices), aux_index=int(self.shard.aux_index)
        )
        try:
            self.sender.send(
                self._pages(self.shard.page_indices),
                state_indices=self._states(self.shard.state_indices),
            )
            self.sent = True
            # A successful stock sender call has posted all physical handles.
            self.posted = True
        except BaseException as error:
            self._launch_error = error
            handles = tuple(getattr(self.sender, "xfer_handles", ()))
            self.posted = bool(handles)
            fence_failed_launch = getattr(self.sender, "fence_failed_launch", None)
            if fence_failed_launch is None:
                if self.posted:
                    raise UnfencedNixlDirect(
                        "sender launch failed after an untracked NIXL post"
                    ) from error
                return KVPoll.Failed
            status = fence_failed_launch(error)
            self.sent = True
            return status
        return self.sender.poll()

    def _target_step(self) -> int:
        if self.receiver is None:
            if not self.runtime.manager.try_ensure_parallel_info(
                self.shard.bootstrap_addr
            ):
                return KVPoll.Bootstrapping
            self.receiver = self.runtime.receiver_class(
                mgr=self.runtime.manager,
                bootstrap_addr=self.shard.bootstrap_addr,
                bootstrap_room=int(self.shard.room),
            )
            self.receiver.init(prefill_dp_rank=int(self.shard.prefill_dp_rank))
            if self.receiver.poll() == KVPoll.Failed:
                return KVPoll.Failed
            try:
                self.receiver.send_metadata(
                    self._pages(self.shard.page_indices),
                    aux_index=int(self.shard.aux_index),
                    state_indices=self._states(self.shard.state_indices),
                )
            except BaseException as error:
                self._launch_error = error
                self.posted = bool(
                    getattr(self.receiver, "started_transfer", False)
                )
                if not self.posted:
                    return KVPoll.Failed
                # Destination addresses may already be visible.  Do not claim
                # failure until the stock receiver observes its terminal state.
            else:
                self.posted = bool(
                    getattr(self.receiver, "started_transfer", True)
                )
        return self.receiver.poll()

    def step(self) -> int:
        return (
            self._source_step()
            if self.endpoint is DirectEndpoint.SOURCE
            else self._target_step()
        )

    @property
    def launch_error(self) -> Optional[BaseException]:
        return self._launch_error

    def cleanup(self) -> None:
        with self._lock:
            if self._cleaned:
                return
            receiver = self.receiver
            if receiver is not None:
                receiver.clear()
            if self.shard is not None:
                _cleanup_room(
                    self.runtime.manager,
                    int(self.shard.room),
                    self.shard.bootstrap_addr,
                )
            self._cleaned = True


@dataclass(slots=True)
class _DirectHandle:
    future: Future
    cancel: threading.Event
    operation: NixlDirectOperation
    start_signal: Any


class _PhysicalStartSignal:
    """Race-safe one-shot callback for the first real NIXL post."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callback: Optional[Callable[[], None]] = None
        self._fired = False

    def install(self, callback: Callable[[], None]) -> None:
        call = False
        with self._lock:
            if self._callback is not None:
                raise RuntimeError("Direct start callback was installed twice")
            self._callback = callback
            call = self._fired
        if call:
            callback()

    def fire(self) -> None:
        callback = None
        with self._lock:
            if self._fired:
                return
            self._fired = True
            callback = self._callback
        if callback is not None:
            callback()


class NixlDirectIOExecutor(TransferExecutor):
    """Dedicated polling pool that converts stock NIXL into edge completion."""

    def __init__(self, *, max_workers: int = 4, poll_interval: float = 0.001):
        self.poll_interval = max(0.0001, float(poll_interval))
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="agentic-nixl-direct",
        )

    def _run(
        self,
        operation: NixlDirectOperation,
        cancel: threading.Event,
        start_signal: _PhysicalStartSignal,
    ) -> PhysicalProgress:
        if not operation.wait_ready(cancel, self.poll_interval):
            return PhysicalProgress(
                PhysicalState.CANCELLED, FenceKind.NOT_POSTED
            )
        while True:
            if cancel.is_set() and not operation.posted:
                return PhysicalProgress(
                    PhysicalState.CANCELLED, FenceKind.NOT_POSTED
                )
            try:
                status = operation.step()
            except UnfencedNixlDirect:
                raise
            except BaseException as error:
                if operation.posted:
                    raise UnfencedNixlDirect(
                        "Direct transport status is unknown after NIXL post"
                    ) from error
                return PhysicalProgress(
                    PhysicalState.FAILED, FenceKind.NOT_POSTED, str(error)
                )
            if operation.posted:
                start_signal.fire()
            if status == KVPoll.Success:
                if cancel.is_set():
                    return PhysicalProgress(
                        PhysicalState.CANCELLED, FenceKind.CANCEL_DRAINED
                    )
                return PhysicalProgress(
                    PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE
                )
            if status == KVPoll.Failed:
                detail = str(operation.launch_error or "NIXL Direct failed")
                if cancel.is_set():
                    return PhysicalProgress(
                        PhysicalState.CANCELLED,
                        FenceKind.CANCEL_DRAINED,
                        detail,
                    )
                return PhysicalProgress(
                    PhysicalState.FAILED,
                    (
                        FenceKind.ERROR_DRAINED
                        if operation.posted
                        else FenceKind.NOT_POSTED
                    ),
                    detail,
                )
            time.sleep(self.poll_interval)

    def submit(self, attempt: TransferAttempt, notify: Callable[[], None]):
        operation = attempt.payload
        if not isinstance(operation, NixlDirectOperation):
            raise TypeError("NIXL Direct requires NixlDirectOperation payload")
        cancel = threading.Event()
        start_signal = _PhysicalStartSignal()
        future = self._pool.submit(self._run, operation, cancel, start_signal)
        handle = _DirectHandle(future, cancel, operation, start_signal)
        future.add_done_callback(lambda _future: notify())
        return handle

    def install_started_callback(
        self, handle: _DirectHandle, callback: Callable[[], None]
    ) -> None:
        handle.start_signal.install(callback)

    def progress(self, handle: _DirectHandle) -> PhysicalProgress:
        if not handle.future.done():
            return PhysicalProgress(PhysicalState.INFLIGHT)
        if handle.future.cancelled():
            return PhysicalProgress(PhysicalState.CANCELLED, FenceKind.NOT_POSTED)
        return handle.future.result()

    def request_cancel(
        self, handle: _DirectHandle, notify: Callable[[], None]
    ) -> None:
        handle.cancel.set()
        if handle.future.cancel():
            notify()

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)


def _path(direction: DirectDirection) -> TransferPath:
    return (
        TransferPath.D2P_DIRECT
        if direction is DirectDirection.D2P
        else TransferPath.P2D_DIRECT
    )


def make_nixl_direct_handler(
    *,
    direction: DirectDirection,
    endpoint: DirectEndpoint,
    runtime: AgenticDirectRuntime,
    descriptor: Callable[[GroupCommand], NixlDirectShard],
    commit: Optional[
        Callable[[GroupCommand, NixlDirectShard, Any], None]
    ] = None,
    abort: Optional[
        Callable[[GroupCommand, NixlDirectShard, Any], None]
    ] = None,
) -> CallbackPathHandler:
    expected_path = _path(DirectDirection(direction))
    endpoint = DirectEndpoint(endpoint)

    def prepare(command: GroupCommand) -> PreparedRankTransfer:
        header = command.payload.get("agentic_data_plane", {})
        if header.get("path") != expected_path.value:
            raise ValueError("Direct command direction does not match rank adapter")
        shard = descriptor(command)
        if not isinstance(shard, NixlDirectShard):
            raise TypeError("Direct descriptor must return NixlDirectShard")
        operation = NixlDirectOperation(runtime, endpoint, shard)
        return PreparedRankTransfer(operation)

    def committed(command, prepared, completion) -> None:
        operation = prepared.transfer_payload
        if commit is not None:
            commit(command, operation.shard, completion)
        operation.cleanup()

    def aborted(command, prepared, completion) -> None:
        operation = prepared.transfer_payload
        if abort is not None:
            abort(command, operation.shard, completion)
        operation.cleanup()

    return CallbackPathHandler(prepare, commit=committed, abort=aborted)


@dataclass(frozen=True, slots=True)
class NixlDirectRankPath:
    path: TransferPath
    runtime: AgenticDirectRuntime
    executor: NixlDirectIOExecutor
    handler: CallbackPathHandler


def create_nixl_direct_rank_path(
    *,
    direction: DirectDirection,
    endpoint: DirectEndpoint,
    kv_pool: Any,
    server_args: Any,
    engine_rank: int,
    pp_rank: int,
    gpu_id: int,
    total_kv_heads: int,
    descriptor: Callable[[GroupCommand], NixlDirectShard],
    bootstrap_port: Optional[int] = None,
    commit: Optional[Callable[[GroupCommand, NixlDirectShard, Any], None]] = None,
    abort: Optional[Callable[[GroupCommand, NixlDirectShard, Any], None]] = None,
    max_workers: int = 4,
    poll_interval: float = 0.001,
) -> NixlDirectRankPath:
    """Create the isolated stock-NIXL runtime plus its V2 queue adapter."""

    endpoint = DirectEndpoint(endpoint)
    runtime = create_agentic_direct_runtime(
        role=(
            DisaggregationMode.PREFILL
            if endpoint is DirectEndpoint.SOURCE
            else DisaggregationMode.DECODE
        ),
        kv_pool=kv_pool,
        server_args=server_args,
        engine_rank=int(engine_rank),
        pp_rank=int(pp_rank),
        gpu_id=int(gpu_id),
        total_kv_heads=int(total_kv_heads),
        bootstrap_port=bootstrap_port,
    )
    executor = NixlDirectIOExecutor(
        max_workers=max_workers, poll_interval=poll_interval
    )
    handler = make_nixl_direct_handler(
        direction=direction,
        endpoint=endpoint,
        runtime=runtime,
        descriptor=descriptor,
        commit=commit,
        abort=abort,
    )
    return NixlDirectRankPath(_path(DirectDirection(direction)), runtime, executor, handler)


__all__ = [
    "DirectDirection",
    "DirectEndpoint",
    "NixlDirectIOExecutor",
    "NixlDirectOperation",
    "NixlDirectRankPath",
    "NixlDirectShard",
    "UnfencedNixlDirect",
    "create_nixl_direct_rank_path",
    "make_nixl_direct_handler",
]
