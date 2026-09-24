from __future__ import annotations

import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
    GroupCommand,
    Owner,
)
from sglang.srt.disaggregation.agentic_nixl_direct_adapter import (
    DirectDirection,
    DirectEndpoint,
    NixlDirectIOExecutor,
    NixlDirectOperation,
    NixlDirectShard,
    make_nixl_direct_handler,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    FenceKind,
    PhysicalState,
    TransferAttempt,
    TransferPath,
)
from sglang.srt.disaggregation.base.conn import KVPoll


class _Bus:
    def __init__(self, *, auto_complete=True, raise_after_post=False):
        self.lock = threading.Lock()
        self.metadata = False
        self.sent = False
        self.completed = False
        self.auto_complete = auto_complete
        self.raise_after_post = raise_after_post


class _Manager:
    def __init__(self, bus):
        self.bus = bus
        self.request_status = {}
        self.failure_records = {}
        self.required_prefill_response_num_table = {}
        self.prefill_response_tracker = {}
        self.transfer_infos = {}
        self.transfer_statuses = {}
        self.addr_to_rooms_tracker = defaultdict(set)

    def try_ensure_parallel_info(self, _bootstrap_addr):
        return True


class _Sender:
    def __init__(self, mgr, bootstrap_addr, bootstrap_room, **_kwargs):
        self.kv_mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.xfer_handles = []
        self.sent = False
        self.launch_failed = False
        self.launch_exception = None

    def init(self, count, aux_index=None):
        assert count > 0 and aux_index is not None

    def send(self, _indices, state_indices=None):
        del state_indices
        with self.kv_mgr.bus.lock:
            self.xfer_handles.append("native-handle")
            self.sent = True
            self.kv_mgr.bus.sent = True
            if self.kv_mgr.bus.raise_after_post:
                raise RuntimeError("partial native launch")
            if self.kv_mgr.bus.auto_complete:
                self.kv_mgr.bus.completed = True

    def fence_failed_launch(self, error):
        self.launch_failed = True
        self.launch_exception = error
        return self.poll()

    def poll(self):
        with self.kv_mgr.bus.lock:
            if self.launch_failed:
                return (
                    KVPoll.Failed
                    if self.kv_mgr.bus.completed
                    else KVPoll.Transferring
                )
            if self.sent:
                return (
                    KVPoll.Success
                    if self.kv_mgr.bus.completed
                    else KVPoll.Transferring
                )
            return (
                KVPoll.WaitingForInput
                if self.kv_mgr.bus.metadata
                else KVPoll.Bootstrapping
            )


class _Receiver:
    def __init__(self, mgr, bootstrap_addr, bootstrap_room):
        self.kv_mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.started_transfer = False
        self.cleared = False

    def init(self, prefill_dp_rank):
        assert prefill_dp_rank == 0

    def send_metadata(self, _indices, aux_index=None, state_indices=None):
        del state_indices
        assert aux_index is not None
        with self.kv_mgr.bus.lock:
            self.started_transfer = True
            self.kv_mgr.bus.metadata = True

    def poll(self):
        with self.kv_mgr.bus.lock:
            if not self.started_transfer:
                return KVPoll.WaitingForInput
            return (
                KVPoll.Success
                if self.kv_mgr.bus.completed
                else KVPoll.Transferring
            )

    def clear(self):
        self.cleared = True


def _runtime(bus, *, source):
    return SimpleNamespace(
        manager=_Manager(bus),
        bootstrap_addr="source:30000" if source else None,
        sender_class=_Sender,
        receiver_class=_Receiver,
    )


def _shard():
    return NixlDirectShard(
        "source:30000",
        17,
        (1, 2, 3),
        state_indices=(4,),
        aux_index=0,
        destination_tp_ranks=(0,),
    )


def _command(direction):
    path = (
        TransferPath.D2P_DIRECT
        if direction is DirectDirection.D2P
        else TransferPath.P2D_DIRECT
    )
    return GroupCommand(
        GenerationKey("run", "snapshot", 1),
        1,
        1,
        CommandKind.PREPARE,
        Owner.D_GPU if direction is DirectDirection.D2P else Owner.P_GPU,
        Owner.PREFILL_READY
        if direction is DirectDirection.D2P
        else Owner.DECODE_READY,
        "lease",
        {
            "agentic_data_plane": {
                "version": 1,
                "path": path.value,
                "operation": "direct",
            },
            "transfer": {},
        },
    )


def _attempt(path, operation):
    return TransferAttempt("snapshot", "1", "lease", path, operation)


def _wait(executor, handle, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = executor.progress(handle)
        if progress.state is not PhysicalState.INFLIGHT:
            return progress
        time.sleep(0.001)
    raise AssertionError("Direct executor did not become terminal")


@pytest.mark.parametrize("direction", list(DirectDirection))
def test_source_and_target_use_stock_nixl_until_real_dma_fence(direction):
    bus = _Bus(auto_complete=True)
    source_runtime = _runtime(bus, source=True)
    target_runtime = _runtime(bus, source=False)
    command = _command(direction)
    source_handler = make_nixl_direct_handler(
        direction=direction,
        endpoint=DirectEndpoint.SOURCE,
        runtime=source_runtime,
        descriptor=lambda _command: _shard(),
    )
    target_handler = make_nixl_direct_handler(
        direction=direction,
        endpoint=DirectEndpoint.TARGET,
        runtime=target_runtime,
        descriptor=lambda _command: _shard(),
    )
    source = source_handler.prepare(command).transfer_payload
    target = target_handler.prepare(command).transfer_payload
    path = (
        TransferPath.D2P_DIRECT
        if direction is DirectDirection.D2P
        else TransferPath.P2D_DIRECT
    )
    executor = NixlDirectIOExecutor(max_workers=2, poll_interval=0.0002)
    events = [threading.Event(), threading.Event()]
    try:
        source_handle = executor.submit(_attempt(path, source), events[0].set)
        target_handle = executor.submit(_attempt(path, target), events[1].set)
        assert events[0].wait(3) and events[1].wait(3)
        for progress in (
            executor.progress(source_handle),
            executor.progress(target_handle),
        ):
            assert progress.state is PhysicalState.SUCCEEDED
            assert progress.fence is FenceKind.DMA_COMPLETE
        assert bus.metadata and bus.sent and bus.completed
    finally:
        executor.close()

def test_cancel_after_post_waits_for_native_terminal_fence():
    bus = _Bus(auto_complete=False)
    bus.metadata = True
    executor = NixlDirectIOExecutor(max_workers=1, poll_interval=0.0002)
    operation = NixlDirectOperation(
        _runtime(bus, source=True), DirectEndpoint.SOURCE, _shard()
    )
    notified = threading.Event()
    handle = executor.submit(
        _attempt(TransferPath.D2P_DIRECT, operation), notified.set
    )
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not bus.sent:
            time.sleep(0.001)
        assert bus.sent
        executor.request_cancel(handle, notified.set)
        assert executor.progress(handle).state is PhysicalState.INFLIGHT
        with bus.lock:
            bus.completed = True
        assert notified.wait(3)
        progress = _wait(executor, handle)
        assert progress.state is PhysicalState.CANCELLED
        assert progress.fence is FenceKind.CANCEL_DRAINED
    finally:
        executor.close()


def test_partial_native_launch_is_not_failed_until_handle_is_fenced():
    bus = _Bus(auto_complete=False, raise_after_post=True)
    bus.metadata = True
    executor = NixlDirectIOExecutor(max_workers=1, poll_interval=0.0002)
    operation = NixlDirectOperation(
        _runtime(bus, source=True), DirectEndpoint.SOURCE, _shard()
    )
    notified = threading.Event()
    handle = executor.submit(
        _attempt(TransferPath.P2D_DIRECT, operation), notified.set
    )
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not bus.sent:
            time.sleep(0.001)
        assert bus.sent
        assert executor.progress(handle).state is PhysicalState.INFLIGHT
        with bus.lock:
            bus.completed = True
        assert notified.wait(3)
        progress = _wait(executor, handle)
        assert progress.state is PhysicalState.FAILED
        assert progress.fence is FenceKind.ERROR_DRAINED
    finally:
        executor.close()
