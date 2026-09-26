from __future__ import annotations

import threading
import time
import queue

import pytest

from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
    GroupCommand,
    LinkCapacityEdge,
    LinkParticipant,
    Owner,
    TCPGroupRelayServer,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    CallbackPathHandler,
    GroupTransferPlan,
    PreparedRankTransfer,
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_multinode_runtime import (
    AgenticMultiNodeRuntime,
    RuntimeCloseBlocked,
    _PathCommandDispatcher,
    _command_worker_counts,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    AgenticTransferQueues,
    FenceKind,
    PhysicalProgress,
    PhysicalState,
    TransferPath,
)


class _ImmediateExecutor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = set()

    def submit(self, attempt, _notify):
        return attempt

    def progress(self, handle):
        with self._lock:
            cancelled = handle.key in self._cancelled
        if cancelled:
            return PhysicalProgress(
                PhysicalState.CANCELLED, FenceKind.CANCEL_DRAINED
            )
        return PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE)

    def request_cancel(self, handle, notify):
        with self._lock:
            self._cancelled.add(handle.key)
        notify()


def _dispatcher_command(request_id, path, kind, seq=1):
    return GroupCommand(
        key=GenerationKey("run", request_id, 0),
        attempt=1,
        command_seq=seq,
        kind=kind,
        source_owner=Owner.D_GPU,
        target_owner=Owner.P_GPU,
        lease_id=f"lease-{request_id}",
        payload={
            "agentic_data_plane": {
                "version": 1,
                "path": path.value,
                "operation": "direct",
            }
        },
    )


def test_path_dispatcher_has_no_cross_path_head_of_line_blocking():
    host_entered = threading.Event()
    host_release = threading.Event()
    direct_done = threading.Event()
    errors = []

    class Executor:
        def handle(self, command):
            path = command.payload.get("agentic_data_plane", {}).get("path")
            if command.kind is CommandKind.PREPARE and path == "d2p_host":
                host_entered.set()
                assert host_release.wait(2)
            if command.kind is CommandKind.PREPARE and path == "d2p_direct":
                direct_done.set()

    dispatcher = _PathCommandDispatcher(
        executor=Executor(),
        workers={path: 1 for path in TransferPath},
        after_command=lambda _command: None,
        on_fatal=errors.append,
    )
    dispatcher.start()
    host = _dispatcher_command("host", TransferPath.D2P_HOST, CommandKind.PREPARE)
    direct = _dispatcher_command(
        "direct", TransferPath.D2P_DIRECT, CommandKind.PREPARE
    )
    dispatcher.submit(host)
    assert host_entered.wait(1)
    dispatcher.submit(direct)
    assert direct_done.wait(1)
    host_release.set()
    dispatcher.submit(
        _dispatcher_command("host", TransferPath.D2P_HOST, CommandKind.FINALIZE, 2)
    )
    dispatcher.submit(
        _dispatcher_command(
            "direct", TransferPath.D2P_DIRECT, CommandKind.FINALIZE, 2
        )
    )
    dispatcher.close(2)
    assert errors == []


def test_host_prepare_order_is_serial_but_direct_uses_all_control_lanes():
    counts = _command_worker_counts(_queues().snapshot())
    assert counts == {
        TransferPath.D2P_DIRECT: 4,
        TransferPath.D2P_HOST: 1,
        TransferPath.P2D_DIRECT: 4,
        TransferPath.P2D_HOST: 1,
    }

    first_entered = threading.Event()
    first_release = threading.Event()
    second_entered = threading.Event()
    errors = []

    class Executor:
        def handle(self, command):
            if command.kind is not CommandKind.PREPARE:
                return
            if command.key.request_id == "first":
                first_entered.set()
                assert first_release.wait(2)
            else:
                second_entered.set()

    dispatcher = _PathCommandDispatcher(
        executor=Executor(),
        workers=counts,
        after_command=lambda _command: None,
        on_fatal=errors.append,
    )
    dispatcher.start()
    dispatcher.submit(
        _dispatcher_command("first", TransferPath.D2P_HOST, CommandKind.PREPARE)
    )
    assert first_entered.wait(1)
    dispatcher.submit(
        _dispatcher_command("second", TransferPath.D2P_HOST, CommandKind.PREPARE)
    )
    assert not second_entered.wait(0.05)
    first_release.set()
    assert second_entered.wait(1)
    for name in ("first", "second"):
        dispatcher.submit(
            _dispatcher_command(name, TransferPath.D2P_HOST, CommandKind.FINALIZE, 2)
        )
    dispatcher.close(2)
    assert errors == []


class _ManualDrainExecutor:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._handle = None
        self._notify = None
        self._cancelled = False
        self._drained = False

    def submit(self, attempt, notify):
        with self._condition:
            self._handle = attempt
            self._notify = notify
            self._condition.notify_all()
        return attempt

    def progress(self, _handle):
        with self._condition:
            if self._drained:
                return PhysicalProgress(
                    PhysicalState.CANCELLED, FenceKind.CANCEL_DRAINED
                )
        return PhysicalProgress(PhysicalState.INFLIGHT)

    def request_cancel(self, _handle, _notify):
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def wait_started(self, timeout=5.0):
        with self._condition:
            return self._condition.wait_for(lambda: self._handle is not None, timeout)

    def wait_cancelled(self, timeout=5.0):
        with self._condition:
            return self._condition.wait_for(lambda: self._cancelled, timeout)

    def drain(self):
        with self._condition:
            self._drained = True
            notify = self._notify
        assert notify is not None
        notify()


def _queues(direct_executor=None) -> AgenticTransferQueues:
    executors = {path: _ImmediateExecutor() for path in TransferPath}
    if direct_executor is not None:
        executors[TransferPath.D2P_DIRECT] = direct_executor
    return AgenticTransferQueues(
        executors,
        lanes={path: 4 for path in TransferPath},
        pending_capacity={path: 32 for path in TransferPath},
    )


def _link(tp_size: int):
    link_id = "p0--d0"
    relay = TCPGroupRelayServer(
        run_id="run",
        token="secret",
        links={
            link_id: {
                "coordinator": {"endpoint_group": "p0", "rank": 0},
                "endpoints": [
                    {
                        "endpoint_group": "p0",
                        "role": "prefill",
                        "size": tp_size,
                    },
                    {
                        "endpoint_group": "d0",
                        "role": "decode",
                        "size": tp_size,
                    },
                ],
            }
        },
    )
    participants = tuple(
        LinkParticipant(role, group, rank)
        for role, group in (("prefill", "p0"), ("decode", "d0"))
        for rank in range(tp_size)
    )
    return relay, link_id, participants, LinkParticipant("prefill", "p0", 0)


def _plan(name: str = "request") -> GroupTransferPlan:
    return GroupTransferPlan(
        key=GenerationKey("run", name, 1),
        path=TransferPath.D2P_DIRECT,
        operation=TransferOperation.DIRECT,
        source_owner=Owner.D_GPU,
        target_owner=Owner.PREFILL_READY,
        lease_id=f"lease-{name}",
        payload={"parent_tokens": 1024, "prompt_tokens": 32},
    )


def _runtimes(tp_size: int, *, failing=None, queue_factory=None):
    relay, link_id, participants, coordinator = _link(tp_size)
    committed = []
    aborted = []
    runtimes = {}
    for participant in participants:
        should_fail = participant == failing

        def prepare(command, participant=participant, should_fail=should_fail):
            if should_fail:
                raise RuntimeError("injected follower prepare failure")
            return PreparedRankTransfer(
                {
                    "participant": (
                        participant.role,
                        participant.endpoint_group,
                        participant.rank,
                    ),
                    "attempt": command.attempt,
                }
            )

        handler = CallbackPathHandler(prepare)
        runtimes[participant] = AgenticMultiNodeRuntime(
            address=relay.address,
            run_id="run",
            link_id=link_id,
            token="secret",
            participant=participant,
            endpoint_tp_size=tp_size,
            participants=participants,
            coordinator_participant=coordinator,
            queues=_queues()
            if queue_factory is None
            else queue_factory(participant),
            handlers={path: handler for path in TransferPath},
            on_committed=lambda plan, attempt: committed.append(
                (plan.key, attempt)
            ),
            on_aborted=lambda plan, attempt, reason: aborted.append(
                (plan.key, attempt, reason)
            ),
        )
    assert relay.wait_connected(link_id, timeout=5)
    for runtime in runtimes.values():
        runtime.start()
    return relay, runtimes, coordinator, committed, aborted


def _close(relay, runtimes, coordinator):
    errors = []
    ordered = [runtimes[coordinator]] + [
        runtime
        for participant, runtime in runtimes.items()
        if participant != coordinator
    ]
    for runtime in ordered:
        try:
            runtime.close(timeout=5)
        except BaseException as error:  # keep closing the remaining queue threads
            errors.append(error)
    relay.close()
    if errors:
        raise errors[0]


def _report_ready(runtimes, plan):
    target_generation = int(
        plan.payload.get("target_generation", plan.key.generation)
    )
    target_key = GenerationKey(
        plan.key.run_id, plan.key.request_id, target_generation
    )
    for participant, runtime in runtimes.items():
        if participant.role == "decode":
            runtime.report_source_ready(plan.key)
        if participant.role == "prefill":
            runtime.report_registered(target_key)


def _activate_target(runtimes, role="prefill"):
    rank0 = next(
        runtime
        for participant, runtime in runtimes.items()
        if participant.role == role and participant.rank == 0
    )
    ticket = rank0.take_activation_ticket(timeout=10)
    for participant, runtime in runtimes.items():
        if participant.role == role:
            runtime.activate_staged(ticket)
            runtime.confirm_scheduler_adopted(ticket)
    return ticket


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_decode_rank0_proposal_commits_one_atomic_two_endpoint_transaction(tp_size):
    relay, runtimes, coordinator, committed, aborted = _runtimes(tp_size)
    plan = _plan(f"tp{tp_size}")
    try:
        # D rank zero proposes; only fixed P rank zero may make and publish the
        # decision.  All 2*TP participants join the same attempt/barrier.
        _report_ready(runtimes, plan)
        assert runtimes[LinkParticipant("decode", "d0", 0)].submit(plan) is None
        _activate_target(runtimes)
        terminal = runtimes[coordinator].take_terminal(timeout=10)
        assert terminal.committed
        assert terminal.key == plan.key
        assert terminal.results is not None
        assert len(terminal.results) == 2 * tp_size
        assert committed == [(plan.key, terminal.attempt)]
        assert aborted == []
        assert all(runtime.wait_idle(5) for runtime in runtimes.values())
        assert runtimes[coordinator].coordinator is not None
        assert (
            len(runtimes[coordinator].coordinator.participants) == 2 * tp_size
        )
    finally:
        _close(relay, runtimes, coordinator)


def test_tp8_decode_capacity_edge_reaches_p0_without_lifecycle_attempt():
    relay, runtimes, coordinator, _committed, _aborted = _runtimes(8)
    seen = []
    arrived = threading.Event()
    p0 = runtimes[coordinator]

    def consume(edge):
        seen.append(edge)
        arrived.set()

    p0._on_capacity_edge_callback = consume
    lifecycle = p0.coordinator
    assert lifecycle is not None
    records_before = dict(lifecycle._coordinator._records)
    try:
        sequence = runtimes[
            LinkParticipant("decode", "d0", 0)
        ].notify_capacity_available(16384)
        assert sequence == 1
        assert arrived.wait(5)
        assert len(seen) == 1 and isinstance(seen[0], LinkCapacityEdge)
        assert seen[0].participant == LinkParticipant("decode", "d0", 0)
        assert seen[0].available_tokens == 16384
        assert p0.orchestrator is not None and p0.orchestrator.active_count == 0
        assert dict(lifecycle._coordinator._records) == records_before
    finally:
        _close(relay, runtimes, coordinator)


def test_tp8_follower_failure_broadcasts_cancel_and_drains_all_participants():
    failing = LinkParticipant("decode", "d0", 6)
    relay, runtimes, coordinator, committed, aborted = _runtimes(
        8, failing=failing
    )
    plan = _plan("failure")
    try:
        _report_ready(runtimes, plan)
        runtimes[LinkParticipant("decode", "d0", 0)].submit(plan)
        terminal = runtimes[coordinator].take_terminal(timeout=10)
        assert not terminal.committed
        assert "prepare failed" in terminal.reason
        assert committed == []
        assert len(aborted) == 1
        assert all(runtime.wait_idle(5) for runtime in runtimes.values())
        # Ownership never moved to P: a follower failure cannot be turned into
        # a fabricated successful barrier by the other fifteen ranks.
        lifecycle = runtimes[coordinator].coordinator
        assert lifecycle is not None
        assert lifecycle.record(plan.key).owner is Owner.D_GPU
    finally:
        _close(relay, runtimes, coordinator)


def test_tp8_out_of_order_readiness_gates_prepare_and_target_scheduler_ticket():
    relay, runtimes, coordinator, committed, aborted = _runtimes(8)
    plan = _plan("readiness-barrier")
    d0 = runtimes[LinkParticipant("decode", "d0", 0)]
    p0 = runtimes[LinkParticipant("prefill", "p0", 0)]
    target_key = GenerationKey("run", "readiness-barrier", 1)
    try:
        assert d0.submit(plan) is None
        time.sleep(0.05)
        assert p0.orchestrator is not None
        assert p0.orchestrator.active_count == 0

        prerequisites = [
            (runtimes[LinkParticipant("decode", "d0", rank)], "source")
            for rank in reversed(range(8))
        ] + [
            (runtimes[LinkParticipant("prefill", "p0", rank)], "registered")
            for rank in range(8)
        ]
        held_runtime, held_kind = prerequisites.pop(5)
        for runtime, kind in prerequisites:
            if kind == "source":
                runtime.report_source_ready(plan.key)
            else:
                runtime.report_registered(target_key)
        # Exact duplicates cannot manufacture the one missing rank.
        prerequisites[0][0].report_source_ready(plan.key)
        time.sleep(0.05)
        assert p0.orchestrator.active_count == 0

        if held_kind == "source":
            held_runtime.report_source_ready(plan.key)
        else:
            held_runtime.report_registered(target_key)
        ticket = p0.take_activation_ticket(timeout=10)
        assert ticket.target_role == "prefill"
        assert ticket.target_generation == 1
        assert ticket.target_key == target_key
        with pytest.raises(queue.Empty):
            d0.take_activation_ticket(timeout=0.05)

        target_runtimes = [
            runtimes[LinkParticipant("prefill", "p0", rank)]
            for rank in range(8)
        ]
        for runtime in target_runtimes[:-1]:
            runtime.activate_staged(ticket)
            runtime.confirm_scheduler_adopted(ticket)
        time.sleep(0.05)
        # Source HBM was released at ACTIVATE, but the control attempt remains
        # live until every target rank confirms native scheduler adoption.
        assert p0.orchestrator.active_count == 1
        target_runtimes[-1].activate_staged(ticket)
        target_runtimes[-1].confirm_scheduler_adopted(ticket)
        terminal = p0.take_terminal(timeout=10)
        assert terminal.committed
        assert committed == [(plan.key, terminal.attempt)]
        assert aborted == []
    finally:
        _close(relay, runtimes, coordinator)


def test_close_retains_active_dma_until_cancel_drained_fence():
    manual = {}

    def queue_factory(participant):
        executor = _ManualDrainExecutor()
        manual[participant] = executor
        return _queues(executor)

    relay, runtimes, coordinator, committed, aborted = _runtimes(
        1, queue_factory=queue_factory
    )
    plan = _plan("shutdown-fence")
    p0 = runtimes[coordinator]
    try:
        _report_ready(runtimes, plan)
        runtimes[LinkParticipant("decode", "d0", 0)].submit(plan)
        deadline = time.monotonic() + 5
        for executor in manual.values():
            assert executor.wait_started(max(0.0, deadline - time.monotonic()))

        with pytest.raises(RuntimeCloseBlocked):
            p0.close(timeout=0.05)
        assert all(executor.wait_cancelled(5) for executor in manual.values())
        assert all(runtime.executor.active_count == 1 for runtime in runtimes.values())

        for executor in manual.values():
            executor.drain()
        terminal = p0.take_terminal(timeout=5)
        assert not terminal.committed
        assert committed == [] and len(aborted) == 1
        assert all(runtime.wait_idle(5) for runtime in runtimes.values())
    finally:
        _close(relay, runtimes, coordinator)
