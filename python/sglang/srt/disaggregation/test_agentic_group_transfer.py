from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
    GenerationTerminal,
    GroupCommand,
    LinkLifecycleCoordinator,
    LinkParticipant,
    LinkRankAck,
    Owner,
    RankAck,
    RankPhase,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    AuthorityPathHandler,
    CallbackPathHandler,
    GroupTransferPlan,
    PreparedRankTransfer,
    RankLocalCommandExecutor,
    RankZeroLinkOrchestrator,
    RemoteHostLoadExecutor,
    RemoteHostLoadPayload,
    TargetLeaseKind,
    TransferOperation,
    plan_endpoint_groups,
)
from sglang.srt.disaggregation.agentic_memory_authority import AgenticMemoryAuthority
from sglang.srt.disaggregation.agentic_remote_host import HostShard, ReadReceipt
from sglang.srt.disaggregation.agentic_remote_host_worker import UnfencedRemoteRead
from sglang.srt.disaggregation.agentic_transfer_queues import (
    AgenticTransferQueues,
    FenceKind,
    PhysicalProgress,
    PhysicalState,
    TransferPath,
)


class ManualExecutor:
    def __init__(self):
        self._condition = threading.Condition()
        self._next = 0
        self.records = {}

    def submit(self, attempt, notify):
        with self._condition:
            self._next += 1
            handle = self._next
            self.records[handle] = {
                "notify": notify,
                "progress": PhysicalProgress(PhysicalState.INFLIGHT),
                "cancel": False,
            }
            self._condition.notify_all()
            return handle

    def progress(self, handle):
        with self._condition:
            return self.records[handle]["progress"]

    def request_cancel(self, handle, _notify):
        with self._condition:
            self.records[handle]["cancel"] = True
            self._condition.notify_all()

    def wait_handle(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self.records:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("physical executor was not started")
                self._condition.wait(remaining)
            return min(self.records)

    def wait_cancel(self, handle, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self.records[handle]["cancel"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("physical executor was not cancelled")
                self._condition.wait(remaining)

    def finish(self, handle, progress):
        with self._condition:
            self.records[handle]["progress"] = progress
            notify = self.records[handle]["notify"]
        notify()


def make_queues():
    executors = {path: ManualExecutor() for path in TransferPath}
    queues = AgenticTransferQueues(
        executors,
        lanes={path: 2 for path in TransferPath},
        pending_capacity={path: 8 for path in TransferPath},
    )
    return queues, executors


def command(kind, seq, *, path=TransferPath.D2P_DIRECT, operation="direct"):
    payload = {}
    if kind in {CommandKind.PREPARE, CommandKind.START}:
        payload = {
            "agentic_data_plane": {
                "version": 1,
                "path": path.value,
                "operation": operation,
            },
            "transfer": {"parent_tokens": 64, "prompt_tokens": 128},
        }
    return GroupCommand(
        GenerationKey("run", "request", 3),
        1,
        seq,
        kind,
        Owner.D_GPU,
        Owner.PREFILL_READY,
        "logical-lease",
        payload,
    )


def wait_phase(values, phase, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for value in tuple(values):
            ack = value.ack if isinstance(value, LinkRankAck) else value
            if ack.phase is phase:
                return ack
        time.sleep(0.001)
    raise AssertionError(f"missing {phase.name} ACK")


def test_rank_executor_releases_source_only_after_group_handoff():
    queues, executors = make_queues()
    acks, failures, released = [], [], []
    handler = CallbackPathHandler(
        lambda _cmd: PreparedRankTransfer(("immutable", "descriptor")),
        commit=lambda cmd, _prepared, completion: released.append(
            (cmd.attempt, completion.fence)
        ),
    )
    rank = RankLocalCommandExecutor(
        participant=LinkParticipant("source", "d", 0),
        queues=queues,
        handlers={path: handler for path in TransferPath},
        emit_ack=acks.append,
        report_failure=failures.append,
    )
    rank.handle(command(CommandKind.PREPARE, 1))
    rank.handle(command(CommandKind.START, 2))
    handle = executors[TransferPath.D2P_DIRECT].wait_handle()
    executors[TransferPath.D2P_DIRECT].finish(
        handle, PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE)
    )
    wait_phase(acks, RankPhase.DMA_DONE)
    assert released == []
    rank.handle(command(CommandKind.RELEASE, 3))
    wait_phase(acks, RankPhase.BOUND)
    assert released == []
    rank.handle(command(CommandKind.HANDOFF, 4))
    wait_phase(acks, RankPhase.RELEASED)
    assert released == [(1, FenceKind.DMA_COMPLETE)]
    rank.handle(command(CommandKind.ACTIVATE, 5))
    wait_phase(acks, RankPhase.STAGED)
    assert released == [(1, FenceKind.DMA_COMPLETE)]
    rank.handle(command(CommandKind.SCHEDULER_ACTIVATE, 6))
    wait_phase(acks, RankPhase.ACTIVATION_ARMED)
    rank.handle(command(CommandKind.PUBLISH_ACTIVATION, 7))
    wait_phase(acks, RankPhase.ACTIVATION_READY)
    rank.handle(command(CommandKind.ISSUE_ACTIVATION_TICKET, 8))
    wait_phase(acks, RankPhase.ACTIVATED)
    rank.handle(command(CommandKind.FINALIZE, 9))
    wait_phase(acks, RankPhase.FINALIZED)
    assert released == [(1, FenceKind.DMA_COMPLETE)]
    assert failures == []
    queues.close()


def test_rank_cancel_waits_for_real_drain_before_failed_ack():
    queues, executors = make_queues()
    acks, aborted = [], []
    handler = CallbackPathHandler(
        lambda _cmd: PreparedRankTransfer("descriptor"),
        abort=lambda _cmd, _prepared, completion: aborted.append(completion.fence),
    )
    rank = RankLocalCommandExecutor(
        participant=LinkParticipant("source", "d", 2),
        queues=queues,
        handlers={path: handler for path in TransferPath},
        emit_ack=acks.append,
        report_failure=lambda _failure: None,
    )
    rank.handle(
        command(
            CommandKind.PREPARE,
            1,
            path=TransferPath.D2P_HOST,
            operation="host_store",
        )
    )
    rank.handle(
        command(
            CommandKind.START,
            2,
            path=TransferPath.D2P_HOST,
            operation="host_store",
        )
    )
    handle = executors[TransferPath.D2P_HOST].wait_handle()
    rank.handle(command(CommandKind.CANCEL, 3, path=TransferPath.D2P_HOST))
    executors[TransferPath.D2P_HOST].wait_cancel(handle)
    assert all(value.ack.phase is not RankPhase.FAILED_DRAINED for value in acks)
    executors[TransferPath.D2P_HOST].finish(
        handle,
        PhysicalProgress(PhysicalState.CANCELLED, FenceKind.CANCEL_DRAINED),
    )
    assert not wait_phase(acks, RankPhase.FAILED_DRAINED).ok
    assert aborted == []
    rank.handle(command(CommandKind.ABORT_FINALIZE, 4, path=TransferPath.D2P_HOST))
    assert wait_phase(acks, RankPhase.ABORTED).ok
    assert aborted == [FenceKind.CANCEL_DRAINED]
    queues.close()


def make_plan(key, operation, source, target, lease):
    return GroupTransferPlan(
        key,
        TransferPath.D2P_DIRECT
        if operation is TransferOperation.DIRECT
        else TransferPath.D2P_HOST,
        operation,
        source,
        target,
        lease,
        {"parent_tokens": 64, "prompt_tokens": 128},
    )


def send_ack(
    orchestrator, participant, command_value, phase, *, result=None, ok=True
):
    orchestrator.on_ack(
        LinkRankAck(
            participant,
            RankAck(
                command_value.key,
                command_value.attempt,
                command_value.command_seq,
                participant.rank,
                phase,
                command_value.lease_id,
                ok=ok,
                result={} if result is None else result,
            ),
        )
    )


def drive_attempt(orchestrator, coordinator, commands, plan, *, offered=False):
    attempt = 1 if offered else orchestrator.begin(plan)
    prepare = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = commands[-1]
    for phase in (RankPhase.DMA_SUBMITTED, RankPhase.DMA_DONE):
        for participant in coordinator.participants:
            send_ack(orchestrator, participant, start, phase)
    release = commands[-1]
    assert release.payload["transfer"] == dict(plan.payload)
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, release, RankPhase.BOUND)
    if plan.operation is not TransferOperation.HOST_STORE:
        handoff = commands[-1]
        assert handoff.kind is CommandKind.HANDOFF
        for participant in coordinator.participants:
            send_ack(orchestrator, participant, handoff, RankPhase.RELEASED)
        activate = commands[-1]
        assert activate.kind is CommandKind.ACTIVATE
        for participant in coordinator.participants:
            send_ack(orchestrator, participant, activate, RankPhase.STAGED)
        ticket = commands[-1]
        assert ticket.kind is CommandKind.ISSUE_ACTIVATION_TICKET
        assert ticket.payload["transfer"] == dict(plan.payload)
        for participant in coordinator.participants:
            send_ack(orchestrator, participant, ticket, RankPhase.ACTIVATED)
    finalize = commands[-1]
    assert finalize.kind is CommandKind.FINALIZE
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, finalize, RankPhase.FINALIZED)
    return attempt


def test_activation_ticket_retains_next_generation_for_d2p():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 2), ("target", "p", 2)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator, broadcast=commands.append
    )
    plan = GroupTransferPlan(
        GenerationKey("run", "next-turn", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "next-turn-lease",
        {"parent_tokens": 64, "prompt_tokens": 128, "target_generation": 1},
    )
    drive_attempt(orchestrator, coordinator, commands, plan)
    ticket = next(
        command
        for command in commands
        if command.kind is CommandKind.ISSUE_ACTIVATION_TICKET
    )
    assert ticket.key.generation == 0
    assert ticket.payload["transfer"]["target_generation"] == 1


def test_materialized_callback_fires_once_after_every_tp_rank_finishes_dma():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "p-d", [("source", "p", 2), ("target", "d", 2)]
    )
    commands, materialized = [], []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        on_materialized=lambda plan, attempt: materialized.append((plan, attempt)),
    )
    plan = GroupTransferPlan(
        GenerationKey("run", "prepared-edge", 0),
        TransferPath.P2D_DIRECT,
        TransferOperation.DIRECT,
        Owner.P_GPU,
        Owner.D_GPU,
        "prepared-edge",
        {"decode_reservation_id": "reservation-1"},
        source_group="p",
        target_group="d",
    )
    assert orchestrator.begin(plan) == 1
    command = commands[-1]
    for participant in coordinator.participants[:-1]:
        send_ack(orchestrator, participant, command, RankPhase.PREPARED)
    assert materialized == []

    send_ack(
        orchestrator,
        coordinator.participants[-1],
        command,
        RankPhase.PREPARED,
    )

    assert materialized == []
    assert commands[-1].kind is CommandKind.START
    start = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, start, RankPhase.DMA_SUBMITTED)
    for participant in coordinator.participants[:-1]:
        send_ack(orchestrator, participant, start, RankPhase.DMA_DONE)
    assert materialized == []

    send_ack(
        orchestrator,
        coordinator.participants[-1],
        start,
        RankPhase.DMA_DONE,
    )
    assert materialized == [(plan, 1)]


def test_direct_commit_waits_for_source8_and_target8_fences():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 8), ("target", "p", 8)]
    )
    commands, committed = [], []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        on_committed=lambda _plan, attempt: committed.append(attempt),
    )
    plan = make_plan(
        GenerationKey("run", "direct", 0),
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.PREFILL_READY,
        "direct-lease",
    )
    attempt = orchestrator.begin(plan)
    prepare = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, start, RankPhase.DMA_SUBMITTED)
    for participant in coordinator.participants[:-1]:
        send_ack(orchestrator, participant, start, RankPhase.DMA_DONE)
    assert coordinator.record(plan.key).owner is Owner.D_GPU
    assert commands[-1].kind is CommandKind.START
    send_ack(orchestrator, coordinator.participants[-1], start, RankPhase.DMA_DONE)
    assert coordinator.record(plan.key).owner is Owner.D_GPU
    release = commands[-1]
    for participant in coordinator.participants[:-1]:
        send_ack(orchestrator, participant, release, RankPhase.BOUND)
    assert coordinator.record(plan.key).owner is Owner.D_GPU
    send_ack(orchestrator, coordinator.participants[-1], release, RankPhase.BOUND)
    assert coordinator.record(plan.key).owner is Owner.D_GPU
    assert committed == []
    handoff = commands[-1]
    assert handoff.kind is CommandKind.HANDOFF
    for participant in coordinator.participants[:-1]:
        send_ack(orchestrator, participant, handoff, RankPhase.RELEASED)
    assert coordinator.record(plan.key).owner is Owner.D_GPU
    send_ack(orchestrator, coordinator.participants[-1], handoff, RankPhase.RELEASED)
    assert coordinator.record(plan.key).owner is Owner.D_GPU
    activate = commands[-1]
    assert activate.kind is CommandKind.ACTIVATE
    assert committed == []
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, activate, RankPhase.STAGED)
    ticket = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, ticket, RankPhase.ACTIVATED)
    assert coordinator.record(plan.key).owner is Owner.PREFILL_READY
    finalize = commands[-1]
    assert finalize.kind is CommandKind.FINALIZE
    assert committed == []
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, finalize, RankPhase.FINALIZED)
    assert committed == [1]


def test_rank_zero_reuses_io_lane_after_group_dma_done_before_bound():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 2), ("target", "p", 2)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
    )
    first = make_plan(
        GenerationKey("run", "first", 0),
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "first-lease",
    )
    second = make_plan(
        GenerationKey("run", "second", 0),
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "second-lease",
    )

    assert orchestrator.offer(first) == 1
    assert orchestrator.offer(second) is None
    assert [value.key for value in commands if value.kind is CommandKind.PREPARE] == [
        first.key
    ]

    prepare = next(
        value
        for value in commands
        if value.kind is CommandKind.PREPARE and value.key == first.key
    )
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, start, RankPhase.DMA_SUBMITTED)
    assert not any(
        value.kind is CommandKind.PREPARE and value.key == second.key
        for value in commands
    )
    for participant in coordinator.participants[:-1]:
        send_ack(orchestrator, participant, start, RankPhase.DMA_DONE)
    assert [value.key for value in commands if value.kind is CommandKind.PREPARE] == [
        first.key
    ]

    send_ack(
        orchestrator,
        coordinator.participants[-1],
        start,
        RankPhase.DMA_DONE,
    )
    prepares = [
        value.key for value in commands if value.kind is CommandKind.PREPARE
    ]
    assert prepares == [first.key, second.key]
    second_prepare = next(
        value
        for value in commands
        if value.kind is CommandKind.PREPARE and value.key == second.key
    )
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, second_prepare, RankPhase.PREPARED)
    assert any(
        value.kind is CommandKind.START and value.key == second.key
        for value in commands
    )
    release = next(
        value
        for value in commands
        if value.key == first.key and value.kind is CommandKind.RELEASE
    )
    assert release.kind is CommandKind.RELEASE
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, release, RankPhase.BOUND)
    handoff = commands[-1]
    assert handoff.kind is CommandKind.HANDOFF
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, handoff, RankPhase.RELEASED)
    assert [value.key for value in commands if value.kind is CommandKind.PREPARE] == [
        first.key, second.key
    ]
    assert second_prepare.key == second.key


def test_rank_zero_admits_disjoint_endpoint_pairs_independently():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run",
        "global",
        [
            ("prefill", "p0", 1),
            ("prefill", "p1", 1),
            ("decode", "d0", 1),
            ("decode", "d1", 1),
            ("decode", "d2", 1),
        ],
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={(TransferPath.P2D_DIRECT, "direct"): 1},
    )

    def plan(name, source, target):
        return GroupTransferPlan(
            GenerationKey("run", name, 0),
            TransferPath.P2D_DIRECT,
            TransferOperation.DIRECT,
            Owner.P_GPU,
            Owner.D_GPU,
            name,
            {},
            source_group=source,
            target_group=target,
        )

    first = plan("first", "p0", "d0")
    disjoint = plan("disjoint", "p1", "d1")
    shared_source = plan("shared-source", "p0", "d2")

    assert orchestrator.offer(first) == 1
    assert orchestrator.offer(disjoint) == 1
    assert orchestrator.offer(shared_source) is None
    prepares = [value.key for value in commands if value.kind is CommandKind.PREPARE]
    assert prepares == [first.key, disjoint.key]

    first_prepare = next(value for value in commands if value.key == first.key)
    first_participants = coordinator.participants_for(first.key, 1)
    for participant in first_participants:
        send_ack(orchestrator, participant, first_prepare, RankPhase.PREPARED)
    first_start = next(
        value
        for value in commands
        if value.key == first.key and value.kind is CommandKind.START
    )
    for phase in (RankPhase.DMA_SUBMITTED, RankPhase.DMA_DONE):
        for participant in first_participants:
            send_ack(orchestrator, participant, first_start, phase)

    prepares = [value.key for value in commands if value.kind is CommandKind.PREPARE]
    assert prepares == [first.key, disjoint.key, shared_source.key]


def test_rank_zero_host_store_and_restore_admit_independently():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 2), ("target", "p", 2)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={
            (TransferPath.D2P_HOST, "host_store"): 1,
            (TransferPath.D2P_HOST, "host_restore"): 1,
        },
    )
    store = GroupTransferPlan(
        GenerationKey("run", "store", 0),
        TransferPath.D2P_HOST,
        TransferOperation.HOST_STORE,
        Owner.D_GPU,
        Owner.D_HOST,
        "store-lease",
        {},
    )
    restore = GroupTransferPlan(
        GenerationKey("run", "restore", 0),
        TransferPath.D2P_HOST,
        TransferOperation.HOST_RESTORE,
        Owner.D_HOST,
        Owner.P_GPU,
        "restore-lease",
        {},
    )

    assert orchestrator.offer(store) == 1
    assert orchestrator.offer(restore) == 1
    assert [value.key for value in commands] == [store.key, restore.key]


def test_direct_and_host_restore_have_independent_io_windows():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 2), ("target", "p", 2)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 4 for path in TransferPath},
        operation_lanes={
            (TransferPath.D2P_DIRECT, "direct"): 4,
            (TransferPath.D2P_HOST, "host_restore"): 4,
        },
    )
    direct = GroupTransferPlan(
        GenerationKey("run", "direct-target", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "direct-target-lease",
        {},
    )
    restore = GroupTransferPlan(
        GenerationKey("run", "restore-target", 0),
        TransferPath.D2P_HOST,
        TransferOperation.HOST_RESTORE,
        Owner.D_HOST,
        Owner.P_GPU,
        "restore-target-lease",
        {},
    )

    assert orchestrator.offer(direct) == 1
    assert orchestrator.offer(restore) == 1
    assert [value.key for value in commands] == [direct.key, restore.key]


def test_shared_network_gate_bounds_disjoint_direct_and_host_restore():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run",
        "fabric",
        [
            ("prefill", "p0", 1),
            ("prefill", "p1", 1),
            ("decode", "d0", 1),
            ("decode", "d1", 1),
        ],
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 4 for path in TransferPath},
        operation_lanes={
            (TransferPath.D2P_DIRECT, "direct"): 4,
            (TransferPath.D2P_HOST, "host_restore"): 4,
        },
        shared_network_lanes=1,
    )
    direct = GroupTransferPlan(
        GenerationKey("run", "direct-shared-rail", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "direct-shared-rail",
        {},
        source_group="d0",
        target_group="p0",
    )
    restore = GroupTransferPlan(
        GenerationKey("run", "restore-shared-rail", 0),
        TransferPath.D2P_HOST,
        TransferOperation.HOST_RESTORE,
        Owner.D_HOST,
        Owner.P_GPU,
        "restore-shared-rail",
        {},
        source_group="d1",
        target_group="p1",
    )

    assert orchestrator.offer(direct) == 1
    assert orchestrator.offer(restore) is None
    assert [value.key for value in commands] == [direct.key]

    prepare = commands[0]
    participants = coordinator.participants_for(direct.key, 1)
    for participant in participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = next(
        value
        for value in commands
        if value.key == direct.key and value.kind is CommandKind.START
    )
    for phase in (RankPhase.DMA_SUBMITTED, RankPhase.DMA_DONE):
        for participant in participants:
            send_ack(orchestrator, participant, start, phase)

    assert any(
        value.key == restore.key and value.kind is CommandKind.PREPARE
        for value in commands
    )


def test_shared_network_gate_is_full_duplex_across_directions():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run",
        "fabric",
        [
            ("prefill", "p0", 1),
            ("decode", "d0", 1),
        ],
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 4 for path in TransferPath},
        operation_lanes={
            (TransferPath.D2P_DIRECT, "direct"): 4,
            (TransferPath.P2D_DIRECT, "direct"): 4,
        },
        shared_network_lanes=1,
    )
    d2p = GroupTransferPlan(
        GenerationKey("run", "d2p-full-duplex", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "d2p-full-duplex",
        {},
        source_group="d0",
        target_group="p0",
    )
    p2d = GroupTransferPlan(
        GenerationKey("run", "p2d-full-duplex", 0),
        TransferPath.P2D_DIRECT,
        TransferOperation.DIRECT,
        Owner.P_GPU,
        Owner.D_GPU,
        "p2d-full-duplex",
        {},
        source_group="p0",
        target_group="d0",
    )

    assert orchestrator.offer(d2p) == 1
    assert orchestrator.offer(p2d) == 1
    assert [value.key for value in commands] == [d2p.key, p2d.key]
    counts, _oldest = orchestrator.progress_diagnostics()
    assert counts["network_d2p_active"] == 1
    assert counts["network_p2d_active"] == 1
    assert counts["network_dma_active"] == 2


def test_shared_network_gate_supports_asymmetric_direction_limits():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run",
        "fabric",
        [
            ("prefill", "p0", 1),
            ("prefill", "p1", 1),
            ("decode", "d0", 1),
            ("decode", "d1", 1),
        ],
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 4 for path in TransferPath},
        operation_lanes={
            (TransferPath.D2P_DIRECT, "direct"): 4,
            (TransferPath.P2D_DIRECT, "direct"): 4,
        },
        shared_network_lanes=4,
        shared_network_lanes_by_direction={"d2p": 1, "p2d": 2},
    )

    def plan(name, path, source, target, source_owner, target_owner):
        return GroupTransferPlan(
            GenerationKey("run", name, 0),
            path,
            TransferOperation.DIRECT,
            source_owner,
            target_owner,
            name,
            {},
            source_group=source,
            target_group=target,
        )

    d2p0 = plan(
        "d2p-asym-0", TransferPath.D2P_DIRECT, "d0", "p0", Owner.D_GPU, Owner.P_GPU
    )
    d2p1 = plan(
        "d2p-asym-1", TransferPath.D2P_DIRECT, "d1", "p1", Owner.D_GPU, Owner.P_GPU
    )
    p2d0 = plan(
        "p2d-asym-0", TransferPath.P2D_DIRECT, "p0", "d0", Owner.P_GPU, Owner.D_GPU
    )
    p2d1 = plan(
        "p2d-asym-1", TransferPath.P2D_DIRECT, "p1", "d1", Owner.P_GPU, Owner.D_GPU
    )

    assert orchestrator.offer(d2p0) == 1
    assert orchestrator.offer(d2p1) is None
    assert orchestrator.offer(p2d0) == 1
    assert orchestrator.offer(p2d1) == 1
    assert [command.key for command in commands] == [d2p0.key, p2d0.key, p2d1.key]
    counts, _oldest = orchestrator.progress_diagnostics()
    assert counts["network_d2p_active"] == 1
    assert counts["network_p2d_active"] == 2


def test_shared_network_gate_reserves_d2p_capacity_for_direct():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run",
        "fabric",
        [
            ("prefill", "p0", 1),
            ("prefill", "p1", 1),
            ("decode", "d0", 1),
            ("decode", "d1", 1),
        ],
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 4 for path in TransferPath},
        operation_lanes={
            (TransferPath.D2P_HOST, "host_restore"): 4,
            (TransferPath.D2P_DIRECT, "direct"): 4,
        },
        shared_network_lanes_by_direction={"d2p": 2, "p2d": 2},
        shared_network_direct_reserve_by_direction={"d2p": 1},
    )

    def plan(name, operation, source, target):
        return GroupTransferPlan(
            GenerationKey("run", name, 0),
            (
                TransferPath.D2P_DIRECT
                if operation is TransferOperation.DIRECT
                else TransferPath.D2P_HOST
            ),
            operation,
            Owner.D_GPU if operation is TransferOperation.DIRECT else Owner.D_HOST,
            Owner.P_GPU,
            name,
            {},
            source_group=source,
            target_group=target,
        )

    host0 = plan("host-reserve-0", TransferOperation.HOST_RESTORE, "d0", "p0")
    host1 = plan("host-reserve-1", TransferOperation.HOST_RESTORE, "d1", "p1")
    direct = plan("direct-reserve", TransferOperation.DIRECT, "d1", "p1")

    assert orchestrator.offer(host0) == 1
    assert orchestrator.offer(host1) is None
    assert orchestrator.offer(direct) == 1
    assert [command.key for command in commands] == [host0.key, direct.key]
    counts, _oldest = orchestrator.progress_diagnostics()
    assert counts["network_d2p_active"] == 2
    assert counts["network_d2p_direct_active"] == 1


def test_shared_network_gate_reserves_host_capacity_only_while_host_waits():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run",
        "fabric",
        [
            ("prefill", "p0", 1),
            ("prefill", "p1", 1),
            ("decode", "d0", 1),
            ("decode", "d1", 1),
        ],
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 4 for path in TransferPath},
        operation_lanes={
            (TransferPath.D2P_HOST, "host_restore"): 4,
            (TransferPath.D2P_DIRECT, "direct"): 4,
        },
        shared_network_lanes_by_direction={"d2p": 2, "p2d": 2},
        shared_network_host_reserve_by_direction={"d2p": 1},
    )

    def plan(name, operation, source, target):
        return GroupTransferPlan(
            GenerationKey("run", name, 0),
            (
                TransferPath.D2P_DIRECT
                if operation is TransferOperation.DIRECT
                else TransferPath.D2P_HOST
            ),
            operation,
            Owner.D_GPU if operation is TransferOperation.DIRECT else Owner.D_HOST,
            Owner.P_GPU,
            name,
            {},
            source_group=source,
            target_group=target,
        )

    direct0 = plan("host-fair-direct-0", TransferOperation.DIRECT, "d0", "p0")
    direct1 = plan("host-fair-direct-1", TransferOperation.DIRECT, "d1", "p1")
    host = plan("host-fair-restore", TransferOperation.HOST_RESTORE, "d0", "p0")
    direct2 = plan("host-fair-direct-2", TransferOperation.DIRECT, "d1", "p1")

    # Host capacity is not left idle in anticipation of future work.
    assert orchestrator.offer(direct0) == 1
    assert orchestrator.offer(direct1) == 1
    assert orchestrator.offer(host) is None
    assert orchestrator.offer(direct2) is None

    prepare = next(
        value
        for value in commands
        if value.key == direct0.key and value.kind is CommandKind.PREPARE
    )
    participants = coordinator.participants_for(direct0.key, 1)
    for participant in participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = next(
        value
        for value in commands
        if value.key == direct0.key and value.kind is CommandKind.START
    )
    for phase in (RankPhase.DMA_SUBMITTED, RankPhase.DMA_DONE):
        for participant in participants:
            send_ack(orchestrator, participant, start, phase)

    # Once a slot opens, the durable Host waiter gets its promised share;
    # the newer Direct remains queued until that recovery progresses.
    assert any(
        value.key == host.key and value.kind is CommandKind.PREPARE
        for value in commands
    )
    assert not any(
        value.key == direct2.key and value.kind is CommandKind.PREPARE
        for value in commands
    )


def test_direct_admission_deadline_applies_while_prepare_is_incomplete():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 1), ("target", "p", 1)]
    )
    commands = []
    rejected = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={(TransferPath.D2P_DIRECT, "direct"): 1},
        on_aborted=lambda plan, attempt, reason: rejected.append(
            (plan, attempt, reason)
        ),
    )
    first = GroupTransferPlan(
        GenerationKey("run", "first-deadline", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "first-deadline",
        {},
    )
    expired = GroupTransferPlan(
        GenerationKey("run", "expired-deadline", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "expired-deadline",
        {},
        admission_deadline=time.monotonic() + 60,
    )

    assert orchestrator.offer(first) == 1
    assert orchestrator.offer(expired) is None
    assert orchestrator.expire_admissions(time.monotonic() + 120) == 1
    assert commands[-1].key == first.key
    assert rejected == [
        (expired, 0, "direct admission deadline expired before PREPARE")
    ]


def test_expired_direct_is_rejected_before_prepare_even_when_lane_is_free():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 1), ("target", "p", 1)]
    )
    commands, rejected = [], []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        on_aborted=lambda plan, attempt, reason: rejected.append(
            (plan.key, attempt, reason)
        ),
    )
    expired = GroupTransferPlan(
        GenerationKey("run", "already-expired", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "already-expired",
        {},
        admission_deadline=time.monotonic() - 1,
    )

    assert orchestrator.offer(expired) is None
    assert commands == []
    assert rejected == [
        (
            expired.key,
            0,
            "direct admission deadline expired before PREPARE",
        )
    ]
    assert orchestrator.active_count == 0
    assert orchestrator.wait_idle(0)


def test_wait_idle_waits_for_immediate_expired_rejection_callback():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 1), ("target", "p", 1)]
    )
    entered, release = threading.Event(), threading.Event()
    rejected = []

    def on_aborted(plan, attempt, reason):
        rejected.append((plan.key, attempt, reason))
        entered.set()
        assert release.wait(2)

    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=lambda _command: None,
        on_aborted=on_aborted,
    )
    expired = GroupTransferPlan(
        GenerationKey("run", "blocking-immediate", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "blocking-immediate",
        {},
        admission_deadline=time.monotonic() - 1,
    )
    worker = threading.Thread(target=lambda: orchestrator.offer(expired))
    worker.start()
    assert entered.wait(1)
    assert not orchestrator.wait_idle(0)
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert orchestrator.wait_idle(1)
    assert len(rejected) == 1


def test_wait_idle_waits_for_deferred_rejection_dispatch():
    for method_name in ("expire_admissions", "cancel_active"):
        coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
            "run", method_name, [("source", "d", 1), ("target", "p", 1)]
        )
        entered, release = threading.Event(), threading.Event()
        rejected = []

        def on_aborted(plan, attempt, reason):
            rejected.append((plan.key, attempt, reason))
            entered.set()
            assert release.wait(2)

        orchestrator = RankZeroLinkOrchestrator(
            coordinator,
            broadcast=lambda _command: None,
            on_aborted=on_aborted,
        )
        plan_value = make_plan(
            GenerationKey("run", f"blocking-{method_name}", 0),
            TransferOperation.DIRECT,
            Owner.D_GPU,
            Owner.P_GPU,
            f"blocking-{method_name}",
        )
        with orchestrator._changed:
            orchestrator._deferred_rejections.append(
                (plan_value, "direct admission deadline expired before PREPARE")
            )
        if method_name == "expire_admissions":
            target = orchestrator.expire_admissions
        else:
            target = lambda: orchestrator.cancel_active("runtime_shutdown")
        worker = threading.Thread(target=target)
        worker.start()
        assert entered.wait(1)
        assert not orchestrator.wait_idle(0)
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert orchestrator.wait_idle(1)
        assert len(rejected) == 1


def test_lane_release_does_not_start_an_expired_pending_direct():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 1), ("target", "p", 1)]
    )
    commands, rejected = [], []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={(TransferPath.D2P_DIRECT, "direct"): 1},
        on_aborted=lambda plan, attempt, reason: rejected.append(
            (plan.key, attempt, reason)
        ),
    )
    first = make_plan(
        GenerationKey("run", "lane-holder", 0),
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "lane-holder",
    )
    expired = GroupTransferPlan(
        GenerationKey("run", "expired-at-release", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "expired-at-release",
        {},
        admission_deadline=time.monotonic() + 0.02,
    )
    assert orchestrator.offer(first) == 1
    assert orchestrator.offer(expired) is None
    time.sleep(0.03)

    prepare = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = commands[-1]
    for phase in (RankPhase.DMA_SUBMITTED, RankPhase.DMA_DONE):
        for participant in coordinator.participants:
            send_ack(orchestrator, participant, start, phase)

    assert not any(command.key == expired.key for command in commands)
    assert rejected == []
    assert orchestrator.expire_admissions() == 1
    assert rejected == [
        (
            expired.key,
            0,
            "direct admission deadline expired before PREPARE",
        )
    ]


def test_shutdown_rejects_lane_pending_plan_instead_of_dropping_it():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 1), ("target", "p", 1)]
    )
    commands, rejected = [], []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={(TransferPath.D2P_DIRECT, "direct"): 1},
        on_aborted=lambda plan, attempt, reason: rejected.append(
            (plan.key, attempt, reason)
        ),
    )
    first = make_plan(
        GenerationKey("run", "shutdown-active", 0),
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "shutdown-active",
    )
    pending = make_plan(
        GenerationKey("run", "shutdown-pending", 0),
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "shutdown-pending",
    )
    assert orchestrator.offer(first) == 1
    assert orchestrator.offer(pending) is None

    assert orchestrator.cancel_active("runtime_shutdown") == 2
    assert rejected == [(pending.key, 0, "runtime_shutdown")]
    assert commands[-1].kind is CommandKind.CANCEL


def test_shutdown_does_not_cancel_a_partially_posted_start():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("decode", "d", 4), ("prefill", "p", 4)]
    )
    commands, rejected = [], []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={(TransferPath.D2P_DIRECT, "direct"): 1},
        on_aborted=lambda plan, attempt, reason: rejected.append(
            (plan.key, attempt, reason)
        ),
    )
    plan_value = GroupTransferPlan(
        GenerationKey("run", "shutdown-partial-start", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "shutdown-partial-start",
        {},
        source_group="d",
        target_group="p",
    )
    assert orchestrator.offer(plan_value) == 1
    prepare = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = commands[-1]
    assert start.kind is CommandKind.START
    for participant in coordinator.participants[:3]:
        send_ack(orchestrator, participant, start, RankPhase.DMA_SUBMITTED)

    command_count = len(commands)
    assert orchestrator.cancel_active("runtime_shutdown") == 0
    assert len(commands) == command_count
    assert all(command.kind is not CommandKind.CANCEL for command in commands)
    assert rejected == []
    assert coordinator.record(plan_value.key).owner is Owner.D_GPU
    assert orchestrator.active_count == 1
    assert not orchestrator.wait_idle(0)


def test_prepare_broadcast_failure_quarantines_active_lane():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 1), ("target", "p", 1)]
    )

    def fail(_command):
        raise OSError("control link failed")

    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=fail,
        path_lanes={path: 1 for path in TransferPath},
    )
    plan_value = make_plan(
        GenerationKey("run", "broadcast-failure", 0),
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "broadcast-failure",
    )
    import pytest

    with pytest.raises(OSError, match="control link failed"):
        orchestrator.offer(plan_value)
    assert orchestrator.active_count == 1
    assert not orchestrator.wait_idle(0)


def test_admission_deadline_stops_after_all_ranks_prepare():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 1), ("target", "p", 1)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={(TransferPath.D2P_DIRECT, "direct"): 1},
    )
    plan_value = GroupTransferPlan(
        GenerationKey("run", "started-deadline", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "started-deadline",
        {},
        admission_deadline=time.monotonic() + 60,
    )
    assert orchestrator.offer(plan_value) == 1
    prepare = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    assert commands[-1].kind is CommandKind.START

    assert orchestrator.expire_admissions(time.monotonic() + 120) == 0
    assert commands[-1].kind is CommandKind.START


def test_admission_deadline_stops_after_all_ranks_submit_dma():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("source", "d", 1), ("target", "p", 1)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={(TransferPath.D2P_DIRECT, "direct"): 1},
    )
    plan_value = GroupTransferPlan(
        GenerationKey("run", "submitted-deadline", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "submitted-deadline",
        {},
        admission_deadline=time.monotonic() + 60,
    )
    assert orchestrator.offer(plan_value) == 1
    prepare = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = commands[-1]
    assert start.kind is CommandKind.START
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, start, RankPhase.DMA_SUBMITTED)

    assert orchestrator.expire_admissions(time.monotonic() + 120) == 0
    assert commands[-1].kind is CommandKind.START


def test_partial_tp_dma_submission_is_never_cancelled_by_admission_deadline():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-p", [("decode", "d", 4), ("prefill", "p", 4)]
    )
    commands, rejected = [], []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        path_lanes={path: 1 for path in TransferPath},
        operation_lanes={(TransferPath.D2P_DIRECT, "direct"): 1},
        on_aborted=lambda plan, attempt, reason: rejected.append(
            (plan.key, attempt, reason)
        ),
    )
    plan_value = GroupTransferPlan(
        GenerationKey("run", "partial-submit", 0),
        TransferPath.D2P_DIRECT,
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.P_GPU,
        "partial-submit",
        {},
        source_group="d",
        target_group="p",
        admission_deadline=time.monotonic() + 60,
    )
    assert orchestrator.offer(plan_value) == 1
    prepare = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = commands[-1]
    assert start.kind is CommandKind.START
    for participant in coordinator.participants[:3]:
        send_ack(orchestrator, participant, start, RankPhase.DMA_SUBMITTED)

    assert orchestrator.expire_admissions(time.monotonic() + 120) == 0
    assert commands[-1].kind is CommandKind.START
    assert rejected == []
    assert coordinator.record(plan_value.key).owner is Owner.D_GPU
    assert orchestrator.active_count == 1


def test_tp8_host_store_collects_source_descriptors_only_for_active_attempt():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-host", [("source", "d", 8), ("target", "p", 8)]
    )
    commands, committed_results = [], []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=commands.append,
        on_committed_results=lambda plan, attempt, results: committed_results.append(
            (plan, attempt, results)
        ),
    )
    plan = make_plan(
        GenerationKey("run", "host-shards", 0),
        TransferOperation.HOST_STORE,
        Owner.D_GPU,
        Owner.D_HOST,
        "host-store-lease",
    )
    attempt = orchestrator.begin(plan)
    prepare = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, prepare, RankPhase.PREPARED)
    start = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, start, RankPhase.DMA_SUBMITTED)
    for participant in coordinator.participants:
        result = {}
        if participant.role == "source":
            shard = HostShard(
                snapshot_id=plan.key.snapshot_id,
                export_id=f"export-{participant.rank}",
                tp_rank=participant.rank,
                tp_size=8,
                layout="a" * 64,
                token_count=128,
                address=4096 * (participant.rank + 1),
                byte_size=8192,
                metadata_b64="eA==",
                peer_id=f"d-r{participant.rank}",
            )
            result = {"host_shard": shard.to_dict()}
        send_ack(
            orchestrator,
            participant,
            start,
            RankPhase.DMA_DONE,
            result=result,
        )

    release = commands[-1]
    assert release.kind is CommandKind.RELEASE
    wire_results = release.payload["rank_results"]
    assert len(wire_results) == 16
    wire_shards = [
        HostShard.from_dict(value["result"]["host_shard"])
        for value in wire_results
        if "host_shard" in value["result"]
    ]
    assert [shard.tp_rank for shard in wire_shards] == list(range(8))

    for participant in coordinator.participants:
        send_ack(orchestrator, participant, release, RankPhase.BOUND)
    finalize = commands[-1]
    assert finalize.kind is CommandKind.FINALIZE
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, finalize, RankPhase.FINALIZED)
    assert orchestrator.active_count == 0
    assert not coordinator._coordinator._active
    assert len(committed_results) == 1
    callback_results = committed_results[0][2]
    assert len(callback_results) == 16
    assert sum("host_shard" in result for result in callback_results.values()) == 8


def test_host_store_then_restore_uses_one_owner_chain():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "host", [("source", "d", 2), ("target", "p", 2)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(coordinator, broadcast=commands.append)
    key = GenerationKey("run", "host", 0)
    store = make_plan(
        key,
        TransferOperation.HOST_STORE,
        Owner.D_GPU,
        Owner.D_HOST,
        "store",
    )
    drive_attempt(orchestrator, coordinator, commands, store)
    assert coordinator.record(key).owner is Owner.D_HOST
    restore = make_plan(
        key,
        TransferOperation.HOST_RESTORE,
        Owner.D_HOST,
        Owner.PREFILL_READY,
        "restore",
    )
    drive_attempt(orchestrator, coordinator, commands, restore)
    assert coordinator.record(key).owner is Owner.PREFILL_READY


def test_no_io_endpoint_joins_barrier_without_consuming_lane():
    queues, _executors = make_queues()
    acks = []
    handler = CallbackPathHandler(
        lambda _command: PreparedRankTransfer(None, requires_io=False)
    )
    rank = RankLocalCommandExecutor(
        participant=LinkParticipant("target", "p", 7),
        queues=queues,
        handlers={path: handler for path in TransferPath},
        emit_ack=acks.append,
        report_failure=lambda failure: (_ for _ in ()).throw(
            AssertionError(failure)
        ),
    )
    rank.handle(
        command(
            CommandKind.PREPARE,
            1,
            path=TransferPath.D2P_HOST,
            operation="host_store",
        )
    )
    rank.handle(
        command(
            CommandKind.START,
            2,
            path=TransferPath.D2P_HOST,
            operation="host_store",
        )
    )
    assert [value.ack.phase for value in acks] == [
        RankPhase.PREPARED,
        RankPhase.DMA_SUBMITTED,
        RankPhase.DMA_DONE,
    ]
    assert acks[-1].ack.detail == "no_io_required"
    queues.close()


class ListAllocator:
    page_size = 64

    def __init__(self, size=1024):
        self.free_values = list(range(size))
        self.live = set()

    def available_size(self):
        return len(self.free_values)

    def alloc(self, count):
        if count > len(self.free_values):
            return None
        result = self.free_values[:count]
        del self.free_values[:count]
        self.live.update(result)
        return result

    def free(self, values):
        values = list(values)
        assert set(values).issubset(self.live)
        self.live.difference_update(values)
        self.free_values.extend(values)


def test_authority_handler_publishes_only_on_group_handoff():
    authority = AgenticMemoryAuthority(ListAllocator())
    handler = AuthorityPathHandler(
        authority,
        lease_kind=TargetLeaseKind.PREFILL_WORKSET,
        owner="d2p",
        payload_builder=lambda _command, lease: tuple(lease.device_indices),
        ready_queue="prefill-ready",
    )
    queues, executors = make_queues()
    acks = []
    rank = RankLocalCommandExecutor(
        participant=LinkParticipant("target", "p", 0),
        queues=queues,
        handlers={path: handler for path in TransferPath},
        emit_ack=acks.append,
        report_failure=lambda failure: (_ for _ in ()).throw(
            AssertionError(failure)
        ),
    )
    rank.handle(command(CommandKind.PREPARE, 1))
    rank.handle(command(CommandKind.START, 2))
    handle = executors[TransferPath.D2P_DIRECT].wait_handle()
    executors[TransferPath.D2P_DIRECT].finish(
        handle, PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE)
    )
    wait_phase(acks, RankPhase.DMA_DONE)
    assert authority.take_ready("prefill-ready", timeout=0.01) == ()
    rank.handle(command(CommandKind.RELEASE, 3))
    wait_phase(acks, RankPhase.BOUND)
    assert authority.take_ready("prefill-ready", timeout=0.01) == ()
    rank.handle(command(CommandKind.HANDOFF, 4))
    wait_phase(acks, RankPhase.RELEASED)
    assert authority.take_ready("prefill-ready", timeout=0.01) == ()
    rank.handle(command(CommandKind.ACTIVATE, 5))
    wait_phase(acks, RankPhase.STAGED)
    # ACTIVATE is the all-rank ready-publication barrier.  Rank zero issues
    # the scheduler ticket only after every participant emitted STAGED.
    assert len(authority.take_ready("prefill-ready", timeout=0.2)) == 1
    queues.close()


def test_tp8_missing_bound_keeps_all_ready_and_source_release_invisible():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "tp8-handoff", [("source", "d", 8), ("target", "p", 8)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(coordinator, broadcast=commands.append)
    queues, _executors = make_queues()
    emitted = []
    released = []
    ready = []
    executors = {}

    for participant in coordinator.participants:
        visible = released if participant.role == "source" else ready
        handler = CallbackPathHandler(
            lambda _command: PreparedRankTransfer(None, requires_io=False),
            commit=lambda _command, _prepared, _completion, visible=visible,
            participant=participant: visible.append(participant),
        )
        executors[participant] = RankLocalCommandExecutor(
            participant=participant,
            queues=queues,
            handlers={path: handler for path in TransferPath},
            emit_ack=emitted.append,
            report_failure=lambda failure: (_ for _ in ()).throw(
                AssertionError(failure)
            ),
        )

    plan = make_plan(
        GenerationKey("run", "tp8-barrier", 0),
        TransferOperation.DIRECT,
        Owner.D_GPU,
        Owner.PREFILL_READY,
        "tp8-barrier-lease",
    )
    orchestrator.begin(plan)

    def deliver(command_value, participants):
        del emitted[:]
        for participant in participants:
            executors[participant].handle(command_value)
        acks = tuple(emitted)
        del emitted[:]
        for ack in acks:
            orchestrator.on_ack(ack)

    participants = coordinator.participants
    deliver(commands[-1], participants)  # PREPARE
    deliver(commands[-1], participants)  # START (no-I/O physical fence)
    release = commands[-1]
    assert release.kind is CommandKind.RELEASE
    deliver(release, participants[:-1])
    assert commands[-1] is release
    assert ready == []
    assert released == []

    deliver(release, participants[-1:])
    handoff = commands[-1]
    assert handoff.kind is CommandKind.HANDOFF
    assert ready == []
    assert released == []
    deliver(handoff, participants[:-1])
    assert commands[-1] is handoff
    assert ready == []
    # HANDOFF itself proves that rank zero observed every target BOUND ACK.
    # Source ranks may release as soon as they consume that command; they do
    # not wait for unrelated target HANDOFF ACKs or scheduler activation.
    assert len(released) == 8
    deliver(handoff, participants[-1:])
    activate = commands[-1]
    assert activate.kind is CommandKind.ACTIVATE
    deliver(activate, participants)
    ticket = commands[-1]
    assert ticket.kind is CommandKind.ISSUE_ACTIVATION_TICKET
    assert len(ready) == 8
    assert len(released) == 8
    deliver(ticket, participants)
    for participant in participants:
        if participant.role == "target":
            executors[participant].activate_staged(
                ticket.key,
                ticket.attempt,
                ticket.lease_id,
            )
            executors[participant].confirm_scheduler_adopted(
                ticket.key,
                ticket.attempt,
                ticket.lease_id,
            )
    staged_acks = tuple(emitted)
    del emitted[:]
    for ack in staged_acks:
        orchestrator.on_ack(ack)
    finalize = commands[-1]
    assert finalize.kind is CommandKind.FINALIZE
    deliver(finalize, participants)
    assert len(ready) == 8
    assert len(released) == 8
    assert orchestrator.active_count == 0
    queues.close()


class FakeRemoteWorker:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def load(self, shard, *, attempt_id, cancel_check, **_kwargs):
        self.entered.set()
        while not self.release.wait(0.001):
            if cancel_check():
                raise RuntimeError("drained cancellation")
        return ReadReceipt(
            "snapshot",
            "export",
            0,
            1,
            "a" * 64,
            2,
            attempt_id,
        )


def test_remote_host_adapter_reports_after_read_fence():
    worker = FakeRemoteWorker()
    executor = RemoteHostLoadExecutor(worker, max_workers=1)
    payload = RemoteHostLoadPayload({"host": "shard"}, device_indices=(1, 2))
    notifications = []
    handle = executor.submit(
        SimpleNamespace(attempt_id="1", payload=payload),
        lambda: notifications.append(True),
    )
    assert worker.entered.wait(1.0)
    assert executor.progress(handle).state is PhysicalState.INFLIGHT
    worker.release.set()
    deadline = time.monotonic() + 1.0
    while not notifications and time.monotonic() < deadline:
        time.sleep(0.001)
    progress = executor.progress(handle)
    assert progress.state is PhysicalState.SUCCEEDED
    assert progress.fence is FenceKind.DMA_COMPLETE
    assert executor.receipt(handle).read_id == "1"
    assert progress.result["read_receipt"]["read_id"] == "1"
    executor.close()


def test_remote_host_adapter_never_fabricates_unknown_drain_fence():
    class BrokenRemoteWorker:
        def load(self, *_args, **_kwargs):
            raise RuntimeError("unexpected worker failure")

    executor = RemoteHostLoadExecutor(BrokenRemoteWorker(), max_workers=1)
    payload = RemoteHostLoadPayload({"host": "shard"}, device_indices=(1, 2))
    handle = executor.submit(
        SimpleNamespace(attempt_id="broken", payload=payload), lambda: None
    )
    deadline = time.monotonic() + 1.0
    while not handle.future.done() and time.monotonic() < deadline:
        time.sleep(0.001)
    try:
        executor.progress(handle)
    except UnfencedRemoteRead as error:
        assert "failed without a drain receipt" in str(error)
    else:
        raise AssertionError("unknown READ failure was incorrectly fenced")
    executor.close()
def test_host_eviction_selects_only_the_source_tp_group():
    participants = tuple(
        LinkParticipant(role, group, rank)
        for role, group in (("prefill", "p0"), ("decode", "d0"))
        for rank in range(2)
    )
    plan = GroupTransferPlan(
        key=GenerationKey("run", "evict", 1),
        path=TransferPath.D2P_HOST,
        operation=TransferOperation.HOST_EVICT,
        source_owner=Owner.D_HOST,
        target_owner=Owner.NONE,
        lease_id="evict:evict:1",
        payload={"kind": "d2p_host_evict"},
        source_group="d0",
    )

    assert plan_endpoint_groups(plan, participants) == ("d0",)


def test_host_eviction_commits_none_owner_and_explicit_terminal():
    coordinator = LinkLifecycleCoordinator.from_endpoint_sizes(
        "run", "d-host", [("decode", "d0", 2)]
    )
    commands = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator, broadcast=commands.append
    )
    plan = GroupTransferPlan(
        key=GenerationKey("run", "evict-terminal", 1),
        path=TransferPath.D2P_HOST,
        operation=TransferOperation.HOST_EVICT,
        source_owner=Owner.D_HOST,
        target_owner=Owner.NONE,
        lease_id="evict:terminal",
        payload={"kind": "d2p_host_evict"},
        source_group="d0",
    )

    drive_attempt(orchestrator, coordinator, commands, plan)

    record = coordinator.record(plan.key)
    assert record.owner is Owner.NONE
    assert record.terminal is GenerationTerminal.EVICTED
