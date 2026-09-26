from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
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
)
from sglang.srt.disaggregation.agentic_memory_authority import AgenticMemoryAuthority
from sglang.srt.disaggregation.agentic_remote_host import HostShard, ReadReceipt
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
    assert released == []
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
    rank.handle(command(CommandKind.PREPARE, 1, path=TransferPath.D2P_HOST))
    rank.handle(command(CommandKind.START, 2, path=TransferPath.D2P_HOST))
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


def send_ack(orchestrator, participant, command_value, phase, *, result=None):
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
                result={} if result is None else result,
            ),
        )
    )


def drive_attempt(orchestrator, coordinator, commands, plan):
    attempt = orchestrator.begin(plan)
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
        scheduler_activate = commands[-1]
        assert scheduler_activate.kind is CommandKind.SCHEDULER_ACTIVATE
        for participant in coordinator.participants:
            send_ack(
                orchestrator,
                participant,
                scheduler_activate,
                RankPhase.ACTIVATION_ARMED,
            )
        publish = commands[-1]
        assert publish.kind is CommandKind.PUBLISH_ACTIVATION
        for participant in coordinator.participants:
            send_ack(orchestrator, participant, publish, RankPhase.ACTIVATION_READY)
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
    scheduler_activate = commands[-1]
    assert scheduler_activate.kind is CommandKind.SCHEDULER_ACTIVATE
    for participant in coordinator.participants:
        send_ack(
            orchestrator, participant, scheduler_activate, RankPhase.ACTIVATION_ARMED
        )
    publish = commands[-1]
    for participant in coordinator.participants:
        send_ack(orchestrator, participant, publish, RankPhase.ACTIVATION_READY)
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
    assert authority.take_ready("prefill-ready", timeout=0.01) == ()
    scheduler_activate = command(CommandKind.SCHEDULER_ACTIVATE, 6)
    rank.handle(scheduler_activate)
    rank.handle(command(CommandKind.PUBLISH_ACTIVATION, 7))
    rank.handle(command(CommandKind.ISSUE_ACTIVATION_TICKET, 8))
    rank.activate_staged(
        scheduler_activate.key,
        scheduler_activate.attempt,
        scheduler_activate.lease_id,
    )
    rank.confirm_scheduler_adopted(
        scheduler_activate.key,
        scheduler_activate.attempt,
        scheduler_activate.lease_id,
    )
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
    assert released == []
    deliver(handoff, participants[-1:])
    activate = commands[-1]
    assert activate.kind is CommandKind.ACTIVATE
    deliver(activate, participants)
    scheduler_activate = commands[-1]
    assert scheduler_activate.kind is CommandKind.SCHEDULER_ACTIVATE
    assert ready == []
    assert len(released) == 8
    deliver(scheduler_activate, participants)
    publish = commands[-1]
    assert publish.kind is CommandKind.PUBLISH_ACTIVATION
    deliver(publish, participants)
    ticket = commands[-1]
    assert ticket.kind is CommandKind.ISSUE_ACTIVATION_TICKET
    deliver(ticket, participants)
    assert ready == []
    for participant in participants:
        if participant.role == "target":
            executors[participant].activate_staged(
                scheduler_activate.key,
                scheduler_activate.attempt,
                scheduler_activate.lease_id,
            )
            executors[participant].confirm_scheduler_adopted(
                scheduler_activate.key,
                scheduler_activate.attempt,
                scheduler_activate.lease_id,
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
