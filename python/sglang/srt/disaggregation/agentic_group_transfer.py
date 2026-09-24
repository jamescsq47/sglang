"""TP-group orchestration for the agentic KV data plane.

This module connects three deliberately separate layers:

* :mod:`agentic_group_protocol` owns the rank-zero lifecycle decision;
* :mod:`agentic_transfer_queues` owns bounded rank-local physical I/O lanes;
* :mod:`agentic_memory_authority` owns rank-local accelerator pages.

The orchestrator never routes requests or allocates pages.  Rank zero publishes
one immutable command, every rank executes the same attempt for its shard, and
ownership changes only after every rank reports a real physical fence.  No
filesystem, shared-directory scan, timer poll, or scheduler callback is used.

Host-store attempts use the same commit sequence as Direct transfers.  Thus a
source-release hook runs immediately after every rank has made its Host shard
durable, while a Host-restore attempt publishes its target lease only after all
ranks have completed their reads.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol

from sglang.srt.disaggregation.agentic_group_protocol import (
    AttemptOutcome,
    CommandKind,
    GenerationKey,
    GroupCommand,
    LinkFailure,
    LinkIntent,
    LinkLifecycleCoordinator,
    LinkParticipant,
    LinkRankAck,
    Owner,
    RankAck,
    RankPhase,
)
from sglang.srt.disaggregation.agentic_memory_authority import (
    AgenticMemoryAuthority,
    PhysicalMemoryLease,
    RequestGenerationAttempt,
)
from sglang.srt.disaggregation.agentic_remote_host_worker import (
    DrainedRemoteRead,
    RemoteHostRankWorker,
    UnfencedRemoteRead,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    AgenticTransferQueues,
    FenceKind,
    PhysicalProgress,
    PhysicalState,
    TransferAttempt,
    TransferCompletion,
    TransferExecutor,
    TransferPath,
)


class TransferOperation(str, Enum):
    """Physical operation represented by one group attempt."""

    DIRECT = "direct"
    HOST_STORE = "host_store"
    HOST_RESTORE = "host_restore"


class TargetLeaseKind(str, Enum):
    """Optional destination reservation made by a rank-local authority."""

    NONE = "none"
    PREFILL_WORKSET = "prefill_workset"
    DECODE_RESERVATION = "decode_reservation"


@dataclass(frozen=True, slots=True)
class GroupTransferPlan:
    """Immutable rank-zero intent for one request-generation attempt."""

    key: GenerationKey
    path: TransferPath
    operation: TransferOperation
    source_owner: Owner
    target_owner: Owner
    lease_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.lease_id:
            raise ValueError("lease_id must be non-empty")
        value = copy.deepcopy(dict(self.payload))
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "payload", MappingProxyType(value))

    def command_payload(self) -> Mapping[str, Any]:
        return {
            "agentic_data_plane": {
                "version": 1,
                "path": self.path.value,
                "operation": self.operation.value,
            },
            "transfer": copy.deepcopy(dict(self.payload)),
        }


@dataclass(frozen=True, slots=True)
class PreparedRankTransfer:
    """Rank-local state produced by PREPARE and retained through RELEASE."""

    transfer_payload: Any
    physical_lease_id: Optional[int] = None
    requires_io: bool = True


class RankPathHandler(Protocol):
    """Rank-local resource hooks; none of these hooks may choose a route."""

    def prepare(self, command: GroupCommand) -> PreparedRankTransfer:
        ...

    def begin_io(
        self, command: GroupCommand, prepared: PreparedRankTransfer
    ) -> None:
        ...

    def finish_io(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        ...

    def prepare_handoff(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        """Prepare local binding without publishing ready or releasing source."""
        ...

    def commit(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        ...

    def abort(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: Optional[TransferCompletion],
    ) -> None:
        ...


class CallbackPathHandler:
    """Small adapter used for native NIXL/Host implementations.

    The callbacks receive only the immutable command and retained rank-local
    preparation result.  This is the intended integration seam for existing
    NIXL senders, local D2H workers, and source-release callbacks.
    """

    def __init__(
        self,
        prepare: Callable[[GroupCommand], PreparedRankTransfer],
        *,
        begin_io: Optional[
            Callable[[GroupCommand, PreparedRankTransfer], None]
        ] = None,
        finish_io: Optional[
            Callable[
                [GroupCommand, PreparedRankTransfer, TransferCompletion], None
            ]
        ] = None,
        commit: Optional[
            Callable[
                [GroupCommand, PreparedRankTransfer, TransferCompletion], None
            ]
        ] = None,
        prepare_handoff: Optional[
            Callable[
                [GroupCommand, PreparedRankTransfer, TransferCompletion], None
            ]
        ] = None,
        abort: Optional[
            Callable[
                [
                    GroupCommand,
                    PreparedRankTransfer,
                    Optional[TransferCompletion],
                ],
                None,
            ]
        ] = None,
    ) -> None:
        self._prepare = prepare
        self._begin_io = begin_io or (lambda _command, _prepared: None)
        self._finish_io = finish_io or (
            lambda _command, _prepared, _completion: None
        )
        self._commit = commit or (
            lambda _command, _prepared, _completion: None
        )
        self._prepare_handoff = prepare_handoff or (
            lambda _command, _prepared, _completion: None
        )
        self._abort = abort or (
            lambda _command, _prepared, _completion: None
        )

    def prepare(self, command: GroupCommand) -> PreparedRankTransfer:
        return self._prepare(command)

    def begin_io(
        self, command: GroupCommand, prepared: PreparedRankTransfer
    ) -> None:
        self._begin_io(command, prepared)

    def finish_io(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        self._finish_io(command, prepared, completion)

    def commit(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        self._commit(command, prepared, completion)

    def prepare_handoff(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        self._prepare_handoff(command, prepared, completion)

    def abort(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: Optional[TransferCompletion],
    ) -> None:
        self._abort(command, prepared, completion)


class AuthorityPathHandler:
    """Reserve and publish one destination lease through the sole authority.

    This handler deliberately manages destination leases only.  Source pages
    stay under their existing owner until ``source_release`` runs after the
    all-rank commit.  That avoids inventing a second allocator free list and
    also keeps an aborted outgoing transfer reusable by its current owner.
    """

    def __init__(
        self,
        authority: AgenticMemoryAuthority,
        *,
        lease_kind: TargetLeaseKind,
        owner: str,
        payload_builder: Callable[
            [GroupCommand, Optional[PhysicalMemoryLease]], Any
        ],
        ready_queue: Optional[str] = None,
        source_release: Optional[
            Callable[
                [GroupCommand, PreparedRankTransfer, TransferCompletion], None
            ]
        ] = None,
    ) -> None:
        if lease_kind is TargetLeaseKind.NONE and ready_queue is not None:
            raise ValueError("a ready queue requires a destination lease")
        self._authority = authority
        self._lease_kind = lease_kind
        self._owner = str(owner)
        self._payload_builder = payload_builder
        self._ready_queue = ready_queue
        self._source_release = source_release

    @staticmethod
    def _transfer_values(command: GroupCommand) -> Mapping[str, Any]:
        value = command.payload.get("transfer", {})
        if not isinstance(value, Mapping):
            raise ValueError("transfer payload must be a mapping")
        return value

    def prepare(self, command: GroupCommand) -> PreparedRankTransfer:
        values = self._transfer_values(command)
        key = RequestGenerationAttempt(
            request_id=command.key.request_id,
            generation=command.key.generation,
            attempt=command.attempt,
        )
        lease: Optional[PhysicalMemoryLease]
        if self._lease_kind is TargetLeaseKind.PREFILL_WORKSET:
            lease = self._authority.reserve_prefill_workset(
                key,
                owner=self._owner,
                parent_tokens=int(values["parent_tokens"]),
                prompt_tokens=int(values["prompt_tokens"]),
                state_slot_counts=values.get("state_slot_counts"),
            )
        elif self._lease_kind is TargetLeaseKind.DECODE_RESERVATION:
            lease = self._authority.reserve_decode(
                key,
                owner=self._owner,
                prompt_tokens=int(values["prompt_tokens"]),
                decode_growth_tokens=int(values["decode_growth_tokens"]),
                state_slot_counts=values.get("state_slot_counts"),
            )
        else:
            lease = None
        if self._lease_kind is not TargetLeaseKind.NONE and lease is None:
            raise MemoryError("rank-local target workset is unavailable")
        payload = self._payload_builder(command, lease)
        return PreparedRankTransfer(
            transfer_payload=payload,
            physical_lease_id=None if lease is None else lease.lease_id,
        )

    def begin_io(
        self, command: GroupCommand, prepared: PreparedRankTransfer
    ) -> None:
        if prepared.physical_lease_id is not None and not self._authority.begin_io(
            prepared.physical_lease_id, str(command.attempt)
        ):
            raise RuntimeError("destination lease could not enter I/O ownership")

    def finish_io(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        lease_id = prepared.physical_lease_id
        if lease_id is None:
            return
        success = completion.state is PhysicalState.SUCCEEDED
        if not self._authority.complete_io(
            lease_id, str(command.attempt), success=success
        ):
            raise RuntimeError("stale or duplicate physical I/O completion")

    def commit(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        if prepared.physical_lease_id is not None and self._ready_queue is not None:
            event = self._authority.publish_ready(
                prepared.physical_lease_id, self._ready_queue
            )
            if event is None:
                raise RuntimeError("committed target lease was not publishable")
        if self._source_release is not None:
            self._source_release(command, prepared, completion)

    def prepare_handoff(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: TransferCompletion,
    ) -> None:
        # Generic authority binding is represented by the already-complete
        # destination lease.  Model-specific providers may perform local Radix
        # binding here, but must not publish ready or release source ownership.
        return None

    def abort(
        self,
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        completion: Optional[TransferCompletion],
    ) -> None:
        lease_id = prepared.physical_lease_id
        if lease_id is None:
            return
        # A cancelled destination never becomes scheduler-visible.
        self._authority.request_release(lease_id)
        self._authority.commit_release(lease_id, reason="group_attempt_aborted")


@dataclass(frozen=True, slots=True)
class LocalAttemptFailure:
    key: GenerationKey
    attempt: int
    participant: LinkParticipant
    lease_id: str
    detail: str


@dataclass(slots=True)
class _LocalAttempt:
    prepare_command: GroupCommand
    path: TransferPath
    operation: TransferOperation
    handler: RankPathHandler
    prepared: Optional[PreparedRankTransfer] = None
    transfer: Optional[TransferAttempt] = None
    start_command: Optional[GroupCommand] = None
    cancel_command: Optional[GroupCommand] = None
    completion: Optional[TransferCompletion] = None
    failure_detail: Optional[str] = None
    submit_ack_sent: bool = False
    completion_pending: bool = False
    abort_done: bool = False
    handoff_prepared: bool = False
    activation_staged: bool = False
    scheduler_activate_command: Optional[GroupCommand] = None
    scheduler_publish_command: Optional[GroupCommand] = None
    activation_ticket_command: Optional[GroupCommand] = None
    activation_published: bool = False
    scheduler_activated: bool = False


class RankLocalCommandExecutor:
    """Execute rank-zero commands for exactly one TP rank.

    PREPARE may reserve local destination pages.  START only enters the
    already-selected path queue.  RELEASE performs the committed handoff and
    source release.  A physical failure is first reported to rank zero; the
    failed/drained ACK is emitted only after rank zero broadcasts CANCEL.
    """

    def __init__(
        self,
        *,
        participant: LinkParticipant,
        queues: AgenticTransferQueues,
        handlers: Mapping[Any, RankPathHandler],
        emit_ack: Callable[[LinkRankAck], None],
        report_failure: Callable[[LocalAttemptFailure], None],
    ) -> None:
        for key in handlers:
            if isinstance(key, TransferPath):
                continue
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], TransferPath)
                or not isinstance(key[1], TransferOperation)
            ):
                raise ValueError(
                    "handler keys must be TransferPath or (path, operation)"
                )
        self.participant = participant
        self._queues = queues
        self._handlers = dict(handlers)
        self._emit_ack = emit_ack
        self._report_failure = report_failure
        self._attempts: dict[tuple[GenerationKey, int], _LocalAttempt] = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)

    def _handler_for(
        self, path: TransferPath, operation: TransferOperation
    ) -> RankPathHandler:
        handler = self._handlers.get((path, operation))
        if handler is None:
            handler = self._handlers.get(path)
        if handler is None:
            raise ValueError(
                f"missing handler for {path.value}/{operation.value}"
            )
        return handler

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._attempts)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: not self._attempts, timeout)

    @staticmethod
    def _identity(command: GroupCommand) -> tuple[GenerationKey, int]:
        return command.key, command.attempt

    @staticmethod
    def _plan(command: GroupCommand) -> tuple[TransferPath, TransferOperation]:
        header = command.payload.get("agentic_data_plane", {})
        if not isinstance(header, Mapping) or int(header.get("version", 0)) != 1:
            raise ValueError("missing agentic data-plane v1 command header")
        return TransferPath(header["path"]), TransferOperation(header["operation"])

    def _ack(
        self,
        command: GroupCommand,
        phase: RankPhase,
        *,
        ok: bool = True,
        detail: str = "",
        result: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._emit_ack(
            LinkRankAck(
                participant=self.participant,
                ack=RankAck(
                    key=command.key,
                    attempt=command.attempt,
                    command_seq=command.command_seq,
                    rank=self.participant.rank,
                    phase=phase,
                    lease_id=command.lease_id,
                ok=ok,
                detail=detail,
                result=result or {},
                ),
            )
        )

    def handle(self, command: GroupCommand) -> None:
        if command.kind is CommandKind.PREPARE:
            self._prepare(command)
        elif command.kind is CommandKind.START:
            self._start(command)
        elif command.kind is CommandKind.CANCEL:
            self._cancel(command)
        elif command.kind is CommandKind.RELEASE:
            self._release(command)
        elif command.kind is CommandKind.HANDOFF:
            self._handoff(command)
        elif command.kind is CommandKind.ACTIVATE:
            self._activate(command)
        elif command.kind is CommandKind.SCHEDULER_ACTIVATE:
            self._scheduler_activate(command)
        elif command.kind is CommandKind.PUBLISH_ACTIVATION:
            self._publish_activation(command)
        elif command.kind is CommandKind.ISSUE_ACTIVATION_TICKET:
            self._issue_activation_ticket(command)
        elif command.kind is CommandKind.FINALIZE:
            self._finalize(command)
        elif command.kind is CommandKind.ABORT_FINALIZE:
            self._abort_finalize(command)
        else:  # pragma: no cover - exhaustive enum guard
            raise ValueError(f"unsupported command {command.kind}")

    def _prepare(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        path, operation = self._plan(command)
        with self._changed:
            if identity in self._attempts:
                raise RuntimeError("duplicate PREPARE command")
            local = _LocalAttempt(
                prepare_command=command,
                path=path,
                operation=operation,
                handler=self._handler_for(path, operation),
            )
            self._attempts[identity] = local
            self._changed.notify_all()
        try:
            prepared = local.handler.prepare(command)
            if not isinstance(prepared, PreparedRankTransfer):
                raise TypeError("path prepare hook returned an invalid result")
        except BaseException as error:
            detail = f"prepare failed: {type(error).__name__}: {error}"
            with self._lock:
                local.failure_detail = detail
            self._report_failure(
                LocalAttemptFailure(
                    command.key,
                    command.attempt,
                    self.participant,
                    command.lease_id,
                    detail,
                )
            )
            return
        with self._lock:
            local.prepared = prepared
        self._ack(command, RankPhase.PREPARED)

    def _start(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or local.prepared is None:
                raise RuntimeError("START arrived before successful PREPARE")
            if local.start_command is not None:
                raise RuntimeError("duplicate START command")
            local.start_command = command
            transfer = TransferAttempt(
                snapshot_id=command.key.snapshot_id,
                attempt_id=str(command.attempt),
                lease_id=command.lease_id,
                path=local.path,
                payload=local.prepared.transfer_payload,
            )
            local.transfer = transfer
            requires_io = local.prepared.requires_io
        if not requires_io:
            # The endpoint still joins the same 16-participant transaction,
            # but this Host leg has no physical operation on this side.
            self._ack(command, RankPhase.DMA_SUBMITTED)
            now = time.monotonic()
            completion = TransferCompletion(
                attempt=transfer,
                state=PhysicalState.SUCCEEDED,
                fence=FenceKind.NO_IO_REQUIRED,
                error=None,
                submitted_at=now,
                completed_at=now,
            )
            with self._lock:
                local.submit_ack_sent = True
                local.completion = completion
            self._ack(command, RankPhase.DMA_DONE, detail="no_io_required")
            return
        try:
            local.handler.begin_io(command, local.prepared)
            accepted = self._queues.submit(
                transfer,
                lambda completion, identity=identity: self._completed(
                    identity, completion
                ),
            )
            if not accepted:
                raise RuntimeError("transfer identity is already live")
        except BaseException as error:
            detail = f"submit failed: {type(error).__name__}: {error}"
            now = time.monotonic()
            completion = TransferCompletion(
                attempt=transfer,
                state=PhysicalState.FAILED,
                fence=FenceKind.NOT_POSTED,
                error=detail,
                submitted_at=now,
                completed_at=now,
            )
            local.handler.finish_io(command, local.prepared, completion)
            with self._lock:
                local.failure_detail = detail
                local.completion = completion
            self._report_failure(
                LocalAttemptFailure(
                    command.key,
                    command.attempt,
                    self.participant,
                    command.lease_id,
                    detail,
                )
            )
            return

        # Emit DMA_SUBMITTED before consuming even an immediately completed
        # queue callback.  This preserves the group protocol's phase ordering.
        self._ack(command, RankPhase.DMA_SUBMITTED)
        pending = False
        with self._lock:
            local.submit_ack_sent = True
            pending = local.completion_pending
            local.completion_pending = False
        if pending:
            self._finish_completion(identity)

    def _completed(
        self,
        identity: tuple[GenerationKey, int],
        completion: TransferCompletion,
    ) -> None:
        with self._lock:
            local = self._attempts.get(identity)
            if local is None:
                raise RuntimeError("completion belongs to an unknown attempt")
            if local.completion is not None:
                raise RuntimeError("physical attempt completed twice")
            local.completion = completion
            if not local.submit_ack_sent:
                local.completion_pending = True
                return
        self._finish_completion(identity)

    def _finish_completion(self, identity: tuple[GenerationKey, int]) -> None:
        with self._lock:
            local = self._attempts[identity]
            command = local.start_command
            completion = local.completion
            prepared = local.prepared
        if command is None or completion is None or prepared is None:
            raise RuntimeError("incomplete rank-local completion state")
        local.handler.finish_io(command, prepared, completion)
        if completion.state is PhysicalState.SUCCEEDED:
            self._ack(command, RankPhase.DMA_DONE, result=completion.result)
            return
        detail = completion.error or completion.state.value
        with self._lock:
            local.failure_detail = detail
            cancel = local.cancel_command
        self._report_failure(
            LocalAttemptFailure(
                command.key,
                command.attempt,
                self.participant,
                command.lease_id,
                detail,
            )
        )
        if cancel is not None:
            self._finish_abort(identity)

    def _cancel(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None:
                raise RuntimeError("CANCEL belongs to an unknown attempt")
            if local.cancel_command is not None:
                raise RuntimeError("duplicate CANCEL command")
            local.cancel_command = command
            transfer = local.transfer
            completion = local.completion
        if transfer is None:
            self._finish_abort(identity)
            return
        if completion is not None:
            self._finish_abort(identity)
            return
        if not self._queues.cancel(transfer):
            raise RuntimeError("live transfer disappeared before cancellation")

    def _finish_abort(self, identity: tuple[GenerationKey, int]) -> None:
        with self._lock:
            local = self._attempts[identity]
            if local.abort_done:
                return
            cancel = local.cancel_command
            prepared = local.prepared
            completion = local.completion
            if cancel is None:
                return
            if local.transfer is not None and completion is None:
                return
            local.abort_done = True
        detail = local.failure_detail or str(cancel.payload.get("reason", "cancelled"))
        result = {} if completion is None else dict(completion.result)
        result["fence"] = (
            FenceKind.NOT_POSTED.value
            if completion is None
            else completion.fence.value
        )
        self._ack(
            cancel,
            RankPhase.FAILED_DRAINED,
            ok=False,
            detail=detail,
            result=result,
        )

    def _abort_finalize(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or not local.abort_done or local.cancel_command is None:
                raise RuntimeError("ABORT_FINALIZE arrived before local drain")
            prepared = local.prepared
            completion = local.completion
        if prepared is not None:
            local.handler.abort(command, prepared, completion)
        self._ack(
            command,
            RankPhase.ABORTED,
            detail=local.failure_detail or "cancelled",
        )
        with self._changed:
            self._attempts.pop(identity, None)
            self._changed.notify_all()

    def _release(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or local.prepared is None or local.completion is None:
                raise RuntimeError("RELEASE arrived before physical completion")
            if local.completion.state is not PhysicalState.SUCCEEDED:
                raise RuntimeError("failed physical attempt cannot commit")
        try:
            local.handler.prepare_handoff(command, local.prepared, local.completion)
        except BaseException as error:
            detail = f"handoff preparation failed: {type(error).__name__}: {error}"
            with self._lock:
                local.failure_detail = detail
            self._report_failure(
                LocalAttemptFailure(
                    command.key,
                    command.attempt,
                    self.participant,
                    command.lease_id,
                    detail,
                )
            )
            return
        with self._lock:
            local.handoff_prepared = True
        self._ack(command, RankPhase.BOUND)

    def _handoff(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if (
                local is None
                or local.prepared is None
                or local.completion is None
                or not local.handoff_prepared
            ):
                raise RuntimeError("HANDOFF arrived before local bind preparation")
            if local.completion.state is not PhysicalState.SUCCEEDED:
                raise RuntimeError("failed physical attempt cannot hand off")
        self._ack(command, RankPhase.RELEASED)

    def _activate(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if (
                local is None
                or local.prepared is None
                or local.completion is None
                or not local.handoff_prepared
            ):
                raise RuntimeError("ACTIVATE arrived before local handoff staging")
            if local.completion.state is not PhysicalState.SUCCEEDED:
                raise RuntimeError("failed physical attempt cannot activate")
        # All target ranks are already bound before this command exists.  A
        # source shard may therefore release immediately; target shards remain
        # invisible until their endpoint TP0 carries SCHEDULER_ACTIVATE on the
        # native scheduler broadcast.
        source_role = self._owner_role(command.source_owner)
        if self.participant.role in {source_role, "source"}:
            local.handler.commit(command, local.prepared, local.completion)
        with self._lock:
            local.activation_staged = True
        self._ack(command, RankPhase.STAGED)

    @staticmethod
    def _owner_role(owner: Owner) -> Optional[str]:
        if owner in {Owner.P_GPU, Owner.P_HOST, Owner.PREFILL_READY}:
            return "prefill"
        if owner in {Owner.D_GPU, Owner.D_HOST, Owner.DECODE_READY}:
            return "decode"
        return None

    @staticmethod
    def _scheduler_target(owner: Owner) -> bool:
        return owner in {
            Owner.P_GPU,
            Owner.PREFILL_READY,
            Owner.D_GPU,
            Owner.DECODE_READY,
        }

    def _scheduler_activate(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or not local.activation_staged:
                raise RuntimeError("SCHEDULER_ACTIVATE arrived before local staging")
            if local.scheduler_activate_command is not None:
                raise RuntimeError("duplicate SCHEDULER_ACTIVATE command")
            local.scheduler_activate_command = command
        self._ack(command, RankPhase.ACTIVATION_ARMED)

    def _publish_activation(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or local.scheduler_activate_command is None:
                raise RuntimeError("PUBLISH_ACTIVATION arrived before local arm")
            if local.scheduler_publish_command is not None:
                raise RuntimeError("duplicate PUBLISH_ACTIVATION command")
            local.scheduler_publish_command = command
        self._ack(command, RankPhase.ACTIVATION_READY)

    def _issue_activation_ticket(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or local.scheduler_publish_command is None:
                raise RuntimeError("activation ticket arrived before ready barrier")
            if local.activation_ticket_command is not None:
                raise RuntimeError("duplicate activation ticket command")
            local.activation_ticket_command = command
            scheduler_activated = local.scheduler_activated
        target_role = self._owner_role(command.target_owner)
        if self._scheduler_target(command.target_owner) and self.participant.role in {
            target_role,
            "target",
        }:
            if scheduler_activated:
                self._ack(command, RankPhase.ACTIVATED)
            return
        self._ack(command, RankPhase.ACTIVATED, detail="no_scheduler_target")

    def activate_staged(
        self, key: GenerationKey, attempt: int, lease_id: str
    ) -> None:
        """Publish one staged lease, without claiming scheduler adoption.

        The native TP broadcast has two distinct boundaries: publishing the
        rank-local ready lease and actually inserting the request into the
        scheduler queue.  Only ``confirm_scheduler_adopted`` crosses the
        latter boundary and emits ``ACTIVATED``.
        """

        identity = (key, int(attempt))
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or local.scheduler_publish_command is None:
                raise RuntimeError("scheduler activation has no staged local attempt")
            command = local.scheduler_publish_command
            if command.lease_id != str(lease_id):
                raise RuntimeError("scheduler activation lease does not match")
            if local.activation_published:
                return
            prepared = local.prepared
            completion = local.completion
        if prepared is None or completion is None:
            raise RuntimeError("scheduler activation lacks physical completion")
        local.handler.commit(command, prepared, completion)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None:
                raise RuntimeError("published activation disappeared")
            local.activation_published = True

    def confirm_scheduler_adopted(
        self, key: GenerationKey, attempt: int, lease_id: str
    ) -> None:
        """Acknowledge activation only after native scheduler queue insertion."""

        identity = (key, int(attempt))
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or not local.activation_published:
                raise RuntimeError("scheduler adopted an unpublished activation")
            command = local.activation_ticket_command
            if command is not None and command.lease_id != str(lease_id):
                raise RuntimeError("scheduler adoption lease does not match")
            if local.scheduler_activated:
                return
            local.scheduler_activated = True
        # TCP ticket delivery may trail the native TP broadcast.  In that
        # case _issue_activation_ticket observes scheduler_activated and emits
        # the ACK; no polling or fabricated success is needed here.
        if command is not None:
            self._ack(command, RankPhase.ACTIVATED)

    def _finalize(self, command: GroupCommand) -> None:
        identity = self._identity(command)
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or not local.activation_staged:
                raise RuntimeError("FINALIZE arrived before local staging")
        # Source release happened at ACTIVATE and target publication happened
        # through the endpoint scheduler broadcast.  FINALIZE only retires the
        # short-lived physical attempt after both facts are fenced.
        self._ack(command, RankPhase.FINALIZED)
        with self._changed:
            self._attempts.pop(identity, None)
            self._changed.notify_all()


@dataclass(slots=True)
class _LinkAttempt:
    plan: GroupTransferPlan
    phase: CommandKind = CommandKind.PREPARE
    dma_results: dict[LinkParticipant, Mapping[str, Any]] = None
    handoff_payload: Mapping[str, Any] = None
    abort_reason: str = ""

    def __post_init__(self) -> None:
        if self.dma_results is None:
            self.dma_results = {}
        if self.handoff_payload is None:
            self.handoff_payload = MappingProxyType({})


class RankZeroLinkOrchestrator:
    """Drive one atomic source-TP + target-TP ownership transaction.

    The wrapped :class:`LinkLifecycleCoordinator` maps endpoint-qualified ACKs
    into one participant set.  Therefore commit and source release cannot
    happen after only one TP group completes.
    """

    def __init__(
        self,
        coordinator: LinkLifecycleCoordinator,
        *,
        broadcast: Callable[[GroupCommand], None],
        on_committed: Optional[Callable[[GroupTransferPlan, int], None]] = None,
        on_committed_results: Optional[
            Callable[
                [
                    GroupTransferPlan,
                    int,
                    Mapping[LinkParticipant, Mapping[str, Any]],
                ],
                None,
            ]
        ] = None,
        on_aborted: Optional[
            Callable[[GroupTransferPlan, int, str], None]
        ] = None,
        decide_intent: Optional[
            Callable[[LinkIntent], Optional[GroupTransferPlan]]
        ] = None,
    ) -> None:
        self._coordinator = coordinator
        self._broadcast = broadcast
        self._on_committed = on_committed
        self._on_committed_results = on_committed_results
        self._on_aborted = on_aborted
        self._decide_intent = decide_intent
        self._active: dict[tuple[GenerationKey, int], _LinkAttempt] = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: not self._active, timeout)

    def cancel_active(self, reason: str) -> int:
        """Cancel uncommitted attempts; committed RELEASE attempts only drain."""

        commands = []
        with self._changed:
            for (key, attempt_id), state in tuple(self._active.items()):
                if state.phase not in {
                    CommandKind.PREPARE,
                    CommandKind.START,
                    CommandKind.RELEASE,
                    CommandKind.HANDOFF,
                }:
                    continue
                if self._coordinator.outcome(key, attempt_id) is not AttemptOutcome.ACTIVE:
                    continue
                command = self._coordinator.request_abort(key, attempt_id, reason)
                state.abort_reason = str(reason)
                state.phase = CommandKind.CANCEL
                commands.append(command)
            self._changed.notify_all()
        for command in commands:
            self._broadcast(command)
        return len(commands)

    def begin(self, plan: GroupTransferPlan) -> int:
        with self._changed:
            command = self._coordinator.begin_attempt(
                plan.key,
                source_owner=plan.source_owner,
                target_owner=plan.target_owner,
                lease_id=plan.lease_id,
                required_commit_phase=RankPhase.RELEASED,
                payload=plan.command_payload(),
            )
            identity = (plan.key, command.attempt)
            if identity in self._active:
                raise RuntimeError("duplicate active link attempt")
            self._active[identity] = _LinkAttempt(plan)
            self._changed.notify_all()
            self._broadcast(command)
            return command.attempt

    def on_intent(self, intent: LinkIntent) -> Optional[int]:
        """Let fixed rank zero turn an endpoint proposal into a command.

        Endpoint rank zero can only send :class:`LinkIntent`; it never calls
        ``broadcast``.  The configured decision callback may reject the
        proposal or return the one immutable plan that this coordinator then
        publishes to every source/target participant.
        """

        if self._decide_intent is None:
            raise RuntimeError("this coordinator does not accept link intents")
        plan = self._decide_intent(intent)
        if plan is None:
            return None
        if plan.key != intent.key:
            raise ValueError("intent decision changed the request-generation key")
        return self.begin(plan)

    def on_ack(self, value: LinkRankAck) -> None:
        with self._lock:
            ack = value.ack
            changed = self._coordinator.apply_ack(value)
            if not changed:
                return
            identity = (ack.key, ack.attempt)
            state = self._active[identity]
            if ack.phase is RankPhase.DMA_DONE:
                state.dma_results[value.participant] = MappingProxyType(
                    copy.deepcopy(dict(ack.result))
                )
            elif ack.phase is RankPhase.FAILED_DRAINED:
                state.dma_results[value.participant] = MappingProxyType(
                    copy.deepcopy(dict(ack.result))
                )
            if state.phase is CommandKind.PREPARE and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.PREPARED
            ):
                command = self._coordinator.issue_start(
                    ack.key,
                    ack.attempt,
                    payload=state.plan.command_payload(),
                )
                state.phase = CommandKind.START
                self._broadcast(command)
                return
            if state.phase is CommandKind.START and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.DMA_DONE
            ):
                missing = set(self._coordinator.participants) - set(
                    state.dma_results
                )
                if missing:
                    raise RuntimeError(
                        f"committed link lacks DMA results from {sorted(map(str, missing))}"
                    )
                rank_results = [
                    {
                        "participant": {
                            "role": participant.role,
                            "endpoint_group": participant.endpoint_group,
                            "rank": participant.rank,
                        },
                        "result": dict(state.dma_results[participant]),
                    }
                    for participant in self._coordinator.participants
                ]
                state.handoff_payload = MappingProxyType(
                    {"rank_results": rank_results}
                )
                command = self._coordinator.issue_release(
                    ack.key,
                    ack.attempt,
                    payload=state.handoff_payload,
                )
                state.phase = CommandKind.RELEASE
                self._broadcast(command)
                return
            if state.phase is CommandKind.RELEASE and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.BOUND
            ):
                command = self._coordinator.issue_handoff(
                    ack.key,
                    ack.attempt,
                    payload=state.handoff_payload,
                )
                state.phase = CommandKind.HANDOFF
                self._broadcast(command)
                return
            if state.phase is CommandKind.HANDOFF and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.RELEASED
            ):
                command = self._coordinator.issue_activate(
                    ack.key,
                    ack.attempt,
                    payload=state.handoff_payload,
                )
                state.phase = CommandKind.ACTIVATE
                self._broadcast(command)
                return
            if state.phase is CommandKind.ACTIVATE and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.STAGED
            ):
                command = self._coordinator.issue_scheduler_activate(
                    ack.key,
                    ack.attempt,
                    payload=state.handoff_payload,
                )
                state.phase = CommandKind.SCHEDULER_ACTIVATE
                self._broadcast(command)
                return
            if (
                state.phase is CommandKind.SCHEDULER_ACTIVATE
                and self._coordinator.group_reached(
                    ack.key, ack.attempt, RankPhase.ACTIVATION_ARMED
                )
            ):
                command = self._coordinator.issue_publish_activation(
                    ack.key,
                    ack.attempt,
                    payload=state.handoff_payload,
                )
                state.phase = CommandKind.PUBLISH_ACTIVATION
                self._broadcast(command)
                return
            if (
                state.phase is CommandKind.PUBLISH_ACTIVATION
                and self._coordinator.group_reached(
                    ack.key, ack.attempt, RankPhase.ACTIVATION_READY
                )
            ):
                command = self._coordinator.issue_activation_ticket(
                    ack.key,
                    ack.attempt,
                    payload=state.handoff_payload,
                )
                state.phase = CommandKind.ISSUE_ACTIVATION_TICKET
                self._broadcast(command)
                return
            if (
                state.phase is CommandKind.ISSUE_ACTIVATION_TICKET
                and self._coordinator.group_reached(
                    ack.key, ack.attempt, RankPhase.ACTIVATED
                )
            ):
                self._coordinator.commit(ack.key, ack.attempt)
                command = self._coordinator.issue_finalize(
                    ack.key,
                    ack.attempt,
                    payload=state.handoff_payload,
                )
                state.phase = CommandKind.FINALIZE
                self._broadcast(command)
                return
            if state.phase is CommandKind.FINALIZE and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.FINALIZED
            ):
                self._coordinator.retire(ack.key, ack.attempt)
                self._active.pop(identity, None)
                self._changed.notify_all()
                if self._on_committed is not None:
                    self._on_committed(state.plan, ack.attempt)
                if self._on_committed_results is not None:
                    self._on_committed_results(
                        state.plan,
                        ack.attempt,
                        MappingProxyType(dict(state.dma_results)),
                    )
                return
            if state.phase is CommandKind.CANCEL and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.FAILED_DRAINED
            ):
                self._coordinator.complete_abort(ack.key, ack.attempt)
                missing = set(self._coordinator.participants) - set(
                    state.dma_results
                )
                if missing:
                    raise RuntimeError(
                        f"drained abort lacks results from {sorted(map(str, missing))}"
                    )
                rank_results = [
                    {
                        "participant": {
                            "role": participant.role,
                            "endpoint_group": participant.endpoint_group,
                            "rank": participant.rank,
                        },
                        "result": dict(state.dma_results[participant]),
                    }
                    for participant in self._coordinator.participants
                ]
                command = self._coordinator.issue_abort_finalize(
                    ack.key,
                    ack.attempt,
                    payload={"rank_results": rank_results},
                )
                state.phase = CommandKind.ABORT_FINALIZE
                self._broadcast(command)
                return
            if state.phase is CommandKind.ABORT_FINALIZE and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.ABORTED
            ):
                self._coordinator.retire(ack.key, ack.attempt)
                self._active.pop(identity, None)
                self._changed.notify_all()
                if self._on_aborted is not None:
                    self._on_aborted(
                        state.plan,
                        ack.attempt,
                        state.abort_reason or ack.detail,
                    )

    def on_local_failure(self, failure: LocalAttemptFailure) -> None:
        with self._lock:
            identity = (failure.key, failure.attempt)
            state = self._active.get(identity)
            if state is None or state.phase is CommandKind.CANCEL:
                return
            if self._coordinator.outcome(
                failure.key, failure.attempt
            ) is not AttemptOutcome.ACTIVE:
                return
            reason = f"{failure.participant}: {failure.detail}"
            command = self._coordinator.request_abort(
                failure.key,
                failure.attempt,
                reason,
            )
            state.abort_reason = reason
            state.phase = CommandKind.CANCEL
            self._broadcast(command)

    def on_link_failure(self, failure: LinkFailure) -> None:
        """Consume the relay-authenticated failure at fixed rank zero."""

        self.on_local_failure(
            LocalAttemptFailure(
                key=failure.key,
                attempt=failure.attempt,
                participant=failure.participant,
                lease_id=failure.lease_id,
                detail=failure.detail,
            )
        )


@dataclass(frozen=True, slots=True)
class RemoteHostLoadPayload:
    """Opaque local payload for one Host→GPU shard read."""

    shard: Any
    device_indices: Any = None
    page_indices: Any = None
    state_indices: Any = None


@dataclass(slots=True)
class _RemoteReadHandle:
    future: Future
    cancel: threading.Event
    receipt: Any = None


class RemoteHostLoadExecutor(TransferExecutor):
    """Turn the existing blocking Host worker into an edge-driven queue engine."""

    def __init__(
        self,
        worker: RemoteHostRankWorker,
        *,
        max_workers: int = 4,
    ) -> None:
        self._worker = worker
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="agentic-host-load",
        )

    def submit(
        self, attempt: TransferAttempt, notify: Callable[[], None]
    ) -> _RemoteReadHandle:
        payload = attempt.payload
        if not isinstance(payload, RemoteHostLoadPayload):
            raise TypeError("remote Host executor requires RemoteHostLoadPayload")
        cancel = threading.Event()

        def run():
            return self._worker.load(
                payload.shard,
                attempt_id=attempt.attempt_id,
                device_indices=payload.device_indices,
                page_indices=payload.page_indices,
                state_indices=payload.state_indices,
                cancel_check=cancel.is_set,
            )

        future = self._pool.submit(run)
        handle = _RemoteReadHandle(future=future, cancel=cancel)
        future.add_done_callback(lambda _future: notify())
        return handle

    def progress(self, handle: _RemoteReadHandle) -> PhysicalProgress:
        if not handle.future.done():
            return PhysicalProgress(PhysicalState.INFLIGHT)
        if handle.future.cancelled():
            return PhysicalProgress(
                PhysicalState.CANCELLED, FenceKind.NOT_POSTED
            )
        try:
            handle.receipt = handle.future.result()
        except UnfencedRemoteRead:
            # Preserve the lane and lease: the queue treats this as fatal and
            # never fabricates a drained proof.
            raise
        except DrainedRemoteRead as error:
            handle.receipt = error.receipt
            state = (
                PhysicalState.CANCELLED
                if handle.cancel.is_set()
                else PhysicalState.FAILED
            )
            fence = (
                FenceKind.CANCEL_DRAINED
                if handle.cancel.is_set()
                else FenceKind.ERROR_DRAINED
            )
            return PhysicalProgress(
                state,
                fence,
                str(error),
                result={"read_receipt": error.receipt.to_dict()},
            )
        except BaseException as error:
            state = (
                PhysicalState.CANCELLED
                if handle.cancel.is_set()
                else PhysicalState.FAILED
            )
            fence = (
                FenceKind.CANCEL_DRAINED
                if handle.cancel.is_set()
                else FenceKind.ERROR_DRAINED
            )
            return PhysicalProgress(state, fence, str(error))
        return PhysicalProgress(
            PhysicalState.SUCCEEDED,
            FenceKind.DMA_COMPLETE,
            result={"read_receipt": handle.receipt.to_dict()},
        )

    def request_cancel(
        self, handle: _RemoteReadHandle, notify: Callable[[], None]
    ) -> None:
        handle.cancel.set()
        if handle.future.cancel():
            notify()

    @staticmethod
    def receipt(handle: _RemoteReadHandle) -> Any:
        return handle.receipt

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)
