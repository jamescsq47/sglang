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
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

from sglang.srt.disaggregation.agentic_group_protocol import (
    AttemptOutcome,
    CommandKind,
    GenerationKey,
    GenerationTerminal,
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
    HOST_EVICT = "host_evict"


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
    source_group: str = ""
    target_group: str = ""
    admission_deadline: float = 0.0

    def __post_init__(self) -> None:
        if not self.lease_id:
            raise ValueError("lease_id must be non-empty")
        if self.source_group and not self.source_group.strip():
            raise ValueError("source_group must be a non-empty identity")
        if self.target_group and not self.target_group.strip():
            raise ValueError("target_group must be a non-empty identity")
        value = copy.deepcopy(dict(self.payload))
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "payload", MappingProxyType(value))
        deadline = float(self.admission_deadline)
        if deadline < 0:
            raise ValueError("admission deadline must be non-negative")
        object.__setattr__(self, "admission_deadline", deadline)

    def command_payload(self) -> Mapping[str, Any]:
        return {
            "agentic_data_plane": {
                "version": 1,
                "path": self.path.value,
                "operation": self.operation.value,
                "source_group": self.source_group,
                "target_group": self.target_group,
            },
            "transfer": copy.deepcopy(dict(self.payload)),
        }


def plan_endpoint_groups(
    plan: GroupTransferPlan, participants: Sequence[LinkParticipant]
) -> tuple[str, ...]:
    """Resolve the exact logical endpoint groups for one physical attempt."""

    available = tuple(
        dict.fromkeys(participant.endpoint_group for participant in participants)
    )
    if plan.operation in {
        TransferOperation.HOST_STORE,
        TransferOperation.HOST_EVICT,
    }:
        requested = (plan.source_group,) if plan.source_group else ()
    else:
        requested = tuple(
            value for value in (plan.source_group, plan.target_group) if value
        )
    if not requested:
        return available
    if len(set(requested)) != len(requested):
        raise ValueError("source and target endpoint groups must differ")
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"plan selects unknown endpoint groups: {sorted(unknown)}")
    return requested


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
    source_released: bool = False


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
            log = logger.debug if isinstance(error, MemoryError) else logger.exception
            log(
                "Agentic V2 rank-local PREPARE failed: participant=%s "
                "key=%s path=%s operation=%s error=%s",
                self.participant,
                command.key.snapshot_id,
                path.value,
                operation.value,
                error,
            )
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
                channel=local.operation.value,
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
                lambda identity=identity: self._submitted(identity),
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

        # DMA_SUBMITTED is emitted by _submitted only at the executor's
        # physical-start edge.  For Direct this means a real sender post or
        # receiver-ready event, not Future creation or queue admission.

    def _submitted(self, identity: tuple[GenerationKey, int]) -> None:
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or local.start_command is None:
                raise RuntimeError("physical start belongs to an unknown attempt")
            if local.submit_ack_sent:
                return
            local.submit_ack_sent = True
            command = local.start_command
        self._ack(command, RankPhase.DMA_SUBMITTED)
        pending = False
        with self._lock:
            local = self._attempts.get(identity)
            if local is None:
                raise RuntimeError("started attempt disappeared")
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
                if completion.fence is FenceKind.NOT_POSTED:
                    # A Direct attempt can fail or be cancelled before the
                    # first real NIXL post.  Such an attempt intentionally
                    # has no DMA_SUBMITTED ACK; report the drained failure so
                    # rank zero can cancel the group instead of waiting for a
                    # start edge that can never exist.
                    local.completion_pending = False
                else:
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
        retire = getattr(local.handler, "retire", None)
        if callable(retire):
            retire(command)
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
        # A Host-store RELEASE command exists only after rank zero observed a
        # durable DMA fence from every source shard.  There is no target GPU
        # binding to wait for, so retaining the completed request in source
        # HBM through the later scheduler activation protocol is unnecessary
        # and directly reduces useful Decode/Prefill concurrency.
        if local.operation is TransferOperation.HOST_STORE:
            source_role = self._owner_role(command.source_owner)
            if self.participant.role in {source_role, "source"}:
                local.handler.commit(command, local.prepared, local.completion)
                with self._lock:
                    local.source_released = True
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
        # Rank zero emits HANDOFF only after every target shard is bound.  The
        # source can therefore release here; scheduler publication remains a
        # later, target-only transaction.  Keeping source pages until ACTIVATE
        # made source HBM residency depend on unrelated scheduler control RTTs.
        source_role = self._owner_role(command.source_owner)
        if (
            self.participant.role in {source_role, "source"}
            and not local.source_released
        ):
            local.handler.commit(command, local.prepared, local.completion)
            with self._lock:
                local.source_released = True
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
        # Source pages were released by HANDOFF after the all-target bind
        # barrier.  Publish each target's rank-local ready lease here.  Rank
        # zero waits for every STAGED ACK before issuing the one scheduler
        # ticket, so no scheduler rank can consume a partially published TP
        # group.
        source_role = self._owner_role(command.source_owner)
        if (
            self.participant.role in {source_role, "source"}
            and not local.source_released
        ):
            local.handler.commit(command, local.prepared, local.completion)
        with self._lock:
            if self.participant.role in {source_role, "source"}:
                local.source_released = True
            local.activation_staged = True
        target_role = self._owner_role(command.target_owner)
        if self._scheduler_target(command.target_owner) and self.participant.role in {
            target_role,
            "target",
        }:
            local.handler.commit(command, local.prepared, local.completion)
            with self._lock:
                local.activation_published = True
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
            if local is None or not local.activation_staged:
                raise RuntimeError("activation ticket arrived before ready barrier")
            if local.activation_ticket_command is not None:
                raise RuntimeError("duplicate activation ticket command")
            target_role = self._owner_role(command.target_owner)
            if (
                self._scheduler_target(command.target_owner)
                and self.participant.role in {target_role, "target"}
                and not local.activation_published
            ):
                raise RuntimeError("activation ticket arrived before local publish")
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
        """Validate TP0's ticket against an already-published ready lease.

        Physical completion publishes the local lease before rank zero emits
        this ticket.  The scheduler path therefore performs no memory or
        network work; it only validates ordering before queue adoption.
        """

        identity = (key, int(attempt))
        with self._lock:
            local = self._attempts.get(identity)
            if local is None or not local.activation_published:
                raise RuntimeError("scheduler activation has no staged local attempt")
            if local.prepare_command.lease_id != str(lease_id):
                raise RuntimeError("scheduler activation lease does not match")

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
            if local is None or not (
                local.activation_staged
                or (
                    local.operation is TransferOperation.HOST_STORE
                    and local.handoff_prepared
                )
            ):
                raise RuntimeError("FINALIZE arrived before local staging")
        # The transfer transaction is already complete.  FINALIZE only retires
        # short-lived attempt state; ready leases remain owned by the memory
        # authority until the scheduler consumes them.
        retire = getattr(local.handler, "retire", None)
        if callable(retire):
            retire(command)
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
    dma_slot_held: bool = False
    started_at: float = 0.0
    prepared_at: float = 0.0
    dma_submitted_at: float = 0.0
    dma_done_at: float = 0.0
    bound_at: float = 0.0
    released_at: float = 0.0

    def __post_init__(self) -> None:
        if self.started_at <= 0:
            self.started_at = time.monotonic()
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
        path_lanes: Optional[Mapping[TransferPath, int]] = None,
        operation_lanes: Optional[
            Mapping[tuple[TransferPath, str], int]
        ] = None,
        shared_network_lanes: int = 0,
        shared_network_lanes_by_direction: Optional[Mapping[str, int]] = None,
        shared_network_direct_reserve_by_direction: Optional[
            Mapping[str, int]
        ] = None,
        shared_network_host_reserve_by_direction: Optional[
            Mapping[str, int]
        ] = None,
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
        on_materialized: Optional[Callable[[GroupTransferPlan, int], None]] = None,
        decide_intent: Optional[
            Callable[[LinkIntent], Optional[GroupTransferPlan]]
        ] = None,
    ) -> None:
        self._coordinator = coordinator
        self._broadcast = broadcast
        self._on_committed = on_committed
        self._on_committed_results = on_committed_results
        self._on_aborted = on_aborted
        self._on_materialized = on_materialized
        self._decide_intent = decide_intent
        self._active: dict[tuple[GenerationKey, int], _LinkAttempt] = {}
        self._path_lanes = {
            path: max(1, int((path_lanes or {}).get(path, 1)))
            for path in TransferPath
        }
        self._operation_lanes = {
            (path, str(operation)): max(1, int(value))
            for (path, operation), value in (operation_lanes or {}).items()
        }
        # A host may expose fewer physical network rails than logical P/D
        # endpoint groups.  Per-endpoint queue limits alone then admit dozens
        # of simultaneous large KV transfers onto one NIC, increasing every
        # transfer's latency and causing Direct admission cascades.  The rail
        # is full duplex, so bound D->P and P->D independently.  A single
        # combined counter incorrectly lets a D->P recovery burst starve P->D
        # delivery even though RX and TX can progress concurrently.  Source-
        # local Host stores do not use the rail and remain independent.
        default_network_lanes = max(0, int(shared_network_lanes))
        directional_network_lanes = shared_network_lanes_by_direction or {}
        self._shared_network_lanes = {
            direction: max(
                0,
                int(directional_network_lanes.get(direction, default_network_lanes)),
            )
            for direction in ("d2p", "p2d")
        }
        direct_reserves = shared_network_direct_reserve_by_direction or {}
        self._shared_network_direct_reserve = {
            direction: min(
                self._shared_network_lanes[direction],
                max(0, int(direct_reserves.get(direction, 0))),
            )
            for direction in ("d2p", "p2d")
        }
        host_reserves = shared_network_host_reserve_by_direction or {}
        self._shared_network_host_reserve = {
            direction: min(
                self._shared_network_lanes[direction],
                max(0, int(host_reserves.get(direction, 0))),
            )
            for direction in ("d2p", "p2d")
        }
        self._shared_network_active = {"d2p": 0, "p2d": 0}
        self._shared_network_direct_active = {"d2p": 0, "p2d": 0}
        self._shared_drain_cursor = 0
        admission_keys = set(TransferPath) | set(self._operation_lanes)
        # Physical queues are rank-local.  A single global counter here made
        # four disjoint P<->D pairs share the same four lanes, even though the
        # pairs use different GPUs, NIXL endpoints and queue workers.  Account
        # admission per participating endpoint group instead: each endpoint
        # may use at most its local lane budget, while disjoint pairs progress
        # independently.
        self._path_active: dict[tuple[object, str], int] = {}
        self._path_pending = {key: deque() for key in admission_keys}
        self._queued_keys: set[GenerationKey] = set()
        # A lane completion can discover an expired queued Direct while the
        # orchestrator lock is held.  Defer its fallback callback rather than
        # re-entering policy/lifecycle code under that lock.
        self._deferred_rejections: deque[tuple[GroupTransferPlan, str]] = deque()
        self._rejection_callbacks_active = 0
        self._rejection_callback_error: Optional[BaseException] = None
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def progress_diagnostics(self) -> tuple[dict[str, int], tuple[str, ...]]:
        """Return a compact, read-only view of stalled coordinator work.

        This is deliberately sampled by the control thread rather than on
        every command/ACK.  It makes multi-endpoint head-of-line stalls
        observable without adding traffic to the data plane.
        """

        now = time.monotonic()
        with self._lock:
            counts: dict[str, int] = {}
            oldest: list[tuple[float, str]] = []
            for state in self._active.values():
                phase = state.phase.value
                counts[phase] = counts.get(phase, 0) + 1
                age = max(0.0, now - state.started_at)
                if age >= 1.0:
                    oldest.append(
                        (
                            age,
                            f"{state.plan.key.snapshot_id}@{phase}/"
                            f"{state.plan.source_group}->{state.plan.target_group}"
                            f"/{state.plan.path.value}",
                        )
                    )
            pending = sum(len(values) for values in self._path_pending.values())
            pending += len(self._deferred_rejections)
            if pending:
                counts["admission_pending"] = pending
            if self._rejection_callbacks_active:
                counts["rejection_callback"] = self._rejection_callbacks_active
            if self._rejection_callback_error is not None:
                counts["rejection_callback_failed"] = 1
            if any(self._shared_network_lanes.values()):
                d2p = self._shared_network_active["d2p"]
                p2d = self._shared_network_active["p2d"]
                counts["network_dma_active"] = d2p + p2d
                counts["network_d2p_active"] = d2p
                counts["network_p2d_active"] = p2d
                counts["network_d2p_direct_active"] = (
                    self._shared_network_direct_active["d2p"]
                )
                counts["network_p2d_direct_active"] = (
                    self._shared_network_direct_active["p2d"]
                )
        oldest.sort(reverse=True)
        return counts, tuple(
            f"{label}:{age:.1f}s" for age, label in oldest[:8]
        )

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        with self._changed:
            return self._changed.wait_for(
                lambda: not self._active
                and not any(self._path_pending.values())
                and not self._deferred_rejections
                and self._rejection_callbacks_active == 0
                and self._rejection_callback_error is None,
                # A rejected plan is not idle until policy has received the
                # fallback notification and transferred its ownership.
                # ``expire_admissions`` drains this queue on the control tick.
                timeout,
            )

    @staticmethod
    def _abortable(state: _LinkAttempt) -> bool:
        # RELEASE is issued only after every shard has a real DMA completion.
        # From that point onward some ranks may already have bound/published
        # the destination or released the source, so rollback would fabricate
        # a globally consistent owner.  Retain ownership and fail loudly.
        # START may already be physically posted on only a subset of TP ranks.
        # Withdrawing its peers would strand those shards forever.  Only an
        # attempt that has not crossed the all-rank PREPARED fence is safe to
        # cancel.
        return state.phase is CommandKind.PREPARE

    def _admission_key(self, plan: GroupTransferPlan):
        # Admission limits physical I/O execution only.  Target workset
        # ownership is enforced by AgenticMemoryAuthority and must not merge
        # independent Direct and Host queues into a shared FIFO.  PREPARE is
        # issued only after this operation-specific lane is available, so a
        # queued transfer cannot reserve target HBM while waiting for an I/O
        # worker.
        key = (plan.path, plan.operation.value)
        return key if key in self._operation_lanes else plan.path

    def _admission_limit(self, key) -> int:
        if isinstance(key, tuple):
            return self._operation_lanes[key]
        return self._path_lanes[key]

    def _admission_resources(
        self, plan: GroupTransferPlan
    ) -> tuple[tuple[object, str], ...]:
        admission = self._admission_key(plan)
        return tuple(
            (admission, endpoint_group)
            for endpoint_group in plan_endpoint_groups(
                plan, self._coordinator.participants
            )
        )

    def _dma_slot_available_locked(self, plan: GroupTransferPlan) -> bool:
        limit = self._admission_limit(self._admission_key(plan))
        endpoint_available = all(
            self._path_active.get(resource, 0) < limit
            for resource in self._admission_resources(plan)
        )
        if not endpoint_available:
            return False
        if not self._uses_shared_network(plan):
            return True
        direction = self._network_direction(plan)
        active = self._shared_network_active[direction]
        limit = self._shared_network_lanes[direction]
        if active >= limit:
            return False
        if plan.operation is TransferOperation.HOST_RESTORE:
            host_active = active - self._shared_network_direct_active[direction]
            return host_active < limit - self._shared_network_direct_reserve[direction]
        if (
            plan.operation is TransferOperation.DIRECT
            and self._has_pending_host_restore_locked(direction)
        ):
            # A continuous stream of fresh Direct attempts must not starve
            # snapshots whose source HBM was already released to Host.  This
            # reservation is demand-driven: without a Host waiter, Direct may
            # still consume the whole physical rail.
            direct_limit = limit - self._shared_network_host_reserve[direction]
            return self._shared_network_direct_active[direction] < direct_limit
        return True

    def _has_pending_host_restore_locked(self, direction: str) -> bool:
        return any(
            candidate.operation is TransferOperation.HOST_RESTORE
            and self._network_direction(candidate) == direction
            for pending in self._path_pending.values()
            for candidate in pending
        )

    def _uses_shared_network(self, plan: GroupTransferPlan) -> bool:
        if plan.operation not in {
            TransferOperation.DIRECT,
            TransferOperation.HOST_RESTORE,
        }:
            return False
        return self._shared_network_lanes[self._network_direction(plan)] > 0

    @staticmethod
    def _network_direction(plan: GroupTransferPlan) -> str:
        if plan.path in {TransferPath.D2P_DIRECT, TransferPath.D2P_HOST}:
            return "d2p"
        if plan.path in {TransferPath.P2D_DIRECT, TransferPath.P2D_HOST}:
            return "p2d"
        raise ValueError(f"unsupported network path {plan.path.value}")

    def _reserve_dma_slot_locked(self, plan: GroupTransferPlan) -> None:
        if not self._dma_slot_available_locked(plan):
            raise RuntimeError("group DMA endpoint admission is unavailable")
        for resource in self._admission_resources(plan):
            self._path_active[resource] = self._path_active.get(resource, 0) + 1
        if self._uses_shared_network(plan):
            direction = self._network_direction(plan)
            self._shared_network_active[direction] += 1
            if plan.operation is TransferOperation.DIRECT:
                self._shared_network_direct_active[direction] += 1

    def _free_dma_slot_locked(self, plan: GroupTransferPlan) -> None:
        for resource in self._admission_resources(plan):
            active = self._path_active.get(resource, 0) - 1
            if active < 0:
                raise RuntimeError("group DMA endpoint admission underflow")
            if active:
                self._path_active[resource] = active
            else:
                self._path_active.pop(resource, None)
        if self._uses_shared_network(plan):
            direction = self._network_direction(plan)
            self._shared_network_active[direction] -= 1
            if self._shared_network_active[direction] < 0:
                raise RuntimeError("shared network admission underflow")
            if plan.operation is TransferOperation.DIRECT:
                self._shared_network_direct_active[direction] -= 1
                if self._shared_network_direct_active[direction] < 0:
                    raise RuntimeError("shared network Direct admission underflow")

    def _drain_one_pending_locked(self, admission) -> bool:
        """Admit at most one feasible plan from one operation queue."""

        pending = self._path_pending[admission]
        if not pending:
            return False
        kept = deque()
        admitted = False
        now = time.monotonic()
        while pending:
            plan = pending.popleft()
            if plan.admission_deadline and plan.admission_deadline <= now:
                self._queued_keys.discard(plan.key)
                self._deferred_rejections.append(
                    (plan, "direct admission deadline expired before PREPARE")
                )
                continue
            if admitted or not self._dma_slot_available_locked(plan):
                kept.append(plan)
                continue
            self._queued_keys.remove(plan.key)
            self._reserve_dma_slot_locked(plan)
            try:
                self._begin_locked(plan, dma_slot_held=True)
            except BaseException:
                if not any(key == plan.key for key, _attempt in self._active):
                    self._free_dma_slot_locked(plan)
                kept.extend(pending)
                self._path_pending[admission] = kept
                raise
            admitted = True
        self._path_pending[admission] = kept
        return admitted

    def _drain_shared_pending_locked(self) -> None:
        """Fairly share one physical rail across independent path queues."""

        admissions = tuple(self._path_pending)
        if not admissions:
            return
        while True:
            progressed = False
            for offset in range(len(admissions)):
                index = (self._shared_drain_cursor + offset) % len(admissions)
                admission = admissions[index]
                if self._drain_one_pending_locked(admission):
                    self._shared_drain_cursor = (index + 1) % len(admissions)
                    progressed = True
                    break
            if not progressed:
                break

    def _drain_pending_locked(self, admission) -> None:
        """Start every feasible request without cross-endpoint head blocking."""

        pending = self._path_pending[admission]
        if not pending:
            return
        kept = deque()
        while pending:
            plan = pending.popleft()
            if (
                plan.admission_deadline
                and plan.admission_deadline <= time.monotonic()
            ):
                self._queued_keys.discard(plan.key)
                self._deferred_rejections.append(
                    (
                        plan,
                        "direct admission deadline expired before PREPARE",
                    )
                )
                continue
            if not self._dma_slot_available_locked(plan):
                kept.append(plan)
                continue
            self._queued_keys.remove(plan.key)
            self._reserve_dma_slot_locked(plan)
            try:
                self._begin_locked(plan, dma_slot_held=True)
            except BaseException:
                if not any(key == plan.key for key, _attempt in self._active):
                    self._free_dma_slot_locked(plan)
                kept.extend(pending)
                self._path_pending[admission] = kept
                raise
        self._path_pending[admission] = kept

    def _arm_rejection_callbacks_locked(
        self, rejected: list[tuple[GroupTransferPlan, str]]
    ) -> None:
        if self._on_aborted is not None:
            self._rejection_callbacks_active += len(rejected)

    def _dispatch_rejection_callbacks(
        self, rejected: list[tuple[GroupTransferPlan, str]]
    ) -> None:
        callback = self._on_aborted
        if callback is None:
            return
        first_error: Optional[BaseException] = None
        for plan, reason in rejected:
            error: Optional[BaseException] = None
            try:
                callback(plan, 0, reason)
            except BaseException as current_error:
                error = current_error
                if first_error is None:
                    first_error = current_error
            finally:
                with self._changed:
                    if error is not None and self._rejection_callback_error is None:
                        self._rejection_callback_error = error
                    self._rejection_callbacks_active -= 1
                    if self._rejection_callbacks_active < 0:
                        raise RuntimeError("rejection callback accounting underflow")
                    self._changed.notify_all()
        if first_error is not None:
            raise first_error

    def cancel_active(self, reason: str) -> int:
        """Cancel queued/PREPARE attempts; START must reach a real fence."""

        commands = []
        rejected: list[tuple[GroupTransferPlan, str]] = []
        with self._changed:
            rejected.extend(self._deferred_rejections)
            self._deferred_rejections.clear()
            for pending in self._path_pending.values():
                rejected.extend((plan, str(reason)) for plan in pending)
                pending.clear()
            self._queued_keys.clear()
            for (key, attempt_id), state in tuple(self._active.items()):
                if not self._abortable(state):
                    continue
                if self._coordinator.outcome(key, attempt_id) is not AttemptOutcome.ACTIVE:
                    continue
                command = self._coordinator.request_abort(key, attempt_id, reason)
                state.abort_reason = str(reason)
                state.phase = CommandKind.CANCEL
                commands.append(command)
            self._arm_rejection_callbacks_locked(rejected)
            self._changed.notify_all()
        self._dispatch_rejection_callbacks(rejected)
        for command in commands:
            self._broadcast(command)
        return len(commands) + len(rejected)

    def offer(self, plan: GroupTransferPlan) -> Optional[int]:
        """Admit one immutable attempt using a rank-zero-owned path lane.

        Every physical operation is lane-first.  In particular a Direct plan
        waiting for a lane owns neither a target workset nor a partially
        posted TP transfer, so its short admission deadline can safely route
        it to Host.  Once PREPARE reaches every rank, START is non-cancellable
        and drains to its real all-rank DMA fence.
        """

        expired = False
        with self._changed:
            if plan.key in self._queued_keys or any(
                key == plan.key for key, _attempt in self._active
            ):
                raise RuntimeError("request-generation already has a link attempt")
            if (
                plan.admission_deadline
                and plan.admission_deadline <= time.monotonic()
            ):
                expired = True
                if self._on_aborted is not None:
                    self._rejection_callbacks_active += 1
            else:
                admission = self._admission_key(plan)
                if not self._dma_slot_available_locked(plan):
                    self._path_pending[admission].append(plan)
                    self._queued_keys.add(plan.key)
                    self._changed.notify_all()
                    return None
                self._reserve_dma_slot_locked(plan)
                try:
                    return self._begin_locked(plan, dma_slot_held=True)
                except BaseException:
                    # begin_attempt installs the authoritative ownership record
                    # before broadcasting PREPARE.  If broadcast fails after
                    # that point, quarantine the lane and active attempt: a
                    # partial remote delivery cannot be safely rolled back or
                    # reused as though nothing happened.
                    if not any(key == plan.key for key, _attempt in self._active):
                        self._free_dma_slot_locked(plan)
                    raise
        if expired:
            self._dispatch_rejection_callbacks(
                [(plan, "direct admission deadline expired before PREPARE")]
            )
        return None

    def _begin_locked(
        self, plan: GroupTransferPlan, *, dma_slot_held: bool
    ) -> int:
        command = self._coordinator.begin_attempt(
            plan.key,
            source_owner=plan.source_owner,
            target_owner=plan.target_owner,
            lease_id=plan.lease_id,
            required_commit_phase=(
                RankPhase.BOUND
                if plan.operation is TransferOperation.HOST_STORE
                else RankPhase.RELEASED
            ),
            endpoint_groups=plan_endpoint_groups(
                plan, self._coordinator.participants
            ),
            payload=plan.command_payload(),
        )
        identity = (plan.key, command.attempt)
        if identity in self._active:
            raise RuntimeError("duplicate active link attempt")
        self._active[identity] = _LinkAttempt(
            plan, dma_slot_held=bool(dma_slot_held)
        )
        self._changed.notify_all()
        self._broadcast(command)
        return command.attempt

    def _release_dma_slot_locked(self, state: _LinkAttempt) -> None:
        if not state.dma_slot_held:
            return
        state.dma_slot_held = False
        admission = self._admission_key(state.plan)
        self._free_dma_slot_locked(state.plan)
        if self._uses_shared_network(state.plan):
            self._drain_shared_pending_locked()
        else:
            self._drain_pending_locked(admission)
        self._drain_prepared_direct_locked(admission)
        self._changed.notify_all()

    def _start_prepared_direct_locked(
        self,
        key: GenerationKey,
        attempt: int,
        state: _LinkAttempt,
    ) -> bool:
        """Start an admitted Direct copy once its physical lane is free."""

        if (
            state.plan.operation is not TransferOperation.DIRECT
            or state.phase is not CommandKind.PREPARE
            or state.prepared_at <= 0
        ):
            return False
        if not state.dma_slot_held:
            if not self._dma_slot_available_locked(state.plan):
                return False
            self._reserve_dma_slot_locked(state.plan)
            state.dma_slot_held = True
        command = self._coordinator.issue_start(
            key,
            int(attempt),
            payload=state.plan.command_payload(),
        )
        state.phase = CommandKind.START
        self._broadcast(command)
        return True

    def _drain_prepared_direct_locked(self, admission) -> None:
        for (key, attempt), state in tuple(self._active.items()):
            if self._admission_key(state.plan) != admission:
                continue
            self._start_prepared_direct_locked(key, attempt, state)

    def expire_admissions(self, now: Optional[float] = None) -> int:
        """Expire Direct plans only before the all-rank PREPARED fence.

        Lane-first admission means a lane-pending plan has not allocated a
        target workset.  An active PREPARE may also be cancelled before any
        START exists.  After all ranks are PREPARED, START may partially post
        across TP ranks and must drain to a real physical fence; cancelling at
        that point can strand a posted shard waiting for a peer that withdrew.
        """

        now = time.monotonic() if now is None else float(now)
        rejected: list[tuple[GroupTransferPlan, str]] = []
        commands = []
        with self._changed:
            rejected.extend(self._deferred_rejections)
            self._deferred_rejections.clear()
            for admission, pending in self._path_pending.items():
                kept = deque()
                while pending:
                    plan = pending.popleft()
                    if plan.admission_deadline and plan.admission_deadline <= now:
                        self._queued_keys.discard(plan.key)
                        rejected.append(
                            (
                                plan,
                                "direct admission deadline expired before PREPARE",
                            )
                        )
                    else:
                        kept.append(plan)
                self._path_pending[admission] = kept
            for (key, attempt), state in tuple(self._active.items()):
                if not (
                    state.phase is CommandKind.PREPARE
                    and state.prepared_at <= 0
                    and state.plan.admission_deadline
                    and state.plan.admission_deadline <= now
                ):
                    continue
                if self._coordinator.outcome(key, attempt) is not AttemptOutcome.ACTIVE:
                    continue
                command = self._coordinator.request_abort(
                    key, attempt, "direct admission deadline expired"
                )
                state.abort_reason = "direct admission deadline expired"
                state.phase = CommandKind.CANCEL
                commands.append(command)
            self._arm_rejection_callbacks_locked(rejected)
            self._changed.notify_all()
        for command in commands:
            self._broadcast(command)
        self._dispatch_rejection_callbacks(rejected)
        return len(rejected) + len(commands)

    @staticmethod
    def _log_slow_ownership(state: _LinkAttempt) -> None:
        end = state.bound_at if state.plan.operation is TransferOperation.HOST_STORE else state.released_at
        if end <= 0 or end - state.started_at < 1.0:
            return
        prepared = state.prepared_at or state.started_at
        dma_done = state.dma_done_at or prepared
        bound = state.bound_at or dma_done
        logger.warning(
            "Agentic V2 slow ownership handoff: key=%s path=%s operation=%s "
            "tokens=%d total_s=%.3f prepare_s=%.3f dma_s=%.3f "
            "bind_s=%.3f release_s=%.3f",
            state.plan.key.snapshot_id,
            state.plan.path.value,
            state.plan.operation.value,
            int(state.plan.payload.get("token_count", 0)),
            end - state.started_at,
            prepared - state.started_at,
            dma_done - prepared,
            bound - dma_done,
            max(0.0, end - bound),
        )

    def begin(self, plan: GroupTransferPlan) -> int:
        with self._changed:
            return self._begin_locked(plan, dma_slot_held=False)

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
        return self.offer(plan)

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
                state.prepared_at = time.monotonic()
                if state.plan.operation is TransferOperation.DIRECT:
                    self._start_prepared_direct_locked(
                        ack.key, ack.attempt, state
                    )
                else:
                    command = self._coordinator.issue_start(
                        ack.key,
                        ack.attempt,
                        payload=state.plan.command_payload(),
                    )
                    state.phase = CommandKind.START
                    self._broadcast(command)
                return
            if (
                state.phase is CommandKind.START
                and state.dma_submitted_at <= 0
                and self._coordinator.group_reached(
                    ack.key, ack.attempt, RankPhase.DMA_SUBMITTED
                )
            ):
                # Kept as transfer diagnostics.  With lane-first admission,
                # the short deadline ended at the all-rank PREPARED fence.
                state.dma_submitted_at = time.monotonic()
            if state.phase is CommandKind.START and self._coordinator.group_reached(
                ack.key, ack.attempt, RankPhase.DMA_DONE
            ):
                state.dma_done_at = time.monotonic()
                # Every target shard now owns both its physical workset and
                # the complete transferred data.  Unlike PREPARED, this edge
                # cannot normally fall back from Direct to Host, so it is the
                # causal point at which an external router may replace a
                # projected capacity charge with the allocator's real charge.
                if self._on_materialized is not None:
                    self._on_materialized(state.plan, ack.attempt)
                # The physical executor is reusable once every shard crossed
                # its real DMA fence.  Binding, source release and scheduler
                # publication retain their ownership leases, not the I/O
                # lane.  Holding the lane through those control barriers
                # serialized unrelated transfers and recreated scheduler
                # backpressure in the data plane.
                self._release_dma_slot_locked(state)
                participants = self._coordinator.participants_for(
                    ack.key, ack.attempt
                )
                missing = set(participants) - set(
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
                    for participant in participants
                ]
                # Later activation commands still need the immutable target
                # generation from the original plan.  The source and target
                # generations differ on D→P returns.
                state.handoff_payload = MappingProxyType(
                    {
                        "rank_results": rank_results,
                        "transfer": dict(state.plan.payload),
                    }
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
                state.bound_at = time.monotonic()
                if state.plan.operation is TransferOperation.HOST_STORE:
                    # Durable Host ownership and source-HBM release are both
                    # complete at this barrier; no scheduler activation
                    # follows a Store-only attempt.
                    self._release_dma_slot_locked(state)
                    self._log_slow_ownership(state)
                    self._coordinator.commit(ack.key, ack.attempt)
                    command = self._coordinator.issue_finalize(
                        ack.key,
                        ack.attempt,
                        payload=state.handoff_payload,
                    )
                    state.phase = CommandKind.FINALIZE
                    self._broadcast(command)
                    return
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
                state.released_at = time.monotonic()
                # Ownership has now left the source HBM and the target is
                # fully bound.  The next attempt may safely reserve memory;
                # later ready publication and scheduler adoption do not hold
                # a data-plane lane.
                self._release_dma_slot_locked(state)
                self._log_slow_ownership(state)
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
                # ACTIVATE publishes every rank-local ready lease.  Only after
                # all ranks acknowledge that edge does rank zero issue the
                # single immutable scheduler ticket.  The former ARM/PUBLISH
                # round trips duplicated this same barrier and allowed the
                # ownership tail to grow by seconds at TP=8.
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
                if state.plan.operation is TransferOperation.HOST_EVICT:
                    self._coordinator.mark_terminal(
                        ack.key, GenerationTerminal.EVICTED
                    )
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
                participants = self._coordinator.participants_for(
                    ack.key, ack.attempt
                )
                missing = set(participants) - set(
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
                    for participant in participants
                ]
                command = self._coordinator.issue_abort_finalize(
                    ack.key,
                    ack.attempt,
                    payload={"rank_results": rank_results},
                )
                state.phase = CommandKind.ABORT_FINALIZE
                self._broadcast(command)
                self._release_dma_slot_locked(state)
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
            if not self._abortable(state):
                raise RuntimeError(
                    reason
                    + "; failure followed the all-rank DMA fence, so ownership "
                    "was retained instead of attempting an unsafe rollback"
                )
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
    ready_event: Any = None


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
            if payload.ready_event is not None:
                payload.ready_event.synchronize()
            device_indices = payload.device_indices
            if callable(device_indices):
                device_indices = device_indices()
            return self._worker.load(
                payload.shard,
                attempt_id=attempt.attempt_id,
                device_indices=device_indices,
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
            # RemoteHostRankWorker converts every recoverable READ failure
            # into DrainedRemoteRead together with its authoritative receipt.
            # An arbitrary worker exception has no such proof and therefore
            # must never be advertised as *_DRAINED.
            raise UnfencedRemoteRead(
                "remote Host READ failed without a drain receipt"
            ) from error
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
