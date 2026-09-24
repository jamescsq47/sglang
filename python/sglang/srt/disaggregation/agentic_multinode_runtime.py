"""Event-driven TP link control runtime for multi-node agentic PD.

This module owns control flow only.  Physical NIXL and Host-copy behavior is
injected through four path handlers and four transfer executors.  It never
reads a marker, ledger, directory, or shared filesystem path.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
    GroupCommand,
    GroupDisconnectedError,
    LinkCapacityEdge,
    LinkDisconnected,
    LinkFailure,
    LinkIntent,
    LinkLifecycleCoordinator,
    LinkParticipant,
    LinkReadiness,
    LinkRankAck,
    Owner,
    ReadinessPhase,
    TCPRankAgent,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    GroupTransferPlan,
    LocalAttemptFailure,
    RankLocalCommandExecutor,
    RankPathHandler,
    RankZeroLinkOrchestrator,
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    AgenticTransferQueues,
    TransferPath,
)


class RuntimeState(str, Enum):
    NEW = "new"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class RuntimeCloseBlocked(RuntimeError):
    """The runtime retained ownership because a physical fence is incomplete."""


@dataclass(frozen=True, slots=True)
class RuntimeTerminal:
    key: GenerationKey
    attempt: int
    committed: bool
    reason: str = ""
    results: Optional[Mapping[LinkParticipant, Mapping[str, Any]]] = None


@dataclass(frozen=True, slots=True)
class EndpointActivationTicket:
    """Target-TP activation carried by SGLang's native scheduler broadcast."""

    key: GenerationKey
    attempt: int
    lease_id: str
    target_role: str
    target_generation: int

    @property
    def target_key(self) -> GenerationKey:
        return GenerationKey(
            self.key.run_id, self.key.request_id, self.target_generation
        )


@dataclass(slots=True)
class _LocalSubmission:
    plan: GroupTransferPlan
    done: threading.Event
    attempt: Optional[int] = None
    error: Optional[BaseException] = None


_STOP = object()
_PLAN_INTENT = "group_transfer_v1"


def _owner_role(owner: Owner) -> Optional[str]:
    if owner in {Owner.P_GPU, Owner.P_HOST, Owner.PREFILL_READY}:
        return "prefill"
    if owner in {Owner.D_GPU, Owner.D_HOST, Owner.DECODE_READY}:
        return "decode"
    return None


def _scheduler_target(owner: Owner) -> bool:
    return owner in {
        Owner.P_GPU,
        Owner.PREFILL_READY,
        Owner.D_GPU,
        Owner.DECODE_READY,
    }


def _target_key(plan: GroupTransferPlan) -> GenerationKey:
    generation = int(plan.payload.get("target_generation", plan.key.generation))
    return GenerationKey(plan.key.run_id, plan.key.request_id, generation)


def _readiness_requirements(
    plan: GroupTransferPlan, participants: Sequence[LinkParticipant]
) -> frozenset[tuple[LinkParticipant, GenerationKey, ReadinessPhase]]:
    required: set[tuple[LinkParticipant, GenerationKey, ReadinessPhase]] = set()
    source_role = _owner_role(plan.source_owner)
    target_role = _owner_role(plan.target_owner)
    if plan.operation in {TransferOperation.DIRECT, TransferOperation.HOST_STORE}:
        if source_role is not None:
            required.update(
                (participant, plan.key, ReadinessPhase.SOURCE_READY)
                for participant in participants
                if participant.role == source_role
            )
    if plan.operation in {TransferOperation.DIRECT, TransferOperation.HOST_RESTORE}:
        if target_role is not None:
            key = _target_key(plan)
            required.update(
                (participant, key, ReadinessPhase.REGISTERED)
                for participant in participants
                if participant.role == target_role
            )
    return frozenset(required)


def _remaining(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _plan_payload(plan: GroupTransferPlan) -> dict:
    return {
        "path": plan.path.value,
        "operation": plan.operation.value,
        "source_owner": plan.source_owner.value,
        "target_owner": plan.target_owner.value,
        "lease_id": plan.lease_id,
        "transfer": dict(plan.payload),
    }


def _plan_from_intent(intent: LinkIntent) -> GroupTransferPlan:
    if intent.kind != _PLAN_INTENT:
        raise ValueError(f"unsupported link intent kind {intent.kind!r}")
    value = intent.payload
    return GroupTransferPlan(
        key=intent.key,
        path=TransferPath(str(value["path"])),
        operation=TransferOperation(str(value["operation"])),
        source_owner=Owner(str(value["source_owner"])),
        target_owner=Owner(str(value["target_owner"])),
        lease_id=str(value["lease_id"]),
        payload=value.get("transfer") or {},
    )


class AgenticMultiNodeRuntime:
    """One rank's control service for a source-TP + target-TP link.

    The fixed link coordinator is normally P rank zero.  Only it constructs a
    lifecycle coordinator and publishes commands.  A non-coordinator endpoint
    rank zero (normally D rank zero) can only submit an intent.  Followers only
    execute commands and report physical fences.
    """

    def __init__(
        self,
        *,
        address: tuple[str, int],
        run_id: str,
        link_id: str,
        token: str,
        participant: LinkParticipant,
        endpoint_tp_size: int,
        participants: Sequence[LinkParticipant],
        coordinator_participant: LinkParticipant,
        queues: AgenticTransferQueues,
        handlers: Mapping[object, RankPathHandler],
        decide_intent: Optional[
            Callable[[LinkIntent, GroupTransferPlan], Optional[GroupTransferPlan]]
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
        on_capacity_edge: Optional[Callable[[LinkCapacityEdge], None]] = None,
        retain_terminals: bool = True,
    ) -> None:
        members = tuple(participants)
        if participant not in members or coordinator_participant not in members:
            raise ValueError("local and coordinator participants must belong to link")
        if coordinator_participant.rank != 0:
            raise ValueError("fixed link coordinator must be endpoint rank zero")
        if int(endpoint_tp_size) < 1:
            raise ValueError("endpoint_tp_size must be positive")

        self.participant = participant
        self.coordinator_participant = coordinator_participant
        self._participants = members
        self._queues = queues
        self._state = RuntimeState.NEW
        self._accepting = False
        self._state_condition = threading.Condition()
        self._fatal: Optional[BaseException] = None
        self._stop = threading.Event()
        self._command_queue: queue.Queue[object] = queue.Queue()
        self._control_queue: queue.Queue[object] = queue.Queue()
        self._proposal_sequence = itertools.count(1)
        self._terminals: queue.Queue[RuntimeTerminal] = queue.Queue()
        self._retain_terminals = bool(retain_terminals)
        self._activation_tickets: queue.Queue[EndpointActivationTicket] = queue.Queue()
        self._readiness: set[
            tuple[LinkParticipant, GenerationKey, ReadinessPhase]
        ] = set()
        self._pending_plans: dict[GenerationKey, GroupTransferPlan] = {}
        self._decide_intent_callback = decide_intent
        self._on_committed_callback = on_committed
        self._on_committed_results_callback = on_committed_results
        self._on_aborted_callback = on_aborted
        self._on_capacity_edge_callback = on_capacity_edge

        self.agent = TCPRankAgent(
            address,
            run_id=run_id,
            group_id=link_id,
            token=token,
            rank=participant.rank,
            tp_size=int(endpoint_tp_size),
            endpoint_group=participant.endpoint_group,
            endpoint_role=participant.role,
        )
        self.executor = RankLocalCommandExecutor(
            participant=participant,
            queues=queues,
            handlers=handlers,
            emit_ack=self._emit_ack,
            report_failure=self._report_failure,
        )

        self.coordinator: Optional[LinkLifecycleCoordinator] = None
        self.orchestrator: Optional[RankZeroLinkOrchestrator] = None
        if participant == coordinator_participant:
            self.coordinator = LinkLifecycleCoordinator(run_id, link_id, members)
            self.orchestrator = RankZeroLinkOrchestrator(
                self.coordinator,
                broadcast=self.agent.publish,
                decide_intent=self._decide_intent,
                on_committed_results=self._on_committed_with_results,
                on_aborted=self._on_aborted,
            )

        label = f"{participant.endpoint_group}-r{participant.rank}"
        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            name=f"agentic-control-recv-{label}",
            daemon=True,
        )
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name=f"agentic-control-owner-{label}",
            daemon=True,
        )
        self._command_thread = threading.Thread(
            target=self._command_loop,
            name=f"agentic-control-command-{label}",
            daemon=True,
        )

    @property
    def state(self) -> RuntimeState:
        with self._state_condition:
            return self._state

    @property
    def fatal_error(self) -> Optional[BaseException]:
        with self._state_condition:
            return self._fatal

    def _record_fatal(self, error: BaseException) -> None:
        with self._state_condition:
            if self._fatal is None:
                self._fatal = error
            self._accepting = False
            if self._state not in {RuntimeState.CLOSING, RuntimeState.CLOSED}:
                self._state = RuntimeState.FAILED
            self._state_condition.notify_all()

    def check_health(self) -> None:
        with self._state_condition:
            if self._fatal is not None:
                raise RuntimeError("agentic multi-node runtime failed") from self._fatal
        self._queues.check_health()

    def start(self) -> None:
        with self._state_condition:
            if self._state is not RuntimeState.NEW:
                raise RuntimeError(f"runtime cannot start from {self._state.value}")
            self._state = RuntimeState.RUNNING
            self._accepting = True
        self._command_thread.start()
        self._control_thread.start()
        self._receive_thread.start()

    def _emit_ack(self, value: LinkRankAck) -> None:
        if value.participant != self.participant:
            raise ValueError("executor emitted an ACK for another participant")
        self.agent.send_ack(value.ack)

    def _report_failure(self, failure: LocalAttemptFailure) -> None:
        if failure.participant != self.participant:
            raise ValueError("executor reported failure for another participant")
        self.agent.report_failure(
            failure.key,
            failure.attempt,
            failure.lease_id,
            failure.detail,
        )

    def _receive_loop(self) -> None:
        try:
            while not self._stop.is_set():
                event = self.agent.receive_event()
                if isinstance(event, GroupCommand):
                    self._command_queue.put(event)
                elif self.orchestrator is not None and isinstance(
                    event,
                    (
                        LinkRankAck,
                        LinkFailure,
                        LinkIntent,
                        LinkCapacityEdge,
                        LinkReadiness,
                        LinkDisconnected,
                    ),
                ):
                    self._control_queue.put(event)
                else:
                    raise RuntimeError(
                        f"unexpected control event for {self.participant}: {event!r}"
                    )
        except (EOFError, OSError) as error:
            if not self._stop.is_set():
                self._record_fatal(error)
        except BaseException as error:
            self._record_fatal(error)

    def _command_loop(self) -> None:
        while True:
            value = self._command_queue.get()
            if value is _STOP:
                return
            try:
                self.executor.handle(value)
                if (
                    value.kind is CommandKind.ISSUE_ACTIVATION_TICKET
                    and self.participant.rank == 0
                    and _scheduler_target(value.target_owner)
                    and self.participant.role == _owner_role(value.target_owner)
                ):
                    self._activation_tickets.put_nowait(
                        EndpointActivationTicket(
                            value.key,
                            value.attempt,
                            value.lease_id,
                            self.participant.role,
                            int(
                                value.payload.get("transfer", {}).get(
                                    "target_generation", value.key.generation
                                )
                            ),
                        )
                    )
            except BaseException as error:
                # The executor itself reports ordinary PREPARE/START failures.
                # Reaching here means lifecycle state is uncertain; retain all
                # ownership and fail the runtime rather than fabricating a fence.
                self._record_fatal(error)

    def _decide_intent(self, intent: LinkIntent) -> Optional[GroupTransferPlan]:
        candidate = _plan_from_intent(intent)
        if self._decide_intent_callback is None:
            return candidate
        return self._decide_intent_callback(intent, candidate)

    def _on_committed_with_results(
        self,
        plan: GroupTransferPlan,
        attempt: int,
        results: Mapping[LinkParticipant, Mapping[str, Any]],
    ) -> None:
        if self._retain_terminals:
            self._terminals.put(
                RuntimeTerminal(plan.key, attempt, True, results=results)
            )
        if self._on_committed_callback is not None:
            self._on_committed_callback(plan, attempt)
        if self._on_committed_results_callback is not None:
            self._on_committed_results_callback(plan, attempt, results)
        self._retire_readiness(plan)

    def _on_aborted(
        self, plan: GroupTransferPlan, attempt: int, reason: str
    ) -> None:
        if self._retain_terminals:
            self._terminals.put(RuntimeTerminal(plan.key, attempt, False, reason))
        if self._on_aborted_callback is not None:
            self._on_aborted_callback(plan, attempt, reason)

    def _control_loop(self) -> None:
        while True:
            value = self._control_queue.get()
            if value is _STOP:
                return
            try:
                orchestrator = self.orchestrator
                if orchestrator is None:
                    raise RuntimeError("non-coordinator received an owner event")
                if isinstance(value, _LocalSubmission):
                    value.attempt = self._offer_plan(value.plan)
                    value.done.set()
                elif isinstance(value, LinkRankAck):
                    orchestrator.on_ack(value)
                elif isinstance(value, LinkFailure):
                    orchestrator.on_link_failure(value)
                elif isinstance(value, LinkIntent):
                    with self._state_condition:
                        accepting = self._accepting
                    if accepting:
                        plan = self._decide_intent(value)
                        if plan is not None:
                            if plan.key != value.key:
                                raise ValueError(
                                    "intent decision changed request-generation key"
                                )
                            self._offer_plan(plan)
                elif isinstance(value, LinkReadiness):
                    self._accept_readiness(value)
                elif isinstance(value, LinkCapacityEdge):
                    callback = self._on_capacity_edge_callback
                    if callback is not None:
                        callback(value)
                elif isinstance(value, LinkDisconnected):
                    assert self.coordinator is not None
                    self.coordinator.apply_event(value)
                    raise GroupDisconnectedError(
                        f"link participant disconnected: {value.participant}"
                    )
                else:
                    raise RuntimeError(f"unsupported owner event {value!r}")
            except BaseException as error:
                if isinstance(value, _LocalSubmission):
                    value.error = error
                    value.done.set()
                self._record_fatal(error)

    def _offer_plan(self, plan: GroupTransferPlan) -> Optional[int]:
        orchestrator = self.orchestrator
        if orchestrator is None:
            raise RuntimeError("only the link coordinator may offer a plan")
        requirements = _readiness_requirements(plan, self._participants)
        if requirements.issubset(self._readiness):
            return orchestrator.begin(plan)
        current = self._pending_plans.get(plan.key)
        if current is not None and current != plan:
            raise RuntimeError("request-generation already has another pending plan")
        self._pending_plans[plan.key] = plan
        return None

    def _accept_readiness(self, value: LinkReadiness) -> None:
        if value.participant not in self._participants:
            raise ValueError("readiness participant is outside the link")
        identity = (value.participant, value.key, value.phase)
        if identity in self._readiness:
            return
        self._readiness.add(identity)
        for key, plan in tuple(self._pending_plans.items()):
            requirements = _readiness_requirements(plan, self._participants)
            if not requirements.issubset(self._readiness):
                continue
            del self._pending_plans[key]
            assert self.orchestrator is not None
            self.orchestrator.begin(plan)

    def _retire_readiness(self, plan: GroupTransferPlan) -> None:
        for item in _readiness_requirements(plan, self._participants):
            self._readiness.discard(item)

    def submit(
        self,
        plan: GroupTransferPlan,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[int]:
        """Submit locally at P0, or send a proposal from another endpoint's r0."""

        with self._state_condition:
            if self._state is not RuntimeState.RUNNING or not self._accepting:
                raise RuntimeError("runtime is not accepting link attempts")
            if self._fatal is not None:
                raise RuntimeError("runtime failed") from self._fatal

        if self.orchestrator is not None:
            submission = _LocalSubmission(plan, threading.Event())
            self._control_queue.put(submission)
            if not submission.done.wait(timeout):
                raise TimeoutError("coordinator did not accept local plan")
            if submission.error is not None:
                raise RuntimeError("coordinator rejected local plan") from submission.error
            return submission.attempt

        if self.participant.rank != 0:
            raise RuntimeError("only endpoint rank zero may propose a link attempt")
        self.agent.propose_intent(
            plan.key,
            next(self._proposal_sequence),
            _PLAN_INTENT,
            _plan_payload(plan),
        )
        return None

    def notify_capacity_available(self, available_tokens: int) -> int:
        """Emit one ephemeral D-capacity wake-up from Decode rank zero."""

        with self._state_condition:
            if self._state is not RuntimeState.RUNNING or not self._accepting:
                raise RuntimeError("runtime is not accepting capacity edges")
            if self._fatal is not None:
                raise RuntimeError("runtime failed") from self._fatal
        if self.participant.role != "decode" or self.participant.rank != 0:
            raise RuntimeError("only Decode endpoint rank zero reports capacity")
        return self.agent.send_capacity_edge(int(available_tokens))

    def report_registered(self, key: GenerationKey) -> None:
        self.agent.report_readiness(key, ReadinessPhase.REGISTERED)

    def report_source_ready(self, key: GenerationKey) -> None:
        self.agent.report_readiness(key, ReadinessPhase.SOURCE_READY)

    def take_activation_ticket(
        self, timeout: Optional[float] = None
    ) -> EndpointActivationTicket:
        """Target endpoint rank zero consumes this before native TP broadcast."""

        if self.participant.rank != 0:
            raise RuntimeError("only endpoint rank zero owns activation tickets")
        return self._activation_tickets.get(timeout=timeout)

    def activate_staged(self, ticket: EndpointActivationTicket) -> None:
        """Publish a ticket already broadcast by this endpoint's TP0 scheduler."""

        if ticket.target_role != self.participant.role:
            raise ValueError("activation ticket belongs to another endpoint role")
        self.executor.activate_staged(
            ticket.key, ticket.attempt, ticket.lease_id
        )

    def confirm_scheduler_adopted(self, ticket: EndpointActivationTicket) -> None:
        """Report success only after this rank inserted the request for compute."""

        if ticket.target_role != self.participant.role:
            raise ValueError("activation ticket belongs to another endpoint role")
        self.executor.confirm_scheduler_adopted(
            ticket.key, ticket.attempt, ticket.lease_id
        )

    def take_terminal(self, timeout: Optional[float] = None) -> RuntimeTerminal:
        return self._terminals.get(timeout=timeout)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        if self.orchestrator is not None and not self.orchestrator.wait_idle(
            _remaining(deadline)
        ):
            return False
        return self.executor.wait_idle(_remaining(deadline))

    def close(self, *, timeout: float = 5.0) -> None:
        """Close only after every attempt has a real terminal physical fence.

        If the deadline expires, the runtime and its memory remain live and the
        method raises.  It never closes an active queue or releases an unfenced
        lease merely to make shutdown finish.
        """

        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._state_condition:
            if self._state is RuntimeState.CLOSED:
                return
            if self._state is RuntimeState.NEW:
                self._accepting = False
            else:
                self._state = RuntimeState.CLOSING
                self._accepting = False

        if self.orchestrator is not None:
            self.orchestrator.cancel_active("runtime_shutdown")
        if not self.wait_idle(_remaining(deadline)):
            raise RuntimeCloseBlocked(
                "active link attempt has not reached RELEASED or FAILED_DRAINED"
            )

        try:
            self._queues.close(timeout=_remaining(deadline) or 0.0)
        except BaseException as error:
            raise RuntimeCloseBlocked(
                "physical transfer queues are not safely idle"
            ) from error

        self._stop.set()
        self.agent.close()
        self._command_queue.put(_STOP)
        self._control_queue.put(_STOP)
        for thread in (
            self._receive_thread,
            self._command_thread,
            self._control_thread,
        ):
            if thread.ident is not None:
                thread.join(timeout=_remaining(deadline))
        if any(
            thread.ident is not None and thread.is_alive()
            for thread in (
                self._receive_thread,
                self._command_thread,
                self._control_thread,
            )
        ):
            raise RuntimeCloseBlocked("control threads did not stop")
        with self._state_condition:
            self._state = RuntimeState.CLOSED
            self._state_condition.notify_all()


__all__ = [
    "AgenticMultiNodeRuntime",
    "EndpointActivationTicket",
    "RuntimeCloseBlocked",
    "RuntimeState",
    "RuntimeTerminal",
]
