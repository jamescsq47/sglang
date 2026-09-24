"""Scheduler-facing composition root for the agentic multi-node V2 path.

This module owns no physical transfer implementation.  It composes the one
rank-local memory authority, the role-specific scheduler bridge, four bounded
transfer queues, and the TP link runtime.  A physical provider must supply
real Direct/Host executors and rank handlers; startup fails closed when that
provider is absent.

All scheduler calls are non-blocking.  Request arrivals and compute-complete
events enter one controller queue; the controller thread is the only caller
that creates link attempts.  Transport threads commit target binding/ready and
source release through provider handlers after the all-rank physical fence.
No filesystem path, directory scan, or scheduler transport poll is used.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from sglang.srt.disaggregation.agentic_decode_memory_bridge import (
    AgenticDMemorySchedulerBridge,
    DecodeCompleteItem,
)
from sglang.srt.disaggregation.agentic_group_protocol import (
    GenerationKey,
    LinkCapacityEdge,
    LinkIntent,
    LinkParticipant,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    GroupTransferPlan,
    RankPathHandler,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata
from sglang.srt.disaggregation.agentic_memory_authority import (
    AgenticMemoryAuthority,
)
from sglang.srt.disaggregation.agentic_memory_scheduler_bridge import (
    AgenticPMemorySchedulerBridge,
    PrefillCompleteItem,
)
from sglang.srt.disaggregation.agentic_multinode_runtime import (
    AgenticMultiNodeRuntime,
    EndpointActivationTicket,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    AgenticTransferQueues,
    TransferExecutor,
    TransferPath,
)


class CompositeRuntimeError(RuntimeError):
    pass


class RequestPhase(str, Enum):
    ARRIVED = "arrived"
    INIT_SUBMITTED = "init_submitted"
    READY = "ready"
    COMPUTE_COMPLETE = "compute_complete"
    TRANSFER_SUBMITTED = "transfer_submitted"
    HANDOFF_COMPLETE = "handoff_complete"


@dataclass(slots=True)
class RuntimeRequestRecord:
    key: GenerationKey
    req: Any
    parent_key: Optional[GenerationKey]
    phase: RequestPhase = RequestPhase.ARRIVED
    submitted_attempt: Optional[int] = None


@dataclass(frozen=True, slots=True)
class _CommittedEvent:
    plan: GroupTransferPlan
    attempt: int


@dataclass(frozen=True, slots=True)
class _CommittedResultsEvent:
    plan: GroupTransferPlan
    attempt: int
    results: Mapping[LinkParticipant, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class _AbortedEvent:
    plan: GroupTransferPlan
    attempt: int
    reason: str


@dataclass(frozen=True, slots=True)
class _CapacityEdgeEvent:
    edge: LinkCapacityEdge


class RequestGenerationRegistry:
    """Small in-memory registry; it never owns physical pages."""

    def __init__(self, run_id: str) -> None:
        self.run_id = str(run_id)
        self._records: dict[GenerationKey, RuntimeRequestRecord] = {}
        self._children: dict[GenerationKey, GenerationKey] = {}
        self._lock = threading.RLock()

    def register(self, req: Any) -> RuntimeRequestRecord:
        metadata = AgenticRequestMetadata.from_req(req)
        if metadata is None:
            raise ValueError("V2 request is missing agentic generation metadata")
        key = GenerationKey(self.run_id, metadata.request_id, metadata.generation)
        parent_key = (
            None
            if metadata.parent_generation is None
            else GenerationKey(
                self.run_id, metadata.request_id, metadata.parent_generation
            )
        )
        with self._lock:
            current = self._records.get(key)
            if current is not None:
                if current.req is not req or current.parent_key != parent_key:
                    raise RuntimeError("request-generation identity was reused")
                return current
            if parent_key is not None:
                old_child = self._children.get(parent_key)
                if old_child is not None and old_child != key:
                    raise RuntimeError("one parent generation has multiple live children")
                self._children[parent_key] = key
            record = RuntimeRequestRecord(key, req, parent_key)
            self._records[key] = record
            return record

    def get(self, key: GenerationKey) -> Optional[RuntimeRequestRecord]:
        with self._lock:
            return self._records.get(key)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._records

    def child_of(self, key: GenerationKey) -> Optional[RuntimeRequestRecord]:
        with self._lock:
            child = self._children.get(key)
            return None if child is None else self._records.get(child)

    def mark_submitted(self, key: GenerationKey, attempt: Optional[int]) -> None:
        with self._lock:
            record = self._records.get(key)
            if record is None:
                raise RuntimeError("transfer belongs to an unknown request-generation")
            if record.phase is RequestPhase.TRANSFER_SUBMITTED:
                if record.submitted_attempt != attempt:
                    raise RuntimeError("request-generation was submitted twice")
                return
            if record.phase is not RequestPhase.COMPUTE_COMPLETE:
                raise RuntimeError("transfer submitted before compute completion")
            record.phase = RequestPhase.TRANSFER_SUBMITTED
            record.submitted_attempt = attempt

    def mark_init_submitted(self, key: GenerationKey, attempt: Optional[int]) -> None:
        with self._lock:
            record = self._records.get(key)
            if record is None or record.phase is not RequestPhase.ARRIVED:
                raise RuntimeError("initial admission has an invalid request phase")
            record.phase = RequestPhase.INIT_SUBMITTED
            record.submitted_attempt = attempt

    def mark_init_ready(self, key: GenerationKey) -> None:
        with self._lock:
            record = self._records.get(key)
            if record is None or record.phase is not RequestPhase.INIT_SUBMITTED:
                raise RuntimeError("initial admission committed in an invalid phase")
            record.phase = RequestPhase.READY

    def mark_compute_complete(self, key: GenerationKey) -> RuntimeRequestRecord:
        with self._lock:
            record = self._records.get(key)
            if record is None:
                raise RuntimeError("completion belongs to an unknown request-generation")
            if record.phase in {RequestPhase.ARRIVED, RequestPhase.READY}:
                record.phase = RequestPhase.COMPUTE_COMPLETE
            elif record.phase is not RequestPhase.COMPUTE_COMPLETE:
                raise RuntimeError("request-generation completed in an invalid phase")
            return record

    def mark_handoff_complete(self, key: GenerationKey) -> None:
        """Record transport ownership commit, not application finality."""

        with self._lock:
            record = self._records.get(key)
            if record is not None:
                record.phase = RequestPhase.HANDOFF_COMPLETE

    def retire_after_source_fence(self, key: GenerationKey) -> bool:
        """Drop scheduler objects after this endpoint relinquishes ownership.

        This registry is not the lifecycle ledger.  Once a real source fence
        released the local physical lease, retaining ``Req`` objects here
        would make memory grow with every agent turn.  Target-side records are
        untouched and stay live until their own compute/transfer handoff.
        """

        with self._lock:
            record = self._records.get(key)
            if record is None:
                return False
            if record.phase not in {
                RequestPhase.COMPUTE_COMPLETE,
                RequestPhase.TRANSFER_SUBMITTED,
                RequestPhase.HANDOFF_COMPLETE,
            }:
                raise RuntimeError("request registry retired before source fence")
            self._records.pop(key)
            if record.parent_key is not None:
                self._children.pop(record.parent_key, None)
            self._children.pop(key, None)
            return True

    def retire_final(self, key: GenerationKey) -> bool:
        """Drop one application-terminal generation after native release."""

        with self._lock:
            record = self._records.pop(key, None)
            if record is None:
                return False
            if record.parent_key is not None:
                self._children.pop(record.parent_key, None)
            self._children.pop(key, None)
            return True


@dataclass(frozen=True, slots=True)
class CompositeContext:
    scheduler: Any
    config: Any
    registry: RequestGenerationRegistry
    authority: AgenticMemoryAuthority
    p_memory_bridge: Optional[AgenticPMemorySchedulerBridge]
    d_memory_bridge: Optional[AgenticDMemorySchedulerBridge]


class RuntimePhysicalProvider(Protocol):
    """Fail-closed integration contract for physical KV/state ownership.

    Path handlers must perform target bind+ready and source release in their
    ``commit`` callback.  They must not publish ready or release source pages
    on a bare local DMA completion.  The provider is also responsible for the
    model-specific attention/recurrent-state layout.
    """

    def state_allocators(self, scheduler: Any) -> Sequence[Any]:
        ...

    def default_state_slot_counts(self, scheduler: Any) -> Sequence[int]:
        ...

    def executors(
        self, context: CompositeContext
    ) -> Mapping[TransferPath, TransferExecutor]:
        ...

    def handlers(
        self, context: CompositeContext
    ) -> Mapping[TransferPath, RankPathHandler]:
        ...

    def lanes(self, context: CompositeContext) -> Mapping[TransferPath, int]:
        ...

    def pending_capacity(
        self, context: CompositeContext
    ) -> Mapping[TransferPath, int]:
        ...

    def on_request(
        self, context: CompositeContext, record: RuntimeRequestRecord
    ) -> Optional[GroupTransferPlan]:
        """Register an arrival and optionally return its TP-atomic init plan.

        Only endpoint rank zero may return a plan.  In particular an initial P
        request must not reserve/publish independently on every rank: it uses
        a no-I/O group transaction and becomes scheduler-visible only after
        every shard has acknowledged RELEASE.
        """

    def plan_prefill_complete(
        self,
        context: CompositeContext,
        record: RuntimeRequestRecord,
        item: PrefillCompleteItem,
    ) -> Optional[GroupTransferPlan]:
        ...

    def plan_decode_complete(
        self,
        context: CompositeContext,
        record: RuntimeRequestRecord,
        item: DecodeCompleteItem,
    ) -> Optional[GroupTransferPlan]:
        ...

    def decide_intent(
        self,
        context: CompositeContext,
        intent: LinkIntent,
        candidate: GroupTransferPlan,
    ) -> Optional[GroupTransferPlan]:
        ...

    def on_committed(
        self, context: CompositeContext, plan: GroupTransferPlan, attempt: int
    ) -> None:
        ...

    def on_committed_results(
        self,
        context: CompositeContext,
        plan: GroupTransferPlan,
        attempt: int,
        results: Mapping[LinkParticipant, Mapping[str, Any]],
    ) -> None:
        ...

    def on_aborted(
        self,
        context: CompositeContext,
        plan: GroupTransferPlan,
        attempt: int,
        reason: str,
    ) -> None:
        ...

    def memory_available(self, *, remote_role: str) -> None:
        """Consume a coalesced remote allocator-capacity edge."""

        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class RuntimeFactoryOptions:
    low_level_factory: Callable[..., Any] = AgenticMultiNodeRuntime


_STOP = object()
_provider_factory: Optional[Callable[[Any, Any], RuntimePhysicalProvider]] = None
_provider_factory_lock = threading.Lock()


def install_physical_provider_factory(
    factory: Callable[[Any, Any], RuntimePhysicalProvider],
) -> None:
    """Install the process-local physical provider before scheduler startup."""

    if not callable(factory):
        raise TypeError("physical provider factory must be callable")
    global _provider_factory
    with _provider_factory_lock:
        if _provider_factory is not None and _provider_factory is not factory:
            raise RuntimeError("a physical provider factory is already installed")
        _provider_factory = factory


def _all_paths(value: Mapping[TransferPath, Any], name: str) -> dict:
    result = dict(value)
    if set(result) != set(TransferPath):
        raise ValueError(f"{name} must cover all four transfer paths")
    return result


class SchedulerAgenticMultinodeRuntime:
    """Non-blocking scheduler facade over one rank's V2 controller."""

    def __init__(
        self,
        scheduler: Any,
        config: Any,
        provider: RuntimePhysicalProvider,
        *,
        options: RuntimeFactoryOptions = RuntimeFactoryOptions(),
    ) -> None:
        self.scheduler = scheduler
        self.config = config
        self.provider = provider
        self.registry = RequestGenerationRegistry(config.run_id)
        state_allocators = tuple(provider.state_allocators(scheduler))
        state_counts = tuple(provider.default_state_slot_counts(scheduler))
        if len(state_allocators) != len(state_counts):
            raise ValueError("state allocator/count descriptions differ")
        self.authority = AgenticMemoryAuthority(
            scheduler.token_to_kv_pool_allocator,
            state_allocators=state_allocators,
        )
        role = str(config.role)
        self.p_memory_bridge = (
            AgenticPMemorySchedulerBridge(
                self.authority, default_state_slot_counts=state_counts
            )
            if role == "prefill"
            else None
        )
        self.d_memory_bridge = (
            AgenticDMemorySchedulerBridge(
                self.authority, default_state_slot_counts=state_counts
            )
            if role == "decode"
            else None
        )
        if role not in {"prefill", "decode"}:
            raise ValueError("scheduler runtime role must be prefill or decode")
        self.context = CompositeContext(
            scheduler,
            config,
            self.registry,
            self.authority,
            self.p_memory_bridge,
            self.d_memory_bridge,
        )
        executors = _all_paths(provider.executors(self.context), "executors")
        handlers = _all_paths(provider.handlers(self.context), "handlers")
        lanes = _all_paths(provider.lanes(self.context), "lanes")
        capacities = _all_paths(
            provider.pending_capacity(self.context), "pending capacities"
        )
        self.transfer_queues = AgenticTransferQueues(
            executors, lanes=lanes, pending_capacity=capacities
        )

        participants = tuple(
            LinkParticipant(endpoint_role, endpoint_group, rank)
            for endpoint_role, endpoint_group in (
                (config.role, config.endpoint_group),
                (config.peer_role, config.peer_group),
            )
            for rank in range(int(config.tp_size))
        )
        local = LinkParticipant(role, config.endpoint_group, scheduler.tp_rank)
        coordinator_role = (
            config.role
            if config.coordinator_group == config.endpoint_group
            else config.peer_role
        )
        coordinator = LinkParticipant(
            coordinator_role, config.coordinator_group, 0
        )
        self._low_level = options.low_level_factory(
            address=config.endpoint,
            run_id=config.run_id,
            link_id=config.group_id,
            token=config.control_token,
            participant=local,
            endpoint_tp_size=config.tp_size,
            participants=participants,
            coordinator_participant=coordinator,
            queues=self.transfer_queues,
            handlers=handlers,
            decide_intent=self._decide_intent,
            on_committed=self._on_committed,
            on_committed_results=self._on_committed_results,
            on_aborted=self._on_aborted,
            on_capacity_edge=self._on_capacity_edge,
            retain_terminals=False,
        )
        self._events: queue.Queue[Any] = queue.Queue()
        self._fatal: Optional[BaseException] = None
        self._closed = False
        self._capacity_edge_pending = False
        self._lock = threading.RLock()
        self._controller = threading.Thread(
            target=self._controller_loop,
            name=f"agentic-scheduler-controller-{role}-r{scheduler.tp_rank}",
            daemon=True,
        )
        if self.p_memory_bridge is not None:
            self.p_memory_bridge.install_completion_sink(self._events.put_nowait)
            self.p_memory_bridge.install_final_release_sink(
                self._on_final_release
            )
        if self.d_memory_bridge is not None:
            self.d_memory_bridge.install_completion_sink(self._events.put_nowait)
            self.d_memory_bridge.install_final_release_sink(
                self._on_final_release
            )
        self._low_level.start()
        if role == "decode" and int(scheduler.tp_rank) == 0:
            self.authority.install_capacity_available_sink(
                self._notify_d_capacity_available
            )
        install_submitter = getattr(provider, "install_submitter", None)
        if install_submitter is not None:
            install_submitter(self._provider_submit)
        self._controller.start()

    def _on_final_release(
        self, attempt: RequestGenerationAttempt, _reason: str
    ) -> None:
        key = GenerationKey(
            self.config.run_id, attempt.request_id, attempt.generation
        )
        self.registry.retire_final(key)

    def submit_request(self, req: Any, *, is_retracted: bool = False) -> None:
        if is_retracted:
            raise RuntimeError("V2 does not re-admit a retracted native request")
        with self._lock:
            if self._closed or self._fatal is not None:
                raise RuntimeError("V2 runtime is not accepting requests")
        record = self.registry.register(req)
        # Registration is an endpoint-local fact.  Every TP rank reports it;
        # only P0 uses the all-rank readiness barrier to start an attempt.
        self._low_level.report_registered(record.key)
        self._events.put_nowait(record)

    def _key_from_completion(self, item: Any) -> GenerationKey:
        key = item.lease.key
        return GenerationKey(self.config.run_id, key.request_id, key.generation)

    def _submit_plan(self, record: RuntimeRequestRecord, plan: GroupTransferPlan) -> None:
        if plan.key != record.key:
            raise RuntimeError("provider changed completion request-generation")
        attempt = self._low_level.submit(plan)
        self.registry.mark_submitted(record.key, attempt)

    def _submit_init_plan(
        self, record: RuntimeRequestRecord, plan: GroupTransferPlan
    ) -> None:
        if plan.key != record.key:
            raise RuntimeError("initial admission changed request-generation")
        attempt = self._low_level.submit(plan)
        self.registry.mark_init_submitted(record.key, attempt)

    def _provider_submit(self, plan: GroupTransferPlan) -> Optional[int]:
        """Event-driven provider submission, including a remote-parent key.

        A P-side policy actor may retain a D intent until the matching child
        request arrives, or retain a committed Host descriptor until a target
        workset is available.  Those plans name the parent generation and do
        not necessarily have a local registry record.  Submission is still
        rank-zero-only and goes through the one group coordinator.
        """

        if int(self.scheduler.tp_rank) != 0:
            raise RuntimeError("only endpoint rank zero may submit provider plans")
        return self._low_level.submit(plan)

    def _controller_loop(self) -> None:
        while True:
            event = self._events.get()
            if event is _STOP:
                return
            try:
                if isinstance(event, RuntimeRequestRecord):
                    plan = self.provider.on_request(self.context, event)
                    if plan is not None:
                        if int(self.scheduler.tp_rank) != 0:
                            raise RuntimeError(
                                "only endpoint rank zero may submit an init plan"
                            )
                        self._submit_init_plan(event, plan)
                elif isinstance(event, PrefillCompleteItem):
                    key = self._key_from_completion(event)
                    record = self.registry.mark_compute_complete(key)
                    plan = self.provider.plan_prefill_complete(
                        self.context, record, event
                    )
                    # The physical source exists on every rank before TP0 is
                    # allowed to issue the immutable group plan.
                    self._low_level.report_source_ready(key)
                    if int(self.scheduler.tp_rank) == 0:
                        if plan is not None:
                            self._submit_plan(record, plan)
                    elif plan is not None:
                        raise RuntimeError("follower proposed a Prefill plan")
                elif isinstance(event, DecodeCompleteItem):
                    key = self._key_from_completion(event)
                    record = self.registry.mark_compute_complete(key)
                    plan = self.provider.plan_decode_complete(
                        self.context, record, event
                    )
                    self._low_level.report_source_ready(key)
                    if int(self.scheduler.tp_rank) == 0:
                        if plan is not None:
                            self._submit_plan(record, plan)
                    elif plan is not None:
                        raise RuntimeError("follower proposed a Decode plan")
                elif isinstance(event, _CommittedEvent):
                    self._handle_committed(event.plan, event.attempt)
                elif isinstance(event, _CommittedResultsEvent):
                    callback = getattr(
                        self.provider, "on_committed_results", None
                    )
                    if callback is not None:
                        callback(
                            self.context,
                            event.plan,
                            event.attempt,
                            event.results,
                        )
                elif isinstance(event, _AbortedEvent):
                    self._handle_aborted(
                        event.plan, event.attempt, event.reason
                    )
                elif isinstance(event, _CapacityEdgeEvent):
                    with self._lock:
                        self._capacity_edge_pending = False
                    callback = getattr(self.provider, "memory_available", None)
                    if callback is not None:
                        callback(remote_role=event.edge.participant.role)
                else:
                    raise TypeError(f"unsupported controller event {event!r}")
            except BaseException as error:
                with self._lock:
                    if self._fatal is None:
                        self._fatal = error

    def _decide_intent(
        self, intent: LinkIntent, candidate: GroupTransferPlan
    ) -> Optional[GroupTransferPlan]:
        return self.provider.decide_intent(self.context, intent, candidate)

    def _on_committed(self, plan: GroupTransferPlan, attempt: int) -> None:
        # Low-level invokes this on its coordinator/control thread.  Never
        # call provider policy or submit a successor attempt from there.
        self._events.put_nowait(_CommittedEvent(plan, attempt))

    def _handle_committed(self, plan: GroupTransferPlan, attempt: int) -> None:
        self.provider.on_committed(self.context, plan, attempt)
        record = self.registry.get(plan.key)
        if record is not None and plan.payload.get("kind") == "initial_prefill":
            self.registry.mark_init_ready(plan.key)

    def _on_committed_results(
        self,
        plan: GroupTransferPlan,
        attempt: int,
        results: Mapping[LinkParticipant, Mapping[str, Any]],
    ) -> None:
        self._events.put_nowait(
            _CommittedResultsEvent(plan, attempt, dict(results))
        )

    def _on_aborted(
        self, plan: GroupTransferPlan, attempt: int, reason: str
    ) -> None:
        self._events.put_nowait(_AbortedEvent(plan, attempt, reason))

    def _on_capacity_edge(self, edge: LinkCapacityEdge) -> None:
        """Coalesce transient D-capacity hints before entering policy code."""

        with self._lock:
            if self._closed or self._fatal is not None or self._capacity_edge_pending:
                return
            self._capacity_edge_pending = True
        self._events.put_nowait(_CapacityEdgeEvent(edge))

    def _notify_d_capacity_available(self, available_tokens: int) -> None:
        """Allocator edge sink; ownership release never depends on this hint."""

        try:
            self._low_level.notify_capacity_available(int(available_tokens))
        except BaseException as error:
            # The memory release has already committed.  Record transport
            # failure without pretending it can be rolled back.
            with self._lock:
                if not self._closed and self._fatal is None:
                    self._fatal = error

    def _handle_aborted(
        self, plan: GroupTransferPlan, attempt: int, reason: str
    ) -> None:
        self.provider.on_aborted(self.context, plan, attempt, reason)
        # An aborted Direct attempt may immediately fall back to Host.  It is
        # not an ownership handoff and must not terminalize the local record.

    def progress_nonblocking(self) -> None:
        """Health edge only; scheduler never advances a physical transfer."""

        with self._lock:
            fatal = self._fatal
            closed = self._closed
        if fatal is not None:
            raise CompositeRuntimeError("V2 controller failed") from fatal
        if closed:
            raise CompositeRuntimeError("V2 runtime is closed")
        self._low_level.check_health()

    def is_idle(self) -> bool:
        """Return whether V2 owns no request, memory lease, or queued DMA."""

        if not self.registry.is_empty() or self.authority.active_lease_count():
            return False
        return all(
            snapshot.pending == 0 and snapshot.active == 0
            for snapshot in self.transfer_queues.snapshot().values()
        )

    def take_activation_ticket(
        self, timeout: Optional[float] = None
    ) -> EndpointActivationTicket:
        """TP0-only scheduler edge; no transport progress is done here."""

        return self._low_level.take_activation_ticket(timeout=timeout)

    def activate_staged(self, ticket: EndpointActivationTicket) -> None:
        """Publish a staged lease in native TP broadcast order on every rank."""

        self._low_level.activate_staged(ticket)

    def confirm_scheduler_adopted(self, ticket: EndpointActivationTicket) -> None:
        """Cross the lifecycle fence after native scheduler queue insertion."""

        self._low_level.confirm_scheduler_adopted(ticket)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self.p_memory_bridge is not None:
            self.p_memory_bridge.install_completion_sink(None)
            self.p_memory_bridge.install_final_release_sink(None)
        if self.d_memory_bridge is not None:
            self.d_memory_bridge.install_completion_sink(None)
            self.d_memory_bridge.install_final_release_sink(None)
        self.authority.install_capacity_available_sink(None)
        # Keep the policy thread alive while the link runtime drains terminal
        # callbacks; otherwise an abort/commit callback can be silently lost.
        self._low_level.close()
        self._events.put(_STOP)
        self._controller.join(timeout=5.0)
        if self._controller.is_alive():
            raise CompositeRuntimeError("V2 controller did not stop")
        self.provider.close()


def create_runtime(
    scheduler: Any,
    config: Any,
    *,
    draft_token_to_kv_pool: Any = None,
    draft_model_config: Any = None,
    physical_provider: Optional[RuntimePhysicalProvider] = None,
    options: RuntimeFactoryOptions = RuntimeFactoryOptions(),
) -> SchedulerAgenticMultinodeRuntime:
    """Build the scheduler runtime or fail before any legacy path is created."""

    if draft_token_to_kv_pool is not None or draft_model_config is not None:
        raise ValueError("multi-node V2 does not yet transfer speculative state")
    provider = physical_provider
    if provider is None:
        with _provider_factory_lock:
            factory = _provider_factory
        if factory is None:
            from sglang.srt.disaggregation.agentic_default_physical_provider import (
                create_default_physical_provider,
            )

            factory = create_default_physical_provider
        provider = factory(scheduler, config)
    return SchedulerAgenticMultinodeRuntime(
        scheduler, config, provider, options=options
    )


__all__ = [
    "CompositeContext",
    "CompositeRuntimeError",
    "RequestGenerationRegistry",
    "RuntimeFactoryOptions",
    "RuntimePhysicalProvider",
    "RuntimeRequestRecord",
    "SchedulerAgenticMultinodeRuntime",
    "create_runtime",
    "install_physical_provider_factory",
]
