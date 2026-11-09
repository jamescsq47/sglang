"""Canonical local Router records, protocol state aliases and historical logger."""
from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from sglang.srt.disaggregation.agentic_host_staging import HostStageState

logger = logging.getLogger(
    __package__ if __package__ == "dualpd_engine_late_binding_router"
    else f"{__package__}.late_binding_router"
)

# Complete-snapshot Host eviction is optional.  The Router must also run
# against environments that implement the core staging lifecycle but predate
# the EVICTING/RECOMPUTE_REQUIRED extension.
_HOST_STAGE_EVICTING = getattr(HostStageState, "EVICTING", None)
_HOST_STAGE_RECOMPUTE_REQUIRED = getattr(
    HostStageState, "RECOMPUTE_REQUIRED", None
)


@dataclass(frozen=True)
class DecodeLoad:
    url: str
    used_tokens: int
    capacity_tokens: int
    running: int
    waiting: int
    prealloc: int
    transfer: int
    max_running: int
    running_kv_tokens: int = 0
    prealloc_tokens: int = 0
    transfer_tokens: int = 0
    physical_used_tokens: int = 0


@dataclass(frozen=True)
class DecodeReservation:
    reservation_id: str
    url: str
    prompt_tokens: int
    admission_tokens: int
    request_count: int
    rooms: tuple[int, ...]
    created_at: float
    draining: bool = False
    p2d_host_snapshot_id: Optional[str] = None
    prefill_domain: Optional[int] = None


@dataclass
class _PrefillWorkReservation:
    domain: int
    tokens: int
    requests: int = 1
    direct_workset_tokens: int = 0
    released: bool = False
    route_pending: bool = False


@dataclass(eq=False)
class _PrefillAdmissionWaiter:
    parent_turn: bool
    enqueued_at: float
    sequence: int


@dataclass(eq=False)
class _PReadyAdmission:
    """One immutable Prefill completion waiting for ordered D admission.

    Request coroutines produce these records and then sleep on ``future``.
    A single admission broker consumes the head of every logical P queue,
    reserves several feasible Decode destinations from one load snapshot, and
    commits each P in ``ready_sequence`` order.  Slow/capacity-bound heads keep
    one explicit owner without blocking the other P queues.
    """

    domain: int
    sequence: int
    submitted_key: Any
    enqueued_at: float
    dispatch: Callable[[], Awaitable[Any]]
    future: asyncio.Future
    finished: asyncio.Event
    request: Optional[dict[str, Any]] = None
    rooms: tuple[int, ...] = ()
    prompt_tokens: int = 0
    commit: Optional[Callable[[DecodeReservation], Awaitable[Any]]] = None
    prepare: Optional[
        Callable[[], Awaitable[Optional[DecodeReservation]]]
    ] = None
    dispatch_task: Optional[asyncio.Task] = None
    cancel_requested: bool = False
    initial_reservation: Optional[DecodeReservation] = None
    prepare_complete: bool = False
    host_staged: bool = False
    ownership_started: bool = False
    commit_started: bool = False
    commit_predecessor: Optional[asyncio.Future] = None
    commit_done: Optional[asyncio.Future] = None


@dataclass(frozen=True)
class _GenerationResponse:
    payload: dict[str, Any]
    status: int
    completed_at: float


class _PrefillRedirect(RuntimeError):
    def __init__(self, domain: int, route: str):
        super().__init__(f"redirect Prefill to P{domain} via {route}")
        self.domain = domain
        self.route = route
