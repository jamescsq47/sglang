"""Late-binding PD router for the agentic serving experiments.

The stock SGLang PD router selects a prefill/decode pair before either request
is dispatched.  In P-ready mode the prefill worker can compute without a
decode destination, so this router deliberately delays decode dispatch until
the prefill worker publishes ``<bootstrap_room>.ready``.

This module only changes routing/admission timing.  KV transfer, HiCache and
Mooncake continue to use the SGLang implementations selected by the workers.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import aiohttp
import orjson
from fastapi.responses import ORJSONResponse, StreamingResponse
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    unpack_agentic_extra_key,
)
from sglang.srt.disaggregation.agentic_host_staging import (
    HostStageState,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.p2d_host_staging import (
    P2D_CUSTOM_PREFILL_DOMAIN,
    P2D_CUSTOM_SNAPSHOT_ID,
    p2d_snapshot_id,
)
from sglang.srt.disaggregation.agentic_prefill_pressure import (
    SharedPrefillPressureReservations,
)
from sglang_router.mini_lb import (
    AIOHTTP_STREAM_READ_CHUNK_SIZE,
    MiniLoadBalancer,
)


logger = logging.getLogger(__name__)

# Complete-snapshot Host eviction is optional.  The Router must also run
# against environments that implement the core staging lifecycle but predate
# the EVICTING/RECOMPUTE_REQUIRED extension.
_HOST_STAGE_EVICTING = getattr(HostStageState, "EVICTING", None)
_HOST_STAGE_RECOMPUTE_REQUIRED = getattr(HostStageState, "RECOMPUTE_REQUIRED", None)


def _sync_json_get(url: str, timeout: float) -> Any:
    """Fetch one tiny control-plane JSON document outside the ASGI loop."""

    request = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return orjson.loads(response.read())


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    reservation_id: str = ""
    host_snapshot_id: Optional[str] = None
    host_assignment_pending: bool = False
    host_assignment_task: Optional[asyncio.Future] = None
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
    prepare: Optional[Callable[[], Awaitable[Optional[DecodeReservation]]]] = None
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


class _PrefillAdmissionGate:
    """Bound hidden HTTP/bootstrap work before requests reach P.

    Parent turns always precede initial requests.  This is intentionally a
    strict agentic-serving priority boundary: a New request must never consume
    the short D-HBM Direct deadline of a returned parent turn.  FIFO is kept
    within each class.
    """

    def __init__(self, limit: int, new_aging_seconds: float):
        if limit <= 0:
            raise ValueError("P admission limit must be positive")
        self.limit = limit
        self.new_aging_seconds = max(0.0, new_aging_seconds)
        self.active = 0
        self._sequence = 0
        self._waiters: list[_PrefillAdmissionWaiter] = []
        self._condition = asyncio.Condition()

    def _next_waiter(self, now: float) -> _PrefillAdmissionWaiter:
        def key(waiter: _PrefillAdmissionWaiter):
            # Never promote an aged New request above a parent.  GPU-side
            # admission has its own P-ready soft caps, while this gate only
            # orders the small HTTP/tokenizer work window in front of P.
            queue_class = 0 if waiter.parent_turn else 1
            return queue_class, waiter.enqueued_at, waiter.sequence

        return min(self._waiters, key=key)

    async def acquire(self, *, parent_turn: bool) -> float:
        waiter = _PrefillAdmissionWaiter(
            parent_turn=parent_turn,
            enqueued_at=time.monotonic(),
            sequence=self._sequence,
        )
        self._sequence += 1
        async with self._condition:
            self._waiters.append(waiter)
            try:
                while True:
                    if self.active < self.limit and self._next_waiter(time.monotonic()) is waiter:
                        self._waiters.remove(waiter)
                        self.active += 1
                        self._condition.notify_all()
                        return time.monotonic() - waiter.enqueued_at
                    # Aging changes priority even when no admission completes.
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=max(0.01, self.new_aging_seconds),
                        )
                    except asyncio.TimeoutError:
                        continue
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                raise

    async def release(self) -> None:
        async with self._condition:
            if self.active <= 0:
                raise RuntimeError("P admission release without active request")
            self.active -= 1
            self._condition.notify_all()


class LateBindingMiniLoadBalancer(MiniLoadBalancer):
    """Dispatch P first and bind D only after the exact P workload is ready."""

    def __init__(self, router_args):
        super().__init__(router_args)
        ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
        if not ready_dir:
            raise ValueError("Late-binding requires SGLANG_PD_P_READY_DIR on router, P and D")
        self.p_ready_dir = Path(ready_dir)
        self.p_ready_dir.mkdir(parents=True, exist_ok=True)
        self.ready_timeout = _env_float("SGLANG_PD_LATE_BIND_READY_TIMEOUT_S", 600.0)
        self.ready_poll_interval = _env_float("SGLANG_PD_LATE_BIND_POLL_INTERVAL_S", 0.02)
        self.max_prefill_inflight = _env_int("SGLANG_PD_LATE_BIND_MAX_PREFILL_INFLIGHT", 64)
        self.prefill_accept_timeout = _env_float("SGLANG_PD_LATE_BIND_ACCEPT_TIMEOUT_S", self.ready_timeout)
        self.prefill_queue_timeout = _env_float("SGLANG_PD_LATE_BIND_QUEUE_TIMEOUT_S", 3600.0)
        self.prefill_new_aging_seconds = _env_float("SGLANG_PD_LATE_BIND_NEW_AGING_S", 10.0)
        # Admission bounds the small HTTP/tokenizer bootstrap window in front
        # of each P.  A single global gate unnecessarily couples independent P
        # workers: pressure on P0 must not consume P1's admission capacity.
        self._prefill_admissions = [
            _PrefillAdmissionGate(self.max_prefill_inflight, self.prefill_new_aging_seconds) for _ in self.prefill_urls
        ]
        # Retain the legacy attribute for single-P users and lightweight test
        # fixtures constructed without __init__.
        self._prefill_admission = self._prefill_admissions[0]
        self.load_timeout = _env_float("SGLANG_PD_LATE_BIND_LOAD_TIMEOUT_S", 2.0)
        self.reservation_timeout = _env_float("SGLANG_PD_LATE_BIND_RESERVATION_TIMEOUT_S", 120.0)
        self.decode_headroom_tokens = _env_int("SGLANG_PD_LATE_BIND_DECODE_HEADROOM_TOKENS", 512)
        self.max_decode_admission_tokens = _env_int("SGLANG_PD_LATE_BIND_MAX_ADMISSION_TOKENS", 4096)
        self.request_load_weight = _env_float("SGLANG_PD_LATE_BIND_REQUEST_LOAD_WEIGHT", 0.05)
        self.transfer_request_weight = _env_float("SGLANG_PD_LATE_BIND_TRANSFER_REQUEST_WEIGHT", 2.0)
        self.context_token_floor = _env_int("SGLANG_PD_LATE_BIND_CONTEXT_TOKEN_FLOOR", 2048)
        self.context_token_ceiling = _env_int("SGLANG_PD_LATE_BIND_CONTEXT_TOKEN_CEILING", 8192)
        self.wait_for_feasible_decode = _env_bool("SGLANG_PD_LATE_BIND_WAIT_FOR_FEASIBLE", True)
        self.target_decode_kv_fraction = _env_float("SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION", 1.0)
        if not (0.0 < self.target_decode_kv_fraction <= 1.0):
            raise ValueError("SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION must be in (0, 1]")
        self.no_capacity_poll_interval = _env_float("SGLANG_PD_LATE_BIND_NO_CAPACITY_POLL_S", 0.01)
        self.load_cache_ttl = _env_float("SGLANG_PD_LATE_BIND_LOAD_CACHE_TTL_S", 0.20)
        self._selection_lock = asyncio.Lock()
        # Request coroutines only publish immutable P-ready admissions.  One
        # broker owns D-capacity accounting for every P, scans in completion
        # order, and pipelines multiple capacity-feasible generations.  The
        # sequence is a fairness hint: one Host-staged generation must not
        # block an unrelated generation that can enter D immediately.
        self._p_ready_submitted_sequences: set[Any] = set()
        self._p_ready_fifo_waiters: dict[int, dict[int, _PReadyAdmission]] = {}
        self._p_ready_fifo_events: dict[int, asyncio.Event] = {}
        self._p_ready_fifo_dispatchers: dict[int, asyncio.Task] = {}
        self._p_ready_fifo_active: dict[int, dict[int, _PReadyAdmission]] = {}
        self._p_ready_broker_event = asyncio.Event()
        self._p_ready_broker_task: Optional[asyncio.Task] = None
        self._p_ready_commit_tails: dict[int, asyncio.Future] = {}
        self._p_ready_admission_window_per_p = _env_int("SGLANG_PD_LATE_BIND_ADMISSION_WINDOW_PER_P", 32)
        self._p_ready_stage_lanes_per_p = _env_int("SGLANG_AGENTIC_KV_P2D_D2H_WORKERS", 4)
        self._p_ready_stage_semaphores: dict[int, asyncio.Semaphore] = {}
        # P->D Host ownership is request-local.  Capacity-bound completions may
        # enter a bounded multi-lane staging pipeline, while a separate commit
        # chain keeps their eventual D admission strictly FIFO.
        # Multi-P runs can have hundreds of HTTP handlers waiting for P-ready.
        # Polling and JSON-decoding the same /dev/shm directory independently
        # in every handler creates an avoidable O(waiters * ready_files) control
        # path.  One watcher owns the directory scan and wakes every matching
        # waiter in a batch.  The single-P path intentionally keeps its proven
        # per-request behavior.
        self._p_ready_monitor_task: Optional[asyncio.Task] = None
        self._p_ready_waiters: dict[int, set[asyncio.Future]] = {}
        self._p_ready_snapshot: dict[int, dict[str, Any]] = {}
        # Only an attempt currently owned by a Router coroutine may
        # participate in the P-ready FIFO.  A redirected P request can race
        # with abort and publish a late marker after Router has moved the
        # generation to another P.  Such an orphan must never become the FIFO
        # head and stop every later P->D handoff.
        self._active_prefill_attempts: dict[int, str] = {}
        self._reservations: dict[str, DecodeReservation] = {}
        self._last_loads: dict[str, DecodeLoad] = {}
        self._load_cache: list[DecodeLoad] = []
        # ``_load_cache_at`` is the publication time used only for TTL.
        # Reservation accounting uses the causal sampling epoch below: a D
        # admission can be considered observed only by a poll that started
        # after that admission completed.
        self._load_cache_at = 0.0
        self._load_cache_sample_started_at = 0.0
        self._load_sample_started_at_by_url: dict[str, float] = {}
        self._load_refresh_task: Optional[asyncio.Task] = None
        # A reservation remains charged after D consumes its P-ready marker
        # until a newer load snapshot includes that allocation.  This lets the
        # hot dispatch path use stale-while-revalidate snapshots safely.
        self._admitted_reservation_at: dict[str, float] = {}
        self._legacy_load_urls: set[str] = (
            set(self.decode_urls) if _env_bool("SGLANG_PD_LATE_BIND_FORCE_LEGACY_LOADS") else set()
        )
        self._prefill_index = 0
        # Ablation-only policy: preserve all capacity/ownership checks, but
        # replace both load-aware P and D choices with deterministic random
        # choices among the currently feasible workers.
        self.ablation_random_routing = _env_bool("SGLANG_PD_ABLATION_RANDOM_ROUTING")
        self._routing_rng = random.Random(_env_int("SGLANG_PD_ABLATION_RANDOM_SEED", 2026))
        # Router-owned shadow queues model Prefill compute work.  A cached P
        # pressure snapshot additionally prevents Direct from targeting a P
        # that cannot fit the complete parent+suffix workset.  Selection never
        # performs an HTTP request or waits for a P scheduler.
        self._prefill_work_lock = asyncio.Lock()
        self._prefill_pending_tokens = [0] * len(self.prefill_urls)
        self._prefill_pending_requests = [0] * len(self.prefill_urls)
        self._prefill_direct_pending_tokens = [0] * len(self.prefill_urls)
        self._prefill_work_tiebreak = 0
        self._prefill_pressure_task: Optional[asyncio.Task] = None
        self._host_capacity_task: Optional[asyncio.Task] = None
        self._prefill_pressure_domains: list[dict[str, Any]] = []
        self._prefill_pressure_at = 0.0
        # Start time of the HTTP sample currently represented by
        # ``_prefill_pressure_domains``.  Direct shadow credit is retained
        # until a sample started after the physical Direct completion has
        # observed the new P allocation.  This makes the hand-off from Router
        # accounting to allocator accounting atomic from later selectors'
        # point of view.
        self._prefill_pressure_sample_started_at = 0.0
        self._prefill_pressure_interval = _env_float("SGLANG_AGENTIC_KV_PREFILL_LOAD_INTERVAL_S", 0.20)
        pressure_path = os.getenv("SGLANG_AGENTIC_KV_PREFILL_LOAD_PATH", "")
        if not pressure_path:
            pressure_path = str(self.p_ready_dir / "early-claims" / "prefill-loads.json")
        self._prefill_pressure_path = Path(pressure_path)
        self._host_capacity_path = Path(
            os.getenv(
                "SGLANG_AGENTIC_KV_HOST_CAPACITY_PATH",
                str(self.p_ready_dir / "early-claims" / "host-capacities.json"),
            )
        )
        reservation_path = os.getenv(
            "SGLANG_AGENTIC_KV_PREFILL_RESERVATION_PATH",
            f"{pressure_path}.reservations",
        )
        self._prefill_pressure_reservations = (
            SharedPrefillPressureReservations(
                reservation_path,
                ttl_seconds=_env_float("SGLANG_AGENTIC_KV_PREFILL_RESERVATION_TTL_S", 5.0),
            )
            if reservation_path
            else None
        )
        staging_path = os.getenv("SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH", "")
        self._d2p_host_ledger = SharedHostStagingLedger(staging_path) if staging_path else None
        # One logical request-generation may outlive an HTTP client's timeout.
        # Keep the actual P->D dispatch detached from that client and let every
        # retry await the same task.  Without this fence, a retry can select a
        # second D and duplicate both Decode compute and KV lifecycle writes.
        self._generation_lock = asyncio.Lock()
        self._generation_tasks: dict[str, asyncio.Task[tuple[dict[str, Any], int]]] = {}
        self._generation_results: dict[str, _GenerationResponse] = {}
        # All long-lived generation tasks share one connector.  Creating and
        # tearing down a ClientSession per request is unsafe at c512: each
        # generation keeps both P and D HTTP transports alive for its whole
        # serving lifetime, and rapid session teardown/retry can recycle a
        # socket fd while uvloop still owns its previous transport.
        self._backend_session: Optional[aiohttp.ClientSession] = None
        # Load sampling is a control plane and must not share connector
        # queues/sockets with hundreds of long-lived P/D generation requests.
        # One small persistent pool isolates /get_load progress without
        # returning to unsafe per-request ClientSession creation.
        self._load_session: Optional[aiohttp.ClientSession] = None
        self.generation_result_ttl = _env_float("SGLANG_PD_GENERATION_RESULT_TTL_S", 3600.0)
        self.max_generation_results = _env_int("SGLANG_PD_MAX_GENERATION_RESULTS", 8192)
        self.numa_domains = _env_bool("SGLANG_PD_LATE_BIND_NUMA_DOMAINS")
        self.dynamic_prefill_domains = _env_bool("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS")
        self.global_decode = _env_bool("SGLANG_PD_LATE_BIND_GLOBAL_DECODE")
        self.p2d_host_staging = _env_bool("SGLANG_AGENTIC_KV_P2D_HOST_STAGING")
        self.p2d_host_spill_delay = _env_float("SGLANG_AGENTIC_KV_P2D_SPILL_DELAY_SECONDS", 0.5)
        self.p2d_host_ledger = None
        self._p2d_host_offered_snapshots: set[str] = set()
        if self.p2d_host_staging:
            p2d_ledger_path = os.getenv(
                "SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH",
                f"{os.getenv('SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH', '')}.p2d",
            )
            if not p2d_ledger_path or p2d_ledger_path == ".p2d":
                raise ValueError("P->D Host staging requires a ledger path")
            self.p2d_host_ledger = SharedHostStagingLedger(p2d_ledger_path)
            # Preserve ownership across a Router restart.  The hot broker can
            # then reject Host-owned snapshots without parsing the complete
            # ledger for every ordinary Direct admission.
            self._p2d_host_offered_snapshots.update(self.p2d_host_ledger.snapshot_entries())
        if self.numa_domains and (len(self.prefill_urls) < 2 or len(self.decode_urls) % len(self.prefill_urls) != 0):
            raise ValueError(
                "NUMA-domain routing requires at least two P workers and an " "equal contiguous D partition per P"
            )
        if self.dynamic_prefill_domains and not self.numa_domains:
            raise ValueError("dynamic Prefill domains require SGLANG_PD_LATE_BIND_NUMA_DOMAINS")
        self.early_claim_store = None
        if _env_bool("SGLANG_AGENTIC_KV_EARLY_CLAIM"):
            early_claim_dir = os.environ.get("SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR", "")
            if not early_claim_dir:
                early_claim_dir = str(self.p_ready_dir / "early-claims")
            self.early_claim_store = AgenticEarlyClaimStore(early_claim_dir)
        logger.info(
            "Late-binding PD enabled: ready_dir=%s decodes=%d headroom=%d "
            "early_claim=%s max_prefill_inflight_per_p=%d",
            self.p_ready_dir,
            len(self.decode_urls),
            self.decode_headroom_tokens,
            self.early_claim_store is not None,
            self.max_prefill_inflight,
        )

    def select_pair(self):
        """Select only P here; the endpoint passes the D placeholder to us."""
        if not self.prefill_urls:
            raise RuntimeError("No prefill servers available")
        if getattr(self, "ablation_random_routing", False):
            index = self._routing_rng.randrange(len(self.prefill_urls))
        else:
            index = self._prefill_index % len(self.prefill_urls)
        self._prefill_index += 1
        return (
            self.prefill_urls[index],
            self.prefill_bootstrap_ports[index],
            None,
        )

    def _choose_decode_score(self, candidates, *, drain: bool = False):
        """Choose one capacity-accounted D score for normal or ablation runs."""

        if getattr(self, "ablation_random_routing", False):
            # Sort first so a fixed ablation seed is independent of response
            # arrival order from the asynchronous load probes.
            return self._routing_rng.choice(sorted(candidates, key=lambda item: item[4].url))
        if drain:
            return min(
                candidates,
                key=lambda item: (item[2], item[1], item[4].url),
            )
        return min(
            candidates,
            key=lambda item: (item[1], item[2], item[4].url),
        )

    @staticmethod
    def _rooms(modified_request: dict[str, Any]) -> tuple[int, ...]:
        rooms = modified_request["bootstrap_room"]
        if not isinstance(rooms, list):
            rooms = [rooms]
        return tuple(int(room) for room in rooms)

    def _ready_path(self, room: int) -> Path:
        return self.p_ready_dir / f"{room}.ready"

    def _accepted_path(self, room: int) -> Path:
        return self.p_ready_dir / f"{room}.accepted"

    def _request_domain(self, metadata, rooms: tuple[int, ...]) -> int:
        if not getattr(self, "numa_domains", False):
            return 0
        key = metadata.request_id if metadata is not None else str(rooms[0])
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % len(self.prefill_urls)

    @staticmethod
    def _request_input_tokens(request: dict[str, Any]) -> int:
        input_ids = request.get("input_ids")
        if not isinstance(input_ids, list):
            custom_params = request.get("custom_params")
            hint = custom_params.get("agentic_prompt_token_count") if isinstance(custom_params, dict) else None
            # This is only shadow admission accounting.  Bound the hint to a
            # generous finite range; the model server remains authoritative
            # for actual tokenization and context-length validation.
            if isinstance(hint, int) and not isinstance(hint, bool) and 0 < hint <= 10_000_000:
                return hint
            return 1
        if not input_ids:
            return 1
        if isinstance(input_ids[0], list):
            return max(1, sum(len(item) for item in input_ids))
        return max(1, len(input_ids))

    def _estimated_prefill_tokens(
        self,
        request: dict[str, Any],
        snapshot_tokens: Optional[int] = None,
    ) -> int:
        current_tokens = self._request_input_tokens(request)
        if snapshot_tokens is None:
            return current_tokens
        # The reusable D snapshot contains the previous prompt and generated
        # response.  The remaining suffix is the tool result plus chat framing.
        return max(1, current_tokens - max(0, int(snapshot_tokens)))

    async def _reserve_prefill_work(
        self,
        tokens: int,
        *,
        domain: Optional[int] = None,
    ) -> _PrefillWorkReservation:
        """Account ordinary Prefill work.

        Parent Direct routing deliberately does not use this path: its P
        choice and complete-workset debit are performed without awaiting this
        lock by :meth:`_reserve_direct_prefill_work_nowait`.
        """

        tokens = max(1, int(tokens))
        async with self._prefill_work_lock:
            if domain is None:
                count = len(self.prefill_urls)
                start = self._prefill_work_tiebreak % count
                candidates = list(range(count))
                pressure = getattr(self, "_prefill_pressure_domains", [])
                pressure_by_domain = {int(row.get("domain", -1)): row for row in pressure}
                if getattr(self, "ablation_random_routing", False):
                    domain = self._routing_rng.choice(sorted(candidates))
                else:
                    domain = min(
                        candidates,
                        key=lambda candidate: (
                            (
                                self._prefill_pressure_score(
                                    pressure_by_domain.get(candidate, {}),
                                    pending_tokens=self._prefill_pending_tokens[candidate],
                                    pending_requests=self._prefill_pending_requests[candidate],
                                )
                                if pressure_by_domain.get(candidate) is not None
                                else float(self._prefill_pending_tokens[candidate])
                            ),
                            (candidate - start) % count,
                        ),
                    )
                self._prefill_work_tiebreak = (domain + 1) % count
            if not 0 <= domain < len(self.prefill_urls):
                raise RuntimeError(f"invalid Prefill domain {domain}")
            self._prefill_pending_tokens[domain] += tokens
            self._prefill_pending_requests[domain] += 1
            reservation = _PrefillWorkReservation(
                domain=domain,
                tokens=tokens,
                reservation_id=uuid.uuid4().hex,
            )
            logger.info(
                "PD_P_WORK_RESERVE P=%d tokens=%d pending_tokens=%d "
                "pending_requests=%d",
                domain,
                tokens,
                self._prefill_pending_tokens[domain],
                self._prefill_pending_requests[domain],
            )
            return reservation

    def _reserve_direct_prefill_work_nowait(
        self,
        tokens: int,
        *,
        direct_workset_tokens: int,
        domain: Optional[int] = None,
        already_committed: bool = False,
    ) -> _PrefillWorkReservation:
        """Select a Direct P and debit its complete workset without yielding.

        The Router's asyncio event loop is the single writer for these local
        counters, so this non-awaiting operation is the atomic selection/CAS
        boundary.  The hot path intentionally considers only physical P KV
        use plus outstanding workset debits.  Scheduler queues, Host arenas,
        and P->D delivery pressure do not participate in Direct routing.

        If no sampled P can fit, the least-used P is only a routing target;
        no capacity is charged and its physical allocator remains free to
        reject the claim, after which D follows the normal Direct->Slow path.
        """

        started_at = time.monotonic()
        tokens = max(1, int(tokens))
        requested_workset = max(1, int(direct_workset_tokens))
        count = len(self.prefill_urls)
        if not hasattr(self, "_prefill_direct_pending_tokens"):
            self._prefill_direct_pending_tokens = [0] * count

        pressure_max_age = max(
            1.0,
            5.0 * float(getattr(self, "_prefill_pressure_interval", 0.2)),
        )
        pressure_is_fresh = (
            time.monotonic() - float(getattr(self, "_prefill_pressure_at", 0.0))
            <= pressure_max_age
        )
        rows = {
            int(row.get("domain", index)): row
            for index, row in enumerate(getattr(self, "_prefill_pressure_domains", []))
        }
        candidates = [domain] if domain is not None else list(range(count))
        if any(candidate is None or not 0 <= int(candidate) < count for candidate in candidates):
            raise RuntimeError(f"invalid Prefill domain {domain}")
        candidates = [int(candidate) for candidate in candidates]

        effective_used: dict[int, int] = {}
        feasible: list[int] = []
        if pressure_is_fresh:
            for candidate in candidates:
                row = rows.get(candidate)
                if row is None:
                    continue
                capacity = max(0, int(row.get("hbm_capacity_tokens", 0)))
                used = max(0, int(row.get("hbm_used_tokens", 0)))
                used += max(0, int(self._prefill_direct_pending_tokens[candidate]))
                effective_used[candidate] = used
                if capacity > 0 and used + requested_workset <= capacity:
                    feasible.append(candidate)

        start = self._prefill_work_tiebreak % count
        selectable = feasible or list(effective_used) or candidates
        if getattr(self, "ablation_random_routing", False):
            selected = self._routing_rng.choice(sorted(selectable))
        else:
            selected = min(
                selectable,
                key=lambda candidate: (
                    effective_used.get(candidate, 0),
                    (candidate - start) % count,
                ),
            )
        self._prefill_work_tiebreak = (selected + 1) % count

        # A completed Direct already has a physical P allocation.  Charge the
        # bridge even if the cached sample is stale; a causally newer sample
        # will take over this accounting in _settle_direct_workset_after_pressure.
        charged_workset = requested_workset if (selected in feasible or already_committed) else 0
        self._prefill_pending_tokens[selected] += tokens
        self._prefill_pending_requests[selected] += 1
        self._prefill_direct_pending_tokens[selected] += charged_workset
        reservation = _PrefillWorkReservation(
            domain=selected,
            tokens=tokens,
            direct_workset_tokens=charged_workset,
            reservation_id=uuid.uuid4().hex,
        )
        logger.info(
            "PD_P_DIRECT_SELECT P=%d tokens=%d workset_tokens=%d "
            "charged_workset_tokens=%d effective_kv_used=%d feasible=%s "
            "decision_ms=%.3f",
            selected,
            tokens,
            requested_workset,
            charged_workset,
            effective_used.get(selected, -1),
            selected in feasible or already_committed,
            1000.0 * (time.monotonic() - started_at),
        )
        return reservation

    async def _settle_direct_workset(self, reservation: Optional[_PrefillWorkReservation]) -> None:
        """Drop Router shadow credit when no physical-accounting bridge is needed.

        Fallback has no Direct allocation on the originally selected P and can
        settle immediately.  Direct success must instead use
        ``_settle_direct_workset_after_pressure`` so cached physical pressure
        and Router shadow accounting cannot both omit the same workset.
        """

        if reservation is None or reservation.direct_workset_tokens <= 0:
            return
        async with self._prefill_work_lock:
            if reservation.direct_workset_tokens <= 0:
                return
            domain = reservation.domain
            self._prefill_direct_pending_tokens[domain] -= reservation.direct_workset_tokens
            if self._prefill_direct_pending_tokens[domain] < 0:
                raise RuntimeError("Prefill Direct shadow accounting underflow")
            reservation.direct_workset_tokens = 0

    async def _settle_direct_workset_after_pressure(
        self,
        reservation: Optional[_PrefillWorkReservation],
        *,
        direct_terminal_at: float,
    ) -> None:
        """Hand Direct credit to a causally newer physical-HBM sample.

        A Direct completion and the periodic ``/get_load`` sampling run in
        different tasks.  Clearing Router credit immediately would leave a
        deterministic interval in which the cached sample is still old while
        the shadow has already disappeared.  Keep the shadow until a sample
        whose fetch began after completion has been published.  Cancellation
        is safe: the dispatch owner ultimately calls ``_release_prefill_work``
        and removes any remaining credit exactly once.
        """

        if reservation is None:
            return
        if reservation.direct_workset_tokens <= 0:
            # No capacity bridge was charged.  Routing intent nevertheless
            # has to end as soon as Direct reaches a terminal state.
            await self._settle_direct_workset(reservation)
            return
        poll_interval = max(
            0.01,
            min(
                0.05,
                float(getattr(self, "_prefill_pressure_interval", 0.20)),
            ),
        )
        while not reservation.released and reservation.direct_workset_tokens > 0:
            if float(getattr(self, "_prefill_pressure_sample_started_at", 0.0)) >= direct_terminal_at:
                await self._settle_direct_workset(reservation)
                return
            await asyncio.sleep(poll_interval)

    async def _fetch_prefill_hbm_pressure(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[int, int, int, list[int]]:
        """Read one logical P without putting the query on its GPU loop."""

        timeout = aiohttp.ClientTimeout(total=self.load_timeout)
        async with session.get(f"{url}/get_load", timeout=timeout) as response:
            response.raise_for_status()
            rows = await response.json()
        if isinstance(rows, dict):
            rows = [rows]
        rank_used = [int(row.get("num_physical_used_tokens", row.get("num_tokens", 0))) for row in rows]
        rank_capacity = [int(row.get("max_total_num_tokens", 0)) for row in rows]
        rank_free = [max(0, capacity - used) for used, capacity in zip(rank_used, rank_capacity)]
        # One logical token consumes one KV slot on every TP rank.  Therefore
        # group feasibility is bounded by the weakest rank, never by the sum.
        capacity = min(rank_capacity, default=0)
        free = min(rank_free, default=0)
        used = max(0, capacity - free)
        waiting = sum(int(row.get("num_waiting_reqs", 0)) for row in rows)
        return used, capacity, waiting, rank_free

    @staticmethod
    def _write_prefill_pressure(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
        temporary.write_bytes(orjson.dumps(payload))
        os.replace(temporary, path)

    def _prefill_arena_bytes(self) -> list[int]:
        used = [0] * len(self.prefill_urls)
        ledger = self._d2p_host_ledger
        if ledger is None:
            return used
        live_states = {
            HostStageState.HOST_RESERVED.value,
            HostStageState.HOST_WRITING.value,
            HostStageState.HOST_READY.value,
            HostStageState.H2D_LOADING.value,
            # ABORTING likewise owns its extent until D is quiescent.
            HostStageState.ABORTING.value,
        }
        # EVICTING still owns its TP-local extents; RECOMPUTE_REQUIRED does
        # not.  Add the former only when this environment exposes it.
        if _HOST_STAGE_EVICTING is not None:
            live_states.add(_HOST_STAGE_EVICTING.value)
        for entry in ledger.snapshot_entries().values():
            if entry.get("state") not in live_states:
                continue
            domain = int(entry.get("arena_domain", -1))
            if 0 <= domain < len(used):
                used[domain] += int(entry.get("byte_size", 0))
        return used

    def _d2p_recovery_reservations_by_domain(self) -> list[tuple[int, int]]:
        """Return live Router-assigned Slow worksets per logical P."""

        totals = [[0, 0] for _ in self.prefill_urls]
        ledger = self._d2p_host_ledger
        if ledger is None:
            return [(0, 0) for _ in self.prefill_urls]
        for entry in ledger.snapshot_entries().values():
            domain = int(entry.get("recovery_prefill_domain", -1))
            if not 0 <= domain < len(totals):
                continue
            if entry.get("state") in {
                HostStageState.CONSUMED.value,
                HostStageState.FAILED.value,
                HostStageState.RECOMPUTE_REQUIRED.value,
            }:
                continue
            if not entry.get("recovery_reservation_id"):
                continue
            claims = entry.get("recovery_claims", {})
            if claims and all(value.get("phase") in {"leased", "io_inflight", "handed"} for value in claims.values()):
                # Physical /get_load accounting has taken over once every
                # rank owns its exact workset lease.
                continue
            totals[domain][0] += max(1, int(entry.get("recovery_workset_tokens", 1)))
            totals[domain][1] += 1
        return [(tokens, requests) for tokens, requests in totals]

    def _p2d_pressure_by_domain(
        self, ledger_entries: Optional[dict[str, dict[str, Any]]] = None
    ) -> list[dict[str, int]]:
        """Snapshot P->D delivery work that ordinary P load omits.

        Completed Prefill requests disappear from ``pending_tokens`` before
        their KV has necessarily reached D.  Count Router inflight/queued work
        and durable P->D Host ownership separately so D->P routing cannot
        mistake a delivery-blocked P for an idle one.
        """

        count = len(self.prefill_urls)
        rows = [
            {
                "p2d_inflight_tokens": 0,
                "p2d_inflight_requests": 0,
                "p2d_host_tokens": 0,
                "p2d_host_requests": 0,
                "p2d_host_bytes": 0,
            }
            for _ in range(count)
        ]
        durable_host_snapshots: set[str] = set()
        ledger = getattr(self, "p2d_host_ledger", None)
        if ledger_entries is None:
            ledger_entries = {} if ledger is None else ledger.snapshot_entries()
        if ledger_entries:
            live_states = {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
            }
            for snapshot_id, entry in ledger_entries.items():
                if entry.get("state") not in live_states:
                    continue
                domain = int(entry.get("prefill_domain", -1))
                if not 0 <= domain < count:
                    continue
                durable_host_snapshots.add(str(snapshot_id))
                rows[domain]["p2d_host_tokens"] += max(0, int(entry.get("token_count", 0)))
                rows[domain]["p2d_host_requests"] += 1
                rows[domain]["p2d_host_bytes"] += max(0, int(entry.get("byte_size", 0)))

        seen: set[tuple[int, int]] = set()
        for containers in (
            getattr(self, "_p_ready_fifo_waiters", {}),
            getattr(self, "_p_ready_fifo_active", {}),
        ):
            for domain, admissions in containers.items():
                if not 0 <= int(domain) < count:
                    continue
                for admission in admissions.values():
                    # Once a D reservation or durable P->D Host owner exists,
                    # that generation is accounted below by its authoritative
                    # delivery state.  Counting the queue record as well would
                    # charge one blocked generation twice and distort P choice.
                    if admission.initial_reservation is not None:
                        continue
                    p2d_snapshot = self._p2d_snapshot_for_rooms(admission.rooms)
                    if p2d_snapshot is not None and p2d_snapshot in durable_host_snapshots:
                        continue
                    key = (int(domain), int(admission.sequence))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows[int(domain)]["p2d_inflight_tokens"] += max(0, int(admission.prompt_tokens))
                    rows[int(domain)]["p2d_inflight_requests"] += max(1, len(admission.rooms))
        for reservation in getattr(self, "_reservations", {}).values():
            domain = reservation.prefill_domain
            if domain is None or not 0 <= int(domain) < count:
                continue
            p2d_snapshot = reservation.p2d_host_snapshot_id
            if p2d_snapshot is None:
                p2d_snapshot = self._p2d_snapshot_for_rooms(reservation.rooms)
            if p2d_snapshot in durable_host_snapshots:
                continue
            rows[int(domain)]["p2d_inflight_tokens"] += max(0, int(reservation.prompt_tokens))
            rows[int(domain)]["p2d_inflight_requests"] += max(1, int(reservation.request_count))
        return rows

    @staticmethod
    def _prefill_pressure_score(
        row: dict[str, Any],
        *,
        pending_tokens: int,
        pending_requests: int,
    ) -> float:
        capacity = max(1, int(row.get("hbm_capacity_tokens", 0)))
        arena_capacity = max(1, int(row.get("arena_capacity_bytes", 0)))
        p2d_capacity = max(1, int(row.get("p2d_arena_capacity_bytes", arena_capacity)))
        return (
            (
                max(0, int(pending_tokens))
                + max(0, int(row.get("p2d_inflight_tokens", 0)))
                + max(0, int(row.get("p2d_host_tokens", 0)))
                + max(0, int(row.get("d_slow_reserved_tokens", 0)))
            )
            / capacity
            + max(0, int(row.get("hbm_used_tokens", 0))) / capacity
            + 2.0 * max(0, int(row.get("arena_used_bytes", 0))) / arena_capacity
            + 2.0 * max(0, int(row.get("p2d_host_bytes", 0))) / p2d_capacity
            + 0.01
            * (
                max(0, int(pending_requests))
                + max(0, int(row.get("scheduler_waiting", 0)))
                + max(0, int(row.get("p2d_inflight_requests", 0)))
                + max(0, int(row.get("p2d_host_requests", 0)))
                + max(0, int(row.get("d_slow_reserved_requests", 0)))
            )
        )

    async def _prefill_pressure_monitor_loop(self) -> None:
        """Publish a nonblocking shared snapshot for D-rank0 slow routing."""

        tp_size = max(1, int(os.getenv("SGLANG_AGENTIC_KV_TP_SIZE", "1")))
        arena_capacity = int(float(os.getenv("SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB", "128")) * (1024**3) * tp_size)
        async with aiohttp.ClientSession() as session:
            while True:
                sample_started_at = time.monotonic()
                fetched = await asyncio.gather(
                    *(self._fetch_prefill_hbm_pressure(session, url) for url in self.prefill_urls),
                    return_exceptions=True,
                )
                async with self._prefill_work_lock:
                    pending_tokens = list(self._prefill_pending_tokens)
                    pending_requests = list(self._prefill_pending_requests)
                arena_used = await asyncio.to_thread(self._prefill_arena_bytes)
                recovery_reservations = await asyncio.to_thread(self._d2p_recovery_reservations_by_domain)
                p2d_ledger = getattr(self, "p2d_host_ledger", None)
                p2d_entries = {} if p2d_ledger is None else await asyncio.to_thread(p2d_ledger.snapshot_entries)
                p2d_pressure = self._p2d_pressure_by_domain(p2d_entries)
                reservation_totals = (
                    {}
                    if self._prefill_pressure_reservations is None
                    else await asyncio.to_thread(self._prefill_pressure_reservations.totals)
                )
                p2d_arena_capacity = int(
                    float(
                        os.getenv(
                            "SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB",
                            "32",
                        )
                    )
                    * (1024**3)
                    * tp_size
                )
                domains = []
                for domain, result in enumerate(fetched):
                    if isinstance(result, BaseException):
                        used = capacity = waiting = 0
                        rank_free = []
                    else:
                        used, capacity, waiting, rank_free = result
                    reserved_tokens, reserved_requests = reservation_totals.get(domain, (0, 0))
                    recovery_tokens, recovery_requests = recovery_reservations[domain]
                    domains.append(
                        {
                            "domain": domain,
                            "pending_tokens": pending_tokens[domain],
                            "pending_requests": pending_requests[domain],
                            "scheduler_waiting": waiting,
                            "hbm_used_tokens": used,
                            "hbm_capacity_tokens": capacity,
                            "rank_free_tokens": rank_free,
                            "min_rank_free_tokens": min(rank_free, default=0),
                            "arena_used_bytes": arena_used[domain],
                            "arena_capacity_bytes": arena_capacity,
                            "p2d_arena_capacity_bytes": p2d_arena_capacity,
                            **p2d_pressure[domain],
                            "d_slow_reserved_tokens": (reserved_tokens + recovery_tokens),
                            "d_slow_reserved_requests": (reserved_requests + recovery_requests),
                        }
                    )
                payload = {
                    "version": 1,
                    "published_at": time.time(),
                    "domains": domains,
                }
                # Publish the sample and its causal epoch together, without an
                # intervening await.  A Direct terminal may therefore either
                # keep its bridge credit or rely on this physical snapshot,
                # but selectors can never miss both.
                self._prefill_pressure_domains = domains
                self._prefill_pressure_sample_started_at = sample_started_at
                self._prefill_pressure_at = time.monotonic()
                try:
                    await asyncio.to_thread(
                        self._write_prefill_pressure,
                        self._prefill_pressure_path,
                        payload,
                    )
                except OSError:
                    logger.exception("Failed to publish Prefill pressure snapshot")
                await asyncio.sleep(self._prefill_pressure_interval)

    async def _host_capacity_monitor_loop(self) -> None:
        """Publish Host-only capacity without waiting for P HTTP samples."""

        tp_size = max(1, int(os.getenv("SGLANG_AGENTIC_KV_TP_SIZE", "1")))
        arena_capacity = int(float(os.getenv("SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB", "128")) * (1024**3) * tp_size)
        while True:
            used = await asyncio.to_thread(self._prefill_arena_bytes)
            payload = {
                "version": 1,
                "published_at": time.time(),
                "domains": [
                    {
                        "domain": domain,
                        "arena_used_bytes": used[domain],
                        "arena_capacity_bytes": arena_capacity,
                    }
                    for domain in range(len(self.prefill_urls))
                ],
            }
            try:
                await asyncio.to_thread(
                    self._write_prefill_pressure,
                    self._host_capacity_path,
                    payload,
                )
            except OSError:
                logger.exception("Failed to publish Host capacity snapshot")
            await asyncio.sleep(self._prefill_pressure_interval)

    def _ensure_prefill_pressure_monitor(self) -> None:
        if not getattr(self, "dynamic_prefill_domains", False):
            return
        if not hasattr(self, "_prefill_pressure_interval"):
            self._prefill_pressure_interval = 0.20
        if not hasattr(self, "_prefill_pressure_sample_started_at"):
            self._prefill_pressure_sample_started_at = 0.0
        if not hasattr(self, "_prefill_pressure_path"):
            self._prefill_pressure_path = self.p_ready_dir / "early-claims" / "prefill-loads.json"
        if not hasattr(self, "_host_capacity_path"):
            self._host_capacity_path = self.p_ready_dir / "early-claims" / "host-capacities.json"
        if not hasattr(self, "_d2p_host_ledger"):
            self._d2p_host_ledger = None
        task = getattr(self, "_prefill_pressure_task", None)
        if task is None or task.done():
            self._prefill_pressure_task = asyncio.create_task(self._prefill_pressure_monitor_loop())
        host_task = getattr(self, "_host_capacity_task", None)
        if host_task is None or host_task.done():
            self._host_capacity_task = asyncio.create_task(self._host_capacity_monitor_loop())

    async def _move_prefill_work(self, reservation: _PrefillWorkReservation, domain: int) -> None:
        if not 0 <= domain < len(self.prefill_urls):
            raise RuntimeError(f"invalid Prefill domain {domain}")
        if reservation.released or reservation.domain == domain:
            return
        async with self._prefill_work_lock:
            if reservation.released or reservation.domain == domain:
                return
            previous = reservation.domain
            self._prefill_pending_tokens[previous] -= reservation.tokens
            self._prefill_pending_requests[previous] -= reservation.requests
            self._prefill_pending_tokens[domain] += reservation.tokens
            self._prefill_pending_requests[domain] += reservation.requests
            if reservation.direct_workset_tokens:
                self._prefill_direct_pending_tokens[previous] -= reservation.direct_workset_tokens
                self._prefill_direct_pending_tokens[domain] += reservation.direct_workset_tokens
            reservation.domain = domain
            logger.info(
                "PD_P_WORK_MOVE from_P=%d to_P=%d tokens=%d pending_tokens=%s",
                previous,
                domain,
                reservation.tokens,
                self._prefill_pending_tokens,
            )

    async def _resize_prefill_work(self, reservation: _PrefillWorkReservation, tokens: int) -> None:
        """Correct shadow accounting when reuse falls back to recompute."""

        tokens = max(1, int(tokens))
        if reservation.released or reservation.tokens == tokens:
            return
        async with self._prefill_work_lock:
            if reservation.released or reservation.tokens == tokens:
                return
            domain = reservation.domain
            previous = reservation.tokens
            self._prefill_pending_tokens[domain] += tokens - previous
            reservation.tokens = tokens
            logger.info(
                "PD_P_WORK_RESIZE P=%d old_tokens=%d tokens=%d " "pending_tokens=%d",
                domain,
                previous,
                tokens,
                self._prefill_pending_tokens[domain],
            )

    async def _release_prefill_work(self, reservation: Optional[_PrefillWorkReservation]) -> None:
        if reservation is None or reservation.released:
            return
        cancel_host_assignment = None
        host_assignment_task = None
        async with self._prefill_work_lock:
            if reservation.released:
                return
            domain = reservation.domain
            self._prefill_pending_tokens[domain] -= reservation.tokens
            self._prefill_pending_requests[domain] -= reservation.requests
            self._prefill_direct_pending_tokens[domain] -= reservation.direct_workset_tokens
            reservation.direct_workset_tokens = 0
            reservation.released = True
            if reservation.host_assignment_pending and reservation.host_snapshot_id:
                cancel_host_assignment = (
                    reservation.host_snapshot_id,
                    reservation.reservation_id,
                )
                reservation.host_assignment_pending = False
                host_assignment_task = reservation.host_assignment_task
            if self._prefill_pending_tokens[domain] < 0 or self._prefill_pending_requests[domain] < 0:
                raise RuntimeError("Prefill shadow queue accounting underflow")
            logger.info(
                "PD_P_WORK_RELEASE P=%d tokens=%d pending_tokens=%d " "pending_requests=%d",
                domain,
                reservation.tokens,
                self._prefill_pending_tokens[domain],
                self._prefill_pending_requests[domain],
            )
        cancelled = False
        if host_assignment_task is not None:
            try:
                await self._finish_physical_control_operation(host_assignment_task)
            except asyncio.CancelledError:
                cancelled = True
        if cancel_host_assignment is not None and self._d2p_host_ledger is not None:
            snapshot_id, reservation_id = cancel_host_assignment
            try:
                await self._finish_physical_control_operation(
                    asyncio.to_thread(
                        self._d2p_host_ledger.cancel_d2p_recovery_assignment,
                        snapshot_id,
                        reservation_id=reservation_id,
                    )
                )
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    @staticmethod
    def _prefill_domain_numa_nodes(domain: int) -> set[int]:
        configured = os.getenv("SGLANG_AGENTIC_KV_PREFILL_TP_NUMA_DOMAINS", "").strip()
        if not configured:
            return {int(domain)}
        groups = configured.split(";")
        if not 0 <= int(domain) < len(groups):
            return {int(domain)}
        return {int(value) for value in groups[int(domain)].split(",") if value.strip()}

    async def _assign_host_recovery_work(
        self,
        request: dict[str, Any],
        parent,
        reservation: Optional[_PrefillWorkReservation],
        host_entry: dict[str, Any],
    ) -> Optional[_PrefillWorkReservation]:
        """Choose P and reserve its complete Slow workset atomically.

        The storage arena is already durable and is deliberately ignored by
        the capacity score.  Every candidate must fit on its weakest TP rank.
        Router shadow credit and the per-generation ledger assignment are
        committed under one lock; a failed ledger CAS rolls the shadow back.
        """

        full_workset_tokens = self._request_input_tokens(request)
        compute_tokens = self._estimated_prefill_tokens(request, host_entry.get("token_count"))
        assigned_domain = host_entry.get("recovery_prefill_domain")
        pressure_max_age = max(1.0, 5.0 * float(getattr(self, "_prefill_pressure_interval", 0.2)))
        if assigned_domain is None and time.monotonic() - self._prefill_pressure_at > pressure_max_age:
            return None

        created_here = reservation is None
        previous_state = None
        async with self._prefill_work_lock:
            if reservation is not None and reservation.released:
                return None
            previous_state = (
                None
                if reservation is None
                else (
                    reservation.domain,
                    reservation.tokens,
                    reservation.direct_workset_tokens,
                    reservation.reservation_id,
                )
            )
            rows = {int(row.get("domain", index)): row for index, row in enumerate(self._prefill_pressure_domains)}
            if assigned_domain is None:
                remaining: dict[int, int] = {}
                for domain in range(len(self.prefill_urls)):
                    row = rows.get(domain)
                    if row is None:
                        continue
                    free = int(
                        row.get(
                            "min_rank_free_tokens",
                            int(row.get("hbm_capacity_tokens", 0)) - int(row.get("hbm_used_tokens", 0)),
                        )
                    )
                    shadow = self._prefill_direct_pending_tokens[domain]
                    if reservation is not None and reservation.domain == domain:
                        shadow -= reservation.direct_workset_tokens
                    free -= max(0, shadow)
                    free -= max(0, int(row.get("d_slow_reserved_tokens", 0)))
                    if free >= full_workset_tokens:
                        remaining[domain] = free - full_workset_tokens
                if not remaining:
                    return None
                best_domain = max(remaining, key=lambda domain: (remaining[domain], -domain))
                storage_domain = int(host_entry.get("arena_domain", -1))
                storage_numa = self._prefill_domain_numa_nodes(storage_domain)
                local = [domain for domain in remaining if storage_numa & self._prefill_domain_numa_nodes(domain)]
                if local:
                    best_local = max(local, key=lambda domain: (remaining[domain], -domain))
                    capacity = max(
                        1,
                        int(rows.get(best_domain, {}).get("hbm_capacity_tokens", 1)),
                    )
                    slack = max(
                        _env_int(
                            "SGLANG_AGENTIC_KV_RESTORE_LOCALITY_SLACK_TOKENS",
                            8192,
                        ),
                        int(
                            capacity
                            * _env_float(
                                "SGLANG_AGENTIC_KV_RESTORE_LOCALITY_SLACK_FRACTION",
                                0.05,
                            )
                        ),
                    )
                    if remaining[best_local] >= remaining[best_domain] - slack:
                        best_domain = best_local
                assigned_domain = best_domain

            domain = int(assigned_domain)
            if not 0 <= domain < len(self.prefill_urls):
                raise RuntimeError(f"invalid Slow recovery P{domain}")
            if reservation is None:
                reservation = _PrefillWorkReservation(
                    domain=domain,
                    tokens=compute_tokens,
                    direct_workset_tokens=full_workset_tokens,
                    reservation_id=uuid.uuid4().hex,
                )
                self._prefill_pending_tokens[domain] += compute_tokens
                self._prefill_pending_requests[domain] += 1
                self._prefill_direct_pending_tokens[domain] += full_workset_tokens
            else:
                previous = reservation.domain
                if previous != domain:
                    self._prefill_pending_tokens[previous] -= reservation.tokens
                    self._prefill_pending_requests[previous] -= reservation.requests
                    self._prefill_direct_pending_tokens[previous] -= reservation.direct_workset_tokens
                    self._prefill_pending_tokens[domain] += reservation.tokens
                    self._prefill_pending_requests[domain] += reservation.requests
                    self._prefill_direct_pending_tokens[domain] += reservation.direct_workset_tokens
                    reservation.domain = domain
                delta = full_workset_tokens - reservation.direct_workset_tokens
                self._prefill_direct_pending_tokens[domain] += delta
                reservation.direct_workset_tokens = full_workset_tokens
                if reservation.tokens != compute_tokens:
                    self._prefill_pending_tokens[domain] += compute_tokens - reservation.tokens
                    reservation.tokens = compute_tokens

            assignment_id = str(
                host_entry.get("recovery_reservation_id") or reservation.reservation_id or uuid.uuid4().hex
            )
            reservation.reservation_id = assignment_id
            # Publish cleanup identity before starting the non-cancellable
            # ledger CAS.  If the request is cancelled while the worker
            # thread is still committing, cleanup can join that operation and
            # cancel the exact assignment instead of leaking an orphan lease.
            reservation.host_snapshot_id = parent.snapshot_id
            reservation.host_assignment_pending = True
            reservation.route_pending = False

        assignment_task = asyncio.create_task(
            asyncio.to_thread(
                self._d2p_host_ledger.assign_d2p_recovery,
                parent.snapshot_id,
                prefill_domain=domain,
                reservation_id=assignment_id,
                workset_tokens=full_workset_tokens,
                lease_seconds=_env_float(
                    "SGLANG_AGENTIC_KV_RECOVERY_ASSIGNMENT_TTL_S",
                    120.0,
                ),
            )
        )
        reservation.host_assignment_task = assignment_task
        try:
            committed = await self._finish_physical_control_operation(assignment_task)
            # Finalization belongs to the same cancellation-safe transaction
            # as the ledger CAS.  A cancellation while waiting for this lock
            # must still remove the exact ledger assignment and all shadow
            # capacity before it can escape to the caller.
            async with self._prefill_work_lock:
                if reservation.host_assignment_task is assignment_task:
                    reservation.host_assignment_task = None
                if reservation.released:
                    return None
                if committed is None:
                    # Eviction or another Router retry won the generation CAS.
                    # Roll back exactly the accounting changed above.
                    if created_here:
                        self._prefill_pending_tokens[domain] -= reservation.tokens
                        self._prefill_pending_requests[domain] -= reservation.requests
                        self._prefill_direct_pending_tokens[domain] -= reservation.direct_workset_tokens
                        reservation.released = True
                    else:
                        previous_domain, previous_tokens, previous_direct, previous_id = previous_state
                        self._prefill_pending_tokens[domain] -= reservation.tokens
                        self._prefill_pending_requests[domain] -= reservation.requests
                        self._prefill_direct_pending_tokens[domain] -= reservation.direct_workset_tokens
                        self._prefill_pending_tokens[previous_domain] += previous_tokens
                        self._prefill_pending_requests[previous_domain] += reservation.requests
                        self._prefill_direct_pending_tokens[previous_domain] += previous_direct
                        reservation.domain = previous_domain
                        reservation.tokens = previous_tokens
                        reservation.direct_workset_tokens = previous_direct
                        reservation.reservation_id = previous_id
                    reservation.host_snapshot_id = None
                    reservation.host_assignment_pending = False
                    return None
                # Start the accounting handoff only after the assignment CAS
                # is globally visible.  A monitor sample that began before
                # this point may have read the old ledger and cannot replace
                # the Router shadow credit safely.
                committed_at = time.monotonic()
                asyncio.create_task(
                    self._settle_direct_workset_after_pressure(
                        reservation,
                        direct_terminal_at=committed_at,
                    )
                )
                return reservation
        except BaseException:
            # The physical task is attached to the reservation.  Release
            # joins it before cancelling by reservation_id, so neither a
            # running CAS nor post-CAS lock cancellation can leak ownership.
            await self._release_prefill_work(reservation)
            raise

    def _bind_prefill_domain(self, request: dict[str, Any], domain: int) -> str:
        prefill_server = self.prefill_urls[domain]
        parsed = urllib.parse.urlparse(prefill_server)
        hostname = parsed.hostname
        if hostname is None:
            raise ValueError(f"Invalid P URL {prefill_server!r}")
        port = self.prefill_bootstrap_ports[domain]
        batch = isinstance(request.get("bootstrap_room"), list)
        request["bootstrap_host"] = [hostname] * len(request["bootstrap_room"]) if batch else hostname
        request["bootstrap_port"] = [port] * len(request["bootstrap_room"]) if batch else port
        return prefill_server

    def _prefill_admission_for_domain(self, domain: int) -> _PrefillAdmissionGate:
        """Return the independent HTTP/bootstrap admission gate for one P."""
        gates = getattr(self, "_prefill_admissions", None)
        if gates is None:
            # Compatibility with tests and embedders that build the router via
            # __new__ and provide the historical single gate explicitly.
            return self._prefill_admission
        if domain < 0 or domain >= len(gates):
            raise ValueError(f"Invalid Prefill domain P{domain}")
        return gates[domain]

    def _domain_decode_urls(self, domain: int) -> set[str]:
        if not getattr(self, "numa_domains", False) or getattr(self, "global_decode", False):
            return set(self.decode_urls)
        width = len(self.decode_urls) // len(self.prefill_urls)
        return set(self.decode_urls[domain * width : (domain + 1) * width])

    @staticmethod
    def _set_p2d_host_metadata(request: dict[str, Any], snapshot_id: str, prefill_domain: int) -> None:
        values = {
            P2D_CUSTOM_SNAPSHOT_ID: str(snapshot_id),
            P2D_CUSTOM_PREFILL_DOMAIN: int(prefill_domain),
        }
        # /generate carries custom metadata inside sampling_params, whereas
        # /v1/chat/completions carries it at the request top level.  Do not
        # manufacture the other protocol's container: unknown nested fields
        # can be silently discarded by the Chat schema, which previously made
        # D construct a native NIXL receiver for a snapshot already released
        # by P into the Host arena.
        sampling = request.get("sampling_params")
        if isinstance(sampling, dict):
            custom = dict(sampling.get("custom_params") or {})
            custom.update(values)
            sampling["custom_params"] = custom
            return
        custom = dict(request.get("custom_params") or {})
        custom.update(values)
        request["custom_params"] = custom

    def _p2d_snapshot_for_rooms(self, rooms: tuple[int, ...]) -> Optional[str]:
        """Return the P->D Host identity supported by the current request."""

        if getattr(self, "p2d_host_ledger", None) is None or len(rooms) != 1:
            return None
        return p2d_snapshot_id(rooms[0])

    def _publish_p2d_host_offer(
        self,
        snapshot_id: str,
        rooms: tuple[int, ...],
        prompt_tokens: int,
        domain: int,
        *,
        source: str = "backpressure",
    ) -> str:
        """Publish one idempotent P->D Host offer without scheduling delay."""

        offered = self.p2d_host_ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": int(rooms[0]),
                "token_count": int(prompt_tokens),
                "prefill_domain": int(domain),
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": int(os.getenv("SGLANG_AGENTIC_KV_TP_SIZE", "1")),
            }
        )
        # The Router is the sole publisher of P->D control offers.  Keep a
        # process-local index so the hot Direct-admission path can prove that
        # no Host owner exists without decoding the complete shared ledger for
        # every P-ready request.  Once present, an id is never removed: the
        # ledger tombstone remains the physical ownership authority.
        if not hasattr(self, "_p2d_host_offered_snapshots"):
            self._p2d_host_offered_snapshots = set()
        self._p2d_host_offered_snapshots.add(snapshot_id)
        logger.info(
            "PD_P2D_HOST_OFFER snapshot=%s rooms=%s P=%d " "prompt_tokens=%d state=%s source=%s",
            snapshot_id,
            rooms,
            domain,
            prompt_tokens,
            offered.get("state"),
            source,
        )
        return snapshot_id

    async def _stage_p2d_until_durable(
        self,
        rooms: tuple[int, ...],
        prompt_tokens: int,
        domain: int,
    ) -> bool:
        """Move one blocked P-ready generation under durable Host ownership.

        This is the producer half of the P->D pipeline.  It never waits for or
        reserves D HBM: once the complete Host snapshot is committed, P may
        release its source pages and the ordered consumer will admit it to D
        later.  Returning ``False`` means Host staging is unavailable for this
        request shape and the caller must retain P HBM while waiting for D.
        """

        snapshot_id = self._p2d_snapshot_for_rooms(rooms)
        if snapshot_id is None:
            return False
        await self._finish_physical_control_operation(
            asyncio.to_thread(
                self._publish_p2d_host_offer,
                snapshot_id,
                rooms,
                prompt_tokens,
                domain,
                source="p_ready_pipeline",
            )
        )

        deadline = time.monotonic() + self.ready_timeout
        while True:
            entry = await asyncio.to_thread(self.p2d_host_ledger.get, snapshot_id)
            state = None if entry is None else entry.get("state")
            if state in {
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
                HostStageState.CONSUMED.value,
            }:
                return True
            if state == HostStageState.REJECTED.value:
                # No Host writer ever took ownership.  This is the explicit
                # RETAIN_P result used when one TP producer cannot reserve its
                # local Arena extent (or native admission wins the offer).
                # The caller still owns the complete P KV and may safely use
                # the ordinary Direct/D-capacity path.
                return False
            if state in {
                HostStageState.ABORTING.value,
                HostStageState.FAILED.value,
            }:
                reason = None if entry is None else entry.get("reason")
                raise RuntimeError(
                    f"P->D Host staging terminated for {snapshot_id}: " f"state={state} reason={reason}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for durable P->D Host snapshot " f"{snapshot_id}")
            await asyncio.sleep(self.ready_poll_interval)

    async def _retry_p_ready_direct_after_grace(
        self, admission: _PReadyAdmission, *, not_before: float
    ) -> Optional[DecodeReservation]:
        """Retry one P-ready generation against a causally fresh D snapshot.

        The P->D grace period is useful only if capacity released during that
        interval can win before Host ownership is published.  Keep this retry
        outside the physical staging lane and charge any winning reservation
        under the same selection lock as the initial broker admission.
        """

        urls = self._domain_decode_urls(admission.domain)
        if not await self._observe_decode_load_after(self._load_http_session(), urls=urls, not_before=not_before):
            return None
        loads = list(self._load_cache)
        async with self._selection_lock:
            if admission.cancel_requested:
                return None
            self._prune_accounted_reservations()
            return self._try_reserve_direct_ready_locked(admission, loads)

    def _abort_unsubmitted_p2d(self, snapshot_id: Optional[str], reason: str) -> None:
        """Return attempt-owned P->D storage to its physical owner safely."""

        if snapshot_id is None or getattr(self, "p2d_host_ledger", None) is None:
            return
        state = self.p2d_host_ledger.abort_unsubmitted_p2d(snapshot_id, reason=reason)
        logger.info(
            "PD_P2D_HOST_ABORT snapshot=%s state=%s reason=%s",
            snapshot_id,
            state,
            reason,
        )

    async def _resolve_dynamic_prefill_work(
        self,
        request: dict[str, Any],
        metadata: Optional[AgenticRequestMetadata],
        arrival_at: Optional[float],
    ) -> _PrefillWorkReservation:
        """Choose and account P before publishing physical Direct admission.

        :meth:`_late_dispatch` publishes the targeted arrival immediately
        after this choice.  P's exact-size workset allocator, rather than the
        short HTTP/tokenizer admission gate, is the authority that decides
        whether the parent+suffix KV can occupy HBM.
        """

        parent = None if metadata is None else metadata.parent
        store = self.early_claim_store
        if parent is None or store is None:
            return await self._reserve_prefill_work(self._estimated_prefill_tokens(request))

        deadline = time.monotonic() + self.ready_timeout
        reservation: Optional[_PrefillWorkReservation] = None
        try:
            while True:
                route = store.read_route(
                    parent,
                    max_age_seconds=max(
                        self.ready_timeout,
                        getattr(self, "generation_result_ttl", 3600.0),
                    ),
                )
                if route is None or route.get("route") == "direct_ready":
                    # Route files are wakeup hints, not ownership authority.
                    # If D committed Shared Host but its best-effort marker was
                    # lost, recover the Slow route from this generation's
                    # exact ledger record so D can release its durable source.
                    host_ledger = getattr(self, "_d2p_host_ledger", None)
                    host_entry = (
                        None
                        if host_ledger is None
                        else await asyncio.to_thread(
                            host_ledger.get, parent.snapshot_id
                        )
                    )
                    if host_entry is not None and host_entry.get("state") in {
                        HostStageState.HOST_WRITING.value,
                        HostStageState.HOST_READY.value,
                    }:
                        route = {
                            "route": "host_writing",
                            "prefill_domain": int(
                                host_entry.get("arena_domain", 0)
                            ),
                            "snapshot_tokens": host_entry.get("token_count"),
                        }
                if route is not None:
                    mode = route.get("route")
                    snapshot_tokens = route.get("snapshot_tokens")
                    if mode == "direct_ready":
                        if reservation is None:
                            reservation = self._reserve_direct_prefill_work_nowait(
                                self._estimated_prefill_tokens(request, snapshot_tokens),
                                direct_workset_tokens=self._request_input_tokens(request),
                            )
                            logger.info(
                                "PD_PREFILL_ROUTE snapshot=%s route=%s " "selected_P=%d estimated_tokens=%d",
                                parent.snapshot_id,
                                mode,
                                reservation.domain,
                                reservation.tokens,
                            )
                        reservation.route_pending = True
                        return reservation
                    elif mode in {"direct_complete", "host_writing", "host_ready"}:
                        if mode in {"host_writing", "host_ready"}:
                            # D has irreversibly left Direct admission.  Drop
                            # the old P shadow before selecting a Slow
                            # recovery destination; otherwise stale Direct
                            # debt can make every P look full while their
                            # physical HBM is mostly empty.
                            await self._settle_direct_workset(reservation)
                            host_ledger = getattr(self, "_d2p_host_ledger", None)
                            host_entry = (
                                None
                                if host_ledger is None
                                else await asyncio.to_thread(host_ledger.get, parent.snapshot_id)
                            )
                            host_state = None if host_entry is None else host_entry.get("state")
                            if _HOST_STAGE_EVICTING is not None and host_state == _HOST_STAGE_EVICTING.value:
                                await asyncio.sleep(self.ready_poll_interval)
                                continue
                            if (
                                _HOST_STAGE_RECOMPUTE_REQUIRED is not None
                                and host_state == _HOST_STAGE_RECOMPUTE_REQUIRED.value
                            ):
                                store.publish_route(
                                    parent,
                                    route="recompute",
                                    prefill_domain=int(route["prefill_domain"]),
                                )
                                store.remove_arrival(parent)
                                await self._release_prefill_work(reservation)
                                return await self._reserve_prefill_work(self._request_input_tokens(request))
                            if host_state != HostStageState.HOST_READY.value:
                                await asyncio.sleep(self.ready_poll_interval)
                                continue
                            reservation = await self._assign_host_recovery_work(
                                request,
                                parent,
                                reservation,
                                host_entry,
                            )
                            if reservation is None:
                                await asyncio.sleep(self.ready_poll_interval)
                                continue
                            domain = reservation.domain
                            store.publish_route(
                                parent,
                                route="host_ready",
                                prefill_domain=domain,
                                arena_numa_node=int(host_entry.get("arena_numa_node", -1)),
                                snapshot_tokens=host_entry.get("token_count"),
                            )
                            logger.info(
                                "PD_PREFILL_ROUTE snapshot=%s route=host_ready "
                                "storage_P=%s recovery_P=%d estimated_tokens=%d",
                                parent.snapshot_id,
                                host_entry.get("arena_domain"),
                                domain,
                                reservation.tokens,
                            )
                            return reservation
                        domain = int(route["prefill_domain"])
                        if not 0 <= domain < len(self.prefill_urls):
                            raise RuntimeError(f"invalid Prefill domain {domain}")
                        if reservation is None:
                            reservation = self._reserve_direct_prefill_work_nowait(
                                self._estimated_prefill_tokens(request, snapshot_tokens),
                                domain=domain,
                                direct_workset_tokens=self._request_input_tokens(request),
                                # D/P already completed the ownership handoff.
                                # Keep exact Router shadow until a causally
                                # newer physical pressure sample observes it.
                                already_committed=True,
                            )
                        else:
                            await self._move_prefill_work(reservation, domain)
                        logger.info(
                            "PD_PREFILL_ROUTE snapshot=%s route=%s P=%d " "estimated_tokens=%d",
                            parent.snapshot_id,
                            mode,
                            domain,
                            reservation.tokens,
                        )
                        if mode == "direct_complete":
                            reservation.route_pending = True
                        return reservation
                    elif mode == "recompute":
                        store.remove_arrival(parent)
                        await self._release_prefill_work(reservation)
                        return await self._reserve_prefill_work(self._request_input_tokens(request))
                if store.read_final(parent, not_before=0.0, max_age_seconds=self.ready_timeout) is not None:
                    store.remove_arrival(parent)
                    await self._release_prefill_work(reservation)
                    return await self._reserve_prefill_work(self._request_input_tokens(request))
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for P route of {parent.snapshot_id}")
                await asyncio.sleep(self.ready_poll_interval)
        except BaseException:
            await self._release_prefill_work(reservation)
            raise

    async def _watch_dynamic_prefill_route(
        self,
        request: dict[str, Any],
        metadata: AgenticRequestMetadata,
        reservation: _PrefillWorkReservation,
    ) -> dict[str, Any]:
        """Observe the Direct outcome without delaying the first P submit."""

        parent = metadata.parent
        store = self.early_claim_store
        if parent is None or store is None:
            return {"action": "settled", "route": "none"}
        deadline = time.monotonic() + self.ready_timeout
        while True:
            route = store.read_route(
                parent,
                max_age_seconds=max(
                    self.ready_timeout,
                    getattr(self, "generation_result_ttl", 3600.0),
                ),
            )
            if route is None or route.get("route") == "direct_ready":
                host_ledger = getattr(self, "_d2p_host_ledger", None)
                host_entry = (
                    None
                    if host_ledger is None
                    else await asyncio.to_thread(
                        host_ledger.get, parent.snapshot_id
                    )
                )
                if host_entry is not None and host_entry.get("state") in {
                    HostStageState.HOST_WRITING.value,
                    HostStageState.HOST_READY.value,
                }:
                    route = {
                        "route": "host_writing",
                        "prefill_domain": int(host_entry.get("arena_domain", 0)),
                        "snapshot_tokens": host_entry.get("token_count"),
                    }
            if route is not None:
                mode = route.get("route")
                if mode in {"direct_complete", "host_writing", "host_ready"}:
                    if mode in {"host_writing", "host_ready"}:
                        # Direct and Slow own disjoint admission phases.  Once
                        # D publishes Host fallback, its Direct shadow must no
                        # longer participate in Slow capacity selection.
                        await self._settle_direct_workset(reservation)
                        host_ledger = getattr(self, "_d2p_host_ledger", None)
                        host_entry = (
                            None
                            if host_ledger is None
                            else await asyncio.to_thread(host_ledger.get, parent.snapshot_id)
                        )
                        host_state = None if host_entry is None else host_entry.get("state")
                        if _HOST_STAGE_EVICTING is not None and host_state == _HOST_STAGE_EVICTING.value:
                            await asyncio.sleep(self.ready_poll_interval)
                            continue
                        if (
                            _HOST_STAGE_RECOMPUTE_REQUIRED is not None
                            and host_state == _HOST_STAGE_RECOMPUTE_REQUIRED.value
                        ):
                            store.publish_route(
                                parent,
                                route="recompute",
                                prefill_domain=int(route["prefill_domain"]),
                            )
                            store.remove_arrival(parent)
                            await self._settle_direct_workset(reservation)
                            await self._resize_prefill_work(reservation, self._request_input_tokens(request))
                            return {"action": "recompute", "route": "host_evicted"}
                        if host_state != HostStageState.HOST_READY.value:
                            await asyncio.sleep(self.ready_poll_interval)
                            continue
                        previous_domain = reservation.domain
                        assigned = await self._assign_host_recovery_work(
                            request,
                            parent,
                            reservation,
                            host_entry,
                        )
                        if assigned is None:
                            await asyncio.sleep(self.ready_poll_interval)
                            continue
                        domain = assigned.domain
                        store.publish_route(
                            parent,
                            route="host_ready",
                            prefill_domain=domain,
                            arena_numa_node=int(host_entry.get("arena_numa_node", -1)),
                            snapshot_tokens=host_entry.get("token_count"),
                        )
                        return {
                            "action": ("redirect" if domain != previous_domain else "settled"),
                            "route": "host_ready",
                            "domain": domain,
                        }
                    domain = int(route["prefill_domain"])
                    if not 0 <= domain < len(self.prefill_urls):
                        raise RuntimeError(f"invalid Prefill domain {domain}")
                    if mode == "direct_complete" and domain == reservation.domain:
                        await self._settle_direct_workset_after_pressure(
                            reservation,
                            direct_terminal_at=time.monotonic(),
                        )
                    else:
                        # Host fallback has no physical allocation on the
                        # originally selected P.  A cross-domain terminal also
                        # cannot hand that domain's shadow to this reservation.
                        await self._settle_direct_workset(reservation)
                    if domain != reservation.domain:
                        logger.info(
                            "PD_PREFILL_REDIRECT snapshot=%s route=%s " "from_P=%d to_P=%d",
                            parent.snapshot_id,
                            mode,
                            reservation.domain,
                            domain,
                        )
                        return {
                            "action": "redirect",
                            "route": mode,
                            "domain": domain,
                        }
                    return {"action": "settled", "route": mode, "domain": domain}
                if mode == "recompute":
                    store.remove_arrival(parent)
                    await self._settle_direct_workset(reservation)
                    await self._resize_prefill_work(reservation, self._request_input_tokens(request))
                    return {"action": "recompute", "route": mode}
            if store.read_final(parent, not_before=0.0, max_age_seconds=self.ready_timeout) is not None:
                store.remove_arrival(parent)
                await self._settle_direct_workset(reservation)
                await self._resize_prefill_work(reservation, self._request_input_tokens(request))
                return {"action": "recompute", "route": "final"}
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for P route of {parent.snapshot_id}")
            await asyncio.sleep(self.ready_poll_interval)

    @staticmethod
    def _raise_prefill_redirect(route_task: Optional[asyncio.Task]) -> None:
        if route_task is None or not route_task.done():
            return
        outcome = route_task.result()
        if outcome.get("action") == "redirect":
            raise _PrefillRedirect(int(outcome["domain"]), str(outcome["route"]))

    @staticmethod
    def _set_prefill_attempt_rid(request: dict[str, Any], *, replace: bool) -> None:
        rid = request.get("rid")
        rooms = request.get("bootstrap_room")
        batch = isinstance(rooms, list)
        if batch:
            size = len(rooms)
            if replace or rid is None:
                request["rid"] = [uuid.uuid4().hex for _ in range(size)]
            elif isinstance(rid, list):
                if len(rid) != size:
                    raise ValueError("rid/bootstrap_room batch size mismatch")
            else:
                # Mirror GenerateReqInput._normalize_rid(): a scalar batch rid
                # is a supported shorthand, not a malformed request.
                request["rid"] = [f"{rid}_{index}" for index in range(size)]
            return
        if isinstance(rid, list):
            raise ValueError("rid list requires batched bootstrap_room")
        if replace or rid is None:
            request["rid"] = uuid.uuid4().hex

    @staticmethod
    def _replace_prefill_attempt_rooms(request: dict[str, Any]) -> None:
        room = request.get("bootstrap_room")
        if isinstance(room, list):
            request["bootstrap_room"] = [uuid.uuid4().int & ((1 << 63) - 1) for _ in room]
        else:
            request["bootstrap_room"] = uuid.uuid4().int & ((1 << 63) - 1)

    @staticmethod
    def _attempt_rids(request: dict[str, Any], rooms: tuple[int, ...]) -> tuple[str, ...]:
        value = request.get("rid")
        if isinstance(value, list):
            if len(value) != len(rooms):
                raise ValueError("rid/bootstrap_room batch size mismatch")
            return tuple(str(item) for item in value)
        if len(rooms) != 1 or value is None:
            raise ValueError("missing rid for Prefill attempt")
        return (str(value),)

    def _activate_prefill_attempt(self, request: dict[str, Any], rooms: tuple[int, ...]) -> None:
        owners = getattr(self, "_active_prefill_attempts", None)
        if owners is None:
            owners = {}
            self._active_prefill_attempts = owners
        claims = tuple(zip(rooms, self._attempt_rids(request, rooms)))
        # Validate the complete TP group before publishing any ownership. A
        # duplicate room must fail atomically rather than stealing one rank of
        # an already-live attempt.
        pending: dict[int, str] = {}
        for room, rid in claims:
            room = int(room)
            existing = owners.get(room, pending.get(room))
            if existing is not None and existing != rid:
                raise RuntimeError(f"bootstrap_room {room} already owned by rid={existing}")
            pending[room] = rid
        owners.update(pending)

    def _deactivate_prefill_attempt(self, request: dict[str, Any], rooms: tuple[int, ...]) -> None:
        owners = getattr(self, "_active_prefill_attempts", None)
        if owners is None:
            return
        for room, rid in zip(rooms, self._attempt_rids(request, rooms)):
            if owners.get(int(room)) == rid:
                owners.pop(int(room), None)

    def _p_ready_marker_is_owned(self, room: int, payload: dict[str, Any]) -> bool:
        owners = getattr(self, "_active_prefill_attempts", None)
        # Compatibility for focused tests/embedders constructed via __new__.
        if owners is None:
            return True
        expected_rid = owners.get(int(room))
        return expected_rid is not None and str(payload.get("rid", "")) == expected_rid

    async def _abort_prefill_attempt(
        self,
        session: aiohttp.ClientSession,
        prefill_server: str,
        request: dict[str, Any],
        prefill_task: Optional[asyncio.Task],
    ) -> bool:
        rids = request.get("rid")
        if not isinstance(rids, list):
            rids = [] if rids is None else [rids]
        aborted = bool(rids)
        for rid in rids:
            try:
                response = await session.post(f"{prefill_server}/abort_request", json={"rid": rid})
                if response.status >= 400:
                    aborted = False
                    logger.warning(
                        "PD_PREFILL_REDIRECT abort failed P=%s rid=%s status=%d",
                        prefill_server,
                        rid,
                        response.status,
                    )
                release = getattr(response, "release", None)
                if release is not None:
                    release()
            except Exception:
                aborted = False
                logger.exception(
                    "PD_PREFILL_REDIRECT abort request failed P=%s rid=%s",
                    prefill_server,
                    rid,
                )
        await self._dispose_response_task(prefill_task)
        return aborted

    async def _abort_decode_attempt(
        self,
        session: aiohttp.ClientSession,
        decode_server: str,
        request: dict[str, Any],
    ) -> bool:
        """Cancel one submitted D attempt before releasing Router ownership.

        Cancelling the aiohttp response task alone only closes the HTTP client;
        it does not guarantee that SGLang removes a request which is still in
        Decode prealloc or transfer.  Explicitly abort every TP request id so
        D can fence/clear its receiver and return any destination allocation.
        """

        rids = request.get("rid")
        if not isinstance(rids, list):
            rids = [] if rids is None else [rids]
        aborted = bool(rids)
        for rid in rids:
            try:
                response = await session.post(f"{decode_server}/abort_request", json={"rid": rid})
                if response.status >= 400:
                    aborted = False
                    logger.warning(
                        "PD_DECODE_ABORT failed D=%s rid=%s status=%d",
                        decode_server,
                        rid,
                        response.status,
                    )
                release = getattr(response, "release", None)
                if release is not None:
                    release()
            except Exception:
                aborted = False
                logger.exception(
                    "PD_DECODE_ABORT request failed D=%s rid=%s",
                    decode_server,
                    rid,
                )
        return aborted

    @staticmethod
    async def _dispose_response_task(task: Optional[asyncio.Task]) -> None:
        """Cancel an HTTP task or return its completed response to the pool."""

        if task is None:
            return
        if not task.done():
            task.cancel()
        results = await asyncio.gather(task, return_exceptions=True)
        if results and isinstance(results[0], aiohttp.ClientResponse):
            results[0].release()

    @staticmethod
    async def _finish_physical_control_operation(awaitable: Awaitable[Any]) -> Any:
        """Join a non-cancellable ledger write before propagating cancellation."""

        task = asyncio.ensure_future(awaitable)
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                # ``to_thread`` keeps running.  Cleanup must observe its final
                # state, so defer logical cancellation until it is joined.
                cancelled = True
                continue
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _scheduled_path(self, room: int) -> Path:
        return self.p_ready_dir / f"{room}.scheduled"

    async def _wait_until_prefill_accepted(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> None:
        deadline = time.monotonic() + self.prefill_accept_timeout
        while True:
            self._raise_prefill_redirect(route_task)
            paths = [self._accepted_path(room) for room in rooms]
            if all(path.exists() for path in paths):
                for path in paths:
                    path.unlink(missing_ok=True)
                return
            if prefill_task.done():
                response = prefill_task.result()
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        "Prefill failed before scheduler acceptance: " f"status={response.status} body={body}"
                    )
            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(
                    "Timed out waiting " f"{self.prefill_accept_timeout}s for P-accepted rooms {missing}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    async def _wait_until_prefill_scheduled(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> None:
        deadline = time.monotonic() + self.prefill_queue_timeout
        while True:
            self._raise_prefill_redirect(route_task)
            paths = [self._scheduled_path(room) for room in rooms]
            if all(path.exists() for path in paths):
                for path in paths:
                    path.unlink(missing_ok=True)
                return
            if prefill_task.done():
                response = prefill_task.result()
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError("Prefill failed while queued: " f"status={response.status} body={body}")
            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(
                    "Timed out waiting " f"{self.prefill_queue_timeout}s in P queue for rooms {missing}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    def _publish_parent_arrival(
        self,
        request: dict[str, Any],
        *,
        target_prefill_domain: Optional[int] = None,
        arrived_at: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Publish next-turn arrival before P's GPU scheduler sees the request."""

        store = self.early_claim_store
        if store is None:
            return None
        sampling = request.get("sampling_params")
        custom_params = sampling.get("custom_params") if isinstance(sampling, dict) else None
        try:
            metadata = AgenticRequestMetadata.from_custom_params(custom_params)
            if metadata is None:
                envelope = unpack_agentic_extra_key(request.get("extra_key"))
                if envelope is not None:
                    _, custom_params = envelope
                    metadata = AgenticRequestMetadata.from_custom_params(custom_params)
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed agentic early-claim metadata")
            return None
        if metadata is None or metadata.parent is None:
            return None
        try:
            payload = store.publish_arrival(
                metadata.parent,
                target_prefill_domain=target_prefill_domain,
                prompt_token_count=self._request_input_tokens(request),
                arrived_at=arrived_at,
            )
        except OSError:
            # Early claim is a performance hint.  A marker write failure must
            # not turn an otherwise valid serving request into an HTTP error.
            logger.exception(
                "Failed to publish agentic early-claim marker for snapshot=%s",
                metadata.parent.snapshot_id,
            )
            return None
        logger.info(
            "PD_EARLY_CLAIM_ARRIVAL snapshot=%s generation=%d arrived_at=%.6f " "prompt_tokens=%d P=%s",
            metadata.parent.snapshot_id,
            metadata.generation,
            payload["arrived_at"],
            payload["prompt_token_count"],
            target_prefill_domain,
        )
        return payload

    @staticmethod
    def _agentic_generation_key(request: dict[str, Any]) -> Optional[str]:
        sampling = request.get("sampling_params")
        custom_params = sampling.get("custom_params") if isinstance(sampling, dict) else None
        try:
            metadata = AgenticRequestMetadata.from_custom_params(custom_params)
            if metadata is None:
                envelope = unpack_agentic_extra_key(request.get("extra_key"))
                if envelope is not None:
                    _, params = envelope
                    metadata = AgenticRequestMetadata.from_custom_params(params)
        except (TypeError, ValueError):
            return None
        return None if metadata is None else metadata.current.snapshot_id

    def _ensure_generation_dedup_state(self) -> None:
        """Support focused tests that construct the router via ``__new__``."""

        if not hasattr(self, "_generation_lock"):
            self._generation_lock = asyncio.Lock()
            self._generation_tasks = {}
            self._generation_results = {}
            self.generation_result_ttl = 3600.0
            self.max_generation_results = 8192

    def _prune_generation_results(self, now: float) -> None:
        expired = [
            key
            for key, result in self._generation_results.items()
            if now - result.completed_at > self.generation_result_ttl
        ]
        for key in expired:
            self._generation_results.pop(key, None)
        overflow = len(self._generation_results) - self.max_generation_results
        if overflow > 0:
            oldest = sorted(
                self._generation_results.items(),
                key=lambda item: item[1].completed_at,
            )[:overflow]
            for key, _ in oldest:
                self._generation_results.pop(key, None)

    async def _record_generation_task(
        self,
        key: str,
        task: asyncio.Task[tuple[dict[str, Any], int]],
    ) -> None:
        try:
            payload, status = task.result()
        except BaseException:
            async with self._generation_lock:
                if self._generation_tasks.get(key) is task:
                    self._generation_tasks.pop(key, None)
            return
        async with self._generation_lock:
            if self._generation_tasks.get(key) is task:
                self._generation_tasks.pop(key, None)
                # Cache only successful logical generations. A transport 5xx
                # is an attempt failure, not the result of the agent turn. If
                # cached, all later HTTP retries replay the same stale wire
                # generation and can never recover after D has torn it down.
                if status < 400:
                    self._generation_results[key] = _GenerationResponse(
                        payload=payload,
                        status=status,
                        completed_at=time.monotonic(),
                    )
                    self._prune_generation_results(time.monotonic())

    async def _generate_once(
        self,
        modified_request: dict[str, Any],
        prefill_server: str,
        endpoint: str,
    ) -> tuple[dict[str, Any], int]:
        session = self._backend_http_session()
        prefill_response, decode_response = await self._late_dispatch(
            session, modified_request, prefill_server, endpoint, {}
        )
        try:
            if modified_request.get("return_logprob", False):
                assert prefill_response is not None
                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()
                if "meta_info" in ret_json and "input_token_logprobs" in ret_json["meta_info"]:
                    ret_json["meta_info"]["input_token_logprobs"] = (
                        prefill_json["meta_info"]["input_token_logprobs"]
                        + ret_json["meta_info"]["input_token_logprobs"]
                    )
            else:
                ret_json = await decode_response.json()
            return ret_json, decode_response.status
        finally:
            if prefill_response is not None:
                prefill_response.release()
            decode_response.release()

    def _backend_http_session(self) -> aiohttp.ClientSession:
        """Return the Router-owned backend pool without an await-time race."""

        session = getattr(self, "_backend_session", None)
        if session is None or session.closed:
            # Construction contains no await, so event-loop tasks cannot both
            # install different sessions between the check and assignment.
            connector = aiohttp.TCPConnector(
                limit=_env_int("SGLANG_PD_ROUTER_HTTP_CONNECTION_LIMIT", 2048),
                limit_per_host=_env_int("SGLANG_PD_ROUTER_HTTP_CONNECTION_LIMIT_PER_HOST", 512),
                # The generation path can have hundreds of concurrent P/D
                # requests.  Reusing a bounded pool is essential: forcing one
                # new socket per turn creates an FD-close/reuse storm in
                # uvloop at c512.  The launch script keeps the SGLang server
                # timeout longer than this client-side idle lifetime.
                keepalive_timeout=_env_float("SGLANG_PD_ROUTER_HTTP_KEEPALIVE_S", 30.0),
            )
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
            self._backend_session = session
        return session

    def _load_http_session(self) -> aiohttp.ClientSession:
        """Return the persistent, generation-independent load-control pool."""

        session = getattr(self, "_load_session", None)
        if session is None or session.closed:
            connector = aiohttp.TCPConnector(
                limit=_env_int("SGLANG_PD_ROUTER_LOAD_CONNECTION_LIMIT", 32),
                limit_per_host=_env_int("SGLANG_PD_ROUTER_LOAD_CONNECTION_LIMIT_PER_HOST", 4),
                # SGLang's scheduler-backed load endpoint can close an idle
                # keep-alive socket while saturated.  A later reuse then
                # spends the whole control timeout on a dead transport and
                # leaves Router with a stale full-D view.  Load replies are
                # tiny and loopback-local, so use one fresh short connection
                # per sample while retaining one bounded connector/session.
                force_close=True,
            )
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.load_timeout),
            )
            self._load_session = session
        return session

    async def close(self) -> None:
        """Close Router-owned HTTP resources during ASGI shutdown."""

        for waiters in getattr(self, "_p_ready_fifo_waiters", {}).values():
            for admission in waiters.values():
                if not admission.future.done():
                    admission.future.cancel()
                admission.finished.set()
        for task in getattr(self, "_p_ready_fifo_dispatchers", {}).values():
            task.cancel()
        broker = getattr(self, "_p_ready_broker_task", None)
        if broker is not None:
            broker.cancel()
        active_tasks: list[asyncio.Task] = []
        for active in getattr(self, "_p_ready_fifo_active", {}).values():
            for admission in active.values():
                if not admission.future.done():
                    admission.future.cancel()
                if admission.dispatch_task is not None:
                    admission.dispatch_task.cancel()
                    active_tasks.append(admission.dispatch_task)
        monitor = getattr(self, "_p_ready_monitor_task", None)
        if monitor is not None:
            monitor.cancel()
        tasks = list(getattr(self, "_p_ready_fifo_dispatchers", {}).values())
        if broker is not None:
            tasks.append(broker)
        tasks.extend(active_tasks)
        if monitor is not None:
            tasks.append(monitor)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        refresh = getattr(self, "_load_refresh_task", None)
        if refresh is not None and not refresh.done():
            refresh.cancel()
            await asyncio.gather(refresh, return_exceptions=True)
        sessions = (
            getattr(self, "_backend_session", None),
            getattr(self, "_load_session", None),
        )
        for session in sessions:
            if session is not None and not session.closed:
                await session.close()

    async def _generate_singleflight(
        self,
        key: str,
        modified_request: dict[str, Any],
        prefill_server: str,
        endpoint: str,
    ) -> tuple[dict[str, Any], int]:
        self._ensure_generation_dedup_state()
        now = time.monotonic()
        async with self._generation_lock:
            self._prune_generation_results(now)
            completed = self._generation_results.get(key)
            if completed is not None:
                logger.info("PD_GENERATION_DEDUP cache_hit generation=%s", key)
                return completed.payload, completed.status
            task = self._generation_tasks.get(key)
            if task is None:
                # The request is mutated while binding its P/NUMA destination;
                # isolate the elected producer from a retry's request object.
                request_copy = orjson.loads(orjson.dumps(modified_request))
                # A newly elected producer is a new physical wire attempt even
                # when its logical request-generation key is unchanged. Never
                # reuse a bootstrap room or TP mailbox identity from a failed
                # transfer whose DMA may still be fenced or quarantined.
                self._replace_prefill_attempt_rooms(request_copy)
                self._set_prefill_attempt_rid(request_copy, replace=True)
                task = asyncio.create_task(self._generate_once(request_copy, prefill_server, endpoint))
                self._generation_tasks[key] = task
                task.add_done_callback(
                    lambda done, generation=key: asyncio.create_task(self._record_generation_task(generation, done))
                )
                logger.info("PD_GENERATION_DEDUP producer generation=%s", key)
            else:
                logger.warning("PD_GENERATION_DEDUP joined_retry generation=%s", key)
        # Client cancellation must not cancel the elected serving operation.
        return await asyncio.shield(task)

    async def _wait_until_prefill_ready(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> int:
        if getattr(self, "dynamic_prefill_domains", False):
            return await self._wait_until_prefill_ready_shared(rooms, prefill_task, route_task)
        deadline = time.monotonic() + self.ready_timeout
        while True:
            paths = [self._ready_path(room) for room in rooms]
            if all(path.exists() for path in paths):
                prompt_tokens = 0
                for path in paths:
                    try:
                        payload = orjson.loads(path.read_bytes())
                    except (OSError, orjson.JSONDecodeError):
                        # Backward compatibility with the old marker containing
                        # only a request id.  Late binding still works, but its
                        # local token reservation is conservative/unknown.
                        continue
                    prompt_tokens += int(payload.get("num_kv_tokens", 0))
                return prompt_tokens
            self._raise_prefill_redirect(route_task)

            # A non-streaming P response should not finish before transfer.  A
            # streaming response can return headers early, so only treat an
            # already returned HTTP error as terminal.
            if prefill_task.done():
                try:
                    response = prefill_task.result()
                except Exception:
                    raise
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(f"Prefill failed before P-ready: status={response.status} body={body}")

            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(f"Timed out waiting {self.ready_timeout}s for P-ready rooms {missing}")
            await asyncio.sleep(self.ready_poll_interval)

    def _scan_p_ready_markers(self) -> dict[int, dict[str, Any]]:
        snapshot: dict[int, dict[str, Any]] = {}
        for path in self.p_ready_dir.glob("*.ready"):
            try:
                room = int(path.stem)
                payload = orjson.loads(path.read_bytes())
                if not isinstance(payload, dict):
                    continue
                payload["_path"] = path
                snapshot[room] = payload
            except (OSError, orjson.JSONDecodeError, TypeError, ValueError):
                # Atomic rename makes partial files impossible, but D may
                # unlink a marker between glob and read.  The next scan is the
                # authoritative state in either case.
                continue
        return snapshot

    async def _p_ready_monitor_loop(self) -> None:
        try:
            while True:
                snapshot = await asyncio.to_thread(self._scan_p_ready_markers)
                self._p_ready_snapshot = snapshot
                for event in getattr(self, "_p_ready_fifo_events", {}).values():
                    event.set()
                broker_event = getattr(self, "_p_ready_broker_event", None)
                if broker_event is not None:
                    broker_event.set()
                for room, futures in list(self._p_ready_waiters.items()):
                    payload = snapshot.get(room)
                    if payload is None:
                        continue
                    for future in tuple(futures):
                        if not future.done():
                            future.set_result(payload)
                await asyncio.sleep(self.ready_poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Shared P-ready watcher failed")
            raise

    def _ensure_p_ready_monitor(self) -> None:
        task = self._p_ready_monitor_task
        if task is None or task.done():
            self._p_ready_monitor_task = asyncio.create_task(self._p_ready_monitor_loop(), name="pd-p-ready-monitor")

    async def _wait_until_prefill_ready_shared(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> int:
        self._ensure_p_ready_monitor()
        loop = asyncio.get_running_loop()
        futures: dict[int, asyncio.Future] = {}
        for room in rooms:
            future = loop.create_future()
            futures[room] = future
            self._p_ready_waiters.setdefault(room, set()).add(future)
            payload = self._p_ready_snapshot.get(room)
            if payload is not None:
                future.set_result(payload)

        deadline = time.monotonic() + self.ready_timeout
        try:
            while True:
                if all(future.done() for future in futures.values()):
                    return sum(int(future.result().get("num_kv_tokens", 0)) for future in futures.values())
                self._raise_prefill_redirect(route_task)
                if prefill_task.done():
                    response = prefill_task.result()
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError("Prefill failed before P-ready: " f"status={response.status} body={body}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [room for room, future in futures.items() if not future.done()]
                    raise TimeoutError(f"Timed out waiting {self.ready_timeout}s for P-ready rooms {missing}")
                pending = [future for future in futures.values() if not future.done()]
                await asyncio.wait(
                    pending,
                    timeout=min(remaining, 0.25),
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            for room, future in futures.items():
                waiters = self._p_ready_waiters.get(room)
                if waiters is None:
                    continue
                waiters.discard(future)
                if not waiters:
                    self._p_ready_waiters.pop(room, None)

    def _p_ready_sequence(self, rooms: tuple[int, ...]) -> int:
        """Read P's authoritative completion sequence from ready markers."""

        sequences: list[int] = []
        for room in rooms:
            path = self._ready_path(room)
            try:
                payload = getattr(self, "_p_ready_snapshot", {}).get(room)
                if payload is None:
                    payload = orjson.loads(path.read_bytes())
                sequence = payload.get("ready_sequence")
                if sequence is not None:
                    sequences.append(int(sequence))
                    continue
            except (OSError, orjson.JSONDecodeError, TypeError, ValueError):
                pass
            # Compatibility with markers produced before ready_sequence was
            # added.  Dedicated run directories make mtime a stable fallback.
            sequences.append(path.stat().st_mtime_ns)
        return min(sequences)

    def _oldest_p_ready_sequence(self, domain: int = 0) -> Optional[int]:
        sequences: list[int] = []
        if getattr(self, "dynamic_prefill_domains", False):
            markers = tuple(getattr(self, "_p_ready_snapshot", {}).values())
        else:
            markers = tuple(self.p_ready_dir.glob("*.ready"))
        for marker in markers:
            try:
                if isinstance(marker, dict):
                    payload = marker
                    path = payload.get("_path")
                else:
                    path = marker
                    payload = orjson.loads(path.read_bytes())
                # SGLang startup probes use the reserved bootstrap room 0.
                # Some probe variants carry a random rid instead of the
                # HEALTH_CHECK_ prefix, so filtering only by rid can leave
                # sequence zero permanently at the head of the business FIFO.
                if path is not None and str(path.stem) == "0":
                    continue
                if (
                    getattr(self, "dynamic_prefill_domains", False)
                    and int(payload.get("prefill_domain", -1)) != domain
                ):
                    continue
                room = int(path.stem) if path is not None else -1
                if not self._p_ready_marker_is_owned(room, payload):
                    # A successful redirect can still race with the old P
                    # scheduler for a few milliseconds.  Ignore and remove
                    # its late marker; strict FIFO applies only to attempts
                    # that still have a Router owner.
                    if path is None:
                        self._p_ready_snapshot.pop(room, None)
                        continue
                    try:
                        current = orjson.loads(path.read_bytes())
                    except (OSError, orjson.JSONDecodeError, TypeError):
                        self._p_ready_snapshot.pop(room, None)
                        continue
                    if not self._p_ready_marker_is_owned(room, current):
                        # Re-check the on-disk rid immediately before unlink:
                        # the monitor snapshot may predate an atomic marker
                        # replacement for a newly active attempt.
                        try:
                            latest = orjson.loads(path.read_bytes())
                        except (OSError, orjson.JSONDecodeError, TypeError):
                            latest = None
                        if latest is not None and self._p_ready_marker_is_owned(room, latest):
                            payload = latest
                            payload["_path"] = path
                            self._p_ready_snapshot[room] = payload
                        else:
                            path.unlink(missing_ok=True)
                            self._p_ready_snapshot.pop(room, None)
                            continue
                    else:
                        payload = current
                        payload["_path"] = path
                        self._p_ready_snapshot[room] = payload
                # SGLang's startup/health probes are real Prefill requests and
                # therefore publish normal P-ready markers.  They have no
                # matching Router generation coroutine, so admitting them to
                # the FIFO would make every workload request wait forever
                # behind an unconsumable sequence number.
                if str(payload.get("rid", "")).startswith("HEALTH_CHECK_"):
                    continue
                sequence = payload.get("ready_sequence")
                sequence = int(sequence) if sequence is not None else path.stat().st_mtime_ns
                key = (domain, sequence) if getattr(self, "dynamic_prefill_domains", False) else sequence
                if key not in self._p_ready_submitted_sequences:
                    sequences.append(sequence)
            except (OSError, orjson.JSONDecodeError, TypeError, ValueError):
                continue
        return min(sequences) if sequences else None

    def _ensure_p_ready_admission_state(self) -> None:
        """Initialize broker state for lightweight/test-created routers."""

        if not hasattr(self, "_p_ready_fifo_waiters"):
            self._p_ready_fifo_waiters = {}
        if not hasattr(self, "_p_ready_fifo_events"):
            self._p_ready_fifo_events = {}
        if not hasattr(self, "_p_ready_fifo_dispatchers"):
            self._p_ready_fifo_dispatchers = {}
        if not hasattr(self, "_p_ready_fifo_active"):
            self._p_ready_fifo_active = {}
        if not hasattr(self, "_p_ready_broker_event"):
            self._p_ready_broker_event = asyncio.Event()
        if not hasattr(self, "_p_ready_broker_task"):
            self._p_ready_broker_task = None
        if not hasattr(self, "_p_ready_commit_tails"):
            self._p_ready_commit_tails = {}
        if not hasattr(self, "_p_ready_admission_window_per_p"):
            self._p_ready_admission_window_per_p = 32
        if not hasattr(self, "_p_ready_stage_lanes_per_p"):
            self._p_ready_stage_lanes_per_p = 4
        if not hasattr(self, "_p_ready_stage_semaphores"):
            self._p_ready_stage_semaphores = {}
        if not hasattr(self, "_p2d_host_offered_snapshots"):
            self._p2d_host_offered_snapshots = set()

    def _p_ready_stage_semaphore(self, domain: int) -> asyncio.Semaphore:
        """Bound concurrent P-HBM -> Host ownership transitions per P."""

        self._ensure_p_ready_admission_state()
        semaphore = self._p_ready_stage_semaphores.get(domain)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max(1, self._p_ready_stage_lanes_per_p))
            self._p_ready_stage_semaphores[domain] = semaphore
        return semaphore

    def _wake_p_ready_fifo(self, domain: int) -> None:
        self._ensure_p_ready_admission_state()
        events = getattr(self, "_p_ready_fifo_events", None)
        if events is None:
            self._p_ready_fifo_events = {}
            events = self._p_ready_fifo_events
        events.setdefault(domain, asyncio.Event()).set()
        broker_event = getattr(self, "_p_ready_broker_event", None)
        if broker_event is not None:
            broker_event.set()

    def _try_reserve_direct_ready_locked(
        self,
        admission: _PReadyAdmission,
        loads_snapshot: list[DecodeLoad],
    ) -> Optional[DecodeReservation]:
        """Reserve one ordinary P-ready request without control-plane I/O.

        This is the common path.  The caller owns ``_selection_lock`` and has
        already obtained one cluster load snapshot for the whole broker batch.
        Host-owned or capacity-bound heads return ``None`` and retain the
        existing full state machine as their single slow-path owner.
        """

        if admission.request is None or admission.commit is None:
            return None
        if admission.cancel_requested:
            return None
        p2d_snapshot = self._p2d_snapshot_for_rooms(admission.rooms)
        if p2d_snapshot is not None and p2d_snapshot in getattr(self, "_p2d_host_offered_snapshots", set()):
            return None

        loads = [load for load in loads_snapshot if load.url in self._domain_decode_urls(admission.domain)]
        if not loads:
            return None

        requested_decode = self._requested_decode_tokens(admission.request)
        decode_headroom = self.decode_headroom_tokens
        if requested_decode is not None:
            decode_headroom = min(requested_decode, self.max_decode_admission_tokens)
        admission_tokens = admission.prompt_tokens + decode_headroom * len(admission.rooms)
        average_context = self._average_running_context(loads)
        transfer_weight = max(1.0, getattr(self, "transfer_request_weight", 2.0))
        scored: list[tuple[bool, float, float, int, DecodeLoad]] = []
        for load in loads:
            reserved_prompt, reserved_admission, reserved_reqs = self._reserved_for(load.url)
            free_after_pending = load.capacity_tokens - load.used_tokens - reserved_admission
            projected_kv = (load.used_tokens + reserved_admission + admission_tokens) / load.capacity_tokens
            feasible = free_after_pending >= admission_tokens and projected_kv <= getattr(
                self, "target_decode_kv_fraction", 1.0
            )
            queued_or_handoff_reqs = max(load.waiting, load.prealloc + load.transfer)
            projected_decode_reqs = load.running + queued_or_handoff_reqs + reserved_reqs + len(admission.rooms)
            projected_compute_kv = (
                load.running_kv_tokens
                + load.prealloc_tokens
                + load.transfer_tokens
                + reserved_prompt
                + admission.prompt_tokens
            )
            work_score = (
                projected_decode_reqs
                + projected_compute_kv / average_context
                + (transfer_weight - 1.0) * load.transfer
            )
            scored.append(
                (
                    not feasible,
                    work_score,
                    projected_kv,
                    projected_decode_reqs,
                    load,
                )
            )

        feasible = [item for item in scored if not item[0]]
        if not feasible and self.wait_for_feasible_decode:
            return None
        candidates = feasible or scored
        _, work_score, projected_kv, projected_decode_reqs, selected = self._choose_decode_score(candidates)
        reservation = DecodeReservation(
            reservation_id=uuid.uuid4().hex,
            url=selected.url,
            prompt_tokens=admission.prompt_tokens,
            admission_tokens=admission_tokens,
            request_count=len(admission.rooms),
            rooms=admission.rooms,
            created_at=time.monotonic(),
            prefill_domain=admission.domain,
        )
        self._reservations[reservation.reservation_id] = reservation
        logger.info(
            "PD_LATE_BIND_BATCH rooms=%s P=%d D=%s prompt_tokens=%d "
            "admission_tokens=%d D_used=%d/%d running=%d waiting=%d "
            "prealloc=%d transfer=%d projected_decode_reqs=%d "
            "average_context=%.1f work_score=%.4f projected_kv=%.4f",
            admission.rooms,
            admission.domain,
            selected.url,
            admission.prompt_tokens,
            admission_tokens,
            selected.used_tokens,
            selected.capacity_tokens,
            selected.running,
            selected.waiting,
            selected.prealloc,
            selected.transfer,
            projected_decode_reqs,
            average_context,
            work_score,
            projected_kv,
        )
        return reservation

    async def _reserve_p_ready_direct_batch(
        self, admissions: list[_PReadyAdmission]
    ) -> dict[_PReadyAdmission, Optional[DecodeReservation]]:
        """Reserve the next fair-scan candidate per P from one D snapshot."""

        eligible = [
            admission for admission in admissions if admission.request is not None and admission.commit is not None
        ]
        reservations = {admission: None for admission in admissions}
        if not eligible:
            return reservations
        try:
            loads = await self._all_decode_loads(self._load_http_session())
        except Exception:
            logger.warning(
                "P-ready batch load snapshot failed; falling back to the " "per-request capacity state machine",
                exc_info=True,
            )
            return reservations
        async with self._selection_lock:
            self._prune_accounted_reservations()
            reservations.update(
                {admission: self._try_reserve_direct_ready_locked(admission, loads) for admission in eligible}
            )
            return reservations

    def _activate_p_ready_admission(self, admission: _PReadyAdmission) -> None:
        waiters = self._p_ready_fifo_waiters.setdefault(admission.domain, {})
        waiters.pop(admission.sequence, None)
        self._p_ready_submitted_sequences.add(admission.submitted_key)
        active = self._p_ready_fifo_active.setdefault(admission.domain, {})
        active[admission.sequence] = admission
        loop = asyncio.get_running_loop()
        # ``ready_sequence`` remains the fair scan order, not a cross-request
        # correctness dependency.  A Host-staged or temporarily infeasible
        # generation must not block a later generation that can enter D now.
        admission.commit_predecessor = None
        admission.commit_done = loop.create_future()

        waited = time.monotonic() - admission.enqueued_at
        if waited >= 0.5:
            logger.info(
                "PD_P_READY_FIFO_DISPATCH P=%d sequence=%d wait_s=%.3f " "submitted=%d queued=%d broker_batch=true",
                admission.domain,
                admission.sequence,
                waited,
                len(self._p_ready_submitted_sequences),
                len(waiters),
            )

    async def _run_p_ready_admission(
        self,
        admission: _PReadyAdmission,
        reservation: Optional[DecodeReservation],
    ) -> None:
        prepared_reservation: Optional[DecodeReservation] = None
        try:
            prepared_reservation = reservation
            admission.initial_reservation = reservation
            admission.ownership_started = bool(reservation is not None or admission.prepare is not None)
            if prepared_reservation is None and admission.prepare is not None:
                spill_delay = max(0.0, float(getattr(self, "p2d_host_spill_delay", 0.0)))
                if spill_delay > 0:
                    grace_started_at = time.monotonic()
                    await asyncio.sleep(spill_delay)
                    if admission.cancel_requested:
                        raise asyncio.CancelledError
                    prepared_reservation = await self._retry_p_ready_direct_after_grace(
                        admission, not_before=grace_started_at
                    )
                # Preparation owns only the physical handoff boundary.  Up to
                # the number of real D2H lanes may run concurrently; after a
                # complete Host snapshot is durable, P can immediately release
                # the request-generation HBM even while ordered D admission is
                # still waiting for capacity.
                if prepared_reservation is None:
                    async with self._p_ready_stage_semaphore(admission.domain):
                        prepared_reservation = await admission.prepare()
            admission.prepare_complete = True
            admission.host_staged = admission.prepare is not None and prepared_reservation is None
            if admission.cancel_requested:
                raise asyncio.CancelledError
            admission.commit_started = True
            if prepared_reservation is None:
                result = await admission.dispatch()
            else:
                assert admission.commit is not None
                result = await admission.commit(prepared_reservation)
        except BaseException as exc:
            self._p_ready_submitted_sequences.discard(admission.submitted_key)
            if not admission.future.done():
                if isinstance(exc, asyncio.CancelledError):
                    admission.future.cancel()
                else:
                    admission.future.set_exception(exc)
        else:
            if not admission.future.done():
                admission.future.set_result(result)
        finally:
            # A D reservation is only handed off when commit starts.  Router
            # shutdown/caller cancellation can otherwise strand capacity while
            # this admission waits behind its FIFO predecessor.  Host-owned
            # snapshots need the matching fence-aware abort instead: their P
            # pages or D2H may already be under physical staging ownership.
            if not admission.commit_started:
                if prepared_reservation is not None:
                    async with self._selection_lock:
                        self._reservations.pop(prepared_reservation.reservation_id, None)
                        self._admitted_reservation_at.pop(prepared_reservation.reservation_id, None)
                        self._load_cache_at = 0.0
                elif admission.prepare is not None:
                    self._abort_unsubmitted_p2d(
                        self._p2d_snapshot_for_rooms(admission.rooms),
                        "p_ready_commit_cancelled",
                    )
            if admission.commit_done is not None and not admission.commit_done.done():
                admission.commit_done.set_result(None)
            active = self._p_ready_fifo_active.get(admission.domain)
            if active is not None:
                active.pop(admission.sequence, None)
                if not active:
                    self._p_ready_fifo_active.pop(admission.domain, None)
            admission.finished.set()
            self._p_ready_broker_event.set()

    def _cancel_unowned_p_ready_admission(self, admission: _PReadyAdmission) -> bool:
        """Cancel an active FIFO record before it owns any physical state."""

        if admission.ownership_started or admission.commit_started:
            return False
        admission.cancel_requested = True
        if not admission.future.done():
            admission.future.cancel()
        if admission.commit_done is not None and not admission.commit_done.done():
            admission.commit_done.set_result(None)
        active = self._p_ready_fifo_active.get(admission.domain)
        if active is not None:
            active.pop(admission.sequence, None)
            if not active:
                self._p_ready_fifo_active.pop(admission.domain, None)
        self._p_ready_submitted_sequences.discard(admission.submitted_key)
        task = admission.dispatch_task
        if task is not None and not task.done():
            task.cancel()
        admission.finished.set()
        self._p_ready_broker_event.set()
        return True

    def _next_p_ready_heads(self) -> list[_PReadyAdmission]:
        """Return the next authoritative fair-scan candidate per P."""

        heads: list[_PReadyAdmission] = []
        domains = sorted(self._p_ready_fifo_waiters)
        for domain in domains:
            waiters = self._p_ready_fifo_waiters.get(domain, {})
            if not waiters:
                continue
            if len(self._p_ready_fifo_active.get(domain, {})) >= max(1, self._p_ready_admission_window_per_p):
                continue
            oldest = self._oldest_p_ready_sequence(domain)
            sequence = min(waiters) if oldest is None else oldest
            admission = waiters.get(sequence)
            if admission is not None:
                heads.append(admission)
        return heads

    async def _p_ready_admission_broker_loop(self) -> None:
        """Batch D admission while keeping each P's commit order strict."""

        try:
            while True:
                heads = self._next_p_ready_heads()
                if not heads:
                    self._p_ready_broker_event.clear()
                    # Close the race with a producer that enqueued immediately
                    # before clear().
                    if self._next_p_ready_heads():
                        self._p_ready_broker_event.set()
                        continue
                    await self._p_ready_broker_event.wait()
                    continue

                reservations = await self._reserve_p_ready_direct_batch(heads)
                for admission in heads:
                    if admission.cancel_requested:
                        continue
                    reservation = reservations.get(admission)
                    admission.initial_reservation = reservation
                    # Capacity-bound generations prepare independently in a
                    # bounded lane pool.  They do not gate later feasible
                    # generations from the same P.
                    self._activate_p_ready_admission(admission)
                    admission.dispatch_task = asyncio.create_task(
                        self._run_p_ready_admission(
                            admission,
                            reservation,
                        ),
                        name=(f"pd-p-ready-admission-{admission.domain}-" f"{admission.sequence}"),
                    )
                # Let commit tasks issue their D POSTs, then scan the next
                # candidates without waiting for receiver completion.
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("P-ready admission broker failed")
            for waiters in self._p_ready_fifo_waiters.values():
                for admission in waiters.values():
                    if not admission.future.done():
                        admission.future.set_exception(exc)
            raise

    async def _dispatch_p_ready_in_order(
        self,
        sequence: int,
        domain: int,
        dispatch: Callable[[], Awaitable[Any]],
        *,
        request: Optional[dict[str, Any]] = None,
        rooms: tuple[int, ...] = (),
        prompt_tokens: int = 0,
        commit: Optional[Callable[[DecodeReservation], Awaitable[Any]]] = None,
        prepare: Optional[Callable[[], Awaitable[Optional[DecodeReservation]]]] = None,
    ) -> Any:
        """Publish an immutable admission and await the shared broker."""

        self._ensure_p_ready_admission_state()
        waiters = self._p_ready_fifo_waiters.setdefault(domain, {})
        if sequence in waiters:
            raise RuntimeError(f"duplicate P-ready sequence P={domain} sequence={sequence}")
        key: Any = (domain, sequence) if getattr(self, "dynamic_prefill_domains", False) else sequence
        future = asyncio.get_running_loop().create_future()
        admission = _PReadyAdmission(
            domain=domain,
            sequence=sequence,
            submitted_key=key,
            enqueued_at=time.monotonic(),
            dispatch=dispatch,
            future=future,
            finished=asyncio.Event(),
            request=request,
            rooms=rooms,
            prompt_tokens=prompt_tokens,
            commit=commit,
            prepare=prepare,
        )
        waiters[sequence] = admission
        task = getattr(self, "_p_ready_broker_task", None)
        if task is None or task.done():
            self._p_ready_broker_task = asyncio.create_task(
                self._p_ready_admission_broker_loop(),
                name="pd-p-ready-admission-broker",
            )
        self._p_ready_broker_event.set()
        try:
            # Shielding keeps caller cancellation from silently cancelling the
            # shared Future while the dispatcher continues mutating transport
            # state.  Cancellation is handled explicitly below.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if waiters.get(sequence) is admission:
                waiters.pop(sequence, None)
                admission.cancel_requested = True
                if not admission.future.done():
                    admission.future.cancel()
                admission.finished.set()
                self._wake_p_ready_fifo(domain)
            else:
                # Once the dispatcher starts an admission it owns a small
                # transaction: choose Direct/Host, bind one D reservation and
                # submit the Decode request.  A transient client cancellation
                # must not abort that transaction halfway through a Host D2H
                # write.  Doing so invalidates the producer grant and turns a
                # physically recoverable P-ready snapshot into a retry/full
                # recompute.  Defer cancellation until the dispatcher reaches
                # its commit boundary; the outer request cleanup can then
                # dispose the already-submitted Decode response safely.
                # If no reservation, Host offer or commit has started, this
                # active record is only waiting behind its FIFO predecessor
                # and is still safe to cancel immediately.
                cancelled_unowned = self._cancel_unowned_p_ready_admission(admission)
                if not cancelled_unowned:
                    while not admission.finished.is_set():
                        try:
                            await asyncio.shield(admission.finished.wait())
                        except asyncio.CancelledError:
                            continue
            raise
        except BaseException:
            if waiters.get(sequence) is admission:
                waiters.pop(sequence, None)
                admission.finished.set()
                self._wake_p_ready_fifo(domain)
            raise

    @staticmethod
    def _requested_decode_tokens(request: dict[str, Any]) -> Optional[int]:
        for key in ("max_new_tokens", "max_completion_tokens", "max_tokens"):
            value = request.get(key)
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    return None
        sampling = request.get("sampling_params")
        if isinstance(sampling, dict) and sampling.get("max_new_tokens") is not None:
            try:
                return max(0, int(sampling["max_new_tokens"]))
            except (TypeError, ValueError):
                return None
        return None

    async def _fetch_decode_load(self, session: aiohttp.ClientSession, url: str) -> DecodeLoad:
        if url in self._legacy_load_urls:
            return await self._fetch_decode_load_legacy(session, url)
        endpoint = f"{url}/v1/loads?include=disagg,memory,queues"
        try:
            async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=self.load_timeout)) as response:
                response.raise_for_status()
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # SGLang 0.5.10 has a known-in-this-environment FastAPI middleware
            # incompatibility where an included APIRouter can make /v1/loads
            # return 500.  /get_load comes from the same scheduler snapshot and
            # includes physical + queued/prealloc/transfer token demand.
            self._legacy_load_urls.add(url)
            return await self._fetch_decode_load_legacy(session, url)

        loads = payload.get("loads") or []
        if not loads:
            raise RuntimeError(f"D load endpoint returned no ranks: {url}")
        physical_used = sum(int(load.get("num_used_tokens", 0)) for load in loads)
        capacity = sum(int(load.get("max_total_num_tokens", 0)) for load in loads)
        running = sum(int(load.get("num_running_reqs", 0)) for load in loads)
        waiting = sum(int(load.get("num_waiting_reqs", 0)) for load in loads)
        max_running = sum(int(load.get("max_running_requests", 0)) for load in loads)
        prealloc = 0
        transfer = 0
        prealloc_tokens = 0
        transfer_tokens = 0
        for load in loads:
            disagg = load.get("disaggregation") or {}
            prealloc += int(disagg.get("decode_prealloc_queue_reqs", 0))
            transfer += int(disagg.get("decode_transfer_queue_reqs", 0))
            prealloc_tokens += int(disagg.get("decode_prealloc_queue_tokens", 0))
            transfer_tokens += int(disagg.get("decode_transfer_queue_tokens", 0))
        if capacity <= 0:
            raise RuntimeError(f"D load endpoint returned invalid KV capacity: {url}")
        running_kv_tokens = sum(int(load.get("num_running_kv_tokens", 0)) for load in loads)
        # SGLang's num_used_tokens is already the non-evictable occupancy:
        # max_total - (allocator.available + radix.evictable).  In particular,
        # it also includes protected direct/Host-staging snapshots that are not
        # represented by running_kv_tokens.  Those pages cannot be promised to
        # a new P->D handoff, so use the scheduler's value as the hard bound.
        hard_used = physical_used
        return DecodeLoad(
            url=url,
            used_tokens=hard_used,
            capacity_tokens=capacity,
            running=running,
            waiting=waiting,
            prealloc=prealloc,
            transfer=transfer,
            max_running=max_running,
            running_kv_tokens=running_kv_tokens,
            prealloc_tokens=prealloc_tokens,
            transfer_tokens=transfer_tokens,
            physical_used_tokens=physical_used,
        )

    async def _fetch_decode_load_legacy(self, session: aiohttp.ClientSession, url: str) -> DecodeLoad:
        # /get_load is a tiny scheduler-control RPC.  Under c512 the Router's
        # ASGI loop also owns hundreds of long-lived generation coroutines and
        # synchronous tmpfs lifecycle operations; driving this RPC on the same
        # loop caused false 2s timeouts while a separate client reached every
        # D immediately.  Move only the network wait/JSON read to a worker;
        # publication and reservation accounting remain on the Router loop.
        rows = await asyncio.to_thread(_sync_json_get, f"{url}/get_load", self.load_timeout)
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            raise RuntimeError(f"D legacy load endpoint returned no ranks: {url}")

        reported_capacity = sum(int(row.get("max_total_num_tokens", 0)) for row in rows)
        reported_max_running = sum(int(row.get("max_running_requests", 0)) for row in rows)
        cached = self._last_loads.get(url)
        if reported_capacity > 0:
            capacity = reported_capacity
            max_running = reported_max_running
        elif cached is not None:
            capacity = cached.capacity_tokens
            max_running = cached.max_running
        else:
            server_info = await asyncio.to_thread(_sync_json_get, f"{url}/server_info", self.load_timeout)
            internal = server_info.get("internal_states") or []
            capacity = sum(int((state.get("memory_usage") or {}).get("token_capacity", 0)) for state in internal)
            max_running = sum(int(state.get("effective_max_running_requests_per_dp", 0)) for state in internal)
        if capacity <= 0:
            raise RuntimeError(f"D server_info returned invalid KV capacity: {url}")

        total_reqs = sum(int(row.get("num_reqs", 0)) for row in rows)
        waiting = sum(int(row.get("num_waiting_reqs", 0)) for row in rows)
        prealloc = sum(int(row.get("decode_prealloc_queue_reqs", 0)) for row in rows)
        transfer = sum(int(row.get("decode_transfer_queue_reqs", 0)) for row in rows)
        physical_used = sum(int(row.get("num_physical_used_tokens", row.get("num_tokens", 0))) for row in rows)
        running_kv_tokens = sum(int(row.get("num_running_kv_tokens", 0)) for row in rows)
        prealloc_tokens = sum(int(row.get("decode_prealloc_queue_tokens", 0)) for row in rows)
        transfer_tokens = sum(int(row.get("decode_transfer_queue_tokens", 0)) for row in rows)
        return DecodeLoad(
            url=url,
            # Despite the compatibility field's historical name, SGLang fills
            # num_physical_used_tokens from _get_token_info()[0], which already
            # excludes evictable Radix pages and includes protected snapshots.
            used_tokens=physical_used,
            capacity_tokens=capacity,
            running=max(0, total_reqs - waiting),
            waiting=waiting,
            prealloc=prealloc if prealloc or transfer else waiting,
            transfer=transfer,
            max_running=max_running,
            running_kv_tokens=running_kv_tokens,
            prealloc_tokens=prealloc_tokens,
            transfer_tokens=transfer_tokens,
            physical_used_tokens=physical_used,
        )

    async def _refresh_decode_loads(self, session: aiohttp.ClientSession) -> list[DecodeLoad]:
        sample_started_at = time.monotonic()
        results = await asyncio.gather(
            *(self._fetch_decode_load(session, url) for url in self.decode_urls),
            return_exceptions=True,
        )
        loads: list[DecodeLoad] = []
        fresh_loads: dict[str, DecodeLoad] = {}
        for url, result in zip(self.decode_urls, results):
            if isinstance(result, Exception):
                cached = self._last_loads.get(url)
                if cached is None:
                    logger.warning("Skipping D with unavailable load endpoint %s: %s", url, result)
                    continue
                logger.warning("Using last load snapshot for D %s: %s", url, result)
                loads.append(cached)
            else:
                fresh_loads[url] = result
                loads.append(result)
        if not loads:
            raise RuntimeError("No decode server has a usable /v1/loads response")
        if not fresh_loads:
            # Reusing old rows is safe only as a conservative fallback.  It is
            # not a fresh observation and must not renew the cache TTL or its
            # causal epoch; otherwise repeated control-plane failures can pin
            # a once-full D snapshot forever after the real GPU drains.
            if self._load_cache:
                return self._load_cache
            raise RuntimeError("No decode server returned a fresh load snapshot")
        published_at = time.monotonic()
        # Concurrent polls may complete out of order. Publish only the newest
        # sampling epoch; otherwise a slow old response could move the causal
        # watermark backwards and overwrite newer D state.
        if sample_started_at >= getattr(self, "_load_cache_sample_started_at", 0.0):
            self._last_loads.update(fresh_loads)
            epochs = getattr(self, "_load_sample_started_at_by_url", None)
            if epochs is None:
                self._load_sample_started_at_by_url = {}
                epochs = self._load_sample_started_at_by_url
            for url in fresh_loads:
                epochs[url] = sample_started_at
            self._load_cache = loads
            self._load_cache_sample_started_at = sample_started_at
            self._load_cache_at = published_at
            return loads
        # This poll lost an out-of-order publication race. Returning its old
        # rows while reservation accounting uses the newer global epoch would
        # combine two incompatible views and could double-spend D capacity.
        # The caller must use the same authoritative snapshot as accounting.
        return self._load_cache

    async def _refresh_decode_loads_background(self) -> None:
        try:
            await self._refresh_decode_loads(self._load_http_session())
        except Exception:
            # The current cached snapshot plus local reservations remains a
            # conservative admission view.  A later dispatch will retry.
            logger.warning("Background D load refresh failed", exc_info=True)

    async def _all_decode_loads(self, session: aiohttp.ClientSession, *, force: bool = False) -> list[DecodeLoad]:
        # ``session`` remains in the signature for lightweight tests and old
        # callers, but production load RPCs always use their isolated control
        # connector rather than the long-lived generation data plane.
        load_session = None if session is None else self._load_http_session()
        now = time.monotonic()
        fresh = self._load_cache and now - self._load_cache_at < self.load_cache_ttl
        if not force and fresh:
            return self._load_cache

        if not force and not self._load_cache:
            # The first c512 burst can otherwise launch one four-endpoint poll
            # per request before any cache exists. Share exactly one initial
            # sample; shielding keeps a cancelled HTTP request from cancelling
            # the cluster-wide bootstrap result for every other waiter.
            task = self._load_refresh_task
            if task is None or task.done():
                task = asyncio.create_task(self._refresh_decode_loads(load_session))
                self._load_refresh_task = task
            return await asyncio.shield(task)

        if not force and self._load_cache:
            # Never put HTTP load polling in the per-request reservation
            # critical path.  Dispatch immediately from the last snapshot and
            # refresh it independently; local token/request reservations make
            # stale admission conservative rather than allowing oversubscribe.
            task = self._load_refresh_task
            if task is None or task.done():
                self._load_refresh_task = asyncio.create_task(self._refresh_decode_loads_background())
            return self._load_cache

        return await self._refresh_decode_loads(load_session)

    def _decode_loads_are_fresh_since(self, urls: set[str], not_before: float) -> bool:
        """Return whether every candidate D was sampled after ``not_before``.

        P->D Host spill costs an extra Host round trip even though recovery can
        now target any D domain.  Do not choose that path from a cached
        all-full observation that predates the P-ready snapshot itself.
        """

        epochs = getattr(self, "_load_sample_started_at_by_url", {})
        return bool(urls) and all(float(epochs.get(url, 0.0)) >= not_before for url in urls)

    async def _observe_decode_load_after(
        self,
        session: aiohttp.ClientSession,
        *,
        urls: set[str],
        not_before: float,
    ) -> bool:
        """Single-flight one causal D-load observation before Host spill.

        The ordinary stale-while-revalidate path already starts a shared
        refresh.  Await that task first; if it began too early, start exactly
        one newer shared refresh.  Failure returns ``False`` so Host spill can
        still guarantee progress instead of retaining P HBM indefinitely.
        """

        for _ in range(2):
            if self._decode_loads_are_fresh_since(urls, not_before):
                return True
            task = self._load_refresh_task
            if task is None or task.done():
                task = asyncio.create_task(self._refresh_decode_loads(self._load_http_session()))
                self._load_refresh_task = task
            try:
                await asyncio.shield(task)
            except Exception:
                logger.warning(
                    "Fresh D load observation failed before P->D Host spill",
                    exc_info=True,
                )
                return False
        return self._decode_loads_are_fresh_since(urls, not_before)

    def _reserved_for(self, url: str, *, exclude_id: Optional[str] = None) -> tuple[int, int, int]:
        prompt = admission = requests = 0
        for reservation_id, reservation in self._reservations.items():
            if reservation_id == exclude_id:
                continue
            admitted_at = self._admitted_reservation_at.get(reservation_id)
            sample_epoch = getattr(self, "_load_sample_started_at_by_url", {}).get(
                reservation.url,
                getattr(self, "_load_cache_sample_started_at", 0.0),
            )
            if admitted_at is not None and sample_epoch >= admitted_at:
                continue
            if reservation.url == url:
                prompt += reservation.prompt_tokens
                admission += reservation.admission_tokens
                requests += reservation.request_count
        return prompt, admission, requests

    def _prune_accounted_reservations(self) -> None:
        accounted = []
        epochs = getattr(self, "_load_sample_started_at_by_url", {})
        global_epoch = getattr(self, "_load_cache_sample_started_at", 0.0)
        for reservation_id, admitted_at in self._admitted_reservation_at.items():
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                accounted.append(reservation_id)
                continue
            if epochs.get(reservation.url, global_epoch) >= admitted_at:
                accounted.append(reservation_id)
        for reservation_id in accounted:
            self._admitted_reservation_at.pop(reservation_id, None)
            self._reservations.pop(reservation_id, None)

    def _average_running_context(self, loads: list[DecodeLoad]) -> float:
        running = sum(load.running for load in loads)
        running_tokens = sum(load.running_kv_tokens for load in loads)
        observed = running_tokens / running if running > 0 and running_tokens > 0 else 4096.0
        floor = max(1, getattr(self, "context_token_floor", 2048))
        ceiling = max(floor, getattr(self, "context_token_ceiling", 8192))
        return min(float(ceiling), max(float(floor), observed))

    def _finalize_p2d_route(
        self,
        reservation: DecodeReservation,
        *,
        snapshot_id: Optional[str],
        state: Optional[str],
        domain: int,
    ) -> Optional[DecodeReservation]:
        """Atomically bind one capacity credit to Direct or Host input.

        A future D credit can be created before P->D Host ownership settles.
        Finalization is therefore shared by both immediately-feasible and
        draining admissions.  ``None`` means a concurrent Host claim won and
        the selector must observe its final locality/readiness before submit.
        """

        if snapshot_id is None:
            return reservation
        if state is None or state == HostStageState.REJECTED.value:
            return reservation
        if state == HostStageState.OFFERED.value:
            cancelled = self.p2d_host_ledger.reject_unclaimed_offer(
                snapshot_id,
                reason="decode_capacity_available",
            )
            if not cancelled:
                return None
            return reservation
        if state in {
            HostStageState.HOST_RESERVED.value,
            HostStageState.HOST_WRITING.value,
            HostStageState.ABORTING.value,
            HostStageState.H2D_LOADING.value,
        }:
            return None
        if state == HostStageState.HOST_READY.value:
            return replace(
                reservation,
                p2d_host_snapshot_id=snapshot_id,
                prefill_domain=domain,
            )
        if state == HostStageState.FAILED.value:
            entry = self.p2d_host_ledger.get(snapshot_id) or {}
            raise RuntimeError(
                f"P->D Host snapshot {snapshot_id} failed after taking "
                f"exclusive ownership: {entry.get('reason', 'unknown')}"
            )
        if state == HostStageState.CONSUMED.value:
            raise RuntimeError(
                f"P->D Host snapshot {snapshot_id} was already consumed before " "this Router attempt submitted Decode"
            )
        # Custom P->D staging has no lower storage tier.  Seeing a D->P-only
        # or unknown state here is a control-plane invariant violation, never
        # permission to resurrect the native sender.
        raise RuntimeError(f"P->D Host snapshot {snapshot_id} has invalid route state {state!r}")

    async def _select_and_reserve_decode(
        self,
        session: aiohttp.ClientSession,
        request: dict[str, Any],
        rooms: tuple[int, ...],
        prompt_tokens: int,
        domain: int = 0,
    ) -> DecodeReservation:
        requested_decode = self._requested_decode_tokens(request)
        decode_headroom = self.decode_headroom_tokens
        if requested_decode is not None:
            decode_headroom = min(requested_decode, self.max_decode_admission_tokens)
        admission_tokens = prompt_tokens + decode_headroom * len(rooms)
        deadline = time.monotonic() + self.ready_timeout
        wait_started = time.monotonic()
        next_wait_log = time.monotonic() + 5.0
        next_missing_domain_log = time.monotonic()
        draining_reservation: Optional[DecodeReservation] = None
        p2d_snapshot = self._p2d_snapshot_for_rooms(rooms)
        p2d_offer_published = bool(
            p2d_snapshot is not None and p2d_snapshot in getattr(self, "_p2d_host_offered_snapshots", set())
        )
        host_freshness_attempted = False
        force_load_refresh = False

        try:
            while True:
                # Both network sampling and the flock/JSON Host ledger live
                # outside the cluster-wide D capacity lock.  A provisional D
                # credit protects capacity while the Host CAS is finalized.
                loads_snapshot = await self._all_decode_loads(session, force=force_load_refresh)
                force_load_refresh = False
                p2d_entry = (
                    await asyncio.to_thread(self.p2d_host_ledger.get, p2d_snapshot)
                    if p2d_snapshot is not None
                    else None
                )
                if p2d_entry is not None and p2d_snapshot is not None:
                    p2d_offer_published = True
                    if not hasattr(self, "_p2d_host_offered_snapshots"):
                        self._p2d_host_offered_snapshots = set()
                    self._p2d_host_offered_snapshots.add(p2d_snapshot)
                p2d_state = None if p2d_entry is None else p2d_entry.get("state")
                p2d_claimed = p2d_state in {
                    HostStageState.HOST_RESERVED.value,
                    HostStageState.HOST_WRITING.value,
                    HostStageState.HOST_READY.value,
                    HostStageState.H2D_LOADING.value,
                    HostStageState.CONSUMED.value,
                }
                p2d_ready = p2d_state == HostStageState.HOST_READY.value

                candidate: Optional[DecodeReservation] = None
                candidate_is_new = False
                candidate_stats: Optional[tuple[DecodeLoad, float, float, int, float]] = None
                publish_host_offer = False
                refresh_before_host_offer = False
                refresh_urls: set[str] = set()
                missing_loads = False
                stale_ownership = False

                async with self._selection_lock:
                    self._prune_accounted_reservations()
                    # The event-loop publisher records the id synchronously
                    # with ledger.offer().  If it ran while the lock was being
                    # acquired, discard the stale lock-free ledger read.
                    if (
                        p2d_snapshot is not None
                        and p2d_entry is None
                        and p2d_snapshot in getattr(self, "_p2d_host_offered_snapshots", set())
                    ):
                        stale_ownership = True
                        loads: list[DecodeLoad] = []
                    else:
                        # Host ownership describes where the durable bytes
                        # live, not which Decode domain must consume them.
                        # Keep late binding global after spill as well: a D on
                        # another NUMA node can read the shared tmpfs extent
                        # through its independent Host->HBM worker.  The Host
                        # owner is released only after the group-level H2D
                        # completion, so this does not change ownership or TP
                        # atomicity.
                        allowed_urls = self._domain_decode_urls(domain)
                        loads = [load for load in loads_snapshot if load.url in allowed_urls]
                        missing_loads = not loads

                    if not stale_ownership and loads:
                        if draining_reservation is not None:
                            if not any(load.url == draining_reservation.url for load in loads):
                                self._reservations.pop(draining_reservation.reservation_id, None)
                                draining_reservation = None

                        average_context = self._average_running_context(loads)
                        transfer_weight = max(
                            1.0,
                            getattr(self, "transfer_request_weight", 2.0),
                        )
                        scored: list[tuple[bool, float, float, int, DecodeLoad]] = []
                        for load in loads:
                            # A draining reservation is only a future-capacity
                            # hint.  Re-score every D without counting that
                            # hint so an observation made while all D workers
                            # were full cannot pin this request to a now-heavier
                            # destination.
                            draining_id = (
                                draining_reservation.reservation_id if draining_reservation is not None else None
                            )
                            (
                                reserved_prompt,
                                reserved_admission,
                                reserved_reqs,
                            ) = self._reserved_for(load.url, exclude_id=draining_id)
                            free_after_pending = load.capacity_tokens - load.used_tokens - reserved_admission
                            projected_kv = (
                                load.used_tokens + reserved_admission + admission_tokens
                            ) / load.capacity_tokens
                            feasible = free_after_pending >= admission_tokens and projected_kv <= getattr(
                                self, "target_decode_kv_fraction", 1.0
                            )
                            queued_or_handoff_reqs = max(load.waiting, load.prealloc + load.transfer)
                            projected_decode_reqs = load.running + queued_or_handoff_reqs + reserved_reqs + len(rooms)
                            projected_compute_kv = (
                                load.running_kv_tokens
                                + load.prealloc_tokens
                                + load.transfer_tokens
                                + reserved_prompt
                                + prompt_tokens
                            )
                            work_score = (
                                projected_decode_reqs
                                + projected_compute_kv / average_context
                                + (transfer_weight - 1.0) * load.transfer
                            )
                            scored.append(
                                (
                                    not feasible,
                                    work_score,
                                    projected_kv,
                                    projected_decode_reqs,
                                    load,
                                )
                            )

                        feasible = [item for item in scored if not item[0]]
                        if (
                            candidate is None
                            and not (p2d_claimed and not p2d_ready)
                            and (feasible or not self.wait_for_feasible_decode)
                        ):
                            choice = self._choose_decode_score(feasible or scored)
                            _, work_score, projected_kv, projected_reqs, selected = choice
                            if draining_reservation is not None and selected.url == draining_reservation.url:
                                candidate = draining_reservation
                            else:
                                if draining_reservation is not None:
                                    old_url = draining_reservation.url
                                    self._reservations.pop(draining_reservation.reservation_id, None)
                                    draining_reservation = None
                                    logger.info(
                                        "PD_LATE_BIND_DRAIN_RESELECT rooms=%s " "old_D=%s new_D=%s reason=least_work",
                                        rooms,
                                        old_url,
                                        selected.url,
                                    )
                                candidate = DecodeReservation(
                                    reservation_id=uuid.uuid4().hex,
                                    url=selected.url,
                                    prompt_tokens=prompt_tokens,
                                    admission_tokens=admission_tokens,
                                    request_count=len(rooms),
                                    rooms=rooms,
                                    created_at=time.monotonic(),
                                )
                                candidate_is_new = True
                                self._reservations[candidate.reservation_id] = candidate
                            candidate_stats = (
                                selected,
                                work_score,
                                projected_kv,
                                projected_reqs,
                                average_context,
                            )

                        if candidate is None:
                            waited = time.monotonic() - wait_started
                            if draining_reservation is None:
                                draining_urls = {item.url for item in self._reservations.values() if item.draining}
                                drain_choices = [item for item in scored if item[4].url not in draining_urls]
                                if drain_choices:
                                    choice = self._choose_decode_score(drain_choices, drain=True)
                                    (
                                        _,
                                        work_score,
                                        projected_kv,
                                        projected_reqs,
                                        selected,
                                    ) = choice
                                    draining_reservation = DecodeReservation(
                                        reservation_id=uuid.uuid4().hex,
                                        url=selected.url,
                                        prompt_tokens=prompt_tokens,
                                        admission_tokens=admission_tokens,
                                        request_count=len(rooms),
                                        rooms=rooms,
                                        created_at=time.monotonic(),
                                        draining=True,
                                    )
                                    self._reservations[draining_reservation.reservation_id] = draining_reservation
                                    logger.info(
                                        "PD_LATE_BIND_DRAIN_RESERVE rooms=%s "
                                        "D=%s admission_tokens=%d D_used=%d/%d "
                                        "projected_decode_reqs=%d "
                                        "projected_kv=%.4f work_score=%.4f",
                                        rooms,
                                        selected.url,
                                        admission_tokens,
                                        selected.used_tokens,
                                        selected.capacity_tokens,
                                        projected_reqs,
                                        projected_kv,
                                        work_score,
                                    )
                            wants_host_offer = bool(
                                p2d_snapshot is not None
                                and not p2d_offer_published
                                and waited >= self.p2d_host_spill_delay
                            )
                            if wants_host_offer:
                                refresh_urls = self._domain_decode_urls(domain)
                                # In production ``_all_decode_loads`` owns the
                                # authoritative cache.  Lightweight tests may
                                # inject loads directly without causal epochs.
                                causal_tracking = bool(self._load_cache)
                                refresh_before_host_offer = bool(
                                    causal_tracking
                                    and not host_freshness_attempted
                                    and not self._decode_loads_are_fresh_since(refresh_urls, wait_started)
                                )
                                publish_host_offer = not refresh_before_host_offer

                if stale_ownership:
                    continue
                if missing_loads:
                    now = time.monotonic()
                    if now >= next_missing_domain_log:
                        logger.warning(
                            "No current D load sample in domain %d; waiting " "for the local endpoint",
                            domain,
                        )
                        next_missing_domain_log = now + 5.0
                    if now >= deadline:
                        raise TimeoutError(
                            f"No usable D worker in domain {domain} before " "P-ready admission deadline"
                        )
                    self._load_cache_at = 0.0
                    force_load_refresh = True
                    await asyncio.sleep(self.no_capacity_poll_interval)
                    continue

                if refresh_before_host_offer:
                    host_freshness_attempted = True
                    await self._observe_decode_load_after(
                        session,
                        urls=refresh_urls,
                        not_before=wait_started,
                    )
                    # Re-run global feasibility against the causally fresh
                    # snapshot.  Only a still-full result may become the
                    # NUMA-local P->D Host owner.
                    continue

                if candidate is not None:
                    candidate_settled = False
                    try:
                        finalized = await asyncio.to_thread(
                            self._finalize_p2d_route,
                            candidate,
                            snapshot_id=p2d_snapshot,
                            state=p2d_state,
                            domain=domain,
                        )
                        async with self._selection_lock:
                            if finalized is None:
                                if candidate_is_new:
                                    self._reservations.pop(candidate.reservation_id, None)
                            else:
                                self._reservations[finalized.reservation_id] = finalized
                            candidate_settled = True
                    except BaseException:
                        if candidate_is_new and not candidate_settled:
                            # No await is needed for rollback: Router state is
                            # event-loop local, so this runs before another
                            # selector can observe the stale credit.
                            self._reservations.pop(candidate.reservation_id, None)
                        raise
                    if finalized is None:
                        continue

                    selected, work_score, projected_kv, projected_reqs, average = candidate_stats
                    logger.info(
                        "PD_LATE_BIND rooms=%s D=%s prompt_tokens=%d "
                        "admission_tokens=%d D_used=%d/%d running=%d "
                        "waiting=%d prealloc=%d transfer=%d "
                        "projected_decode_reqs=%d average_context=%.1f "
                        "work_score=%.4f projected_kv=%.4f "
                        "policy=feasible_least_work",
                        rooms,
                        selected.url,
                        prompt_tokens,
                        admission_tokens,
                        selected.used_tokens,
                        selected.capacity_tokens,
                        selected.running,
                        selected.waiting,
                        selected.prealloc,
                        selected.transfer,
                        projected_reqs,
                        average,
                        work_score,
                        projected_kv,
                    )
                    if finalized.p2d_host_snapshot_id is not None:
                        logger.info(
                            "PD_P2D_HOST_BIND snapshot=%s rooms=%s P=%d D=%s "
                            "policy=global_host_feasible_least_work",
                            p2d_snapshot,
                            rooms,
                            domain,
                            selected.url,
                        )
                    return finalized

                if publish_host_offer and p2d_snapshot is not None:
                    await self._finish_physical_control_operation(
                        asyncio.to_thread(
                            self._publish_p2d_host_offer,
                            p2d_snapshot,
                            rooms,
                            prompt_tokens,
                            domain,
                            source="selector_backpressure",
                        )
                    )
                    p2d_offer_published = True

                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting {self.ready_timeout}s for feasible "
                        f"Decode capacity for P-ready rooms {rooms} "
                        f"({admission_tokens} tokens)"
                    )
                if now >= next_wait_log:
                    logger.info(
                        "PD_LATE_BIND_WAIT rooms=%s admission_tokens=%d: " "all D workers full",
                        rooms,
                        admission_tokens,
                    )
                    next_wait_log = now + 5.0
                await asyncio.sleep(self.no_capacity_poll_interval)
        except BaseException:
            if draining_reservation is not None:
                async with self._selection_lock:
                    self._reservations.pop(draining_reservation.reservation_id, None)
                    self._load_cache_at = 0.0
            if p2d_snapshot is not None:
                self._abort_unsubmitted_p2d(p2d_snapshot, "decode_selection_aborted")
            raise

    async def _release_reservation_when_admitted(
        self, reservation: DecodeReservation, ready_key: Optional[Any] = None
    ) -> None:
        deadline = time.monotonic() + self.reservation_timeout
        admitted = False
        try:
            while time.monotonic() < deadline:
                # D removes P-ready only after it has allocated destination KV.
                if all(not self._ready_path(room).exists() for room in reservation.rooms):
                    admitted = True
                    return
                await asyncio.sleep(self.ready_poll_interval)
            logger.warning(
                "Late-binding reservation timed out: rooms=%s D=%s",
                reservation.rooms,
                reservation.url,
            )
        finally:
            if ready_key is not None:
                self._p_ready_submitted_sequences.discard(ready_key)
            async with self._selection_lock:
                if admitted:
                    self._admitted_reservation_at[reservation.reservation_id] = time.monotonic()
                else:
                    self._reservations.pop(reservation.reservation_id, None)

    async def _late_dispatch(
        self,
        session: aiohttp.ClientSession,
        modified_request: dict[str, Any],
        prefill_server: str,
        endpoint: str,
        headers: dict[str, str],
    ):
        rooms = self._rooms(modified_request)
        sampling = modified_request.get("sampling_params")
        custom_params = sampling.get("custom_params") if isinstance(sampling, dict) else None
        try:
            metadata = AgenticRequestMetadata.from_custom_params(custom_params)
            # Chat Completions carries lifecycle metadata at the request top
            # level and in the router-safe extra_key envelope, rather than in
            # Generate's sampling_params.  All later route watching depends on
            # recognizing the parent here; otherwise Router chooses an
            # arbitrary P while D may publish the Slow snapshot to a different
            # NUMA-local P, leaving both sides waiting forever.
            if metadata is None:
                envelope = unpack_agentic_extra_key(modified_request.get("extra_key"))
                if envelope is not None:
                    _, envelope_params = envelope
                    metadata = AgenticRequestMetadata.from_custom_params(envelope_params)
        except (TypeError, ValueError):
            metadata = None
        parent_turn = metadata is not None and metadata.parent is not None
        prefill_work: Optional[_PrefillWorkReservation] = None
        # Preserve the application-visible tool-return time independently of
        # the short HTTP/tokenizer admission gate.  Dynamic routing publishes
        # an untargeted marker immediately so D can observe the tool ACK; once
        # a P is selected, the targeted marker is published before admission
        # so that P's exact-size workset allocator can claim Direct KV without
        # waiting behind unrelated HTTP requests.
        arrival_at: Optional[float] = time.time() if parent_turn else None
        if getattr(self, "dynamic_prefill_domains", False):
            self._ensure_prefill_pressure_monitor()
            # Notify D immediately that the tool result has returned.  The
            # marker starts untargeted; Router then charges its local shadow
            # queues and targets the lighter P for Direct.  A failed Direct is
            # moved to the D worker's NUMA-local P when host_ready appears.
            arrival = self._publish_parent_arrival(modified_request)
            if arrival is not None:
                arrival_at = float(arrival["arrived_at"])
            prefill_work = await self._resolve_dynamic_prefill_work(
                modified_request,
                metadata,
                arrival_at,
            )
            domain = prefill_work.domain
        else:
            domain = self._request_domain(metadata, rooms)
        self._set_prefill_attempt_rid(modified_request, replace=False)
        route_task: Optional[asyncio.Task] = None
        prefill_task: Optional[asyncio.Task] = None
        decode_task: Optional[asyncio.Task] = None
        admission_task: Optional[asyncio.Task] = None
        p2d_attempt_snapshot: Optional[str] = None
        reservation: Optional[DecodeReservation] = None
        ready_sequence: Optional[int] = None
        ready_key: Optional[Any] = None
        parent_admission: Optional[_PrefillAdmissionGate] = None
        parent_admission_domain: Optional[int] = None

        async def release_parent_admission() -> None:
            nonlocal parent_admission
            nonlocal parent_admission_domain
            if parent_admission is None:
                return
            gate = parent_admission
            parent_admission = None
            parent_admission_domain = None
            await gate.release()

        try:
            while True:
                if getattr(self, "numa_domains", False):
                    prefill_server = self._bind_prefill_domain(modified_request, domain)
                rooms = self._rooms(modified_request)
                self._activate_prefill_attempt(modified_request, rooms)
                prefill_admission = self._prefill_admission_for_domain(domain)
                if parent_turn:
                    # Physical KV admission and HTTP/tokenizer admission have
                    # deliberately separate lifetimes.  Publishing the target
                    # first is safe because P claims DIRECT_READY only after
                    # atomically leasing the complete parent+suffix workset.
                    # If that lease is unavailable, P leaves the manifest
                    # untouched and D takes the ordinary timeout-to-Slow path.
                    self._publish_parent_arrival(
                        modified_request,
                        target_prefill_domain=(domain if getattr(self, "dynamic_prefill_domains", False) else None),
                        arrived_at=arrival_at,
                    )
                    if (
                        route_task is None
                        and prefill_work is not None
                        and prefill_work.route_pending
                        and metadata is not None
                    ):
                        route_task = asyncio.create_task(
                            self._watch_dynamic_prefill_route(
                                modified_request,
                                metadata,
                                prefill_work,
                            )
                        )
                    if parent_admission is not None:
                        if parent_admission_domain != domain:
                            raise RuntimeError(
                                "parent admission cannot move before the old " "Prefill attempt is quiescent"
                            )
                        admission_wait = 0.0
                    else:
                        admission_wait = await prefill_admission.acquire(parent_turn=True)
                        parent_admission = prefill_admission
                        parent_admission_domain = domain
                else:
                    admission_wait = await prefill_admission.acquire(parent_turn=False)
                if admission_wait >= 1.0:
                    logger.info(
                        "PD_P_ADMISSION rooms=%s parent_turn=%s wait_s=%.3f " "P=%d active=%d limit=%d",
                        rooms,
                        parent_turn,
                        admission_wait,
                        domain,
                        prefill_admission.active,
                        self.max_prefill_inflight,
                    )
                prefill_submit_at = time.monotonic()
                prefill_task = asyncio.create_task(
                    session.post(
                        f"{prefill_server}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    )
                )
                try:
                    await self._wait_until_prefill_accepted(rooms, prefill_task)
                    accept_wait = time.monotonic() - prefill_submit_at
                    if parent_turn and accept_wait >= 0.1:
                        logger.info(
                            "PD_P_PARENT_ACCEPT rooms=%s wait_s=%.3f " "tokenizer_inflight_limit=%d",
                            rooms,
                            accept_wait,
                            self.max_prefill_inflight,
                        )
                finally:
                    # This gate protects only the short HTTP/tokenizer accept
                    # window.  Parent KV remains protected independently by
                    # its complete-workset lease through Prefill and P->D
                    # handoff, so retaining the admission slot for that whole
                    # lifetime would only delay later Direct claims.
                    if parent_turn:
                        await release_parent_admission()
                    else:
                        await prefill_admission.release()
                try:
                    await self._wait_until_prefill_scheduled(rooms, prefill_task, route_task)
                    prompt_tokens = await self._wait_until_prefill_ready(rooms, prefill_task, route_task)
                    logger.info(
                        "PD_ROUTER_READY_OBSERVED rooms=%s P=%d prompt_tokens=%d",
                        rooms,
                        domain,
                        prompt_tokens,
                    )
                    break
                except _PrefillRedirect as redirect:
                    old_rooms = rooms
                    old_server = prefill_server
                    self._deactivate_prefill_attempt(modified_request, old_rooms)
                    aborted = await self._abort_prefill_attempt(session, old_server, modified_request, prefill_task)
                    if not aborted:
                        raise RuntimeError(
                            "Cannot safely redirect Prefill because abort was not " f"acknowledged by {old_server}"
                        )
                    prefill_task = None
                    # The explicit abort acknowledgement is the quiescence
                    # boundary for the old P.  The short admission lease has
                    # normally already been released at HTTP acceptance; this
                    # remains idempotent for failures before acceptance.
                    await release_parent_admission()
                    for room in old_rooms:
                        self._accepted_path(room).unlink(missing_ok=True)
                        self._scheduled_path(room).unlink(missing_ok=True)
                        self._ready_path(room).unlink(missing_ok=True)
                        self._p_ready_snapshot.pop(room, None)
                    await self._move_prefill_work(prefill_work, redirect.domain)
                    domain = redirect.domain
                    route_task = None
                    self._replace_prefill_attempt_rooms(modified_request)
                    self._set_prefill_attempt_rid(modified_request, replace=True)
                    logger.info(
                        "PD_PREFILL_REDIRECT_SUBMIT route=%s from_P=%s to_P=%d " "old_rooms=%s new_rooms=%s",
                        redirect.route,
                        old_server,
                        domain,
                        old_rooms,
                        self._rooms(modified_request),
                    )
                    continue
            if route_task is not None and not route_task.done():
                route_task.cancel()
                await asyncio.gather(route_task, return_exceptions=True)
            await self._release_prefill_work(prefill_work)
            # Host staging is owned independently by each admission state
            # machine; it cannot gate the path choice of later P-ready work.
            p2d_attempt_snapshot = self._p2d_snapshot_for_rooms(rooms)
            # The request coroutine now becomes a pure producer.  The per-P
            # dispatcher owns FIFO order, future D-capacity credit and the
            # actual Decode POST as one state transition.
            ready_sequence = self._p_ready_sequence(rooms)
            ready_key = (domain, ready_sequence) if getattr(self, "dynamic_prefill_domains", False) else ready_sequence

            async def commit_ready_reservation(
                selected_reservation: DecodeReservation,
            ) -> DecodeReservation:
                nonlocal admission_task
                nonlocal decode_task
                nonlocal p2d_attempt_snapshot
                nonlocal reservation
                reservation = selected_reservation
                if reservation.p2d_host_snapshot_id is not None:
                    self._set_p2d_host_metadata(
                        modified_request,
                        reservation.p2d_host_snapshot_id,
                        int(reservation.prefill_domain),
                    )
                decode_task = asyncio.create_task(
                    session.post(
                        f"{reservation.url}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    )
                )
                # Decode now owns either the native transfer or the Host
                # restore. Its reservation cleanup is the sole lifecycle
                # authority from this point onward.
                p2d_attempt_snapshot = None
                # Preserve P-ready submission order, but never hold the global
                # FIFO lock while one D waits to allocate destination KV.  The
                # independent reservation task keeps capacity charged until D
                # removes the ready marker, so later submissions cannot
                # double-spend that space or globally head-of-line block.
                admission_task = asyncio.create_task(self._release_reservation_when_admitted(reservation, ready_key))
                await asyncio.sleep(0)
                return reservation

            async def dispatch_ready_snapshot() -> DecodeReservation:
                logger.info(
                    "PD_ROUTER_SELECT_ENTER rooms=%s P=%d sequence=%d",
                    rooms,
                    domain,
                    ready_sequence,
                )
                selected_reservation = await self._select_and_reserve_decode(
                    session, modified_request, rooms, prompt_tokens, domain
                )
                if selected_reservation is None:
                    raise RuntimeError("P->D admission returned without a Decode reservation")
                return await commit_ready_reservation(selected_reservation)

            async def prepare_ready_snapshot() -> Optional[DecodeReservation]:
                """Persist one blocked P completion without waiting for D."""

                await self._stage_p2d_until_durable(rooms, prompt_tokens, domain)
                # Both outcomes continue at the request-local commit boundary:
                # HOST_READY means P pages may already be released; REJECTED
                # means the complete P KV is retained.  In neither case may
                # this bounded D2H preparation lane wait for D capacity.
                # ``dispatch_ready_snapshot`` selects Direct versus Host from
                # the authoritative manifest after the FIFO predecessor.
                return None

            await self._dispatch_p_ready_in_order(
                ready_sequence,
                domain,
                dispatch_ready_snapshot,
                request=modified_request,
                rooms=rooms,
                prompt_tokens=prompt_tokens,
                commit=commit_ready_reservation,
                prepare=prepare_ready_snapshot,
            )
            # P has finished its part once its HTTP response arrives.  Unless
            # prompt logprobs were explicitly requested, consume that small
            # response immediately and return its connection to the pool
            # instead of pinning one P socket for the entire Decode lifetime.
            # ``decode_task`` is already running, so this does not serialize P
            # and D execution.
            prefill_response = await prefill_task
            if not modified_request.get("return_logprob", False):
                try:
                    read = getattr(prefill_response, "read", None)
                    if read is not None:
                        await read()
                finally:
                    release = getattr(prefill_response, "release", None)
                    if release is not None:
                        release()
                prefill_response = None
            decode_response = await decode_task
            await admission_task
            return prefill_response, decode_response
        except BaseException:
            await self._release_prefill_work(prefill_work)
            if route_task is not None and not route_task.done():
                route_task.cancel()
                await asyncio.gather(route_task, return_exceptions=True)
            if ready_key is not None:
                self._p_ready_submitted_sequences.discard(ready_key)
            # Once the Decode POST has been submitted, closing its HTTP task is
            # not a destination cleanup fence. Explicit abort makes D remove a
            # request still parked in prealloc/transfer before Router drops the
            # reservation and a retry starts a fresh physical generation.
            if reservation is not None and decode_task is not None:
                await self._abort_decode_attempt(session, reservation.url, modified_request)
            await self._dispose_response_task(prefill_task)
            await self._dispose_response_task(decode_task)
            if admission_task is not None and not admission_task.done():
                admission_task.cancel()
                await asyncio.gather(admission_task, return_exceptions=True)
            if reservation is not None:
                async with self._selection_lock:
                    self._reservations.pop(reservation.reservation_id, None)
                if reservation.p2d_host_snapshot_id is not None and getattr(self, "p2d_host_ledger", None) is not None:
                    # The Host receiver may already own an H2D DMA even when
                    # the HTTP task is cancelled.  Use the same fence-aware
                    # abort protocol as selection cleanup: HOST_READY may
                    # close immediately, while H2D_LOADING/CONSUMED remain
                    # solely under Decode's physical completion authority.
                    self._abort_unsubmitted_p2d(
                        reservation.p2d_host_snapshot_id,
                        "router_dispatch_failed",
                    )
            for room in rooms:
                try:
                    self._accepted_path(room).unlink(missing_ok=True)
                    self._scheduled_path(room).unlink(missing_ok=True)
                    self._ready_path(room).unlink(missing_ok=True)
                except OSError:
                    pass
            await release_parent_admission()
            raise
        finally:
            self._abort_unsubmitted_p2d(p2d_attempt_snapshot, "router_attempt_ended_before_d_submit")
            self._deactivate_prefill_attempt(modified_request, rooms)
            await release_parent_admission()

    async def generate(self, modified_request, prefill_server, decode_server, endpoint) -> ORJSONResponse:
        assert decode_server is None, "Late-binding D must not be selected early"
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        generation_key = self._agentic_generation_key(modified_request)
        if generation_key is None:
            ret_json, status = await self._generate_once(modified_request, prefill_server, endpoint)
        else:
            ret_json, status = await self._generate_singleflight(
                generation_key, modified_request, prefill_server, endpoint
            )
        return ORJSONResponse(content=ret_json, status_code=status)

    async def generate_stream(self, modified_request, prefill_server, decode_server, endpoint="generate"):
        assert decode_server is None, "Late-binding D must not be selected early"

        async def stream_results():
            session = self._backend_http_session()
            prefill_response = decode_response = None
            try:
                prefill_response, decode_response = await self._late_dispatch(
                    session, modified_request, prefill_server, endpoint, {}
                )
                if modified_request.get("return_logprob", False):
                    assert prefill_response is not None
                    prefill_chunks = [chunk async for chunk in prefill_response.content]
                    first = orjson.loads(prefill_chunks[0].decode("utf-8")[5:].strip())
                    async for chunk in decode_response.content:
                        decoded = chunk.decode("utf-8")
                        if decoded.startswith("data:") and "[DONE]" not in decoded:
                            result = orjson.loads(decoded[5:].strip())
                            result["meta_info"]["input_token_logprobs"] = (
                                first["meta_info"]["input_token_logprobs"]
                                + result["meta_info"]["input_token_logprobs"]
                            )
                            yield b"data: " + orjson.dumps(result) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_response.content.iter_chunked(AIOHTTP_STREAM_READ_CHUNK_SIZE):
                        yield chunk
            finally:
                if prefill_response is not None:
                    prefill_response.release()
                if decode_response is not None:
                    decode_response.release()

        return StreamingResponse(stream_results(), media_type="text/event-stream")
