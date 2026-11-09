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
    RequestGeneration,
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
from sglang_router.mini_lb import (
    AIOHTTP_STREAM_READ_CHUNK_SIZE,
    MiniLoadBalancer,
)


logger = logging.getLogger(__name__)

# Historical public records/helpers remain aliases to one implementation.
from .local_types import (
    DecodeLoad,
    DecodeReservation,
    _PrefillWorkReservation,
    _PrefillAdmissionWaiter,
    _PReadyAdmission,
    _GenerationResponse,
    _PrefillRedirect,
    _HOST_STAGE_EVICTING,
    _HOST_STAGE_RECOMPUTE_REQUIRED,
)
from .local_config import _env_float, _env_int, _env_bool
from .local_capacity import LocalCapacityMixin, _sync_json_get
from .local_admission import LocalAdmissionMixin, _PrefillAdmissionGate
from .local_http import LocalHTTPMixin


class LateBindingMiniLoadBalancer(
    LocalCapacityMixin, LocalAdmissionMixin, LocalHTTPMixin, MiniLoadBalancer
):
    """Dispatch P first and bind D only after the exact P workload is ready."""

    def __init__(self, router_args):
        super().__init__(router_args)
        ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
        if not ready_dir:
            raise ValueError(
                "Late-binding requires SGLANG_PD_P_READY_DIR on router, P and D"
            )
        self.p_ready_dir = Path(ready_dir)
        self.p_ready_dir.mkdir(parents=True, exist_ok=True)
        self.ready_timeout = _env_float("SGLANG_PD_LATE_BIND_READY_TIMEOUT_S", 600.0)
        self.ready_poll_interval = _env_float(
            "SGLANG_PD_LATE_BIND_POLL_INTERVAL_S", 0.02
        )
        self.max_prefill_inflight = _env_int(
            "SGLANG_PD_LATE_BIND_MAX_PREFILL_INFLIGHT", 64
        )
        self.prefill_accept_timeout = _env_float(
            "SGLANG_PD_LATE_BIND_ACCEPT_TIMEOUT_S", self.ready_timeout
        )
        self.prefill_queue_timeout = _env_float(
            "SGLANG_PD_LATE_BIND_QUEUE_TIMEOUT_S", 3600.0
        )
        # Admission bounds the small HTTP/tokenizer bootstrap window in front
        # of each P.  A single global gate unnecessarily couples independent P
        # workers: pressure on P0 must not consume P1's admission capacity.
        self._prefill_admissions = [
            _PrefillAdmissionGate(self.max_prefill_inflight)
            for _ in self.prefill_urls
        ]
        # Retain the legacy attribute for single-P users and lightweight test
        # fixtures constructed without __init__.
        self._prefill_admission = self._prefill_admissions[0]
        self.load_timeout = _env_float("SGLANG_PD_LATE_BIND_LOAD_TIMEOUT_S", 2.0)
        self.reservation_timeout = _env_float(
            "SGLANG_PD_LATE_BIND_RESERVATION_TIMEOUT_S", 120.0
        )
        self.decode_headroom_tokens = _env_int(
            "SGLANG_PD_LATE_BIND_DECODE_HEADROOM_TOKENS", 512
        )
        self.max_decode_admission_tokens = _env_int(
            "SGLANG_PD_LATE_BIND_MAX_ADMISSION_TOKENS", 4096
        )
        self.request_load_weight = _env_float(
            "SGLANG_PD_LATE_BIND_REQUEST_LOAD_WEIGHT", 0.05
        )
        self.transfer_request_weight = _env_float(
            "SGLANG_PD_LATE_BIND_TRANSFER_REQUEST_WEIGHT", 2.0
        )
        # A D with free KV is not necessarily able to start another native
        # P->D Direct handoff: Decode separately bounds receivers whose
        # destination metadata / DMA has not reached a terminal state.  Keep
        # the Router's late-binding decision within that same admission
        # window.  Zero preserves the historical unbounded Router policy.
        self.max_direct_transfer_inflight = _env_int(
            "SGLANG_PD_LATE_BIND_MAX_DIRECT_TRANSFER_INFLIGHT", 0
        )
        if self.max_direct_transfer_inflight < 0:
            raise ValueError(
                "SGLANG_PD_LATE_BIND_MAX_DIRECT_TRANSFER_INFLIGHT must be >= 0"
            )
        self.context_token_floor = _env_int(
            "SGLANG_PD_LATE_BIND_CONTEXT_TOKEN_FLOOR", 2048
        )
        self.context_token_ceiling = _env_int(
            "SGLANG_PD_LATE_BIND_CONTEXT_TOKEN_CEILING", 8192
        )
        self.wait_for_feasible_decode = _env_bool(
            "SGLANG_PD_LATE_BIND_WAIT_FOR_FEASIBLE", True
        )
        self.target_decode_kv_fraction = _env_float(
            "SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION", 1.0
        )
        if not (0.0 < self.target_decode_kv_fraction <= 1.0):
            raise ValueError(
                "SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION must be in (0, 1]"
            )
        self.no_capacity_poll_interval = _env_float(
            "SGLANG_PD_LATE_BIND_NO_CAPACITY_POLL_S", 0.01
        )
        self.load_cache_ttl = _env_float(
            "SGLANG_PD_LATE_BIND_LOAD_CACHE_TTL_S", 0.20
        )
        self._selection_lock = asyncio.Lock()
        # Request coroutines only publish immutable P-ready admissions.  One
        # broker owns D-capacity accounting for every P, scans in completion
        # order, and pipelines multiple capacity-feasible generations.  The
        # sequence is a fairness hint: one Host-staged generation must not
        # block an unrelated generation that can enter D immediately.
        self._p_ready_submitted_sequences: set[Any] = set()
        self._p_ready_fifo_waiters: dict[
            int, dict[int, _PReadyAdmission]
        ] = {}
        self._p_ready_fifo_events: dict[int, asyncio.Event] = {}
        self._p_ready_fifo_dispatchers: dict[int, asyncio.Task] = {}
        self._p_ready_fifo_active: dict[int, dict[int, _PReadyAdmission]] = {}
        self._p_ready_broker_event = asyncio.Event()
        self._p_ready_broker_task: Optional[asyncio.Task] = None
        self._p_ready_commit_tails: dict[int, asyncio.Future] = {}
        self._p_ready_admission_window_per_p = _env_int(
            "SGLANG_PD_LATE_BIND_ADMISSION_WINDOW_PER_P", 32
        )
        self._p_ready_stage_lanes_per_p = _env_int(
            "SGLANG_AGENTIC_KV_P2D_D2H_WORKERS", 4
        )
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
            set(self.decode_urls)
            if _env_bool("SGLANG_PD_LATE_BIND_FORCE_LEGACY_LOADS")
            else set()
        )
        self._prefill_index = 0
        # Ablation-only policy: preserve all capacity/ownership checks, but
        # replace both load-aware P and D choices with deterministic random
        # choices among the currently feasible workers.
        self.ablation_random_routing = _env_bool(
            "SGLANG_PD_ABLATION_RANDOM_ROUTING"
        )
        self._routing_rng = random.Random(
            _env_int("SGLANG_PD_ABLATION_RANDOM_SEED", 2026)
        )
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
        self._prefill_pressure_domains: list[dict[str, Any]] = []
        self._prefill_pressure_at = 0.0
        # Start time of the HTTP sample currently represented by
        # ``_prefill_pressure_domains``.  Direct shadow credit is retained
        # until a sample started after the physical Direct completion has
        # observed the new P allocation.  This makes the hand-off from Router
        # accounting to allocator accounting atomic from later selectors'
        # point of view.
        self._prefill_pressure_sample_started_at = 0.0
        self._prefill_pressure_interval = _env_float(
            "SGLANG_AGENTIC_KV_PREFILL_LOAD_INTERVAL_S", 0.20
        )
        pressure_path = os.getenv("SGLANG_AGENTIC_KV_PREFILL_LOAD_PATH", "")
        if not pressure_path:
            pressure_path = str(
                self.p_ready_dir / "early-claims" / "prefill-loads.json"
            )
        self._prefill_pressure_path = Path(pressure_path)
        staging_path = os.getenv("SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH", "")
        self._d2p_host_ledger = (
            SharedHostStagingLedger(staging_path) if staging_path else None
        )
        self._multinode_static_recovery = _env_bool(
            "SGLANG_AGENTIC_MULTINODE_ENABLED", False
        )
        if self._multinode_static_recovery and len(self.prefill_urls) != 1:
            raise ValueError("multi-node V1 requires exactly one logical Prefill group")
        self._multinode_recovery_parents: dict[str, int] = {}
        self._slow_congestion = None
        if os.getenv("SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE", "0").lower() in {"1", "true"}:
            from sglang.srt.disaggregation.agentic_slow_congestion import SlowRecoveryCongestion
            lanes = max(1, _env_int("SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT", 2)) * len(self.prefill_urls)
            self._slow_congestion = SlowRecoveryCongestion(
                _env_int("SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH", 4 * lanes),
                _env_int("SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW", lanes),
            )
        # One logical request-generation may outlive an HTTP client's timeout.
        # Keep the actual P->D dispatch detached from that client and let every
        # retry await the same task.  Without this fence, a retry can select a
        # second D and duplicate both Decode compute and KV lifecycle writes.
        self._generation_lock = asyncio.Lock()
        self._generation_tasks: dict[
            str, asyncio.Task[tuple[dict[str, Any], int]]
        ] = {}
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
        self.generation_result_ttl = _env_float(
            "SGLANG_PD_GENERATION_RESULT_TTL_S", 3600.0
        )
        self.max_generation_results = _env_int(
            "SGLANG_PD_MAX_GENERATION_RESULTS", 8192
        )
        self.numa_domains = _env_bool("SGLANG_PD_LATE_BIND_NUMA_DOMAINS")
        self.dynamic_prefill_domains = _env_bool(
            "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS"
        )
        self.global_decode = _env_bool("SGLANG_PD_LATE_BIND_GLOBAL_DECODE")
        # Ablation-only classic P->D ordering: reserve and submit one Decode
        # destination before Prefill starts.  The custom Router stays active,
        # so D->P Direct/Slow routing and request-generation ownership are
        # otherwise identical to the full method.
        self.p2d_prebind_ablation = _env_bool(
            "SGLANG_PD_ABLATION_P2D_PREBIND"
        )
        # P->D destination-first ablation.  D's native allocator remains the
        # only capacity authority, but P is not submitted until D publishes
        # proof that the matching request owns physical destination KV.
        self.p2d_native_prealloc_ablation = _env_bool(
            "SGLANG_PD_ABLATION_P2D_NATIVE_PREALLOC"
        )
        # The stock MiniLB has no D->P placement protocol.  When D-first P/D
        # submission is composed with dynamic D->P placement, a
        # provisional P can be superseded after both HTTP requests already
        # own physical state.  For the P->D ablation, settle that existing
        # D->P decision before starting the destination-first pair.
        self.p2d_native_route_settled = _env_bool(
            "SGLANG_PD_ABLATION_P2D_ROUTE_SETTLED"
        )
        # Pair native D preallocation with one capacity-backed logical P
        # credit. Unlike the ordinary placement shadow, this mode waits for
        # enough uncommitted P HBM for the request's complete workset before
        # submitting its matching D request. Parent evacuation is still
        # announced immediately, so a missed Direct window drains D->Host
        # without holding the old D while this credit waits.
        self.p2d_paired_credit_ablation = _env_bool(
            "SGLANG_PD_ABLATION_P2D_PAIRED_CREDIT"
        )
        if self.p2d_native_route_settled and not self.p2d_native_prealloc_ablation:
            raise ValueError(
                "P->D route-settled mode requires native-preallocation ablation"
            )
        if self.p2d_paired_credit_ablation and not (
            self.p2d_native_prealloc_ablation
            and self.p2d_native_route_settled
            and self.dynamic_prefill_domains
        ):
            raise ValueError(
                "P->D paired-credit mode requires native preallocation, "
                "route settlement, and dynamic Prefill domains"
            )
        self.p2d_host_staging = _env_bool(
            "SGLANG_AGENTIC_KV_P2D_HOST_STAGING"
        )
        if self.p2d_prebind_ablation and self.p2d_native_prealloc_ablation:
            raise ValueError(
                "P->D pre-binding and native-preallocation ablations are "
                "mutually exclusive"
            )
        if (
            self.p2d_prebind_ablation
            or self.p2d_native_prealloc_ablation
        ) and self.p2d_host_staging:
            raise ValueError(
                "P->D preallocation ablation requires P->D Host staging disabled"
            )
        self.p2d_host_spill_delay = _env_float(
            "SGLANG_AGENTIC_KV_P2D_SPILL_DELAY_SECONDS", 0.5
        )
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
            self._p2d_host_offered_snapshots.update(
                self.p2d_host_ledger.snapshot_entries()
            )
        if self.numa_domains and len(self.prefill_urls) < 2:
            raise ValueError(
                "NUMA-domain routing requires at least two P workers"
            )
        if (
            self.numa_domains
            and not self.global_decode
            and len(self.decode_urls) % len(self.prefill_urls) != 0
        ):
            raise ValueError(
                "NUMA-local Decode routing requires an equal contiguous D "
                "partition per P; set SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1 "
                "to share arbitrary D counts across P domains"
            )
        if self.dynamic_prefill_domains and not self.numa_domains:
            raise ValueError(
                "dynamic Prefill domains require SGLANG_PD_LATE_BIND_NUMA_DOMAINS"
            )
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
