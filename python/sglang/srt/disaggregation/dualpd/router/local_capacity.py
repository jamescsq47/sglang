"""Capacity sampling and shadow reservations on the one concrete Router instance."""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional
import aiohttp
import orjson
from sglang.srt.disaggregation.agentic_host_staging import HostStageState

from .local_types import (
    DecodeLoad,
    DecodeReservation,
    _HOST_STAGE_EVICTING,
    _PReadyAdmission,
    _PrefillWorkReservation,
    logger,
)

def _sync_json_get(url: str, timeout: float) -> Any:
    """Fetch one tiny control-plane JSON document outside the ASGI loop."""

    request = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return orjson.loads(response.read())


class LocalCapacityMixin:
    """Method-only component; constructor, locks and queues belong to the Router."""

    def _choose_decode_score(self, candidates, *, drain: bool = False):
        """Choose one capacity-accounted D score for normal or ablation runs."""

        if getattr(self, "ablation_random_routing", False):
            # Sort first so a fixed ablation seed is independent of response
            # arrival order from the asynchronous load probes.
            return self._routing_rng.choice(
                sorted(candidates, key=lambda item: item[4].url)
            )
        if drain:
            return min(
                candidates,
                key=lambda item: (item[2], item[1], item[4].url),
            )
        return min(
            candidates,
            key=lambda item: (item[1], item[2], item[4].url),
        )

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
            hint = (
                custom_params.get("agentic_prompt_token_count")
                if isinstance(custom_params, dict)
                else None
            )
            # This is only shadow admission accounting.  Bound the hint to a
            # generous finite range; the model server remains authoritative
            # for actual tokenization and context-length validation.
            if (
                isinstance(hint, int)
                and not isinstance(hint, bool)
                and 0 < hint <= 10_000_000
            ):
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
        direct_workset_tokens: int = 0,
    ) -> _PrefillWorkReservation:
        tokens = max(1, int(tokens))
        direct_workset_tokens = max(0, int(direct_workset_tokens))
        async with self._prefill_work_lock:
            if domain is None:
                count = len(self.prefill_urls)
                start = self._prefill_work_tiebreak % count
                candidates = list(range(count))
                pressure = getattr(self, "_prefill_pressure_domains", [])
                pressure_at = float(getattr(self, "_prefill_pressure_at", 0.0))
                pressure_max_age = max(
                    1.0,
                    5.0 * float(getattr(self, "_prefill_pressure_interval", 0.2)),
                )
                pressure_by_domain = {
                    int(row.get("domain", -1)): row for row in pressure
                }
                direct_pending = getattr(
                    self,
                    "_prefill_direct_pending_tokens",
                    [0] * count,
                )
                pressure_fresh = time.monotonic() - pressure_at <= pressure_max_age
                remaining_by_domain: dict[int, int] = {}
                if pressure_fresh:
                    for candidate in candidates:
                        row = pressure_by_domain.get(candidate)
                        if row is None:
                            continue
                        capacity = int(row.get("hbm_capacity_tokens", 0))
                        if capacity <= 0:
                            continue
                        remaining_by_domain[candidate] = (
                            capacity
                            - int(row.get("hbm_used_tokens", 0))
                            - direct_pending[candidate]
                        )
                    feasible = [
                        candidate
                        for candidate, remaining in remaining_by_domain.items()
                        if remaining >= direct_workset_tokens
                    ]
                    if feasible:
                        candidates = feasible
                    elif remaining_by_domain:
                        candidates = list(remaining_by_domain)
                if getattr(self, "ablation_random_routing", False):
                    domain = self._routing_rng.choice(sorted(candidates))
                elif remaining_by_domain:
                    # D->P routing deliberately uses one cheap signal: choose
                    # the P with the most uncommitted KV capacity.  Local
                    # Router-local full-workset reservations are deducted so
                    # stale samples cannot herd several requests onto one P.
                    # The exact-size allocator remains authoritative.
                    domain = max(
                        candidates,
                        key=lambda candidate: (
                            remaining_by_domain[candidate],
                            -((candidate - start) % count),
                        ),
                    )
                else:
                    # Startup or a stale/missing P sample has no meaningful
                    # remaining-capacity ordering.  Preserve the local shadow
                    # fallback until the next asynchronous sample arrives.
                    domain = min(
                        candidates,
                        key=lambda candidate: (
                            self._prefill_pending_tokens[candidate],
                            (candidate - start) % count,
                        ),
                    )
                self._prefill_work_tiebreak = (domain + 1) % count
            if not 0 <= domain < len(self.prefill_urls):
                raise RuntimeError(f"invalid Prefill domain {domain}")
            self._prefill_pending_tokens[domain] += tokens
            self._prefill_pending_requests[domain] += 1
            if not hasattr(self, "_prefill_direct_pending_tokens"):
                self._prefill_direct_pending_tokens = [0] * len(self.prefill_urls)
            self._prefill_direct_pending_tokens[domain] += direct_workset_tokens
            reservation = _PrefillWorkReservation(
                domain=domain,
                tokens=tokens,
                direct_workset_tokens=direct_workset_tokens,
            )
            logger.info(
                "PD_P_WORK_RESERVE P=%d tokens=%d pending_tokens=%d "
                "pending_requests=%d direct_workset_tokens=%d "
                "direct_pending_tokens=%d",
                domain,
                tokens,
                self._prefill_pending_tokens[domain],
                self._prefill_pending_requests[domain],
                direct_workset_tokens,
                self._prefill_direct_pending_tokens[domain],
            )
            return reservation

    async def _reserve_paired_prefill_credit(
        self, request: dict[str, Any]
    ) -> _PrefillWorkReservation:
        """Wait for and atomically claim one feasible full P workset.

        Ordinary D->P placement deliberately falls back to the least-loaded P
        when every sampled P is full. That is useful for normal queueing but
        is not a capacity credit: composing it with native D preallocation can
        fill D with requests whose corresponding P worksets cannot run. This
        ablation admits the D request only after this stricter claim succeeds.
        """

        required = self._request_input_tokens(request)
        last_log_at = 0.0
        while True:
            async with self._prefill_work_lock:
                count = len(self.prefill_urls)
                pressure = getattr(self, "_prefill_pressure_domains", [])
                pressure_at = float(getattr(self, "_prefill_pressure_at", 0.0))
                pressure_max_age = max(
                    1.0,
                    5.0 * float(getattr(self, "_prefill_pressure_interval", 0.2)),
                )
                pressure_by_domain = {
                    int(row.get("domain", -1)): row for row in pressure
                }
                direct_pending = getattr(
                    self, "_prefill_direct_pending_tokens", [0] * count
                )
                pressure_fresh = time.monotonic() - pressure_at <= pressure_max_age
                remaining_by_domain: dict[int, int] = {}
                if pressure_fresh:
                    for candidate in range(count):
                        row = pressure_by_domain.get(candidate)
                        if row is None:
                            continue
                        capacity = int(row.get("hbm_capacity_tokens", 0))
                        if capacity <= 0:
                            continue
                        remaining_by_domain[candidate] = (
                            capacity
                            - int(row.get("hbm_used_tokens", 0))
                            - direct_pending[candidate]
                        )
                feasible = [
                    candidate
                    for candidate, remaining in remaining_by_domain.items()
                    if remaining >= required
                ]
                if feasible:
                    start = self._prefill_work_tiebreak % count
                    domain = max(
                        feasible,
                        key=lambda candidate: (
                            remaining_by_domain[candidate],
                            -((candidate - start) % count),
                        ),
                    )
                    self._prefill_work_tiebreak = (domain + 1) % count
                    self._prefill_pending_tokens[domain] += required
                    self._prefill_pending_requests[domain] += 1
                    self._prefill_direct_pending_tokens[domain] += required
                    reservation = _PrefillWorkReservation(
                        domain=domain,
                        tokens=required,
                        direct_workset_tokens=required,
                        route_pending=True,
                    )
                    logger.info(
                        "PD_PAIRED_P_CREDIT P=%d tokens=%d remaining_before=%d "
                        "direct_pending_tokens=%d",
                        domain,
                        required,
                        remaining_by_domain[domain],
                        self._prefill_direct_pending_tokens[domain],
                    )
                    return reservation
            now = time.monotonic()
            if now - last_log_at >= 10.0:
                logger.info(
                    "PD_PAIRED_P_CREDIT_WAIT tokens=%d pressure_fresh=%s "
                    "remaining=%s",
                    required,
                    pressure_fresh,
                    remaining_by_domain,
                )
                last_log_at = now
            await asyncio.sleep(self.ready_poll_interval)

    async def _settle_direct_workset(
        self, reservation: Optional[_PrefillWorkReservation]
    ) -> None:
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
            self._prefill_direct_pending_tokens[domain] -= (
                reservation.direct_workset_tokens
            )
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

        if reservation is None or reservation.direct_workset_tokens <= 0:
            return
        poll_interval = max(
            0.01,
            min(
                0.05,
                float(getattr(self, "_prefill_pressure_interval", 0.20)),
            ),
        )
        while not reservation.released and reservation.direct_workset_tokens > 0:
            if (
                float(
                    getattr(self, "_prefill_pressure_sample_started_at", 0.0)
                )
                >= direct_terminal_at
            ):
                await self._settle_direct_workset(reservation)
                return
            await asyncio.sleep(poll_interval)

    async def _fetch_prefill_hbm_pressure(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[int, int]:
        """Read one logical P without putting the query on its GPU loop."""

        timeout = aiohttp.ClientTimeout(total=self.load_timeout)
        async with session.get(f"{url}/get_load", timeout=timeout) as response:
            response.raise_for_status()
            rows = await response.json()
        if isinstance(rows, dict):
            rows = [rows]
        used = sum(
            int(row.get("num_physical_used_tokens", row.get("num_tokens", 0)))
            for row in rows
        )
        capacity = sum(int(row.get("max_total_num_tokens", 0)) for row in rows)
        return used, capacity

    @staticmethod
    def _write_prefill_pressure(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
        temporary.write_bytes(orjson.dumps(payload))
        os.replace(temporary, path)

    def _prefill_arena_bytes(self, recovery_parents=None):
        used = [0] * len(self.prefill_urls)
        ledger = self._d2p_host_ledger
        if ledger is None:
            return used if recovery_parents is None else (used, 0)
        q = 0
        if recovery_parents is not None:
            from sglang.srt.disaggregation.agentic_slow_congestion import waiting_for_recovery
        live_states = {
            HostStageState.HOST_RESERVED.value,
            HostStageState.HOST_WRITING.value,
            HostStageState.HOST_READY.value,
            HostStageState.H2D_LOADING.value,
            # The Host extent remains authoritative until the recovered HBM
            # workset has been inserted and pinned, including retry windows.
            HostStageState.HBM_READY.value,
            HostStageState.RETRY_PENDING.value,
            # ABORTING likewise owns its extent until D is quiescent.
            HostStageState.ABORTING.value,
        }
        # EVICTING still owns its TP-local extents; RECOMPUTE_REQUIRED does
        # not.  Add the former only when this environment exposes it.
        if _HOST_STAGE_EVICTING is not None:
            live_states.add(_HOST_STAGE_EVICTING.value)
        for snapshot_id, entry in ledger.snapshot_entries().items():
            # Source-local D Host storage has a different physical owner from
            # the only P group. Commit its recovery destination through the
            # normal ledger CAS before P can import/pin it. Reuse this existing
            # background scan; never wait for Host readiness in HTTP dispatch.
            if (
                getattr(self, "_multinode_static_recovery", False)
                and snapshot_id in getattr(self, "_multinode_recovery_parents", {})
                and entry.get("source_host_node")
                and entry.get("recovery_domain") is None
                and entry.get("state") == HostStageState.HOST_READY.value
            ):
                ledger.assign_d2p_recovery_domain(snapshot_id, 0)
            # Same existing ledger read/iteration, unique generation IDs;
            # never sum per-P caches or scan arrival marker directories.
            if recovery_parents is not None and snapshot_id in recovery_parents:
                q += int(waiting_for_recovery(entry))
            if entry.get("state") not in live_states:
                continue
            domain = int(entry.get("arena_domain", -1))
            if 0 <= domain < len(used):
                used[domain] += int(entry.get("byte_size", 0))
        return used if recovery_parents is None else (used, q)

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
            ledger_entries = (
                {} if ledger is None else ledger.snapshot_entries()
            )
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
                rows[domain]["p2d_host_tokens"] += max(
                    0, int(entry.get("token_count", 0))
                )
                rows[domain]["p2d_host_requests"] += 1
                rows[domain]["p2d_host_bytes"] += max(
                    0, int(entry.get("byte_size", 0))
                )

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
                    if (
                        p2d_snapshot is not None
                        and p2d_snapshot in durable_host_snapshots
                    ):
                        continue
                    key = (int(domain), int(admission.sequence))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows[int(domain)]["p2d_inflight_tokens"] += max(
                        0, int(admission.prompt_tokens)
                    )
                    rows[int(domain)]["p2d_inflight_requests"] += max(
                        1, len(admission.rooms)
                    )
        for reservation in getattr(self, "_reservations", {}).values():
            domain = reservation.prefill_domain
            if domain is None or not 0 <= int(domain) < count:
                continue
            p2d_snapshot = reservation.p2d_host_snapshot_id
            if p2d_snapshot is None:
                p2d_snapshot = self._p2d_snapshot_for_rooms(reservation.rooms)
            if p2d_snapshot in durable_host_snapshots:
                continue
            rows[int(domain)]["p2d_inflight_tokens"] += max(
                0, int(reservation.prompt_tokens)
            )
            rows[int(domain)]["p2d_inflight_requests"] += max(
                1, int(reservation.request_count)
            )
        return rows

    async def _prefill_pressure_monitor_loop(self) -> None:
        """Publish independent P-HBM and Shared-Host capacity signals."""

        tp_size = max(1, int(os.getenv("SGLANG_AGENTIC_KV_TP_SIZE", "1")))
        arena_capacity = int(
            float(os.getenv("SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB", "128"))
            * (1024**3)
            * tp_size
        )
        async with aiohttp.ClientSession() as session:
            while True:
                sample_started_at = time.monotonic()
                fetched = await asyncio.gather(
                    *(
                        self._fetch_prefill_hbm_pressure(session, url)
                        for url in self.prefill_urls
                    ),
                    return_exceptions=True,
                )
                policy = getattr(self, "_slow_congestion", None)
                if policy is None:
                    arena_used = await asyncio.to_thread(self._prefill_arena_bytes)
                else:
                    arena_used, q = await asyncio.to_thread(
                        self._prefill_arena_bytes, frozenset(policy.parents)
                    )
                domains = []
                for domain, result in enumerate(fetched):
                    if isinstance(result, BaseException):
                        used = capacity = 0
                    else:
                        used, capacity = result
                    domains.append(
                        {
                            "domain": domain,
                            "hbm_used_tokens": used,
                            "hbm_capacity_tokens": capacity,
                            "arena_used_bytes": arena_used[domain],
                            "arena_capacity_bytes": arena_capacity,
                        }
                    )
                payload = {
                    "version": 1,
                    "published_at": time.time(),
                    "domains": domains,
                }
                if policy is not None:
                    previous = policy.latest
                    payload["slow_recovery"] = policy.sample(q)
                    if policy.latest is not previous:
                        logger.info("PD_SLOW_CONGESTION q=%d congested=%s high=%d low=%d",
                                    q, policy.blocked, policy.high, policy.low)
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

    def _ensure_prefill_pressure_monitor(self) -> None:
        if not getattr(self, "dynamic_prefill_domains", False) and getattr(
            self, "_slow_congestion", None
        ) is None and not getattr(self, "_multinode_static_recovery", False):
            return
        if not hasattr(self, "_prefill_pressure_interval"):
            self._prefill_pressure_interval = 0.20
        if not hasattr(self, "_prefill_pressure_sample_started_at"):
            self._prefill_pressure_sample_started_at = 0.0
        if not hasattr(self, "_prefill_pressure_path"):
            self._prefill_pressure_path = (
                self.p_ready_dir / "early-claims" / "prefill-loads.json"
            )
        if not hasattr(self, "_d2p_host_ledger"):
            self._d2p_host_ledger = None
        task = getattr(self, "_prefill_pressure_task", None)
        if task is None or task.done():
            self._prefill_pressure_task = asyncio.create_task(
                self._prefill_pressure_monitor_loop()
            )

    async def _move_prefill_work(
        self, reservation: _PrefillWorkReservation, domain: int
    ) -> None:
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
                self._prefill_direct_pending_tokens[previous] -= (
                    reservation.direct_workset_tokens
                )
                self._prefill_direct_pending_tokens[domain] += (
                    reservation.direct_workset_tokens
                )
            reservation.domain = domain
            logger.info(
                "PD_P_WORK_MOVE from_P=%d to_P=%d tokens=%d pending_tokens=%s",
                previous,
                domain,
                reservation.tokens,
                self._prefill_pending_tokens,
            )

    async def _move_prefill_work_to_max_remaining_kv(
        self,
        reservation: _PrefillWorkReservation,
        required_workset_tokens: int,
    ) -> int:
        """Atomically retarget an existing request to the roomiest P HBM."""

        required_workset_tokens = max(1, int(required_workset_tokens))
        async with self._prefill_work_lock:
            if reservation.released:
                raise RuntimeError("cannot route a released Prefill reservation")
            count = len(self.prefill_urls)
            pressure = getattr(self, "_prefill_pressure_domains", [])
            pressure_at = float(getattr(self, "_prefill_pressure_at", 0.0))
            pressure_max_age = max(
                1.0,
                5.0 * float(getattr(self, "_prefill_pressure_interval", 0.2)),
            )
            if time.monotonic() - pressure_at > pressure_max_age:
                pressure = []
            pressure_by_domain = {
                int(row.get("domain", -1)): row for row in pressure
            }
            remaining = {}
            for domain in range(count):
                row = pressure_by_domain.get(domain)
                if row is None:
                    continue
                capacity = int(row.get("hbm_capacity_tokens", 0))
                if capacity <= 0:
                    continue
                pending = self._prefill_direct_pending_tokens[domain]
                if domain == reservation.domain:
                    pending -= reservation.direct_workset_tokens
                remaining[domain] = (
                    capacity - int(row.get("hbm_used_tokens", 0)) - pending
                )
            feasible = [
                domain
                for domain, available in remaining.items()
                if available >= required_workset_tokens
            ]
            candidates = feasible or list(remaining)
            if candidates:
                domain = max(candidates, key=lambda item: (remaining[item], -item))
            else:
                domain = min(
                    range(count), key=lambda item: self._prefill_pending_tokens[item]
                )
            if domain != reservation.domain:
                previous = reservation.domain
                self._prefill_pending_tokens[previous] -= reservation.tokens
                self._prefill_pending_requests[previous] -= reservation.requests
                self._prefill_direct_pending_tokens[previous] -= (
                    reservation.direct_workset_tokens
                )
                self._prefill_pending_tokens[domain] += reservation.tokens
                self._prefill_pending_requests[domain] += reservation.requests
                self._prefill_direct_pending_tokens[domain] += (
                    reservation.direct_workset_tokens
                )
                reservation.domain = domain
                logger.info(
                    "PD_P_WORK_MOVE_MAX_REMAINING from_P=%d to_P=%d "
                    "workset_tokens=%d",
                    previous,
                    domain,
                    required_workset_tokens,
                )
            return domain

    async def _resize_prefill_work(
        self, reservation: _PrefillWorkReservation, tokens: int
    ) -> None:
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
                "PD_P_WORK_RESIZE P=%d old_tokens=%d tokens=%d "
                "pending_tokens=%d",
                domain,
                previous,
                tokens,
                self._prefill_pending_tokens[domain],
            )

    async def _release_prefill_work(
        self, reservation: Optional[_PrefillWorkReservation]
    ) -> None:
        if reservation is None or reservation.released:
            return
        async with self._prefill_work_lock:
            if reservation.released:
                return
            domain = reservation.domain
            self._prefill_pending_tokens[domain] -= reservation.tokens
            self._prefill_pending_requests[domain] -= reservation.requests
            self._prefill_direct_pending_tokens[domain] -= (
                reservation.direct_workset_tokens
            )
            reservation.direct_workset_tokens = 0
            reservation.released = True
            if (
                self._prefill_pending_tokens[domain] < 0
                or self._prefill_pending_requests[domain] < 0
            ):
                raise RuntimeError("Prefill shadow queue accounting underflow")
            logger.info(
                "PD_P_WORK_RELEASE P=%d tokens=%d pending_tokens=%d "
                "pending_requests=%d",
                domain,
                reservation.tokens,
                self._prefill_pending_tokens[domain],
                self._prefill_pending_requests[domain],
            )

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
        if (
            p2d_snapshot is not None
            and p2d_snapshot
            in getattr(self, "_p2d_host_offered_snapshots", set())
        ):
            return None

        loads = [
            load
            for load in loads_snapshot
            if load.url in self._domain_decode_urls(admission.domain)
        ]
        if not loads:
            return None

        requested_decode = self._requested_decode_tokens(admission.request)
        decode_headroom = self.decode_headroom_tokens
        if requested_decode is not None:
            decode_headroom = min(
                requested_decode, self.max_decode_admission_tokens
            )
        admission_tokens = admission.prompt_tokens + decode_headroom * len(
            admission.rooms
        )
        average_context = self._average_running_context(loads)
        transfer_weight = max(1.0, getattr(self, "transfer_request_weight", 2.0))
        scored: list[tuple[bool, float, float, int, DecodeLoad]] = []
        for load in loads:
            reserved_prompt, reserved_admission, reserved_reqs = self._reserved_for(
                load.url
            )
            free_after_pending = (
                load.capacity_tokens - load.used_tokens - reserved_admission
            )
            projected_kv = (
                load.used_tokens + reserved_admission + admission_tokens
            ) / load.capacity_tokens
            transfer_feasible = (
                self.max_direct_transfer_inflight == 0
                or load.prealloc + load.transfer + reserved_reqs
                < self.max_direct_transfer_inflight
            )
            feasible = (
                free_after_pending >= admission_tokens
                and projected_kv
                <= getattr(self, "target_decode_kv_fraction", 1.0)
                and transfer_feasible
            )
            queued_or_handoff_reqs = max(
                load.waiting, load.prealloc + load.transfer
            )
            projected_decode_reqs = (
                load.running
                + queued_or_handoff_reqs
                + reserved_reqs
                + len(admission.rooms)
            )
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
        _, work_score, projected_kv, projected_decode_reqs, selected = (
            self._choose_decode_score(candidates)
        )
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

    async def _fetch_decode_load(
        self, session: aiohttp.ClientSession, url: str
    ) -> DecodeLoad:
        if url in self._legacy_load_urls:
            return await self._fetch_decode_load_legacy(session, url)
        endpoint = f"{url}/v1/loads?include=disagg,memory,queues"
        try:
            async with session.get(
                endpoint, timeout=aiohttp.ClientTimeout(total=self.load_timeout)
            ) as response:
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
        running_kv_tokens = sum(
            int(load.get("num_running_kv_tokens", 0)) for load in loads
        )
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

    async def _fetch_decode_load_legacy(
        self, session: aiohttp.ClientSession, url: str
    ) -> DecodeLoad:
        # /get_load is a tiny scheduler-control RPC.  Under c512 the Router's
        # ASGI loop also owns hundreds of long-lived generation coroutines and
        # synchronous tmpfs lifecycle operations; driving this RPC on the same
        # loop caused false 2s timeouts while a separate client reached every
        # D immediately.  Move only the network wait/JSON read to a worker;
        # publication and reservation accounting remain on the Router loop.
        rows = await asyncio.to_thread(
            _sync_json_get, f"{url}/get_load", self.load_timeout
        )
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            raise RuntimeError(f"D legacy load endpoint returned no ranks: {url}")

        reported_capacity = sum(
            int(row.get("max_total_num_tokens", 0)) for row in rows
        )
        reported_max_running = sum(
            int(row.get("max_running_requests", 0)) for row in rows
        )
        cached = self._last_loads.get(url)
        if reported_capacity > 0:
            capacity = reported_capacity
            max_running = reported_max_running
        elif cached is not None:
            capacity = cached.capacity_tokens
            max_running = cached.max_running
        else:
            server_info = await asyncio.to_thread(
                _sync_json_get, f"{url}/server_info", self.load_timeout
            )
            internal = server_info.get("internal_states") or []
            capacity = sum(
                int((state.get("memory_usage") or {}).get("token_capacity", 0))
                for state in internal
            )
            max_running = sum(
                int(state.get("effective_max_running_requests_per_dp", 0))
                for state in internal
            )
        if capacity <= 0:
            raise RuntimeError(f"D server_info returned invalid KV capacity: {url}")

        total_reqs = sum(int(row.get("num_reqs", 0)) for row in rows)
        waiting = sum(int(row.get("num_waiting_reqs", 0)) for row in rows)
        prealloc = sum(int(row.get("decode_prealloc_queue_reqs", 0)) for row in rows)
        transfer = sum(int(row.get("decode_transfer_queue_reqs", 0)) for row in rows)
        physical_used = sum(
            int(row.get("num_physical_used_tokens", row.get("num_tokens", 0)))
            for row in rows
        )
        running_kv_tokens = sum(
            int(row.get("num_running_kv_tokens", 0)) for row in rows
        )
        prealloc_tokens = sum(
            int(row.get("decode_prealloc_queue_tokens", 0)) for row in rows
        )
        transfer_tokens = sum(
            int(row.get("decode_transfer_queue_tokens", 0)) for row in rows
        )
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

    async def _refresh_decode_loads(
        self, session: aiohttp.ClientSession
    ) -> list[DecodeLoad]:
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
        if sample_started_at >= getattr(
            self, "_load_cache_sample_started_at", 0.0
        ):
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

    async def _all_decode_loads(
        self, session: aiohttp.ClientSession, *, force: bool = False
    ) -> list[DecodeLoad]:
        # ``session`` remains in the signature for lightweight tests and old
        # callers, but production load RPCs always use their isolated control
        # connector rather than the long-lived generation data plane.
        load_session = None if session is None else self._load_http_session()
        now = time.monotonic()
        fresh = (
            self._load_cache
            and now - self._load_cache_at < self.load_cache_ttl
        )
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
                self._load_refresh_task = asyncio.create_task(
                    self._refresh_decode_loads_background()
                )
            return self._load_cache

        return await self._refresh_decode_loads(load_session)

    def _decode_loads_are_fresh_since(
        self, urls: set[str], not_before: float
    ) -> bool:
        """Return whether every candidate D was sampled after ``not_before``.

        P->D Host spill costs an extra Host round trip even though recovery can
        now target any D domain.  Do not choose that path from a cached
        all-full observation that predates the P-ready snapshot itself.
        """

        epochs = getattr(self, "_load_sample_started_at_by_url", {})
        return bool(urls) and all(
            float(epochs.get(url, 0.0)) >= not_before for url in urls
        )

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
                task = asyncio.create_task(
                    self._refresh_decode_loads(self._load_http_session())
                )
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

    def _reserved_for(
        self, url: str, *, exclude_id: Optional[str] = None
    ) -> tuple[int, int, int]:
        prompt = admission = requests = 0
        for reservation_id, reservation in self._reservations.items():
            if reservation_id == exclude_id:
                continue
            admitted_at = self._admitted_reservation_at.get(reservation_id)
            sample_epoch = getattr(
                self, "_load_sample_started_at_by_url", {}
            ).get(
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
                f"P->D Host snapshot {snapshot_id} was already consumed before "
                "this Router attempt submitted Decode"
            )
        # Custom P->D staging has no lower storage tier.  Seeing a D->P-only
        # or unknown state here is a control-plane invariant violation, never
        # permission to resurrect the native sender.
        raise RuntimeError(
            f"P->D Host snapshot {snapshot_id} has invalid route state {state!r}"
        )

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
            decode_headroom = min(
                requested_decode, self.max_decode_admission_tokens
            )
        admission_tokens = prompt_tokens + decode_headroom * len(rooms)
        deadline = time.monotonic() + self.ready_timeout
        wait_started = time.monotonic()
        next_wait_log = time.monotonic() + 5.0
        next_missing_domain_log = time.monotonic()
        draining_reservation: Optional[DecodeReservation] = None
        p2d_snapshot = self._p2d_snapshot_for_rooms(rooms)
        p2d_offer_published = bool(
            p2d_snapshot is not None
            and p2d_snapshot
            in getattr(self, "_p2d_host_offered_snapshots", set())
        )
        host_freshness_attempted = False
        force_load_refresh = False

        try:
            while True:
                # Both network sampling and the flock/JSON Host ledger live
                # outside the cluster-wide D capacity lock.  A provisional D
                # credit protects capacity while the Host CAS is finalized.
                loads_snapshot = await self._all_decode_loads(
                    session, force=force_load_refresh
                )
                force_load_refresh = False
                p2d_entry = (
                    await asyncio.to_thread(
                        self.p2d_host_ledger.get, p2d_snapshot
                    )
                    if p2d_snapshot is not None
                    else None
                )
                if p2d_entry is not None and p2d_snapshot is not None:
                    p2d_offer_published = True
                    if not hasattr(self, "_p2d_host_offered_snapshots"):
                        self._p2d_host_offered_snapshots = set()
                    self._p2d_host_offered_snapshots.add(p2d_snapshot)
                p2d_state = (
                    None if p2d_entry is None else p2d_entry.get("state")
                )
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
                candidate_stats: Optional[
                    tuple[DecodeLoad, float, float, int, float]
                ] = None
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
                        and p2d_snapshot
                        in getattr(self, "_p2d_host_offered_snapshots", set())
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
                        loads = [
                            load
                            for load in loads_snapshot
                            if load.url in allowed_urls
                        ]
                        missing_loads = not loads

                    if not stale_ownership and loads:
                        if draining_reservation is not None:
                            if not any(
                                load.url == draining_reservation.url
                                for load in loads
                            ):
                                self._reservations.pop(
                                    draining_reservation.reservation_id, None
                                )
                                draining_reservation = None

                        average_context = self._average_running_context(loads)
                        transfer_weight = max(
                            1.0,
                            getattr(self, "transfer_request_weight", 2.0),
                        )
                        scored: list[
                            tuple[bool, float, float, int, DecodeLoad]
                        ] = []
                        for load in loads:
                            # A draining reservation is only a future-capacity
                            # hint.  Re-score every D without counting that
                            # hint so an observation made while all D workers
                            # were full cannot pin this request to a now-heavier
                            # destination.
                            draining_id = (
                                draining_reservation.reservation_id
                                if draining_reservation is not None
                                else None
                            )
                            (
                                reserved_prompt,
                                reserved_admission,
                                reserved_reqs,
                            ) = self._reserved_for(
                                load.url, exclude_id=draining_id
                            )
                            free_after_pending = (
                                load.capacity_tokens
                                - load.used_tokens
                                - reserved_admission
                            )
                            projected_kv = (
                                load.used_tokens
                                + reserved_admission
                                + admission_tokens
                            ) / load.capacity_tokens
                            transfer_feasible = (
                                self.max_direct_transfer_inflight == 0
                                or load.prealloc + load.transfer + reserved_reqs
                                < self.max_direct_transfer_inflight
                            )
                            feasible = (
                                free_after_pending >= admission_tokens
                                and projected_kv
                                <= getattr(
                                    self, "target_decode_kv_fraction", 1.0
                                )
                                and transfer_feasible
                            )
                            queued_or_handoff_reqs = max(
                                load.waiting, load.prealloc + load.transfer
                            )
                            projected_decode_reqs = (
                                load.running
                                + queued_or_handoff_reqs
                                + reserved_reqs
                                + len(rooms)
                            )
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
                            _, work_score, projected_kv, projected_reqs, selected = (
                                choice
                            )
                            if (
                                draining_reservation is not None
                                and selected.url == draining_reservation.url
                            ):
                                candidate = draining_reservation
                            else:
                                if draining_reservation is not None:
                                    old_url = draining_reservation.url
                                    self._reservations.pop(
                                        draining_reservation.reservation_id, None
                                    )
                                    draining_reservation = None
                                    logger.info(
                                        "PD_LATE_BIND_DRAIN_RESELECT rooms=%s "
                                        "old_D=%s new_D=%s reason=least_work",
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
                                self._reservations[
                                    candidate.reservation_id
                                ] = candidate
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
                                draining_urls = {
                                    item.url
                                    for item in self._reservations.values()
                                    if item.draining
                                }
                                drain_choices = [
                                    item
                                    for item in scored
                                    if item[4].url not in draining_urls
                                ]
                                if drain_choices:
                                    choice = self._choose_decode_score(
                                        drain_choices, drain=True
                                    )
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
                                    self._reservations[
                                        draining_reservation.reservation_id
                                    ] = draining_reservation
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
                                    and not self._decode_loads_are_fresh_since(
                                        refresh_urls, wait_started
                                    )
                                )
                                publish_host_offer = not refresh_before_host_offer

                if stale_ownership:
                    continue
                if missing_loads:
                    now = time.monotonic()
                    if now >= next_missing_domain_log:
                        logger.warning(
                            "No current D load sample in domain %d; waiting "
                            "for the local endpoint",
                            domain,
                        )
                        next_missing_domain_log = now + 5.0
                    if now >= deadline:
                        raise TimeoutError(
                            f"No usable D worker in domain {domain} before "
                            "P-ready admission deadline"
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
                                    self._reservations.pop(
                                        candidate.reservation_id, None
                                    )
                            else:
                                self._reservations[
                                    finalized.reservation_id
                                ] = finalized
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

                    selected, work_score, projected_kv, projected_reqs, average = (
                        candidate_stats
                    )
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
                        "PD_LATE_BIND_WAIT rooms=%s admission_tokens=%d: "
                        "all D workers full",
                        rooms,
                        admission_tokens,
                    )
                    next_wait_log = now + 5.0
                await asyncio.sleep(self.no_capacity_poll_interval)
        except BaseException:
            if draining_reservation is not None:
                async with self._selection_lock:
                    self._reservations.pop(
                        draining_reservation.reservation_id, None
                    )
                    self._load_cache_at = 0.0
            if p2d_snapshot is not None:
                self._abort_unsubmitted_p2d(
                    p2d_snapshot, "decode_selection_aborted"
                )
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
                    self._admitted_reservation_at[
                        reservation.reservation_id
                    ] = time.monotonic()
                else:
                    self._reservations.pop(reservation.reservation_id, None)
