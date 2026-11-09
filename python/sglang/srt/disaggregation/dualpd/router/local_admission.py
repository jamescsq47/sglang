"""P-ready and ledger admission coordination; no independent ownership authority."""
from __future__ import annotations

import asyncio
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
import orjson
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    RequestGeneration,
)
from .local_dependencies import HostStageState
from .local_dependencies import (
    P2D_CUSTOM_PREFILL_DOMAIN,
    P2D_CUSTOM_SNAPSHOT_ID,
    p2d_snapshot_id,
)

from .local_types import (
    DecodeReservation,
    _HOST_STAGE_EVICTING,
    _HOST_STAGE_RECOMPUTE_REQUIRED,
    _PReadyAdmission,
    _PrefillAdmissionWaiter,
    _PrefillWorkReservation,
    logger,
)

class _PrefillAdmissionGate:
    """Bound hidden HTTP/bootstrap work before requests reach P.

    This gate is FIFO across parent and initial turns. KV source affects only
    the independent Direct/Slow I/O credits inside P, never request admission.
    """

    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("P admission limit must be positive")
        self.limit = limit
        self.active = 0
        self._sequence = 0
        self._waiters: list[_PrefillAdmissionWaiter] = []
        self._condition = asyncio.Condition()

    def _next_waiter(self, now: float) -> _PrefillAdmissionWaiter:
        del now
        return min(self._waiters, key=lambda waiter: waiter.sequence)

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
                    if (
                        self.active < self.limit
                        and self._next_waiter(time.monotonic()) is waiter
                    ):
                        self._waiters.remove(waiter)
                        self.active += 1
                        self._condition.notify_all()
                        return time.monotonic() - waiter.enqueued_at
                    await self._condition.wait()
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


class LocalAdmissionMixin:
    """Method-only component; constructor, locks and queues belong to the Router."""

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

    def _d_reserved_path(self, room: int) -> Path:
        return self.p_ready_dir / f"{room}.d_reserved"

    def _bind_prefill_domain(
        self, request: dict[str, Any], domain: int
    ) -> str:
        prefill_server = self.prefill_urls[domain]
        parsed = urllib.parse.urlparse(prefill_server)
        hostname = parsed.hostname
        if hostname is None:
            raise ValueError(f"Invalid P URL {prefill_server!r}")
        port = self.prefill_bootstrap_ports[domain]
        batch = isinstance(request.get("bootstrap_room"), list)
        request["bootstrap_host"] = (
            [hostname] * len(request["bootstrap_room"]) if batch else hostname
        )
        request["bootstrap_port"] = (
            [port] * len(request["bootstrap_room"]) if batch else port
        )
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
        if (
            not getattr(self, "numa_domains", False)
            or getattr(self, "global_decode", False)
        ):
            return set(self.decode_urls)
        width = len(self.decode_urls) // len(self.prefill_urls)
        return set(self.decode_urls[domain * width : (domain + 1) * width])

    @staticmethod
    def _set_p2d_host_metadata(
        request: dict[str, Any], snapshot_id: str, prefill_domain: int
    ) -> None:
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
            "PD_P2D_HOST_OFFER snapshot=%s rooms=%s P=%d "
            "prompt_tokens=%d state=%s source=%s",
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
                    f"P->D Host staging terminated for {snapshot_id}: "
                    f"state={state} reason={reason}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for durable P->D Host snapshot "
                    f"{snapshot_id}"
                )
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
        if not await self._observe_decode_load_after(
            self._load_http_session(), urls=urls, not_before=not_before
        ):
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
        state = self.p2d_host_ledger.abort_unsubmitted_p2d(
            snapshot_id, reason=reason
        )
        logger.info(
            "PD_P2D_HOST_ABORT snapshot=%s state=%s reason=%s",
            snapshot_id,
            state,
            reason,
        )

    async def _commit_slow_recovery_domain(
        self,
        parent: RequestGeneration,
        route: dict[str, Any],
        domain: int,
    ) -> Optional[int]:
        """Commit one Host-independent H2D/Prefill destination.

        ``None`` is the explicit outcome when pressure eviction wins the
        HOST_READY assignment CAS.  The caller then resizes/releases its
        provisional incremental-work reservation and performs full Prefill.
        """

        ledger = getattr(self, "_d2p_host_ledger", None)
        if ledger is None:
            raise RuntimeError("dynamic Slow recovery requires a Host ledger")
        deadline = time.monotonic() + self.ready_timeout
        while True:
            assigned = await asyncio.to_thread(
                ledger.assign_d2p_recovery_domain, parent.snapshot_id, domain
            )
            entry = await asyncio.to_thread(ledger.get, parent.snapshot_id)
            if assigned:
                break
            existing = None if entry is None else entry.get("recovery_domain")
            if existing is not None:
                domain = int(existing)
                break
            state = None if entry is None else entry.get("state")
            if (
                _HOST_STAGE_EVICTING is not None
                and state == _HOST_STAGE_EVICTING.value
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for Host eviction fence of "
                        f"{parent.snapshot_id}"
                    )
                await asyncio.sleep(self.ready_poll_interval)
                continue
            if (
                _HOST_STAGE_RECOMPUTE_REQUIRED is not None
                and state == _HOST_STAGE_RECOMPUTE_REQUIRED.value
            ):
                self.early_claim_store.publish_route(
                    parent,
                    route="recompute",
                    prefill_domain=int(route.get("prefill_domain", domain)),
                )
                self.early_claim_store.remove_arrival(parent)
                logger.info(
                    "PD_SLOW_RECOVERY_ROUTE snapshot=%s state=%s "
                    "action=full_recompute",
                    parent.snapshot_id,
                    state,
                )
                return None
            raise RuntimeError(
                f"cannot assign Slow recovery P for {parent.snapshot_id}: "
                f"state={state}"
            )
        host_domain = route.get("arena_domain")
        if host_domain is None and entry is not None:
            host_domain = entry.get("arena_domain")
        self.early_claim_store.publish_route(
            parent,
            route="host_ready",
            prefill_domain=domain,
            arena_domain=(None if host_domain is None else int(host_domain)),
            arena_numa_node=route.get("arena_numa_node"),
            snapshot_tokens=route.get("snapshot_tokens"),
        )
        logger.info(
            "PD_SLOW_RECOVERY_ROUTE snapshot=%s host=%s P=%d "
            "policy=max_remaining_kv",
            parent.snapshot_id,
            host_domain,
            domain,
        )
        return domain

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
            return await self._reserve_prefill_work(
                self._estimated_prefill_tokens(request)
            )

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
                if route is not None:
                    mode = route.get("route")
                    snapshot_tokens = route.get("snapshot_tokens")
                    if mode == "direct_ready":
                        if reservation is None:
                            reservation = await self._reserve_prefill_work(
                                self._estimated_prefill_tokens(
                                    request, snapshot_tokens
                                ),
                                direct_workset_tokens=self._request_input_tokens(
                                    request
                                ),
                            )
                            logger.info(
                                "PD_PREFILL_ROUTE snapshot=%s route=%s "
                                "selected_P=%d estimated_tokens=%d",
                                parent.snapshot_id,
                                mode,
                                reservation.domain,
                                reservation.tokens,
                            )
                        reservation.route_pending = True
                        return reservation
                    elif mode in {"direct_complete", "host_writing", "host_ready"}:
                        if mode in {"host_writing", "host_ready"}:
                            host_ledger = getattr(self, "_d2p_host_ledger", None)
                            host_entry = (
                                None
                                if host_ledger is None
                                else await asyncio.to_thread(
                                    host_ledger.get, parent.snapshot_id
                                )
                            )
                            host_state = (
                                None if host_entry is None else host_entry.get("state")
                            )
                            if (
                                _HOST_STAGE_EVICTING is not None
                                and host_state == _HOST_STAGE_EVICTING.value
                            ):
                                await asyncio.sleep(self.ready_poll_interval)
                                continue
                            if (
                                _HOST_STAGE_RECOMPUTE_REQUIRED is not None
                                and host_state
                                == _HOST_STAGE_RECOMPUTE_REQUIRED.value
                            ):
                                store.publish_route(
                                    parent,
                                    route="recompute",
                                    prefill_domain=int(route["prefill_domain"]),
                                )
                                store.remove_arrival(parent)
                                await self._release_prefill_work(reservation)
                                return await self._reserve_prefill_work(
                                    self._request_input_tokens(request)
                                )
                            # Host placement is complete before selecting the
                            # independent H2D/Prefill P. Do not bind recovery
                            # to the arena owner while D2H is still running.
                            if (
                                mode == "host_writing"
                                or host_state != HostStageState.HOST_READY.value
                            ):
                                await asyncio.sleep(self.ready_poll_interval)
                                continue
                            if reservation is None:
                                reservation = await self._reserve_prefill_work(
                                    self._estimated_prefill_tokens(
                                        request, snapshot_tokens
                                    ),
                                    direct_workset_tokens=self._request_input_tokens(
                                        request
                                    ),
                                )
                            domain = await self._commit_slow_recovery_domain(
                                parent, route, reservation.domain
                            )
                            if domain is None:
                                await self._release_prefill_work(reservation)
                                reservation = None
                                return await self._reserve_prefill_work(
                                    self._request_input_tokens(request)
                                )
                            if domain != reservation.domain:
                                await self._move_prefill_work(reservation, domain)
                            logger.info(
                                "PD_PREFILL_ROUTE snapshot=%s route=host_ready "
                                "P=%d estimated_tokens=%d",
                                parent.snapshot_id,
                                domain,
                                reservation.tokens,
                            )
                            return reservation
                        domain = int(route["prefill_domain"])
                        if not 0 <= domain < len(self.prefill_urls):
                            raise RuntimeError(f"invalid Prefill domain {domain}")
                        if reservation is None:
                            reservation = await self._reserve_prefill_work(
                                self._estimated_prefill_tokens(
                                    request, snapshot_tokens
                                ),
                                domain=domain,
                                direct_workset_tokens=self._request_input_tokens(
                                    request
                                ),
                            )
                        else:
                            await self._move_prefill_work(reservation, domain)
                        logger.info(
                            "PD_PREFILL_ROUTE snapshot=%s route=%s P=%d "
                            "estimated_tokens=%d",
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
                        return await self._reserve_prefill_work(
                            self._request_input_tokens(request)
                        )
                if store.read_final(
                    parent, not_before=0.0, max_age_seconds=self.ready_timeout
                ) is not None:
                    store.remove_arrival(parent)
                    await self._release_prefill_work(reservation)
                    return await self._reserve_prefill_work(
                        self._request_input_tokens(request)
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for P route of {parent.snapshot_id}"
                    )
                await asyncio.sleep(self.ready_poll_interval)
        except BaseException:
            await self._release_prefill_work(reservation)
            raise

    async def _watch_dynamic_prefill_route(
        self,
        request: dict[str, Any],
        metadata: AgenticRequestMetadata,
        reservation: _PrefillWorkReservation,
        *,
        fixed_domain: bool = False,
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
            if route is not None:
                mode = route.get("route")
                if mode in {"direct_complete", "host_writing", "host_ready"}:
                    if mode in {"host_writing", "host_ready"}:
                        host_ledger = getattr(self, "_d2p_host_ledger", None)
                        host_entry = (
                            None
                            if host_ledger is None
                            else await asyncio.to_thread(
                                host_ledger.get, parent.snapshot_id
                            )
                        )
                        host_state = (
                            None if host_entry is None else host_entry.get("state")
                        )
                        if (
                            _HOST_STAGE_EVICTING is not None
                            and host_state == _HOST_STAGE_EVICTING.value
                        ):
                            await asyncio.sleep(self.ready_poll_interval)
                            continue
                        if (
                            _HOST_STAGE_RECOMPUTE_REQUIRED is not None
                            and host_state
                            == _HOST_STAGE_RECOMPUTE_REQUIRED.value
                        ):
                            store.publish_route(
                                parent,
                                route="recompute",
                                prefill_domain=int(route["prefill_domain"]),
                            )
                            store.remove_arrival(parent)
                            await self._settle_direct_workset(reservation)
                            await self._resize_prefill_work(
                                reservation, self._request_input_tokens(request)
                            )
                            return {"action": "recompute", "route": "host_evicted"}
                        if (
                            mode == "host_writing"
                            or host_state != HostStageState.HOST_READY.value
                        ):
                            await asyncio.sleep(self.ready_poll_interval)
                            continue
                        original_domain = reservation.domain
                        domain = reservation.domain
                        if not fixed_domain:
                            domain = await self._move_prefill_work_to_max_remaining_kv(
                                reservation, self._request_input_tokens(request),
                            )
                        domain = await self._commit_slow_recovery_domain(
                            parent, route, domain
                        )
                        if domain is None:
                            # Max-KV selection moved only Router accounting;
                            # eviction won before any redirect was committed.
                            # The existing HTTP request still targets the
                            # original P, so move its full-recompute charge
                            # back there before resizing it.
                            if reservation.domain != original_domain:
                                await self._move_prefill_work(
                                    reservation, original_domain
                                )
                            await self._settle_direct_workset(reservation)
                            await self._resize_prefill_work(
                                reservation, self._request_input_tokens(request)
                            )
                            return {
                                "action": "recompute",
                                "route": "host_evicted",
                            }
                        if domain != reservation.domain:
                            await self._move_prefill_work(reservation, domain)
                        if domain != original_domain:
                            logger.info(
                                "PD_PREFILL_REDIRECT snapshot=%s route=host_ready "
                                "from_P=%d to_P=%d",
                                parent.snapshot_id,
                                original_domain,
                                domain,
                            )
                        if domain != reservation.domain:
                            raise RuntimeError("Slow recovery reservation move lost")
                        # The request was already submitted for Direct. Make
                        # the HTTP attempt follow the newly selected recovery
                        # P whenever it differs from that original domain.
                        return (
                            {
                                "action": "redirect",
                                "route": "host_ready",
                                "domain": domain,
                            }
                            if domain != original_domain
                            else {
                                "action": "settled",
                                "route": "host_ready",
                                "domain": domain,
                            }
                        )
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
                            "PD_PREFILL_REDIRECT snapshot=%s route=%s "
                            "from_P=%d to_P=%d",
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
                    await self._resize_prefill_work(
                        reservation, self._request_input_tokens(request)
                    )
                    return {"action": "recompute", "route": mode}
            if store.read_final(
                parent, not_before=0.0, max_age_seconds=self.ready_timeout
            ) is not None:
                store.remove_arrival(parent)
                await self._settle_direct_workset(reservation)
                await self._resize_prefill_work(
                    reservation, self._request_input_tokens(request)
                )
                return {"action": "recompute", "route": "final"}
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for P route of {parent.snapshot_id}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    def _p_ready_marker_is_owned(self, room: int, payload: dict[str, Any]) -> bool:
        owners = getattr(self, "_active_prefill_attempts", None)
        # Compatibility for focused tests/embedders constructed via __new__.
        if owners is None:
            return True
        expected_rid = owners.get(int(room))
        return expected_rid is not None and str(payload.get("rid", "")) == expected_rid

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
                        "Prefill failed before scheduler acceptance: "
                        f"status={response.status} body={body}"
                    )
            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(
                    "Timed out waiting "
                    f"{self.prefill_accept_timeout}s for P-accepted rooms {missing}"
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
                    raise RuntimeError(
                        "Prefill failed while queued: "
                        f"status={response.status} body={body}"
                    )
            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(
                    "Timed out waiting "
                    f"{self.prefill_queue_timeout}s in P queue for rooms {missing}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    async def _wait_until_decode_reserved(
        self,
        rooms: tuple[int, ...],
        decode_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> None:
        """Fence P submission on physical destination-KV ownership at D."""

        deadline = time.monotonic() + self.prefill_queue_timeout
        paths = [self._d_reserved_path(room) for room in rooms]
        while True:
            self._raise_prefill_redirect(route_task)
            if all(path.exists() for path in paths):
                for path in paths:
                    path.unlink(missing_ok=True)
                return
            if decode_task.done():
                response = decode_task.result()
                body = await response.text()
                response.release()
                raise RuntimeError(
                    "Decode returned before destination KV reservation: "
                    f"status={response.status} body={body}"
                )
            if time.monotonic() >= deadline:
                missing = [
                    room for room, path in zip(rooms, paths) if not path.exists()
                ]
                raise TimeoutError(
                    "Timed out waiting "
                    f"{self.prefill_queue_timeout}s for D-reserved rooms {missing}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    async def _wait_until_prefill_ready(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> int:
        if getattr(self, "dynamic_prefill_domains", False):
            return await self._wait_until_prefill_ready_shared(
                rooms, prefill_task, route_task
            )
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
                    raise RuntimeError(
                        f"Prefill failed before P-ready: status={response.status} body={body}"
                    )

            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(
                    f"Timed out waiting {self.ready_timeout}s for P-ready rooms {missing}"
                )
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
            self._p_ready_monitor_task = asyncio.create_task(
                self._p_ready_monitor_loop(), name="pd-p-ready-monitor"
            )

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
                    return sum(
                        int(future.result().get("num_kv_tokens", 0))
                        for future in futures.values()
                    )
                self._raise_prefill_redirect(route_task)
                if prefill_task.done():
                    response = prefill_task.result()
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            "Prefill failed before P-ready: "
                            f"status={response.status} body={body}"
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [
                        room for room, future in futures.items() if not future.done()
                    ]
                    raise TimeoutError(
                        f"Timed out waiting {self.ready_timeout}s for P-ready rooms {missing}"
                    )
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
                if getattr(self, "dynamic_prefill_domains", False) and int(
                    payload.get("prefill_domain", -1)
                ) != domain:
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
                        if latest is not None and self._p_ready_marker_is_owned(
                            room, latest
                        ):
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
                sequence = (
                    int(sequence)
                    if sequence is not None
                    else path.stat().st_mtime_ns
                )
                key = (
                    (domain, sequence)
                    if getattr(self, "dynamic_prefill_domains", False)
                    else sequence
                )
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

    async def _reserve_p_ready_direct_batch(
        self, admissions: list[_PReadyAdmission]
    ) -> dict[_PReadyAdmission, Optional[DecodeReservation]]:
        """Reserve the next fair-scan candidate per P from one D snapshot."""

        eligible = [
            admission
            for admission in admissions
            if admission.request is not None
            and admission.commit is not None
        ]
        reservations = {admission: None for admission in admissions}
        if not eligible:
            return reservations
        try:
            loads = await self._all_decode_loads(self._load_http_session())
        except Exception:
            logger.warning(
                "P-ready batch load snapshot failed; falling back to the "
                "per-request capacity state machine",
                exc_info=True,
            )
            return reservations
        async with self._selection_lock:
            self._prune_accounted_reservations()
            reservations.update(
                {
                    admission: self._try_reserve_direct_ready_locked(
                        admission, loads
                    )
                    for admission in eligible
                }
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
                "PD_P_READY_FIFO_DISPATCH P=%d sequence=%d wait_s=%.3f "
                "submitted=%d queued=%d broker_batch=true",
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
            admission.ownership_started = bool(
                reservation is not None or admission.prepare is not None
            )
            if prepared_reservation is None and admission.prepare is not None:
                spill_delay = max(
                    0.0, float(getattr(self, "p2d_host_spill_delay", 0.0))
                )
                if spill_delay > 0:
                    grace_started_at = time.monotonic()
                    await asyncio.sleep(spill_delay)
                    if admission.cancel_requested:
                        raise asyncio.CancelledError
                    prepared_reservation = (
                        await self._retry_p_ready_direct_after_grace(
                            admission, not_before=grace_started_at
                        )
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
            admission.host_staged = (
                admission.prepare is not None and prepared_reservation is None
            )
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
                        self._reservations.pop(
                            prepared_reservation.reservation_id, None
                        )
                        self._admitted_reservation_at.pop(
                            prepared_reservation.reservation_id, None
                        )
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

    def _cancel_unowned_p_ready_admission(
        self, admission: _PReadyAdmission
    ) -> bool:
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
            if len(self._p_ready_fifo_active.get(domain, {})) >= max(
                1, self._p_ready_admission_window_per_p
            ):
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
                        name=(
                            f"pd-p-ready-admission-{admission.domain}-"
                            f"{admission.sequence}"
                        ),
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
        prepare: Optional[
            Callable[[], Awaitable[Optional[DecodeReservation]]]
        ] = None,
    ) -> Any:
        """Publish an immutable admission and await the shared broker."""

        self._ensure_p_ready_admission_state()
        waiters = self._p_ready_fifo_waiters.setdefault(domain, {})
        if sequence in waiters:
            raise RuntimeError(
                f"duplicate P-ready sequence P={domain} sequence={sequence}"
            )
        key: Any = (
            (domain, sequence)
            if getattr(self, "dynamic_prefill_domains", False)
            else sequence
        )
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
                cancelled_unowned = self._cancel_unowned_p_ready_admission(
                    admission
                )
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
