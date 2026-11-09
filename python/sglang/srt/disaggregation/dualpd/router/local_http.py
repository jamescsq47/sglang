"""Backend HTTP attempts, generation deduplication and exact existing dispatch."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import replace
from typing import Any, Awaitable, Optional
import aiohttp
import orjson
from fastapi.responses import ORJSONResponse, StreamingResponse
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    unpack_agentic_extra_key,
)
from sglang_router.mini_lb import AIOHTTP_STREAM_READ_CHUNK_SIZE

from .local_types import (
    DecodeReservation,
    _GenerationResponse,
    _PrefillRedirect,
    _PrefillWorkReservation,
    logger,
)

from .local_config import _env_float, _env_int

class LocalHTTPMixin:
    """Method-only component; constructor, locks and queues belong to the Router."""

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
            request["bootstrap_room"] = [
                uuid.uuid4().int & ((1 << 63) - 1) for _ in room
            ]
        else:
            request["bootstrap_room"] = uuid.uuid4().int & ((1 << 63) - 1)

    @staticmethod
    def _attempt_rids(
        request: dict[str, Any], rooms: tuple[int, ...]
    ) -> tuple[str, ...]:
        value = request.get("rid")
        if isinstance(value, list):
            if len(value) != len(rooms):
                raise ValueError("rid/bootstrap_room batch size mismatch")
            return tuple(str(item) for item in value)
        if len(rooms) != 1 or value is None:
            raise ValueError("missing rid for Prefill attempt")
        return (str(value),)

    def _activate_prefill_attempt(
        self, request: dict[str, Any], rooms: tuple[int, ...]
    ) -> None:
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
                raise RuntimeError(
                    f"bootstrap_room {room} already owned by rid={existing}"
                )
            pending[room] = rid
        owners.update(pending)

    def _deactivate_prefill_attempt(
        self, request: dict[str, Any], rooms: tuple[int, ...]
    ) -> None:
        owners = getattr(self, "_active_prefill_attempts", None)
        if owners is None:
            return
        for room, rid in zip(rooms, self._attempt_rids(request, rooms)):
            if owners.get(int(room)) == rid:
                owners.pop(int(room), None)

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
                response = await session.post(
                    f"{prefill_server}/abort_request", json={"rid": rid}
                )
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
                response = await session.post(
                    f"{decode_server}/abort_request", json={"rid": rid}
                )
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

        task = asyncio.create_task(awaitable)
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
        custom_params = (
            sampling.get("custom_params") if isinstance(sampling, dict) else None
        )
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
            "PD_EARLY_CLAIM_ARRIVAL snapshot=%s generation=%d arrived_at=%.6f "
            "prompt_tokens=%d P=%s",
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
        custom_params = (
            sampling.get("custom_params") if isinstance(sampling, dict) else None
        )
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
                if (
                    "meta_info" in ret_json
                    and "input_token_logprobs" in ret_json["meta_info"]
                ):
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
                limit_per_host=_env_int(
                    "SGLANG_PD_ROUTER_HTTP_CONNECTION_LIMIT_PER_HOST", 512
                ),
                # The generation path can have hundreds of concurrent P/D
                # requests.  Reusing a bounded pool is essential: forcing one
                # new socket per turn creates an FD-close/reuse storm in
                # uvloop at c512.  The launch script keeps the SGLang server
                # timeout longer than this client-side idle lifetime.
                keepalive_timeout=_env_float(
                    "SGLANG_PD_ROUTER_HTTP_KEEPALIVE_S", 30.0
                ),
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
                limit_per_host=_env_int(
                    "SGLANG_PD_ROUTER_LOAD_CONNECTION_LIMIT_PER_HOST", 4
                ),
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
                task = asyncio.create_task(
                    self._generate_once(request_copy, prefill_server, endpoint)
                )
                self._generation_tasks[key] = task
                task.add_done_callback(
                    lambda done, generation=key: asyncio.create_task(
                        self._record_generation_task(generation, done)
                    )
                )
                logger.info("PD_GENERATION_DEDUP producer generation=%s", key)
            else:
                logger.warning("PD_GENERATION_DEDUP joined_retry generation=%s", key)
        # Client cancellation must not cancel the elected serving operation.
        return await asyncio.shield(task)

    async def _wait_prebound_prefill_or_redirect(
        self,
        prefill_task: asyncio.Task,
        decode_task: asyncio.Task,
        route_task: Optional[asyncio.Task],
    ) -> Any:
        """Wait for native Prefill while keeping D->P rerouting responsive."""

        deadline = time.monotonic() + self.ready_timeout
        decode_checked = False
        while True:
            self._raise_prefill_redirect(route_task)
            if prefill_task.done():
                response = prefill_task.result()
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        "Pre-bound Prefill failed: "
                        f"status={response.status} body={body}"
                    )
                self._raise_prefill_redirect(route_task)
                return response
            if decode_task.done() and not decode_checked:
                response = decode_task.result()
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        "Pre-bound Decode failed before Prefill: "
                        f"status={response.status} body={body}"
                    )
                decode_checked = True
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting "
                    f"{self.ready_timeout}s for native pre-bound Prefill"
                )
            waits = {prefill_task}
            if not decode_task.done():
                waits.add(decode_task)
            if route_task is not None and not route_task.done():
                waits.add(route_task)
            await asyncio.wait(
                waits,
                timeout=min(0.25, self.ready_poll_interval),
                return_when=asyncio.FIRST_COMPLETED,
            )

    async def _prebound_dispatch_with_metadata(
        self,
        session: aiohttp.ClientSession,
        modified_request: dict[str, Any],
        prefill_server: str,
        endpoint: str,
        headers: dict[str, str],
        metadata: Optional[AgenticRequestMetadata],
    ):
        """Run classic pre-bound P->D while preserving the D->P method.

        The Router reserves one D before submitting either backend request.
        Workers disable compute-ahead P-ready gating under the same global
        ablation flag, so D performs native preallocation and P waits on its
        normal bootstrap metadata.  No P->D Host owner can exist in this mode.
        """

        if getattr(self, "p2d_host_staging", False):
            raise RuntimeError("pre-bound P->D cannot use P->D Host staging")
        if modified_request.get("return_logprob", False):
            raise RuntimeError(
                "pre-bound P->D ablation does not support return_logprob"
            )

        rooms = self._rooms(modified_request)
        parent_turn = metadata is not None and metadata.parent is not None
        arrival_at: Optional[float] = time.time() if parent_turn else None
        prefill_work: Optional[_PrefillWorkReservation] = None
        if getattr(self, "dynamic_prefill_domains", False):
            self._ensure_prefill_pressure_monitor()
            arrival = self._publish_parent_arrival(modified_request)
            if arrival is not None:
                arrival_at = float(arrival["arrived_at"])
            prefill_work = await self._resolve_dynamic_prefill_work(
                modified_request, metadata, arrival_at
            )
            domain = prefill_work.domain
        else:
            domain = self._request_domain(metadata, rooms)

        self._set_prefill_attempt_rid(modified_request, replace=False)
        while True:
            if getattr(self, "numa_domains", False):
                prefill_server = self._bind_prefill_domain(
                    modified_request, domain
                )
            rooms = self._rooms(modified_request)
            self._activate_prefill_attempt(modified_request, rooms)
            attempt_active = True
            route_task: Optional[asyncio.Task] = None
            reservation: Optional[DecodeReservation] = None
            prefill_task: Optional[asyncio.Task] = None
            decode_task: Optional[asyncio.Task] = None
            p_gate: Optional[_PrefillAdmissionGate] = None
            p_gate_held = False
            keep_d_credit = False
            keep_p_credit = False
            try:
                if parent_turn:
                    self._publish_parent_arrival(
                        modified_request,
                        target_prefill_domain=(
                            domain
                            if getattr(self, "dynamic_prefill_domains", False)
                            else None
                        ),
                        arrived_at=arrival_at,
                    )
                    if (
                        prefill_work is not None
                        and prefill_work.route_pending
                        and metadata is not None
                    ):
                        route_task = asyncio.create_task(
                            self._watch_dynamic_prefill_route(
                                modified_request, metadata, prefill_work
                            )
                        )

                reservation = await self._select_and_reserve_decode(
                    session,
                    modified_request,
                    rooms,
                    self._request_input_tokens(modified_request),
                    domain,
                )
                if reservation is None:
                    raise RuntimeError(
                        "P->D pre-binding returned without a Decode reservation"
                    )
                logger.info(
                    "PD_P2D_PREBIND rooms=%s P=%d D=%s admission_tokens=%d",
                    rooms,
                    domain,
                    reservation.url,
                    reservation.admission_tokens,
                )
                # D is posted first. Its native preallocation publishes the
                # bootstrap metadata consumed by P. P is forced non-streaming
                # so its HTTP completion is also the native P->D completion
                # fence; the client-facing D request keeps its original mode.
                decode_task = asyncio.create_task(
                    session.post(
                        f"{reservation.url}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    )
                )
                p_gate = self._prefill_admission_for_domain(domain)
                await p_gate.acquire(parent_turn=parent_turn)
                p_gate_held = True
                prefill_request = dict(modified_request)
                prefill_request["stream"] = False
                prefill_task = asyncio.create_task(
                    session.post(
                        f"{prefill_server}/{endpoint}",
                        json=prefill_request,
                        headers=headers,
                    )
                )
                try:
                    await self._wait_until_prefill_accepted(
                        rooms, prefill_task, route_task
                    )
                finally:
                    await p_gate.release()
                    p_gate_held = False

                prefill_response = await self._wait_prebound_prefill_or_redirect(
                    prefill_task, decode_task, route_task
                )
                try:
                    await prefill_response.read()
                finally:
                    prefill_response.release()
                prefill_response = None
                # The non-streaming native P response is the P->D transfer
                # fence. Hand shadow credit to a causally newer D load sample.
                async with self._selection_lock:
                    self._admitted_reservation_at[
                        reservation.reservation_id
                    ] = time.monotonic()
                keep_d_credit = True
                await self._release_prefill_work(prefill_work)
                if route_task is not None and not route_task.done():
                    route_task.cancel()
                    await asyncio.gather(route_task, return_exceptions=True)
                decode_response = await decode_task
                return prefill_response, decode_response
            except _PrefillRedirect as redirect:
                p_aborted = True
                d_aborted = True
                if prefill_task is not None:
                    p_aborted = await self._abort_prefill_attempt(
                        session,
                        prefill_server,
                        modified_request,
                        prefill_task,
                    )
                    prefill_task = None
                if decode_task is not None and reservation is not None:
                    d_aborted = await self._abort_decode_attempt(
                        session, reservation.url, modified_request
                    )
                    await self._dispose_response_task(decode_task)
                    decode_task = None
                if not p_aborted or not d_aborted:
                    keep_p_credit = not p_aborted
                    keep_d_credit = True
                    raise RuntimeError(
                        "Cannot safely redirect pre-bound P->D attempt: "
                        f"P_abort={p_aborted} D_abort={d_aborted}"
                    )
                if reservation is not None:
                    async with self._selection_lock:
                        self._admitted_reservation_at[
                            reservation.reservation_id
                        ] = time.monotonic()
                    reservation = None
                for room in rooms:
                    self._accepted_path(room).unlink(missing_ok=True)
                    self._scheduled_path(room).unlink(missing_ok=True)
                    self._ready_path(room).unlink(missing_ok=True)
                self._deactivate_prefill_attempt(modified_request, rooms)
                attempt_active = False
                await self._move_prefill_work(prefill_work, redirect.domain)
                domain = redirect.domain
                self._replace_prefill_attempt_rooms(modified_request)
                self._set_prefill_attempt_rid(modified_request, replace=True)
                keep_p_credit = True
                continue
            except BaseException:
                p_aborted = prefill_task is None
                d_aborted = decode_task is None
                if prefill_task is not None:
                    p_aborted = await self._abort_prefill_attempt(
                        session,
                        prefill_server,
                        modified_request,
                        prefill_task,
                    )
                    prefill_task = None
                if decode_task is not None and reservation is not None:
                    d_aborted = await self._abort_decode_attempt(
                        session, reservation.url, modified_request
                    )
                    await self._dispose_response_task(decode_task)
                    decode_task = None
                keep_p_credit = not p_aborted
                keep_d_credit = reservation is not None
                if d_aborted and reservation is not None:
                    async with self._selection_lock:
                        self._admitted_reservation_at[
                            reservation.reservation_id
                        ] = time.monotonic()
                raise
            finally:
                if p_gate_held and p_gate is not None:
                    await p_gate.release()
                if route_task is not None and not route_task.done():
                    route_task.cancel()
                    await asyncio.gather(route_task, return_exceptions=True)
                if reservation is not None and not keep_d_credit:
                    async with self._selection_lock:
                        self._reservations.pop(
                            reservation.reservation_id, None
                        )
                if not keep_p_credit:
                    await self._release_prefill_work(prefill_work)
                if attempt_active:
                    self._deactivate_prefill_attempt(modified_request, rooms)

    async def _native_prealloc_dispatch_with_metadata(
        self,
        session: aiohttp.ClientSession,
        modified_request: dict[str, Any],
        prefill_server: str,
        endpoint: str,
        headers: dict[str, str],
        metadata: Optional[AgenticRequestMetadata],
    ):
        """Run Direct-only P->D with a strict destination-first KV fence.

        D receives the request first and its native allocator remains the sole
        capacity authority.  Only a worker-authored marker proving physical KV
        ownership releases the matching P request.  No Router reservation,
        admission gate, or explicit request-count cap is used.
        """

        if getattr(self, "p2d_host_staging", False):
            raise RuntimeError(
                "native P->D preallocation cannot use P->D Host staging"
            )

        rooms = self._rooms(modified_request)
        parent_turn = metadata is not None and metadata.parent is not None
        arrival_at: Optional[float] = time.time() if parent_turn else None
        prefill_work: Optional[_PrefillWorkReservation] = None
        if getattr(self, "dynamic_prefill_domains", False):
            self._ensure_prefill_pressure_monitor()
            arrival = self._publish_parent_arrival(modified_request)
            if arrival is not None:
                arrival_at = float(arrival["arrived_at"])
            if getattr(self, "p2d_paired_credit_ablation", False):
                # The untargeted arrival above immediately starts the old-D
                # Direct/Slow decision. Only H2D recovery waits until this
                # request owns both a feasible P credit and native D space.
                prefill_work = await self._reserve_paired_prefill_credit(
                    modified_request
                )
            else:
                prefill_work = await self._resolve_dynamic_prefill_work(
                    modified_request, metadata, arrival_at
                )
            domain = prefill_work.domain
        else:
            domain = self._request_domain(metadata, rooms)

        self._set_prefill_attempt_rid(modified_request, replace=False)
        while True:
            if getattr(self, "numa_domains", False):
                prefill_server = self._bind_prefill_domain(
                    modified_request, domain
                )
            rooms = self._rooms(modified_request)
            self._activate_prefill_attempt(modified_request, rooms)
            attempt_active = True
            retain_prefill_work = False
            route_task: Optional[asyncio.Task] = None
            prefill_task: Optional[asyncio.Task] = None
            decode_task: Optional[asyncio.Task] = None
            decode_server: Optional[str] = None
            try:
                # Reserve the next-turn destination before targeting the
                # parent recovery at P.  Targeting first can fill P HBM with
                # recovered worksets whose matching D requests are still
                # blocked in the preallocation queue; at high concurrency
                # that creates a P-HBM <-> D-HBM resource cycle.  The D
                # endpoint is independent of the eventual same-node P sender,
                # so a later D->P Host redirect may safely retarget P while
                # retaining this physical destination allocation.
                decode_candidates = sorted(
                    self._domain_decode_urls(domain)
                )
                if not decode_candidates:
                    raise RuntimeError(
                        f"No Decode worker available for Prefill domain {domain}"
                    )
                decode_server = self._routing_rng.choice(decode_candidates)
                logger.info(
                    "PD_P2D_D_FIRST_PREALLOC rooms=%s P=%d D=%s",
                    rooms,
                    domain,
                    decode_server,
                )

                for room in rooms:
                    self._d_reserved_path(room).unlink(missing_ok=True)
                decode_task = asyncio.create_task(
                    session.post(
                        f"{decode_server}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    )
                )
                await self._wait_until_decode_reserved(
                    rooms, decode_task, None
                )
                logger.info(
                    "PD_P2D_D_RESERVED rooms=%s P=%d D=%s",
                    rooms,
                    domain,
                    decode_server,
                )

                if parent_turn:
                    self._publish_parent_arrival(
                        modified_request,
                        target_prefill_domain=(
                            domain
                            if getattr(
                                self, "dynamic_prefill_domains", False
                            )
                            else None
                        ),
                        arrived_at=arrival_at,
                    )
                    if (
                        prefill_work is not None
                        and prefill_work.route_pending
                        and metadata is not None
                    ):
                        route_task = asyncio.create_task(
                            self._watch_dynamic_prefill_route(
                                modified_request,
                                metadata,
                                prefill_work,
                                fixed_domain=getattr(
                                    self, "p2d_paired_credit_ablation", False
                                ),
                            )
                        )
                        if getattr(
                            self, "p2d_native_route_settled", False
                        ):
                            # D already owns the next-turn destination.  Now
                            # allow the selected P to claim/recover parent KV,
                            # and settle that route before P forward.  A Slow
                            # recovery may select another same-node P; NIXL's
                            # reserved D endpoint can receive from either P.
                            original_domain = domain
                            outcome = await route_task
                            route_task = None
                            prefill_work.route_pending = False
                            outcome_domain = outcome.get("domain")
                            if (
                                outcome.get("action") == "redirect"
                                and outcome_domain is not None
                                and int(outcome_domain) != prefill_work.domain
                            ):
                                await self._move_prefill_work(
                                    prefill_work, int(outcome_domain)
                                )
                            domain = prefill_work.domain
                            logger.info(
                                "PD_P2D_NATIVE_ROUTE_SETTLED rooms=%s P=%d "
                                "route=%s action=%s",
                                rooms,
                                domain,
                                outcome.get("route"),
                                outcome.get("action"),
                            )
                            if domain != original_domain:
                                prefill_server = self._bind_prefill_domain(
                                    modified_request, domain
                                )
                                logger.info(
                                    "PD_P2D_D_RESERVED_P_RETARGET rooms=%s "
                                    "from_P=%d to_P=%d D=%s",
                                    rooms,
                                    original_domain,
                                    domain,
                                    decode_server,
                                )

                prefill_task = asyncio.create_task(
                    session.post(
                        f"{prefill_server}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    )
                )

                prefill_response = await self._wait_prebound_prefill_or_redirect(
                    prefill_task, decode_task, route_task
                )
                if not modified_request.get("return_logprob", False):
                    try:
                        await prefill_response.read()
                    finally:
                        prefill_response.release()
                    prefill_response = None

                await self._release_prefill_work(prefill_work)
                if route_task is not None and not route_task.done():
                    route_task.cancel()
                    await asyncio.gather(route_task, return_exceptions=True)
                decode_response = await decode_task
                return prefill_response, decode_response
            except _PrefillRedirect as redirect:
                p_aborted = True
                d_aborted = True
                if prefill_task is not None:
                    p_aborted = await self._abort_prefill_attempt(
                        session,
                        prefill_server,
                        modified_request,
                        prefill_task,
                    )
                    prefill_task = None
                if decode_task is not None and decode_server is not None:
                    d_aborted = await self._abort_decode_attempt(
                        session, decode_server, modified_request
                    )
                    await self._dispose_response_task(decode_task)
                    decode_task = None
                if not p_aborted or not d_aborted:
                    raise RuntimeError(
                        "Cannot safely redirect native P->D attempt: "
                        f"P_abort={p_aborted} D_abort={d_aborted}"
                    )
                for room in rooms:
                    self._d_reserved_path(room).unlink(missing_ok=True)
                    self._accepted_path(room).unlink(missing_ok=True)
                    self._scheduled_path(room).unlink(missing_ok=True)
                    self._ready_path(room).unlink(missing_ok=True)
                self._deactivate_prefill_attempt(modified_request, rooms)
                attempt_active = False
                await self._move_prefill_work(prefill_work, redirect.domain)
                retain_prefill_work = True
                domain = redirect.domain
                self._replace_prefill_attempt_rooms(modified_request)
                self._set_prefill_attempt_rid(modified_request, replace=True)
                continue
            except BaseException:
                for room in rooms:
                    self._d_reserved_path(room).unlink(missing_ok=True)
                if prefill_task is not None:
                    await self._abort_prefill_attempt(
                        session,
                        prefill_server,
                        modified_request,
                        prefill_task,
                    )
                    prefill_task = None
                if decode_task is not None and decode_server is not None:
                    await self._abort_decode_attempt(
                        session, decode_server, modified_request
                    )
                    await self._dispose_response_task(decode_task)
                    decode_task = None
                raise
            finally:
                if route_task is not None and not route_task.done():
                    route_task.cancel()
                    await asyncio.gather(route_task, return_exceptions=True)
                if not retain_prefill_work:
                    await self._release_prefill_work(prefill_work)
                if attempt_active:
                    self._deactivate_prefill_attempt(modified_request, rooms)

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
                    metadata = AgenticRequestMetadata.from_custom_params(
                        envelope_params
                    )
        except (TypeError, ValueError):
            metadata = None
        policy = getattr(self, "_slow_congestion", None)
        parent_id = metadata.parent.snapshot_id if metadata is not None and metadata.parent is not None else None
        if policy is not None or getattr(self, "_multinode_static_recovery", False):
            self._ensure_prefill_pressure_monitor()
        if policy is not None and parent_id is not None:
            policy.enter(parent_id)
        remote_parents = getattr(self, "_multinode_recovery_parents", None)
        if getattr(self, "_multinode_static_recovery", False) and parent_id is not None:
            remote_parents[parent_id] = remote_parents.get(parent_id, 0) + 1
        try:
            if getattr(self, "p2d_native_prealloc_ablation", False):
                dispatch = self._native_prealloc_dispatch_with_metadata
            elif getattr(self, "p2d_prebind_ablation", False):
                dispatch = self._prebound_dispatch_with_metadata
            else:
                dispatch = self._late_dispatch_with_metadata
            return await dispatch(
                session, modified_request, prefill_server, endpoint, headers, metadata
            )
        finally:
            if getattr(self, "_multinode_static_recovery", False) and parent_id is not None:
                remaining = remote_parents[parent_id] - 1
                if remaining:
                    remote_parents[parent_id] = remaining
                else:
                    remote_parents.pop(parent_id, None)
            if policy is not None and parent_id is not None:
                policy.leave(parent_id)

    async def _late_dispatch_with_metadata(
        self, session, modified_request, prefill_server, endpoint, headers, metadata
    ):
        rooms = self._rooms(modified_request)
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
        pressure_handoff_task: Optional[asyncio.Task] = None

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
                    prefill_server = self._bind_prefill_domain(
                        modified_request, domain
                    )
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
                        target_prefill_domain=(
                            domain
                            if getattr(self, "dynamic_prefill_domains", False)
                            else None
                        ),
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
                                "parent admission cannot move before the old "
                                "Prefill attempt is quiescent"
                            )
                        admission_wait = 0.0
                    else:
                        admission_wait = await prefill_admission.acquire(
                            parent_turn=True
                        )
                        parent_admission = prefill_admission
                        parent_admission_domain = domain
                else:
                    admission_wait = await prefill_admission.acquire(
                        parent_turn=False
                    )
                if admission_wait >= 1.0:
                    logger.info(
                        "PD_P_ADMISSION rooms=%s parent_turn=%s wait_s=%.3f "
                        "P=%d active=%d limit=%d",
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
                            "PD_P_PARENT_ACCEPT rooms=%s wait_s=%.3f "
                            "tokenizer_inflight_limit=%d",
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
                    await self._wait_until_prefill_scheduled(
                        rooms, prefill_task, route_task
                    )
                    # The scheduled marker is published only after P owns the
                    # complete parent+suffix workset.  Keep Router shadow
                    # credit until a pressure fetch that began after this
                    # physical allocation, then remove it exactly once.  This
                    # applies equally to Direct and Slow H2D admission.
                    if (
                        prefill_work is not None
                        and prefill_work.direct_workset_tokens > 0
                        and pressure_handoff_task is None
                    ):
                        pressure_handoff_task = asyncio.create_task(
                            self._settle_direct_workset_after_pressure(
                                prefill_work,
                                direct_terminal_at=time.monotonic(),
                            )
                        )
                    prompt_tokens = await self._wait_until_prefill_ready(
                        rooms, prefill_task, route_task
                    )
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
                    aborted = await self._abort_prefill_attempt(
                        session, old_server, modified_request, prefill_task
                    )
                    if not aborted:
                        raise RuntimeError(
                            "Cannot safely redirect Prefill because abort was not "
                            f"acknowledged by {old_server}"
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
                        "PD_PREFILL_REDIRECT_SUBMIT route=%s from_P=%s to_P=%d "
                        "old_rooms=%s new_rooms=%s",
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
            if pressure_handoff_task is not None:
                await asyncio.gather(
                    pressure_handoff_task, return_exceptions=True
                )
            # Host staging is owned independently by each admission state
            # machine; it cannot gate the path choice of later P-ready work.
            p2d_attempt_snapshot = self._p2d_snapshot_for_rooms(rooms)
            # The request coroutine now becomes a pure producer.  The per-P
            # dispatcher owns FIFO order, future D-capacity credit and the
            # actual Decode POST as one state transition.
            ready_sequence = self._p_ready_sequence(rooms)
            ready_key = (
                (domain, ready_sequence)
                if getattr(self, "dynamic_prefill_domains", False)
                else ready_sequence
            )

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
                admission_task = asyncio.create_task(
                    self._release_reservation_when_admitted(
                        reservation, ready_key
                    )
                )
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
                    raise RuntimeError(
                        "P->D admission returned without a Decode reservation"
                    )
                return await commit_ready_reservation(selected_reservation)

            async def prepare_ready_snapshot() -> Optional[DecodeReservation]:
                """Persist one blocked P completion without waiting for D."""

                await self._stage_p2d_until_durable(
                    rooms, prompt_tokens, domain
                )
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
            if pressure_handoff_task is not None:
                await asyncio.gather(
                    pressure_handoff_task, return_exceptions=True
                )
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
                await self._abort_decode_attempt(
                    session, reservation.url, modified_request
                )
            await self._dispose_response_task(prefill_task)
            await self._dispose_response_task(decode_task)
            if admission_task is not None and not admission_task.done():
                admission_task.cancel()
                await asyncio.gather(admission_task, return_exceptions=True)
            if reservation is not None:
                async with self._selection_lock:
                    self._reservations.pop(reservation.reservation_id, None)
                if (
                    reservation.p2d_host_snapshot_id is not None
                    and getattr(self, "p2d_host_ledger", None) is not None
                ):
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
            self._abort_unsubmitted_p2d(
                p2d_attempt_snapshot, "router_attempt_ended_before_d_submit"
            )
            self._deactivate_prefill_attempt(modified_request, rooms)
            await release_parent_admission()

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert decode_server is None, "Late-binding D must not be selected early"
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        generation_key = self._agentic_generation_key(modified_request)
        if generation_key is None:
            ret_json, status = await self._generate_once(
                modified_request, prefill_server, endpoint
            )
        else:
            ret_json, status = await self._generate_singleflight(
                generation_key, modified_request, prefill_server, endpoint
            )
        return ORJSONResponse(content=ret_json, status_code=status)

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
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
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk
            finally:
                if prefill_response is not None:
                    prefill_response.release()
                if decode_response is not None:
                    decode_response.release()

        return StreamingResponse(stream_results(), media_type="text/event-stream")
