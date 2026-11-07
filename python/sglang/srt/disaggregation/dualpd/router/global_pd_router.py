#!/usr/bin/env python3
"""File-free HTTP router for one global multi-endpoint DualPD fabric.

Each generation may choose a new logical P and D group.  The selected endpoint
identities are injected into the immutable request metadata consumed by the
TCP lifecycle authority; this keeps HTTP dispatch and KV ownership routing in
the same transaction.  Local inflight counts provide a nonblocking least-load
tie-breaker.  Physical page reservation remains authoritative on both TP
ranks and is never inferred from this HTTP-side hint.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import asyncio
import copy
import uuid
from dataclasses import dataclass

import aiohttp

import sglang_router.mini_lb as mini_lb_module
import uvicorn
from fastapi import HTTPException
from fastapi.responses import ORJSONResponse, StreamingResponse
from sglang.srt.disaggregation.agentic_group_protocol import (
    GenerationKey,
    send_application_final,
)
from sglang_router.launch_router import parse_router_args


@dataclass(slots=True)
class _DecodeReservation:
    reservation_id: str
    tokens: int
    expires_at: float
    materialized_after_started_epoch: int | None = None


class GlobalEndpointMiniLoadBalancer(mini_lb_module.MiniLoadBalancer):
    def __init__(self, args):
        super().__init__(args)
        raw = json.loads(os.environ["DUALPD_ROUTER_ENDPOINT_GROUPS"])
        self._groups = {str(url).rstrip("/"): str(group) for url, group in raw.items()}
        endpoints = [*self.prefill_urls, *self.decode_urls]
        missing = [url for url in endpoints if url.rstrip("/") not in self._groups]
        if missing:
            raise ValueError(f"router group mapping lacks endpoints: {missing}")
        self._inflight = {url.rstrip("/"): 0 for url in endpoints}
        self._load = {url.rstrip("/"): (0, 0, 0) for url in endpoints}
        self._prefill_reserved_tokens = {
            url.rstrip("/"): 0 for url in self.prefill_urls
        }
        self._pending_prefill_reservations = {
            url.rstrip("/"): 0 for url in self.prefill_urls
        }
        # /get_load is sampled periodically.  Without a projected D charge,
        # every P-ready callback arriving in the same sample interval sees the
        # same empty Decode endpoint and creates a thundering herd.  These
        # Reservations bridge selection -> target materialization becoming visible in
        # /get_load.  They must not span the complete cross-node DMA, otherwise
        # the same workset is counted once physically and once here.  The D
        # memory authority remains the sole allocator and rejects real races.
        self._decode_reserved_tokens = {
            url.rstrip("/"): 0 for url in self.decode_urls
        }
        self._decode_reservations = {
            url.rstrip("/"): {} for url in self.decode_urls
        }
        self._decode_reservation_index = {}
        self._load_epoch = {url.rstrip("/"): 0 for url in self.decode_urls}
        self._load_started_epoch = {
            url.rstrip("/"): 0 for url in self.decode_urls
        }
        self._decode_growth_reserve_tokens = max(
            0,
            int(
                os.getenv(
                    "SGLANG_AGENTIC_MULTINODE_DECODE_GROWTH_TOKENS", "0"
                )
            ),
        )
        self._default_prefill_reservation = max(
            1,
            int(os.getenv("DUALPD_ROUTER_PREFILL_RESERVATION_TOKENS", "16384")),
        )
        self._tp_size = max(1, int(os.getenv("DUALPD_ROUTER_TP_SIZE", "1")))
        self._tie = 0
        self._poll_task = None
        self._ready_waiters = {}
        self._early_ready = {}
        # One immutable agent generation may be replayed by the application
        # transport after its client connection disappears.  Keep the actual
        # P/D backend operation alive and let a retry join that operation
        # instead of issuing a second request with the same lifecycle key.
        # The lifecycle registry must remain strict: two physical executions
        # of one generation are an ownership bug, not something it should
        # silently merge.
        self._generation_tasks = {}
        self._generation_completed_at = {}
        self._generation_retry_ttl = max(
            1.0,
            float(os.getenv("DUALPD_ROUTER_GENERATION_RETRY_TTL_SECONDS", "300")),
        )
        self._generation_retry_cache_max = max(
            1,
            int(os.getenv("DUALPD_ROUTER_GENERATION_RETRY_CACHE_MAX", "4096")),
        )
        # ``self.timeout`` protects control-plane admission.  A Decode call
        # may legitimately run much longer under c128, so it must not inherit
        # that short deadline and cause the same generation to be retried.
        self._generation_timeout = float(
            os.getenv("DUALPD_ROUTER_GENERATION_TIMEOUT_SECONDS", "3600")
        )
        if self._generation_timeout <= 0:
            raise ValueError("generation timeout must be positive")
        # This is a crash-recovery bound only.  Normal release is causal:
        # all target TP ranks finish DMA, then one newer /get_load sample replaces
        # this Router shadow charge with the allocator's physical charge.
        self._decode_reservation_seconds = max(
            self._generation_timeout,
            float(
                os.getenv(
                    "DUALPD_ROUTER_DECODE_RESERVATION_SECONDS",
                    str(self._generation_timeout),
                )
            ),
        )

    def _backend_timeout(self):
        return aiohttp.ClientTimeout(
            total=self._generation_timeout,
            connect=self.timeout,
            sock_connect=self.timeout,
            sock_read=self._generation_timeout,
        )

    def _least(self, values):
        start = self._tie
        self._tie += 1
        return min(
            enumerate(values),
            key=lambda item: (
                (
                    (
                        self._load[item[1].rstrip("/")][0]
                        + self._prefill_reserved_tokens.get(
                            item[1].rstrip("/"), 0
                        )
                    )
                    / self._load[item[1].rstrip("/")][1]
                    if self._load[item[1].rstrip("/")][1] > 0
                    else 0.0
                ),
                self._load[item[1].rstrip("/")][2],
                self._inflight[item[1].rstrip("/")],
                (item[0] - start) % len(values),
            ),
        )[0]

    async def _sample(self, session, url):
        endpoint = url.rstrip("/")
        sample_epoch = 0
        if endpoint in self._load_started_epoch:
            self._load_started_epoch[endpoint] += 1
            sample_epoch = self._load_started_epoch[endpoint]
        try:
            async with session.get(
                f"{url.rstrip('/')}/get_load",
                timeout=aiohttp.ClientTimeout(total=1.0),
            ) as response:
                response.raise_for_status()
                rows = await response.json()
            if isinstance(rows, dict):
                rows = [rows]
            used = sum(
                int(row.get("num_physical_used_tokens", row.get("num_tokens", 0)))
                for row in rows
            )
            capacity = sum(
                max(
                    0,
                    int(row.get("max_total_num_tokens", 0))
                    - (
                        self._decode_growth_reserve_tokens
                        if endpoint in self._decode_reservations
                        else 0
                    ),
                )
                for row in rows
            )
            waiting = sum(int(row.get("num_waiting_reqs", 0)) for row in rows)
            self._load[endpoint] = (used, capacity, waiting)
            if endpoint in self._load_epoch:
                self._load_epoch[endpoint] = sample_epoch
                self._release_sampled_decode_reservations(endpoint)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError):
            # Physical TP reservation is still authoritative.  A stale load
            # sample may affect balancing but can never grant memory.
            return

    async def _poll(self):
        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.gather(
                    *(self._sample(session, url) for url in self._inflight),
                    return_exceptions=True,
                )
                await asyncio.sleep(0.2)

    async def start_polling(self):
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll())

    async def close(self):
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        tasks = tuple(self._generation_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._generation_tasks.clear()
        self._generation_completed_at.clear()

    def select_pair(self):
        """Select only P here; D is deliberately late-bound after Prefill."""

        pidx = self._least(self.prefill_urls)
        purl = self.prefill_urls[pidx]
        endpoint = purl.rstrip("/")
        self._inflight[endpoint] += 1
        # select_pair() runs before the request body reaches generate().  Book
        # a conservative token reservation now, then replace it with the exact
        # prompt/workset estimate synchronously at the start of generate().
        # The event loop cannot interleave another selection between those two
        # steps, so concurrent requests never all choose the same stale /get_load
        # sample.
        self._prefill_reserved_tokens[endpoint] += (
            self._default_prefill_reservation * self._tp_size
        )
        self._pending_prefill_reservations[endpoint] += 1
        # MiniLB's public handler requires a placeholder D URL.  generate()
        # ignores it and selects the real endpoint only on P-ready.
        return purl, self.prefill_bootstrap_ports[pidx], self.decode_urls[0]

    @staticmethod
    def _prompt_tokens(request):
        custom = request.get("custom_params") or {}
        value = custom.get("agentic_prompt_token_count")
        if value is not None:
            return max(1, int(value))
        values = request.get("input_ids")
        if isinstance(values, list) and (not values or isinstance(values[0], int)):
            return max(1, len(values))
        return None

    def _materialize_prefill_reservation(self, purl, request):
        endpoint = purl.rstrip("/")
        if self._pending_prefill_reservations[endpoint] <= 0:
            raise RuntimeError("missing provisional Prefill reservation")
        self._pending_prefill_reservations[endpoint] -= 1
        exact = self._prompt_tokens(request) or self._default_prefill_reservation
        self._prefill_reserved_tokens[endpoint] += (
            (exact - self._default_prefill_reservation) * self._tp_size
        )
        return exact

    def _annotate_prefill(self, request, purl):
        custom = dict(request.get("custom_params") or {})
        custom["agentic_prefill_group"] = self._groups[purl.rstrip("/")]
        custom.pop("agentic_decode_group", None)
        request["custom_params"] = custom

    def _annotate_decode(self, request, durl, reservation_id):
        custom = dict(request.get("custom_params") or {})
        custom["agentic_decode_group"] = self._groups[durl.rstrip("/")]
        custom["agentic_decode_reservation_id"] = str(reservation_id)
        request["custom_params"] = custom

    @staticmethod
    def _room(request):
        room = request.get("bootstrap_room")
        if isinstance(room, list):
            if len(room) != 1:
                raise ValueError("global late binding currently requires batch size one")
            room = room[0]
        return str(room)

    async def publish_prefill_ready(self, payload):
        room = str(payload.get("bootstrap_room"))
        if room in {"", "None"}:
            raise ValueError("Prefill-ready callback omitted bootstrap_room")
        waiter = self._ready_waiters.pop(room, None)
        if waiter is None:
            self._early_ready[room] = dict(payload)
        elif not waiter.done():
            waiter.set_result(dict(payload))

    async def _wait_prefill_ready(self, request):
        room = self._room(request)
        ready = self._early_ready.pop(room, None)
        if ready is not None:
            return ready
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if room in self._ready_waiters:
            raise RuntimeError(f"duplicate Prefill-ready waiter for room {room}")
        self._ready_waiters[room] = future
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        finally:
            self._ready_waiters.pop(room, None)

    async def _select_decode(self, required_tokens, reservation_id=None):
        required_tokens = max(0, int(required_tokens))
        reservation_id = str(reservation_id or uuid.uuid4().hex)
        if reservation_id in self._decode_reservation_index:
            raise RuntimeError("duplicate Decode reservation identity")
        deadline = asyncio.get_running_loop().time() + self.timeout
        while True:
            self._prune_decode_reservations()
            feasible = []
            for index, url in enumerate(self.decode_urls):
                endpoint = url.rstrip("/")
                used, capacity, waiting = self._load[endpoint]
                projected = used + self._decode_reserved_tokens[endpoint]
                if capacity <= 0 or capacity - projected >= required_tokens:
                    feasible.append(
                        (index, url, projected, capacity, waiting)
                    )
            if feasible:
                start = self._tie
                self._tie += 1
                _index, selected, *_ = min(
                    feasible,
                    key=lambda item: (
                        item[2] / item[3] if item[3] > 0 else 0.0,
                        item[4],
                        self._inflight[item[1].rstrip("/")],
                        (item[0] - start) % len(self.decode_urls),
                    ),
                )
                endpoint = selected.rstrip("/")
                self._inflight[endpoint] += 1
                self._decode_reserved_tokens[endpoint] += required_tokens
                reservation = _DecodeReservation(
                    reservation_id=reservation_id,
                    tokens=required_tokens,
                    expires_at=(
                        time.monotonic() + self._decode_reservation_seconds
                    ),
                )
                self._decode_reservations[endpoint][reservation_id] = reservation
                self._decode_reservation_index[reservation_id] = endpoint
                return selected
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("no Decode TP group can reserve the complete workset")
            await asyncio.sleep(0.02)

    def _prune_decode_reservations(self, now=None):
        now = time.monotonic() if now is None else float(now)
        for endpoint, reservations in self._decode_reservations.items():
            expired = [
                reservation_id
                for reservation_id, reservation in reservations.items()
                if reservation.expires_at <= now
            ]
            for reservation_id in expired:
                self._release_decode_reservation(reservation_id)
            if self._decode_reserved_tokens[endpoint] < 0:
                raise RuntimeError("Decode token reservation underflow")

    def _release_decode_reservation(self, reservation_id):
        reservation_id = str(reservation_id)
        endpoint = self._decode_reservation_index.pop(reservation_id, None)
        if endpoint is None:
            return False
        reservation = self._decode_reservations[endpoint].pop(reservation_id)
        self._decode_reserved_tokens[endpoint] -= reservation.tokens
        if self._decode_reserved_tokens[endpoint] < 0:
            raise RuntimeError("Decode token reservation underflow")
        return True

    async def publish_decode_materialized(self, payload):
        reservation_id = str(payload.get("reservation_id", ""))
        target_group = str(payload.get("target_group", ""))
        if not reservation_id or not target_group:
            raise ValueError("Decode materialized callback omitted reservation identity")
        endpoint = self._decode_reservation_index.get(reservation_id)
        if endpoint is None:
            # Idempotent retry after a newer physical sample already consumed
            # the shadow reservation.
            return
        if self._groups.get(endpoint) != target_group:
            raise ValueError("Decode PREPARED target differs from Router reservation")
        reservation = self._decode_reservations[endpoint][reservation_id]
        if reservation.materialized_after_started_epoch is None:
            # Capture requests that had already started, not merely samples
            # that had completed.  A GET issued before DMA_DONE but returning
            # afterward is not allowed to consume this shadow reservation.
            reservation.materialized_after_started_epoch = (
                self._load_started_epoch[endpoint]
            )

    def _release_sampled_decode_reservations(self, endpoint):
        epoch = self._load_epoch[endpoint]
        ready = [
            reservation_id
            for reservation_id, reservation in self._decode_reservations[
                endpoint
            ].items()
            if reservation.materialized_after_started_epoch is not None
            and reservation.materialized_after_started_epoch < epoch
        ]
        for reservation_id in ready:
            self._release_decode_reservation(reservation_id)

    def _release(self, purl, durl=None, *, prefill_tokens=0):
        endpoint = purl.rstrip("/")
        self._inflight[endpoint] -= 1
        if prefill_tokens:
            self._prefill_reserved_tokens[endpoint] -= (
                int(prefill_tokens) * self._tp_size
            )
            if self._prefill_reserved_tokens[endpoint] < 0:
                raise RuntimeError("Prefill token reservation underflow")
        if durl is not None:
            self._inflight[durl.rstrip("/")] -= 1

    @staticmethod
    async def _settle_backend_task(task, label):
        """Cancel and consume one backend request without masking D success.

        A non-logprob response is wholly owned by Decode.  Prefill's HTTP
        response carries no user data and may be lost after its readiness
        callback and ownership handoff have already succeeded.  Waiting for
        that unused response before Decode used to strand the client for the
        full generation timeout and made the outer HTTP retry replay the same
        immutable generation.  Always observe the task, but treat it as
        cleanup once Decode has supplied the authoritative response.
        """

        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            logging.warning("%s backend cleanup failed: %r", label, error)

    @staticmethod
    def _generation_key(request):
        custom = request.get("custom_params") or {}
        request_id = custom.get("agentic_request_id")
        generation = custom.get("agentic_generation")
        if request_id is None or generation is None:
            return None
        return str(request_id), int(generation)

    def _prune_generation_tasks(self):
        now = asyncio.get_running_loop().time()
        expired = [
            key
            for key, completed_at in self._generation_completed_at.items()
            if now - completed_at >= self._generation_retry_ttl
        ]
        for key in expired:
            self._generation_completed_at.pop(key, None)
            self._generation_tasks.pop(key, None)

        overflow = len(self._generation_tasks) - self._generation_retry_cache_max
        if overflow <= 0:
            return
        oldest = sorted(
            self._generation_completed_at.items(), key=lambda item: item[1]
        )
        for key, _completed_at in oldest[:overflow]:
            self._generation_completed_at.pop(key, None)
            self._generation_tasks.pop(key, None)

    def _record_generation_done(self, key, task):
        if self._generation_tasks.get(key) is task:
            self._generation_completed_at[key] = asyncio.get_running_loop().time()

    def _release_duplicate_prefill_selection(self, request, purl):
        # MiniLB calls select_pair() before generate(), so even an idempotent
        # retry arrives with a fresh provisional P charge.  Consume and
        # release that shadow charge without touching either physical worker.
        prefill_tokens = self._materialize_prefill_reservation(purl, request)
        self._release(purl, prefill_tokens=prefill_tokens)

    async def generate(self, request, purl, durl, endpoint):
        key = self._generation_key(request)
        if key is None:
            return await self._generate_once(request, purl, durl, endpoint)

        self._prune_generation_tasks()
        task = self._generation_tasks.get(key)
        if task is not None:
            self._release_duplicate_prefill_selection(request, purl)
        else:
            task = asyncio.create_task(
                self._generate_once(request, purl, durl, endpoint),
                name=f"dualpd-generation-{key[0]}-{key[1]}",
            )
            self._generation_tasks[key] = task
            task.add_done_callback(
                lambda completed, generation_key=key: self._record_generation_done(
                    generation_key, completed
                )
            )

        # A caller disconnect cancels only this waiter.  P/D ownership and
        # transport continue to their normal fence, and a retry joins `task`.
        return await asyncio.shield(task)

    async def _generate_once(self, request, purl, durl, endpoint):
        del durl
        prefill_tokens = self._materialize_prefill_reservation(purl, request)
        prefill_request = copy.deepcopy(request)
        self._annotate_prefill(prefill_request, purl)
        selected_d = None
        decode_reservation_id = None
        p_released = False
        prefill_task = None
        decode_task = None
        try:
            async with aiohttp.ClientSession(
                timeout=self._backend_timeout()
            ) as session:
                async def run_prefill():
                    nonlocal p_released
                    try:
                        return await session.post(
                            f"{purl}/{endpoint}", json=prefill_request
                        )
                    finally:
                        if not p_released:
                            self._release(purl, prefill_tokens=prefill_tokens)
                            p_released = True

                prefill_task = asyncio.create_task(run_prefill())
                ready = await self._wait_prefill_ready(prefill_request)
                decode_reservation_id = f"{self._room(prefill_request)}:{uuid.uuid4().hex}"
                selected_d = await self._select_decode(
                    ready.get("required_tokens", 0), decode_reservation_id
                )
                decode_request = copy.deepcopy(prefill_request)
                self._annotate_decode(
                    decode_request, selected_d, decode_reservation_id
                )
                decode_task = asyncio.create_task(
                    session.post(f"{selected_d}/{endpoint}", json=decode_request)
                )
                if "return_logprob" in request:
                    prefill_response = await prefill_task
                    decode_response = await decode_task
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
                    # Decode owns the user-visible result.  Do not let an
                    # already-unneeded P response hold this request open.
                    decode_response = await decode_task
                    ret_json = await decode_response.json()
                return ORJSONResponse(
                    content=ret_json, status_code=decode_response.status
                )
        finally:
            await self._settle_backend_task(prefill_task, "Prefill")
            await self._settle_backend_task(decode_task, "Decode")
            if not p_released:
                self._release(purl, prefill_tokens=prefill_tokens)
            if selected_d is not None:
                self._inflight[selected_d.rstrip("/")] -= 1
            if decode_reservation_id is not None:
                self._release_decode_reservation(decode_reservation_id)

    async def generate_stream(self, request, purl, durl, endpoint="generate"):
        del durl
        prefill_tokens = self._materialize_prefill_reservation(purl, request)
        prefill_request = copy.deepcopy(request)
        self._annotate_prefill(prefill_request, purl)

        async def stream_results():
            selected_d = None
            decode_reservation_id = None
            p_released = False
            prefill_task = None
            decode_task = None
            try:
                async with aiohttp.ClientSession(
                    timeout=self._backend_timeout()
                ) as session:
                    async def run_prefill():
                        nonlocal p_released
                        try:
                            return await session.post(
                                f"{purl}/{endpoint}", json=prefill_request
                            )
                        finally:
                            if not p_released:
                                self._release(
                                    purl, prefill_tokens=prefill_tokens
                                )
                                p_released = True

                    prefill_task = asyncio.create_task(run_prefill())
                    ready = await self._wait_prefill_ready(prefill_request)
                    decode_reservation_id = (
                        f"{self._room(prefill_request)}:{uuid.uuid4().hex}"
                    )
                    selected_d = await self._select_decode(
                        ready.get("required_tokens", 0), decode_reservation_id
                    )
                    decode_request = copy.deepcopy(prefill_request)
                    self._annotate_decode(
                        decode_request, selected_d, decode_reservation_id
                    )
                    decode_task = asyncio.create_task(
                        session.post(f"{selected_d}/{endpoint}", json=decode_request)
                    )
                    if request.get("return_logprob", False):
                        prefill_response = await prefill_task
                        decode_response = await decode_task
                        prefill_chunks = [chunk async for chunk in prefill_response.content]
                        first = json.loads(
                            prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                        )
                        async for chunk in decode_response.content:
                            decoded = chunk.decode("utf-8")
                            if decoded.startswith("data:") and "[DONE]" not in decoded:
                                value = json.loads(decoded[5:].strip("\n"))
                                value["meta_info"]["input_token_logprobs"] = (
                                    first["meta_info"]["input_token_logprobs"]
                                    + value["meta_info"]["input_token_logprobs"]
                                )
                                yield b"data: " + json.dumps(value).encode() + b"\n\n"
                            else:
                                yield chunk
                    else:
                        decode_response = await decode_task
                        async for chunk in decode_response.content.iter_chunked(65536):
                            yield chunk
            finally:
                await self._settle_backend_task(prefill_task, "Prefill")
                await self._settle_backend_task(decode_task, "Decode")
                if not p_released:
                    self._release(purl, prefill_tokens=prefill_tokens)
                if selected_d is not None:
                    self._inflight[selected_d.rstrip("/")] -= 1
                if decode_reservation_id is not None:
                    self._release_decode_reservation(decode_reservation_id)

        return StreamingResponse(stream_results(), media_type="text/event-stream")


@mini_lb_module.app.post("/dualpd/prefill_ready")
async def prefill_ready(payload: dict):
    router = mini_lb_module.lb
    if not isinstance(router, GlobalEndpointMiniLoadBalancer):
        raise HTTPException(status_code=503, detail="global DualPD router not ready")
    try:
        await router.publish_prefill_ready(payload)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"ok": True}


@mini_lb_module.app.post("/dualpd/decode_materialized")
async def decode_materialized(payload: dict):
    router = mini_lb_module.lb
    if not isinstance(router, GlobalEndpointMiniLoadBalancer):
        raise HTTPException(status_code=503, detail="global DualPD router not ready")
    try:
        await router.publish_decode_materialized(payload)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"ok": True}


@mini_lb_module.app.post("/dualpd/application_final")
async def application_final(payload: dict):
    """Forward harness terminality to the in-memory lifecycle authority."""

    try:
        endpoint = os.environ["SGLANG_AGENTIC_GROUP_ENDPOINT"].removeprefix(
            "tcp://"
        )
        host, port = endpoint.rsplit(":", 1)
        key = GenerationKey(
            os.environ["SGLANG_AGENTIC_GROUP_RUN_ID"],
            str(payload["request_id"]),
            int(payload["generation"]),
        )
        await asyncio.to_thread(
            send_application_final,
            (host, int(port)),
            run_id=os.environ["SGLANG_AGENTIC_GROUP_RUN_ID"],
            group_id=os.environ["SGLANG_AGENTIC_GROUP_GROUP_ID"],
            token=os.environ["SGLANG_AGENTIC_GROUP_TOKEN"],
            key=key,
        )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"ok": True}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_router_args(sys.argv[1:])
    args.mini_lb = True
    router = GlobalEndpointMiniLoadBalancer(args)
    mini_lb_module.lb = router
    mini_lb_module.app.router.add_event_handler("startup", router.start_polling)
    mini_lb_module.app.router.add_event_handler("shutdown", router.close)
    if router.enable_trace:
        mini_lb_module.process_tracing_init(router.otlp_traces_endpoint, "sglang")
        mini_lb_module.trace_set_thread_info("Mini lb")
    uvicorn.run(mini_lb_module.app, host=router.host, port=router.port, loop="asyncio")


if __name__ == "__main__":
    main()
