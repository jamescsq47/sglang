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
import asyncio

import aiohttp

import sglang_router.mini_lb as mini_lb_module
import uvicorn
from fastapi import HTTPException
from sglang.srt.disaggregation.agentic_group_protocol import (
    GenerationKey,
    send_application_final,
)
from sglang_router.launch_router import parse_router_args


from .tcp_capacity import TCPCapacityMixin, _DecodeReservation
from .tcp_http import TCPHTTPMixin


class GlobalEndpointMiniLoadBalancer(
    TCPCapacityMixin, TCPHTTPMixin, mini_lb_module.MiniLoadBalancer
):
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
