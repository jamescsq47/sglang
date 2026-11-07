import asyncio
import copy
import json
from types import SimpleNamespace

from sglang.srt.disaggregation.agentic_group_protocol import GenerationKey
from sglang.srt.disaggregation.dualpd.router import global_pd_router
from sglang.srt.disaggregation.dualpd.router.global_pd_router import (
    GlobalEndpointMiniLoadBalancer,
    application_final,
)


def _router(monkeypatch):
    prefill = [("http://p0:1", 11), ("http://p1:2", 12)]
    decode = ["http://d0:3", "http://d1:4"]
    monkeypatch.setenv(
        "DUALPD_ROUTER_ENDPOINT_GROUPS",
        json.dumps(
            {
                "http://p0:1": "p0",
                "http://p1:2": "p1",
                "http://d0:3": "d0",
                "http://d1:4": "d1",
            }
        ),
    )
    return GlobalEndpointMiniLoadBalancer(
        SimpleNamespace(
            prefill_urls=prefill,
            decode_urls=decode,
            request_timeout_secs=10,
            host="127.0.0.1",
            port=8000,
            policy="random",
            pd_disaggregation=True,
            otlp_traces_endpoint=None,
            enable_trace=False,
        )
    )


def test_application_final_targets_fixed_group_coordinator(monkeypatch):
    observed = []

    monkeypatch.setenv("SGLANG_AGENTIC_GROUP_ENDPOINT", "tcp://10.1.2.3:29590")
    monkeypatch.setenv("SGLANG_AGENTIC_GROUP_RUN_ID", "run-fixed")
    monkeypatch.setenv("SGLANG_AGENTIC_GROUP_GROUP_ID", "p0--d0")
    monkeypatch.setenv("SGLANG_AGENTIC_GROUP_TOKEN", "secret")
    monkeypatch.setattr(
        global_pd_router,
        "send_application_final",
        lambda address, **kwargs: observed.append((address, kwargs)),
    )

    assert asyncio.run(
        application_final({"request_id": "instance-17", "generation": 9})
    ) == {"ok": True}
    assert observed == [
        (
            ("10.1.2.3", 29590),
            {
                "run_id": "run-fixed",
                "group_id": "p0--d0",
                "token": "secret",
                "key": GenerationKey("run-fixed", "instance-17", 9),
            },
        )
    ]
def test_entry_selection_does_not_early_bind_or_charge_decode(monkeypatch):
    router = _router(monkeypatch)

    purl, _bootstrap, placeholder = router.select_pair()

    assert purl in router.prefill_urls
    assert placeholder == router.decode_urls[0]
    assert sum(router._inflight[url] for url in router.decode_urls) == 0


def test_prefill_selection_charges_projected_tokens_before_next_request(monkeypatch):
    router = _router(monkeypatch)
    router._load["http://p0:1"] = (100, 1000, 0)
    router._load["http://p1:2"] = (100, 1000, 0)

    first, _bootstrap, _decode = router.select_pair()
    second, _bootstrap, _decode = router.select_pair()

    assert first != second
    assert router._prefill_reserved_tokens[first] == 16384
    assert router._prefill_reserved_tokens[second] == 16384


def test_exact_prefill_reservation_replaces_provisional_and_releases(monkeypatch):
    router = _router(monkeypatch)
    purl, _bootstrap, _decode = router.select_pair()

    exact = router._materialize_prefill_reservation(
        purl,
        {
            "input_ids": [1, 2],
            "custom_params": {"agentic_prompt_token_count": 9000},
        },
    )

    assert exact == 9000
    assert router._prefill_reserved_tokens[purl] == 9000
    router._release(purl, prefill_tokens=exact)
    assert router._prefill_reserved_tokens[purl] == 0
    assert router._inflight[purl] == 0


def test_prefill_ready_is_edge_triggered_even_before_waiter(monkeypatch):
    router = _router(monkeypatch)
    request = {"bootstrap_room": 123}

    async def exercise():
        await router.publish_prefill_ready(
            {"bootstrap_room": 123, "required_tokens": 99}
        )
        return await router._wait_prefill_ready(request)

    assert asyncio.run(exercise())["required_tokens"] == 99


def test_decode_late_binding_filters_capacity_then_chooses_lower_load(monkeypatch):
    router = _router(monkeypatch)
    router._load["http://d0:3"] = (950, 1000, 0)
    router._load["http://d1:4"] = (200, 1000, 2)

    selected = asyncio.run(router._select_decode(100))

    assert selected == "http://d1:4"
    assert router._inflight[selected] == 1


def test_decode_selection_charges_projected_tokens_before_next_sample(monkeypatch):
    router = _router(monkeypatch)
    router._load["http://d0:3"] = (100, 1000, 0)
    router._load["http://d1:4"] = (100, 1000, 0)

    async def select_twice():
        return await router._select_decode(300), await router._select_decode(300)

    first, second = asyncio.run(select_twice())

    assert first != second
    assert router._decode_reserved_tokens[first] == 300
    assert router._decode_reserved_tokens[second] == 300


def test_decode_projected_reservation_expires_without_touching_allocator(
    monkeypatch,
):
    router = _router(monkeypatch)
    router._load["http://d0:3"] = (100, 1000, 0)
    router._load["http://d1:4"] = (900, 1000, 0)

    selected = asyncio.run(router._select_decode(200, "reservation-1"))
    assert router._decode_reserved_tokens[selected] == 200

    expires = router._decode_reservations[selected]["reservation-1"].expires_at
    router._prune_decode_reservations(expires)

    assert router._decode_reserved_tokens[selected] == 0
    assert not router._decode_reservations[selected]


def test_decode_projected_reservation_only_bridges_target_prepare(monkeypatch):
    monkeypatch.delenv("DUALPD_ROUTER_DECODE_RESERVATION_SECONDS", raising=False)
    router = _router(monkeypatch)

    assert router._decode_reservation_seconds == router._generation_timeout


def test_decode_materialized_replaces_shadow_only_after_later_started_sample(
    monkeypatch,
):
    router = _router(monkeypatch)
    router._load["http://d0:3"] = (100, 1000, 0)
    router._load["http://d1:4"] = (900, 1000, 0)

    selected = asyncio.run(router._select_decode(200, "reservation-2"))
    assert selected == "http://d0:3"
    asyncio.run(
        router.publish_decode_materialized(
            {"reservation_id": "reservation-2", "target_group": "d0"}
        )
    )

    assert router._decode_reserved_tokens[selected] == 200
    router._load_epoch[selected] += 1
    router._release_sampled_decode_reservations(selected)
    assert router._decode_reserved_tokens[selected] == 0
    assert "reservation-2" not in router._decode_reservation_index


def test_inflight_old_load_sample_cannot_consume_materialized_shadow(monkeypatch):
    router = _router(monkeypatch)
    router._load["http://d0:3"] = (100, 1000, 0)
    router._load["http://d1:4"] = (900, 1000, 0)
    selected = asyncio.run(router._select_decode(200, "reservation-race"))
    router._load_started_epoch[selected] = 1
    asyncio.run(
        router.publish_decode_materialized(
            {"reservation_id": "reservation-race", "target_group": "d0"}
        )
    )

    router._load_epoch[selected] = 1
    router._release_sampled_decode_reservations(selected)
    assert router._decode_reserved_tokens[selected] == 200

    router._load_started_epoch[selected] = 2
    router._load_epoch[selected] = 2
    router._release_sampled_decode_reservations(selected)
    assert router._decode_reserved_tokens[selected] == 0


def test_decode_materialized_rejects_wrong_target(monkeypatch):
    router = _router(monkeypatch)
    asyncio.run(router._select_decode(200, "reservation-3"))

    async def publish_wrong_target():
        await router.publish_decode_materialized(
            {"reservation_id": "reservation-3", "target_group": "missing"}
        )

    import pytest

    with pytest.raises(ValueError, match="target differs"):
        asyncio.run(publish_wrong_target())


def test_generation_timeout_is_separate_from_control_timeout(monkeypatch):
    monkeypatch.setenv("DUALPD_ROUTER_GENERATION_TIMEOUT_SECONDS", "900")
    router = _router(monkeypatch)

    timeout = router._backend_timeout()

    assert router.timeout == 10
    assert timeout.total == 900
    assert timeout.connect == 10
    assert timeout.sock_read == 900


def test_non_logprob_result_does_not_wait_for_unused_prefill_response(monkeypatch):
    router = _router(monkeypatch)
    router._load["http://d0:3"] = (0, 1000, 0)
    router._load["http://d1:4"] = (900, 1000, 0)
    purl, _bootstrap, placeholder = router.select_pair()
    observed = {"prefill_cancelled": False}

    class Response:
        status = 200

        async def json(self):
            return {"choice": "decode-owned"}

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **_kwargs):
            if url.startswith(purl):
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    observed["prefill_cancelled"] = True
                    raise
            return Response()

    async def ready(_request):
        return {"required_tokens": 100}

    monkeypatch.setattr(global_pd_router.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(router, "_wait_prefill_ready", ready)

    request = {
        "bootstrap_room": "room-1",
        "input_ids": [1, 2, 3],
        "custom_params": {"agentic_prompt_token_count": 3},
    }
    response = asyncio.run(
        router.generate(request, purl, placeholder, "v1/chat/completions")
    )

    assert json.loads(response.body) == {"choice": "decode-owned"}
    assert observed["prefill_cancelled"] is True
    assert router._inflight[purl] == 0
    assert router._prefill_reserved_tokens[purl] == 0
    assert sum(router._inflight[url] for url in router.decode_urls) == 0
    assert not router._decode_reservation_index


def test_generation_retry_joins_live_backend_after_frontend_cancel(monkeypatch):
    router = _router(monkeypatch)
    router._load["http://d0:3"] = (0, 1000, 0)
    router._load["http://d1:4"] = (900, 1000, 0)
    first_purl, _bootstrap, placeholder = router.select_pair()
    decode_release = asyncio.Event()
    calls = {"prefill": 0, "decode": 0}

    class Response:
        status = 200

        async def json(self):
            return {"choice": "single-physical-generation"}

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **_kwargs):
            if url.startswith(first_purl):
                calls["prefill"] += 1
                await asyncio.Future()
            calls["decode"] += 1
            await decode_release.wait()
            return Response()

    async def ready(_request):
        return {"required_tokens": 100}

    monkeypatch.setattr(global_pd_router.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(router, "_wait_prefill_ready", ready)
    request = {
        "bootstrap_room": "first-room",
        "input_ids": [1, 2, 3],
        "custom_params": {
            "agentic_request_id": "trajectory-1",
            "agentic_generation": 7,
            "agentic_prompt_token_count": 3,
        },
    }

    async def exercise():
        first = asyncio.create_task(
            router.generate(request, first_purl, placeholder, "v1/chat/completions")
        )
        while calls["decode"] == 0:
            await asyncio.sleep(0)
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass

        retry_purl, _bootstrap, retry_placeholder = router.select_pair()
        retry_request = copy.deepcopy(request)
        retry_request["bootstrap_room"] = "retry-room"
        retry = asyncio.create_task(
            router.generate(
                retry_request,
                retry_purl,
                retry_placeholder,
                "v1/chat/completions",
            )
        )
        await asyncio.sleep(0)
        decode_release.set()
        response = await retry
        await router.close()
        return response

    response = asyncio.run(exercise())
    assert json.loads(response.body) == {"choice": "single-physical-generation"}
    assert calls == {"prefill": 1, "decode": 1}
    assert sum(router._inflight.values()) == 0
    assert sum(router._prefill_reserved_tokens.values()) == 0
    assert not router._decode_reservation_index
