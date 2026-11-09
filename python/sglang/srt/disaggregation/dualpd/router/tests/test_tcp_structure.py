"""TCP boundaries preserve native callbacks, records and one concrete owner."""
import ast
import asyncio
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import typing
from types import SimpleNamespace

import httpx
import pytest

from sglang.srt.disaggregation.dualpd.router import (
    global_pd_router as root,
    tcp_capacity as capacity,
    tcp_http as http,
)

ENGINE = Path(__file__).resolve().parents[7]
ROUTER = ENGINE / "python/sglang/srt/disaggregation/dualpd/router"
spec = importlib.util.spec_from_file_location("tcp_structure_verifier", ENGINE / "tools/verify_dualpd_router_cleanup.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def tcp_router(monkeypatch):
    endpoints = {"http://p:1":"p", "http://d:2":"d"}
    monkeypatch.setenv("DUALPD_ROUTER_ENDPOINT_GROUPS", json.dumps(endpoints))
    return root.GlobalEndpointMiniLoadBalancer(SimpleNamespace(
        prefill_urls=[("http://p:1",11)], decode_urls=["http://d:2"],
        request_timeout_secs=10, host="127.0.0.1", port=8000, policy="random",
        pd_disaggregation=True, otlp_traces_endpoint=None, enable_trace=False))


@pytest.mark.parametrize("stream", [False, True])
def test_backend_failure_or_stream_close_settles_tasks_and_all_charges(tcp_router, monkeypatch, stream):
    router = tcp_router
    closed = []
    cancelled = []
    async def ready(request):
        return {"required_tokens":3}
    monkeypatch.setattr(router, "_wait_prefill_ready", ready)
    class Content:
        async def iter_chunked(self, size):
            yield b"one chunk"
            await asyncio.Event().wait()
    class Session:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            closed.append(True)
        async def post(self, url, **kwargs):
            if url.startswith("http://p:1"):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append("prefill")
            if not stream:
                raise RuntimeError("decode backend failed")
            return SimpleNamespace(content=Content(),status=200)
    monkeypatch.setattr(root.aiohttp, "ClientSession", Session)
    async def exercise():
        purl, _, durl = router.select_pair()
        request = {"input_ids":[1,2],"bootstrap_room":77}
        if stream:
            response = await router.generate_stream(request,purl,durl)
            assert await anext(response.body_iterator) == b"one chunk"
            await response.body_iterator.aclose()
        else:
            with pytest.raises(RuntimeError,match="decode backend failed"):
                await router.generate(request,purl,durl,"generate")
        assert cancelled == ["prefill"] and closed == [True]
        assert not any(router._inflight.values())
        assert not any(router._prefill_reserved_tokens.values())
        assert not any(router._pending_prefill_reservations.values())
        assert not any(router._decode_reserved_tokens.values())
        assert not router._decode_reservation_index
    asyncio.run(exercise())


def test_close_cancels_poll_and_live_generations_on_same_instance(tcp_router):
    router = tcp_router
    async def exercise():
        async def pending():
            await asyncio.Event().wait()
        poll = asyncio.create_task(pending())
        generation = asyncio.create_task(pending())
        router._poll_task = poll
        router._generation_tasks[("request",1)] = generation
        router._generation_completed_at[("older",0)] = 0
        await asyncio.sleep(0)
        await router.close()
        assert poll.cancelled() and generation.cancelled()
        assert router._poll_task is None and not router._generation_tasks
        assert not router._generation_completed_at
    asyncio.run(exercise())


def test_no_capacity_timeout_never_grants_or_charges_target(tcp_router):
    router = tcp_router
    router._load["http://d:2"] = (100,100,0)
    router.timeout = 0
    async def exercise():
        with pytest.raises(TimeoutError,match="complete workset"):
            await router._select_decode(1)
        assert not any(router._inflight.values())
        assert not any(router._decode_reserved_tokens.values())
        assert not router._decode_reservation_index
    asyncio.run(exercise())


def test_ready_waiter_cancellation_removes_only_its_own_waiter(tcp_router):
    router = tcp_router
    async def exercise():
        task = asyncio.create_task(router._wait_prefill_ready({"bootstrap_room":88}))
        await asyncio.sleep(0)
        assert set(router._ready_waiters) == {"88"}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not router._ready_waiters and not router._early_ready
    asyncio.run(exercise())


def test_native_asgi_callbacks_reach_split_methods_with_causal_release(tcp_router, monkeypatch):
    router = tcp_router
    monkeypatch.setattr(root.mini_lb_module,"lb",router)
    async def exercise():
        transport = httpx.ASGITransport(app=root.mini_lb_module.app)
        async with httpx.AsyncClient(transport=transport,base_url="http://router") as client:
            response = await client.post("/dualpd/prefill_ready",json={})
            assert response.status_code == 400
            response = await client.post("/dualpd/prefill_ready",json={"bootstrap_room":99,"required_tokens":3})
            assert response.status_code == 200 and response.json() == {"ok":True}
            assert (await router._wait_prefill_ready({"bootstrap_room":99}))["required_tokens"] == 3
            selected = await router._select_decode(3,"reservation")
            router._load_started_epoch[selected] = 4
            response = await client.post("/dualpd/decode_materialized",json={"reservation_id":"reservation","target_group":"wrong"})
            assert response.status_code == 400
            response = await client.post("/dualpd/decode_materialized",json={"reservation_id":"reservation","target_group":"d"})
            assert response.status_code == 200
            router._load_epoch[selected] = 4
            router._release_sampled_decode_reservations(selected)
            assert router._decode_reserved_tokens[selected] == 3
            router._load_epoch[selected] = 5
            router._release_sampled_decode_reservations(selected)
            assert router._decode_reserved_tokens[selected] == 0
            response = await client.post("/dualpd/decode_materialized",json={"reservation_id":"reservation","target_group":"d"})
            assert response.status_code == 200
    asyncio.run(exercise())


def test_concrete_constructor_and_all_method_owners_are_preserved():
    concrete = root.GlobalEndpointMiniLoadBalancer
    assert concrete.__bases__ == (capacity.TCPCapacityMixin, http.TCPHTTPMixin,
                                  root.mini_lb_module.MiniLoadBalancer)
    names = set()
    for cls, expected in [(concrete, 8), (capacity.TCPCapacityMixin, 12), (http.TCPHTTPMixin, 9)]:
        methods = {name: value for name, value in vars(cls).items()
                   if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod))}
        assert len(methods) == expected
        if cls is not concrete:
            assert "__init__" not in methods
        assert not names & set(methods)
        names.update(methods)
        for name, value in methods.items():
            assert inspect.getattr_static(concrete, name) is value
            typing.get_type_hints(value.__func__ if isinstance(value, (staticmethod, classmethod)) else value)
    assert len(names) == 29


def test_tcp_record_alias_old_pickle_fields_and_annotations():
    assert root._DecodeReservation is capacity._DecodeReservation
    value = root._DecodeReservation("reservation", 1234, 56.7, 9)
    # The unchanged slotted dataclass requires protocol>=2 for instances.
    with pytest.raises(TypeError):
        pickle.dumps(value, protocol=0)
    legacy_class = pickle.dumps(root._DecodeReservation, protocol=0).replace(
        b"sglang.srt.disaggregation.dualpd.router.tcp_capacity\n",
        b"sglang.srt.disaggregation.dualpd.router.global_pd_router\n")
    assert pickle.loads(legacy_class) is root._DecodeReservation
    legacy = pickle.dumps(value, protocol=2).replace(
        b"sglang.srt.disaggregation.dualpd.router.tcp_capacity\n",
        b"sglang.srt.disaggregation.dualpd.router.global_pd_router\n")
    restored = pickle.loads(legacy)
    assert type(restored) is root._DecodeReservation and restored == value
    assert typing.get_type_hints(root._DecodeReservation) == {
        "reservation_id": str, "tokens": int, "expires_at": float,
        "materialized_after_started_epoch": int | None,
    }


def test_complete_tcp_module_reconstructs_callbacks_main_and_all_bodies():
    result = verifier.verify_tcp()
    assert result["methods_ast_identical"] == 29
    assert result["record_ast_identical"]
    assert result["callbacks_main_and_service_guard_ast_identical"]
    assert result["complete_original_module_reconstructed"]
    assert result["existing_tests_unchanged"]


def test_helpers_are_lazy_and_module_service_registers_callbacks_once(tmp_path):
    code = '''
import asyncio,importlib,json,os,runpy,sys
from types import SimpleNamespace
import aiohttp,uvicorn
import sglang_router.mini_lb as mini
import sglang_router.launch_router as launcher
target = "sglang.srt.disaggregation.dualpd.router.global_pd_router"
assert target not in sys.modules
paths = ("/dualpd/prefill_ready", "/dualpd/decode_materialized", "/dualpd/application_final")
def counts():
 return {path:sum(route.path == path for route in mini.app.routes) for path in paths}
before = counts()
def forbidden(*args, **kwargs):
 raise AssertionError("component import/main started a task or HTTP session")
asyncio.create_task = forbidden
aiohttp.ClientSession = forbidden
capacity = importlib.import_module("sglang.srt.disaggregation.dualpd.router.tcp_capacity")
http = importlib.import_module("sglang.srt.disaggregation.dualpd.router.tcp_http")
assert counts() == before and target not in sys.modules
args = SimpleNamespace(prefill_urls=[("http://p:1",11)],decode_urls=["http://d:2"],
 request_timeout_secs=10,host="127.0.0.1",port=8000,policy="random",
 pd_disaggregation=True,otlp_traces_endpoint=None,enable_trace=False)
os.environ["DUALPD_ROUTER_ENDPOINT_GROUPS"] = json.dumps({"http://p:1":"p","http://d:2":"d"})
launcher.parse_router_args = lambda argv: args
observed = []
uvicorn.run = lambda app, **kwargs: observed.append((app,kwargs))
module = runpy.run_module(target,run_name="__main__",alter_sys=True)
assert target not in sys.modules, "helper imported canonical root back during -m startup"
assert counts() == {path:value+1 for path,value in before.items()}
assert len(observed) == 1 and observed[0][0] is mini.app
assert observed[0][1] == {"host":"127.0.0.1","port":8000,"loop":"asyncio"}
assert mini.lb.__class__ is module["GlobalEndpointMiniLoadBalancer"]
assert module["_DecodeReservation"] is capacity._DecodeReservation
assert mini.lb._poll_task is None and not mini.lb._generation_tasks
print(json.dumps({"callback_registrations_once":True,"no_import_tasks":True}))
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env=dict(os.environ), capture_output=True, text=True,
                            check=True, timeout=90)
    assert json.loads(result.stdout.splitlines()[-1])["callback_registrations_once"]


@pytest.mark.parametrize("mutation", ["body", "missing", "extra", "record", "callback",
                                      "guard", "mro", "missing_import", "wrong_import",
                                      "back_import", "constructor", "old_assertion"])
def test_tcp_verifier_rejects_body_import_callback_and_gate_mutations(mutation):
    filename = {
        "body":"tcp_capacity.py", "missing":"tcp_capacity.py", "extra":"tcp_http.py",
        "record":"tcp_capacity.py", "callback":"global_pd_router.py", "guard":"global_pd_router.py",
        "mro":"global_pd_router.py", "missing_import":"tcp_capacity.py", "wrong_import":"tcp_http.py",
        "back_import":"tcp_http.py", "constructor":"tcp_http.py", "old_assertion":"tests/test_global_pd_router.py",
    }[mutation]
    source = (ROUTER / filename).read_text()
    if mutation == "body":
        source = source.replace("capacity - projected >= required_tokens", "capacity - projected < required_tokens")
    elif mutation == "missing":
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TCPCapacityMixin")
        cls.body.remove(next(node for node in cls.body if isinstance(node, ast.FunctionDef)))
        source = ast.unparse(tree)
    elif mutation == "extra":
        source += "\n    def unexpected(self):\n        return None\n"
    elif mutation == "record":
        source = source.replace("expires_at: float", "expires_at: int")
    elif mutation == "callback":
        source = source.replace('"/dualpd/prefill_ready"', '"/dualpd/wrong_ready"')
    elif mutation == "guard":
        source = source.replace('if __name__ == "__main__":', 'if __name__ == "wrong_entry":')
    elif mutation == "mro":
        source = source.replace("TCPCapacityMixin, TCPHTTPMixin, mini_lb_module.MiniLoadBalancer",
                                "TCPHTTPMixin, TCPCapacityMixin, mini_lb_module.MiniLoadBalancer")
    elif mutation == "missing_import":
        source = source.replace("import time\n", "")
    elif mutation == "wrong_import":
        source = source.replace("import copy\n", "import json as copy\n")
    elif mutation == "back_import":
        source += "\nfrom .global_pd_router import GlobalEndpointMiniLoadBalancer\n"
    elif mutation == "constructor":
        source += "\n    def __init__(self):\n        pass\n"
    elif mutation == "old_assertion":
        source = source.replace("assert first != second", "assert first == second")
    assert source != (ROUTER / filename).read_text(), "Mutation fixture did not change anything"
    with pytest.raises(AssertionError):
        verifier.verify_tcp({filename:source})
