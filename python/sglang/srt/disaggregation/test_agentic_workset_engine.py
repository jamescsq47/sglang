"""Startup selection must not change legacy/D behavior or import a live pool."""
from types import SimpleNamespace as NS
import pytest

from sglang.srt.disaggregation.agentic_workset_engine import (
    controller_requested, validate_controller_configuration,
)


def config(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://localhost:1234")
    monkeypatch.setenv("SGLANG_PD_P_READY_DIR", "/run/test-identity-only")
    monkeypatch.setenv("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "disabled")
    return NS(server_args=NS(disaggregation_mode="prefill"),
              req_to_token_pool=NS(), waiting_queue=[], agentic_kv_waiting_queue=[])


def test_default_and_decode_are_unchanged(monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_P_WORKSET_CONTROLLER", raising=False)
    assert not controller_requested(NS(disaggregation_mode="prefill"))
    monkeypatch.setenv("SGLANG_AGENTIC_P_WORKSET_CONTROLLER", "1")
    assert controller_requested(NS(disaggregation_mode="prefill"))
    assert not controller_requested(NS(disaggregation_mode="decode"))
    assert not controller_requested(NS(disaggregation_mode="null"))


def test_invalid_switch_not_silently_ignored(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_P_WORKSET_CONTROLLER", "maybe")
    with pytest.raises(ValueError):
        controller_requested(NS(disaggregation_mode="prefill"))


def test_empty_supported_configuration(monkeypatch):
    validate_controller_configuration(config(monkeypatch))


@pytest.mark.parametrize("name", ["enable_hierarchical_cache", "enable_hicache_storage",
    "is_hybrid_swa", "enable_priority_preemption", "is_mixed_chunk", "enable_lora"])
def test_unconverted_paths_rejected(monkeypatch, name):
    scheduler = config(monkeypatch)
    setattr(scheduler, name, True)
    with pytest.raises(ValueError, match="unconverted"):
        validate_controller_configuration(scheduler)


@pytest.mark.parametrize("name", ["pp_size", "dp_size", "attn_cp_size"])
def test_unsupported_parallelism(monkeypatch, name):
    scheduler = config(monkeypatch)
    setattr(scheduler.server_args, name, 2)
    with pytest.raises(ValueError, match="unconverted"):
        validate_controller_configuration(scheduler)


@pytest.mark.parametrize("name", ["waiting_queue", "agentic_kv_waiting_queue"])
def test_runtime_replacement_forbidden(monkeypatch, name):
    scheduler = config(monkeypatch)
    setattr(scheduler, name, [object()])
    with pytest.raises(ValueError, match="before request ingress"):
        validate_controller_configuration(scheduler)


def test_missing_socket_and_legacy_watermark(monkeypatch):
    scheduler = config(monkeypatch)
    monkeypatch.delenv("SGLANG_AGENTIC_CONTROL_ENDPOINT")
    with pytest.raises(ValueError, match="socket"):
        validate_controller_configuration(scheduler)
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://localhost:1234")
    monkeypatch.setenv("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "continuous")
    with pytest.raises(ValueError, match="throttling"):
        validate_controller_configuration(scheduler)


def test_streaming_prebind_without_deferred_p_ready_is_rejected(monkeypatch):
    scheduler = config(monkeypatch)
    monkeypatch.delenv("SGLANG_PD_P_READY_DIR")
    with pytest.raises(ValueError, match="deferred full-snapshot"):
        validate_controller_configuration(scheduler)


def test_hybrid_requires_tracking_and_owned_checkpoints(monkeypatch):
    scheduler = config(monkeypatch)
    scheduler.req_to_token_pool.mamba_pool = object()
    with pytest.raises(ValueError, match="tracking"):
        validate_controller_configuration(scheduler)
    scheduler.req_to_token_pool.enable_mamba_extra_buffer = True
    with pytest.raises(ValueError, match="request-owned"):
        validate_controller_configuration(scheduler)
    scheduler._agentic_mamba_prefill_admission = True
    validate_controller_configuration(scheduler)


def factory(monkeypatch, *, rank=0, device="cuda", fail=None, fail_shutdown=None):
    """Run the real factory with inert CPU-only boundary implementations."""
    import threading
    import torch
    import sglang.srt.disaggregation.agentic_tp_events as events
    import sglang.srt.disaggregation.agentic_workset_controller as controller_module
    import sglang.srt.disaggregation.agentic_workset_native as native
    import sglang.srt.disaggregation.agentic_workset_runtime as runtime_module
    import sglang.srt.mem_cache.allocator as allocators

    calls, instances = [], {}
    owner = threading.current_thread()
    def step(name):
        calls.append(name)
        if fail == name or fail_shutdown == name:
            raise RuntimeError("injected " + name)

    def current_device():
        assert threading.current_thread() is owner
        step("current_device")
        return 3
    def current_stream(target):
        assert threading.current_thread() is owner
        assert target.index is not None
        instances["stream_device"] = target
        return NS(synchronize=lambda: step("init_sync"))
    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)

    class Allocator:
        size, page_size = 64, 4
        def __init__(self):
            self.device, self.adapter = device, None
        def available_size(self):
            return self.size
        def install_workset_adapter(self, adapter):
            self.adapter = adapter
            step("attention_install")
    class Mamba:
        size = 8
        adapter = None
        def available_size(self):
            return self.size
        def install_workset_adapter(self, adapter):
            self.adapter = adapter
            step("mamba_install")
    class Client:
        def __init__(self, address, **kwargs):
            step("client")
            self.address, self.kwargs = address, kwargs
            instances["client"] = self
        def wait_ready(self):
            step("wait_ready")
        def close(self):
            step("client_close")
    class Controller:
        def __init__(self, ledger):
            step("controller")
            self.ledger = ledger
            instances["controller"] = self
        def shutdown(self):
            step("controller_shutdown")
    class Runtime:
        def __init__(self, client, broker, **kwargs):
            step("runtime")
            self.kwargs = kwargs
            instances["runtime"] = self
        def counts(self):
            return None
        def native_free(self, value):
            return None
        def shutdown(self):
            step("runtime_shutdown")
        def check_health(self):
            step("runtime_health")
    class Bridge:
        def __init__(self, **kwargs):
            step("bridge")
            self.kwargs = kwargs
            instances["bridge"] = self
        def shutdown(self, **kwargs):
            step("bridge_shutdown")
        def check_health(self):
            step("bridge_health")
    monkeypatch.setattr(events, "TPEventClient", Client)
    monkeypatch.setattr(controller_module, "WorksetController", Controller)
    monkeypatch.setattr(runtime_module, "PWorksetRuntime", Runtime)
    monkeypatch.setattr(native, "NativeLastRefFreeAdapter", Bridge)
    monkeypatch.setattr(allocators, "PagedTokenToKVPoolAllocator", Allocator)

    scheduler = config(monkeypatch)
    scheduler.tp_rank, scheduler.tp_size = rank, 8
    scheduler.token_to_kv_pool_allocator = Allocator()
    scheduler.req_to_token_pool.mamba_pool = Mamba()
    scheduler.req_to_token_pool.enable_mamba_extra_buffer = True
    scheduler._agentic_mamba_prefill_admission = True
    for name, value in {"SGLANG_AGENTIC_TP_EVENT_ENDPOINT": "tcp://localhost:2345",
            "SGLANG_AGENTIC_CONTROL_RUN_ID": "run", "SGLANG_AGENTIC_CONTROL_TOKEN": "secret",
            "SGLANG_AGENTIC_CONTROL_GROUP_ID": "P"}.items():
        monkeypatch.setenv(name, value)
    def attach(runtime, *, native_bridge):
        instances["attached"] = (runtime, native_bridge)
        step("attach")
    broker = NS(reference_fence=object(), attach_runtime=attach)
    return NS(scheduler=scheduler, broker=broker, calls=calls, instances=instances)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("device,expected", [("cuda", "cuda:3"), ("cuda:2", "cuda:2"), ("cpu", "cpu")])
def test_real_factory_freezes_main_thread_device_and_single_authority(monkeypatch, rank, device, expected):
    from sglang.srt.disaggregation.agentic_workset_engine import install_controller
    f = factory(monkeypatch, rank=rank, device=device)
    engine = install_controller(f.scheduler, f.broker)
    assert str(engine.runtime.kwargs["device"]) == str(f.broker.device) == expected
    assert ("current_device" in f.calls) is (device == "cuda")
    assert ("init_sync" in f.calls) is device.startswith("cuda")
    if device.startswith("cuda"):
        assert f.calls.index("init_sync") < f.calls.index("runtime")
    assert engine.runtime.kwargs["dedicated_client"] is True
    assert (engine.controller is not None) is (rank == 0)
    assert engine.client.kwargs == dict(run_id="run", token="secret", group="P:workset", rank=rank, size=8)
    assert engine.runtime.kwargs["page_capacity"] == 16
    assert engine.runtime.kwargs["mamba_slots"] == 8
    assert f.scheduler.token_to_kv_pool_allocator.adapter is engine.native_bridge
    assert f.scheduler.req_to_token_pool.mamba_pool.adapter is engine.native_bridge
    assert f.calls.index("attach") < f.calls.index("attention_install") < f.calls.index("mamba_install")
    engine.check_health()
    assert f.calls[-2:] == ["runtime_health", "bridge_health"]


@pytest.mark.parametrize("failure", ["wait_ready", "runtime", "bridge", "attach", "attention_install", "mamba_install"])
def test_partial_factory_failure_closes_controls_without_restoring_native_pool(monkeypatch, failure):
    from sglang.srt.disaggregation.agentic_workset_engine import install_controller
    f = factory(monkeypatch, fail=failure)
    with pytest.raises(RuntimeError, match="injected " + failure):
        install_controller(f.scheduler, f.broker)
    for resource in ("runtime", "controller", "bridge"):
        if resource in f.instances:
            assert resource + "_shutdown" in f.calls
    assert "client_close" in f.calls
    if failure in {"attention_install", "mamba_install"}:
        # No reset/free-list resurrection after even partial installation.
        assert f.scheduler.token_to_kv_pool_allocator.adapter is f.instances["bridge"]


@pytest.mark.parametrize("missing", ["SGLANG_AGENTIC_TP_EVENT_ENDPOINT", "SGLANG_AGENTIC_CONTROL_RUN_ID",
    "SGLANG_AGENTIC_CONTROL_TOKEN", "SGLANG_AGENTIC_CONTROL_GROUP_ID"])
def test_factory_rejects_missing_identity_before_any_control_or_cuda_call(monkeypatch, missing):
    from sglang.srt.disaggregation.agentic_workset_engine import install_controller
    f = factory(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="endpoint/run/token/group"):
        install_controller(f.scheduler, f.broker)
    assert not f.calls


@pytest.mark.parametrize("resource", ["runtime", "controller", "bridge"])
def test_factory_cleanup_error_does_not_mask_install_error_or_skip_close(monkeypatch, resource):
    from sglang.srt.disaggregation.agentic_workset_engine import install_controller
    f = factory(monkeypatch, fail="mamba_install", fail_shutdown=resource + "_shutdown")
    with pytest.raises(RuntimeError, match="injected mamba_install"):
        install_controller(f.scheduler, f.broker)
    assert all(name in f.calls for name in ("runtime_shutdown", "bridge_shutdown", "controller_shutdown", "client_close"))
    assert f.scheduler.token_to_kv_pool_allocator.adapter is f.instances["bridge"]


@pytest.mark.parametrize("invalid", ["wrapped", "attention_live", "mamba_live"])
def test_factory_rejects_second_or_live_address_authority(monkeypatch, invalid):
    from sglang.srt.disaggregation.agentic_workset_engine import install_controller
    f = factory(monkeypatch)
    allocator = f.scheduler.token_to_kv_pool_allocator
    if invalid == "wrapped":
        class Wrapped(type(allocator)):
            pass
        f.scheduler.token_to_kv_pool_allocator = Wrapped()
    elif invalid == "attention_live":
        allocator.available_size = lambda: allocator.size - allocator.page_size
    else:
        pool = f.scheduler.req_to_token_pool.mamba_pool
        pool.available_size = lambda: pool.size - 1
    with pytest.raises(ValueError):
        install_controller(f.scheduler, f.broker)
    assert not f.calls
