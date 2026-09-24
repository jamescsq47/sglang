from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.agentic_group_protocol import (
    GenerationKey,
    LinkCapacityEdge,
    LinkParticipant,
    Owner,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    CallbackPathHandler,
    GroupTransferPlan,
    PreparedRankTransfer,
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_multinode_composite import (
    RequestPhase,
    RuntimeFactoryOptions,
    create_runtime,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    FenceKind,
    PhysicalProgress,
    PhysicalState,
    TransferPath,
)


class FakeAllocator:
    page_size = 64

    def __init__(self, size=4096):
        self.free_values = list(range(size))
        self.live = set()

    def available_size(self):
        return len(self.free_values)

    def alloc(self, count):
        if count > len(self.free_values):
            return None
        values = self.free_values[:count]
        del self.free_values[:count]
        self.live.update(values)
        return torch.tensor(values, dtype=torch.int64)

    def free(self, indices):
        values = [int(value) for value in indices.tolist()]
        assert set(values).issubset(self.live)
        self.live.difference_update(values)
        self.free_values.extend(values)


class ImmediateExecutor:
    def submit(self, attempt, _notify):
        return attempt

    def progress(self, _handle):
        return PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE)

    def request_cancel(self, _handle, notify):
        notify()


class FakeLowLevel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.plans = []
        self.started = False
        self.closed = False
        self.capacity_edges = []
        self.registered = []
        self.source_ready = []

    def start(self):
        self.started = True

    def submit(self, plan):
        self.plans.append(plan)
        return len(self.plans)

    def check_health(self):
        assert self.started and not self.closed

    def notify_capacity_available(self, available_tokens):
        self.capacity_edges.append(int(available_tokens))
        return len(self.capacity_edges)

    def report_registered(self, key):
        self.registered.append(key)

    def report_source_ready(self, key):
        self.source_ready.append(key)

    def take_activation_ticket(self, timeout=None):
        raise queue.Empty

    def activate_staged(self, _ticket):
        pass

    def confirm_scheduler_adopted(self, _ticket):
        pass

    def close(self):
        self.kwargs["queues"].close()
        self.closed = True


class FakeProvider:
    def __init__(self, role):
        self.role = role
        self.closed = False
        self.requests = []
        self.local_completions = []
        self.capacity_edges = []

    def state_allocators(self, _scheduler):
        return ()

    def default_state_slot_counts(self, _scheduler):
        return ()

    def executors(self, _context):
        return {path: ImmediateExecutor() for path in TransferPath}

    def handlers(self, _context):
        handler = CallbackPathHandler(
            lambda _command: PreparedRankTransfer(None, requires_io=False)
        )
        return {path: handler for path in TransferPath}

    def lanes(self, _context):
        return {path: 1 for path in TransferPath}

    def pending_capacity(self, _context):
        return {path: 8 for path in TransferPath}

    def on_request(self, context, record):
        self.requests.append(record.key)
        key = SimpleNamespace(
            request_id=record.key.request_id,
            generation=record.key.generation,
            attempt=1,
        )
        # The bridges require the typed authority identity.
        from sglang.srt.disaggregation.agentic_memory_authority import (
            RequestGenerationAttempt,
        )

        key = RequestGenerationAttempt(key.request_id, key.generation, key.attempt)
        if self.role == "prefill":
            bridge = context.p_memory_bridge
            lease = bridge.reserve_workset(
                key,
                owner="initial",
                parent_tokens=0,
                prompt_tokens=128,
            )
            assert bridge.begin_ingress(lease, "local")
            assert bridge.complete_ingress(lease, "local", success=True)
            bridge.bind_and_publish(
                lease,
                record.req,
                bind_parent=lambda *_: "p-binding",
                release_bound=lambda bound, *_: context.authority.token_allocator.free(
                    bound.device_indices
                ),
            )
        else:
            bridge = context.d_memory_bridge
            lease = bridge.reserve_decode(
                key,
                owner="p2d",
                prompt_tokens=128,
                decode_growth_tokens=128,
            )
            assert bridge.begin_ingress(lease, "local")
            assert bridge.complete_ingress(lease, "local", success=True)
            bridge.bind_and_publish(
                lease,
                record.req,
                bind_prompt=lambda *_: "d-binding",
                release_bound=lambda bound, *_: context.authority.token_allocator.free(
                    bound.device_indices
                ),
            )

    @staticmethod
    def _plan(record, item, path, source, target):
        return GroupTransferPlan(
            key=record.key,
            path=path,
            operation=TransferOperation.DIRECT,
            source_owner=source,
            target_owner=target,
            lease_id=str(item.lease.lease_id),
            payload={"prompt_tokens": item.lease.prompt_tokens},
        )

    def plan_prefill_complete(self, _context, record, item):
        self.local_completions.append((record.key, item.lease.lease_id))
        return self._plan(
            record, item, TransferPath.P2D_DIRECT, Owner.PREFILL_READY, Owner.D_GPU
        )

    def plan_decode_complete(self, _context, record, item):
        self.local_completions.append((record.key, item.lease.lease_id))
        return self._plan(
            record, item, TransferPath.D2P_DIRECT, Owner.D_GPU, Owner.PREFILL_READY
        )

    def decide_intent(self, _context, _intent, candidate):
        return candidate

    def on_committed(self, _context, _plan, _attempt):
        pass

    def on_aborted(self, _context, _plan, _attempt, _reason):
        pass

    def memory_available(self, *, remote_role):
        self.capacity_edges.append(remote_role)

    def close(self):
        self.closed = True


def config(role, *, tp_size=1):
    peer = "decode" if role == "prefill" else "prefill"
    return SimpleNamespace(
        run_id="run",
        role=role,
        endpoint_group="p" if role == "prefill" else "d",
        peer_role=peer,
        peer_group="d" if role == "prefill" else "p",
        coordinator_group="p",
        tp_size=tp_size,
        endpoint=("127.0.0.1", 12345),
        group_id="p--d",
        control_token="0123456789abcdef",
    )


def request(name="r"):
    return SimpleNamespace(
        rid=name,
        sampling_params=SimpleNamespace(
            custom_params={
                "agentic_request_id": name,
                "agentic_generation": 0,
            }
        ),
    )


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition did not become true")


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_scheduler_to_compute_complete_handoff_is_event_driven(role):
    allocator = FakeAllocator()
    scheduler = SimpleNamespace(token_to_kv_pool_allocator=allocator, tp_rank=0)
    provider = FakeProvider(role)
    holder = {}

    def low_level_factory(**kwargs):
        holder["runtime"] = FakeLowLevel(**kwargs)
        return holder["runtime"]

    runtime = create_runtime(
        scheduler,
        config(role),
        physical_provider=provider,
        options=RuntimeFactoryOptions(low_level_factory=low_level_factory),
    )
    req = request(role)
    try:
        runtime.submit_request(req)
        wait_until(lambda: bool(provider.requests))
        if role == "prefill":
            assert runtime.p_memory_bridge.drain_prefill_ready() == (req,)
            assert runtime.p_memory_bridge.publish_prefill_complete(req) is not None
            # Sink delivery means the scheduler-facing completion queue stays empty.
            assert runtime.p_memory_bridge.drain_prefill_complete() == ()
        else:
            assert runtime.d_memory_bridge.drain_decode_ready() == (req,)
            assert runtime.d_memory_bridge.publish_decode_complete(req) is not None
            assert runtime.d_memory_bridge.drain_decode_complete() == ()
        wait_until(lambda: len(holder["runtime"].plans) == 1)
        key = GenerationKey("run", role, 0)
        assert holder["runtime"].plans[0].key == key
        assert runtime.registry.get(key).phase is RequestPhase.TRANSFER_SUBMITTED
        runtime.progress_nonblocking()
    finally:
        runtime.close()
    assert holder["runtime"].closed
    assert provider.closed


def test_factory_uses_builtin_physical_provider_when_not_installed(monkeypatch):
    from sglang.srt.disaggregation import agentic_multinode_composite as module
    from sglang.srt.disaggregation import agentic_default_physical_provider as physical

    monkeypatch.setattr(module, "_provider_factory", None)
    called = []

    def create(scheduler, runtime_config):
        called.append((scheduler, runtime_config))
        raise RuntimeError("builtin provider selected")

    monkeypatch.setattr(physical, "create_default_physical_provider", create)
    scheduler = SimpleNamespace(token_to_kv_pool_allocator=FakeAllocator(), tp_rank=0)
    with pytest.raises(RuntimeError, match="builtin provider selected"):
        create_runtime(scheduler, config("prefill"))
    assert called and called[0][0] is scheduler


def test_registry_retires_req_only_after_source_fence():
    from sglang.srt.disaggregation.agentic_multinode_composite import (
        RequestGenerationRegistry,
    )

    registry = RequestGenerationRegistry("run")
    req = request("retire")
    record = registry.register(req)
    with pytest.raises(RuntimeError, match="before source fence"):
        registry.retire_after_source_fence(record.key)
    registry.mark_compute_complete(record.key)
    assert registry.retire_after_source_fence(record.key)
    assert registry.get(record.key) is None
    assert not registry.retire_after_source_fence(record.key)


def test_remote_decode_capacity_edges_are_coalesced_off_control_thread():
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=FakeAllocator(), tp_rank=0
    )
    provider = FakeProvider("prefill")
    holder = {}

    def low_level_factory(**kwargs):
        holder["runtime"] = FakeLowLevel(**kwargs)
        return holder["runtime"]

    runtime = create_runtime(
        scheduler,
        config("prefill"),
        physical_provider=provider,
        options=RuntimeFactoryOptions(low_level_factory=low_level_factory),
    )
    edge = LinkCapacityEdge(
        "p0--d0",
        LinkParticipant("decode", "d0", 0),
        "session",
        1,
        4096,
    )
    try:
        callback = holder["runtime"].kwargs["on_capacity_edge"]
        callback(edge)
        callback(edge)
        wait_until(lambda: provider.capacity_edges == ["decode"])
    finally:
        runtime.close()


def test_tp_follower_records_completion_but_never_submits_a_plan():
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=FakeAllocator(), tp_rank=1
    )
    provider = FakeProvider("prefill")
    holder = {}

    def low_level_factory(**kwargs):
        holder["runtime"] = FakeLowLevel(**kwargs)
        return holder["runtime"]

    runtime = create_runtime(
        scheduler,
        config("prefill", tp_size=2),
        physical_provider=provider,
        options=RuntimeFactoryOptions(low_level_factory=low_level_factory),
    )
    req = request("follower")
    try:
        runtime.submit_request(req)
        wait_until(lambda: bool(provider.requests))
        assert runtime.p_memory_bridge.drain_prefill_ready() == (req,)
        assert runtime.p_memory_bridge.publish_prefill_complete(req) is not None
        key = GenerationKey("run", "follower", 0)
        wait_until(
            lambda: runtime.registry.get(key).phase
            is RequestPhase.COMPUTE_COMPLETE
        )
        assert provider.local_completions == [(key, 1)]
        assert holder["runtime"].plans == []
    finally:
        runtime.close()
