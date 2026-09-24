import threading
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.agentic_memory_authority import (
    AgenticMemoryAuthority,
    RequestGenerationAttempt,
)
from sglang.srt.disaggregation.agentic_memory_scheduler_bridge import (
    AgenticPMemorySchedulerBridge,
    memory_authority_v2_enabled,
)


class FakeAllocator:
    def __init__(self, size=1024, page_size=64):
        self.page_size = page_size
        self.free_values = list(range(size))
        self.live = set()
        self.alloc_calls = 0

    def available_size(self):
        return len(self.free_values)

    def alloc(self, count):
        self.alloc_calls += 1
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


def test_v2_flag_is_default_off(monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_MEMORY_AUTHORITY_V2", raising=False)
    assert not memory_authority_v2_enabled()


def test_controller_binds_then_scheduler_only_adopts_ready_lease():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    bridge = AgenticPMemorySchedulerBridge(authority)
    lease = bridge.reserve_workset(
        RequestGenerationAttempt("r", 1, 1),
        owner="direct",
        parent_tokens=64,
        prompt_tokens=128,
    )
    assert allocator.alloc_calls == 1
    assert bridge.begin_ingress(lease, "dma")
    assert bridge.complete_ingress(lease, "dma", success=True)

    req = SimpleNamespace(rid="child")
    calls = []

    def bind_parent(bound_lease, bound_req):
        authority.assert_native_guard()
        calls.append(("bind", bound_lease.lease_id, bound_req.rid))
        return {"radix_node": 7}

    def release_bound(bound_lease, bound_req, binding):
        calls.append(("release", bound_lease.lease_id, binding["radix_node"]))
        allocator.free(bound_lease.device_indices)

    first_event = bridge.bind_and_publish(
        lease,
        req,
        bind_parent=bind_parent,
        release_bound=release_bound,
    )
    replay_event = bridge.bind_and_publish(
        lease,
        req,
        bind_parent=bind_parent,
        release_bound=release_bound,
    )
    assert replay_event is first_event
    assert calls == [("bind", lease.lease_id, "child")]

    ready = bridge.take_prefill_ready()
    assert len(ready) == 1
    before_adopt_allocs = allocator.alloc_calls
    assert bridge.adopt_for_prefill(ready[0])
    assert allocator.alloc_calls == before_adopt_allocs
    assert req._agentic_workset_backed

    consumed = bridge.consume_suffix(
        lease, 64, final_prompt_chunk=True
    )
    assert len(consumed) == 64
    assert len(bridge.remaining_suffix_indices(lease)) == 0
    assert bridge.finish_prefill(req)
    assert bridge.release_after_group_fence(
        lease.lease_id, reason="p2d_complete"
    )
    assert calls[-1] == ("release", lease.lease_id, 7)
    assert allocator.available_size() == 1024


def test_scheduler_drain_only_consumes_committed_ready_requests():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    bridge = AgenticPMemorySchedulerBridge(authority)
    lease = bridge.reserve_workset(
        RequestGenerationAttempt("r", 7, 1),
        owner="slow",
        parent_tokens=64,
        prompt_tokens=128,
    )
    req = SimpleNamespace(rid="ready")
    assert bridge.begin_ingress(lease, "h2d")
    assert bridge.drain_prefill_ready() == ()
    assert bridge.complete_ingress(lease, "h2d", success=True)
    bridge.bind_and_publish(
        lease,
        req,
        bind_parent=lambda *_: "bound",
        release_bound=lambda bound_lease, *_: allocator.free(
            bound_lease.device_indices
        ),
    )

    before = allocator.alloc_calls
    assert bridge.drain_prefill_ready() == (req,)
    assert allocator.alloc_calls == before
    assert bridge.drain_prefill_ready() == ()


def test_final_prefill_chunk_consumes_page_padding_ownership():
    allocator = FakeAllocator(page_size=64)
    authority = AgenticMemoryAuthority(allocator)
    bridge = AgenticPMemorySchedulerBridge(authority)
    lease = bridge.reserve_workset(
        RequestGenerationAttempt("r", 8, 1),
        owner="direct",
        parent_tokens=64,
        prompt_tokens=100,
    )
    req = SimpleNamespace(rid="partial-page")
    assert bridge.begin_ingress(lease, "dma")
    assert bridge.complete_ingress(lease, "dma", success=True)
    bridge.bind_and_publish(
        lease,
        req,
        bind_parent=lambda *_: "bound",
        release_bound=lambda *_: None,
    )
    assert bridge.drain_prefill_ready() == (req,)
    assert len(bridge.consume_suffix(lease, 36, final_prompt_chunk=True)) == 36
    assert len(bridge.remaining_suffix_indices(lease)) == 0


def test_prefill_completion_is_event_driven_and_duplicate_safe():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    bridge = AgenticPMemorySchedulerBridge(authority)
    lease = bridge.reserve_workset(
        RequestGenerationAttempt("complete", 1, 4),
        owner="direct",
        parent_tokens=64,
        prompt_tokens=128,
    )
    req = SimpleNamespace(rid="complete")
    assert bridge.begin_ingress(lease, "dma")
    assert bridge.complete_ingress(lease, "dma", success=True)
    bridge.bind_and_publish(
        lease,
        req,
        bind_parent=lambda *_: "bound",
        release_bound=lambda bound_lease, *_: allocator.free(
            bound_lease.device_indices
        ),
    )
    assert bridge.drain_prefill_ready() == (req,)

    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(bridge.publish_prefill_complete(req))
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(result is results[0] for result in results)
    assert bridge.take_prefill_complete() == (results[0],)
    assert bridge.take_prefill_complete() == ()
    # Completion is a handoff, not a scheduler-side free.
    assert allocator.available_size() == 1024 - 128
    assert bridge.release_after_group_fence(
        lease.lease_id, reason="p2d_complete"
    )
    assert allocator.available_size() == 1024


def test_release_cannot_pass_an_inflight_ingress_fence():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    bridge = AgenticPMemorySchedulerBridge(authority)
    lease = bridge.reserve_workset(
        RequestGenerationAttempt("r", 1, 2),
        owner="slow",
        parent_tokens=64,
        prompt_tokens=128,
    )
    assert bridge.begin_ingress(lease, "h2d")
    assert authority.request_release(lease.lease_id)
    assert not authority.commit_release(lease.lease_id)
    assert bridge.complete_ingress(lease, "h2d", success=False)
    assert authority.commit_release(lease.lease_id, reason="cancel")
    assert allocator.available_size() == 1024


def test_bind_failure_releases_no_pages_behind_authority():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    bridge = AgenticPMemorySchedulerBridge(authority)
    lease = bridge.reserve_workset(
        RequestGenerationAttempt("r", 2, 1),
        owner="direct",
        parent_tokens=64,
        prompt_tokens=128,
    )
    assert bridge.begin_ingress(lease, "dma")
    assert bridge.complete_ingress(lease, "dma", success=True)

    def fail_bind(_lease, _req):
        raise RuntimeError("radix failed before mutation")

    released = []

    def release_failed(bound_lease, _req, binding):
        assert binding is None
        released.append(bound_lease.lease_id)
        allocator.free(bound_lease.device_indices)

    try:
        bridge.bind_and_publish(
            lease,
            SimpleNamespace(rid="child"),
            bind_parent=fail_bind,
            release_bound=release_failed,
        )
    except RuntimeError as exc:
        assert "radix failed" in str(exc)
    else:
        raise AssertionError("bind failure must propagate")
    assert authority.active_lease_count() == 1
    assert allocator.available_size() == 1024 - 128
    assert authority.request_release(lease.lease_id)
    assert authority.commit_release(lease.lease_id, reason="bind_failed")
    assert released == [lease.lease_id]
    assert allocator.available_size() == 1024


def test_prefill_abort_releases_through_authority_and_notifies_registry():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    bridge = AgenticPMemorySchedulerBridge(authority)
    lease = bridge.reserve_workset(
        RequestGenerationAttempt("abort", 3, 2),
        owner="direct",
        parent_tokens=64,
        prompt_tokens=128,
    )
    req = SimpleNamespace(rid="abort", req_pool_idx=None)
    assert bridge.begin_ingress(lease, "dma")
    assert bridge.complete_ingress(lease, "dma", success=True)
    bridge.bind_and_publish(
        lease,
        req,
        bind_parent=lambda *_: "bound",
        release_bound=lambda bound_lease, *_: allocator.free(
            bound_lease.device_indices
        ),
    )
    assert bridge.drain_prefill_ready() == (req,)
    terminal = []
    bridge.install_final_release_sink(
        lambda key, reason: terminal.append((key, reason))
    )
    assert bridge.release_abort(req, reason="grammar_accept_failed")
    assert allocator.available_size() == 1024
    assert terminal == [(lease.key, "grammar_accept_failed")]
