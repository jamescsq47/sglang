import threading
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.agentic_decode_memory_bridge import (
    AgenticDMemorySchedulerBridge,
)
from sglang.srt.disaggregation.agentic_memory_authority import (
    AgenticMemoryAuthority,
    RequestGenerationAttempt,
)
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    agentic_finished_req_needs_parent_handoff,
)
from sglang.srt.mem_cache import common as mem_common
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.managers.schedule_batch import ScheduleBatch


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
        assert not self.live.intersection(values)
        self.live.update(values)
        return torch.tensor(values, dtype=torch.int64)

    def free(self, indices):
        values = [int(value) for value in indices.tolist()]
        assert set(values).issubset(self.live)
        self.live.difference_update(values)
        self.free_values.extend(values)


def _ready_decode(allocator, *, growth_tokens=128):
    authority = AgenticMemoryAuthority(
        allocator, decode_growth_reserve_tokens=growth_tokens
    )
    bridge = AgenticDMemorySchedulerBridge(authority)
    lease = bridge.reserve_decode(
        RequestGenerationAttempt("r", 1, 1),
        owner="p2d",
        prompt_tokens=65,
        decode_growth_tokens=growth_tokens,
    )
    req = SimpleNamespace(rid="r")
    grown = []
    assert bridge.begin_ingress(lease, "p2d")
    assert bridge.complete_ingress(lease, "p2d", success=True)

    def bind_prompt(*_args):
        authority.assert_native_guard()
        return "native-request"

    bridge.bind_and_publish(
        lease,
        req,
        bind_prompt=bind_prompt,
        release_bound=lambda bound_lease, *_: (
            allocator.free(bound_lease.device_indices),
            [allocator.free(indices) for indices in grown],
        ),
    )
    assert bridge.drain_decode_ready() == (req,)
    return authority, bridge, lease, req, grown


def test_decode_growth_consumes_credit_without_second_allocation_owner():
    allocator = FakeAllocator()
    authority, bridge, lease, req, grown = _ready_decode(allocator)
    # Imported input is one independently rounded 128-token allocation.
    assert allocator.alloc_calls == 1
    assert authority.available_tokens() == 1024 - 128 - 128

    def allocate(raw):
        indices = raw.alloc(64)
        grown.append(indices)
        return indices

    out = bridge.allocate_decode_growth(
        [req], seq_lens_next=torch.tensor([129]), allocate=allocate
    )
    assert len(out) == 64
    assert allocator.alloc_calls == 2
    assert authority.available_tokens() == 1024 - 128 - 64 - 128

    # A non-page-boundary token uses the existing page and no physical alloc.
    marker = object()
    assert (
        bridge.allocate_decode_growth(
            [req],
            seq_lens_next=torch.tensor([130]),
            allocate=lambda _raw: marker,
        )
        is marker
    )
    assert allocator.alloc_calls == 2

    assert bridge.finish_decode(req)
    assert bridge.begin_egress(req, "d2p")
    assert authority.request_release(lease.lease_id)
    assert not authority.commit_release(lease.lease_id)
    assert bridge.complete_egress(req, "d2p", success=True)
    assert bridge.release_after_group_fence(
        lease.lease_id, reason="d2p_committed"
    )
    assert bridge.release_after_group_fence(
        lease.lease_id, reason="duplicate"
    )
    assert allocator.available_size() == 1024


def test_decode_growth_rejects_foreign_and_renews_exhausted_headroom():
    allocator = FakeAllocator()
    _, bridge, _, req, _ = _ready_decode(allocator, growth_tokens=64)
    foreign = SimpleNamespace(
        _agentic_d_memory_bridge=bridge,
        _agentic_d_memory_lease=None,
    )
    try:
        bridge.allocate_decode_growth(
            [foreign], seq_lens_next=torch.tensor([129]), allocate=lambda _: None
        )
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("foreign request must fail closed")

    first = bridge.allocate_decode_growth(
        [req],
        seq_lens_next=torch.tensor([129]),
        allocate=lambda raw: raw.alloc(64),
    )
    assert len(first) == 64
    second = bridge.allocate_decode_growth(
        [req],
        seq_lens_next=torch.tensor([193]),
        allocate=lambda raw: raw.alloc(64),
    )
    assert len(second) == 64


def test_decode_growth_capacity_shortage_is_retryable_without_allocation(monkeypatch):
    allocator = FakeAllocator(size=192)
    _, bridge, _, req, _ = _ready_decode(allocator, growth_tokens=64)
    first = bridge.allocate_decode_growth(
        [req],
        seq_lens_next=torch.tensor([129]),
        allocate=lambda raw: raw.alloc(64),
    )
    assert len(first) == 64
    assert allocator.available_size() == 0

    allocate_calls = []
    assert (
        bridge.allocate_decode_growth(
            [req],
            seq_lens_next=torch.tensor([193]),
            allocate=lambda raw: allocate_calls.append(raw),
        )
        is None
    )
    assert allocate_calls == []

    monkeypatch.setattr(
        schedule_batch_module, "evict_from_tree_cache", lambda *_args: None
    )
    batch = SimpleNamespace(
        reqs=[req],
        seq_lens_cpu=torch.tensor([192]),
        tree_cache=object(),
        token_to_kv_pool_allocator=allocator,
        new_tokens_required_next_decode=lambda _indices: 64,
    )
    assert not ScheduleBatch.check_decode_mem(batch)


def test_decode_scheduler_does_not_independently_drain_tp_ready_queue():
    calls = []
    req = object()
    scheduler = SimpleNamespace(
        agentic_d_memory_v2_bridge=SimpleNamespace(
            drain_decode_ready=lambda **kwargs: (
                calls.append(("ready", kwargs["max_items"])) or (req,)
            )
        ),
        waiting_queue=[],
        decode_offload_manager=SimpleNamespace(
            check_offload_progress=lambda: calls.append("legacy-offload")
        ),
        disagg_decode_prealloc_queue=SimpleNamespace(
            resume_retracted_reqs=lambda: calls.append("legacy-prealloc")
        ),
    )

    SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

    # TP0 drains immutable activation tickets and broadcasts their exact order
    # through Scheduler.recv_requests; this per-rank loop must not choose work.
    assert scheduler.waiting_queue == []
    assert calls == []


def test_common_decode_allocation_enters_same_authority(monkeypatch):
    allocator = FakeAllocator()
    _, bridge, _, req, grown = _ready_decode(allocator)
    batch = SimpleNamespace(
        reqs=[req],
        seq_lens_cpu=torch.tensor([128]),
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
    )

    def native(_batch, _tokens):
        indices = allocator.alloc(64)
        grown.append(indices)
        return indices

    monkeypatch.setattr(mem_common, "_alloc_for_decode_native", native)
    before = allocator.alloc_calls
    result = mem_common.alloc_for_decode(batch, 1)
    assert len(result) == 64
    assert allocator.alloc_calls == before + 1


def test_common_decode_rejects_mixed_authority_batch(monkeypatch):
    allocator = FakeAllocator()
    _, _, _, req, _ = _ready_decode(allocator)
    batch = SimpleNamespace(
        reqs=[req, SimpleNamespace()],
        seq_lens_cpu=torch.tensor([128, 128]),
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
    )
    monkeypatch.setattr(
        mem_common,
        "_alloc_for_decode_native",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not allocate")),
    )
    try:
        mem_common.alloc_for_decode(batch, 1)
    except RuntimeError as exc:
        assert "mixes" in str(exc)
    else:
        raise AssertionError("mixed ownership must fail before allocation")


def test_decode_completion_is_event_driven_and_duplicate_safe():
    allocator = FakeAllocator()
    _, bridge, lease, req, _ = _ready_decode(allocator)
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(bridge.publish_decode_complete(req))
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(result is results[0] for result in results)
    assert bridge.drain_decode_complete() == (results[0],)
    assert bridge.drain_decode_complete() == ()
    assert allocator.available_size() < 1024
    assert bridge.release_after_group_fence(
        lease.lease_id, reason="d2p_complete"
    )
    assert allocator.available_size() == 1024


def test_true_final_is_released_without_d2p_completion_event():
    allocator = FakeAllocator()
    _, bridge, lease, req, _ = _ready_decode(allocator)
    terminal = []
    bridge.install_final_release_sink(
        lambda key, reason: terminal.append((key, reason))
    )
    req.sampling_params = SimpleNamespace(
        custom_params={
            "agentic_request_id": "trajectory",
            "agentic_generation": 1,
            "agentic_terminal_marker_token_ids": [[9]],
            "agentic_tool_suffix_token_ids": [[7]],
        }
    )
    req.output_ids = [3, 9]
    req.tokenizer = None
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
    assert not agentic_finished_req_needs_parent_handoff(req)
    assert bridge.release_final(req)
    assert bridge.drain_decode_complete() == ()
    assert allocator.available_size() == 1024
    assert terminal == [(lease.key, "application_final")]
    assert not hasattr(req, "_agentic_d_memory_lease")


def test_tool_and_unknown_continue_but_length_does_not():
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            custom_params={
                "agentic_request_id": "trajectory",
                "agentic_generation": 2,
                "agentic_terminal_marker_token_ids": [[9]],
                "agentic_tool_suffix_token_ids": [[7]],
            }
        ),
        output_ids=[3, 7],
        tokenizer=None,
        finished_reason=SimpleNamespace(to_json=lambda: {"type": "stop"}),
    )
    assert agentic_finished_req_needs_parent_handoff(req)
    req.output_ids = [3, 4]
    assert agentic_finished_req_needs_parent_handoff(req)
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length"})
    assert not agentic_finished_req_needs_parent_handoff(req)
