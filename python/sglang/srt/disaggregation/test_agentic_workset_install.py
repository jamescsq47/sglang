"""Controller grants consumed by the existing Direct/Slow/fresh broker."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_workset_device import prepare_workset
from sglang.srt.disaggregation.agentic_workset_ledger import WorksetLedger


def setup(parent=0, prompt=13):
    ledger = WorksetLedger(incarnation="test", page_count=16, page_size=4)
    plan = ledger.grant(snapshot_id="req:g1", attempt_id="attempt", owner="fresh" if not parent else "direct",
                        parent_tokens=parent, prompt_tokens=prompt)
    prepared = prepare_workset(plan, device="cpu", page_capacity=16)
    broker = AgenticPWorksetLeaseBroker(4)
    broker.request(plan.key.snapshot_id, parent, prompt, owner=plan.owner)
    return ledger, broker, prepared


def req(prompt=13):
    return NS(origin_input_ids=list(range(prompt)), req_pool_idx=None,
              prefix_indices=torch.empty(0, dtype=torch.int64), mamba_pool_idx=None,
              mamba_ping_pong_track_buffer=None)


@pytest.mark.parametrize("parent", [0, 4, 8])
def test_install_exact_plan_without_native_allocation(parent):
    ledger, broker, prepared = setup(parent)
    lease = broker.install_prepared(prepared)
    assert lease.controller_plan == prepared.plan
    assert lease.device_indices is prepared.device_indices
    assert lease.parent_allocated_tokens == parent
    assert broker.install_prepared(prepared) is lease
    assert ledger.counts.free_pages == 12
    assert broker.get("req:g1") is lease


def test_unfinished_preparation_cannot_publish_lease():
    _, broker, prepared = setup()
    prepared = replace(prepared, ready_event=NS(query=lambda: False))
    with pytest.raises(RuntimeError, match="device-ready"):
        broker.install_prepared(prepared)
    assert broker.get("req:g1") is None


def test_native_tp_plan_cannot_share_controller_broker():
    _, broker, prepared = setup()
    broker.install_tp_plan(1, (("req:g1", "fresh", 0, 13),))
    with pytest.raises(RuntimeError, match="native TP"):
        broker.install_prepared(prepared)


def test_replay_cannot_replace_live_attempt():
    _, broker, prepared = setup()
    broker.install_prepared(prepared)
    bad = replace(prepared, plan=replace(prepared.plan,
        key=replace(prepared.plan.key, attempt_id="other")))
    with pytest.raises(RuntimeError, match="conflicts"):
        broker.install_prepared(bad)


def test_fresh_cached_prefix_returns_only_private_duplicate_and_consumes_suffix():
    ledger, broker, prepared = setup()
    lease = broker.install_prepared(prepared)
    request = req()
    broker.handoff_fresh_to_req("req:g1", request, lease)
    request.prefix_indices = torch.tensor(list(range(100, 108)))
    returned = []
    broker.adopt_fresh_prefix(request, lease, prefix_tokens=8, pinned=True,
                             return_prefix=lambda plan, n: returned.append((plan, n)))
    broker.adopt_fresh_prefix(request, lease, prefix_tokens=8, pinned=True,
                             return_prefix=lambda *args: pytest.fail("duplicate return"))
    assert returned == [(prepared.plan, 8)]
    assert ledger.counts.free_pages == 12  # Not reusable until real TP fences.
    assert broker.consume_suffix(lease, 4, final_prompt_chunk=False).tolist() == [12, 13, 14, 15]
    assert broker.consume_suffix(lease, 1, final_prompt_chunk=True).tolist() == [16]
    assert request.prefix_indices.tolist() == list(range(100, 108))


def test_failed_private_prefix_retirement_does_not_advance_cursor():
    _, broker, prepared = setup()
    lease = broker.install_prepared(prepared)
    request = req()
    broker.handoff_fresh_to_req("req:g1", request, lease)
    request.prefix_indices = torch.arange(8)
    def reject(*args):
        raise RuntimeError("queue full")
    with pytest.raises(RuntimeError, match="queue full"):
        broker.adopt_fresh_prefix(request, lease, prefix_tokens=8, pinned=True,
                                 return_prefix=reject)
    assert lease.suffix_cursor == lease.cached_prefix_tokens == 0


@pytest.mark.parametrize("tokens,pinned", [(8, False), (7, True), (13, True)])
def test_invalid_prefix_rejected(tokens, pinned):
    _, broker, prepared = setup()
    lease = broker.install_prepared(prepared)
    request = req()
    broker.handoff_fresh_to_req("req:g1", request, lease)
    request.prefix_indices = torch.arange(tokens)
    with pytest.raises(ValueError, match="pinned"):
        broker.adopt_fresh_prefix(request, lease, prefix_tokens=tokens, pinned=pinned,
                                 return_prefix=lambda *args: pytest.fail("invalid return"))


def test_controller_plan_shape_rejected_before_publish():
    _, broker, prepared = setup()
    with pytest.raises(ValueError, match="complete workset"):
        broker.install_prepared(replace(prepared, device_indices=prepared.device_indices[:4]))
    assert broker.get("req:g1") is None


def test_prepared_replay_after_compute_handoff_cannot_resurrect_owner():
    _, broker, prepared = setup()
    lease = broker.install_prepared(prepared)
    broker.handoff_fresh_to_req("req:g1", req(), lease)
    broker.consume_suffix(lease, 13, final_prompt_chunk=True)
    with pytest.raises(RuntimeError, match="handed off or retired"):
        broker.install_prepared(prepared)


def test_controller_install_permanently_excludes_native_allocator_service():
    _, broker, prepared = setup()
    broker.install_prepared(prepared)
    with pytest.raises(RuntimeError, match="native allocator"):
        broker.service(NS())
    with pytest.raises(RuntimeError, match="native TP"):
        broker.install_tp_plan(2, ())


@pytest.mark.parametrize("parent", [0, 4])
def test_controller_hybrid_reserves_two_output_checkpoints_for_every_source(parent):
    state = NS(size=16, mamba_cache=NS(temporal=torch.zeros(1)),
               initialize_slots=lambda indices: None)
    pool = NS(enable_mamba_extra_buffer=True, mamba_ping_pong_track_buffer_size=2)
    broker = AgenticPWorksetLeaseBroker(4, state_allocators=(state,),
        mamba_req_to_token_pool=pool, reserve_mamba_checkpoint=True)
    ledger = WorksetLedger(incarnation="hybrid", page_count=16, page_size=4, mamba_slots=16)
    plan = ledger.grant("r:1", "attempt", owner="direct" if parent else "fresh",
        parent_tokens=parent, prompt_tokens=13,
        checkpoint_slots=int(bool(parent)), runtime_slots=5)
    prepared = prepare_workset(plan, device="cpu", page_capacity=16, mamba_pool=state)
    lease = broker.install_prepared(prepared)
    request = req()
    if parent:
        assert broker.begin_bind("r:1", lease)
        broker.commit_parent_bound("r:1", lease, state_donated_to_radix=True)
        broker.attach_runtime_state_for_bind("r:1", request, lease)
        broker.handoff_to_req("r:1", request, lease)
    else:
        broker.handoff_fresh_to_req("r:1", request, lease)
    assert request._agentic_mamba_prefill_checkpoint.numel() == 2
    assert request.mamba_ping_pong_track_buffer.numel() == 2
