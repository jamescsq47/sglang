"""CPU fault checks for the hybrid adapter; never submit GPU operations."""
from types import SimpleNamespace as NS

import pytest
import torch
import threading
import tempfile
import numpy as np

import sglang.srt.disaggregation.agentic_hybrid_dma as dma
from sglang.srt.disaggregation.agentic_host_staging import H2DLaunchFence
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker


@pytest.fixture
def host_arena():
    from sglang.srt.disaggregation.agentic_host_staging import SharedHostSnapshotArena
    with tempfile.TemporaryDirectory(prefix="q35-lazy-test-", dir="/dev/shm") as directory:
        arena = SharedHostSnapshotArena(directory, 65536, backend="memfd")
        try:
            yield arena
        finally:
            arena.close()


@pytest.mark.parametrize("slots", [None, 1, 2])
def test_actual_arena_lazy_materializes_hybrid_layout_once(monkeypatch, host_arena, slots):
    import sglang.srt.disaggregation.agentic_host_staging as host
    calls = []
    pool = NS(mamba_pool=object()) if slots else NS()
    snapshot = host_arena.create("s", 64, pool, 4096)
    def factory(**kwargs):
        calls.append(kwargs)
        def install(indices):
            assert snapshot._materialized is None
            assert tuple(indices) == tuple(range(1, slots + 1))
        return NS(state_slots=kwargs.get("state_slots", 1), set_state_indices=install, close=lambda **kwargs: None)
    monkeypatch.setattr(host, "SharedMHAHostSnapshot", factory)
    if slots:
        indices = np.arange(1, slots + 1)
        snapshot.set_state_indices(indices)
        indices[:] = 99
    assert not calls
    assert snapshot.materialize() is snapshot
    assert snapshot.materialize() is snapshot
    assert len(calls) == 1
    assert calls[0].get("state_slots") == slots
    assert host_arena.release(snapshot)
    assert host_arena.release(snapshot)
    assert host_arena.used_bytes == 0


def test_lazy_state_failure_closes_unpublished_inner(monkeypatch, host_arena):
    import sglang.srt.disaggregation.agentic_host_staging as host
    closed = []
    snapshot = host_arena.create("s", 64, NS(mamba_pool=object()), 4096)
    snapshot.set_state_indices([1, 2])
    with pytest.raises(ValueError, match="immutable"):
        snapshot.set_state_indices([1, 3])
    def fail(indices):
        raise RuntimeError("state install failure")
    monkeypatch.setattr(host, "SharedMHAHostSnapshot", lambda **kwargs: NS(
        set_state_indices=fail, close=lambda **kwargs: closed.append(kwargs)))
    with pytest.raises(RuntimeError, match="state install failure"):
        snapshot.materialize()
    assert snapshot._materialized is None
    assert closed == [{"unlink": False}]
    snapshot.close()
    snapshot.close()
    with pytest.raises(RuntimeError, match="released"):
        snapshot.set_state_indices([1, 2])


def test_lazy_hybrid_default_h2d_mirrors_once_after_materialize(monkeypatch, host_arena):
    import sglang.srt.disaggregation.agentic_host_staging as host
    snapshot = host_arena.create("s", 64, NS(mamba_pool=object()), 4096)
    installs, mirrors = [], []
    def factory(**kwargs):
        assert "state_slots" not in kwargs  # D2P defaults to one slot.
        return NS(state_slots=1, set_state_indices=lambda values: installs.append(tuple(values)),
                  close=lambda **kwargs: None)
    monkeypatch.setattr(host, "SharedMHAHostSnapshot", factory)
    device_indices = object()  # Simulate the GPU vector without requiring CUDA.
    def mirror(value):
        assert value is device_indices
        mirrors.append(value)
        return [7]
    monkeypatch.setattr(host, "_device_indices_to_host", mirror)
    load = {"record": {"snapshot": snapshot},
            "workset_lease": NS(state_device_indices=(device_indices,))}
    host.AgenticPHostStagingManager._configure_hybrid_h2d_state(load)
    host.AgenticPHostStagingManager._configure_hybrid_h2d_state(load)
    assert mirrors == [device_indices]
    assert installs == [(7,)]
    assert snapshot._pending_state_indices == (7,)
    snapshot.set_state_indices([7])  # Idempotent after materialization.
    with pytest.raises(ValueError, match="immutable"):
        snapshot.set_state_indices([8])
    assert host_arena.release(snapshot)
    assert host_arena.used_bytes == 0


def test_hybrid_h2d_failed_setup_is_not_published(monkeypatch):
    import sglang.srt.disaggregation.agentic_host_staging as host
    def fail(indices):
        raise RuntimeError("invalid destination")
    snapshot = NS(materialize=lambda: None, set_state_indices=fail)
    load = {"record": {"snapshot": snapshot},
            "workset_lease": NS(state_device_indices=(torch.tensor([7]),))}
    with pytest.raises(RuntimeError, match="invalid destination"):
        host.AgenticPHostStagingManager._configure_hybrid_h2d_state(load)
    assert "hybrid_state_configured" not in load


@pytest.mark.parametrize("fence_style", ["legacy", "composite", "unavailable"])
def test_hybrid_host_retry_rebinds_only_after_composite_fence(monkeypatch, host_arena, fence_style):
    import sglang.srt.disaggregation.agentic_host_staging as host
    pool = NS(mamba_pool=NS(mamba_cache=NS(conv=[torch.zeros(1, 16, 2)],
                                         temporal=torch.zeros(1, 16, 2))))
    snapshot = host_arena.create("s", 64, pool, 4096)
    inner = object.__new__(dma.RegisteredHybridHostSnapshot)
    inner.__dict__.update(device_pool=pool, state_slots=1, state_indices=None,
                          _closed=False, _state_bounce_refs=None)
    inner.close = lambda **kwargs: None
    monkeypatch.setattr(host, "SharedMHAHostSnapshot", lambda **kwargs: inner)
    record = {"snapshot": snapshot, "loading": True}
    lease = NS(state_device_indices=(torch.tensor([7]),))
    event = NS(done=False, query=lambda: event.done, synchronize=lambda: None)
    load = {"record": record, "workset_lease": lease, "event": event,
            "request_generation": NS(snapshot_id="s"), "io_attempt": "attempt",
            "io_inflight": True}
    if fence_style != "legacy":
        load["launch_fence"] = host.H2DLaunchFence(event=event, submitted=True,
            armed=True, unavailable=fence_style == "unavailable")
    host.AgenticPHostStagingManager._configure_hybrid_h2d_state(load)
    assert inner.state_indices == (7,)
    calls = []
    manager = NS(loads={"r": load}, host_ready={}, owner="p", tp_rank=0, tp_size=2,
                 _get_state_lock=lambda: threading.RLock(),
                 ledger=NS(request_d2p_retry=lambda *a, **k: True,
                           complete_d2p_retry_rank=lambda *a, **k: True),
                 workset_broker=NS(mark_io_quiesced=lambda *a: calls.append("quiesce") or True,
                                  request_release=lambda *a: calls.append("release")))
    monkeypatch.setattr(host.AgenticPHostStagingManager, "_notify_scheduler", lambda *a: None)
    monkeypatch.setattr(host.AgenticPHostStagingManager, "_release_h2d_lane", lambda *a: None)
    assert not host.AgenticPHostStagingManager._discard_failed_h2d_load(manager, "r", load)
    assert inner.state_indices == (7,)
    assert not calls
    event.done = True
    if fence_style == "unavailable":
        assert not host.AgenticPHostStagingManager._discard_failed_h2d_load(manager, "r", load)
        assert inner.state_indices == (7,) and not calls
        assert load["dma_quarantined"] and manager._h2d_poisoned
        return
    assert host.AgenticPHostStagingManager._discard_failed_h2d_load(manager, "r", load)
    assert calls == ["quiesce", "release"]
    assert inner.state_indices is None and snapshot._pending_state_indices is None
    assert host_arena.used_bytes == 4096 and manager.host_ready["s"] is record
    retry = {"record": record, "workset_lease": NS(state_device_indices=(torch.tensor([9]),))}
    host.AgenticPHostStagingManager._configure_hybrid_h2d_state(retry)
    assert inner.state_indices == (9,)
    # A stale callback from the old attempt must not reset the new binding.
    assert host.AgenticPHostStagingManager._discard_failed_h2d_load(manager, "r", load)
    assert inner.state_indices == (9,)


@pytest.mark.parametrize("indices", [[], [1, 2, 3], [-1]])
def test_lazy_state_rejects_invalid_vectors(host_arena, indices):
    snapshot = host_arena.create("s", 64, NS(mamba_pool=object()), 4096)
    with pytest.raises(ValueError):
        snapshot.set_state_indices(indices)
    assert snapshot._materialized is None


def test_p2d_try_submit_uses_actual_lazy_arena_and_rolls_back(monkeypatch, host_arena):
    from sglang.srt.disaggregation.p2d_host_staging import AgenticPToDHostStagingManager
    monkeypatch.setenv("SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT", "false")
    observed = []
    def prepare(*args, **kwargs):
        snapshot = next(iter(host_arena._active_extents.values()))[0]
        assert snapshot._materialized is None
        observed.append(snapshot._pending_state_indices)
        return None  # Inject group admission rejection after local preparation.
    manager = object.__new__(AgenticPToDHostStagingManager)
    manager.__dict__.update(arena=host_arena, device_pool=NS(mamba_pool=object()),
        _lock=threading.RLock(), _prepared={}, _active={}, _records={}, _results={},
        _byte_size=lambda n: 4096, owner="p", prefill_domain=0, numa_node=0,
        tp_rank=0, tp_size=2, page_size=64, hard_watermark=1.0,
        ledger=NS(get=lambda sid: {"state": "offered", "prefill_domain": 0},
                  prepare_p2d_write_rank=prepare, reject_unclaimed_offer=lambda *args, **kwargs: True))
    req = NS(bootstrap_room=1, origin_input_ids=[5] * 64, output_ids=[9],
             mamba_pool_idx=1, _agentic_p2d_mamba_checkpoint_index=2,
             _agentic_p2d_mamba_checkpoint_tokens=64)
    assert not manager.try_submit(req, list(range(64)))
    assert observed == [(1, 2)]
    assert host_arena.used_bytes == 0


class Event:
    def __init__(self, **kwargs):
        self.records = []

    def record(self, stream):
        self.records.append(stream)


@pytest.mark.parametrize("enabled,hybrid,tracked,expected", [
    (True, True, 896, 896), (True, True, 960, 960),
    (True, True, None, None), (False, True, 896, None),
    (True, False, 896, None),
])
def test_prebuilt_preserves_only_agentic_hybrid_checkpoint(monkeypatch, enabled, hybrid, tracked, expected):
    from sglang.srt.disaggregation.decode_schedule_batch_mixin import cache_pd_decode_committed_req
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", str(enabled).lower())
    original = list(range(970))
    req = NS(fill_ids=original, kv_committed_len=969,
             mamba_last_track_seqlen=tracked, mamba_pool_idx=1 if hybrid else None,
             mamba_next_track_idx=1, mamba_ping_pong_track_buffer=[2, 3])
    def cache(req):
        assert len(req.fill_ids) == 969
        req.mamba_last_track_seqlen = None
    cache_pd_decode_committed_req(NS(cache_unfinished_req=cache, page_size=64), req)
    assert req.fill_ids is original
    assert req.mamba_last_track_seqlen == expected
    assert req.mamba_next_track_idx == 1
    assert req.mamba_ping_pong_track_buffer == [2, 3]


def test_prebuilt_failed_cache_does_not_restore_checkpoint(monkeypatch):
    from sglang.srt.disaggregation.decode_schedule_batch_mixin import cache_pd_decode_committed_req
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", "true")
    original = list(range(916))
    req = NS(fill_ids=original, kv_committed_len=915,
             mamba_last_track_seqlen=896, mamba_pool_idx=1)
    def cache(req):
        req.mamba_last_track_seqlen = None
        raise RuntimeError("cache failure")
    with pytest.raises(RuntimeError, match="cache failure"):
        cache_pd_decode_committed_req(NS(cache_unfinished_req=cache, page_size=64), req)
    assert req.fill_ids is original
    assert req.mamba_last_track_seqlen is None


@pytest.mark.parametrize("hybrid,owner,checks", [
    (True, "candidate", False), (True, "pending", False),
    (True, "empty", True), (False, "pending", True),
])
def test_idle_checker_counts_atomic_hybrid_tp_ownership(monkeypatch, hybrid, owner, checks):
    from sglang.srt.managers.scheduler_runtime_checker_mixin import SchedulerRuntimeCheckerMixin
    from sglang.srt.disaggregation.utils import DisaggregationMode
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", "true")
    calls = []
    offload = NS(ongoing_offload=[], _decode_pending_release_tokens=0,
                 agentic_direct_candidates={"s": {}} if owner == "candidate" else {},
                 _agentic_release_ownership={"s": (object(), 0)} if owner == "pending" else {})
    scheduler = NS(enable_hisparse=False, is_hybrid_ssm=hybrid,
        disaggregation_mode=DisaggregationMode.DECODE, waiting_queue=[],
        disagg_decode_transfer_queue=NS(queue=[]), disagg_decode_prealloc_queue=NS(queue=[]),
        decode_offload_manager=offload, init_new_token_ratio=1,
        check_memory=lambda: calls.append("memory"), check_tree_cache=lambda: calls.append("tree"),
        maybe_sleep_on_idle=lambda: None)
    SchedulerRuntimeCheckerMixin.self_check_during_idle(scheduler)
    assert calls == (["memory", "tree"] if checks else [])
    offload.agentic_direct_candidates.clear()
    offload._agentic_release_ownership.clear()
    calls.clear()
    SchedulerRuntimeCheckerMixin.self_check_during_idle(scheduler)
    assert calls == ["memory", "tree"]


@pytest.mark.parametrize("hybrid,has_lease,checks", [(True, True, False), (True, False, True), (False, True, True)])
def test_prefill_idle_resumes_after_hybrid_lease_release(monkeypatch, hybrid, has_lease, checks):
    from sglang.srt.managers.scheduler_runtime_checker_mixin import SchedulerRuntimeCheckerMixin
    from sglang.srt.disaggregation.utils import DisaggregationMode
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", "true")
    calls = []
    scheduler = NS(enable_hisparse=False, is_hybrid_ssm=hybrid,
        disaggregation_mode=DisaggregationMode.PREFILL, disagg_prefill_inflight_queue=[],
        agentic_p_workset_broker=NS(_lock=threading.RLock(), _leases={"s": object()} if has_lease else {}),
        init_new_token_ratio=1, check_memory=lambda: calls.append("memory"),
        check_tree_cache=lambda: calls.append("tree"), maybe_sleep_on_idle=lambda: None)
    SchedulerRuntimeCheckerMixin.self_check_during_idle(scheduler)
    assert calls == (["memory", "tree"] if checks else [])


@pytest.mark.parametrize("hybrid,enabled,retained,pending,expected", [
    (True, True, True, False, ["offload", "finish"]),
    (True, True, True, True, ["offload"]),
    (True, True, False, False, ["offload", "finalize", "finish"]),
    (False, True, True, False, ["finish", "native_free"]),
    (True, False, True, False, ["finish", "native_free"]),
])
def test_prebuilt_one_token_hybrid_uses_existing_manager(monkeypatch, hybrid, enabled, retained, pending, expected):
    import sglang.srt.managers.scheduler_output_processor_mixin as processor
    from sglang.srt.disaggregation.utils import DisaggregationMode
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", str(enabled).lower())
    calls = []
    monkeypatch.setattr(processor, "release_kv_cache", lambda *args: calls.append("native_free"))
    req = NS(rid="one", mamba_pool_idx=1 if hybrid else None,
        sampling_params=NS(custom_params={"agentic_request_id": "one"}),
        check_finished=lambda: None, finished=lambda: True,
        time_stats=NS(set_decode_prebuilt_finish_time=lambda: None,
                      set_quick_finish_time=lambda: calls.append("finish")))
    manager = NS(offload_kv_cache=lambda req: (calls.append("offload"), retained)[1],
                 is_response_pending=lambda rid: pending,
                 finalize_release_on_finish=lambda req: calls.append("finalize"))
    scheduler = NS(disaggregation_mode=DisaggregationMode.DECODE,
        decode_offload_manager=manager, tree_cache=object(), stream_output=lambda *args: None)
    processor.SchedulerOutputProcessorMixin.process_batch_result_prebuilt(
        scheduler, NS(reqs=[req], return_logprob=False))
    assert calls == expected


@pytest.mark.parametrize("prompt_tokens", [896, 907])
def test_first_sample_snapshot_excludes_uncomputed_token(monkeypatch, prompt_tokens):
    from sglang.srt.disaggregation.agentic_hybrid_transfer import snapshot_token_count_for_req, state_indices_for_req
    monkeypatch.setenv("SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT", "false")
    req = NS(origin_input_ids=[5] * prompt_tokens, output_ids=[248046],
        mamba_pool_idx=1, mamba_last_track_seqlen=896,
        mamba_next_track_idx=1, mamba_ping_pong_track_buffer=[2, 3])
    committed = len(req.origin_input_ids + req.output_ids[:-1])
    assert committed == prompt_tokens
    assert snapshot_token_count_for_req(req, committed, ("mamba",), 64) == 896
    assert int(state_indices_for_req(req, ("mamba",), checkpoint_tokens=896, page_size=64)[0][0]) == 2


def adapter():
    obj = object.__new__(dma.RegisteredHybridHostSnapshot)
    obj.attention = NS(path="unused", file_offset=0)
    obj._closed = False
    obj.state_indices = (1,)
    obj._state_mapping = None
    obj._state_windows = ()
    obj._state_bounce = None
    obj._prepare_state_backing = lambda: None
    return obj


def test_preallocated_host_interface_is_idempotent():
    obj = adapter()
    assert obj.materialize() is obj
    obj.mark_populated()
    obj.mark_populated()
    assert obj.requires_prefault is False
    obj._closed = True
    with pytest.raises(RuntimeError):
        obj.materialize()
    with pytest.raises(RuntimeError):
        obj.mark_populated()


@pytest.mark.parametrize("failure", [False, True])
def test_attention_event_never_publishes_complete_hybrid_fence(monkeypatch, failure):
    monkeypatch.setattr(torch.cuda, "Event", Event)
    obj = adapter()
    fence = H2DLaunchFence(event=Event())
    stream = object()

    def attention(inner):
        inner.submitted = True
        inner.event.record(stream)
        inner.armed = True
        assert not fence.armed
        if failure:
            raise RuntimeError("partial Attention submission")
        return inner.event, ["attention-reference"]

    def state(**kwargs):
        assert not fence.armed
        assert "attention-reference" in kwargs["prior_refs"]
        fence.event.record(stream)
        fence.armed = True
        return fence.event, kwargs["prior_refs"]

    obj._append_state = state
    if failure:
        with pytest.raises(RuntimeError, match="partial Attention"):
            obj._finish_composite(attention, stream=stream, fence=fence, to_host=True)
    else:
        obj._finish_composite(attention, stream=stream, fence=fence, to_host=True)
    assert fence.armed and fence.submitted
    assert fence.event.records == [stream]


def test_mapping_reference_released_exactly_once():
    obj = adapter()
    released = []
    mapping = NS(_users=2, release=lambda windows: released.append(windows))
    obj._state_mapping = mapping
    obj._state_windows = ("window",)
    obj._drop_state_mapping()
    obj._drop_state_mapping()
    assert mapping._users == 1
    assert released == [("window",)]


def test_state_addresses_cannot_change_during_live_extent():
    obj = adapter()
    obj.state_indices = None
    obj.state_slots = 1
    obj.device_pool = NS(mamba_pool=NS(mamba_cache=NS(
        conv=[torch.zeros(2, 8, 3)], temporal=torch.zeros(2, 8, 3, 3))))
    obj.set_state_indices([2])
    obj.set_state_indices([2])
    with pytest.raises(RuntimeError):
        obj.set_state_indices([3])
    with pytest.raises(ValueError):
        obj.set_state_indices([8])


class Allocator:
    def __init__(self, capacity):
        self.free_slots = torch.arange(1, capacity + 1)

    def alloc(self, count):
        if count > len(self.free_slots):
            return None
        result = self.free_slots[:count].clone()
        self.free_slots = self.free_slots[count:]
        return result

    def free(self, indices):
        self.free_slots = torch.cat([self.free_slots, indices])


@pytest.mark.parametrize("hybrid", [False, True])
def test_decode_admission_queries_attention_token_capacity(hybrid):
    from sglang.srt.disaggregation.decode import DecodePreallocQueue
    def wrong_api():
        raise AssertionError("hybrid cache has no scalar evictable_size API")
    cache = NS(supports_mamba=lambda: hybrid, full_evictable_size=lambda: 64,
               evictable_size=wrong_api if hybrid else lambda: 64)
    queue = NS(tree_cache=cache, token_to_kv_pool_allocator=NS(available_size=lambda: 128),
               scheduler=NS(running_batch=NS(reqs=[]), waiting_queue=[], last_batch=None),
               transfer_queue=NS(queue=[]), retracted_queue=[], num_reserved_decode_tokens=64)
    assert DecodePreallocQueue._allocatable_tokens(queue) == 192


@pytest.mark.parametrize("active_already_freed", [False, True])
def test_cancel_handed_runtime_before_native_req_slot_releases_all_states_once(active_already_freed):
    from sglang.srt.mem_cache.common import release_kv_cache, release_unadmitted_mamba_cow
    state = Allocator(3)
    indices = state.alloc(3)
    req = NS(req_pool_idx=None, mamba_pool_idx=indices[0],
             mamba_ping_pong_track_buffer=indices[1:],
             _agentic_mamba_runtime_reserved=True)
    cache = NS(supports_mamba=lambda: True, req_to_token_pool=NS(mamba_pool=state))
    release_unadmitted_mamba_cow(req, cache)
    assert req.mamba_pool_idx is not None and len(state.free_slots) == 0
    if active_already_freed:
        state.free(req.mamba_pool_idx.reshape(1))
        req.mamba_pool_idx = None
    release_kv_cache(req, cache, is_insert=False)
    release_kv_cache(req, cache, is_insert=False)
    assert sorted(state.free_slots.tolist()) == [1, 2, 3]
    assert req.mamba_pool_idx is None and req.mamba_ping_pong_track_buffer is None


@pytest.mark.parametrize("keep", [None, 1])
def test_native_runtime_release_then_duplicate_cleanup_preserves_radix_checkpoint(keep):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.common import release_kv_cache
    state = Allocator(3)
    indices = state.alloc(3)
    req = NS(req_pool_idx=0, mamba_pool_idx=indices[0],
             mamba_ping_pong_track_buffer=indices[1:], _agentic_mamba_runtime_reserved=True)
    pool = NS(mamba_pool=state, enable_mamba_extra_buffer=True,
              mamba_ping_pong_track_buffer_size=2,
              req_index_to_mamba_ping_pong_track_buffer_mapping=indices[1:].reshape(1, 2))
    HybridReqToTokenPool.free_mamba_cache(pool, req, mamba_ping_pong_track_buffer_to_keep=keep)
    req.req_pool_idx = None
    cache = NS(supports_mamba=lambda: True, req_to_token_pool=pool)
    release_kv_cache(req, cache, is_insert=False)
    release_kv_cache(req, cache, is_insert=False)
    assert sorted(state.free_slots.tolist()) == ([1, 2, 3] if keep is None else [1, 2])
    assert req.mamba_ping_pong_track_buffer is None
    assert not req._agentic_mamba_runtime_reserved


def test_mamba_checker_counts_broker_and_handed_req_without_double_count():
    import threading
    from sglang.srt.managers.scheduler_runtime_checker_mixin import SchedulerRuntimeCheckerMixin
    req = NS(req_pool_idx=None, mamba_pool_idx=torch.tensor(9),
             mamba_ping_pong_track_buffer=torch.tensor([10, 11]), _agentic_mamba_runtime_reserved=True)
    broker = NS(_lock=threading.Lock(), _leases={
        "binding": NS(state_device_indices=(torch.tensor([1]),),
                      runtime_state_device_indices=(torch.tensor([2, 3, 4]),)),
        "handed": NS(state_device_indices=(), runtime_state_device_indices=()),
    })
    scheduler = NS(agentic_p_workset_broker=broker, waiting_queue=[req, req], decode_offload_manager=None)
    assert SchedulerRuntimeCheckerMixin._agentic_owned_mamba_slots(scheduler) == {
        "broker": 4, "unpooled_req": 3, "detached_decode": 0,
    }


@pytest.mark.parametrize("slots", [0, 1, 3])
def test_state_shortage_rolls_back_whole_workset(slots):
    attention = Allocator(128)
    state = Allocator(slots)
    broker = AgenticPWorksetLeaseBroker(64, state_allocators=(state,),
        mamba_req_to_token_pool=NS(enable_mamba_extra_buffer=True,
                                 mamba_ping_pong_track_buffer_size=2))
    broker.request("r:1", parent_tokens=64, prompt_tokens=128)
    broker.service(attention)
    assert broker.get("r:1") is None
    assert len(attention.free_slots) == 128
    assert len(state.free_slots) == slots


def test_donated_checkpoint_and_runtime_have_distinct_owners():
    attention, state = Allocator(128), Allocator(4)
    broker = AgenticPWorksetLeaseBroker(64, state_allocators=(state,),
        mamba_req_to_token_pool=NS(enable_mamba_extra_buffer=True,
                                 mamba_ping_pong_track_buffer_size=2))
    broker.request("r:1", parent_tokens=64, prompt_tokens=128)
    broker.service(attention)
    lease = broker.get("r:1")
    assert lease is not None and len(state.free_slots) == 0
    checkpoint = lease.state_device_indices[0].clone()
    assert broker.begin_bind("r:1", lease)
    broker.commit_parent_bound("r:1", lease, state_donated_to_radix=True)
    req = NS(origin_input_ids=list(range(128)), mamba_pool_idx=None)
    broker.attach_runtime_state_for_bind("r:1", req, lease)
    assert not getattr(req, "_agentic_mamba_runtime_reserved", False)
    broker.handoff_to_req("r:1", req, lease)
    assert req._agentic_mamba_runtime_reserved
    assert lease.state_device_indices == lease.runtime_state_device_indices == ()
    runtime = torch.cat([req.mamba_pool_idx.reshape(1), req.mamba_ping_pong_track_buffer])
    assert checkpoint.item() not in runtime.tolist()
    assert sorted(torch.cat([checkpoint, runtime]).tolist()) == [1, 2, 3, 4]


@pytest.mark.parametrize("locked,shared", [(False, False), (True, False), (False, True)])
def test_release_short_cached_parent_preserves_live_shared_branch(locked, shared):
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
    root = NS()
    node = NS(parent=root, full_lock_ref=int(locked), mamba_lock_ref=0,
              children={"other": object()} if shared else {},
              key=NS(extra_key="generation"), mamba_value=torch.tensor([1]))
    freed = []

    def evict(current, _mamba):
        freed.append(current)
        return 64, 1, current, None

    cache = NS(disable=False, page_size=64, root_node=root,
               match_prefix=lambda _: NS(device_indices=torch.arange(64), last_device_node=node),
               _evict_leaf_node=evict)
    req = NS(extra_key="generation", origin_input_ids=[1] * 64, output_ids=[2] * 64)
    count = MambaRadixCache.release_request_generation_cache(cache, req, committed_len=128)
    assert count == (0 if locked or shared else 64)
    assert len(freed) == (0 if locked or shared else 1)


@pytest.mark.parametrize("direction", ["backup", "load"])
def test_state_bounce_partial_submission_retains_references(monkeypatch, direction):
    from contextlib import nullcontext
    from sglang.srt.disaggregation.agentic_hybrid_snapshot import SharedMambaHostSnapshot
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    real_empty_like = torch.empty_like
    monkeypatch.setattr(torch, "empty_like", lambda tensor, **kw: real_empty_like(tensor, device="cpu"))
    obj = object.__new__(SharedMambaHostSnapshot)
    obj.state_slots = 1
    obj.conv = [torch.zeros(1, 1, 2)]
    obj.temporal = torch.zeros(1, 1, 2)
    obj.mamba_pool = NS(mamba_cache=NS(conv=[torch.zeros(1, 3, 2)],
                                     temporal=torch.zeros(1, 3, 2)))
    fence = H2DLaunchFence(event=Event())
    original_copy = torch.Tensor.copy_
    posted = []

    def fail_second_copy(destination, source, *args, **kwargs):
        if kwargs.get("non_blocking"):
            assert fence.copy_refs, "bounce must be held before any CUDA submission"
            posted.append(1)
            if len(posted) == 2:
                raise RuntimeError("partial state DMA")
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_second_copy)
    operation = obj.start_backup_from_device if direction == "backup" else obj.start_load_to_device
    with pytest.raises(RuntimeError, match="partial state DMA"):
        operation(torch.tensor([1]), object(), launch_fence=fence)
    assert len(posted) == 2
    assert fence.submitted and fence.armed
    assert fence.copy_refs[1] and fence.copy_refs[2] is not None
