import os
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticPHostStagingManager,
    HostStageState,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotState
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    CUSTOM_GENERATION,
    CUSTOM_PARENT_GENERATION,
    CUSTOM_REQUEST_ID,
)
from sglang.srt.disaggregation.agentic_tp import (
    rank_env_int,
    rank_scoped_arena_directory,
    request_generation_key,
)
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostStagingManager,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler


def _ledger():
    fd, path = tempfile.mkstemp(prefix="sglang-agentic-tp-", dir="/dev/shm")
    os.close(fd)
    os.unlink(path)
    return SharedHostStagingLedger(path), path


def _rank_offer(rank: int):
    return {
        "snapshot_id": "request:3",
        "request_id": "request",
        "generation": 3,
        "token_count": 128,
        "token_digest": "tokens",
        "logical_hashes": ["a", "b"],
        "byte_size": 1024,
        "d_pid": 100 + rank,
        "source_numa_node": rank,
        "arena_numa_node": rank,
        "arena_domain": 0,
        "tp_rank": rank,
        "tp_size": 2,
    }


def test_request_generation_key_distinguishes_multi_turn_generations():
    first = request_generation_key("request", 1001)
    second = request_generation_key("request", 1002)
    assert first != second
    assert first == request_generation_key("request", 1001)


def test_decode_rank_zero_emits_only_lifecycle_transitions():
    candidate = {
        "manifest": SimpleNamespace(state=SnapshotState.DIRECT_READY),
        "sent": False,
    }
    manager = SimpleNamespace(
        tp_world_size=2,
        tp_rank=0,
        agentic_direct_candidates={"request:3": candidate},
    )

    first = DecodeKVCacheOffloadManager.tp_candidate_commands(manager)
    assert first == [{"snapshot_id": "request:3", "action": "wait"}]
    assert DecodeKVCacheOffloadManager.tp_candidate_commands(manager) == []

    candidate["manifest"].state = SnapshotState.DIRECT_LOADING
    assert DecodeKVCacheOffloadManager.tp_candidate_commands(manager) == [
        {"snapshot_id": "request:3", "action": "direct"}
    ]


def test_decode_follower_only_installs_rank_zero_command():
    candidate = {"tp_command": "wait"}
    manager = SimpleNamespace(
        tp_world_size=2,
        agentic_direct_candidates={"request:3": candidate},
    )

    DecodeKVCacheOffloadManager.apply_tp_candidate_commands(
        manager, [{"snapshot_id": "request:3", "action": "direct"}]
    )
    assert candidate["tp_command"] == "direct"


def test_tp_host_snapshot_requires_all_rank_offers_grants_and_writes():
    ledger, path = _ledger()
    try:
        first = ledger.offer(_rank_offer(0))
        assert first["state"] == "tp_collecting"
        complete_offer = ledger.offer(_rank_offer(1))
        assert complete_offer["state"] == HostStageState.OFFERED.value
        assert complete_offer["byte_size"] == 2048

        owner = "p-group:p0"
        assert ledger.claim_rank("request:3", owner, tp_rank=0, tp_size=2)
        assert ledger.claim_rank("request:3", owner, tp_rank=1, tp_size=2)
        for rank in range(2):
            assert ledger.publish_rank_grant(
                "request:3",
                owner,
                {
                    "kind": "shared_host_extent",
                    "arena_path": f"/dev/shm/rank-{rank}",
                    "byte_size": 1024,
                    "token_count": 128,
                },
                tp_rank=rank,
                tp_size=2,
            )
        assert ledger.get("request:3")["state"] == HostStageState.HOST_WRITING.value
        assert ledger.complete_host_write(
            "request:3", 100, tp_rank=0, tp_size=2
        )
        assert ledger.get("request:3")["state"] == HostStageState.HOST_WRITING.value
        assert ledger.complete_host_write(
            "request:3", 101, tp_rank=1, tp_size=2
        )
        assert ledger.get("request:3")["state"] == HostStageState.HOST_READY.value

        assert ledger.prepare_tp_host_load_rank(
            "request:3", owner, tp_rank=1, tp_size=2
        )
        assert ledger.prepare_tp_host_load_rank(
            "request:3", owner, tp_rank=0, tp_size=2
        )

        assert ledger.complete_host_load_rank(
            "request:3", owner, tp_rank=0, tp_size=2
        )
        assert ledger.get("request:3")["state"] == HostStageState.H2D_LOADING.value
        assert ledger.complete_host_load_rank(
            "request:3", owner, tp_rank=1, tp_size=2
        )
        assert ledger.get("request:3")["state"] == HostStageState.CONSUMED.value
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def test_tp_host_commit_admits_after_manifest_cleanup():
    """The native commit outlives request-level Host manifest cleanup."""

    request = RequestGeneration("host-commit", 4)
    ledger = SimpleNamespace(get=lambda _snapshot_id: None)
    manager = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        owner="p-group:prefill-0",
        ledger=ledger,
        tp_host_commit_snapshot=request.snapshot_id,
    )
    req = SimpleNamespace(
        rid="child",
        _agentic_host_rank_loaded=True,
        _agentic_host_rank_token_count=256,
    )

    assert AgenticPHostStagingManager.gate_request(manager, req, request) is False
    assert req._agentic_kv_gate_complete is True
    assert req._agentic_kv_host_hit_tokens == 256
    assert req._agentic_tp_bootstrap_snapshot_id == request.snapshot_id
    assert not hasattr(req, "_agentic_host_rank_loaded")


def test_tp_host_h2d_executes_rank_zero_load_command_without_peer_markers():
    request = RequestGeneration("host-start", 2)
    starts = []
    event = SimpleNamespace(query=lambda: False)
    snapshot = SimpleNamespace(
        start_load_range_to_device=lambda *args, **kwargs: starts.append(
            (args, kwargs)
        )
        or (event, [object()])
    )
    record = {
        "snapshot": snapshot,
        "offer": {"token_count": 128, "byte_size": 4096},
        "loading": "h2d_prepared",
    }
    load = {
        "record": record,
        "device_indices": list(range(128)),
        "event": None,
        "copy_refs": None,
        "offset": 0,
        "chunk_end": 0,
        "gpu_elapsed_ms": 0.0,
    }
    manager = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        owner="p-group:prefill-0",
        loads={"child": load},
        ledger=SimpleNamespace(),
        h2d_chunk_tokens=64,
        _h2d_stream=object(),
        _h2d_staging=object(),
        _h2d_host_bounce=object(),
        _get_state_lock=nullcontext,
    )
    req = SimpleNamespace(rid="child")

    assert AgenticPHostStagingManager.gate_request(manager, req, request) is True
    assert len(starts) == 1
    assert load["event"] is event
    assert load["chunk_end"] == 64
    assert record["loading"] == "h2d"


def test_rank_local_numa_configuration(monkeypatch):
    monkeypatch.setenv("TP_NUMAS", "0,1")
    monkeypatch.setenv("LEGACY_NUMA", "7")
    assert rank_env_int("LEGACY_NUMA", "TP_NUMAS", tp_rank=0) == 0
    assert rank_env_int("LEGACY_NUMA", "TP_NUMAS", tp_rank=1) == 1
    assert rank_scoped_arena_directory(
        "/dev/shm/p0", tp_rank=1, tp_size=2, numa_node=1
    ) == "/dev/shm/p0/tp-rank-1-numa-1"
    assert rank_scoped_arena_directory(
        "/dev/shm/p0", tp_rank=0, tp_size=1, numa_node=0
    ) == "/dev/shm/p0"


def test_tp_early_direct_completion_waits_for_rank_local_request():
    """A TP shard may complete before its broadcast Req reaches the queue."""

    snapshot_id = "request:3"
    scheduler = SimpleNamespace(
        agentic_early_direct_receives={
            snapshot_id: SimpleNamespace(completed_at=1.0)
        },
        agentic_early_direct_completion_queue=deque([snapshot_id]),
        agentic_early_direct_poll_lock=nullcontext(),
        agentic_kv_waiting_queue=[],
    )

    Scheduler._agentic_bind_completed_waiters(scheduler)

    assert list(scheduler.agentic_early_direct_completion_queue) == [snapshot_id]


def test_tp_direct_stale_offer_aborts_instead_of_blocking_group():
    """A Direct command selected just before D fallback is terminally stale."""

    request = RequestGeneration("stale-direct", 1)
    store = SimpleNamespace(
        load=lambda _request, require_ready=False: SimpleNamespace(
            state=SnapshotState.SLOW_FALLBACK
        )
    )
    scheduler = SimpleNamespace(
        agentic_early_direct_receives={},
        agentic_tp_direct_local_failed=set(),
        _agentic_snapshot_store=lambda: store,
        _agentic_start_early_direct_receive=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale Direct must not start a receiver")
        ),
    )

    assert not Scheduler._agentic_tp_start_direct_shard(
        scheduler, request, arrived_at=time.time(), prefill_domain=0
    )
    assert scheduler.agentic_tp_direct_local_failed == {request.snapshot_id}


def test_tp_host_timeout_command_forces_the_same_recompute_branch():
    parent = RequestGeneration("request", 1)
    req = SimpleNamespace(
        rid="child",
        sampling_params=SimpleNamespace(
            custom_params={
                CUSTOM_REQUEST_ID: "request",
                CUSTOM_GENERATION: 2,
                CUSTOM_PARENT_GENERATION: 1,
            }
        ),
    )
    scheduler = SimpleNamespace(
        tp_size=2,
        _agentic_tp_host_timeout_snapshot=parent.snapshot_id,
        agentic_early_claim_store=None,
        _agentic_bind_early_direct_receive=lambda *_args, **_kwargs: None,
    )

    assert not Scheduler._agentic_should_defer(scheduler, req, 0.0)
    assert req._agentic_kv_gate_complete
    assert req._agentic_kv_fallback == "timeout:shared_host"


def test_tp_direct_command_precedes_slow_command_then_slow_resumes():
    host_parent = RequestGeneration("host-parent", 1)
    direct_parent = RequestGeneration("direct-parent", 1)

    def child(parent, queue_class):
        return SimpleNamespace(
            rid=f"{parent.request_id}-child",
            _agentic_kv_queue_class=queue_class,
            sampling_params=SimpleNamespace(
                custom_params={
                    CUSTOM_REQUEST_ID: parent.request_id,
                    CUSTOM_GENERATION: 2,
                    CUSTOM_PARENT_GENERATION: parent.generation,
                }
            ),
        )

    host_req = child(host_parent, "slow")
    direct_req = child(direct_parent, "fast")
    visited = []
    scheduler = SimpleNamespace(
        tp_size=2,
        _agentic_tp_selected_snapshot=direct_parent.snapshot_id,
        _agentic_tp_host_selected_snapshot=host_parent.snapshot_id,
        _agentic_tp_host_commit_snapshot=None,
        _agentic_tp_host_timeout_snapshot=None,
        agentic_host_staging_manager=SimpleNamespace(),
        agentic_kv_waiting_queue=[(direct_req, 0.0), (host_req, 0.0)],
        _agentic_bind_completed_waiters=lambda: None,
        _agentic_io_active=lambda _req: False,
        _agentic_io_kind=lambda _req: None,
        _agentic_queue_class=lambda req: req._agentic_kv_queue_class,
        _agentic_should_defer=lambda req, *_args, **_kwargs: visited.append(req.rid)
        or True,
    )

    Scheduler._drain_agentic_kv_waiting_queue(scheduler)
    assert visited == [direct_req.rid]

    # Once the Direct command retires, the selected slow restore resumes.
    scheduler._agentic_tp_selected_snapshot = None
    scheduler._agentic_tp_host_selected_snapshot = host_parent.snapshot_id
    visited.clear()
    scheduler.agentic_kv_waiting_queue = [(direct_req, 0.0), (host_req, 0.0)]
    Scheduler._drain_agentic_kv_waiting_queue(scheduler)
    assert visited == [host_req.rid]


def test_tp_p_ready_is_published_only_by_rank_zero():
    """A follower never creates or resurrects the logical P-ready marker."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        def rank(rank_id):
            return SimpleNamespace(
                tp_size=2,
                tp_rank=rank_id,
                _p_ready_publish_sequence=0,
                disagg_prefill_bootstrap_queue=SimpleNamespace(
                    p_ready_dir=directory
                ),
                _write_p_ready_marker=(
                    lambda req, ready_path, ready_sequence, ready_metadata,
                    rank_id=rank_id: SchedulerDisaggregationPrefillMixin._write_p_ready_marker(
                        schedulers[rank_id],
                        req,
                        ready_path,
                        ready_sequence,
                        ready_metadata,
                    )
                ),
            )

        schedulers = [None, None]
        schedulers[0] = rank(0)
        schedulers[1] = rank(1)
        requests = [
            SimpleNamespace(
                rid="tp-ready",
                bootstrap_room=1234,
                origin_input_ids=list(range(128)),
                disagg_p_ready_notified=False,
            )
            for _ in range(2)
        ]
        method = SchedulerDisaggregationPrefillMixin._publish_deferred_prefill_ready
        ready_path = os.path.join(directory, "1234.ready")
        method(schedulers[0], requests[0])
        assert os.path.exists(ready_path)

        os.unlink(ready_path)
        method(schedulers[1], requests[1])
        assert not os.path.exists(ready_path)


def test_tp_direct_rank_zero_selects_bounded_fifo_group_for_all_ranks(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        marker_store = AgenticEarlyClaimStore(directory)
        arrived_at = time.time()
        requests = [RequestGeneration("first", 1), RequestGeneration("second", 2)]
        payloads = [{"arrived_at": arrived_at + index * 0.001} for index in range(2)]
        manifests = {
            request.snapshot_id: SimpleNamespace(
                request=request,
                state=SnapshotState.DIRECT_READY,
                created_at=payload["arrived_at"],
                token_count=1024,
            )
            for request, payload in zip(requests, payloads)
        }
        snapshot_store = SimpleNamespace(
            load=lambda request, require_ready=False: manifests[request.snapshot_id]
        )

        def scheduler(rank):
            value = SimpleNamespace(
                tp_size=2,
                tp_rank=rank,
                agentic_early_claim_store=marker_store,
                agentic_tp_direct_admission_active={},
                agentic_early_direct_admission_queue=deque(
                    (request, payload, manifests[request.snapshot_id])
                    for request, payload in zip(requests, payloads)
                ),
                agentic_early_direct_admission_ids={
                    request.snapshot_id for request in requests
                },
                agentic_early_direct_receives={},
                agentic_early_direct_terminal={},
                agentic_direct_credit_pool=SimpleNamespace(free_tokens=40000),
                server_args=SimpleNamespace(page_size=64),
                started=[],
            )
            value._agentic_start_early_direct_receive = (
                lambda request, *_args, **_kwargs: value.started.append(
                    request.snapshot_id
                )
                or True
            )
            return value

        rank0 = scheduler(0)
        rank1 = scheduler(1)
        method = Scheduler._agentic_admit_queued_direct_receives

        # A follower cannot make an independent choice.
        method(rank1, snapshot_store, 2.0, nullcontext())
        assert rank1.started == []

        method(rank0, snapshot_store, 2.0, nullcontext())
        # Selection is rank-zero-only, but no shard starts before the native
        # TP command is visible to the whole group.
        assert rank0.started == []
        assert list(rank0.agentic_tp_direct_admission_active) == [
            request.snapshot_id for request in requests
        ]
        assert [
            active[0]
            for active in rank0.agentic_tp_direct_admission_active.values()
        ] == requests
        method(rank1, snapshot_store, 2.0, nullcontext())
        assert rank1.started == []
        method(rank0, snapshot_store, 2.0, nullcontext())
        assert rank0.started == []
        assert not rank0.agentic_early_direct_admission_queue


def test_tp_decode_release_uses_native_scheduler_control():
    """Decode release is broadcast at the existing scheduler boundary."""

    released = []
    manager = SimpleNamespace(
        _agentic_tp_pending_releases={"request:3": object()},
        tp_pending_release_snapshot=lambda: "request:3",
        commit_tp_release=lambda snapshot_id: released.append(snapshot_id),
    )
    scheduler = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        disaggregation_mode=DisaggregationMode.DECODE,
        decode_offload_manager=manager,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
    )

    control = Scheduler._agentic_tp_prepare_admission_control(scheduler)
    assert control == {
        Scheduler._AGENTIC_TP_CONTROL_KEY: True,
        "decode_release_snapshot": "request:3",
        "decode_admit_keys": [],
        "decode_transfer_keys": [],
        "decode_transfer_rid": None,
        "decode_transfer_room": None,
        "decode_agentic_commands": [],
    }
    ordinary = object()
    assert Scheduler._agentic_tp_consume_admission_control(
        scheduler, [ordinary, control]
    ) == [ordinary]
    assert released == ["request:3"]


def test_tp_pending_release_accounts_only_uncached_tail():
    """Radix already owns the prefix while native TP release is pending."""

    req = SimpleNamespace(
        kv_allocated_len=4097,
        kv_committed_len=4097,
        cache_protected_len=2048,
        req_pool_idx=7,
    )
    manager = SimpleNamespace(
        page_size=64,
        _decode_pending_release_tokens=64,
        _agentic_tp_pending_releases={"request:3": (req, 0)},
    )
    reserved = DecodeKVCacheOffloadManager.agentic_pending_release_token_count.fget(
        manager
    )
    assert reserved == 64 + 4160 - 2048
    assert (
        DecodeKVCacheOffloadManager.agentic_pending_release_req_count.fget(manager)
        == 1
    )


def test_tp_decode_release_can_resolve_peer_live_candidate():
    """A peer need not have polled terminal state before the native commit."""

    snapshot_id = "request:4"
    req = SimpleNamespace(rid="request", req_pool_idx=7)
    released = []
    cleaned = []
    claims = []
    manager = SimpleNamespace(
        agentic_direct_candidates={snapshot_id: {"req": req}},
        _release_finished_req=lambda value, offset: released.append(
            (value, offset)
        ),
        _cleanup_agentic_direct_sender=lambda candidate: cleaned.append(candidate),
        _agentic_release_early_claim=lambda candidate, reason: claims.append(
            (candidate, reason)
        ),
    )

    DecodeKVCacheOffloadManager.commit_tp_release(manager, snapshot_id)
    assert released == [(req, 0)]
    assert len(cleaned) == 1
    assert claims[0][1] == "tp_release_commit"
    assert manager.agentic_direct_candidates == {}


def test_tp_p2d_peer_claim_suppresses_rank_local_native_completion():
    """One Host claim keeps every TP shard on the same P->D path."""

    req = SimpleNamespace(bootstrap_room=123)
    manager = SimpleNamespace(
        tp_size=2,
        prefill_domain=1,
        owner="p-group:1",
        _targets_this_p=lambda entry: int(entry["prefill_domain"]) == 1,
        ledger=SimpleNamespace(
            get=lambda _snapshot_id: {
                "state": HostStageState.HOST_RESERVED.value,
                "prefill_domain": 1,
                "p_owner": "p-group:1",
            }
        ),
    )
    manager.group_claimed = lambda value: (
        AgenticPToDHostStagingManager.group_claimed(manager, value)
    )

    assert AgenticPToDHostStagingManager.group_claimed(manager, req)
    assert (
        AgenticPToDHostStagingManager.poll(manager, req)
        == int(KVPoll.Transferring)
    )
