import os
import json
import tempfile
import threading
import time
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import sglang.srt.disaggregation.prefill as prefill_module

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
from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostStagingManager,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import (
    AgenticDirectPageCreditPool,
    Scheduler,
)


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


def test_direct_credit_promotes_received_pages_and_replenishes_transit_pool():
    class Allocator:
        def __init__(self):
            self.next_index = 0

        def alloc(self, count):
            result = torch.arange(
                self.next_index, self.next_index + count, dtype=torch.int64
            )
            self.next_index += count
            return result

        def free(self, _indices):
            pass

    allocator = Allocator()
    pool = AgenticDirectPageCreditPool(
        allocator, capacity_tokens=8, page_size=4
    )
    allocation = pool.allocate(4)
    assert allocation is not None
    received_indices = pool.device_view(allocation).clone()
    assert pool.free_tokens == 4

    replacement = allocator.alloc(allocation.allocated_tokens)
    pool.promote_to_ordinary(allocation, replacement)

    # The received KV stays on its original physical pages, while the fixed
    # Direct slots immediately become reusable through ordinary empty pages.
    assert torch.equal(received_indices, torch.arange(0, 4))
    assert pool.free_tokens == 8
    next_allocation = pool.allocate(4)
    assert next_allocation is not None
    assert torch.equal(pool.device_view(next_allocation), replacement)


def test_request_generation_key_distinguishes_multi_turn_generations():
    first = request_generation_key("request", 1001)
    second = request_generation_key("request", 1002)
    assert first != second
    assert first == request_generation_key("request", 1001)


def test_tp_mailbox_reports_complete_generation_without_collective():
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        ranks = [
            TPGroupMailbox(
                "test", tp_rank=rank, tp_size=2, directory=directory
            )
            for rank in range(2)
        ]
        key = request_generation_key("request", 1001)
        ranks[1].publish_local(key, int(KVPoll.Success))
        assert ranks[0].group_status(key) is None
        ranks[0].publish_local(key, int(KVPoll.Success))
        assert ranks[0].group_status(key) == int(KVPoll.Success)
        # Cached reads must still observe an atomic replacement with a new
        # state, while publishing an unchanged state remains a no-op.
        ranks[1].publish_local(key, int(KVPoll.Failed))
        assert ranks[0].group_status(key) == int(KVPoll.Failed)
        ranks[1].publish_local(key, int(KVPoll.Failed))
        assert ranks[0].group_status(key) == int(KVPoll.Failed)
        ranks[0].publish_receipt(key, int(KVPoll.Success))
        assert ranks[1].receipt(key) == int(KVPoll.Success)

        # A later generation of the same agent has independent state.
        assert ranks[0].group_status(request_generation_key("request", 1002)) is None


def test_tp_direct_progress_is_monotonic_until_explicit_clear():
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        mailbox = TPGroupMailbox(
            "direct-progress", tp_rank=0, tp_size=1, directory=directory
        )
        key = "request:1"
        mailbox.publish_local_progress(key, 3)
        mailbox.publish_local_progress(key, 4)
        mailbox.publish_local_progress(key, 3)
        assert mailbox.local_status(key) == 4

        mailbox.publish_local_progress(key, -1)
        mailbox.publish_local_progress(key, 4)
        assert mailbox.local_status(key) == -1

        mailbox.clear_group(key)
        mailbox.publish_local_progress(key, 1)
        assert mailbox.local_status(key) == 1


def test_decode_receiver_lifecycle_lock_serializes_poll_and_clear():
    poll_started = threading.Event()
    allow_poll_to_finish = threading.Event()

    class Receiver:
        def __init__(self):
            self.cleared = False

        def poll(self):
            poll_started.set()
            assert allow_poll_to_finish.wait(timeout=2.0)
            assert not self.cleared
            return KVPoll.Success

        def clear(self):
            self.cleared = True

    receiver = Receiver()
    request = SimpleNamespace(
        bootstrap_room=7,
        bootstrap_host="127.0.0.1",
        output_ids=[],
        return_logprob=False,
        time_stats=SimpleNamespace(set_wait_queue_entry_time=lambda: None),
    )
    decode_req = SimpleNamespace(
        req=request,
        kv_receiver=receiver,
        metadata_buffer_index=0,
    )
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.queue = [decode_req]
    queue.enable_staging = False
    queue._async_progress_enabled = True
    queue._async_poll_lock = threading.Lock()
    queue.scheduler = SimpleNamespace(tp_size=1, server_args=SimpleNamespace())
    queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    queue.metadata_buffers = SimpleNamespace(
        get_buf=lambda _index: (
            torch.tensor([1]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0.0]),
            torch.tensor([0]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([7]),
        )
    )

    poll_thread = threading.Thread(target=queue.background_progress)
    poll_thread.start()
    assert poll_started.wait(timeout=2.0)

    # Decode's scheduler thread must not block behind a slow transport poll.
    started_at = time.monotonic()
    assert queue.pop_transferred() == []
    assert time.monotonic() - started_at < 0.1

    committed = threading.Event()

    def commit():
        assert queue._commit_transfer_to_req(decode_req)
        committed.set()

    commit_thread = threading.Thread(target=commit)
    commit_thread.start()
    time.sleep(0.02)
    assert not committed.is_set()
    allow_poll_to_finish.set()
    poll_thread.join(timeout=2.0)
    commit_thread.join(timeout=2.0)

    assert committed.is_set()
    assert receiver.cleared
    assert decode_req.kv_receiver is None


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


def test_decode_follower_retains_command_until_local_candidate_exists():
    """A one-shot TP0 transition must survive follower publication skew."""

    manager = SimpleNamespace(
        tp_world_size=2,
        agentic_direct_candidates={},
        _agentic_tp_pending_candidate_commands={},
    )
    command = {"snapshot_id": "request:3", "action": "direct"}

    DecodeKVCacheOffloadManager.apply_tp_candidate_commands(manager, [command])
    assert manager._agentic_tp_pending_candidate_commands == {
        "request:3": command
    }

    candidate = {"tp_command": "wait"}
    manager.agentic_direct_candidates["request:3"] = candidate
    assert DecodeKVCacheOffloadManager._apply_tp_candidate_command(
        manager,
        manager._agentic_tp_pending_candidate_commands["request:3"],
    )
    assert candidate["tp_command"] == "direct"
    assert manager._agentic_tp_pending_candidate_commands == {}


def test_slow_path_selects_lowest_pressure_logical_prefill(monkeypatch):
    with tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False) as f:
        json.dump(
            {
                "published_at": time.time(),
                "domains": [
                    {
                        "domain": 0,
                        "pending_tokens": 30000,
                        "hbm_used_tokens": 80000,
                        "hbm_capacity_tokens": 100000,
                        "arena_used_bytes": 80,
                        "arena_capacity_bytes": 100,
                        "pending_requests": 10,
                        "scheduler_waiting": 10,
                    },
                    {
                        "domain": 1,
                        "pending_tokens": 5000,
                        "hbm_used_tokens": 20000,
                        "hbm_capacity_tokens": 100000,
                        "arena_used_bytes": 10,
                        "arena_capacity_bytes": 100,
                        "pending_requests": 1,
                        "scheduler_waiting": 1,
                    },
                ],
            },
            f,
        )
        path = f.name
    try:
        monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "1")
        monkeypatch.setenv("SGLANG_AGENTIC_KV_PREFILL_LOAD_PATH", path)
        monkeypatch.setenv("SGLANG_AGENTIC_KV_PREFILL_TP_NUMA_DOMAINS", "0,1;0,1")
        manager = SimpleNamespace(
            tp_world_size=2,
            agentic_host_staging_client=SimpleNamespace(
                arena_domain=0, arena_numa_node=0
            ),
        )
        manager._prefill_domain_numa_nodes = lambda domain: (
            DecodeKVCacheOffloadManager._prefill_domain_numa_nodes(
                manager, domain
            )
        )
        domain, numa_nodes = DecodeKVCacheOffloadManager._select_slow_prefill_domain(
            manager
        )
        assert domain == 1
        assert numa_nodes == [0, 1]
    finally:
        os.unlink(path)


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
        assert ledger.get("request:3")["state"] == HostStageState.HOST_READY.value
        assert ledger.prepare_tp_host_load_rank(
            "request:3", owner, tp_rank=0, tp_size=2
        )
        assert ledger.get("request:3")["state"] == HostStageState.H2D_LOADING.value

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


def test_tp_p2d_host_commit_is_order_independent():
    """A fast shard may finish before its peer has published a grant."""

    ledger, path = _ledger()
    snapshot_id = "p2d:41"
    owner = "p2d-p-group:prefill-0"
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 41,
                "token_count": 128,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": 2,
            }
        )
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=0, tp_size=2)
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=1, tp_size=2)

        assert ledger.publish_rank_grant(
            snapshot_id,
            owner,
            {
                "kind": "shared_host_extent",
                "arena_path": "/dev/shm/p2d-rank-0",
                "byte_size": 1024,
                "token_count": 128,
            },
            tp_rank=0,
            tp_size=2,
        )
        assert ledger.complete_p2d_host_write_rank(
            snapshot_id, owner, tp_rank=0, tp_size=2
        )
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.HOST_RESERVED.value
        assert current["writer_acks"] == [0]

        assert ledger.publish_rank_grant(
            snapshot_id,
            owner,
            {
                "kind": "shared_host_extent",
                "arena_path": "/dev/shm/p2d-rank-1",
                "byte_size": 1024,
                "token_count": 128,
            },
            tp_rank=1,
            tp_size=2,
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.HOST_WRITING.value
        assert ledger.complete_p2d_host_write_rank(
            snapshot_id, owner, tp_rank=1, tp_size=2
        )
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.HOST_READY.value
        assert current["writer_acks"] == [0, 1]
        assert len(current["grants"]) == 2
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


def test_tp_host_h2d_progresses_on_independent_worker_after_group_prepare():
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
        "request_generation": request,
        "device_indices": list(range(128)),
        "event": None,
        "copy_refs": None,
        "offset": 0,
        "chunk_end": 0,
        "gpu_elapsed_ms": 0.0,
        "start_allowed": False,
        "io_complete": False,
    }
    manager = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        owner="p-group:prefill-0",
        loads={"child": load},
        ledger=SimpleNamespace(
            get=lambda _snapshot_id: {"state": HostStageState.H2D_LOADING.value}
        ),
        h2d_chunk_tokens=64,
        _h2d_stream=object(),
        _h2d_staging=object(),
        _h2d_host_bounce=object(),
        _get_state_lock=nullcontext,
    )
    manager._start_h2d_chunk = lambda selected: (
        AgenticPHostStagingManager._start_h2d_chunk(manager, selected)
    )
    manager._release_completed_h2d_host = lambda selected: False
    req = SimpleNamespace(rid="child")

    # PREPARE may reserve pages, but it must not let a fast rank launch H2D
    # before TP0 has observed every rank's prepared ACK.
    assert (
        AgenticPHostStagingManager.gate_request(
            manager, req, request, allow_prepare=True, allow_start=False
        )
        is True
    )
    assert starts == []

    # Authorization only wakes the independent Slow I/O queue; the scheduler
    # does not launch or poll CUDA work itself.
    assert (
        AgenticPHostStagingManager.gate_request(
            manager, req, request, allow_prepare=True, allow_start=True
        )
        is True
    )
    assert starts == []
    AgenticPHostStagingManager._progress_h2d_loads(manager)
    assert len(starts) == 1
    assert load["event"] is event
    assert load["chunk_end"] == 64
    assert record["loading"] == "h2d"


def test_tp_host_h2d_failure_fails_group_and_recomputes_without_leak():
    request = RequestGeneration("host-failure", 2)
    state = {"value": HostStageState.H2D_LOADING.value}

    class Ledger:
        def get(self, _snapshot_id):
            return {"state": state["value"]}

        def transition(self, _snapshot_id, target, **_kwargs):
            state["value"] = target.value
            return True

    freed_device = []
    released_host = []
    record = {
        "snapshot": object(),
        "offer": {"token_count": 128, "byte_size": 4096},
        "loading": "h2d_prepared",
    }
    load = {
        "record": record,
        "request_generation": request,
        "device_indices": [1, 2],
        "event": None,
        "copy_refs": None,
        "offset": 0,
        "chunk_end": 0,
        "gpu_elapsed_ms": 0.0,
        "start_allowed": True,
        "io_complete": False,
        "host_released": False,
    }
    manager = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        owner="p-group:prefill-0",
        ledger=Ledger(),
        loads={"child": load},
        host_ready={},
        active={},
        aborting={},
        token_allocator=SimpleNamespace(
            free=lambda indices: freed_device.append(tuple(indices))
        ),
        _get_state_lock=nullcontext,
        _start_h2d_chunk=lambda _load: (_ for _ in ()).throw(
            RuntimeError("injected H2D failure")
        ),
        _release_completed_h2d_host=lambda _load: False,
        _release_record=lambda selected: released_host.append(selected),
    )
    manager._discard_failed_h2d_load = lambda rid, selected: (
        AgenticPHostStagingManager._discard_failed_h2d_load(
            manager, rid, selected
        )
    )

    AgenticPHostStagingManager._progress_h2d_loads(manager)
    assert state["value"] == HostStageState.FAILED.value
    assert isinstance(load["io_error"], RuntimeError)

    req = SimpleNamespace(rid="child")
    assert AgenticPHostStagingManager.gate_request(manager, req, request) is False
    assert req._agentic_kv_gate_complete is True
    assert req._agentic_kv_fallback == "shared_host_h2d_failed"
    assert manager.loads == {}
    assert freed_device == [(1, 2)]
    assert released_host == [record]


def test_tp_host_group_state_machine_has_an_explicit_prepare_barrier():
    assert Scheduler._agentic_tp_host_next_action(0) == "prepare"
    assert Scheduler._agentic_tp_host_next_action(1) == "start"
    assert Scheduler._agentic_tp_host_next_action(2) == "bind"
    assert Scheduler._agentic_tp_host_next_action(3) == "commit"
    assert Scheduler._agentic_tp_host_next_action(4) == "clear"


def test_tp_host_completed_dma_waits_for_group_bind_command():
    """A fast Slow shard cannot enter Radix before every TP shard is ready."""

    request = RequestGeneration("host-bind", 1)
    manager = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        loads={
            "child": {
                "io_error": None,
                "io_complete": True,
            }
        },
        _get_state_lock=nullcontext,
    )
    req = SimpleNamespace(rid="child")

    assert (
        AgenticPHostStagingManager.gate_request(
            manager,
            req,
            request,
            allow_prepare=True,
            allow_start=True,
            allow_bind=False,
        )
        is True
    )
    assert not hasattr(req, "_agentic_host_rank_loaded")


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


def test_tp_direct_and_slow_group_commands_progress_together():
    """Independent Direct and Slow ownership transitions do not starve."""

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


def test_tp_direct_worker_defers_failed_page_release_to_owner_scheduler():
    """A TP ingress worker must not free GPU pages outside the model loop."""

    request = RequestGeneration("request", 7)
    entry = SimpleNamespace(
        request=request,
        completed_at=None,
        transport_poll=KVPoll.Failed,
        started_at=time.monotonic(),
        receiver=SimpleNamespace(),
    )
    scheduler = object.__new__(Scheduler)
    scheduler.tp_size = 2
    scheduler.agentic_early_claim_store = object()
    scheduler.agentic_direct_runtime = object()
    scheduler.agentic_early_direct_poll_lock = nullcontext()
    scheduler.agentic_early_direct_receives = {request.snapshot_id: entry}
    scheduler.agentic_early_direct_terminal = {}
    scheduler.agentic_tp_direct_local_failed = set()
    scheduler._agentic_snapshot_store = lambda: object()
    scheduler._agentic_collect_direct_arrivals = lambda _lock: None
    scheduler._agentic_admit_queued_direct_receives = (
        lambda _store, _timeout, _lock: None
    )
    scheduler._agentic_commit_tp_direct_groups = lambda _store: None
    scheduler._agentic_drop_early_direct_receive = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(AssertionError("worker freed TP GPU pages"))
    )

    Scheduler._agentic_poll_early_direct_receives_once(
        scheduler, now=time.monotonic()
    )

    assert scheduler.agentic_tp_direct_local_failed == {request.snapshot_id}
    assert scheduler.agentic_early_direct_receives[request.snapshot_id] is entry


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
    assert visited == [direct_req.rid, host_req.rid]

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


def test_tp_prefill_producer_reports_local_payload_before_logical_ready():
    """Each TP shard reports preparation; enqueue never creates P-ready."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        mailboxes = [
            TPGroupMailbox(
                "p2d-producer", tp_rank=rank, tp_size=2, directory=directory
            )
            for rank in range(2)
        ]
        requests = [
            SimpleNamespace(
                rid="tp-producer",
                bootstrap_room=4321,
                disagg_p_ready_deferred=True,
                disagg_p_ready_notified=False,
                disagg_p_ready_transfer_started=False,
                _async_prefill_transfer_payload=(2, [1, 2], None),
            )
            for _ in range(2)
        ]

        for rank in range(2):
            scheduler = SimpleNamespace(
                tp_size=2,
                agentic_tp_p2d_sender_mailbox=mailboxes[rank],
                _prefill_ready_condition=threading.Condition(),
                _prefill_ready_queue=deque(),
                _prefill_ready_queued_keys=set(),
                _p_ready_publish_sequence=0,
            )
            scheduler._prefill_transfer_key = (
                SchedulerDisaggregationPrefillMixin._prefill_transfer_key
            )
            scheduler._report_tp_prefill_producer_ready = lambda req: (
                SchedulerDisaggregationPrefillMixin._report_tp_prefill_producer_ready(
                    scheduler, req
                )
            )
            scheduler._prefill_queued_keys = lambda: (
                SchedulerDisaggregationPrefillMixin._prefill_queued_keys(scheduler)
            )
            assert SchedulerDisaggregationPrefillMixin._enqueue_deferred_prefill_transfer(
                scheduler, requests[rank]
            )
            assert len(scheduler._prefill_ready_queue) == 1
            assert requests[rank].disagg_p_ready_notified is False

        key = request_generation_key("tp-producer", 4321)
        assert mailboxes[0].group_status(key) == int(KVPoll.Bootstrapping)


def test_tp_prefill_worker_activation_preserves_producer_sequence():
    """Activating the bounded sender must not create a P-ready FIFO hole."""

    request = SimpleNamespace(
        rid="tp-sequence",
        bootstrap_room=4322,
        disagg_p_ready_deferred=True,
        disagg_p_ready_notified=True,
        disagg_p_ready_transfer_started=False,
        _async_prefill_transfer_payload=(2, [1, 2], None),
        _p_ready_sequence=7,
    )
    scheduler = SimpleNamespace(
        tp_size=2,
        _prefill_ready_condition=threading.Condition(),
        _prefill_ready_queue=deque(),
        _prefill_ready_queued_keys=set(),
        _p_ready_publish_sequence=8,
        _prefill_transfer_key=(
            SchedulerDisaggregationPrefillMixin._prefill_transfer_key
        ),
        _report_tp_prefill_producer_ready=lambda _req: None,
    )
    scheduler._prefill_queued_keys = lambda: (
        SchedulerDisaggregationPrefillMixin._prefill_queued_keys(scheduler)
    )

    assert SchedulerDisaggregationPrefillMixin._enqueue_deferred_prefill_transfer(
        scheduler, request
    )
    assert request._p_ready_sequence == 7
    assert scheduler._p_ready_publish_sequence == 8


def test_tp_prefill_batch_control_preserves_identical_order_on_all_ranks():
    control = {
        Scheduler._AGENTIC_TP_CONTROL_KEY: True,
        "direct_commands": [],
        "prefill_transfer_keys": [("first", 10), ("second", 20)],
        "prefill_transfer_statuses": [
            int(KVPoll.WaitingForInput),
            int(KVPoll.WaitingForInput),
        ],
        "prefill_submit_keys": [("first", 10), ("second", 20)],
        "host_snapshot": None,
        "host_action": None,
        "host_timeout_snapshot": None,
    }

    def rank(rank_id):
        return SimpleNamespace(
            tp_size=2,
            tp_rank=rank_id,
            disaggregation_mode=DisaggregationMode.PREFILL,
            _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
            agentic_tp_direct_admission_active={},
            agentic_tp_direct_group_status={},
            agentic_tp_direct_local_admitted=set(),
            agentic_tp_direct_local_failed=set(),
            agentic_early_direct_receives={},
            agentic_tp_direct_visible_order=[],
            agentic_tp_direct_command_visible=False,
            agentic_tp_host_local_admitted=set(),
            agentic_tp_host_active=None,
            agentic_tp_host_active_since=0.0,
            agentic_tp_host_command_visible=False,
            agentic_tp_host_group_status=0,
            agentic_host_staging_manager=None,
        )

    schedulers = [rank(0), rank(1)]
    for scheduler in schedulers:
        ordinary = Scheduler._agentic_tp_consume_admission_control(
            scheduler, [dict(control)]
        )
        assert ordinary == []
        assert scheduler._agentic_tp_prefill_submit_keys == [
            ("first", 10),
            ("second", 20),
        ]
        assert scheduler._agentic_tp_prefill_transfer_group_status == {
            ("first", 10): int(KVPoll.WaitingForInput),
            ("second", 20): int(KVPoll.WaitingForInput),
        }


def test_tp_prefill_batch_submits_each_rank_shard_exactly_once():
    calls = [[], []]

    class Sender:
        def __init__(self, rank):
            self.rank = rank

        def init(self, pages, metadata_index):
            calls[self.rank].append(("init", pages, metadata_index))

        def send(self, page_indices, state_indices):
            calls[self.rank].append(
                ("send", tuple(page_indices), tuple(state_indices))
            )

    for rank in range(2):
        request = SimpleNamespace(
            rid="batch-submit",
            bootstrap_room=99,
            metadata_buffer_index=7,
            disagg_p_ready_transfer_started=False,
            disagg_kv_sender=Sender(rank),
            _async_prefill_transfer_payload=(2, [11, 12], [21, 22]),
            time_stats=SimpleNamespace(
                set_prefill_transfer_queue_entry_time=lambda: None
            ),
        )
        scheduler = SimpleNamespace(
            tp_size=2,
            tp_rank=rank,
            agentic_tp_p2d_sender_mailbox=SimpleNamespace(
                publish_local=lambda *_args: None
            ),
            _prefill_transfer_key=(
                SchedulerDisaggregationPrefillMixin._prefill_transfer_key
            ),
        )
        submit = SchedulerDisaggregationPrefillMixin._submit_tp_prefill_transfer
        assert submit(scheduler, request)
        assert not submit(scheduler, request)

    assert calls[0] == calls[1]
    assert [call[0] for call in calls[0]] == ["init", "send"]


def test_tp_prefill_failure_releases_generation_and_control_state(monkeypatch):
    releases = []
    branch_releases = []
    sender_clears = []
    host_clears = []
    mailbox_clears = []
    monkeypatch.setattr(
        prefill_module,
        "release_kv_cache",
        lambda req, _tree, **kwargs: releases.append((req.rid, kwargs)),
    )
    request = SimpleNamespace(
        rid="failed-transfer",
        bootstrap_room=77,
        origin_input_ids=list(range(128)),
        disagg_kv_sender=SimpleNamespace(clear=lambda: sender_clears.append(True)),
        _agentic_p2d_host_snapshot_id="failed-transfer:1",
        _async_prefill_transfer_payload=(2, [1, 2], [3, 4]),
    )
    scheduler = SimpleNamespace(
        tree_cache=SimpleNamespace(
            release_agentic_request_cache=lambda req, committed_len: (
                branch_releases.append((req.rid, committed_len))
            )
        ),
        _clear_tp_prefill_transfer_mailboxes=lambda req: mailbox_clears.append(
            req.rid
        ),
    )
    p2d_host = SimpleNamespace(
        mark_scheduler_consumed=lambda req: host_clears.append(req.rid)
    )

    SchedulerDisaggregationPrefillMixin._cleanup_failed_prefill_transfer(
        scheduler,
        request,
        p2d_host,
        SimpleNamespace(),
    )

    assert releases == [("failed-transfer", {"is_insert": False})]
    assert branch_releases == [("failed-transfer", 128)]
    assert sender_clears == [True]
    assert host_clears == ["failed-transfer"]
    assert mailbox_clears == ["failed-transfer"]
    assert not hasattr(request, "_async_prefill_transfer_payload")


def test_tp_p_ready_out_of_order_publish_is_nonblocking():
    """TP0 retries a FIFO gap instead of blocking the native TP broadcast."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        scheduler = SimpleNamespace(
            tp_size=2,
            tp_rank=0,
            _p_ready_publish_sequence=2,
            _prefill_ready_next_publish_sequence=0,
            _prefill_ready_publish_condition=threading.Condition(),
            _prefill_transfer_stop=threading.Event(),
            disagg_prefill_bootstrap_queue=SimpleNamespace(p_ready_dir=directory),
        )
        scheduler._write_p_ready_marker = (
            lambda req, ready_path, ready_sequence, ready_metadata: (
                SchedulerDisaggregationPrefillMixin._write_p_ready_marker(
                    scheduler,
                    req,
                    ready_path,
                    ready_sequence,
                    ready_metadata,
                )
            )
        )
        request = SimpleNamespace(
            rid="tp-gap",
            bootstrap_room=4323,
            origin_input_ids=[1, 2],
            disagg_p_ready_notified=False,
            _p_ready_sequence=1,
        )

        started = time.monotonic()
        SchedulerDisaggregationPrefillMixin._publish_deferred_prefill_ready(
            scheduler, request
        )
        assert time.monotonic() - started < 0.1
        assert request.disagg_p_ready_notified is False
        assert not os.path.exists(os.path.join(directory, "4323.ready"))


def test_tp_direct_rank0_background_grant_starts_all_followers(monkeypatch):
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

        mailboxes = [
            TPGroupMailbox(
                "direct-background-grant",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]

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
                agentic_tp_direct_mailbox=mailboxes[rank],
                agentic_tp_direct_local_failed=set(),
                agentic_tp_direct_local_admitted=set(),
                server_args=SimpleNamespace(page_size=64),
                started=[],
            )
            def start(request, *_args, **_kwargs):
                value.started.append(request.snapshot_id)
                return True

            value._agentic_tp_start_direct_shard = start
            return value

        rank0 = scheduler(0)
        rank1 = scheduler(1)
        method = Scheduler._agentic_admit_queued_direct_receives

        # A follower may observe the Router marker first, but cannot make an
        # independent admission decision before TP0 grants that generation.
        method(rank1, snapshot_store, 2.0, nullcontext())
        assert rank1.started == []
        assert len(rank1.agentic_early_direct_admission_queue) == 2

        # TP0 grants every request that fits the Direct reserve without a
        # model-scheduler broadcast.
        method(rank0, snapshot_store, 2.0, nullcontext())
        assert rank0.started == []
        assert list(rank0.agentic_tp_direct_admission_active) == [
            request.snapshot_id for request in requests
        ]
        assert [
            active[0]
            for active in rank0.agentic_tp_direct_admission_active.values()
        ] == requests
        assert not rank0.agentic_early_direct_admission_queue

        Scheduler._agentic_progress_tp_direct_grants(rank0, snapshot_store)
        assert rank0.started == [request.snapshot_id for request in requests]

        # The follower mirrors the exact TP0 grants and starts the same FIFO
        # from its own background worker.
        method(rank1, snapshot_store, 2.0, nullcontext())
        Scheduler._agentic_progress_tp_direct_grants(rank1, snapshot_store)
        assert rank1.started == [request.snapshot_id for request in requests]
        assert list(rank1.agentic_tp_direct_admission_active) == [
            request.snapshot_id for request in requests
        ]
        assert not rank1.agentic_early_direct_admission_queue


def test_tp_direct_group_completion_wakes_router_without_scheduler_tick():
    request = RequestGeneration("direct-complete", 4)
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        mailboxes = [
            TPGroupMailbox(
                "direct-complete-test",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]
        for mailbox in mailboxes:
            mailbox.publish_local(request.snapshot_id, 3)
        completed = SimpleNamespace(state=SnapshotState.CONSUMED)
        commits = []
        store = SimpleNamespace(
            complete_direct_group=lambda manifest, claim_id: (
                commits.append((manifest, claim_id)) or completed
            )
        )
        entry = SimpleNamespace(
            completed_at=time.monotonic(),
            group_committed=False,
            manifest=SimpleNamespace(token_count=1024),
            claim_id="claim",
            prefill_domain=None,
            route_published=False,
            request=request,
            arrived_at=time.time(),
        )
        scheduler = SimpleNamespace(
            tp_rank=0,
            agentic_tp_direct_mailbox=mailboxes[0],
            agentic_early_direct_poll_lock=nullcontext(),
            agentic_tp_direct_admission_active={request.snapshot_id: object()},
            agentic_early_direct_receives={request.snapshot_id: entry},
            agentic_tp_direct_local_failed=set(),
            agentic_tp_direct_local_admitted=set(),
        )

        Scheduler._agentic_commit_tp_direct_groups(scheduler, store)

        assert commits == [(entry.manifest, "claim")]
        assert entry.group_committed
        assert mailboxes[1].receipt(request.snapshot_id) == 3


def test_tp_direct_rank_init_failure_never_releases_group_claim():
    request = RequestGeneration("follower-init-failure", 2)
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
        state=SnapshotState.DIRECT_LOADING,
        claim_id="direct-early-tp:p0:follower-init-failure:2",
        tp_size=2,
        token_count=64,
        kv_layout_hash="layout",
        direct_bootstrap_addr="127.0.0.1:1",
    )
    released_claims = []
    released_credits = []
    allocation = SimpleNamespace(page_indices=torch.arange(64))
    store = SimpleNamespace(
        claim_direct=lambda *_args: manifest,
        load=lambda *_args, **_kwargs: manifest,
        release_direct_claim=lambda *_args: released_claims.append(_args),
    )
    for rank in range(2):
        scheduler = SimpleNamespace(
            tp_size=2,
            tp_rank=rank,
            tree_cache=SimpleNamespace(is_eagle=False),
            agentic_direct_runtime=SimpleNamespace(
                layout_hash="layout",
                manager=SimpleNamespace(
                    try_ensure_parallel_info=lambda *_args: False
                ),
                receiver_class=None,
            ),
            agentic_direct_credit_pool=SimpleNamespace(
                allocate=lambda _tokens: allocation,
                release=lambda value: released_credits.append(value),
            ),
            agentic_direct_poll_requested=None,
            agentic_nixl_control_lock=nullcontext(),
            agentic_early_direct_poll_lock=nullcontext(),
            agentic_early_direct_receives={},
            agentic_tp_direct_local_failed=set(),
        )

        assert not Scheduler._agentic_start_early_direct_receive(
            scheduler,
            request,
            manifest,
            store,
            arrived_at=time.time(),
        )
        assert request.snapshot_id not in scheduler.agentic_early_direct_receives
        assert request.snapshot_id not in scheduler.agentic_tp_direct_local_failed

    assert released_credits == [allocation, allocation]
    assert released_claims == []


def test_tp_direct_scheduler_does_not_timeout_background_start(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "p0")
    request = RequestGeneration("start-timeout", 3)
    arrived_at = time.time() - 60.0
    released = []
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
        state=SnapshotState.DIRECT_LOADING,
        claim_id="direct-early-tp:p0:start-timeout:3",
    )
    store = SimpleNamespace(
        load=lambda *_args, **_kwargs: manifest,
        release_direct_claim=lambda current, claim_id: released.append(
            (current.snapshot_id, claim_id)
        ),
    )
    active = {request.snapshot_id: (request, arrived_at, None, 1024)}
    owner = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        disaggregation_mode=DisaggregationMode.PREFILL,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
        agentic_tp_direct_admission_active=dict(active),
        agentic_tp_direct_group_status={request.snapshot_id: 0},
        agentic_early_direct_receives={},
        _agentic_snapshot_store=lambda: store,
        disagg_prefill_inflight_queue=[],
        agentic_tp_p2d_sender_mailbox=None,
        agentic_tp_p2d_receiver_mailbox=None,
        agentic_host_staging_manager=None,
    )

    control = Scheduler._agentic_tp_prepare_admission_control(owner)

    assert control["direct_commands"][0]["action"] == "poll"
    assert released == []


def test_tp_direct_start_timeout_preserves_concurrently_consumed_group(monkeypatch):
    request = RequestGeneration("completed-during-reduce", 4)
    arrived_at = time.time() - 60.0
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
        state=SnapshotState.CONSUMED,
        claim_id="direct-early-tp:p0:completed-during-reduce:4",
    )
    released = []
    entry = SimpleNamespace(group_committed=True)
    owner = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        disaggregation_mode=DisaggregationMode.PREFILL,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
        agentic_tp_direct_admission_active={
            request.snapshot_id: (request, arrived_at, None, 1024)
        },
        # Model a stale reduction immediately before background commit.
        agentic_tp_direct_group_status={request.snapshot_id: 0},
        agentic_tp_direct_mailbox=SimpleNamespace(
            receipt=lambda _snapshot_id: 3
        ),
        agentic_early_direct_receives={request.snapshot_id: entry},
        _agentic_snapshot_store=lambda: SimpleNamespace(
            load=lambda *_args, **_kwargs: manifest,
            release_direct_claim=lambda *_args: released.append(True),
        ),
        disagg_prefill_inflight_queue=[],
        agentic_tp_p2d_sender_mailbox=None,
        agentic_tp_p2d_receiver_mailbox=None,
        agentic_host_staging_manager=None,
    )

    control = Scheduler._agentic_tp_prepare_admission_control(owner)

    assert control["direct_commands"][0]["action"] == "prepare_bind"
    assert released == []


def test_tp_direct_background_abort_survives_cleanup_error(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "p0")
    request = RequestGeneration("cleanup-error", 5)
    arrived_at = time.time() - 60.0
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
        state=SnapshotState.DIRECT_LOADING,
        claim_id="direct-early-tp:p0:cleanup-error:5",
    )

    def fail_cleanup(*_args):
        raise RuntimeError("injected cleanup failure")

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        mailboxes = [
            TPGroupMailbox(
                "direct-background-abort",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]
        for mailbox in mailboxes:
            mailbox.publish_local_progress(request.snapshot_id, -1)
        owner = SimpleNamespace(
            tp_rank=0,
            agentic_tp_direct_mailbox=mailboxes[0],
            agentic_early_direct_poll_lock=nullcontext(),
            agentic_tp_direct_admission_active={
                request.snapshot_id: (request, arrived_at, None, 1024)
            },
            agentic_early_direct_receives={},
            agentic_tp_direct_local_failed=set(),
            agentic_tp_direct_local_admitted=set(),
        )
        store = SimpleNamespace(
            load=lambda *_args, **_kwargs: manifest,
            release_direct_claim=fail_cleanup,
        )

        Scheduler._agentic_commit_tp_direct_groups(owner, store)

        assert mailboxes[1].receipt(request.snapshot_id) == -1


def test_tp_direct_background_start_timeout_releases_group_claim(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "p0")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT", "0.1")
    request = RequestGeneration("background-timeout", 6)
    arrived_at = time.time() - 1.0
    claim_id = f"direct-early-tp:p0:{request.snapshot_id}"
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
        state=SnapshotState.DIRECT_LOADING,
        claim_id=claim_id,
    )
    released = []
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        mailbox = TPGroupMailbox(
            "direct-background-timeout",
            tp_rank=0,
            tp_size=2,
            directory=directory,
        )
        mailbox.publish_receipt(request.snapshot_id, 1)
        owner = SimpleNamespace(
            tp_rank=0,
            agentic_tp_direct_mailbox=mailbox,
            agentic_early_direct_poll_lock=nullcontext(),
            agentic_tp_direct_admission_active={
                request.snapshot_id: (request, arrived_at, None, 1024)
            },
            agentic_early_direct_receives={},
            agentic_tp_direct_local_failed=set(),
            agentic_tp_direct_local_admitted=set(),
        )
        owner._agentic_abort_tp_direct_grant = (
            lambda selected, store, reason: Scheduler._agentic_abort_tp_direct_grant(
                owner, selected, store, reason=reason
            )
        )
        owner._agentic_tp_start_direct_shard = lambda *_args, **_kwargs: False
        store = SimpleNamespace(
            load=lambda *_args, **_kwargs: manifest,
            release_direct_claim=lambda current, observed_claim: released.append(
                (current.snapshot_id, observed_claim)
            ),
        )

        Scheduler._agentic_progress_tp_direct_grants(owner, store)

        assert released == [(request.snapshot_id, claim_id)]
        assert mailbox.receipt(request.snapshot_id) == -1
        assert request.snapshot_id in owner.agentic_tp_direct_local_failed


def test_tp_direct_bind_control_is_two_phase():
    request = RequestGeneration("two-phase-bind", 7)
    entry = SimpleNamespace(group_committed=True, prepared_req=None)
    receipt = {"value": 3}
    owner = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        disaggregation_mode=DisaggregationMode.PREFILL,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
        agentic_tp_direct_admission_active={
            request.snapshot_id: (request, time.time(), None, 1024)
        },
        agentic_tp_direct_group_status={},
        agentic_tp_direct_mailbox=SimpleNamespace(
            receipt=lambda _snapshot_id: receipt["value"]
        ),
        agentic_early_direct_receives={request.snapshot_id: entry},
        disagg_prefill_inflight_queue=[],
        agentic_tp_p2d_sender_mailbox=None,
        agentic_tp_p2d_receiver_mailbox=None,
        agentic_host_staging_manager=None,
    )

    control = Scheduler._agentic_tp_prepare_admission_control(owner)
    assert control["direct_commands"][0]["action"] == "prepare_bind"

    entry.prepared_req = object()
    receipt["value"] = 4
    control = Scheduler._agentic_tp_prepare_admission_control(owner)
    assert control["direct_commands"][0]["action"] == "commit_bind"

    receipt["value"] = 5
    control = Scheduler._agentic_tp_prepare_admission_control(owner)
    assert control["direct_commands"][0]["action"] == "clear"


def test_tp_direct_prepared_bind_rollback_releases_pin_and_branch():
    req = SimpleNamespace(
        rid="rollback-child",
        _agentic_direct_parent_pin_node=object(),
        _agentic_direct_parent_token_count=1024,
    )
    entry = SimpleNamespace(prepared_req=req)
    pins = []
    releases = []
    owner = SimpleNamespace(
        tree_cache=SimpleNamespace(
            dec_lock_ref=lambda node: pins.append(node),
            release_agentic_request_cache=lambda selected, **kwargs: releases.append(
                (selected, kwargs)
            ),
        )
    )

    Scheduler._agentic_rollback_prepared_direct_bind(owner, entry)

    assert len(pins) == 1
    assert releases == [
        (
            req,
            {"committed_len": 1024, "_defer_if_blocked": False},
        )
    ]
    assert entry.prepared_req is None
    assert not hasattr(req, "_agentic_direct_parent_pin_node")


def test_tp_direct_peer_abort_waits_for_ordered_scheduler_rollback():
    request = RequestGeneration("prepared-peer-abort", 8)
    req = SimpleNamespace(rid="prepared-child")
    entry = SimpleNamespace(prepared_req=req, request=request)
    events = []
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        mailbox = TPGroupMailbox(
            "prepared-peer-abort",
            tp_rank=0,
            tp_size=2,
            directory=directory,
        )
        mailbox.publish_receipt(request.snapshot_id, -1)
        owner = SimpleNamespace(
            tp_size=2,
            tp_rank=0,
            disaggregation_mode=DisaggregationMode.PREFILL,
            _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
            agentic_tp_direct_mailbox=mailbox,
            agentic_early_direct_poll_lock=nullcontext(),
            agentic_tp_direct_admission_active={
                request.snapshot_id: (request, time.time(), None, 1024)
            },
            agentic_tp_direct_group_status={},
            agentic_early_direct_receives={request.snapshot_id: entry},
            agentic_tp_direct_local_failed=set(),
            agentic_tp_direct_local_admitted=set(),
            agentic_tp_host_local_admitted=set(),
            agentic_tp_host_active=None,
            agentic_tp_host_active_since=0.0,
            agentic_tp_host_command_visible=False,
            agentic_tp_host_group_status=0,
            agentic_host_staging_manager=None,
            _agentic_snapshot_store=lambda: object(),
        )
        owner._agentic_rollback_prepared_direct_bind = lambda selected: (
            events.append("rollback"),
            setattr(selected, "prepared_req", None),
        )
        owner._agentic_drop_early_direct_receive = (
            lambda selected, *_args, **_kwargs: (
                events.append("drop"),
                owner.agentic_early_direct_receives.pop(
                    selected.request.snapshot_id, None
                ),
            )
        )
        # The background path observes the group failure but must retain all
        # page ownership until every rank receives the native abort command.
        Scheduler._agentic_progress_tp_direct_grants(owner, object())
        assert owner.agentic_early_direct_receives[request.snapshot_id] is entry
        assert entry.prepared_req is req
        assert events == []

        control = {
            Scheduler._AGENTIC_TP_CONTROL_KEY: True,
            "direct_commands": [
                {
                    "snapshot": request.snapshot_id,
                    "request_id": request.request_id,
                    "generation": request.generation,
                    "action": "abort",
                }
            ],
            "prefill_transfer_keys": [],
            "prefill_transfer_statuses": [],
            "prefill_submit_keys": [],
            "host_snapshot": None,
            "host_action": None,
            "host_timeout_snapshot": None,
        }
        Scheduler._agentic_tp_consume_admission_control(owner, [control])

        assert events == ["rollback", "drop"]
        assert request.snapshot_id not in owner.agentic_early_direct_receives


def test_tp1_direct_arrival_starts_without_scheduler_reservation_queue(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    request = RequestGeneration("tp1-fast", 1)
    arrived_at = time.time()
    manifest = SimpleNamespace(
        request=request,
        state=SnapshotState.DIRECT_READY,
        created_at=arrived_at,
        token_count=1024,
    )
    snapshot_store = SimpleNamespace(
        load=lambda _request, require_ready=False: manifest
    )
    started = []
    scheduler = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        agentic_early_claim_store=object(),
        agentic_tp_direct_admission_active={},
        agentic_early_direct_admission_queue=deque(
            [(request, {"arrived_at": arrived_at}, manifest)]
        ),
        agentic_early_direct_admission_ids={request.snapshot_id},
        agentic_early_direct_receives={},
        agentic_early_direct_terminal={},
        agentic_direct_credit_pool=SimpleNamespace(free_tokens=40000),
        server_args=SimpleNamespace(page_size=64),
        _agentic_start_early_direct_receive=lambda selected, *_args, **_kwargs: (
            started.append(selected.snapshot_id) or True
        ),
    )

    Scheduler._agentic_admit_queued_direct_receives(
        scheduler, snapshot_store, 2.0, nullcontext()
    )

    assert started == [request.snapshot_id]
    assert not scheduler.agentic_early_direct_admission_queue


def test_disabled_compute_ahead_does_not_double_reserve_direct_headroom(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "disabled")
    scheduler = SimpleNamespace(
        agentic_direct_credit_pool=SimpleNamespace(capacity_tokens=40000),
        disagg_prefill_bootstrap_queue=SimpleNamespace(p_ready_dir="/dev/shm"),
        chunked_req=None,
        _p_ready_compute_ahead_throttled=False,
        _get_token_info=lambda: (0, 0.0, 39999, 0),
    )

    method = SchedulerDisaggregationPrefillMixin._should_throttle_p_ready_compute_ahead
    assert not method(scheduler)
    assert not scheduler._p_ready_compute_ahead_throttled
    assert scheduler._p_ready_compute_credit_tokens is None

    scheduler._get_token_info = lambda: (0, 0.0, 50000, 0)
    assert not method(scheduler)
    assert not scheduler._p_ready_compute_ahead_throttled
    assert scheduler._p_ready_compute_credit_tokens is None


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
        agentic_tp_p2d_receiver_mailbox=SimpleNamespace(
            group_status=lambda _key: None,
            publish_receipt=lambda _key, _status: None,
        ),
    )

    control = Scheduler._agentic_tp_prepare_admission_control(scheduler)
    assert control == {
        Scheduler._AGENTIC_TP_CONTROL_KEY: True,
        "decode_release_snapshot": "request:3",
        "decode_admit_keys": [],
        "decode_transfer_keys": [],
        "decode_transfer_statuses": [],
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
