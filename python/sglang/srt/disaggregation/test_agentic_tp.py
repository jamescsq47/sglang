import errno
import os
import json
import mmap
import queue
import shutil
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import sglang.srt.disaggregation.prefill as prefill_module
import sglang.srt.disaggregation.agentic_host_staging as host_staging_module
import sglang.srt.disaggregation.p2d_host_staging as p2d_host_module

from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticDHostStagingClient,
    AgenticNodeLocalRawStore,
    AgenticPHostStagingManager,
    H2DLaunchFence,
    HostStageState,
    LazySharedMHAHostSnapshot,
    P2D_RELEASE_HOST_OWNED,
    SharedHostSnapshotArena,
    SharedHostStagingLedger,
    SharedMHAHostSnapshot,
    _copy_layer_first_host_range,
    create_agentic_storage_controller,
    supports_agentic_kv_spill,
)
from sglang.srt.disaggregation.agentic_early_claim import (
    AgenticDirectoryChangeWatcher,
    AgenticEarlyClaimStore,
    AgenticFileChangeWatcher,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    MooncakeSnapshotStore,
    RequestGeneration,
    SnapshotManifest,
    SnapshotState,
    token_ids_digest,
)
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
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
    TransferStatus,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostLoadManager,
    AgenticPToDHostReceiver,
    AgenticPToDHostStagingManager,
    _RegisteredP2DHostArena,
    _p2d_host_write_committed,
    _raise_if_p2d_host_failed,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import (
    AgenticEarlyDirectReceive,
    AgenticPWorksetLeaseBroker,
    Scheduler,
)
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator


def _ledger():
    fd, path = tempfile.mkstemp(prefix="sglang-agentic-tp-", dir="/dev/shm")
    os.close(fd)
    os.unlink(path)
    return SharedHostStagingLedger(path), path


def test_grouped_paged_free_owns_request_slot_indices():
    """A deferred free must survive reuse of its req_to_token source row."""

    allocator = PagedTokenToKVPoolAllocator.__new__(PagedTokenToKVPoolAllocator)
    allocator.page_size = 64
    allocator.need_sort = False
    allocator.debug_mode = False
    allocator.free_pages = torch.empty(0, dtype=torch.int64)
    allocator.release_pages = torch.empty(0, dtype=torch.int64)
    allocator.is_not_in_free_group = True
    allocator.free_group = []

    request_slot_view = torch.arange(64, 128, dtype=torch.int64)
    allocator.free_group_begin()
    allocator.free(request_slot_view)
    request_slot_view.add_(64)  # Simulate immediate request-slot reuse.
    allocator.free_group_end()

    assert allocator.free_pages.tolist() == [1]


def test_agentic_file_change_watcher_is_edge_triggered(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{}")
    with AgenticFileChangeWatcher(path) as watcher:
        assert watcher.poll(0.0) is False
        path.write_text('{"changed":true}')
        assert watcher.poll(1.0) is True
        assert watcher.poll(0.0) is False


def test_agentic_file_change_watcher_marks_poll_failure_unhealthy():
    class BrokenPoller:
        @staticmethod
        def poll(_timeout_ms):
            raise OSError("watch failed")

    watcher = AgenticFileChangeWatcher.__new__(AgenticFileChangeWatcher)
    watcher._closed = False
    watcher.healthy = True
    watcher.poller = BrokenPoller()

    assert watcher.poll(0.0) is True
    assert watcher.healthy is False


def test_host_ledger_publishes_versioned_snapshot_delta():
    ledger, path = _ledger()
    try:
        with AgenticDirectoryChangeWatcher(ledger.event_directory) as watcher:
            offered = ledger.offer(_rank_offer(0))
            paths, overflow = watcher.poll(1.0)
            assert overflow is False
            assert len(paths) == 1
            event = ledger.read_entry_event(paths[0])
            assert event["snapshot_id"] == offered["snapshot_id"]
            assert event["revision"] == 1
            assert event["entry"]["state"] == "tp_collecting"

            ledger.offer(_rank_offer(1))
            paths, overflow = watcher.poll(1.0)
            assert overflow is False
            event = ledger.read_entry_event(paths[0])
            assert event["revision"] == 2
            assert event["entry"]["state"] == HostStageState.OFFERED.value

            claimed = ledger.claim(offered["snapshot_id"], "p:test")
            assert claimed is not None
            paths, overflow = watcher.poll(1.0)
            assert overflow is False
            assert len(paths) == 1
            event = ledger.read_entry_event(paths[0])
            assert event["revision"] == 3
            assert event["entry"]["state"] == HostStageState.HOST_RESERVED.value
            assert ledger.get(offered["snapshot_id"])["p_owner"] == "p:test"
            assert offered["snapshot_id"] in ledger.snapshot_entries(
                force_refresh=True
            )
            with open(path, encoding="utf-8") as handle:
                # The global file is now relay metadata only. Ordinary Host
                # control never rewrites a map containing all snapshots.
                assert json.load(handle)["entries"] == {}
    finally:
        shutil.rmtree(ledger.event_directory, ignore_errors=True)
        os.unlink(path)


def test_host_ledger_migration_keeps_legacy_global_entry_authoritative():
    with tempfile.TemporaryDirectory(
        prefix="sglang-agentic-ledger-migration-", dir="/dev/shm"
    ) as directory:
        path = os.path.join(directory, "ledger.json")
        ledger = SharedHostStagingLedger(path)
        offered = ledger.offer(_rank_offer(0))
        stale_revision = ledger.read_entry_event(
            ledger._event_path(offered["snapshot_id"])
        )["revision"]
        legacy = dict(offered)
        legacy.update(
            state=HostStageState.REJECTED.value,
            reason="authoritative legacy state",
            _event_revision=stale_revision + 10,
        )
        with open(path, "r+", encoding="utf-8") as handle:
            data = json.load(handle)
            data["entries"] = {offered["snapshot_id"]: legacy}
            handle.seek(0)
            json.dump(data, handle)
            handle.truncate()

        migrated = SharedHostStagingLedger(path)
        current = migrated.get(offered["snapshot_id"])
        assert current["state"] == HostStageState.REJECTED.value
        assert current["reason"] == "authoritative legacy state"
        assert current["_event_revision"] > legacy["_event_revision"]
        with open(path, encoding="utf-8") as handle:
            assert json.load(handle)["entries"] == {}


def test_relay_claim_and_prune_preserve_unrelated_relay_snapshot():
    with tempfile.TemporaryDirectory(
        prefix="sglang-agentic-ledger-relay-", dir="/dev/shm"
    ) as directory:
        path = os.path.join(directory, "ledger.json")
        ledger = SharedHostStagingLedger(path)
        ledger.register_relay(
            relay_id="relay:1",
            pid=222,
            numa_node=1,
            slot_token_count=64,
            slot_count=2,
            d2h_gib_per_second=100.0,
        )

        snapshot_ids = []
        for generation in (3, 4):
            offer = dict(_rank_offer(0))
            offer.update(
                snapshot_id=f"request:{generation}",
                generation=generation,
                tp_size=1,
                source_numa_node=0,
                arena_numa_node=1,
            )
            offered = ledger.offer(offer)
            snapshot_id = offered["snapshot_id"]
            snapshot_ids.append(snapshot_id)
            assert ledger.claim(snapshot_id, "p:test") is not None
            assert ledger.publish_grants(
                snapshot_id,
                "p:test",
                [{"kind": "shared_host_extent"}],
            )
            assigned = ledger.assign_transfer_path(
                snapshot_id,
                source_pid=offer["d_pid"],
                source_numa_node=0,
                arena_numa_node=1,
                direct_cross_numa_gib_per_second=0.01,
                nvlink_gib_per_second=100.0,
                relay_stale_seconds=60.0,
            )
            assert assigned["write_mode"] == "relay"

        claimed = ledger.claim_relay_job("relay:1", 222)
        assert claimed["snapshot_id"] == snapshot_ids[0]
        assert ledger.get(snapshot_ids[1])["relay_job_state"] == "queued"
        with open(path, encoding="utf-8") as handle:
            assert set(json.load(handle)["entries"]) == set(snapshot_ids)

        def make_terminal(entries):
            current = entries[snapshot_ids[0]]
            current["state"] = HostStageState.FAILED.value
            current["updated_at"] = 0.0
            return True, True

        assert ledger._mutate_relay_entry(snapshot_ids[0], make_terminal)
        ledger.prune(older_than_seconds=0.0, consumed_older_than_seconds=0.0)
        assert ledger.get(snapshot_ids[0]) is None
        assert ledger.get(snapshot_ids[1]) is not None
        with open(path, encoding="utf-8") as handle:
            assert set(json.load(handle)["entries"]) == {snapshot_ids[1]}


def test_host_control_applies_only_new_snapshot_delta():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._ledger_event_queue = queue.SimpleQueue()
    manager._ledger_event_ready = threading.Event()
    manager._ledger_entries_cache = {
        "request:3": {"snapshot_id": "request:3", "_event_revision": 2}
    }
    manager._ledger_event_queue.put(
        {
            "snapshot_id": "request:3",
            "revision": 1,
            "entry": {
                "snapshot_id": "request:3",
                "_event_revision": 1,
                "state": HostStageState.OFFERED.value,
            },
        }
    )
    manager._ledger_event_queue.put(
        {
            "snapshot_id": "request:4",
            "revision": 1,
            "entry": {
                "snapshot_id": "request:4",
                "_event_revision": 1,
                "state": HostStageState.HOST_READY.value,
            },
        }
    )
    manager._ledger_event_ready.set()

    assert manager._apply_ledger_events() == {"request:4"}
    assert manager._ledger_entries_cache["request:3"]["_event_revision"] == 2
    assert (
        manager._ledger_entries_cache["request:4"]["state"]
        == HostStageState.HOST_READY.value
    )
    assert manager._ledger_event_ready.is_set() is False


def test_ledger_force_refresh_bypasses_cross_process_cache(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LEDGER_CACHE_SECONDS", "60")
    ledger, path = _ledger()
    peer = SharedHostStagingLedger(path)
    try:
        assert ledger.snapshot_entries() == {}
        peer.offer(_rank_offer(0))
        assert ledger.snapshot_entries() == {}
        assert "request:3" in ledger.snapshot_entries(force_refresh=True)
    finally:
        os.unlink(path)


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


def test_layer_first_host_copy_moves_only_the_requested_token_range():
    source = torch.arange(2 * 2 * 4 * 1 * 2, dtype=torch.int32).view(
        2, 2, 4, 1, 2
    )
    destination = torch.full((2, 2, 6, 1, 2), -1, dtype=torch.int32)

    _copy_layer_first_host_range(
        destination,
        source,
        destination_start=3,
        source_start=1,
        token_count=2,
    )

    torch.testing.assert_close(destination[:, :, 3:5], source[:, :, 1:3])
    assert torch.all(destination[:, :, :3] == -1)
    assert torch.all(destination[:, :, 5:] == -1)


def test_tp1_p2d_completion_poll_does_not_wait_for_reverse_direct_lock():
    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("submitted P->D completion must be lock-free")

        def __exit__(self, *_args):
            return False

    polls = []
    req = SimpleNamespace(
        disagg_p_ready_transfer_started=True,
        disagg_kv_sender=SimpleNamespace(
            kv_mgr=SimpleNamespace(thread_sync_rw_enabled=True),
            poll=lambda: polls.append(True) or KVPoll.Success
        ),
    )
    scheduler = SimpleNamespace(
        agentic_p2d_host_staging_manager=None,
        agentic_nixl_control_lock=ForbiddenLock(),
    )

    result = (
        SchedulerDisaggregationPrefillMixin._prefill_transfer_progress_tp1_req_once(
            scheduler, req
        )
    )

    assert result == int(KVPoll.Success)
    assert polls == [True]


def test_tp1_p2d_completion_keeps_python_lock_without_native_rw_sync():
    class TrackingLock:
        def __init__(self):
            self.held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, *_args):
            self.held = False

    lock = TrackingLock()
    sender = SimpleNamespace(
        kv_mgr=SimpleNamespace(thread_sync_rw_enabled=False),
    )
    sender.poll = lambda: (
        KVPoll.Success
        if lock.held
        else (_ for _ in ()).throw(
            AssertionError("legacy NIXL completion poll must remain locked")
        )
    )
    req = SimpleNamespace(
        disagg_p_ready_transfer_started=True,
        disagg_kv_sender=sender,
    )
    scheduler = SimpleNamespace(
        agentic_p2d_host_staging_manager=None,
        agentic_nixl_control_lock=lock,
    )
    scheduler._prefill_transfer_progress_req_once = lambda request: int(
        request.disagg_kv_sender.poll()
    )

    result = (
        SchedulerDisaggregationPrefillMixin._prefill_transfer_progress_tp1_req_once(
            scheduler, req
        )
    )

    assert result == int(KVPoll.Success)


def test_tp1_p2d_submission_keeps_short_nixl_control_lock():
    class TrackingLock:
        def __init__(self):
            self.held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, *_args):
            self.held = False

    lock = TrackingLock()
    scheduler = SimpleNamespace(
        agentic_p2d_host_staging_manager=None,
        agentic_nixl_control_lock=lock,
    )
    scheduler._prefill_transfer_progress_req_once = lambda _req: (
        int(KVPoll.Transferring)
        if lock.held
        else (_ for _ in ()).throw(AssertionError("NIXL submission is unlocked"))
    )
    req = SimpleNamespace(disagg_p_ready_transfer_started=False)

    result = (
        SchedulerDisaggregationPrefillMixin._prefill_transfer_progress_tp1_req_once(
            scheduler, req
        )
    )

    assert result == int(KVPoll.Transferring)


def _tp1_terminal_scheduler(req, poll, p2d_host):
    key = (str(req.rid), int(req.bootstrap_room))
    return SimpleNamespace(
        disagg_prefill_inflight_queue=[req],
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        _prefill_transfer_async_enabled=True,
        _prefill_transfer_prepare_queue=deque(),
        _prefill_transfer_prepare_keys=set(),
        _prefill_transfer_terminal_queue=deque([key]),
        _prefill_transfer_poll_lock=threading.Lock(),
        _prefill_transfer_key=lambda request: (
            str(request.rid),
            int(request.bootstrap_room),
        ),
        _prefill_transfer_cached_polls=lambda _requests: [poll],
        _prepare_deferred_prefill_transfer=lambda _request: True,
        _enqueue_deferred_prefill_transfer=lambda _request: True,
        _release_prefill_transfer_poll_claims=lambda _requests: None,
        agentic_p2d_host_staging_manager=p2d_host,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(kv_args=SimpleNamespace(kv_item_lens=[]))
        ),
        stream_output=lambda *_args, **_kwargs: None,
        enable_metrics=False,
    )


def test_tp1_p2d_terminal_edge_requeues_when_host_release_is_not_ready():
    release_attempts = []
    p2d_host = SimpleNamespace(
        poll=lambda _req: None,
        prepare_scheduler_release=lambda req: release_attempts.append(req.rid)
        or False,
    )
    req = SimpleNamespace(
        rid="terminal-host-wins",
        bootstrap_room=7,
        disagg_p_ready_deferred=False,
        _async_prefill_transfer_poll=int(KVPoll.Success),
        return_logprob=False,
    )
    scheduler = _tp1_terminal_scheduler(req, int(KVPoll.Success), p2d_host)

    first = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
        scheduler
    )
    second = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
        scheduler
    )

    assert first == second == []
    assert release_attempts == [req.rid, req.rid]
    assert list(scheduler._prefill_transfer_terminal_queue) == [
        (req.rid, req.bootstrap_room)
    ]


def test_tp1_p2d_failed_cleanup_requeues_the_same_terminal_edge():
    cleanup_attempts = []
    sender = SimpleNamespace(
        failure_exception=lambda: RuntimeError("transfer failed")
    )
    req = SimpleNamespace(
        rid="terminal-failed",
        bootstrap_room=9,
        disagg_p_ready_deferred=False,
        _async_prefill_transfer_poll=int(KVPoll.Failed),
        disagg_kv_sender=sender,
        time_stats=SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **_kwargs: None)
        ),
        return_logprob=False,
    )
    scheduler = _tp1_terminal_scheduler(req, int(KVPoll.Failed), None)
    scheduler._cleanup_failed_prefill_transfer = (
        lambda request, *_args: cleanup_attempts.append(request.rid) or False
    )

    SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
        scheduler
    )

    assert cleanup_attempts == [req.rid]
    assert list(scheduler._prefill_transfer_terminal_queue) == [
        (req.rid, req.bootstrap_room)
    ]


def test_tp_p2d_failed_cleanup_defers_terminal_to_host_owner():
    """Native failure must not mask a Host claim that won during cleanup."""

    sender = SimpleNamespace(
        failure_exception=lambda: RuntimeError("native transfer failed"),
        poll=lambda: (_ for _ in ()).throw(
            AssertionError("Host-owned progress must not poll native sender")
        ),
    )
    req = SimpleNamespace(
        rid="native-failed-host-won",
        bootstrap_room=10,
        disagg_p_ready_deferred=False,
        _async_prefill_transfer_poll=int(KVPoll.Failed),
        disagg_kv_sender=sender,
        time_stats=SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **_kwargs: None)
        ),
        return_logprob=False,
    )
    scheduler = _tp1_terminal_scheduler(req, int(KVPoll.Failed), None)
    scheduler.tp_size = 2
    scheduler._cleanup_failed_prefill_transfer = lambda *_args: False

    SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
        scheduler
    )

    assert not hasattr(req, "_agentic_p2d_group_terminal")
    assert list(scheduler._prefill_transfer_terminal_queue) == [
        (req.rid, req.bootstrap_room)
    ]

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        namespace = f"p2d-host-won-{time.time_ns()}"
        sender_mailbox = TPGroupMailbox(
            f"{namespace}-sender", tp_rank=0, tp_size=2, directory=directory
        )
        sender_peer = TPGroupMailbox(
            f"{namespace}-sender", tp_rank=1, tp_size=2, directory=directory
        )
        receiver_mailbox = TPGroupMailbox(
            f"{namespace}-receiver", tp_rank=0, tp_size=2, directory=directory
        )
        key = scheduler._prefill_transfer_key(req)
        sender_peer.publish_local(key, int(KVPoll.Success))
        host_polls = []
        scheduler.tp_rank = 0
        scheduler.agentic_tp_p2d_sender_mailbox = sender_mailbox
        scheduler.agentic_tp_p2d_receiver_mailbox = receiver_mailbox
        scheduler.agentic_p2d_host_staging_manager = SimpleNamespace(
            poll=lambda request: host_polls.append(request.rid)
            or int(KVPoll.Success)
        )

        poll = SchedulerDisaggregationPrefillMixin._prefill_transfer_progress_tp_req_once(
            scheduler, req
        )

    assert poll == int(KVPoll.Success)
    assert host_polls == [req.rid]


def test_node_local_metadata_store_is_cross_instance_and_create_only():
    with tempfile.TemporaryDirectory(
        prefix="sglang-agentic-metadata-", dir="/dev/shm"
    ) as directory:
        first = AgenticNodeLocalRawStore(directory)
        second = AgenticNodeLocalRawStore(directory)

        assert first.put("manifest", b"ready") == 0
        assert second.put("manifest", b"duplicate") != 0
        assert second.get("manifest") == b"ready"
        assert second.is_exist("manifest") == 1
        assert second.upsert("manifest", b"loading") == 0
        assert first.get("manifest") == b"loading"
        assert first.batch_is_exist(["manifest", "missing"]) == [1, 0]
        assert first.batch_remove(["manifest", "missing"]) == [0, -1]
        assert second.get("manifest") == b""


def test_custom_storage_controller_needs_no_native_storage_backend(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_METADATA_DIR", str(tmp_path))
    controller = create_agentic_storage_controller(
        token_allocator=None,
        server_args=SimpleNamespace(hicache_storage_backend=None),
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        model_name="test",
    )
    assert controller.mem_pool_host is None
    assert not controller.storage_backend.supports_kv_spill
    store = controller.storage_backend.agentic_snapshot_store()
    assert store.store.put("claim", b"owner") == 0

    # Direct-only ablations still need the node-local lifecycle store even
    # though no Shared Host Arena or native HiCache backend exists.
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", "true")
    scheduler = SimpleNamespace(
        agentic_storage_controller=controller,
        tree_cache=SimpleNamespace(cache_controller=None),
    )
    assert Scheduler._agentic_snapshot_store(scheduler) is not None


def test_publish_agentic_failure_releases_only_after_failed_cas():
    request = RequestGeneration("failure-cas", 0)
    metadata = SimpleNamespace(current=request, tool_type=None)
    manifest = SimpleNamespace(claim_id="claim", state=SnapshotState.DIRECT_READY)
    routes = []

    class Store:
        lose_cas = True

        def mark_failed(self, *_args, **_kwargs):
            if self.lose_cas:
                raise RuntimeError("concurrent P claim won")
            return SimpleNamespace(state=SnapshotState.FAILED)

    store = Store()
    manager = SimpleNamespace(
        agentic_snapshot_store=store,
        _publish_agentic_route=lambda request, **kwargs: (
            routes.append((request, kwargs)) or True
        ),
    )

    assert not DecodeKVCacheOffloadManager._publish_agentic_failure(
        manager, metadata, "no_host", manifest
    )
    assert routes == []

    store.lose_cas = False
    assert DecodeKVCacheOffloadManager._publish_agentic_failure(
        manager, metadata, "no_host", manifest
    )
    assert routes == [(request, {"route": "recompute"})]


def test_agentic_nixl_ignores_heartbeat_failure_until_data_is_complete():
    room = 17
    bootstrap = "127.0.0.1:9000"
    transfer = TransferStatus(is_failure=True)
    manager = SimpleNamespace(
        transfer_statuses={room: transfer},
        addr_to_rooms_tracker={bootstrap: {room}},
        update_transfer_status=lambda: None,
    )
    receiver = NixlKVReceiver.__new__(NixlKVReceiver)
    receiver.bootstrap_room = room
    receiver.bootstrap_addr = bootstrap
    receiver.kv_mgr = manager
    receiver.started_transfer = True
    receiver.conclude_state = None

    # Heartbeat failure is only control-plane evidence.  A late remote WRITE
    # may still target these pages, so they remain quarantined.
    assert receiver.poll_agentic() == KVPoll.WaitingForInput
    assert room in manager.transfer_statuses

    # Physical completion is the complete set of remote-write notifications;
    # only this state authorizes allocator reuse.
    transfer.received_aux = True
    transfer.num_pp_ranks_expected = 1
    transfer.expected_kvs_per_pp[0] = 1
    transfer.received_kvs_per_pp[0].add(0)
    assert receiver.poll_agentic() == KVPoll.Success
    assert room not in manager.transfer_statuses
    assert room not in manager.addr_to_rooms_tracker[bootstrap]


def test_workset_lease_reserves_parent_and_suffix_then_commits_parent():
    class Allocator:
        def __init__(self):
            self.next_index = 0
            self.freed = []

        def alloc(self, count):
            result = torch.arange(
                self.next_index, self.next_index + count, dtype=torch.int64
            )
            self.next_index += count
            return result

        def free(self, indices):
            self.freed.append(indices.clone())

    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.request("request:3", parent_tokens=4, prompt_tokens=9)
    broker.service(allocator)
    lease = broker.get("request:3")
    assert lease is not None
    assert lease.allocated_tokens == 12
    assert torch.equal(lease.parent_indices, torch.arange(0, 4))
    assert torch.equal(lease.suffix_indices, torch.arange(4, 12))

    assert broker.begin_bind("request:3", lease)
    broker.commit_parent_bound("request:3", lease)
    req = SimpleNamespace(origin_input_ids=list(range(9)))
    broker.handoff_to_req("request:3", req, lease)
    # Cleanup after the ownership commit can be retried.  Repeating the exact
    # same commit is a no-op, while a different Req cannot steal the lease.
    broker.handoff_to_req("request:3", req, lease)
    with pytest.raises(RuntimeError, match="another request"):
        broker.handoff_to_req(
            "request:3",
            SimpleNamespace(origin_input_ids=list(range(9))),
            lease,
        )
    assert broker.get("request:3") is lease
    assert allocator.freed == []
    assert req._agentic_workset_backed is True
    assert torch.equal(
        req._agentic_workset_suffix_indices, torch.arange(4, 12)
    )
    with pytest.raises(RuntimeError, match="page boundary"):
        broker.consume_suffix(lease, 3, final_prompt_chunk=False)
    assert lease.suffix_cursor == 0
    assert lease.state == "handed"
    assert torch.equal(
        broker.consume_suffix(lease, 4, final_prompt_chunk=False),
        torch.arange(4, 8),
    )
    assert broker.get("request:3") is lease
    assert torch.equal(
        broker.consume_suffix(lease, 1, final_prompt_chunk=True),
        torch.arange(8, 9),
    )
    assert broker.get("request:3") is None


def test_workset_lease_rounds_parent_and_suffix_independently():
    class Allocator:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            raise AssertionError("live workset must not be freed")

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.request("unaligned:1", parent_tokens=5, prompt_tokens=10)
    broker.service(Allocator())
    lease = broker.get("unaligned:1")

    assert lease is not None
    assert lease.parent_allocated_tokens == 8
    assert lease.suffix_allocated_tokens == 8
    assert lease.allocated_tokens == 16


def test_host_ready_tombstone_rejects_a_late_direct_intent_but_not_slow():
    snapshot_id = "slow-won-before-marker:0"
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    direct_owner = broker.direct_owner(snapshot_id)
    slow_owner = broker.slow_owner(snapshot_id, "next-request")

    assert not broker.supersede_unstarted(snapshot_id, owner=direct_owner)
    assert broker.owner_is_superseded(snapshot_id, owner=direct_owner)
    assert not broker.request(
        snapshot_id, parent_tokens=4, prompt_tokens=8, owner=direct_owner
    )
    assert broker.request(
        snapshot_id, parent_tokens=4, prompt_tokens=8, owner=slow_owner
    )

    # Consuming the marker is not enough to clear the tombstone: an older TP
    # epoch may still install its already-frozen Direct plan afterwards.
    assert broker.owner_is_superseded(snapshot_id, owner=direct_owner)


def test_direct_terminal_drop_retires_and_tombstones_active_workset():
    """A completed Direct abort cannot leave or recreate an active lease."""

    class Allocator:
        def __init__(self):
            self.freed = []

        @staticmethod
        def alloc(count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.freed.append(indices.clone())

    snapshot_id = "direct-terminal-retire:0"
    request = RequestGeneration("direct-terminal-retire", 0)
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    direct_owner = broker.direct_owner(snapshot_id)
    allocator = Allocator()
    assert broker.request(
        snapshot_id, parent_tokens=4, prompt_tokens=8, owner=direct_owner
    )
    broker.service(allocator)
    lease = broker.get(snapshot_id, owner=direct_owner)
    assert lease is not None

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_p_workset_broker = broker
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_early_direct_receives = {}
    scheduler.agentic_early_direct_terminal = {}
    scheduler._agentic_clear_direct_receiver = lambda *_args: None
    entry = AgenticEarlyDirectReceive(
        request=SimpleNamespace(snapshot_id=snapshot_id),
        manifest=SimpleNamespace(),
        claim_id="direct-claim",
        receiver=SimpleNamespace(clear=lambda: None),
        device_indices=None,
        started_at=time.monotonic(),
        arrived_at=time.time(),
        workset_lease=lease,
        transport_poll=KVPoll.Failed,
    )
    scheduler.agentic_early_direct_receives[snapshot_id] = entry

    scheduler._agentic_drop_early_direct_receive(
        entry,
        snapshot_store=object(),
        release_claim=False,
        reason="direct_setup_deadline_unstarted",
    )

    assert broker.owner_is_superseded(snapshot_id, owner=direct_owner)
    assert lease.state == "releasing"
    broker.service(allocator)
    assert broker.get(snapshot_id) is None
    assert len(allocator.freed) == 1
    assert not broker.request(
        snapshot_id, parent_tokens=4, prompt_tokens=8, owner=direct_owner
    )
    assert broker.request(
        snapshot_id,
        parent_tokens=4,
        prompt_tokens=8,
        owner=broker.slow_owner(snapshot_id, "child"),
    )


def test_tp0_retires_a_superseded_direct_from_an_already_frozen_plan():
    class Allocator:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            pass

    snapshot_id = "host-ready-after-freeze:0"
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner(snapshot_id)

    # HOST_READY is observed while TP0's older epoch is already in flight and
    # before this rank has installed any local intent/lease.
    assert not broker.supersede_unstarted(snapshot_id, owner=owner)
    broker.install_tp_plan(1, [(snapshot_id, owner, 4, 8)])
    broker.service(Allocator())
    # The already-broadcast epoch remains a group decision.  This rank still
    # allocates Direct, then TP0 retires it authoritatively next epoch.
    assert broker.get(snapshot_id, owner=owner) is not None

    _plan, retirements, _handoffs = broker.prepare_tp_control(2)
    assert retirements == (snapshot_id,)
    assert snapshot_id in broker.tp_retire_candidates


def test_tp_retire_commit_prevents_late_materialization_from_frozen_plan():
    """A never-allocated frozen entry is terminal once TP retirement commits."""

    class Allocator:
        def __init__(self):
            self.allocations = []

        def alloc(self, count):
            self.allocations.append(count)
            return torch.arange(count, dtype=torch.int64)

        @staticmethod
        def free(_indices):
            raise AssertionError("unmaterialized work has no pages to free")

    snapshot_id = "retire-before-service:0"
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner(snapshot_id)
    broker.install_tp_plan(1, [(snapshot_id, owner, 4, 8)])

    assert broker.supersede_unstarted(snapshot_id, owner=owner)
    assert broker.owner_has_unretired_work(snapshot_id, owner=owner)
    assert broker.prepare_tp_retire(snapshot_id)
    assert broker.commit_tp_retire(snapshot_id)

    allocator = Allocator()
    broker.service(allocator)
    assert allocator.allocations == []
    assert broker.get(snapshot_id) is None
    assert not broker.owner_has_unretired_work(snapshot_id, owner=owner)


def test_old_direct_plan_cannot_retire_the_new_slow_owner():
    class Allocator:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            pass

    snapshot_id = "direct-to-slow-owner-transition:0"
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    direct_owner = broker.direct_owner(snapshot_id)
    slow_owner = broker.slow_owner(snapshot_id, "next-request")
    allocator = Allocator()

    # An old TP epoch installs Direct after HOST_READY had already superseded
    # it.  The following epoch performs the required group retirement.
    assert not broker.supersede_unstarted(snapshot_id, owner=direct_owner)
    broker.install_tp_plan(1, [(snapshot_id, direct_owner, 4, 8)])
    broker.service(allocator)
    _plan, retirements, _handoffs = broker.prepare_tp_control(2)
    assert retirements == (snapshot_id,)
    assert broker.commit_tp_retire(snapshot_id)
    broker.service(allocator)
    assert broker.get(snapshot_id) is None

    # Slow now owns the same request-generation.  Epoch 2 still names the old
    # Direct owner, but that stale plan may no longer create a snapshot-wide
    # retirement that would kill Slow.
    assert broker.request(snapshot_id, 4, 8, owner=slow_owner)
    plan, retirements, _handoffs = broker.prepare_tp_control(3)
    assert plan == ((snapshot_id, slow_owner, 4, 8),)
    assert retirements == ()
    broker.service(allocator)
    slow_lease = broker.get(snapshot_id, owner=slow_owner)
    assert slow_lease is not None
    assert broker.begin_io_attempt(snapshot_id, slow_lease, "slow-attempt")


def test_frozen_direct_plan_stays_rank_consistent_during_slow_transition():
    class Allocator:
        def __init__(self):
            self.allocations = []
            self.cursor = 0

        def alloc(self, count):
            self.allocations.append(count)
            result = torch.arange(
                self.cursor, self.cursor + count, dtype=torch.int64
            )
            self.cursor += count
            return result

        def free(self, _indices):
            pass

    snapshot_id = "host-ready-between-plan-and-service:0"
    next_snapshot_id = "following-direct:0"
    brokers = [AgenticPWorksetLeaseBroker(page_size=4) for _ in range(2)]
    direct_owner = brokers[0].direct_owner(snapshot_id)
    next_owner = brokers[0].direct_owner(next_snapshot_id)
    slow_owner = brokers[0].slow_owner(snapshot_id, "next-request")
    allocators = [Allocator(), Allocator()]

    # TP0 froze Direct, but HOST_READY and the successor Slow request became
    # authoritative on rank0 before either rank services that epoch.  Rank1
    # has not observed HOST_READY yet.  Rank-local observation order must not
    # change any physical allocation in the already-broadcast plan.
    plan = [
        (snapshot_id, direct_owner, 4, 8),
        (next_snapshot_id, next_owner, 4, 8),
    ]
    for broker in brokers:
        broker.install_tp_plan(1, plan)
    assert brokers[0].supersede_unstarted(snapshot_id, owner=direct_owner)
    # A successor owner cannot enter while the old snapshot-scoped TP
    # retirement is waiting for its group ACK.
    assert not brokers[0].request(snapshot_id, 4, 8, owner=slow_owner)

    for broker, allocator in zip(brokers, allocators):
        broker.service(allocator)

    assert allocators[0].allocations == allocators[1].allocations == [8, 8]
    for broker in brokers:
        assert torch.equal(
            broker.get(snapshot_id, owner=direct_owner).device_indices,
            torch.arange(0, 8),
        )
        assert torch.equal(
            broker.get(next_snapshot_id, owner=next_owner).device_indices,
            torch.arange(8, 16),
        )


def test_workset_lease_does_not_steal_native_chunk_continuation_capacity():
    class Allocator:
        def __init__(self):
            self.available = 20

        def available_size(self):
            return self.available

        def alloc(self, count):
            if count > self.available:
                return None
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            pass

    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.request("direct:1", parent_tokens=4, prompt_tokens=12)

    # The workset needs 12 pages and would fit physically, but the active
    # native chunk already owns the promise of the final 12 pages.
    broker.service(allocator, reserve_tokens=12)
    assert broker.get("direct:1") is None
    assert allocator.available == 20

    # Once the chunk finishes, the same pending intent can be granted.
    broker.service(allocator)
    assert broker.get("direct:1") is not None
    assert allocator.available == 8


def test_tp_workset_allocation_plan_prevents_cross_rank_ownership_inversion():
    class Allocator:
        def __init__(self):
            self.available = 16
            self.allocations = []

        def available_size(self):
            return self.available

        def alloc(self, count):
            if count > self.available:
                return None
            start = 16 - self.available
            self.available -= count
            self.allocations.append(count)
            return torch.arange(start, start + count, dtype=torch.int64)

        def free(self, indices):
            self.available += int(indices.numel())

    rank0 = AgenticPWorksetLeaseBroker(page_size=4)
    rank1 = AgenticPWorksetLeaseBroker(page_size=4)
    owner_a = rank0.direct_owner("a:0")
    owner_b = rank0.direct_owner("b:0")
    assert rank0.request("a:0", 4, 8, owner=owner_a)
    assert rank0.request("b:0", 4, 8, owner=owner_b)
    # Filesystem/HTTP readiness is deliberately observed in the opposite
    # order on the follower.
    assert rank1.request("b:0", 4, 8, owner=owner_b)

    plan = rank0.prepare_tp_plan(1)
    allocator0 = Allocator()
    allocator1 = Allocator()
    rank1.install_tp_plan(1, plan)
    rank0.service(allocator0)
    rank1.service(allocator1)

    assert rank0.get("a:0", owner=owner_a) is not None
    assert rank0.get("b:0", owner=owner_b) is not None
    # TP0's immutable command is sufficient to install a follower intent; a
    # rank-local filesystem marker is not an allocation authority.
    assert rank1.get("a:0", owner=owner_a) is not None
    assert rank1.get("b:0", owner=owner_b) is not None
    assert allocator0.allocations == allocator1.allocations == [8, 8]


def test_tp_workset_epoch_defers_rank0_cancel_until_group_removal():
    class Allocator:
        def __init__(self):
            self.available = 24

        def available_size(self):
            return self.available

        def alloc(self, count):
            if count > self.available:
                return None
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.available += int(indices.numel())

    rank0 = AgenticPWorksetLeaseBroker(page_size=4)
    rank1 = AgenticPWorksetLeaseBroker(page_size=4)
    owner_a = rank0.direct_owner("cancel-a:0")
    owner_b = rank0.direct_owner("cancel-b:0")
    rank0.request("cancel-a:0", 4, 8, owner=owner_a)
    rank0.request("cancel-b:0", 4, 8, owner=owner_b)
    plan1 = rank0.prepare_tp_plan(1)
    rank1.install_tp_plan(1, plan1)

    # This is the exact former race: cancellation lands after TP0 snapshots
    # the plan but before the scheduler services physical allocation.
    assert rank0.cancel_unstarted("cancel-a:0", owner=owner_a)
    allocator0, allocator1 = Allocator(), Allocator()
    rank0.service(allocator0)
    rank1.service(allocator1)
    assert rank0.get("cancel-a:0", owner=owner_a) is not None
    assert rank1.get("cancel-a:0", owner=owner_a) is not None

    plan2 = rank0.prepare_tp_plan(2)
    assert [entry[0] for entry in plan2] == ["cancel-a:0", "cancel-b:0"]
    rank1.install_tp_plan(2, plan2)
    assert rank0.prepare_tp_retire("cancel-a:0")
    assert rank1.prepare_tp_retire("cancel-a:0")
    assert rank0.commit_tp_retire("cancel-a:0")
    assert rank1.commit_tp_retire("cancel-a:0")
    rank0.service(allocator0)
    rank1.service(allocator1)
    assert rank0.get("cancel-a:0", owner=owner_a) is None
    assert rank1.get("cancel-a:0", owner=owner_a) is None
    assert allocator0.available == allocator1.available == 16


def test_tp_workset_epoch_ignores_follower_only_cancel():
    class Allocator:
        def __init__(self):
            self.available = 16

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.available += int(indices.numel())

    rank0 = AgenticPWorksetLeaseBroker(page_size=4)
    rank1 = AgenticPWorksetLeaseBroker(page_size=4)
    owner = rank0.direct_owner("follower-cancel:0")
    rank0.request("follower-cancel:0", 4, 8, owner=owner)
    rank1.request("follower-cancel:0", 4, 8, owner=owner)
    plan1 = rank0.prepare_tp_plan(1)
    rank1.install_tp_plan(1, plan1)
    assert rank1.cancel_unstarted("follower-cancel:0", owner=owner)
    allocator0, allocator1 = Allocator(), Allocator()
    rank0.service(allocator0)
    rank1.service(allocator1)

    # TP0 still owns the generation in epoch 2, overriding the follower's
    # rank-local timeout without freeing either shard.
    plan2 = rank0.prepare_tp_plan(2)
    rank1.install_tp_plan(2, plan2)
    rank0.service(allocator0)
    rank1.service(allocator1)
    assert rank0.get("follower-cancel:0", owner=owner) is not None
    assert rank1.get("follower-cancel:0", owner=owner) is not None
    assert allocator0.available == allocator1.available == 8
    lease0 = rank0.get("follower-cancel:0", owner=owner)
    lease1 = rank1.get("follower-cancel:0", owner=owner)
    assert rank0.begin_io_attempt("follower-cancel:0", lease0, "next")
    assert rank1.begin_io_attempt("follower-cancel:0", lease1, "next")


def test_tp_workset_epoch_defers_release_without_reallocating_old_plan():
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            if count > self.available:
                return None
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.available += int(indices.numel())

    rank0 = AgenticPWorksetLeaseBroker(page_size=4)
    rank1 = AgenticPWorksetLeaseBroker(page_size=4)
    owner = rank0.direct_owner("release-after-plan:0")
    rank0.request("release-after-plan:0", 4, 8, owner=owner)
    plan1 = rank0.prepare_tp_plan(1)
    rank1.install_tp_plan(1, plan1)
    allocator0, allocator1 = Allocator(), Allocator()
    rank0.service(allocator0)
    rank1.service(allocator1)
    old_lease = rank0.get("release-after-plan:0", owner=owner)
    assert old_lease is not None

    # TP0 has frozen epoch 2, then an async transport callback asks to release.
    # Service must retain the exact old lease for this epoch, not free and
    # immediately recreate it from the frozen entry.
    plan2 = rank0.prepare_tp_plan(2)
    rank1.install_tp_plan(2, plan2)
    assert rank0.request_release(
        "release-after-plan:0", old_lease, owner=owner
    )
    assert not rank0.begin_io_attempt(
        "release-after-plan:0", old_lease, "late-io"
    )
    assert not rank0.begin_bind("release-after-plan:0", old_lease)
    rank0.service(allocator0)
    rank1.service(allocator1)
    assert rank0.get("release-after-plan:0", owner=owner) is old_lease
    assert rank1.get("release-after-plan:0", owner=owner) is not None

    assert rank0.prepare_tp_retire("release-after-plan:0")
    assert rank1.prepare_tp_retire("release-after-plan:0")
    assert rank0.commit_tp_retire("release-after-plan:0")
    assert rank1.commit_tp_retire("release-after-plan:0")
    rank0.service(allocator0)
    rank1.service(allocator1)
    assert rank0.get("release-after-plan:0") is None
    assert rank1.get("release-after-plan:0") is None
    assert allocator0.available == allocator1.available == 8
    plan3 = rank0.prepare_tp_plan(3)
    assert plan3 == ()
    rank1.install_tp_plan(3, plan3)


def test_tp_workset_epoch_does_not_resurrect_io_terminal_release():
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            if count > self.available:
                return None
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.available += int(indices.numel())

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    allocator = Allocator()
    owner = broker.direct_owner("io-terminal:0")
    broker.request("io-terminal:0", 4, 8, owner=owner)
    broker.prepare_tp_plan(1)
    broker.service(allocator)
    lease = broker.get("io-terminal:0", owner=owner)
    assert broker.begin_io_attempt("io-terminal:0", lease, "attempt-1")
    broker.mark_io_inflight("io-terminal:0", lease, "attempt-1")

    # Epoch 2 is frozen while DMA is still live.  Its terminal callback may
    # release the old exact lease, but the same epoch must never recreate A.
    assert broker.prepare_tp_plan(2)[0][0] == "io-terminal:0"
    assert not broker.request_release(
        "io-terminal:0", lease, owner=owner, io_attempt="attempt-1"
    )
    assert broker.mark_io_quiesced(
        "io-terminal:0", lease, "attempt-1"
    )
    assert broker.prepare_tp_retire("io-terminal:0")
    assert broker.commit_tp_retire("io-terminal:0")
    broker.service(allocator)
    assert broker.get("io-terminal:0", owner=owner) is None
    assert allocator.available == 8
    assert broker.prepare_tp_plan(3) == ()


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("hybrid", [False, True])
def test_tp_workset_group_retire_waits_for_staggered_rank_fences(tp_size, hybrid):
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            if count > self.available:
                return None
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.available += int(indices.numel())

    state_allocators = [Allocator() for _ in range(tp_size)]
    ranks = [AgenticPWorksetLeaseBroker(
        page_size=4,
        state_allocators=(state,) if hybrid else (),
        mamba_req_to_token_pool=SimpleNamespace(
            enable_mamba_extra_buffer=True, mamba_ping_pong_track_buffer_size=2,
        ) if hybrid else None,
    ) for state in state_allocators]
    owner = ranks[0].direct_owner("stagger:0")
    ranks[0].request("stagger:0", 4, 8, owner=owner)
    plan1 = ranks[0].prepare_tp_plan(1)
    for broker in ranks[1:]:
        broker.install_tp_plan(1, plan1)
    allocators = [Allocator() for _ in ranks]
    for broker, allocator in zip(ranks, allocators):
        broker.service(allocator)
        lease = broker.get("stagger:0", owner=owner)
        assert broker.begin_io_attempt("stagger:0", lease, "attempt")
        broker.mark_io_inflight("stagger:0", lease, "attempt")

    assert all(state.available == (4 if hybrid else 8) for state in state_allocators)

    plan2 = ranks[0].prepare_tp_plan(2)
    for broker in ranks[1:]:
        broker.install_tp_plan(2, plan2)
    leases = [broker.get("stagger:0", owner=owner) for broker in ranks]
    for broker, lease in zip(ranks, leases):
        assert not broker.request_release(
            "stagger:0", lease, owner=owner, io_attempt="attempt"
        )
        assert not broker.prepare_tp_retire("stagger:0")

    # Rank 0 finishes first.  Its pages remain quarantined and the group may
    # not commit while rank 1 still has a live DMA fence.
    assert ranks[0].mark_io_quiesced("stagger:0", leases[0], "attempt")
    assert ranks[0].tp_retire_ready("stagger:0")
    assert not ranks[1].tp_retire_ready("stagger:0")
    assert not ranks[1].commit_tp_retire("stagger:0")
    # Receiver cleanup retries release after the local fence.  This must not
    # turn retire_ready into a local free while another rank is still in DMA.
    for epoch in (3, 4):
        assert ranks[0].request_release("stagger:0", leases[0])
        plan, retiring, _ = ranks[0].prepare_tp_control(epoch)
        for broker in ranks[1:]:
            broker.install_tp_plan(epoch, plan, retiring_ids=retiring)
        for broker, allocator in zip(ranks, allocators):
            broker.service(allocator)
            assert allocator.available == 0
        assert leases[0].state == "retire_ready"
        assert all(lease.state == "release_pending" for lease in leases[1:])
        # Attention pages and Mamba checkpoint/runtime slots share the same
        # all-rank retirement barrier, including repeated local cleanup.
        assert all(state.available == (4 if hybrid else 8) for state in state_allocators)
    for broker, allocator in zip(ranks, allocators):
        broker.service(allocator)
        assert broker.get("stagger:0", owner=owner) is not None

    for broker, lease in zip(ranks[1:], leases[1:]):
        assert broker.mark_io_quiesced("stagger:0", lease, "attempt")
    assert all(broker.tp_retire_ready("stagger:0") for broker in ranks)
    assert all(broker.commit_tp_retire("stagger:0") for broker in ranks)
    for broker, allocator in zip(ranks, allocators):
        broker.service(allocator)
        assert broker.get("stagger:0") is None
        assert allocator.available == 8
    assert ranks[0].prepare_tp_plan(5) == ()
    assert all(state.available == 8 for state in state_allocators)


def test_tp_workset_epoch_rejects_stale_and_shape_mismatch():
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner("shape:0")
    broker.request("shape:0", 4, 8, owner=owner)
    broker.install_tp_plan(3, (("shape:0", owner, 4, 8),))
    # Replaying the exact native broadcast is idempotent; an older epoch or a
    # same-epoch content change is rejected.
    broker.install_tp_plan(3, (("shape:0", owner, 4, 8),))
    with pytest.raises(RuntimeError, match="stale TP workset epoch"):
        broker.install_tp_plan(2, (("shape:0", owner, 4, 8),))
    with pytest.raises(RuntimeError, match="disagrees with local intent"):
        broker.install_tp_plan(4, (("shape:0", owner, 4, 12),))


def test_tp_workset_epoch_replay_includes_authoritative_retirements():
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner("retire-replay:0")
    broker.request("retire-replay:0", 4, 8, owner=owner)
    assert broker.prepare_tp_retire("retire-replay:0")

    plan = broker.prepare_tp_plan(
        3,
        retiring_ids=("retire-replay:0",),
    )
    broker.install_tp_plan(
        3,
        plan,
        retiring_ids=("retire-replay:0",),
    )

    with pytest.raises(RuntimeError, match="replayed with new content"):
        broker.install_tp_plan(3, plan, retiring_ids=())


def test_tp_workset_install_materializes_retire_tombstone_before_return():
    class Allocator:
        def available_size(self):
            return 8

        def alloc(self, count):
            return torch.arange(count)

        def free(self, _indices):
            pass

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner("follower-retire:0")
    broker.request("follower-retire:0", 4, 8, owner=owner)
    plan = broker.prepare_tp_plan(1)
    broker.service(Allocator())
    lease = broker.get("follower-retire:0", owner=owner)
    assert lease is not None

    broker.install_tp_plan(
        2,
        plan,
        retiring_ids=("follower-retire:0",),
    )
    assert broker.tp_retire_candidates == ("follower-retire:0",)
    assert not broker.begin_io_attempt(
        "follower-retire:0", lease, "must-not-start"
    )
    assert not broker.begin_bind("follower-retire:0", lease)


def test_tp_workset_control_freezes_async_retire_with_plan():
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count)

        def free(self, indices):
            self.available += int(indices.numel())

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner("async-retire:0")
    broker.request("async-retire:0", 4, 8, owner=owner)
    broker.prepare_tp_plan(1)
    broker.service(Allocator())
    lease = broker.get("async-retire:0", owner=owner)
    assert lease is not None

    # Model the old scheduler race: it sampled no candidates, then an async
    # completion requested release before the plan was frozen.  The atomic
    # control transaction must include that new terminal decision.
    sampled_before_release = broker.tp_retire_candidates
    assert sampled_before_release == ()
    assert broker.request_release("async-retire:0", lease, owner=owner)
    plan, retiring, handoffs = broker.prepare_tp_control(
        2,
        retiring_ids=sampled_before_release,
    )
    assert plan == (("async-retire:0", owner, 4, 8),)
    assert retiring == ("async-retire:0",)
    assert handoffs == ()
    assert not broker.begin_io_attempt(
        "async-retire:0", lease, "must-not-restart"
    )


def test_tp_workset_final_suffix_broadcasts_group_handoff_without_free():
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count)

        def free(self, indices):
            self.available += int(indices.numel())

    ranks = [AgenticPWorksetLeaseBroker(page_size=4) for _ in range(2)]
    allocators = [Allocator(), Allocator()]
    owner = ranks[0].direct_owner("handed-stagger:0")
    ranks[0].request("handed-stagger:0", 4, 8, owner=owner)
    plan1 = ranks[0].prepare_tp_plan(1)
    ranks[1].install_tp_plan(1, plan1)
    for broker, allocator in zip(ranks, allocators):
        broker.service(allocator)
        lease = broker.get("handed-stagger:0", owner=owner)
        assert lease is not None
        lease.parent_bound = True
        req = SimpleNamespace(origin_input_ids=[0] * 8)
        assert broker.begin_bind("handed-stagger:0", lease)
        broker.handoff_to_req("handed-stagger:0", req, lease)

    # Native TP scheduling consumes the same logical model row on every rank.
    # Only after every shard has transferred ownership to its native Req may
    # TP0 publish the idempotent group handoff acknowledgement.
    for broker in ranks:
        lease = broker.get("handed-stagger:0", owner=owner)
        assert lease is not None
        broker.consume_suffix(lease, 4, final_prompt_chunk=True)
    plan2, retiring, handoffs = ranks[0].prepare_tp_control(2)
    assert plan2 == ()
    assert retiring == ()
    assert handoffs == ("handed-stagger:0",)
    for broker in ranks:
        broker.install_tp_plan(2, plan2)
        assert broker.commit_tp_handoff("handed-stagger:0")
        assert broker.commit_tp_handoff("handed-stagger:0")
    assert all(broker.get("handed-stagger:0") is None for broker in ranks)
    assert allocators[0].available == allocators[1].available == 0


def test_tp_workset_handed_cancel_still_requires_group_retire():
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count)

        def free(self, indices):
            self.available += int(indices.numel())

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    allocator = Allocator()
    owner = broker.direct_owner("handed-cancel:0")
    broker.request("handed-cancel:0", 4, 8, owner=owner)
    broker.prepare_tp_plan(1)
    broker.service(allocator)
    lease = broker.get("handed-cancel:0", owner=owner)
    assert lease is not None
    lease.parent_bound = True
    req = SimpleNamespace(origin_input_ids=[0] * 8)
    assert broker.begin_bind("handed-cancel:0", lease)
    broker.handoff_to_req("handed-cancel:0", req, lease)
    assert broker.prepare_tp_plan(2) == (
        ("handed-cancel:0", owner, 4, 8),
    )

    assert broker.release_handed("handed-cancel:0", lease, req=req)
    assert broker.tp_retire_candidates == ("handed-cancel:0",)
    broker.service(allocator)
    assert broker.get("handed-cancel:0", owner=owner) is lease
    assert allocator.available == 0

    assert broker.prepare_tp_retire("handed-cancel:0")
    assert broker.commit_tp_retire("handed-cancel:0")
    broker.service(allocator)
    assert broker.get("handed-cancel:0") is None
    # Parent pages are Radix-owned; only the unconsumed suffix returns here.
    assert allocator.available == 4


def test_tp_workset_binding_retry_may_finish_one_rank_later():
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count)

        def free(self, indices):
            self.available += int(indices.numel())

    ranks = [AgenticPWorksetLeaseBroker(page_size=4) for _ in range(2)]
    allocators = [Allocator(), Allocator()]
    owner = ranks[0].direct_owner("binding-stagger:0")
    ranks[0].request("binding-stagger:0", 4, 8, owner=owner)
    plan1 = ranks[0].prepare_tp_plan(1)
    ranks[1].install_tp_plan(1, plan1)
    reqs = [SimpleNamespace(origin_input_ids=[0] * 8) for _ in ranks]
    leases = []
    for broker, allocator in zip(ranks, allocators):
        broker.service(allocator)
        lease = broker.get("binding-stagger:0", owner=owner)
        assert lease is not None
        lease.parent_bound = True
        assert broker.begin_bind("binding-stagger:0", lease)
        leases.append(lease)

    ranks[0].handoff_to_req("binding-stagger:0", reqs[0], leases[0])
    plan2 = ranks[0].prepare_tp_plan(2)
    assert plan2 == (("binding-stagger:0", owner, 4, 8),)
    ranks[1].install_tp_plan(2, plan2)
    ranks[1].service(allocators[1])
    assert ranks[1].get("binding-stagger:0", owner=owner).state == "binding"
    assert allocators[0].available == allocators[1].available == 0

    ranks[1].handoff_to_req("binding-stagger:0", reqs[1], leases[1])
    for broker, lease in zip(ranks, leases):
        broker.consume_suffix(lease, 4, final_prompt_chunk=True)
    assert all(broker.get("binding-stagger:0") is None for broker in ranks)
    assert allocators[0].available == allocators[1].available == 0


def test_tp_workset_handoff_commit_rejects_a_follower_rank_split():
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count)

        def free(self, indices):
            self.available += int(indices.numel())

    ranks = [AgenticPWorksetLeaseBroker(page_size=4) for _ in range(2)]
    allocators = [Allocator(), Allocator()]
    snapshot_id = "binding-handoff-stagger:0"
    owner = ranks[0].direct_owner(snapshot_id)
    ranks[0].request(snapshot_id, 4, 8, owner=owner)
    plan1 = ranks[0].prepare_tp_plan(1)
    ranks[1].install_tp_plan(1, plan1)
    reqs = [SimpleNamespace(origin_input_ids=[0] * 8) for _ in ranks]
    leases = []
    for broker, allocator in zip(ranks, allocators):
        broker.service(allocator)
        lease = broker.get(snapshot_id, owner=owner)
        assert lease is not None
        lease.parent_bound = True
        assert broker.begin_bind(snapshot_id, lease)
        leases.append(lease)

    ranks[0].handoff_to_req(snapshot_id, reqs[0], leases[0])
    ranks[0].consume_suffix(leases[0], 4, final_prompt_chunk=True)
    plan2, retiring, handoffs = ranks[0].prepare_tp_control(2)
    assert plan2 == ()
    assert retiring == ()
    assert handoffs == (snapshot_id,)

    ranks[0].install_tp_plan(2, plan2)
    ranks[1].install_tp_plan(2, plan2)
    assert ranks[0].commit_tp_handoff(snapshot_id)
    # A follower that has not consumed the same model row is not allowed to
    # catch up after TP0.  Continuing would assign collective rows to different
    # requests, so the group must fail closed at this boundary.
    assert not ranks[1].commit_tp_handoff(snapshot_id)
    assert ranks[1].get(snapshot_id, owner=owner).state == "binding"


def test_tp1_workset_binding_abort_preserves_original_release_behavior():
    class Allocator:
        def __init__(self):
            self.available = 8

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count)

        def free(self, indices):
            self.available += int(indices.numel())

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    allocator = Allocator()
    snapshot_id = "tp1-bind-abort:0"
    owner = broker.direct_owner(snapshot_id)
    broker.request(snapshot_id, 4, 8, owner=owner)
    broker.service(allocator)
    lease = broker.get(snapshot_id, owner=owner)
    assert lease is not None
    assert broker.begin_bind(snapshot_id, lease)
    assert broker.abort_bind(snapshot_id, lease, parent_bound=False)
    broker.service(allocator)
    assert broker.get(snapshot_id) is None
    assert allocator.available == 8


def test_tp_workset_epoch_releases_then_allocates_in_group_order():
    class Allocator:
        def __init__(self):
            self.available = 8
            self.next_index = 0

        def available_size(self):
            return self.available

        def alloc(self, count):
            if count > self.available:
                return None
            self.available -= count
            result = torch.arange(self.next_index, self.next_index + count)
            self.next_index += count
            return result

        def free(self, indices):
            self.available += int(indices.numel())

    rank0 = AgenticPWorksetLeaseBroker(page_size=4)
    rank1 = AgenticPWorksetLeaseBroker(page_size=4)
    owner_a = rank0.direct_owner("replace-a:0")
    owner_b = rank0.direct_owner("replace-b:0")
    rank0.request("replace-a:0", 4, 8, owner=owner_a)
    plan1 = rank0.prepare_tp_plan(1)
    rank1.install_tp_plan(1, plan1)
    allocator0, allocator1 = Allocator(), Allocator()
    rank0.service(allocator0)
    rank1.service(allocator1)
    lease_a = rank0.get("replace-a:0", owner=owner_a)
    assert rank0.request_release("replace-a:0", lease_a, owner=owner_a)
    rank0.request("replace-b:0", 4, 8, owner=owner_b)

    plan2 = rank0.prepare_tp_plan(2)
    assert [entry[0] for entry in plan2] == ["replace-a:0", "replace-b:0"]
    rank1.install_tp_plan(2, plan2)
    assert rank0.prepare_tp_retire("replace-a:0")
    assert rank1.prepare_tp_retire("replace-a:0")
    assert rank0.commit_tp_retire("replace-a:0")
    assert rank1.commit_tp_retire("replace-a:0")
    rank0.service(allocator0)
    rank1.service(allocator1)
    assert rank0.get("replace-a:0") is rank1.get("replace-a:0") is None
    assert rank0.get("replace-b:0", owner=owner_b) is not None
    assert rank1.get("replace-b:0", owner=owner_b) is not None
    assert allocator0.available == allocator1.available == 0


def test_workset_reserve_never_delays_a_release():
    class Allocator:
        def __init__(self):
            self.available = 16
            self.freed = []

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.freed.append(indices.clone())
            self.available += int(indices.numel())

    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.request("release:1", parent_tokens=4, prompt_tokens=8)
    broker.service(allocator)
    lease = broker.get("release:1")
    assert lease is not None
    assert broker.request_release("release:1", lease)

    broker.service(allocator, reserve_tokens=allocator.available_size())

    assert broker.get("release:1") is None
    assert len(allocator.freed) == 1
    assert allocator.available == 16


def test_cancel_unstarted_reclaims_a_grant_created_after_arrival():
    class Allocator:
        def __init__(self):
            self.available = 16

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.available += int(indices.numel())

    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner("stale:1")
    broker.request("stale:1", parent_tokens=4, prompt_tokens=8, owner=owner)
    broker.service(allocator)
    assert broker.get("stale:1", owner=owner) is not None
    assert allocator.available == 8

    assert broker.cancel_unstarted("stale:1", owner=owner)
    broker.service(allocator, reserve_tokens=16)

    assert broker.get("stale:1", owner=owner) is None
    assert allocator.available == 16


def test_cancel_unstarted_never_reclaims_an_io_owned_workset():
    class Allocator:
        def available_size(self):
            return 16

        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            raise AssertionError("an I/O-owned workset must remain quarantined")

    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner("inflight:1")
    broker.request("inflight:1", parent_tokens=4, prompt_tokens=8, owner=owner)
    broker.service(allocator)
    lease = broker.get("inflight:1", owner=owner)
    assert lease is not None
    assert broker.begin_io_attempt("inflight:1", lease, "attempt:1")

    assert not broker.cancel_unstarted("inflight:1", owner=owner)
    broker.service(allocator)

    assert broker.get("inflight:1", owner=owner) is lease
    assert lease.state == "io_reserved"


def test_scheduler_reserves_exact_unfinished_native_chunk_suffix():
    calls = []
    scheduler = SimpleNamespace(
        page_size=4,
        chunked_req=SimpleNamespace(
            origin_input_ids=list(range(21)),
            output_ids=[],
            fill_ids=list(range(8)),
        ),
        agentic_p_workset_broker=SimpleNamespace(
            service=lambda allocator, *, reserve_tokens: calls.append(reserve_tokens)
        ),
        token_to_kv_pool_allocator=object(),
    )

    Scheduler._agentic_service_p_workset_leases(scheduler)

    assert calls == [16]


def test_scheduler_does_not_reserve_again_for_private_workset_chunk():
    calls = []
    scheduler = SimpleNamespace(
        page_size=4,
        chunked_req=SimpleNamespace(
            origin_input_ids=list(range(21)),
            output_ids=[],
            fill_ids=list(range(8)),
            _agentic_workset_backed=True,
            _agentic_workset_suffix_indices=torch.arange(16),
        ),
        agentic_p_workset_broker=SimpleNamespace(
            service=lambda allocator, *, reserve_tokens: calls.append(reserve_tokens)
        ),
        token_to_kv_pool_allocator=object(),
    )

    Scheduler._agentic_service_p_workset_leases(scheduler)

    assert calls == [0]


def test_workset_release_waits_for_physical_direct_terminal():
    class Allocator:
        def __init__(self):
            self.freed = []

        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.freed.append(indices.clone())

    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner("dma:1")
    broker.request("dma:1", 4, 9, owner=owner)
    broker.service(allocator)
    lease = broker.get("dma:1", owner=owner)
    assert lease is not None

    attempt = "direct:a"
    assert broker.begin_io_attempt("dma:1", lease, attempt)
    broker.mark_io_inflight("dma:1", lease, attempt)
    assert not broker.request_release(
        "dma:1", lease, owner=owner, io_attempt=attempt
    )
    assert lease.state == "release_pending"
    broker.service(allocator)
    assert allocator.freed == []
    assert broker.get("dma:1", owner=owner) is lease

    assert broker.mark_io_quiesced("dma:1", lease, attempt)
    broker.service(allocator)
    assert broker.get("dma:1", owner=owner) is None
    assert len(allocator.freed) == 1
    assert torch.equal(allocator.freed[0], torch.arange(12))


def test_slow_h2d_abort_and_rollback_wait_for_physical_terminal():
    class Allocator:
        def __init__(self):
            self.freed = []

        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.freed.append(indices.clone())

    class Event:
        ready = False

        def query(self):
            return self.ready

        def synchronize(self):
            assert self.ready

    request = RequestGeneration("slow-dma", 1)
    rid = "child-rid"
    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.slow_owner(request.snapshot_id, rid)
    broker.request(request.snapshot_id, 4, 8, owner=owner)
    broker.service(allocator)
    lease = broker.get(request.snapshot_id, owner=owner)
    assert lease is not None
    attempt = f"slow-h2d:p:{rid}:{lease.lease_id}"
    assert broker.begin_io_attempt(request.snapshot_id, lease, attempt)
    broker.mark_io_inflight(request.snapshot_id, lease, attempt)

    event = Event()
    # This focused test models a recovery P's rank-local remote mmap.  The
    # mapping may close after its own H2D fence; physical Host-owner TP shards
    # are covered by the group-atomic test below.
    record = {"remote_host": True}
    ledger_state = {"value": HostStageState.H2D_LOADING.value, "claims": {}}

    class Ledger:
        def request_host_load_failure(self, *_args, **_kwargs):
            ledger_state["value"] = HostStageState.ABORTING.value
            return True

        def get(self, _snapshot_id):
            return {
                "state": ledger_state["value"],
                "recovery_claims": {
                    "0": {
                        "claim_id": owner,
                        "phase": "io_inflight",
                        "lease_id": lease.lease_id,
                    }
                },
            }

        def cancel_d2p_recovery_rank(self, *_args, **_kwargs):
            ledger_state["claims"] = {}
            return True

        def mark_host_load_rank_drained(self, *_args, **_kwargs):
            ledger_state["value"] = HostStageState.FAILED.value
            return True

    load = {
        "record": record,
        "request_generation": request,
        "workset_lease": lease,
        "recovery_claim_id": owner,
        "io_attempt": attempt,
        "io_inflight": True,
        "io_quiesced": False,
        "launch_fence": H2DLaunchFence(
            event=event, submitted=True, armed=True
        ),
    }
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.owner = "p:test"
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.ledger = Ledger()
    manager._state_lock = threading.RLock()
    manager.workset_broker = broker
    manager.tree_cache = SimpleNamespace()
    manager.loads = {rid: load}
    manager.host_ready = {}
    manager._control_wakeup = SimpleNamespace(set=lambda: None)
    manager._h2d_poisoned = False
    released_host = []
    manager._h2d_lane_reservations = {request.snapshot_id: 0}
    manager._release_record = lambda current: released_host.append(current) or True

    req = SimpleNamespace(rid=rid)
    manager.abort_request(rid, request)
    manager.rollback_bound_parent(req, request)
    broker.service(allocator)

    assert lease.state == "io_inflight"
    assert allocator.freed == []
    assert not manager._discard_failed_h2d_load(rid, load)
    broker.service(allocator)
    assert allocator.freed == []

    event.ready = True
    assert manager._discard_failed_h2d_load(rid, load)
    broker.service(allocator)
    assert broker.get(request.snapshot_id, owner=owner) is None
    assert len(allocator.freed) == 1
    assert released_host == [record]
    assert ledger_state["value"] == HostStageState.FAILED.value


def test_tp2_inflight_h2d_abort_waits_for_every_rank_fence():
    class Allocator:
        @staticmethod
        def alloc(count):
            return torch.arange(count, dtype=torch.int64)

        @staticmethod
        def free(_indices):
            return None

    class Event:
        def __init__(self, ready):
            self.ready = ready

        def query(self):
            return self.ready

        def synchronize(self):
            assert self.ready

    ledger, path = _ledger()
    request = RequestGeneration("tp2-inflight-abort", 1)
    snapshot_id = request.snapshot_id
    owner = "p-group:inflight"
    rid = "tp2-inflight-child"
    managers = []
    events = [Event(True), Event(False)]
    released = []

    class Arena:
        def __init__(self, rank):
            self.rank = rank

        def release(self, snapshot):
            # No physical Host shard may be released while another TP rank is
            # still nonterminal.
            assert ledger.get(snapshot_id)["state"] == HostStageState.FAILED.value
            released.append((self.rank, snapshot))
            return True
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 8,
                "byte_size": 4096,
                "tp_size": 2,
            }
        )

        def publish_ready(entries):
            entries[snapshot_id].update(
                p_owner=owner,
                arena_domain=0,
                recovery_domain=0,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_ready, event_snapshot_id=snapshot_id)
        claim_id = AgenticPWorksetLeaseBroker.slow_owner(snapshot_id, rid)
        for rank in range(2):
            broker = AgenticPWorksetLeaseBroker(page_size=4)
            broker.request(snapshot_id, 8, 12, owner=claim_id)
            broker.service(Allocator())
            lease = broker.get(snapshot_id, owner=claim_id)
            assert lease is not None
            assert ledger.claim_d2p_recovery_rank(
                snapshot_id,
                owner,
                tp_rank=rank,
                tp_size=2,
                claim_id=claim_id,
                recovery_domain=0,
            )
            assert ledger.attach_d2p_recovery_lease_rank(
                snapshot_id,
                owner,
                tp_rank=rank,
                tp_size=2,
                claim_id=claim_id,
                lease_id=lease.lease_id,
            )
            attempt = f"slow-h2d:{rank}"
            assert broker.begin_io_attempt(snapshot_id, lease, attempt)
            broker.mark_io_inflight(snapshot_id, lease, attempt)
            assert ledger.mark_d2p_recovery_phase_rank(
                snapshot_id,
                owner,
                tp_rank=rank,
                tp_size=2,
                claim_id=claim_id,
                lease_id=lease.lease_id,
                phase="io_inflight",
            )
            snapshot = object()
            record = {"loading": "h2d", "snapshot": snapshot}
            load = {
                "record": record,
                "request_generation": request,
                "workset_lease": lease,
                "recovery_claim_id": claim_id,
                "io_attempt": attempt,
                "io_inflight": True,
                "io_quiesced": False,
                "launch_fence": H2DLaunchFence(
                    event=events[rank], submitted=True, armed=True
                ),
            }
            manager = AgenticPHostStagingManager.__new__(
                AgenticPHostStagingManager
            )
            manager.owner = owner
            manager.arena_domain = 0
            manager.tp_rank = rank
            manager.tp_size = 2
            manager.ledger = ledger
            manager.arena = Arena(rank)
            manager._state_lock = threading.RLock()
            manager.workset_broker = broker
            manager.loads = {rid: load}
            manager.host_ready = {snapshot_id: record}
            manager._h2d_lane_reservations = {snapshot_id: 0}
            manager._control_wakeup = SimpleNamespace(set=lambda: None)
            manager._h2d_poisoned = False
            managers.append((manager, load))

        for manager, _load in managers:
            manager.abort_request(rid, request)

        assert managers[0][0]._discard_failed_h2d_load(rid, managers[0][1])
        first = ledger.get(snapshot_id)
        assert first["state"] == HostStageState.ABORTING.value
        assert first["loader_drained_ranks"] == [0]
        assert released == []
        assert snapshot_id in managers[0][0].host_ready
        assert not managers[1][0]._discard_failed_h2d_load(rid, managers[1][1])
        assert ledger.get(snapshot_id)["state"] == HostStageState.ABORTING.value
        assert released == []

        events[1].ready = True
        assert managers[1][0]._discard_failed_h2d_load(rid, managers[1][1])
        final = ledger.get(snapshot_id)
        assert final["state"] == HostStageState.FAILED.value
        assert final["loader_drained_ranks"] == [0, 1]
        assert final["recovery_claims"] == {}
        # Production control workers observe this terminal ledger edge on all
        # ranks.  Drive that scan explicitly in the unit test.
        for manager, _load in managers:
            manager._release_consumed_owned_host({snapshot_id: final})
        assert sorted(rank for rank, _snapshot in released) == [0, 1]
        assert all(snapshot_id not in manager.host_ready for manager, _ in managers)
    finally:
        os.unlink(path)


def test_aborted_request_quarantines_direct_dma_without_fake_abort():
    class Allocator:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            raise AssertionError("in-flight DMA pages must remain quarantined")

    class Receiver:
        def abort(self):
            raise AssertionError("local abort is not a physical DMA fence")

    request = RequestGeneration("abort-dma", 1)
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
    )
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner(request.snapshot_id)
    broker.request(request.snapshot_id, 4, 8, owner=owner)
    broker.service(Allocator())
    lease = broker.get(request.snapshot_id, owner=owner)
    assert lease is not None
    attempt = "claim"
    assert broker.begin_io_attempt(request.snapshot_id, lease, attempt)
    broker.mark_io_inflight(request.snapshot_id, lease, attempt)

    req = SimpleNamespace(
        rid="child-rid",
        _agentic_direct_receiver=Receiver(),
        _agentic_direct_manifest=manifest,
        _agentic_direct_workset_lease=lease,
        _agentic_direct_claim_id="claim",
        _agentic_direct_io_attempt=attempt,
        _agentic_direct_indices=torch.arange(4),
        _agentic_direct_started_at=time.monotonic(),
        _agentic_kv_snapshot_store=object(),
    )
    scheduler = SimpleNamespace(
        tree_cache=SimpleNamespace(),
        agentic_p_workset_broker=broker,
        agentic_early_direct_poll_lock=nullcontext(),
        agentic_early_direct_receives={},
    )

    Scheduler._agentic_abort_cleanup(scheduler, req)

    entry = scheduler.agentic_early_direct_receives[request.snapshot_id]
    assert entry.abort_requested
    assert entry.receiver is not None
    assert lease.state == "release_pending"
    assert broker.get(request.snapshot_id, owner=owner) is lease


def test_workset_io_attempt_prevents_stale_quiesce_and_release():
    class Allocator:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            raise AssertionError("active Direct destination must not be freed")

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    owner = broker.direct_owner("race:1")
    broker.request("race:1", 4, 8, owner=owner)
    broker.service(Allocator())
    lease = broker.get("race:1", owner=owner)
    assert lease is not None

    assert broker.begin_io_attempt("race:1", lease, "attempt-a")
    broker.mark_io_inflight("race:1", lease, "attempt-a")
    assert not broker.begin_io_attempt("race:1", lease, "attempt-b")
    assert not broker.mark_io_quiesced("race:1", lease, "attempt-b")
    assert not broker.request_release(
        "race:1", lease, io_attempt="attempt-b"
    )
    assert lease.state == "io_inflight"
    assert lease.io_attempt == "attempt-a"
    assert broker.mark_io_quiesced("race:1", lease, "attempt-a")
    assert lease.state == "active"


def test_transport_release_cannot_reclaim_request_owned_workset():
    class Allocator:
        def __init__(self):
            self.freed = []

        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.freed.append(indices.clone())

    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.request("handed:1", 4, 8)
    broker.service(allocator)
    lease = broker.get("handed:1")
    assert lease is not None
    assert broker.begin_bind("handed:1", lease)
    broker.commit_parent_bound("handed:1", lease)
    req = SimpleNamespace(origin_input_ids=list(range(8)))
    broker.handoff_to_req("handed:1", req, lease)

    assert not broker.request_release("handed:1", lease)
    broker.service(allocator)
    assert broker.get("handed:1") is lease
    assert allocator.freed == []
    assert broker.release_handed("handed:1", lease, req=req)
    broker.service(allocator)
    assert broker.get("handed:1") is None
    assert len(allocator.freed) == 1


def test_workset_lease_release_is_identity_scoped_and_blocks_handoff():
    class Allocator:
        def __init__(self):
            self.next_index = 0
            self.freed = []

        def alloc(self, count):
            result = torch.arange(
                self.next_index, self.next_index + count, dtype=torch.int64
            )
            self.next_index += count
            return result

        def free(self, indices):
            self.freed.append(indices.clone())

    allocator = Allocator()
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.request("same:1", 4, 8)
    broker.service(allocator)
    first = broker.get("same:1")
    assert first is not None
    assert broker.request_release("same:1", first)
    with pytest.raises(RuntimeError, match="disappeared"):
        broker.service(allocator)
        broker.handoff_to_req(
            "same:1", SimpleNamespace(origin_input_ids=list(range(8))), first
        )

    broker.request("same:1", 4, 8)
    broker.service(allocator)
    second = broker.get("same:1")
    assert second is not None and second.lease_id != first.lease_id
    assert not broker.request_release("same:1", first)
    assert not broker.request_release("same:1")
    broker.service(allocator)
    assert broker.get("same:1") is second


def test_workset_bind_ownership_blocks_delayed_io_release():
    class Allocator:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            raise AssertionError("binding workset must not be released")

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.request("bind:1", 4, 8, owner="slow:attempt")
    broker.service(Allocator())
    lease = broker.get("bind:1", owner="slow:attempt")
    assert lease is not None
    assert broker.begin_bind("bind:1", lease)
    assert not broker.request_release("bind:1", lease)
    broker.commit_parent_bound("bind:1", lease)
    req = SimpleNamespace(origin_input_ids=list(range(8)))
    broker.handoff_to_req("bind:1", req, lease)
    assert lease.state == "handed"


def test_workset_owner_isolates_stale_direct_from_slow_restore():
    class Allocator:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            raise AssertionError("stale Direct must not release Slow pages")

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    slow_owner = broker.slow_owner("generation:1", "child-rid")
    direct_owner = broker.direct_owner("generation:1")
    assert broker.request("generation:1", 4, 9, owner=slow_owner)
    broker.service(Allocator())
    slow = broker.get("generation:1", owner=slow_owner)
    assert slow is not None

    assert not broker.request("generation:1", 4, 9, owner=direct_owner)
    assert broker.get("generation:1", owner=direct_owner) is None
    broker.cancel_unstarted("generation:1", owner=direct_owner)
    broker.service(Allocator())
    assert broker.get("generation:1", owner=slow_owner) is slow


def test_workset_handoff_validation_retains_allocator_ownership():
    class Allocator:
        def alloc(self, count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, _indices):
            raise AssertionError("validation must not release behind the caller")

    broker = AgenticPWorksetLeaseBroker(page_size=4)
    allocator = Allocator()
    broker.request("prompt:2", 4, 8)
    broker.service(allocator)
    lease = broker.get("prompt:2")
    assert lease is not None
    assert broker.begin_bind("prompt:2", lease)
    broker.commit_parent_bound("prompt:2", lease)
    with pytest.raises(RuntimeError, match="prompt changed"):
        broker.handoff_to_req(
            "prompt:2", SimpleNamespace(origin_input_ids=list(range(9))), lease
        )
    assert broker.get("prompt:2") is lease
    assert lease.state == "binding"


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


def test_tp_transfer_failure_waits_for_every_rank_physical_terminal():
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        ranks = [
            TPGroupMailbox(
                "transfer-terminal", tp_rank=rank, tp_size=2, directory=directory
            )
            for rank in range(2)
        ]
        key = request_generation_key("transfer", 9)
        ranks[0].publish_local(key, int(KVPoll.Failed))
        ranks[1].publish_local(key, int(KVPoll.Transferring))

        status, cancel = ranks[0].transfer_group_status(key)
        assert status == int(KVPoll.Transferring)
        assert cancel is True

        ranks[1].publish_local(key, int(KVPoll.Failed))
        status, cancel = ranks[0].transfer_group_status(key)
        assert status == int(KVPoll.Failed)
        assert cancel is True


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


def test_tp_decode_group_commit_waits_for_receiver_lifecycle_lock():
    """A TP rank may not skip a group-committed transfer locally."""

    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue._async_progress_enabled = True
    queue._async_poll_lock = threading.Lock()
    queue.scheduler = SimpleNamespace(tp_size=2)

    entered_commit = threading.Event()
    queue._pop_transferred_locked = lambda _keys=None: entered_commit.set() or [1]
    queue._async_poll_lock.acquire()

    result = []
    commit_thread = threading.Thread(
        target=lambda: result.extend(queue.pop_transferred([("rid", 1)]))
    )
    commit_thread.start()
    time.sleep(0.02)
    assert not entered_commit.is_set()

    queue._async_poll_lock.release()
    commit_thread.join(timeout=2.0)
    assert not commit_thread.is_alive()
    assert entered_commit.is_set()
    assert result == [1]


@pytest.mark.parametrize("poll", [KVPoll.Transferring, KVPoll.Success])
def test_async_prealloc_poll_accepts_receiver_progress_after_bind(poll):
    """A stale prealloc snapshot may observe Host H2D already in progress."""

    decode_req = SimpleNamespace(
        waiting_for_input=False,
        kv_receiver=SimpleNamespace(poll=lambda: poll),
        req=SimpleNamespace(rid="host-race", bootstrap_room=17),
    )
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.queue = [decode_req]
    queue._async_progress_enabled = True

    queue._update_handshake_waiters()
    assert decode_req.waiting_for_input is False


def test_async_prealloc_never_repolls_waiting_receiver():
    def stale_poll():
        raise AssertionError("receiver ownership already moved to transfer queue")

    waiting = SimpleNamespace(
        waiting_for_input=True,
        kv_receiver=SimpleNamespace(poll=stale_poll),
        req=SimpleNamespace(rid="waiting", bootstrap_room=1),
    )
    pending = SimpleNamespace(
        waiting_for_input=False,
        kv_receiver=SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
        req=SimpleNamespace(
            rid="pending",
            bootstrap_room=2,
            time_stats=SimpleNamespace(set_bootstrap_done_time=lambda: None),
        ),
    )
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.queue = [waiting, pending]
    queue._async_progress_enabled = True

    queue._update_handshake_waiters()
    assert waiting.waiting_for_input is True
    assert pending.waiting_for_input is True


def test_async_prealloc_metadata_cannot_block_readiness_control():
    metadata_started = threading.Event()
    release_metadata = threading.Event()
    control_steps = []

    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue._async_progress_enabled = True
    queue._async_control_next_at = 0.0
    queue._async_control_interval = 0.0
    queue._resolve_pending_reqs = lambda: control_steps.append("resolve")
    queue._update_handshake_waiters = lambda: control_steps.append("handshake")
    queue._background_update_p_ready = lambda: control_steps.append("p_ready")
    queue._publish_tp_admission_readiness = lambda: control_steps.append("tp_ready")

    def blocked_metadata():
        metadata_started.set()
        assert release_metadata.wait(timeout=2.0)

    queue._background_prepare_metadata = blocked_metadata
    metadata_thread = threading.Thread(target=queue.background_metadata_progress)
    metadata_thread.start()
    assert metadata_started.wait(timeout=2.0)

    queue.background_control_progress()
    assert control_steps == ["resolve", "handshake", "p_ready", "tp_ready"]

    release_metadata.set()
    metadata_thread.join(timeout=2.0)
    assert not metadata_thread.is_alive()


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


def test_tp_generation_producer_election_is_safe_when_follower_arrives_first():
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        store = AgenticEarlyClaimStore(directory)
        request = RequestGeneration("follower-first", 2)
        logical_tp_owner = "decode-0:rid"

        # Any rank may atomically publish the logical TP-engine owner.  Every
        # peer joins the identical owner; a different D engine loses.
        assert store.claim_generation_producer(request, logical_tp_owner)
        assert store.claim_generation_producer(request, logical_tp_owner)
        assert not store.claim_generation_producer(request, "decode-1:rid")


def test_tp_finished_snapshot_follower_first_retains_both_physical_shards(
    monkeypatch,
):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "decode-0")
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        producer_store = AgenticEarlyClaimStore(directory)
        metadata = AgenticRequestMetadata(
            request_id="follower-first-offload",
            generation=1,
            tool_suffix_token_ids=((9,),),
        )
        published = []

        def manager(rank):
            owner = SimpleNamespace(
                tp_world_size=2,
                tp_rank=rank,
                page_size=4,
                agentic_early_claim_store=producer_store,
                agentic_direct_runtime=object(),
                agentic_hostless=True,
            )
            owner._publish_agentic_direct_candidate = (
                lambda req, _metadata, _tokens: published.append((rank, req.rid))
                or True
            )
            return owner

        def req():
            return SimpleNamespace(
                rid="same-tp-request",
                origin_input_ids=[1, 2, 3, 4],
                output_ids=[5, 9],
                tokenizer=None,
                finished_reason=None,
                finished=lambda: True,
            )

        # Rank 1 wins the filesystem race, but both ranks use the same logical
        # TP owner and therefore retain their own physical KV shard.
        assert DecodeKVCacheOffloadManager._offload_agentic_finished_snapshot(
            manager(1), req(), metadata
        )
        assert DecodeKVCacheOffloadManager._offload_agentic_finished_snapshot(
            manager(0), req(), metadata
        )
        assert published == [(1, "same-tp-request"), (0, "same-tp-request")]


def _direct_setup_manager(*, tp_rank, tp_size, publish_offer, publish_route):
    class Sender:
        def __init__(self, **_kwargs):
            pass

        def poll(self):
            return KVPoll.Bootstrapping

    manager = SimpleNamespace(
        tp_rank=tp_rank,
        tp_world_size=tp_size,
        page_size=4,
        agentic_fast_threshold=2.0,
        agentic_direct_runtime=SimpleNamespace(
            bootstrap_addr="127.0.0.1:1",
            layout_hash="layout",
            kv_pool=object(),
            manager=object(),
            sender_class=Sender,
        ),
        agentic_snapshot_store=SimpleNamespace(
            publish_direct_offer=publish_offer,
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int64).view(1, 64),
        ),
        agentic_direct_candidates={},
        _agentic_candidates_lock=threading.RLock(),
        _agentic_tp_pending_candidate_commands={},
        _publish_agentic_route=publish_route,
        wake_decode_io_progress=lambda: None,
    )
    manager._apply_tp_candidate_command = lambda command: (
        DecodeKVCacheOffloadManager._apply_tp_candidate_command(manager, command)
    )
    manager._agentic_candidate_items = lambda: (
        tuple(manager.agentic_direct_candidates.items())
    )
    return manager


def _install_direct_setup_candidate(manager, request_id="direct-setup"):
    request = RequestGeneration(request_id, 1)
    metadata = SimpleNamespace(current=request, tool_type="tool")
    req = SimpleNamespace(req_pool_idx=0, rid=f"{request_id}-rid")
    assert DecodeKVCacheOffloadManager._publish_agentic_direct_candidate(
        manager, req, metadata, list(range(8))
    )
    return manager.agentic_direct_candidates[request.snapshot_id]


def test_direct_setup_route_failure_retains_parent_and_retries(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.debug_kv_digest",
        lambda *_args: None,
    )
    routes = [False, True]
    offers = []
    manager = _direct_setup_manager(
        tp_rank=0,
        tp_size=1,
        publish_offer=lambda manifest: offers.append(manifest.snapshot_id),
        publish_route=lambda *_args, **_kwargs: routes.pop(0),
    )
    candidate = _install_direct_setup_candidate(manager)

    assert not DecodeKVCacheOffloadManager._progress_agentic_direct_candidate_setup(
        manager, candidate, time.monotonic()
    )
    assert candidate["local_prepared"]
    assert candidate["offer_published"]
    assert not candidate["route_published"]
    assert candidate["req"].req_pool_idx == 0
    assert len(offers) == 1

    candidate["setup_retry_at"] = 0.0
    assert DecodeKVCacheOffloadManager._progress_agentic_direct_candidate_setup(
        manager, candidate, time.monotonic()
    )
    assert candidate["setup_committed"]
    assert candidate["route_published"]
    # Retrying the route never republishes or discards the immutable offer.
    assert len(offers) == 1


def test_direct_setup_offer_exception_retains_parent_and_retries(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.debug_kv_digest",
        lambda *_args: None,
    )
    attempts = []

    def publish_offer(manifest):
        attempts.append(manifest.snapshot_id)
        if len(attempts) == 1:
            raise OSError("transient metadata failure")

    routes = []
    manager = _direct_setup_manager(
        tp_rank=0,
        tp_size=1,
        publish_offer=publish_offer,
        publish_route=lambda *_args, **_kwargs: routes.append(True) or True,
    )
    candidate = _install_direct_setup_candidate(manager, "offer-retry")

    assert not DecodeKVCacheOffloadManager._progress_agentic_direct_candidate_setup(
        manager, candidate, time.monotonic()
    )
    assert candidate["local_prepared"]
    assert not candidate["offer_published"]
    assert routes == []
    assert candidate["req"].req_pool_idx == 0

    candidate["setup_retry_at"] = 0.0
    assert DecodeKVCacheOffloadManager._progress_agentic_direct_candidate_setup(
        manager, candidate, time.monotonic()
    )
    assert candidate["setup_committed"]
    assert len(attempts) == 2
    assert routes == [True]


def test_tp_direct_setup_failure_keeps_both_shards_on_rank_zero_decision(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.debug_kv_digest",
        lambda *_args: None,
    )
    publish_attempts = []

    def publish_offer(manifest):
        publish_attempts.append(manifest.snapshot_id)
        if len(publish_attempts) == 1:
            raise OSError("transient rank-zero publication failure")

    rank0 = _direct_setup_manager(
        tp_rank=0,
        tp_size=2,
        publish_offer=publish_offer,
        publish_route=lambda *_args, **_kwargs: True,
    )
    rank1 = _direct_setup_manager(
        tp_rank=1,
        tp_size=2,
        publish_offer=lambda _manifest: (_ for _ in ()).throw(
            AssertionError("follower must not publish the logical offer")
        ),
        publish_route=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("follower must not choose a route")
        ),
    )
    candidate0 = _install_direct_setup_candidate(rank0, "tp-offer-retry")
    candidate1 = _install_direct_setup_candidate(rank1, "tp-offer-retry")

    assert not DecodeKVCacheOffloadManager._progress_agentic_direct_candidate_setup(
        rank0, candidate0, time.monotonic()
    )
    assert DecodeKVCacheOffloadManager._progress_agentic_direct_candidate_setup(
        rank1, candidate1, time.monotonic()
    )
    assert DecodeKVCacheOffloadManager.tp_candidate_commands(rank0) == [
        {"snapshot_id": "tp-offer-retry:1", "action": "wait"}
    ]
    assert candidate0["req"].req_pool_idx == candidate1["req"].req_pool_idx == 0

    candidate0["setup_retry_at"] = 0.0
    assert DecodeKVCacheOffloadManager._progress_agentic_direct_candidate_setup(
        rank0, candidate0, time.monotonic()
    )
    assert candidate0["setup_committed"] and candidate1["setup_committed"]
    assert publish_attempts == ["tp-offer-retry:1", "tp-offer-retry:1"]


def test_final_confirmation_outranks_permanently_failing_direct_setup():
    snapshot_id = "final-before-setup:1"
    candidate = {
        "req": SimpleNamespace(req_pool_idx=0),
        "metadata": SimpleNamespace(current=RequestGeneration("final-before-setup", 1)),
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "staging": False,
        "setup_committed": False,
    }
    completed = []
    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_fast_threshold=2.0,
        agentic_relay_worker=None,
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: True,
        _agentic_complete_final_candidate=lambda value, _now: completed.append(value)
        or True,
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False, progress_class="direct"
    )
    assert completed == [candidate]


def test_slow_path_selects_largest_remaining_host_arena(monkeypatch):
    with tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False) as f:
        json.dump(
            {
                "published_at": time.time(),
                "domains": [
                    {
                        "domain": 0,
                        "pending_tokens": 30000,
                        "hbm_used_tokens": 1000,
                        "hbm_capacity_tokens": 100000,
                        "arena_used_bytes": 80,
                        "arena_capacity_bytes": 100,
                        "pending_requests": 10,
                        "scheduler_waiting": 10,
                    },
                    {
                        "domain": 1,
                        "pending_tokens": 5000,
                        "hbm_used_tokens": 99000,
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
        domain, numa_nodes = DecodeKVCacheOffloadManager._select_slow_host_domain(
            manager, byte_size=20
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


def test_d2p_host_source_is_retained_until_all_radix_binds():
    ledger, path = _ledger()
    snapshot_id = "request:d2p-bind"
    owner = "p-group:p0"
    try:
        for rank in range(2):
            offer = _rank_offer(rank)
            offer["snapshot_id"] = snapshot_id
            offer["request_direction"] = "d2p"
            ledger.offer(offer)
        for rank in range(2):
            assert ledger.claim_rank(
                snapshot_id, owner, tp_rank=rank, tp_size=2
            )
        for rank in range(2):
            assert ledger.publish_rank_grant(
                snapshot_id,
                owner,
                {
                    "kind": "shared_host_extent",
                    "arena_path": f"/dev/shm/d2p-rank-{rank}",
                    "byte_size": 1024,
                    "token_count": 128,
                },
                tp_rank=rank,
                tp_size=2,
            )
        for rank in range(2):
            assert ledger.complete_host_write(
                snapshot_id, 100 + rank, tp_rank=rank, tp_size=2
            )
        for rank in range(2):
            assert ledger.prepare_tp_host_load_rank(
                snapshot_id, owner, tp_rank=rank, tp_size=2
            )
        assert ledger.complete_d2p_host_load_rank(
            snapshot_id, owner, tp_rank=0, tp_size=2
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.H2D_LOADING.value
        assert ledger.complete_d2p_host_load_rank(
            snapshot_id, owner, tp_rank=1, tp_size=2
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.HBM_READY.value
        assert ledger.complete_host_bind_rank(
            snapshot_id, owner, tp_rank=0, tp_size=2
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.HBM_READY.value
        assert ledger.complete_host_bind_rank(
            snapshot_id, owner, tp_rank=1, tp_size=2
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.CONSUMED.value
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def test_d2p_host_owner_and_recovery_p_owner_are_independent():
    ledger, path = _ledger()
    snapshot_id = "request:remote-recovery"
    host_owner = "p-group:host-p0"
    recovery_owner = "p-group:compute-p1"
    claim_id = "slow:remote-recovery"
    try:
        for rank in range(2):
            offer = _rank_offer(rank)
            offer["snapshot_id"] = snapshot_id
            offer["arena_domain"] = 0
            ledger.offer(offer)
        for rank in range(2):
            assert ledger.claim_rank(
                snapshot_id, host_owner, tp_rank=rank, tp_size=2
            )
            assert ledger.publish_rank_grant(
                snapshot_id,
                host_owner,
                {
                    "kind": "shared_host_extent",
                    "arena_path": f"/dev/shm/remote-rank-{rank}",
                    "arena_offset": 0,
                    "byte_size": 1024,
                    "token_count": 128,
                },
                tp_rank=rank,
                tp_size=2,
            )
        for rank in range(2):
            assert ledger.complete_host_write(
                snapshot_id, 100 + rank, tp_rank=rank, tp_size=2
            )

        assert ledger.assign_d2p_recovery_domain(snapshot_id, 1)
        for rank in range(2):
            assert ledger.claim_d2p_recovery_rank(
                snapshot_id,
                recovery_owner,
                tp_rank=rank,
                tp_size=2,
                claim_id=claim_id,
                recovery_domain=1,
            )
            assert ledger.attach_d2p_recovery_lease_rank(
                snapshot_id,
                recovery_owner,
                tp_rank=rank,
                tp_size=2,
                claim_id=claim_id,
                lease_id=1000 + rank,
            )
            assert ledger.prepare_tp_host_load_rank(
                snapshot_id, recovery_owner, tp_rank=rank, tp_size=2
            )
        for rank in range(2):
            assert ledger.complete_d2p_host_load_rank(
                snapshot_id, recovery_owner, tp_rank=rank, tp_size=2
            )
        for rank in range(2):
            assert ledger.complete_host_bind_rank(
                snapshot_id, recovery_owner, tp_rank=rank, tp_size=2
            )

        terminal = ledger.get(snapshot_id)
        assert terminal["state"] == HostStageState.CONSUMED.value
        assert terminal["p_owner"] == host_owner
        assert terminal["recovery_owner"] == recovery_owner
        assert terminal["recovery_domain"] == 1
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def test_remote_recovery_p_maps_host_owner_extent_without_taking_arena_lease():
    with tempfile.NamedTemporaryFile(dir="/dev/shm") as extent:
        extent.truncate(mmap.ALLOCATIONGRANULARITY)
        manager = object.__new__(AgenticPHostStagingManager)
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.owner = "p:compute"
        manager.arena_domain = 1
        manager.device_pool = SimpleNamespace()
        manager.host_ready = {}
        manager._state_lock = threading.RLock()
        entry = {
            "snapshot_id": "request:remote-map",
            "state": HostStageState.HOST_READY.value,
            "p_owner": "p:host",
            "arena_domain": 0,
            "recovery_domain": 1,
            "token_count": 64,
            "token_digest": "digest",
            "byte_size": mmap.ALLOCATIONGRANULARITY,
            "grants": [
                {
                    "arena_path": extent.name,
                    "arena_offset": 0,
                    "byte_size": mmap.ALLOCATIONGRANULARITY,
                    "token_count": 64,
                }
            ],
        }

        record = manager._import_remote_host_record(entry["snapshot_id"], entry)

        assert record is manager.host_ready[entry["snapshot_id"]]
        assert record["remote_host"] is True
        assert record["snapshot"].path == extent.name
        assert manager._release_record(record)
        assert "snapshot" not in record


def test_host_owner_releases_extent_after_remote_recovery_consumed():
    released = []
    snapshot = object()
    manager = object.__new__(AgenticPHostStagingManager)
    manager.owner = "p:host"
    manager.arena_domain = 0
    manager._state_lock = threading.RLock()
    manager.host_ready = {
        "request:remote-consumed": {
            "snapshot": snapshot,
            "loading": False,
            "offer": {"token_count": 64},
        }
    }
    manager.arena = SimpleNamespace(
        release=lambda value: released.append(value) or True
    )
    entries = {
        "request:remote-consumed": {
            "state": HostStageState.CONSUMED.value,
            "p_owner": "p:host",
            "arena_domain": 0,
            "recovery_domain": 1,
        }
    }

    manager._release_consumed_owned_host(entries)

    assert released == [snapshot]
    assert manager.host_ready == {}


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


def test_tp1_p2d_host_uses_same_rank_grant_commit_protocol():
    ledger, path = _ledger()
    snapshot_id = "p2d:42"
    owner = "p2d-p-group:prefill-0"
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 42,
                "token_count": 128,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": 1,
            }
        )
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=0, tp_size=1)
        assert ledger.publish_rank_grant(
            snapshot_id,
            owner,
            {
                "kind": "shared_host_extent",
                "arena_path": "/dev/shm/p2d-tp1-rank-0",
                "byte_size": 1024,
                "token_count": 128,
            },
            tp_rank=0,
            tp_size=1,
        )
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.HOST_WRITING.value
        assert list(current["rank_grants"]) == ["0"]

        assert ledger.complete_p2d_host_write_rank(
            snapshot_id, owner, tp_rank=0, tp_size=1
        )
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.HOST_READY.value
        assert current["writer_acks"] == [0]
        assert len(current["grants"]) == 1
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@pytest.mark.parametrize("tp_size", [1, 2])
def test_p2d_claim_and_grant_are_one_ownership_transaction(tp_size):
    ledger, path = _ledger()
    snapshot_id = f"p2d:atomic-{tp_size}"
    owner = "p2d-p-group:prefill-0"
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 420 + tp_size,
                "token_count": 128,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": tp_size,
            }
        )
        grants = []
        for rank in range(tp_size):
            grant = {
                "kind": "shared_host_extent",
                "arena_path": f"/dev/shm/p2d-atomic-rank-{rank}",
                "arena_offset": rank * 1024,
                "byte_size": 1024,
                "token_count": 128,
            }
            grants.append(grant)
            assert ledger.prepare_p2d_write_rank(
                snapshot_id,
                owner,
                grant,
                tp_rank=rank,
                tp_size=tp_size,
            )
        for rank, grant in enumerate(grants):
            claimed = ledger.claim_p2d_write_rank(
                snapshot_id,
                owner,
                grant,
                tp_rank=rank,
                tp_size=tp_size,
            )
            assert claimed is not None
            current = ledger.get(snapshot_id)
            assert current["state"] == (
                HostStageState.HOST_WRITING.value
                if rank + 1 == tp_size
                else HostStageState.HOST_RESERVED.value
            )
            assert current["p_owner"] == owner
            assert current["claimed_ranks"] == list(range(rank + 1))
            assert len(current["rank_grants"]) == rank + 1
            # No native selector may terminate a snapshot after any physical
            # Host extent became the exclusive owner of its P KV shard.
            assert not ledger.reject_unclaimed_offer(
                snapshot_id, reason="late_native_race"
            )
            assert (
                ledger.arbitrate_p2d_release(snapshot_id, tp_size=tp_size)
                == P2D_RELEASE_HOST_OWNED
            )

        for rank in range(tp_size):
            assert ledger.complete_p2d_host_write_rank(
                snapshot_id, owner, tp_rank=rank, tp_size=tp_size
            )
        assert ledger.get(snapshot_id)["state"] == HostStageState.HOST_READY.value
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@pytest.mark.parametrize("tp_size", [1, 2])
def test_duplicate_control_offer_never_regresses_owned_p2d_state(tp_size):
    ledger, path = _ledger()
    snapshot_id = f"p2d:duplicate-control-{tp_size}"
    owner = "p2d-p-group:prefill-0"
    control = {
        "snapshot_id": snapshot_id,
        "bootstrap_room": 500 + tp_size,
        "token_count": 128,
        "prefill_domain": 0,
        "request_direction": "p2d",
        "control_offer": True,
        "tp_size": tp_size,
    }
    try:
        ledger.offer(control)
        grants = []
        for rank in range(tp_size):
            grant = {
                "kind": "shared_host_extent",
                "arena_path": f"/dev/shm/p2d-control-rank-{rank}",
                "arena_offset": rank * 1024,
                "byte_size": 1024,
                "token_count": 128,
            }
            grants.append(grant)
            assert ledger.prepare_p2d_write_rank(
                snapshot_id,
                owner,
                grant,
                tp_rank=rank,
                tp_size=tp_size,
            )
        for rank, grant in enumerate(grants):
            assert ledger.claim_p2d_write_rank(
                snapshot_id,
                owner,
                grant,
                tp_rank=rank,
                tp_size=tp_size,
            )
            before = ledger.get(snapshot_id)
            replay = ledger.offer(control)
            after = ledger.get(snapshot_id)
            assert replay == before
            assert after == before
            assert after["rank_offers"] == {}

        for rank in range(tp_size):
            assert ledger.complete_p2d_host_write_rank(
                snapshot_id, owner, tp_rank=rank, tp_size=tp_size
            )
        ready = ledger.get(snapshot_id)
        assert ready["state"] == HostStageState.HOST_READY.value
        assert ledger.offer(control) == ready
        assert ledger.get(snapshot_id) == ready
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@pytest.mark.parametrize("rank_order", [(0, 1), (1, 0)])
def test_tp_p2d_managers_join_atomic_host_transaction_in_any_order(rank_order):
    ledger, path = _ledger()
    snapshot_id = "p2d:4242"
    owner = "p2d-p-group:prefill-test"

    class Arena:
        capacity_bytes = 1 << 20
        used_bytes = 0

        def __init__(self, rank):
            self.rank = rank

        def can_reserve(self, *_args):
            return True

        def create(self, *_args):
            return SimpleNamespace(
                path=f"/dev/shm/p2d-manager-rank-{self.rank}",
                offset=self.rank * 4096,
            )

        def release(self, _snapshot):
            pass

    def manager(rank):
        value = AgenticPToDHostStagingManager.__new__(
            AgenticPToDHostStagingManager
        )
        value.ledger = ledger
        value.device_pool = SimpleNamespace(
            layer_num=1,
            head_num=1,
            head_dim=1,
            store_dtype=torch.uint8,
        )
        value.prefill_domain = 0
        value.numa_node = rank
        value.tp_rank = rank
        value.tp_size = 2
        value.owner = owner
        value.hard_watermark = 1.0
        value.arena = Arena(rank)
        value._lock = threading.RLock()
        value._prepared = {}
        value._active = {}
        value._results = {}
        value._records = {}
        value._work = queue.SimpleQueue()
        return value

    managers = [manager(0), manager(1)]
    reqs = [
        SimpleNamespace(
            bootstrap_room=4242,
            origin_input_ids=[1, 2],
            output_ids=[3],
            return_logprob=False,
            cached_tokens=0,
        )
        for _ in range(2)
    ]
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 4242,
                "token_count": 2,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": 2,
            }
        )
        first, second = rank_order
        assert managers[first].has_offer(reqs[first])
        # A single prepared shard owns only a tentative Host extent.  It must
        # not take P-KV ownership before every TP peer has capacity.
        assert not managers[first].try_submit(
            reqs[first], torch.tensor([0, 1])
        )
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.OFFERED.value
        assert current["prepared_ranks"] == [first]
        assert not current.get("claimed_ranks")

        assert managers[second].has_offer(reqs[second])
        assert managers[second].try_submit(reqs[second], torch.tensor([0, 1]))
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.HOST_RESERVED.value
        assert current["claimed_ranks"] == [second]

        # The first rank's offer worker retries after its peer publishes the
        # final extent and joins the group-owned transaction.
        assert managers[first].try_submit(reqs[first], torch.tensor([0, 1]))
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.HOST_WRITING.value
        assert current["claimed_ranks"] == [0, 1]
        assert len(current["rank_grants"]) == 2
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def test_tp_p2d_capacity_failure_rejects_before_any_rank_owns_p_kv():
    ledger, path = _ledger()
    snapshot_id = "p2d:4243"
    owner = "p2d-p-group:prefill-test"
    released = []

    class Arena:
        capacity_bytes = 1 << 20
        used_bytes = 0

        def __init__(self, rank, has_capacity):
            self.rank = rank
            self.has_capacity = has_capacity

        def can_reserve(self, *_args):
            return self.has_capacity

        def create(self, *_args):
            return SimpleNamespace(
                path=f"/dev/shm/p2d-capacity-rank-{self.rank}",
                offset=self.rank * 4096,
            )

        def release(self, snapshot):
            released.append((self.rank, snapshot.path))

    def manager(rank, has_capacity):
        value = AgenticPToDHostStagingManager.__new__(
            AgenticPToDHostStagingManager
        )
        value.ledger = ledger
        value.device_pool = SimpleNamespace(
            layer_num=1,
            head_num=1,
            head_dim=1,
            store_dtype=torch.uint8,
        )
        value.prefill_domain = 0
        value.numa_node = rank
        value.tp_rank = rank
        value.tp_size = 2
        value.owner = owner
        value.hard_watermark = 1.0
        value.arena = Arena(rank, has_capacity)
        value._lock = threading.RLock()
        value._prepared = {}
        value._active = {}
        value._results = {}
        value._records = {}
        value._candidates = {}
        value._work = queue.SimpleQueue()
        return value

    reqs = [
        SimpleNamespace(
            bootstrap_room=4243,
            origin_input_ids=[1, 2],
            output_ids=[3],
            return_logprob=False,
            cached_tokens=0,
        )
        for _ in range(2)
    ]
    managers = [manager(0, True), manager(1, False)]
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 4243,
                "token_count": 2,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": 2,
            }
        )
        assert not managers[0].try_submit(reqs[0], torch.tensor([0, 1]))
        assert not managers[1].try_submit(reqs[1], torch.tensor([0, 1]))
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.REJECTED.value
        assert current.get("p_owner") is None
        assert not current.get("claimed_ranks")

        # The peer that tentatively reserved an extent can now return it and
        # release its untouched P pages through the ordinary native path.
        assert managers[0].cancel_watch(reqs[0])
        assert released == [(0, "/dev/shm/p2d-capacity-rank-0")]
        assert managers[0]._prepared == {}
        assert managers[0]._active == {}
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def test_tp_d2p_failure_and_drain_are_rank_aware():
    ledger, path = _ledger()
    snapshot_id = "request:rank-failure"
    owner = "p-group:p0"
    try:
        for rank, pid in ((0, 100), (1, 101)):
            offer = _rank_offer(rank)
            offer["snapshot_id"] = snapshot_id
            offer["d_pid"] = pid
            ledger.offer(offer)
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=0, tp_size=2)
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=1, tp_size=2)

        assert ledger.fail_host_write(
            snapshot_id,
            101,
            "rank1_failed",
            tp_rank=1,
            tp_size=2,
        )
        current = ledger.get(snapshot_id)
        assert current["state"] == HostStageState.ABORTING.value
        assert current["writer_drained_ranks"] == [1]
        assert current["writer_drained"] is False

        assert ledger.mark_writer_rank_drained(
            snapshot_id, 100, tp_rank=0, tp_size=2
        )
        current = ledger.get(snapshot_id)
        assert current["writer_drained_ranks"] == [0, 1]
        assert current["writer_drained"] is True
        assert ledger.transition(
            snapshot_id, HostStageState.FAILED, owner=owner
        )
    finally:
        os.unlink(path)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (HostStageState.FAILED, "failed"),
        (HostStageState.H2D_LOADING, "host_ready"),
        (HostStageState.CONSUMED, "host_ready"),
    ],
)
def test_tp_d2p_completed_rank_observes_group_terminal_state(state, expected):
    snapshot_id = "request:completed-rank"
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "rank_host_write_complete": True,
    }
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": state.value}
    )
    client._cleanup_write = lambda _candidate: True
    client._cleanup_relay_senders = lambda _candidate: None

    assert client.progress(candidate, []) == expected


def test_tp_d2p_completed_rank_drains_when_peer_aborts():
    snapshot_id = "request:completed-rank-abort"
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "rank_host_write_complete": True,
    }
    drained = []
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.tp_rank = 1
    client.tp_size = 2
    client.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": HostStageState.ABORTING.value},
        mark_writer_rank_drained=lambda *args, **kwargs: drained.append(
            (args, kwargs)
        )
        or True,
    )
    client._cleanup_write = lambda _candidate: True

    assert client.progress(candidate, []) == "waiting"
    assert len(drained) == 1
    assert drained[0][1] == {"tp_rank": 1, "tp_size": 2}


def test_tp_d2p_abort_drain_ack_error_retains_d_source_for_retry():
    snapshot_id = "request:abort-drain-retry"
    candidate = {"manifest": SimpleNamespace(snapshot_id=snapshot_id)}
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.tp_rank = 0
    client.tp_size = 2
    client.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": HostStageState.ABORTING.value},
        mark_writer_rank_drained=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("transient ledger ACK failure")
        ),
    )
    client._cleanup_write = lambda _candidate: True

    # "waiting" keeps the Decode candidate and its source pages live; a
    # terminal "failed" result would let the caller discard the only copy.
    assert client.progress(candidate, []) == "waiting"


def test_d2p_missing_active_ledger_entry_fails_closed_without_recompute():
    snapshot_id = "request:missing-ledger"
    candidate = {"manifest": SimpleNamespace(snapshot_id=snapshot_id)}
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.ledger = SimpleNamespace(get=lambda _snapshot_id: None)
    client._cleanup_write = lambda _candidate: True

    assert client.progress(candidate, []) == "waiting"


def test_d2p_abort_quarantines_unfenced_partial_dma_before_any_release():
    snapshot_id = "slow:unfenced-abort"

    class Snapshot:
        def __init__(self):
            self.closed = False

        def close(self, *, unlink):
            self.closed = True

    snapshot = Snapshot()
    launch_fence = H2DLaunchFence(event=SimpleNamespace(query=lambda: False))
    launch_fence.submitted = True
    launch_fence.unavailable = True
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "arena_write": {
            "snapshot": snapshot,
            "chunks": {
                0: {
                    "event": None,
                    "copy_refs": [object()],
                    "start": 0,
                    "end": 1,
                    "phase": "dma",
                    "launch_fence": launch_fence,
                }
            },
            "next_offset": 1,
            "retry_ranges": [],
            "committed_tokens": 0,
            "gpu_elapsed_ms": 0.0,
        },
    }
    drained = []
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.tp_rank = 0
    client.tp_size = 1
    client._d2h_dma_quarantine = []
    client._d2h_lanes = [
        {"snapshot_id": snapshot_id, "host_bounce": object(), "phase": "dma"}
    ]
    client.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": HostStageState.ABORTING.value},
        mark_writer_rank_drained=lambda *_args, **_kwargs: drained.append(True),
    )

    assert client.progress(candidate, []) == "waiting"
    assert candidate.get("arena_write") is not None
    assert snapshot.closed is False
    assert client._d2h_lanes[0]["snapshot_id"] == snapshot_id
    assert len(client._d2h_dma_quarantine) == 1
    assert drained == []


def test_d2p_cleanup_retries_unregister_before_releasing_lane():
    snapshot_id = "slow:unregister-retry"

    class Snapshot:
        def __init__(self):
            self.close_calls = 0

        def close(self, *, unlink):
            assert unlink is False
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected cudaHostUnregister failure")

    snapshot = Snapshot()
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "arena_write": {
            "snapshot": snapshot,
            "chunks": {0: {"event": SimpleNamespace(query=lambda: True)}},
        },
    }
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client._d2h_lanes = [{"snapshot_id": snapshot_id, "phase": "cpu"}]

    assert client._cleanup_write(candidate) is False
    assert candidate.get("arena_write") is not None
    assert client._d2h_lanes[0] == {"snapshot_id": snapshot_id, "phase": "cpu"}

    assert client._cleanup_write(candidate) is True
    assert candidate.get("arena_write") is None
    assert client._d2h_lanes[0] == {"snapshot_id": None, "phase": "free"}


def test_d2p_slow_writer_uses_independent_copy_lanes():
    class Snapshot:
        def __init__(self):
            self.started = []

        def start_backup_range_from_device(self, indices, **kwargs):
            self.started.append((indices.clone(), kwargs))
            return object(), object()

    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client._d2h_chunk_tokens = 4
    client._d2h_bounce_depth = 2
    client._d2h_dma_limit = 2
    client._d2h_lanes = [
        {
            "stream": object(),
            "staging": object(),
            "host_bounce": object(),
            "snapshot_id": None,
            "phase": "free",
            "dma_group": index // 2,
        }
        for index in range(4)
    ]

    def candidate(snapshot_id):
        return {
            "manifest": SimpleNamespace(snapshot_id=snapshot_id),
            "arena_write": {
                "snapshot": Snapshot(),
                "chunks": {},
                "next_offset": 0,
                "retry_ranges": [],
                "committed_tokens": 0,
                "gpu_elapsed_ms": 0.0,
            },
        }

    first = candidate("slow:1")
    second = candidate("slow:2")
    third = candidate("slow:3")
    fourth = candidate("slow:4")
    indices = torch.arange(8)

    assert client._start_write_chunk(first, indices)
    first_lane = next(iter(first["arena_write"]["chunks"]))
    # DMA(1) completed and its bounce is now CPU-owned; group 0's stream and
    # HBM gather buffer may feed its alternate bounce for the *same snapshot*.
    first["arena_write"]["chunks"][first_lane]["phase"] = "cpu"
    client._d2h_lanes[first_lane]["phase"] = "cpu"
    assert client._start_write_chunk(first, indices)
    assert tuple(first["arena_write"]["chunks"]) == (0, 1)
    # A second snapshot still uses the other physical DMA group.  No third
    # DMA may start while both groups are occupied.
    assert client._start_write_chunk(second, indices)
    assert not client._start_write_chunk(third, indices)
    assert not client._start_write_chunk(fourth, indices)
    assert tuple(second["arena_write"]["chunks"]) == (2,)


def test_d2p_slow_control_uses_sticky_bounded_round_robin_window():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._agentic_candidates_lock = threading.RLock()
    manager.agentic_direct_candidates = {
        f"slow:{index}": {"staging": True} for index in range(10)
    }
    manager._agentic_slow_active_ids = {}
    manager._agentic_slow_progress_cursor = 0
    manager._agentic_slow_active_limit = 6
    manager._agentic_slow_progress_budget = 2
    manager.agentic_host_staging_client = SimpleNamespace(max_active_writes=6)

    slices = [
        tuple(
            snapshot_id
            for snapshot_id, _ in manager._agentic_bounded_slow_candidate_items()
        )
        for _ in range(3)
    ]
    assert slices == [
        ("slow:0", "slow:1"),
        ("slow:2", "slow:3"),
        ("slow:4", "slow:5"),
    ]
    assert tuple(manager._agentic_slow_active_ids) == tuple(
        f"slow:{index}" for index in range(6)
    )

    manager._agentic_candidate_pop("slow:0")
    manager._agentic_bounded_slow_candidate_items()
    assert tuple(manager._agentic_slow_active_ids) == tuple(
        f"slow:{index}" for index in range(1, 7)
    )


def test_d2p_active_host_write_can_progress_without_ledger_poll():
    candidate = {
        "manifest": SimpleNamespace(snapshot_id="slow:local"),
        "arena_write": {
            "snapshot": object(),
            "chunks": {},
            "next_offset": 0,
            "retry_ranges": [],
            "committed_tokens": 0,
            "gpu_elapsed_ms": 0.0,
        },
    }
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.ledger = SimpleNamespace(
        get=lambda _snapshot_id: pytest.fail("active D2H must not poll the ledger")
    )
    client._start_write_chunk = lambda _candidate, _indices: False

    assert client.has_active_local_write(candidate)
    assert client.progress(candidate, [0], local_write_only=True) == "waiting"


def test_shared_arena_spill_capability_has_one_compatibility_rule():
    assert supports_agentic_kv_spill(SimpleNamespace())
    assert supports_agentic_kv_spill(SimpleNamespace(supports_kv_spill=True))
    assert not supports_agentic_kv_spill(
        SimpleNamespace(supports_kv_spill=False)
    )


@pytest.mark.parametrize(
    ("commit_succeeds", "expected"), [(True, "host_ready"), (False, "failed")]
)
def test_tp1_d2p_local_write_uses_final_commit_as_durability_fence(
    commit_succeeds, expected
):
    snapshot_id = "slow:final-fence"

    class Event:
        def query(self):
            return True

    class StartEvent:
        def elapsed_time(self, _event):
            return 1.0

    class Snapshot:
        byte_size = 1024
        _last_d2h_start_event = StartEvent()

        def __init__(self):
            self.commits = []
            self.closed = False

        def commit_backup_range_from_bounce(self, _bounce, **kwargs):
            self.commits.append(kwargs)

        def close(self, *, unlink):
            assert not unlink
            self.closed = True

    snapshot = Snapshot()
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "arena_write": {
            "snapshot": snapshot,
            "chunks": {
                0: {
                    "event": Event(),
                    "copy_refs": object(),
                    "start": 0,
                    "end": 1,
                    "phase": "dma",
                    "start_event": StartEvent(),
                }
            },
            "next_offset": 1,
            "retry_ranges": [],
            "committed_tokens": 0,
            "gpu_elapsed_ms": 0.0,
            "host_copy_elapsed_seconds": 0.0,
            "wall_started_at": time.monotonic(),
        },
    }
    drained = []
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.tp_rank = 0
    client.tp_size = 1
    client._d2h_lanes = [
        {"snapshot_id": snapshot_id, "host_bounce": object(), "phase": "dma"}
    ]
    client.ledger = SimpleNamespace(
        complete_host_write=lambda *args, **kwargs: commit_succeeds,
        mark_writer_rank_drained=lambda *args, **kwargs: drained.append(
            (args, kwargs)
        )
        or True,
        get=lambda _snapshot_id: pytest.fail(
            "TP1 final commit must not perform a second ledger read"
        ),
    )
    client._cleanup_relay_senders = lambda _candidate: None

    # DMA completion, CPU commit retirement and final durability publication
    # are three independently bounded progress stages.
    assert (
        client.progress(candidate, torch.arange(1), local_write_only=True)
        == "waiting"
    )
    assert (
        client.progress(candidate, torch.arange(1), local_write_only=True)
        == "waiting"
    )
    assert client.progress(candidate, torch.arange(1), local_write_only=True) == expected
    assert snapshot.commits == [{"destination_start": 0, "token_count": 1}]
    assert snapshot.closed
    assert client._d2h_lanes[0]["snapshot_id"] is None
    assert bool(drained) is (not commit_succeeds)


def test_d2p_pageable_commit_retains_d_source_and_bounce_until_cpu_fence():
    snapshot_id = "slow:async-cpu-fence"

    class Event:
        def query(self):
            return True

    class StartEvent:
        def elapsed_time(self, _event):
            return 2.0

    class Snapshot:
        byte_size = 2048
        _last_d2h_start_event = StartEvent()

        def __init__(self):
            self.commits = []
            self.closed = False

        def commit_backup_range_from_bounce(self, bounce, **kwargs):
            self.commits.append((bounce, kwargs))

        def close(self, *, unlink):
            assert not unlink
            self.closed = True

    class ControlledPool:
        def __init__(self):
            self.job = None
            self.future = Future()

        def submit(self, function, *args, **kwargs):
            self.job = (function, args, kwargs)
            return self.future

    snapshot = Snapshot()
    bounce = object()
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "arena_write": {
            "snapshot": snapshot,
            "chunks": {
                0: {
                    "event": Event(),
                    "copy_refs": object(),
                    "start": 0,
                    "end": 1,
                    "phase": "dma",
                    "start_event": StartEvent(),
                }
            },
            "next_offset": 1,
            "retry_ranges": [],
            "committed_tokens": 0,
            "gpu_elapsed_ms": 0.0,
            "host_copy_elapsed_seconds": 0.0,
            "wall_started_at": time.monotonic(),
        },
    }
    pool = ControlledPool()
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.tp_rank = 0
    client.tp_size = 1
    client._d2h_lanes = [
        {"snapshot_id": snapshot_id, "host_bounce": bounce, "phase": "dma"}
    ]
    client._d2h_host_copy_pool = pool
    client.ledger = SimpleNamespace(
        complete_host_write=lambda *_args, **_kwargs: True,
        get=lambda _snapshot_id: pytest.fail(
            "TP1 final commit must not perform a second ledger read"
        ),
    )
    client._cleanup_relay_senders = lambda _candidate: None

    assert (
        client.progress(candidate, torch.arange(1), local_write_only=True)
        == "waiting"
    )
    assert snapshot.commits == []
    assert candidate.get("arena_write") is not None
    assert client._d2h_lanes[0]["phase"] == "cpu"

    function, args, kwargs = pool.job
    function(*args, **kwargs)
    pool.future.set_result(0.003)
    assert (
        client.progress(candidate, torch.arange(1), local_write_only=True)
        == "waiting"
    )
    assert client.progress(candidate, torch.arange(1), local_write_only=True) == "host_ready"
    assert snapshot.commits == [
        (bounce, {"destination_start": 0, "token_count": 1})
    ]
    assert snapshot.closed
    assert client._d2h_lanes[0]["phase"] == "free"


def test_d2p_next_dma_overlaps_previous_cpu_commit_without_extra_dma_pressure():
    snapshot_id = "slow:pipelined-commit"

    class Event:
        def __init__(self, ready):
            self.ready = ready

        def query(self):
            return self.ready

    class StartEvent:
        def elapsed_time(self, _event):
            return 1.0

    class Snapshot:
        byte_size = 4096

        def __init__(self):
            self.started = []

        def start_backup_range_from_device(self, indices, **kwargs):
            self.started.append((indices.clone(), kwargs))
            self._last_d2h_start_event = StartEvent()
            return Event(False), object()

        def commit_backup_range_from_bounce(self, *_args, **_kwargs):
            raise AssertionError("controlled Future must fence the CPU commit")

    class ControlledPool:
        def __init__(self):
            self.future = Future()

        def submit(self, *_args, **_kwargs):
            return self.future

    snapshot = Snapshot()
    first_event = Event(True)
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "arena_write": {
            "snapshot": snapshot,
            "chunks": {
                0: {
                    "event": first_event,
                    "copy_refs": object(),
                    "start": 0,
                    "end": 4,
                    "phase": "dma",
                    "start_event": StartEvent(),
                }
            },
            "next_offset": 4,
            "retry_ranges": [],
            "committed_tokens": 0,
            "gpu_elapsed_ms": 0.0,
            "host_copy_elapsed_seconds": 0.0,
            "wall_started_at": time.monotonic(),
        },
    }
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.tp_rank = 0
    client.tp_size = 1
    client._d2h_chunk_tokens = 4
    client._d2h_bounce_depth = 2
    client._d2h_dma_limit = 1
    client._d2h_host_copy_pool = ControlledPool()
    client._d2h_lanes = [
        {
            "snapshot_id": snapshot_id,
            "host_bounce": object(),
            "phase": "dma",
            "stream": object(),
            "staging": object(),
            "dma_group": 0,
        },
        {
            "snapshot_id": None,
            "host_bounce": object(),
            "phase": "free",
            "stream": object(),
            "staging": object(),
            "dma_group": 0,
        },
    ]

    indices = torch.arange(8)
    # Visit 1 hands chunk 0 to the CPU worker.  Its Future remains pending.
    assert client.progress(candidate, indices, local_write_only=True) == "waiting"
    assert client._d2h_lanes[0]["phase"] == "cpu"
    # Visit 2 is allowed to launch chunk 1 into the alternate bounce, while
    # the prior CPU Future still owns bounce 0.  DMA inflight remains capped at 1.
    assert client.progress(candidate, indices, local_write_only=True) == "waiting"
    chunks = candidate["arena_write"]["chunks"]
    assert set(chunks) == {0, 1}
    assert chunks[0]["cpu_future"] is client._d2h_host_copy_pool.future
    assert chunks[1]["phase"] == "dma"
    assert sum(lane["phase"] == "dma" for lane in client._d2h_lanes) == 1


def test_d2p_shared_arena_offer_omits_unused_spill_hashes():
    captured = []
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.ledger = SimpleNamespace(offer=lambda value: captured.append(value) or value)
    client.retain_logical_hashes = False
    client.source_numa_node = 0
    client.arena_numa_node = 0
    client.arena_domain = 0
    client.direct_runtime = None
    client.tp_rank = 0
    client.tp_size = 1
    manifest = SimpleNamespace(
        snapshot_id="slow:no-spill-hashes",
        tool_started_at=1.0,
        direct_room=None,
        kv_layout_hash="layout",
    )
    metadata = SimpleNamespace(
        current=SimpleNamespace(
            request_id="request",
            generation=1,
            storage_id="request:1",
        ),
        tool_type="search",
    )

    client.offer(
        manifest=manifest,
        metadata=metadata,
        token_count=128,
        token_digest="digest",
        logical_hashes=["unused", "unused"],
        byte_size=4096,
    )

    assert "logical_hashes" not in captured[0]

    client.retain_logical_hashes = True
    client.offer(
        manifest=manifest,
        metadata=metadata,
        token_count=128,
        token_digest="digest",
        logical_hashes=["page-0", "page-1"],
        byte_size=4096,
    )
    assert captured[1]["logical_hashes"] == ["page-0", "page-1"]


def test_lazy_shared_host_extent_is_prefaulted_before_grant_publication():
    directory = tempfile.mkdtemp(dir="/dev/shm")
    path = os.path.join(directory, "snapshot.kv")
    snapshot = LazySharedMHAHostSnapshot(
        path=path,
        token_count=1,
        device_pool=SimpleNamespace(),
        byte_size=4 * 1024 * 1024,
    )
    try:
        snapshot.prefault_for_write()
        assert os.stat(path).st_size == snapshot.byte_size
        with open(path, "rb") as file_obj:
            assert file_obj.read(16) == b"\0" * 16
    finally:
        snapshot.close(unlink=True)
        os.rmdir(directory)


def test_lazy_shared_host_extent_enospc_never_touches_sparse_pages(monkeypatch):
    directory = tempfile.mkdtemp(dir="/dev/shm")
    path = os.path.join(directory, "snapshot.kv")
    snapshot = LazySharedMHAHostSnapshot(
        path=path,
        token_count=1,
        device_pool=SimpleNamespace(),
        byte_size=4 * 1024 * 1024,
    )
    memset_called = False

    def fail_fallocate(fd, offset, length):
        raise OSError(errno.ENOSPC, "tmpfs full")

    def record_memset(*args):
        nonlocal memset_called
        memset_called = True

    monkeypatch.setattr(os, "posix_fallocate", fail_fallocate)
    monkeypatch.setattr(host_staging_module, "_HOST_MEMSET", record_memset)
    try:
        with pytest.raises(OSError) as error:
            snapshot.prefault_for_write()
        assert error.value.errno == errno.ENOSPC
        assert not memset_called
    finally:
        snapshot.close(unlink=True)
        os.rmdir(directory)


def test_shared_host_arena_suballocates_preallocated_extent_without_prefault():
    directory = tempfile.mkdtemp(dir="/dev/shm")
    arena = SharedHostSnapshotArena(directory, 16 * 1024 * 1024)
    pool = SimpleNamespace()
    first = arena.create("first", 1, pool, 4 * 1024 * 1024)
    first_path = first.path
    first.prefault_for_write()
    assert not first.requires_prefault
    arena.release(first)

    second = arena.create("second", 1, pool, 2 * 1024 * 1024)
    try:
        assert second.path == first_path
        assert second.allocation_bytes == 2 * 1024 * 1024
        assert not second.requires_prefault
        assert arena.used_bytes == 2 * 1024 * 1024
        assert arena.committed_bytes == 16 * 1024 * 1024
        assert os.stat(second.path).st_size == 16 * 1024 * 1024
        assert second.file_offset == 0
    finally:
        arena.release(second)
        arena.close()
        os.rmdir(directory)


def test_shared_host_arena_release_is_idempotent_and_keeps_backing_pool():
    directory = tempfile.mkdtemp(dir="/dev/shm")
    arena = SharedHostSnapshotArena(directory, 16 * 1024 * 1024)
    snapshot = arena.create("partial", 1, SimpleNamespace(), 4 * 1024 * 1024)
    path = snapshot.path

    arena.release(snapshot)
    arena.release(snapshot)

    assert os.path.exists(path)
    assert arena.used_bytes == 0
    assert arena.committed_bytes == 16 * 1024 * 1024
    assert arena._free_extents == [(0, 16 * 1024 * 1024)]
    arena.close()
    assert not os.path.exists(path)
    os.rmdir(directory)


def test_shared_host_arena_stale_release_cannot_free_recycled_owner():
    directory = tempfile.mkdtemp(dir="/dev/shm")
    arena = SharedHostSnapshotArena(directory, 16 * 1024 * 1024)
    first = arena.create("first", 1, SimpleNamespace(), 4 * 1024 * 1024)
    first.prefault_for_write()
    path = first.path
    arena.release(first)
    second = arena.create("second", 1, SimpleNamespace(), 2 * 1024 * 1024)

    arena.release(first)

    assert arena.used_bytes == 2 * 1024 * 1024
    assert arena._active_extents[id(second)][0] is second
    assert arena._free_extents == [(2 * 1024 * 1024, 14 * 1024 * 1024)]
    arena.release(second)
    arena.close()
    os.rmdir(directory)


def test_shared_host_memfd_arena_is_cross_process_mappable_and_auto_reclaimed():
    directory = tempfile.mkdtemp(dir="/dev/shm")
    arena = SharedHostSnapshotArena(
        directory, 4 * 1024 * 1024, backend="memfd"
    )
    path = arena.path
    assert path.startswith(f"/proc/{os.getpid()}/fd/")
    assert os.readlink(path).startswith("/memfd:sglang-agentic-host-arena-")
    assert os.listdir(directory) == []

    remote_fd = os.open(path, os.O_RDWR)
    remote_mapping = mmap.mmap(remote_fd, 4096, access=mmap.ACCESS_WRITE)
    try:
        remote_mapping[:12] = b"shared-memfd"
        owner_fd = os.open(path, os.O_RDWR)
        try:
            owner_mapping = mmap.mmap(owner_fd, 4096, access=mmap.ACCESS_WRITE)
            try:
                assert owner_mapping[:12] == b"shared-memfd"
            finally:
                owner_mapping.close()
        finally:
            os.close(owner_fd)

        arena.close()
        assert not os.path.exists(path)
        # Existing D/P mappings keep the anonymous object alive long enough
        # to drain in-flight I/O even after the owner closes its descriptor.
        assert remote_mapping[:12] == b"shared-memfd"
    finally:
        remote_mapping.close()
        os.close(remote_fd)
        arena.close()
        os.rmdir(directory)


def test_shared_host_memfd_path_rejects_unrelated_process_descriptors():
    with tempfile.TemporaryFile() as unrelated:
        path = f"/proc/{os.getpid()}/fd/{unrelated.fileno()}"
        with pytest.raises(ValueError, match="not an agentic memfd"):
            host_staging_module._open_shared_host_backing(path, os.O_RDWR)


def _cpu_registered_p2d_arena(page_count=4):
    """Build allocator-only state without requiring CUDA registration."""

    arena = _RegisteredP2DHostArena.__new__(_RegisteredP2DHostArena)
    arena.directory = "/dev/shm/test-registered-p2d"
    arena.path = f"{arena.directory}/registered-arena.kv"
    arena.capacity_bytes = mmap.ALLOCATIONGRANULARITY * page_count
    arena.device_pool = SimpleNamespace(
        layer_num=1,
        head_num=1,
        head_dim=1,
        v_head_dim=1,
        store_dtype=torch.uint8,
        k_buffer=torch.empty(0),
        v_buffer=torch.empty(0),
    )
    arena.raw = torch.zeros(arena.capacity_bytes, dtype=torch.uint8)
    arena.mapping = None
    arena._registered = False
    arena.registration_seconds = 0.0
    arena.used_bytes = 0
    arena._lock = threading.Lock()
    arena._free = [(0, arena.capacity_bytes)]
    arena._active = {}
    arena._closed = False
    return arena


def test_registered_p2d_arena_suballocates_and_coalesces_request_extents():
    arena = _cpu_registered_p2d_arena(page_count=4)
    pool = arena.device_pool
    first = arena.create("first", 1024, pool, 2048)
    second = arena.create("second", 1024, pool, 2048)

    assert first.offset == 0
    assert second.offset == mmap.ALLOCATIONGRANULARITY
    assert arena.used_bytes == 2 * mmap.ALLOCATIONGRANULARITY

    arena.release(first)
    arena.release(second)
    assert arena.used_bytes == 0
    assert arena._free == [(0, arena.capacity_bytes)]


def test_registered_p2d_arena_fragmentation_rejects_before_ledger_claim():
    arena = _cpu_registered_p2d_arena(page_count=4)
    pool = arena.device_pool
    snapshots = [
        arena.create(str(index), 1024, pool, 2048) for index in range(3)
    ]
    arena.release(snapshots[0])
    arena.release(snapshots[2])

    # Three pages are free in aggregate, but no three-page contiguous extent
    # exists.  Admission must reject this before taking Host ledger ownership.
    assert not arena.can_reserve(3 * mmap.ALLOCATIONGRANULARITY, 1.0)
    arena.release(snapshots[1])
    assert arena.can_reserve(3 * mmap.ALLOCATIONGRANULARITY, 1.0)


def test_registered_p2d_arena_stale_release_cannot_free_new_owner():
    arena = _cpu_registered_p2d_arena(page_count=2)
    pool = arena.device_pool
    first = arena.create("first", 1024, pool, 2048)
    arena.release(first)
    second = arena.create("second", 1024, pool, 2048)

    arena.release(first)
    assert arena.used_bytes == mmap.ALLOCATIONGRANULARITY
    assert arena._active[id(second)][0] is second
    arena.release(second)


def test_registered_p2d_arena_rolls_back_extent_when_view_construction_fails(
    monkeypatch,
):
    arena = _cpu_registered_p2d_arena(page_count=4)

    def fail_snapshot(**_kwargs):
        raise ValueError("invalid KV layout")

    monkeypatch.setattr(p2d_host_module, "_RegisteredP2DHostSnapshot", fail_snapshot)
    with pytest.raises(ValueError, match="invalid KV layout"):
        arena.create("bad", 1024, arena.device_pool, 2048)

    assert arena.used_bytes == 0
    assert arena._active == {}
    assert arena._free == [(0, arena.capacity_bytes)]


def test_shared_snapshot_maps_only_its_registered_arena_extent():
    page = mmap.ALLOCATIONGRANULARITY
    fd, path = tempfile.mkstemp(prefix="sglang-p2d-offset-", dir="/dev/shm")
    try:
        os.ftruncate(fd, 2 * page)
        os.pwrite(fd, bytes([17]) * page, 0)
        os.pwrite(fd, bytes([29]) * page, page)
    finally:
        os.close(fd)
    pool = SimpleNamespace(
        layer_num=1,
        head_num=1,
        head_dim=1,
        v_head_dim=1,
        store_dtype=torch.uint8,
        k_buffer=torch.empty(0),
        v_buffer=torch.empty(0),
    )
    snapshot = SharedMHAHostSnapshot(
        path=path,
        token_count=page // 2,
        device_pool=pool,
        byte_size=page,
        create=False,
        file_offset=page,
    )
    try:
        assert torch.all(snapshot.kv_buffer == 29)
    finally:
        snapshot.close()
        os.unlink(path)


def test_p2d_host_extent_is_reserved_before_ledger_claim_and_released_on_loss():
    events = []
    snapshot = SimpleNamespace(path="/dev/shm/test-p2d-arena", offset=0)

    class Arena:
        capacity_bytes = 1024
        used_bytes = 0

        def can_reserve(self, *_args):
            events.append("capacity")
            return True

        def create(self, *_args):
            events.append("reserve")
            return snapshot

        def release(self, value):
            assert value is snapshot
            events.append("release")

    class Ledger:
        def get(self, _snapshot_id):
            return {
                "state": HostStageState.OFFERED.value,
                "prefill_domain": 0,
            }

        def prepare_p2d_write_rank(self, *_args, **_kwargs):
            events.append("prepare")
            return {"state": HostStageState.OFFERED.value}

        def claim_p2d_write_rank(self, *_args, **_kwargs):
            events.append("claim")
            return None

        def reject_unclaimed_offer(self, *_args, **_kwargs):
            events.append("reject")
            return True

        def transition(self, *_args, **_kwargs):
            raise AssertionError("an unclaimed offer must not be failed")

    manager = AgenticPToDHostStagingManager.__new__(AgenticPToDHostStagingManager)
    manager.ledger = Ledger()
    manager.device_pool = SimpleNamespace(
        layer_num=1,
        head_num=1,
        head_dim=1,
        store_dtype=torch.uint8,
    )
    manager.prefill_domain = 0
    manager.numa_node = 0
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.owner = "p"
    manager.hard_watermark = 1.0
    manager.arena = Arena()
    manager._lock = threading.RLock()
    manager._prepared = {}
    manager._active = {}
    manager._results = {}
    manager._records = {}
    req = SimpleNamespace(
        bootstrap_room=77,
        origin_input_ids=[1, 2],
        output_ids=[3],
        return_logprob=False,
        cached_tokens=0,
    )

    assert not manager.try_submit(req, torch.tensor([0, 1]))
    assert events == [
        "capacity",
        "reserve",
        "prepare",
        "claim",
        "release",
        "reject",
    ]
    assert manager._active == {}
    assert manager._records == {}


def test_p2d_manager_close_retains_arena_when_dma_has_no_fence():
    class FinishedThread:
        def join(self, **_kwargs):
            pass

        def is_alive(self):
            return False

    manager = AgenticPToDHostStagingManager.__new__(AgenticPToDHostStagingManager)
    manager._stop = threading.Event()
    manager._candidate_wakeup = threading.Event()
    manager._group_wakeup = threading.Event()
    manager._work = queue.SimpleQueue()
    manager._threads = []
    manager._offer_thread = FinishedThread()
    manager._completion_thread = FinishedThread()
    manager._dma_quarantine = [(object(),)]
    manager.arena = SimpleNamespace(
        close=lambda: (_ for _ in ()).throw(
            AssertionError("an unfenced DMA arena must remain mapped")
        )
    )

    manager.close()
    assert manager._stop.is_set()


def test_p_host_grant_publishes_preallocated_arena_extent():
    published = []
    snapshot = SimpleNamespace(
        path="/dev/shm/prefaulted.kv",
        file_offset=8192,
        byte_size=4096,
        token_count=64,
    )
    record = {
        "offer": {"snapshot_id": "slow:prefaulted"},
        "snapshot": snapshot,
    }
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.active = {"slow:prefaulted": record}
    manager.owner = "p:test"
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.arena_numa_node = 0
    manager.ledger = SimpleNamespace(
        publish_grants=lambda snapshot_id, owner, grants: published.append(
            (snapshot_id, owner, grants)
        )
        or True
    )

    assert published == []
    manager._publish_arena_grant("slow:prefaulted", record)
    assert len(published) == 1
    assert published[0][2][0]["arena_path"] == snapshot.path
    assert published[0][2][0]["arena_offset"] == snapshot.file_offset


def test_shared_host_arena_preallocation_failure_is_atomic(monkeypatch):
    directory = tempfile.mkdtemp(dir="/dev/shm")

    def fail_fallocate(fd, offset, length):
        raise OSError(errno.ENOSPC, "tmpfs full")

    monkeypatch.setattr(os, "posix_fallocate", fail_fallocate)
    try:
        with pytest.raises(OSError) as error:
            SharedHostSnapshotArena(directory, 4 * 1024 * 1024)
        assert error.value.errno == errno.ENOSPC
        assert os.listdir(directory) == []
    finally:
        os.rmdir(directory)


def test_p_host_grant_publish_transient_retains_complete_extent_for_retry():
    snapshot_id = "slow:grant-retry"
    snapshot = SimpleNamespace(
        path="/dev/shm/grant-retry.kv", byte_size=4096, token_count=64
    )
    record = {
        "offer": {"snapshot_id": snapshot_id},
        "snapshot": snapshot,
    }
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.active = {snapshot_id: record}
    manager.aborting = {}
    manager.owner = "p:test"
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.arena_numa_node = 0
    manager.arena = SimpleNamespace(release=lambda _snapshot: pytest.fail("released"))
    manager.ledger = SimpleNamespace(
        publish_grants=lambda *_args, **_kwargs: False,
        get=lambda _snapshot_id: {
            "state": HostStageState.HOST_RESERVED.value,
            "p_owner": "p:test",
            "grants": [],
        },
    )

    manager._publish_arena_grant(snapshot_id, record)

    assert manager.active[snapshot_id] is record
    assert record["grant_publish_pending"] is True


def test_p_host_grant_publish_authoritative_abort_retires_extent_safely():
    snapshot_id = "slow:grant-abort"
    snapshot = SimpleNamespace(
        path="/dev/shm/grant-abort.kv", byte_size=4096, token_count=64
    )
    record = {
        "offer": {"snapshot_id": snapshot_id},
        "snapshot": snapshot,
    }
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.active = {snapshot_id: record}
    manager.aborting = {}
    manager.owner = "p:test"
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.arena_numa_node = 0
    manager.arena = SimpleNamespace(release=lambda _snapshot: pytest.fail("released early"))
    manager.ledger = SimpleNamespace(
        publish_grants=lambda *_args, **_kwargs: False,
        get=lambda _snapshot_id: {
            "state": HostStageState.ABORTING.value,
            "p_owner": "p:test",
            "grants": [],
        },
    )

    manager._publish_arena_grant(snapshot_id, record)

    assert snapshot_id not in manager.active
    assert manager.aborting[snapshot_id] is record
    assert record["free_host_on_abort"] is True


def test_tp_p2d_host_write_ready_is_a_monotonic_boundary():
    """A fast D may advance past HOST_READY before a P rank observes it."""

    assert _p2d_host_write_committed(
        {"state": HostStageState.HOST_READY.value}
    )
    assert _p2d_host_write_committed(
        {"state": HostStageState.H2D_LOADING.value}
    )
    assert _p2d_host_write_committed(
        {"state": HostStageState.CONSUMED.value}
    )
    assert not _p2d_host_write_committed(
        {"state": HostStageState.HOST_WRITING.value}
    )


def test_host_ready_retires_only_the_racing_unstarted_direct_workset():
    """A complete Slow copy supersedes a Direct grant that never started I/O."""

    snapshot_id = "slow-won:0"
    cancelled = []

    class Broker:
        @staticmethod
        def direct_owner(observed_snapshot_id):
            return f"direct:{observed_snapshot_id}"

        @staticmethod
        def supersede_unstarted(observed_snapshot_id, *, owner=None):
            cancelled.append((observed_snapshot_id, owner))
            return True

    record = {
        "offer": {"token_count": 128, "byte_size": 4096},
    }
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.active = {snapshot_id: record}
    manager.host_ready = {}
    manager.workset_broker = Broker()
    manager.ledger = SimpleNamespace(
        get=lambda observed_snapshot_id: (
            {"state": HostStageState.HOST_READY.value}
            if observed_snapshot_id == snapshot_id
            else None
        )
    )

    manager._poll_active()

    assert cancelled == [(snapshot_id, f"direct:{snapshot_id}")]
    assert manager.active == {}
    assert manager.host_ready == {snapshot_id: record}


def test_tp_p2d_host_wait_fails_closed_instead_of_hanging():
    with pytest.raises(RuntimeError, match="terminated in failed"):
        _raise_if_p2d_host_failed(
            "p2d:failed",
            {"state": HostStageState.FAILED.value, "reason": "peer_failed"},
        )
    with pytest.raises(RuntimeError, match="disappeared"):
        _raise_if_p2d_host_failed("p2d:missing", None)


def test_tp_p2d_native_arbitration_rejects_late_host_offer():
    ledger, path = _ledger()
    snapshot_id = "p2d:901"
    try:
        assert ledger.arbitrate_p2d_native(snapshot_id, tp_size=2)
        late = ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 901,
                "token_count": 128,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": 2,
            }
        )
        assert late["state"] == HostStageState.REJECTED.value
        assert late["native_won"] is True
        assert ledger.claim_rank(
            snapshot_id, "p2d-p-group:p0", tp_rank=1, tp_size=2
        ) is None
    finally:
        os.unlink(path)


@pytest.mark.parametrize(
    ("claimed_ranks", "commit", "expected"),
    [
        ((), False, HostStageState.REJECTED),
        ((0,), False, HostStageState.ABORTING),
        ((0, 1), True, HostStageState.FAILED),
    ],
)
def test_router_abort_of_unsubmitted_tp_p2d_preserves_physical_fence(
    claimed_ranks, commit, expected
):
    ledger, path = _ledger()
    snapshot_id = "p2d:905"
    owner = "p2d-p-group:p0"
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 905,
                "token_count": 128,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": 2,
            }
        )
        for rank in claimed_ranks:
            assert ledger.claim_rank(
                snapshot_id, owner, tp_rank=rank, tp_size=2
            )
            assert ledger.publish_rank_grant(
                snapshot_id,
                owner,
                {"kind": "shared_host_extent", "tp_rank": rank},
                tp_rank=rank,
                tp_size=2,
            )
            if commit:
                assert ledger.complete_p2d_host_write_rank(
                    snapshot_id,
                    owner,
                    tp_rank=rank,
                    tp_size=2,
                )

        state = ledger.abort_unsubmitted_p2d(
            snapshot_id, reason="router_cancelled_before_d_submit"
        )

        assert state == expected.value
        assert ledger.get(snapshot_id)["state"] == expected.value
    finally:
        os.unlink(path)


def test_tp_p2d_peer_host_claim_blocks_native_page_release():
    ledger, path = _ledger()
    snapshot_id = "p2d:902"
    owner = "p2d-p-group:p0"
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 902,
                "token_count": 128,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": 2,
            }
        )
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=1, tp_size=2)
        manager = AgenticPToDHostStagingManager.__new__(
            AgenticPToDHostStagingManager
        )
        manager.ledger = ledger
        manager.tp_size = 2
        manager._lock = threading.RLock()
        manager._prepared = {}
        manager._active = {}
        manager._results = {}
        manager._candidates = {}
        req = SimpleNamespace(bootstrap_room=902)

        assert manager.prepare_scheduler_release(req) is False
        assert not getattr(req, "_agentic_p2d_host_terminal", False)
    finally:
        os.unlink(path)


def test_p2d_abort_cannot_release_pages_owned_by_d2h():
    manager = AgenticPToDHostStagingManager.__new__(AgenticPToDHostStagingManager)
    manager._lock = threading.RLock()
    manager._active = {"p2d:903": {}}
    manager._results = {}
    manager._candidates = {}
    req = SimpleNamespace(
        bootstrap_room=903, _agentic_p2d_host_snapshot_id="p2d:903"
    )

    assert manager.cancel_watch(req) is False
    assert not getattr(req, "_agentic_p2d_host_terminal", False)


def test_tp_p2d_d2h_group_barrier_does_not_occupy_copy_lane():
    """Local DMA completion waits for peers in the completion plane only."""

    states = {
        "p2d:first": HostStageState.HOST_WRITING.value,
        "p2d:second": HostStageState.HOST_WRITING.value,
    }
    manager = AgenticPToDHostStagingManager.__new__(AgenticPToDHostStagingManager)
    manager.ledger = SimpleNamespace(
        get=lambda snapshot_id: {"state": states[snapshot_id]}
    )
    manager._lock = threading.RLock()
    manager._active = {"p2d:first": {}, "p2d:second": {}}
    manager._results = {}
    manager._group_pending = {
        "p2d:second": {
            "started_at": time.monotonic(),
            "token_count": 32,
            "byte_size": 64,
            "worker_id": 1,
        },
        "p2d:first": {
            "started_at": time.monotonic(),
            "token_count": 32,
            "byte_size": 64,
            "worker_id": 0,
        },
    }

    # Neither peer group is complete, but both local copy lanes have already
    # returned their records to this independent completion set.
    assert manager._progress_group_completions_once() == 0
    assert set(manager._group_pending) == {"p2d:first", "p2d:second"}

    # Peer ranks may commit in the opposite order without tying up or
    # deadlocking the finite DMA lane pool.
    states["p2d:first"] = HostStageState.HOST_READY.value
    assert manager._progress_group_completions_once() == 1
    states["p2d:second"] = HostStageState.H2D_LOADING.value
    assert manager._progress_group_completions_once() == 1
    assert manager._group_pending == {}
    assert manager._results == {
        "p2d:first": int(KVPoll.Success),
        "p2d:second": int(KVPoll.Success),
    }


def test_tp_p2d_h2d_group_barrier_does_not_occupy_copy_lane():
    states = {
        "p2d:first": HostStageState.H2D_LOADING.value,
        "p2d:second": HostStageState.H2D_LOADING.value,
    }
    manager = AgenticPToDHostLoadManager.__new__(AgenticPToDHostLoadManager)
    manager.ledger = SimpleNamespace(
        get=lambda snapshot_id: {"state": states[snapshot_id]}
    )
    manager._completion_lock = threading.RLock()
    manager.decode_domain = 0
    manager.numa_node = 0
    receivers = {
        snapshot_id: SimpleNamespace(
            snapshot_id=snapshot_id,
            mark_terminal=lambda poll, snapshot_id=snapshot_id, **_kwargs: results.append(
                (snapshot_id, int(poll))
            ),
        )
        for snapshot_id in states
    }
    results = []
    manager._group_pending = {
        snapshot_id: {
            "receiver": receiver,
            "started_at": time.monotonic(),
            "token_count": 32,
            "byte_size": 64,
            "worker_id": worker_id,
        }
        for worker_id, (snapshot_id, receiver) in enumerate(receivers.items())
    }

    assert manager._progress_group_completions_once() == 0
    states["p2d:second"] = HostStageState.CONSUMED.value
    assert manager._progress_group_completions_once() == 1
    states["p2d:first"] = HostStageState.CONSUMED.value
    assert manager._progress_group_completions_once() == 1
    assert manager._group_pending == {}
    assert set(results) == {
        ("p2d:first", int(KVPoll.Success)),
        ("p2d:second", int(KVPoll.Success)),
    }


@pytest.mark.parametrize(
    "terminal_state",
    [HostStageState.FAILED, HostStageState.ABORTING],
)
@pytest.mark.parametrize("release_method", ["prepare_scheduler_release", "cancel_watch"])
def test_tp_p2d_peer_terminal_releases_unsubmitted_local_shard(
    terminal_state, release_method
):
    ledger, path = _ledger()
    snapshot_id = "p2d:904"
    owner = "p2d-p-group:p0"
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 904,
                "token_count": 128,
                "prefill_domain": 0,
                "request_direction": "p2d",
                "control_offer": True,
                "tp_size": 2,
            }
        )
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=1, tp_size=2)
        assert ledger.transition(snapshot_id, terminal_state, owner=owner)

        manager = AgenticPToDHostStagingManager.__new__(
            AgenticPToDHostStagingManager
        )
        manager.ledger = ledger
        manager.tp_size = 2
        manager._lock = threading.RLock()
        manager._prepared = {}
        manager._active = {}
        manager._results = {}
        manager._candidates = {snapshot_id: {}}
        req = SimpleNamespace(bootstrap_room=904)

        assert getattr(manager, release_method)(req) is True
        assert getattr(req, "_agentic_p2d_host_terminal", False)
        assert snapshot_id not in manager._candidates
    finally:
        os.unlink(path)


def test_tp_host_commit_admits_after_manifest_cleanup():
    """The native commit outlives request-level Host manifest cleanup."""

    request = RequestGeneration("host-commit", 4)
    ledger = SimpleNamespace(
        get=lambda _snapshot_id: None,
        mark_d2p_recovery_phase_rank=lambda *_args, **_kwargs: True,
    )
    manager = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        owner="p-group:prefill-0",
        ledger=ledger,
        tp_host_commit_snapshot=request.snapshot_id,
        workset_broker=SimpleNamespace(handoff_to_req=lambda *_args: None),
        token_allocator=object(),
    )
    req = SimpleNamespace(
        rid="child",
        _agentic_host_rank_loaded=True,
        _agentic_host_rank_token_count=256,
        _agentic_host_workset_lease=SimpleNamespace(
            lease_id=1, owner=f"slow:{request.snapshot_id}:child"
        ),
    )

    assert AgenticPHostStagingManager.gate_request(manager, req, request) is False
    assert req._agentic_kv_gate_complete is True
    assert req._agentic_kv_host_hit_tokens == 256
    assert req._agentic_tp_bootstrap_snapshot_id == request.snapshot_id
    assert not hasattr(req, "_agentic_host_rank_loaded")


def test_slow_h2d_lanes_cap_workset_intents_before_hbm_allocation():
    """The fifth Slow snapshot remains Host-only when four lanes are owned."""

    requested = []

    class Broker:
        @staticmethod
        def slow_owner(snapshot_id, rid):
            return f"slow:{snapshot_id}:{rid}"

        def request(self, snapshot_id, parent_tokens, prompt_tokens, *, owner):
            requested.append((snapshot_id, parent_tokens, prompt_tokens, owner))
            return True

        @staticmethod
        def get(_snapshot_id, *, owner=None):
            return None

    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.max_h2d_inflight = 4
    manager._h2d_lane_reservations = {}
    manager.active = {}
    manager.aborting = {}
    manager.loads = {}
    manager.host_ready = {}
    manager._ledger_entries_cache = {}
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.arena_domain = 0
    manager.owner = "p:test"
    manager.workset_broker = Broker()
    manager.ledger = SimpleNamespace(
        get=lambda snapshot_id: manager._ledger_entries_cache.get(snapshot_id),
        claim_d2p_recovery_rank=lambda *_args, **_kwargs: True,
    )

    requests = []
    reqs = []
    for index in range(5):
        request = RequestGeneration(f"slow-{index}", 1)
        req = SimpleNamespace(rid=f"child-{index}", origin_input_ids=[11, 22])
        requests.append(request)
        reqs.append(req)
        manager.host_ready[request.snapshot_id] = {
            "snapshot": SimpleNamespace(_materialized=object()),
            "offer": {
                "token_count": 1,
                "token_digest": token_ids_digest([11]),
                "byte_size": 128,
            },
            "loading": False,
        }
        manager._ledger_entries_cache[request.snapshot_id] = {
            "state": HostStageState.HOST_READY.value,
            "p_owner": manager.owner,
        }

    for request, req in zip(requests, reqs):
        assert manager.gate_request(req, request) is True

    assert len(requested) == 4
    assert set(manager._h2d_lane_reservations.values()) == {0, 1, 2, 3}
    assert requests[4].snapshot_id not in manager._h2d_lane_reservations

    manager._release_h2d_lane(requests[1].snapshot_id)
    assert manager.gate_request(reqs[4], requests[4]) is True
    assert len(requested) == 5
    assert manager._h2d_lane_reservations[requests[4].snapshot_id] == 1


def test_selected_recovery_p_retires_direct_before_claiming_slow_lane():
    """TP frozen Direct cannot pin cross-P Slow claim/lane before retire."""

    class Allocator:
        def __init__(self):
            self.available = 32

        def available_size(self):
            return self.available

        def alloc(self, count):
            self.available -= count
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.available += int(indices.numel())

    snapshot_id = "cross-p-direct-slow:0"
    slow_requests = []
    recovery_claims = []

    request = RequestGeneration("cross-p-direct-slow", 0)
    req = SimpleNamespace(rid="child", origin_input_ids=[11, 22])
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    direct_owner = broker.direct_owner(snapshot_id)
    broker.install_tp_plan(1, [(snapshot_id, direct_owner, 4, 8)])
    allocator = Allocator()
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.max_h2d_inflight = 4
    manager._h2d_lane_reservations = {}
    manager.active = {}
    manager.aborting = {}
    manager.loads = {}
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.arena_domain = 3
    manager.owner = "p:recovery"
    manager.workset_broker = broker
    record = {
        "snapshot": SimpleNamespace(_materialized=object()),
        "offer": {
            "token_count": 1,
            "token_digest": token_ids_digest([11]),
            "byte_size": 128,
        },
        "loading": False,
    }
    manager.host_ready = {snapshot_id: record}
    ledger_entry = {
        "state": HostStageState.HOST_READY.value,
        "p_owner": "p:host-owner",
        "recovery_domain": manager.arena_domain,
        "recovery_owner": manager.owner,
    }
    manager._ledger_entries_cache = {snapshot_id: ledger_entry}
    manager.ledger = SimpleNamespace(
        get=lambda _snapshot_id: ledger_entry,
        claim_d2p_recovery_rank=lambda *_args, **_kwargs: recovery_claims.append(
            (_args, _kwargs)
        )
        or True,
    )

    # The scheduler owns physical release.  Until it has serviced that
    # release, Slow consumes no shared-ledger claim and no finite H2D lane.
    assert manager.gate_request(req, request) is True
    assert broker.owner_is_superseded(snapshot_id, owner=direct_owner)
    assert broker.owner_has_unretired_work(snapshot_id, owner=direct_owner)
    assert recovery_claims == []
    assert slow_requests == []
    assert manager._h2d_lane_reservations == {}

    # The immutable epoch still materializes exactly as broadcast, but Slow
    # remains Host-only while that Direct lease awaits group retirement.
    broker.service(allocator)
    assert broker.get(snapshot_id, owner=direct_owner) is not None
    assert manager.gate_request(req, request) is True
    assert recovery_claims == []
    assert manager._h2d_lane_reservations == {}

    # The next TP control epoch retires the old owner on every rank.  Only
    # after physical service frees its pages may Slow claim Host and a lane.
    _plan, retirements, _handoffs = broker.prepare_tp_control(2)
    assert retirements == (snapshot_id,)
    assert broker.owner_has_unretired_work(snapshot_id, owner=direct_owner)
    assert manager.gate_request(req, request) is True
    assert recovery_claims == []
    assert manager._h2d_lane_reservations == {}
    assert not broker.request(
        snapshot_id,
        parent_tokens=1,
        prompt_tokens=2,
        owner=broker.slow_owner(snapshot_id, req.rid),
    )
    assert broker.commit_tp_retire(snapshot_id)
    broker.service(allocator)
    assert not broker.owner_has_unretired_work(
        snapshot_id, owner=direct_owner
    )

    original_request = broker.request

    def record_slow_request(*args, **kwargs):
        slow_requests.append((args, kwargs))
        return original_request(*args, **kwargs)

    broker.request = record_slow_request
    assert manager.gate_request(req, request) is True
    assert len(recovery_claims) == 1
    assert len(slow_requests) == 1
    assert manager._h2d_lane_reservations == {snapshot_id: 0}


def test_d2p_recovery_claim_fences_eviction_before_workset_allocation():
    """Host pinning wins the lifecycle CAS before any P-HBM lease exists."""

    ledger, path = _ledger()
    snapshot_id = "recovery-fence:1"
    owner = "p:test"
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 128,
                "byte_size": 4096,
                "tp_size": 1,
            }
        )

        def publish_ready(entries):
            current = entries[snapshot_id]
            current.update(
                p_owner=owner,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_ready, event_snapshot_id=snapshot_id)
        claim_id = f"slow:{snapshot_id}:child"
        assert ledger.claim_d2p_recovery_rank(
            snapshot_id,
            owner,
            tp_rank=0,
            tp_size=1,
            claim_id=claim_id,
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.H2D_LOADING.value
        assert not ledger.begin_host_eviction(
            snapshot_id,
            owner,
            tp_size=1,
            reason="pressure",
        )
        assert ledger.attach_d2p_recovery_lease_rank(
            snapshot_id,
            owner,
            tp_rank=0,
            tp_size=1,
            claim_id=claim_id,
            lease_id=17,
        )
        assert ledger.cancel_d2p_recovery_rank(
            snapshot_id,
            owner,
            tp_rank=0,
            tp_size=1,
            claim_id=claim_id,
            lease_id=17,
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.HOST_READY.value
        assert ledger.begin_host_eviction(
            snapshot_id,
            owner,
            tp_size=1,
            reason="pressure",
        )
    finally:
        os.unlink(path)


def test_d2p_recovery_inflight_cannot_be_cancelled_or_evicted():
    ledger, path = _ledger()
    snapshot_id = "recovery-inflight:1"
    owner = "p:test"
    claim_id = f"slow:{snapshot_id}:child"
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 128,
                "byte_size": 4096,
                "tp_size": 1,
            }
        )

        def publish_ready(entries):
            entries[snapshot_id].update(
                p_owner=owner,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_ready, event_snapshot_id=snapshot_id)
        assert ledger.claim_d2p_recovery_rank(
            snapshot_id, owner, tp_rank=0, tp_size=1, claim_id=claim_id
        )
        assert ledger.attach_d2p_recovery_lease_rank(
            snapshot_id,
            owner,
            tp_rank=0,
            tp_size=1,
            claim_id=claim_id,
            lease_id=23,
        )
        assert ledger.mark_d2p_recovery_phase_rank(
            snapshot_id,
            owner,
            tp_rank=0,
            tp_size=1,
            claim_id=claim_id,
            lease_id=23,
            phase="io_inflight",
        )
        assert not ledger.cancel_d2p_recovery_rank(
            snapshot_id,
            owner,
            tp_rank=0,
            tp_size=1,
            claim_id=claim_id,
            lease_id=23,
        )
        assert not ledger.begin_host_eviction(
            snapshot_id, owner, tp_size=1, reason="pressure"
        )
    finally:
        os.unlink(path)


def test_abort_cancels_pinned_recovery_before_workset_grant():
    ledger, path = _ledger()
    request = RequestGeneration("abort-pinned", 1)
    snapshot_id = request.snapshot_id
    owner = "p:test"
    rid = "child-pinned"
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    released = []
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 8,
                "byte_size": 4096,
                "tp_size": 1,
            }
        )

        def publish_ready(entries):
            entries[snapshot_id].update(
                p_owner=owner,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_ready, event_snapshot_id=snapshot_id)
        claim_id = broker.slow_owner(snapshot_id, rid)
        assert ledger.claim_d2p_recovery_rank(
            snapshot_id, owner, tp_rank=0, tp_size=1, claim_id=claim_id
        )

        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager._state_lock = threading.RLock()
        manager.owner = owner
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.ledger = ledger
        manager.workset_broker = broker
        manager.loads = {}
        manager.host_ready = {
            snapshot_id: {
                "loading": "h2d_reserving",
                "snapshot": object(),
                "offer": {"token_count": 8, "byte_size": 4096},
            }
        }
        manager._h2d_lane_reservations = {snapshot_id: 0}
        manager._control_wakeup = SimpleNamespace(set=lambda: None)
        manager._release_record = lambda record: released.append(record) or True

        manager.abort_request(rid, request)

        assert ledger.get(snapshot_id)["state"] == HostStageState.FAILED.value
        assert snapshot_id not in manager.host_ready
        assert snapshot_id not in manager._prestart_recovery_aborts
        assert broker.eviction_blocker(snapshot_id) is None
        assert len(released) == 1
    finally:
        os.unlink(path)


def test_abort_releases_exact_granted_workset_before_host_snapshot():
    class Allocator:
        def __init__(self):
            self.freed = []

        @staticmethod
        def alloc(count):
            return torch.arange(count, dtype=torch.int64)

        def free(self, indices):
            self.freed.append(indices.clone())

    ledger, path = _ledger()
    request = RequestGeneration("abort-leased", 1)
    snapshot_id = request.snapshot_id
    owner = "p:test"
    rid = "child-leased"
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    allocator = Allocator()
    released = []
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 8,
                "byte_size": 4096,
                "tp_size": 1,
            }
        )

        def publish_ready(entries):
            entries[snapshot_id].update(
                p_owner=owner,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_ready, event_snapshot_id=snapshot_id)
        claim_id = broker.slow_owner(snapshot_id, rid)
        assert ledger.claim_d2p_recovery_rank(
            snapshot_id, owner, tp_rank=0, tp_size=1, claim_id=claim_id
        )
        assert broker.request(snapshot_id, 8, 12, owner=claim_id)
        broker.service(allocator)
        lease = broker.get(snapshot_id, owner=claim_id)
        assert lease is not None
        assert ledger.attach_d2p_recovery_lease_rank(
            snapshot_id,
            owner,
            tp_rank=0,
            tp_size=1,
            claim_id=claim_id,
            lease_id=lease.lease_id,
        )

        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager._state_lock = threading.RLock()
        manager.owner = owner
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.ledger = ledger
        manager.workset_broker = broker
        manager.loads = {}
        manager.host_ready = {
            snapshot_id: {
                "loading": "h2d_reserving",
                "snapshot": object(),
                "offer": {"token_count": 8, "byte_size": 4096},
            }
        }
        manager._h2d_lane_reservations = {snapshot_id: 0}
        manager._control_wakeup = SimpleNamespace(set=lambda: None)
        manager._release_record = lambda record: released.append(record) or True

        manager.abort_request(rid, request)
        broker.service(allocator)

        assert ledger.get(snapshot_id)["state"] == HostStageState.FAILED.value
        assert broker.get(snapshot_id, owner=claim_id) is None
        assert len(allocator.freed) == 1
        assert len(released) == 1
    finally:
        os.unlink(path)


def test_redirect_abort_on_host_owner_preserves_remote_recovery_snapshot():
    """An obsolete Direct attempt cannot abort another P's Slow recovery."""

    ledger, path = _ledger()
    request = RequestGeneration("redirect-host-owner", 1)
    snapshot_id = request.snapshot_id
    host_owner = "p:host-domain-0"
    recovery_owner = "p:recovery-domain-1"
    released = []
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 8,
                "byte_size": 4096,
                "tp_size": 1,
            }
        )

        def publish_remote_recovery(entries):
            entries[snapshot_id].update(
                p_owner=host_owner,
                arena_domain=0,
                recovery_domain=1,
                recovery_owner=recovery_owner,
                state=HostStageState.H2D_LOADING.value,
                recovery_claim_id="slow:remote",
                recovery_claims={
                    "0": {"claim_id": "slow:remote", "phase": "pinned"}
                },
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_remote_recovery, event_snapshot_id=snapshot_id)
        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager._state_lock = threading.RLock()
        manager.owner = host_owner
        manager.arena_domain = 0
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.ledger = ledger
        manager.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
        manager.loads = {}
        manager.host_ready = {
            snapshot_id: {
                "loading": False,
                "snapshot": object(),
                "offer": {"token_count": 8, "byte_size": 4096},
            }
        }
        manager._h2d_lane_reservations = {}
        manager._control_wakeup = SimpleNamespace(set=lambda: None)
        manager._release_record = lambda record: released.append(record) or True

        manager.abort_request("obsolete-direct-rid", request)

        entry = ledger.get(snapshot_id)
        assert entry["state"] == HostStageState.H2D_LOADING.value
        assert entry["recovery_owner"] == recovery_owner
        assert snapshot_id in manager.host_ready
        assert released == []
        assert not getattr(manager, "_prestart_recovery_aborts", {})
    finally:
        os.unlink(path)


def test_tp2_selected_unclaimed_abort_drains_late_rank():
    """Every rank ACKs an abort even when rank 0 froze the group first."""

    ledger, path = _ledger()
    request = RequestGeneration("tp2-selected-abort", 1)
    snapshot_id = request.snapshot_id
    owner = "p-group:selected"
    rid = "selected-child"
    managers = []
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 8,
                "byte_size": 4096,
                "tp_size": 2,
            }
        )

        def publish_selected(entries):
            entries[snapshot_id].update(
                p_owner=owner,
                arena_domain=0,
                recovery_domain=0,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_selected, event_snapshot_id=snapshot_id)
        for rank in range(2):
            manager = AgenticPHostStagingManager.__new__(
                AgenticPHostStagingManager
            )
            manager._state_lock = threading.RLock()
            manager.owner = owner
            manager.arena_domain = 0
            manager.tp_rank = rank
            manager.tp_size = 2
            manager.ledger = ledger
            manager.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
            manager.loads = {}
            manager.host_ready = {
                snapshot_id: {
                    "loading": False,
                    "snapshot": object(),
                    "offer": {"token_count": 8, "byte_size": 4096},
                }
            }
            manager._h2d_lane_reservations = {}
            manager._control_wakeup = SimpleNamespace(set=lambda: None)
            manager._release_record = lambda _record: True
            managers.append(manager)

        managers[0].abort_request(rid, request)
        first = ledger.get(snapshot_id)
        assert first["state"] == HostStageState.ABORTING.value
        assert first["loader_drained_ranks"] == [0]

        managers[1].abort_request(rid, request)
        final = ledger.get(snapshot_id)
        assert final["state"] == HostStageState.FAILED.value
        assert final["loader_drained_ranks"] == [0, 1]
        assert final["recovery_claims"] == {}
    finally:
        os.unlink(path)


@pytest.mark.parametrize("with_lease", [False, True])
def test_tp2_prestart_abort_is_group_atomic(with_lease):
    class Allocator:
        @staticmethod
        def alloc(count):
            return torch.arange(count, dtype=torch.int64)

        @staticmethod
        def free(_indices):
            return None

    ledger, path = _ledger()
    request = RequestGeneration(f"tp2-abort-{with_lease}", 1)
    snapshot_id = request.snapshot_id
    owner = "p-group:test"
    rid = "child-tp2"
    managers = []
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 8,
                "byte_size": 4096,
                "tp_size": 2,
            }
        )

        def publish_ready(entries):
            entries[snapshot_id].update(
                p_owner=owner,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_ready, event_snapshot_id=snapshot_id)
        claim_id = AgenticPWorksetLeaseBroker.slow_owner(snapshot_id, rid)
        for rank in range(2):
            assert ledger.claim_d2p_recovery_rank(
                snapshot_id,
                owner,
                tp_rank=rank,
                tp_size=2,
                claim_id=claim_id,
            )
            broker = AgenticPWorksetLeaseBroker(page_size=4)
            if with_lease:
                assert broker.request(snapshot_id, 8, 12, owner=claim_id)
                broker.service(Allocator())
                lease = broker.get(snapshot_id, owner=claim_id)
                assert lease is not None
                assert ledger.attach_d2p_recovery_lease_rank(
                    snapshot_id,
                    owner,
                    tp_rank=rank,
                    tp_size=2,
                    claim_id=claim_id,
                    lease_id=lease.lease_id,
                )
            manager = AgenticPHostStagingManager.__new__(
                AgenticPHostStagingManager
            )
            manager._state_lock = threading.RLock()
            manager.owner = owner
            manager.arena_domain = 0
            manager.tp_rank = rank
            manager.tp_size = 2
            manager.ledger = ledger
            manager.workset_broker = broker
            manager.loads = {}
            manager.host_ready = {
                snapshot_id: {
                    "loading": "h2d_reserving",
                    "snapshot": object(),
                    "offer": {"token_count": 8, "byte_size": 4096},
                }
            }
            manager._h2d_lane_reservations = {snapshot_id: 0}
            manager._control_wakeup = SimpleNamespace(set=lambda: None)
            manager._release_record = lambda _record: True
            managers.append(manager)

        managers[0].abort_request(rid, request)
        first = ledger.get(snapshot_id)
        assert first["state"] == HostStageState.ABORTING.value
        assert first["loader_drained_ranks"] == [0]
        assert set(first["recovery_claims"]) == {"1"}

        managers[1].abort_request(rid, request)
        final = ledger.get(snapshot_id)
        assert final["state"] == HostStageState.FAILED.value
        assert final["loader_drained_ranks"] == [0, 1]
        assert final["recovery_claims"] == {}
        for manager in managers:
            manager._release_consumed_owned_host({snapshot_id: final})
            assert snapshot_id not in manager.host_ready
            assert snapshot_id not in manager._prestart_recovery_aborts
            assert manager.workset_broker.eviction_blocker(snapshot_id) is None
    finally:
        os.unlink(path)


def test_tp2_prestart_abort_loser_yields_to_eviction_owner():
    ledger, path = _ledger()
    request = RequestGeneration("abort-vs-eviction", 1)
    snapshot_id = request.snapshot_id
    owner = "p-group:test"
    released = []
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 8,
                "byte_size": 4096,
                "tp_size": 2,
            }
        )

        def publish_ready(entries):
            entries[snapshot_id].update(
                p_owner=owner,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_ready, event_snapshot_id=snapshot_id)
        assert ledger.begin_host_eviction(
            snapshot_id, owner, tp_size=2, reason="pressure"
        )

        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager._state_lock = threading.RLock()
        manager.owner = owner
        manager.tp_rank = 1
        manager.tp_size = 2
        manager.ledger = ledger
        manager.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
        manager.host_ready = {
            snapshot_id: {
                "loading": "abort_pending",
                "abort_requested": True,
                "snapshot": object(),
                "offer": {"token_count": 8, "byte_size": 4096},
            }
        }
        manager._prestart_recovery_aborts = {
            snapshot_id: {"rid": "child", "request_generation": request}
        }
        manager._h2d_lane_reservations = {snapshot_id: 0}
        manager._release_record = lambda record: released.append(record) or True

        manager._progress_prestart_aborts()

        assert ledger.get(snapshot_id)["state"] == HostStageState.EVICTING.value
        assert released == []
        assert manager.host_ready[snapshot_id]["loading"] is False
        assert "abort_requested" not in manager.host_ready[snapshot_id]
        assert snapshot_id not in manager._prestart_recovery_aborts
    finally:
        os.unlink(path)


def test_prestart_abort_retries_first_ledger_mutation_error():
    ledger, path = _ledger()
    request = RequestGeneration("abort-ledger-retry", 1)
    snapshot_id = request.snapshot_id
    owner = "p:test"
    rid = "child-retry"
    released = []
    try:
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "token_count": 8,
                "byte_size": 4096,
                "tp_size": 1,
            }
        )

        def publish_ready(entries):
            entries[snapshot_id].update(
                p_owner=owner,
                state=HostStageState.HOST_READY.value,
                updated_at=time.time(),
            )
            return True, True

        ledger._mutate(publish_ready, event_snapshot_id=snapshot_id)
        original = ledger.request_host_load_failure
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("transient ledger failure")
            return original(*args, **kwargs)

        ledger.request_host_load_failure = fail_once
        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager._state_lock = threading.RLock()
        manager.owner = owner
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.ledger = ledger
        manager.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
        manager.loads = {}
        manager.host_ready = {
            snapshot_id: {
                "loading": False,
                "snapshot": object(),
                "offer": {"token_count": 8, "byte_size": 4096},
            }
        }
        manager._h2d_lane_reservations = {snapshot_id: 0}
        manager._control_wakeup = SimpleNamespace(set=lambda: None)
        manager._release_record = lambda record: released.append(record) or True

        manager.abort_request(rid, request)
        # A transient CAS failure is fail-closed: no local abort tombstone may
        # precede authoritative ledger ownership.
        assert ledger.get(snapshot_id)["state"] == HostStageState.HOST_READY.value
        assert not getattr(manager, "_prestart_recovery_aborts", {})
        assert released == []

        manager._progress_host_abort_requests()
        assert ledger.get(snapshot_id)["state"] == HostStageState.FAILED.value
        assert snapshot_id not in manager._prestart_recovery_aborts
        assert len(released) == 1
    finally:
        os.unlink(path)


def test_workset_broker_reports_every_eviction_unsafe_owner():
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    lease = SimpleNamespace(
        snapshot_id="eviction-owner:1",
        lease_id=9,
        owner="slow:eviction-owner:1:child",
        allocated_tokens=32,
        state="active",
    )
    broker._leases[lease.snapshot_id] = lease
    assert "state=active" in broker.eviction_blocker(lease.snapshot_id)
    lease.state = "io_inflight"
    assert "state=io_inflight" in broker.eviction_blocker(lease.snapshot_id)
    lease.state = "handed"
    assert "state=handed" in broker.eviction_blocker(lease.snapshot_id)
    lease.state = "releasing"
    assert broker.eviction_blocker(lease.snapshot_id) is None


def test_tp_slow_lane_retries_transient_prepare_without_start_or_leak():
    """A TP ledger hiccup keeps one exact lease and retries before H2D."""

    request = RequestGeneration("slow-prepare-retry", 3)
    req = SimpleNamespace(rid="child-retry", origin_input_ids=[11, 22])
    lease = SimpleNamespace(lease_id=1, parent_indices=[7])
    prepare_calls = []

    class Broker:
        @staticmethod
        def slow_owner(snapshot_id, rid):
            return f"slow:{snapshot_id}:{rid}"

        @staticmethod
        def request(*_args, **_kwargs):
            return True

        @staticmethod
        def get(*_args, **_kwargs):
            return lease

        @staticmethod
        def begin_io_attempt(*_args, **_kwargs):
            return True

    class Ledger:
        @staticmethod
        def get(_snapshot_id):
            return {
                "state": HostStageState.HOST_READY.value,
                "p_owner": "p:test",
            }

        @staticmethod
        def claim_d2p_recovery_rank(*_args, **_kwargs):
            return True

        @staticmethod
        def attach_d2p_recovery_lease_rank(*_args, **_kwargs):
            return True

        @staticmethod
        def prepare_tp_host_load_rank(*_args, **_kwargs):
            prepare_calls.append(True)
            if len(prepare_calls) == 1:
                raise RuntimeError("transient ledger failure")
            return True

    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.max_h2d_inflight = 4
    manager._h2d_lane_reservations = {}
    manager.active = {}
    manager.aborting = {}
    manager.loads = {}
    manager.host_ready = {
        request.snapshot_id: {
            "snapshot": SimpleNamespace(_materialized=object()),
            "offer": {
                "token_count": 1,
                "token_digest": token_ids_digest([11]),
                "byte_size": 128,
            },
            "loading": False,
        }
    }
    manager._ledger_entries_cache = None
    manager.tp_rank = 0
    manager.tp_size = 2
    manager.arena_domain = 0
    manager.owner = "p:test"
    manager.workset_broker = Broker()
    manager.ledger = Ledger()
    manager._control_wakeup = threading.Event()

    # The first prepare fails after the physical lane and workset lease have
    # been selected.  It must not authorize the independent H2D worker.
    assert manager.gate_request(req, request, allow_start=True) is True
    load = manager.loads[req.rid]
    assert load["ledger_prepare_pending"] is True
    assert load["start_allowed"] is False
    assert manager._h2d_lane_reservations == {request.snapshot_id: 0}

    # The next visit retries the same idempotent TP transition.  No second lane
    # or second physical workset is allocated, and only then may H2D start.
    assert manager.gate_request(req, request, allow_start=True) is True
    assert len(prepare_calls) == 2
    assert manager.loads[req.rid] is load
    assert load["ledger_prepare_pending"] is False
    assert load["start_allowed"] is True
    assert manager._h2d_lane_reservations == {request.snapshot_id: 0}


def test_tp1_slow_handoff_failure_retains_host_load_and_lane_for_retry():
    request = RequestGeneration("slow-handoff-tp1", 1)
    req = SimpleNamespace(rid="child-tp1", origin_input_ids=[11, 22], extra_key=None)
    handoff_calls = []
    released_host = []
    lease = SimpleNamespace(
        lease_id=1, owner=f"slow:{request.snapshot_id}:{req.rid}"
    )
    record = {
        "snapshot": object(),
        "offer": {"token_count": 1, "byte_size": 128},
        "loading": "h2d",
    }
    load = {
        "record": record,
        "request_generation": request,
        "device_indices": [7],
        "workset_lease": lease,
        "recovery_claim_id": lease.owner,
        "io_error": None,
        "io_complete": True,
        "radix_bound": True,
        "host_released": False,
    }

    class Broker:
        @staticmethod
        def handoff_to_req(*_args):
            handoff_calls.append(True)
            if len(handoff_calls) == 1:
                raise RuntimeError("transient handoff failure")

    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.owner = "p:test"
    manager.loads = {req.rid: load}
    manager.host_ready = {request.snapshot_id: record}
    manager._h2d_lane_reservations = {request.snapshot_id: 0}
    manager.workset_broker = Broker()
    manager.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": HostStageState.CONSUMED.value},
        complete_host_bind_rank=lambda *_args, **_kwargs: True,
        mark_d2p_recovery_phase_rank=lambda *_args, **_kwargs: True,
    )
    manager._complete_shared_host_manifest = lambda _request: True
    manager._release_record = lambda selected: released_host.append(selected) or True

    assert manager.gate_request(req, request) is True
    assert manager.loads[req.rid] is load
    assert manager.host_ready[request.snapshot_id] is record
    assert manager._h2d_lane_reservations == {request.snapshot_id: 0}
    assert released_host == []

    assert manager.gate_request(req, request) is False
    assert len(handoff_calls) == 2
    assert req.rid not in manager.loads
    assert request.snapshot_id not in manager.host_ready
    assert manager._h2d_lane_reservations == {}
    assert released_host == [record]


def test_d2p_h2d_release_retries_unregister_before_dropping_host_record():
    snapshot_id = "slow:h2d-unregister-retry"
    record = {"snapshot": object()}
    load = {
        "request_generation": SimpleNamespace(snapshot_id=snapshot_id),
        "record": record,
        "host_released": False,
    }
    release_results = iter((False, True))
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.tp_rank = 0
    manager.host_ready = {snapshot_id: record}
    manager.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": HostStageState.CONSUMED.value}
    )
    manager._release_record = lambda selected: (
        selected is record and next(release_results)
    )

    assert manager._release_completed_h2d_host(load) is False
    assert load["host_released"] is False
    assert manager.host_ready[snapshot_id] is record

    assert manager._release_completed_h2d_host(load) is True
    assert load["host_released"] is True
    assert snapshot_id not in manager.host_ready


def test_tp2_slow_handoff_failure_retains_commit_context_for_retry():
    request = RequestGeneration("slow-handoff-tp2", 1)
    req = SimpleNamespace(
        rid="child-tp2",
        origin_input_ids=[11, 22],
        _agentic_host_rank_loaded=True,
        _agentic_host_rank_token_count=1,
        _agentic_host_workset_lease=SimpleNamespace(
            lease_id=1, owner=f"slow:{request.snapshot_id}:child-tp2"
        ),
    )
    handoff_calls = []
    released_host = []
    record = {"snapshot": object(), "offer": {"token_count": 1}}

    class Broker:
        @staticmethod
        def handoff_to_req(*_args):
            handoff_calls.append(True)
            if len(handoff_calls) == 1:
                raise RuntimeError("transient TP handoff failure")

    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.tp_rank = 1
    manager.tp_size = 2
    manager.owner = "p-group:test"
    manager.tp_host_commit_snapshot = request.snapshot_id
    manager.loads = {}
    manager.host_ready = {request.snapshot_id: record}
    manager._h2d_lane_reservations = {request.snapshot_id: 2}
    manager.workset_broker = Broker()
    manager.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": HostStageState.CONSUMED.value},
        mark_d2p_recovery_phase_rank=lambda *_args, **_kwargs: True,
    )
    manager._release_record = lambda selected: released_host.append(selected) or True

    assert manager.gate_request(req, request) is True
    assert req._agentic_host_rank_loaded is True
    assert req._agentic_host_workset_lease is not None
    assert manager.host_ready[request.snapshot_id] is record
    assert manager._h2d_lane_reservations == {request.snapshot_id: 2}
    assert released_host == []

    assert manager.gate_request(req, request) is False
    assert len(handoff_calls) == 2
    assert not hasattr(req, "_agentic_host_rank_loaded")
    assert not hasattr(req, "_agentic_host_workset_lease")
    assert request.snapshot_id not in manager.host_ready
    assert manager._h2d_lane_reservations == {}
    assert released_host == [record]


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
        "workset_lease": SimpleNamespace(lease_id=1),
        "recovery_claim_id": f"slow:{request.snapshot_id}:child",
        "io_attempt": "slow-h2d:test",
        "io_inflight": False,
        "io_quiesced": False,
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
            get=lambda _snapshot_id: {"state": HostStageState.H2D_LOADING.value},
            mark_d2p_recovery_phase_rank=lambda *_args, **_kwargs: True,
        ),
        h2d_chunk_tokens=64,
        _h2d_stream=object(),
        _h2d_poisoned=False,
        _h2d_staging=object(),
        _h2d_host_bounce=object(),
        workset_broker=SimpleNamespace(
            mark_io_inflight=lambda *_args: None,
            mark_io_quiesced=lambda *_args: True,
        ),
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


def test_completed_host_dma_waits_for_scheduler_owned_handoff_before_release():
    """The I/O worker cannot release Host at an all-rank ledger ACK.

    The final TP binder changes the shared ledger to CONSUMED before the
    scheduler's native COMMIT broadcast reaches both ranks.  Host ownership
    must therefore remain local until gate_request() hands the workset to the
    exact live Req on this rank.
    """

    request = RequestGeneration("host-commit-race", 1)
    load = {
        "request_generation": request,
        "io_error": None,
        "io_complete": True,
    }
    released = []
    manager = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        loads={"child": load},
        ledger=SimpleNamespace(
            get=lambda _snapshot_id: {"state": HostStageState.CONSUMED.value}
        ),
        _h2d_poisoned=False,
        _get_state_lock=nullcontext,
        _release_completed_h2d_host=lambda selected: released.append(selected),
    )

    AgenticPHostStagingManager._progress_h2d_loads(manager)

    assert manager.loads == {"child": load}
    assert released == []


def test_tp1_completed_host_dma_releases_lane_without_group_commit_wait():
    """TP=1 must not retain a completed load in the finite H2D lane pool."""

    request = RequestGeneration("host-complete-tp1", 1)
    load = {
        "request_generation": request,
        "io_error": None,
        "io_complete": True,
    }
    released = []
    manager = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        loads={"child": load},
        ledger=SimpleNamespace(
            get=lambda _snapshot_id: {"state": HostStageState.CONSUMED.value}
        ),
        _h2d_poisoned=False,
        _get_state_lock=nullcontext,
        _release_completed_h2d_host=lambda selected: released.append(selected),
    )

    AgenticPHostStagingManager._progress_h2d_loads(manager)

    assert released == [load]


def test_tp_host_h2d_failure_rearms_complete_host_snapshot_without_recompute():
    request = RequestGeneration("host-failure", 2)
    state = {"value": HostStageState.H2D_LOADING.value}

    class Ledger:
        def get(self, _snapshot_id):
            return {"state": state["value"]}

        def transition(self, _snapshot_id, target, **_kwargs):
            state["value"] = target.value
            return True

        def request_d2p_retry(self, _snapshot_id, _owner, **_kwargs):
            state["value"] = HostStageState.RETRY_PENDING.value
            return True

        def complete_d2p_retry_rank(self, _snapshot_id, _owner, **_kwargs):
            state["value"] = HostStageState.HOST_READY.value
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
        "workset_lease": object(),
        "io_attempt": "slow-h2d:test",
        "io_inflight": False,
        "io_quiesced": False,
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
        workset_broker=SimpleNamespace(
            cancel_io_attempt=lambda *_args: True,
            request_release=lambda snapshot_id, *_args: freed_device.append(
                (snapshot_id,)
            ),
        ),
        _get_state_lock=nullcontext,
        _h2d_poisoned=False,
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
    assert state["value"] == HostStageState.RETRY_PENDING.value
    assert isinstance(load["io_error"], RuntimeError)
    AgenticPHostStagingManager._progress_h2d_loads(manager)
    assert state["value"] == HostStageState.HOST_READY.value
    assert manager.loads == {}
    assert manager.host_ready == {request.snapshot_id: record}
    assert freed_device == [(request.snapshot_id,)]
    assert released_host == []


def test_d_host_extent_open_transient_failure_retains_d_source_and_retries():
    snapshot_id = "slow:d2h-open-retry"
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "arena_write": None,
    }
    entry = {
        "state": HostStageState.HOST_WRITING.value,
        "write_mode": "direct",
    }
    failed = []
    client = SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        relay_enabled=False,
        _start_write=lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EAGAIN, "injected transient open failure")
        ),
        ledger=SimpleNamespace(
            fail_host_write=lambda *_args, **_kwargs: failed.append(1)
        ),
    )

    assert (
        AgenticDHostStagingClient.progress(
            client,
            candidate,
            torch.arange(64),
            entry_snapshot=entry,
        )
        == "waiting"
    )
    assert failed == []
    assert candidate["arena_write"] is None
    assert candidate["arena_write_retry_at"] > time.monotonic()


def test_slow_h2d_cuda_error_quarantines_source_and_destination():
    request = RequestGeneration("h2d-fence-error", 1)

    class BrokenEvent:
        def query(self):
            raise RuntimeError("CUDA event state unavailable")

    released = []
    load = {
        "request_generation": request,
        "event": BrokenEvent(),
        "record": object(),
        "workset_lease": object(),
    }
    manager = SimpleNamespace(
        loads={"child": load},
        workset_broker=SimpleNamespace(
            request_release=lambda *_args: released.append("device")
        ),
        _release_record=lambda _record: released.append("host"),
        _get_state_lock=nullcontext,
    )

    assert not AgenticPHostStagingManager._discard_failed_h2d_load(
        manager, "child", load
    )
    assert load["dma_quarantined"] is True
    assert manager.loads["child"] is load
    assert released == []


def test_slow_h2d_partial_launch_without_fence_is_quarantined():
    request = RequestGeneration("h2d-partial-launch", 1)
    released = []
    launch_fence = H2DLaunchFence(event=object())
    launch_fence.submitted = True
    launch_fence.unavailable = True
    load = {
        "request_generation": request,
        "event": None,
        "launch_fence": launch_fence,
        "record": object(),
        "workset_lease": object(),
    }
    manager = SimpleNamespace(
        loads={"child": load},
        workset_broker=SimpleNamespace(
            request_release=lambda *_args, **_kwargs: released.append("device")
        ),
        _release_record=lambda _record: released.append("host"),
        _get_state_lock=nullcontext,
    )

    assert not AgenticPHostStagingManager._discard_failed_h2d_load(
        manager, "child", load
    )
    assert load["dma_quarantined"] is True
    assert manager.loads["child"] is load
    assert released == []


def test_p2d_host_abort_waits_for_physical_h2d_terminal():
    manager = SimpleNamespace()

    before_submit = AgenticPToDHostReceiver(manager, "before-submit")
    before_submit.abort()
    assert before_submit.poll() == int(KVPoll.Failed)

    inflight = AgenticPToDHostReceiver(manager, "inflight")
    inflight._submitted = True
    inflight._poll = int(KVPoll.Transferring)
    inflight.abort()
    assert inflight.poll() == int(KVPoll.Transferring)
    inflight.mark_terminal(KVPoll.Success)
    assert inflight.poll() == int(KVPoll.Failed)

    unfenced = AgenticPToDHostReceiver(manager, "unfenced")
    unfenced._submitted = True
    unfenced.mark_quarantined(RuntimeError("no DMA fence"))
    unfenced.abort()
    assert unfenced.poll() == int(KVPoll.WaitingForInput)


def test_tp_cancel_phase_keeps_peer_h2d_pages_owned_until_fence():
    receiver = AgenticPToDHostReceiver(SimpleNamespace(), "peer-inflight")
    receiver._submitted = True
    receiver._poll = int(KVPoll.Transferring)
    decode_req = SimpleNamespace(
        req=SimpleNamespace(rid="peer", bootstrap_room=77),
        kv_receiver=receiver,
    )
    transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    transfer_queue.queue = [decode_req]
    transfer_queue._async_poll_lock = threading.Lock()

    transfer_queue.abort_agentic_host_transfers([("peer", 77)])

    assert receiver.abort_pending is True
    assert receiver.poll() == int(KVPoll.Transferring)
    receiver.mark_terminal(KVPoll.Success)
    assert receiver.poll() == int(KVPoll.Failed)


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


def _tp1_common_scheduler(req, load):
    scheduler = object.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_kv_waiting_queue = [(req, time.monotonic())]
    scheduler.agentic_early_direct_completion_queue = deque()
    scheduler.agentic_early_direct_poll_lock = nullcontext()
    scheduler.agentic_host_staging_manager = SimpleNamespace(loads={req.rid: load})
    return scheduler


def test_tp1_slow_is_revisited_by_common_admission(monkeypatch):
    """TP=1 uses the same per-tick arrival scan as TP>1."""

    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "1")
    req = SimpleNamespace(rid="slow-child", _agentic_kv_queue_class="slow")
    scheduler = _tp1_common_scheduler(
        req, {"io_complete": True, "ledger_prepare_pending": False}
    )
    visits = []
    scheduler._agentic_should_defer = (
        lambda request, *_args, **_kwargs: visits.append(request.rid) or True
    )

    Scheduler._drain_agentic_kv_waiting_queue_tp1(scheduler)
    Scheduler._drain_agentic_kv_waiting_queue_tp1(scheduler)

    assert visits == [req.rid, req.rid]
    assert [item[0] for item in scheduler.agentic_kv_waiting_queue] == [req]


def test_tp1_uses_arrival_order_independent_of_kv_source(
    monkeypatch,
):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "2")
    fast = [
        SimpleNamespace(rid=f"fast-{index}", _agentic_kv_queue_class="fast")
        for index in range(3)
    ]
    slow = SimpleNamespace(rid="slow", _agentic_kv_queue_class="slow")
    requests = fast + [slow]
    scheduler = object.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_kv_waiting_queue = [
        (req, time.monotonic() - 10) for req in requests
    ]
    scheduler.agentic_early_direct_completion_queue = deque()
    scheduler.agentic_early_direct_poll_lock = nullcontext()
    scheduler.agentic_host_staging_manager = None
    visited = []
    scheduler._agentic_should_defer = (
        lambda req, *_args, **_kwargs: visited.append(req.rid) or True
    )

    Scheduler._drain_agentic_kv_waiting_queue_tp1(scheduler)

    assert visited == [fast[0].rid, fast[1].rid, fast[2].rid, slow.rid]


def test_prefill_admission_puts_owned_worksets_first_without_source_priority():
    fallback_fast = SimpleNamespace(_agentic_kv_queue_class="fast")
    owned_slow = SimpleNamespace(
        _agentic_kv_queue_class="slow",
        _agentic_workset_backed=True,
        _agentic_workset_suffix_indices=torch.arange(8),
    )
    owned_fast = SimpleNamespace(
        _agentic_kv_queue_class="fast",
        _agentic_workset_backed=True,
        _agentic_workset_suffix_indices=torch.arange(4),
    )
    ordinary_slow = SimpleNamespace(_agentic_kv_queue_class="slow")
    fresh = SimpleNamespace(_agentic_kv_queue_class="new")
    scheduler = SimpleNamespace(
        waiting_queue=[fallback_fast, ordinary_slow, owned_slow, fresh, owned_fast]
    )

    Scheduler._prioritize_agentic_prefill_ready(scheduler)

    assert scheduler.waiting_queue == [
        owned_slow,
        owned_fast,
        fallback_fast,
        ordinary_slow,
        fresh,
    ]


def test_tp_prefill_priority_is_independent_of_local_io_completion_order():
    def req(rid, sequence, priority, queue_class, *, workset_backed=False):
        return SimpleNamespace(
            rid=rid,
            _agentic_tp_prefill_sequence=sequence,
            _agentic_tp_prefill_priority=priority,
            _agentic_kv_queue_class=queue_class,
            _agentic_workset_backed=workset_backed,
            _agentic_workset_suffix_indices=(
                torch.arange(8) if workset_backed else None
            ),
        )

    # The older parent has not acquired a workset; the newer two have.  Both
    # ranks must prioritize the group-committed worksets despite observing
    # their local I/O completion in opposite wall-clock order.
    parent0 = req("parent-0", 0, 0, "slow")
    new1 = req("new-1", 1, 1, "new")
    parent2 = req("parent-2", 2, 0, "fast", workset_backed=True)
    new3 = req("new-3", 3, 1, "new", workset_backed=True)
    rank0 = SimpleNamespace(
        tp_size=2,
        waiting_queue=[parent0, new1, parent2, new3],
    )
    # Model the other shard completing Direct/Slow I/O in the opposite order.
    rank1 = SimpleNamespace(
        tp_size=2,
        waiting_queue=[new3, parent2, new1, parent0],
    )

    Scheduler._prioritize_agentic_prefill_ready(rank0)
    Scheduler._prioritize_agentic_prefill_ready(rank1)

    expected = ["parent-2", "new-3", "parent-0", "new-1"]
    assert [item.rid for item in rank0.waiting_queue] == expected
    assert [item.rid for item in rank1.waiting_queue] == expected


def test_tp_prefill_priority_has_deterministic_fallback_for_legacy_requests():
    first = SimpleNamespace(rid="a", _agentic_tp_prefill_priority=1)
    second = SimpleNamespace(rid="b", _agentic_tp_prefill_priority=1)
    rank0 = SimpleNamespace(tp_size=2, waiting_queue=[second, first])
    rank1 = SimpleNamespace(tp_size=2, waiting_queue=[first, second])

    Scheduler._prioritize_agentic_prefill_ready(rank0)
    Scheduler._prioritize_agentic_prefill_ready(rank1)

    assert [item.rid for item in rank0.waiting_queue] == ["a", "b"]
    assert [item.rid for item in rank1.waiting_queue] == ["a", "b"]


def test_tp_direct_worker_defers_failed_page_release_to_owner_scheduler():
    """A TP ingress worker must not free GPU pages outside the model loop."""

    request = RequestGeneration("request", 7)
    entry = SimpleNamespace(
        request=request,
        completed_at=None,
        transport_poll=KVPoll.Failed,
        started_at=time.monotonic(),
        receiver=SimpleNamespace(),
        workset_lease=None,
        abort_requested=False,
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
        agentic_p_workset_broker=SimpleNamespace(
            request_release=lambda *_args, **_kwargs: None
        ),
        _agentic_snapshot_store=lambda: store,
        _agentic_start_early_direct_receive=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale Direct must not start a receiver")
        ),
    )

    assert not Scheduler._agentic_tp_start_direct_shard(
        scheduler, request, arrived_at=time.time(), prefill_domain=0
    )
    assert scheduler.agentic_tp_direct_local_failed == {request.snapshot_id}


def test_stale_direct_arrival_retires_a_granted_unstarted_workset(monkeypatch):
    """D fallback cannot leave its earlier Direct reservation in P HBM."""

    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    request = RequestGeneration("stale-direct-grant", 1)
    arrived_at = time.time()
    manifest = SimpleNamespace(
        request=request,
        state=SnapshotState.SLOW_FALLBACK,
        created_at=arrived_at,
        token_count=1024,
    )
    cancelled = []
    broker = SimpleNamespace(
        owner_is_superseded=lambda *_args, **_kwargs: False,
        cancel_unstarted=lambda snapshot_id, *, owner=None: cancelled.append(
            (snapshot_id, owner)
        )
    )
    scheduler = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        agentic_early_claim_store=object(),
        agentic_tp_direct_admission_active={},
        agentic_early_direct_admission_queue=deque(
            [
                (
                    request,
                    {"arrived_at": arrived_at, "prompt_token_count": 2048},
                    manifest,
                )
            ]
        ),
        agentic_early_direct_admission_ids={request.snapshot_id},
        agentic_early_direct_receives={},
        agentic_early_direct_terminal={},
        agentic_p_workset_broker=broker,
        server_args=SimpleNamespace(page_size=64),
    )

    Scheduler._agentic_admit_queued_direct_receives(
        scheduler,
        SimpleNamespace(load=lambda *_args, **_kwargs: manifest),
        2.0,
        nullcontext(),
    )

    assert cancelled == [
        (
            request.snapshot_id,
            AgenticPWorksetLeaseBroker.direct_owner(request.snapshot_id),
        )
    ]
    assert not scheduler.agentic_early_direct_admission_queue


def test_direct_grant_refreshes_cached_manifest_before_start(monkeypatch):
    """A queued DIRECT_READY object cannot hide D's later Slow fallback."""

    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    request = RequestGeneration("cached-direct-grant", 1)
    arrived_at = time.time()
    direct = SimpleNamespace(
        request=request,
        state=SnapshotState.DIRECT_READY,
        created_at=arrived_at,
        token_count=1024,
    )
    slow = SimpleNamespace(
        request=request,
        state=SnapshotState.SLOW_FALLBACK,
        created_at=arrived_at,
        token_count=1024,
    )
    authoritative = {"manifest": direct}
    get_calls = []
    lease = object()
    cancelled = []

    def get_lease(*_args, **_kwargs):
        get_calls.append(True)
        return None if len(get_calls) == 1 else lease

    broker = SimpleNamespace(
        owner_is_superseded=lambda *_args, **_kwargs: False,
        request=lambda *_args, **_kwargs: True,
        get=get_lease,
        cancel_unstarted=lambda snapshot_id, *, owner=None: cancelled.append(
            (snapshot_id, owner)
        ),
    )
    scheduler = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        agentic_early_claim_store=object(),
        agentic_tp_direct_admission_active={},
        agentic_early_direct_admission_queue=deque(
            [
                (
                    request,
                    {"arrived_at": arrived_at, "prompt_token_count": 2048},
                    direct,
                )
            ]
        ),
        agentic_early_direct_admission_ids={request.snapshot_id},
        agentic_early_direct_receives={},
        agentic_early_direct_terminal={},
        agentic_p_workset_broker=broker,
        agentic_tp_direct_mailbox=SimpleNamespace(
            publish_receipt=lambda *_args: (_ for _ in ()).throw(
                AssertionError("stale grant must not be published")
            )
        ),
        server_args=SimpleNamespace(page_size=64),
    )
    store = SimpleNamespace(
        load=lambda *_args, **_kwargs: authoritative["manifest"]
    )

    # First pass publishes the intent but receives no physical lease, so the
    # queue retains its cached DIRECT_READY object.
    Scheduler._agentic_admit_queued_direct_receives(
        scheduler, store, 2.0, nullcontext()
    )
    assert len(scheduler.agentic_early_direct_admission_queue) == 1
    assert scheduler.agentic_early_direct_admission_queue[0][2] is direct

    # D falls back before the scheduler grants the intent.  The second pass
    # obtains that grant but must refresh the store and retire it immediately.
    authoritative["manifest"] = slow
    Scheduler._agentic_admit_queued_direct_receives(
        scheduler, store, 2.0, nullcontext()
    )

    assert cancelled == [
        (
            request.snapshot_id,
            AgenticPWorksetLeaseBroker.direct_owner(request.snapshot_id),
        )
    ]
    assert not scheduler.agentic_early_direct_admission_queue
    assert not scheduler.agentic_tp_direct_admission_active


def test_tp_host_timeout_is_diagnostic_and_retains_parent():
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
        _agentic_tp_host_actions={parent.snapshot_id: "prepare"},
        agentic_host_staging_manager=SimpleNamespace(
            gate_request=lambda *_args, **_kwargs: True,
            snapshot_ready=lambda *_args, **_kwargs: False,
        ),
        _agentic_io_active=lambda *_args, **_kwargs: False,
    )

    assert Scheduler._agentic_should_defer(scheduler, req, 0.0)
    assert not getattr(req, "_agentic_kv_gate_complete", False)
    assert not hasattr(req, "_agentic_kv_fallback")


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
        schedulers = []

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
            schedulers.append(scheduler)
            assert SchedulerDisaggregationPrefillMixin._enqueue_deferred_prefill_transfer(
                scheduler, requests[rank]
            )
            assert len(scheduler._prefill_ready_queue) == 1
            assert requests[rank].disagg_p_ready_notified is False

        key = request_generation_key("tp-producer", 4321)
        assert mailboxes[0].group_status(key) == int(KVPoll.Bootstrapping)

        report = SchedulerDisaggregationPrefillMixin._report_tp_prefill_producer_ready
        mailboxes[1].publish_local(key, int(KVPoll.WaitingForInput))
        report(schedulers[1], requests[1])
        assert mailboxes[1].local_status(key) == int(KVPoll.WaitingForInput)

        # TP1 never publishes the logical P-ready marker, so its notified flag
        # remains false.  A terminal worker must nevertheless not let the
        # scheduler enqueue this generation a second time or recreate its
        # sender state after cleanup removes it.
        follower = requests[1]
        follower._async_prefill_transfer_consumer_active = False
        schedulers[1]._prefill_ready_queue.clear()
        schedulers[1]._prefill_ready_queued_keys.clear()
        mailboxes[0].clear_group(key)
        assert SchedulerDisaggregationPrefillMixin._enqueue_deferred_prefill_transfer(
            schedulers[1], follower
        )
        assert len(schedulers[1]._prefill_ready_queue) == 0
        assert mailboxes[1].local_status(key) is None


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
        "workset_plan_epoch": 1,
        "workset_allocation_plan": [],
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
            agentic_p_workset_broker=SimpleNamespace(
                    install_tp_plan=lambda *_args, **_kwargs: None
            ),
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


def test_tp_direct_control_round_trip_preserves_local_workset_lease():
    request = RequestGeneration("lease-round-trip", 1)
    lease = object()
    snapshot_id = request.snapshot_id
    control = {
        Scheduler._AGENTIC_TP_CONTROL_KEY: True,
        "workset_plan_epoch": 1,
        "workset_allocation_plan": [],
        "direct_commands": [
            {
                "snapshot": snapshot_id,
                "request_id": request.request_id,
                "generation": request.generation,
                "action": "poll",
                "arrived_at": 1.0,
                "domain": 0,
                "required_tokens": 1024,
            }
        ],
        "prefill_transfer_keys": [],
        "prefill_transfer_statuses": [],
        "prefill_submit_keys": [],
        "host_commands": [],
    }
    direct_lock = threading.RLock()

    class LockCheckedActive(dict):
        def get(self, key, default=None):
            assert direct_lock._is_owned()
            return super().get(key, default)

        def __setitem__(self, key, value):
            assert direct_lock._is_owned()
            return super().__setitem__(key, value)

    active = LockCheckedActive()
    dict.__setitem__(
        active, snapshot_id, (request, 0.5, 0, 1024, lease)
    )
    owner = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        disaggregation_mode=DisaggregationMode.PREFILL,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
        agentic_tp_direct_admission_active=active,
        agentic_tp_direct_group_status={},
        agentic_tp_direct_local_admitted=set(),
        agentic_tp_direct_local_failed=set(),
        agentic_early_direct_receives={},
        agentic_early_direct_poll_lock=direct_lock,
        agentic_p_workset_broker=SimpleNamespace(
            get=lambda *_args, **_kwargs: None,
                install_tp_plan=lambda *_args, **_kwargs: None,
        ),
        agentic_tp_host_local_admitted=set(),
        agentic_tp_host_active=None,
        agentic_tp_host_active_since=0.0,
        agentic_tp_host_command_visible=False,
        agentic_tp_host_group_status=0,
        agentic_host_staging_manager=None,
    )

    assert Scheduler._agentic_tp_consume_admission_control(owner, [control]) == []
    active = owner.agentic_tp_direct_admission_active[snapshot_id]
    assert len(active) == 5
    assert active[4] is lease


def test_delayed_direct_cleanup_cannot_remove_new_attempt_entry():
    request = RequestGeneration("entry-cas", 1)
    old_entry = SimpleNamespace(
        request=request,
        completed_at=time.monotonic(),
        transport_poll=KVPoll.Success,
        workset_lease=None,
        receiver=object(),
        manifest=SimpleNamespace(),
        claim_id="old",
    )
    new_entry = SimpleNamespace(request=request, claim_id="new")
    scheduler = SimpleNamespace(
        tp_size=1,
        agentic_early_direct_poll_lock=threading.RLock(),
        agentic_early_direct_receives={request.snapshot_id: new_entry},
        agentic_early_direct_terminal={},
        agentic_p_workset_broker=SimpleNamespace(
            request_release=lambda *_args, **_kwargs: False
        ),
    )

    Scheduler._agentic_drop_early_direct_receive(
        scheduler,
        old_entry,
        snapshot_store=object(),
        release_claim=False,
        reason="late_old_cleanup",
    )

    assert scheduler.agentic_early_direct_receives[request.snapshot_id] is new_entry


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


def test_tp_prefill_background_progress_submits_without_scheduler_control():
    """TP0 authorizes prepared shards through tmpfs, not a forward tick."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        namespace = f"p2d-background-{time.time_ns()}"
        sender_mailboxes = [
            TPGroupMailbox(
                namespace,
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]
        receiver_mailboxes = [
            TPGroupMailbox(
                f"{namespace}-receiver",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]
        requests = []
        schedulers = []
        submissions = [0, 0]

        class Sender:
            def poll(self):
                return int(KVPoll.WaitingForInput)

        for rank in range(2):
            req = SimpleNamespace(
                rid="background-submit",
                bootstrap_room=123,
                disagg_kv_sender=Sender(),
                disagg_p_ready_notified=False,
                disagg_p_ready_transfer_started=False,
                _async_prefill_transfer_payload=(1, [rank + 1], None),
            )
            scheduler = SimpleNamespace(
                tp_size=2,
                tp_rank=rank,
                agentic_tp_p2d_sender_mailbox=sender_mailboxes[rank],
                agentic_tp_p2d_receiver_mailbox=receiver_mailboxes[rank],
                agentic_p2d_host_staging_manager=None,
            )
            scheduler._prefill_transfer_key = (
                SchedulerDisaggregationPrefillMixin._prefill_transfer_key
            )
            scheduler._publish_deferred_prefill_ready = (
                lambda request: setattr(request, "disagg_p_ready_notified", True)
            )

            def submit(request, rank=rank):
                submissions[rank] += 1
                request.disagg_p_ready_transfer_started = True
                return True

            scheduler._submit_tp_prefill_transfer = submit
            requests.append(req)
            schedulers.append(scheduler)
            sender_mailboxes[rank].publish_local(
                scheduler._prefill_transfer_key(req), int(KVPoll.Bootstrapping)
            )

        progress = (
            SchedulerDisaggregationPrefillMixin._prefill_transfer_progress_tp_req_once
        )
        # Rank0 may publish P-ready, but cannot authorize transfer until rank1
        # has also observed its matching D receiver.
        assert progress(schedulers[0], requests[0]) == int(KVPoll.Transferring)
        assert submissions == [0, 0]
        assert requests[0].disagg_p_ready_notified
        assert progress(schedulers[1], requests[1]) == int(KVPoll.Transferring)
        assert submissions == [0, 0]

        # No scheduler method is called between these background progress
        # steps. TP0 writes the command and both ranks submit exactly once.
        assert progress(schedulers[0], requests[0]) == int(KVPoll.Transferring)
        assert progress(schedulers[1], requests[1]) == int(KVPoll.Transferring)
        assert submissions == [1, 1]


def test_tp_prefill_background_keeps_scheduler_terminal_after_mailbox_cleanup():
    """A late worker must not resurrect a scheduler-retired transfer."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        namespace = f"p2d-retired-terminal-{time.time_ns()}"
        sender = TPGroupMailbox(
            f"{namespace}-sender", tp_rank=1, tp_size=2, directory=directory
        )
        receiver = TPGroupMailbox(
            f"{namespace}-receiver", tp_rank=1, tp_size=2, directory=directory
        )

        class ClearedSender:
            def poll(self):
                raise AssertionError("retired transport must not be polled")

        request = SimpleNamespace(
            rid="retired-transfer",
            bootstrap_room=321,
            disagg_kv_sender=ClearedSender(),
            disagg_p_ready_transfer_started=True,
            _agentic_p2d_group_terminal=int(KVPoll.Success),
        )
        scheduler = SimpleNamespace(
            tp_size=2,
            tp_rank=1,
            agentic_tp_p2d_sender_mailbox=sender,
            agentic_tp_p2d_receiver_mailbox=receiver,
            agentic_p2d_host_staging_manager=None,
        )
        scheduler._prefill_transfer_key = (
            SchedulerDisaggregationPrefillMixin._prefill_transfer_key
        )

        poll = (
            SchedulerDisaggregationPrefillMixin._prefill_transfer_progress_tp_req_once(
                scheduler, request
            )
        )

        assert poll == int(KVPoll.Success)


def test_tp_prefill_cleanup_waits_for_every_scheduler_rank():
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        namespace = f"p2d-cleanup-{time.time_ns()}"

        def mailbox(name, rank):
            return TPGroupMailbox(
                f"{namespace}-{name}",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )

        ranks = []
        for rank in range(2):
            scheduler = SimpleNamespace(
                tp_size=2,
                tp_rank=rank,
                _prefill_transfer_tp_background_enabled=True,
                agentic_tp_p2d_sender_mailbox=mailbox("sender", rank),
                agentic_tp_p2d_receiver_mailbox=mailbox("receiver", rank),
                agentic_tp_p2d_cleanup_mailbox=mailbox("cleanup", rank),
                _prefill_transfer_cleanup_lock=threading.Lock(),
                _prefill_transfer_cleanup_pending=set(),
            )
            scheduler._prefill_transfer_key = (
                SchedulerDisaggregationPrefillMixin._prefill_transfer_key
            )
            ranks.append(scheduler)

        request = SimpleNamespace(rid="cleanup", bootstrap_room=456)
        key = ranks[0]._prefill_transfer_key(request)
        ranks[0].agentic_tp_p2d_sender_mailbox.publish_receipt(
            key, int(KVPoll.Success)
        )
        clear = SchedulerDisaggregationPrefillMixin._clear_tp_prefill_transfer_mailboxes
        clear(ranks[0], request)
        cleanup_once = (
            SchedulerDisaggregationPrefillMixin._prefill_transfer_cleanup_once
        )
        assert cleanup_once(ranks[0]) == 0
        assert ranks[0].agentic_tp_p2d_sender_mailbox.receipt(key) == int(
            KVPoll.Success
        )

        clear(ranks[1], request)
        assert cleanup_once(ranks[0]) == 1
        assert ranks[0].agentic_tp_p2d_sender_mailbox.receipt(key) is None
        assert key not in ranks[0]._prefill_transfer_cleanup_pending

def test_tp_background_terminal_uses_all_rank_sender_reduction():
    """Native TP control must not retire P pages after only one worker stops."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        namespace = f"p2d-terminal-{time.time_ns()}"
        sender = [
            TPGroupMailbox(
                f"{namespace}-sender",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]
        receiver = [
            TPGroupMailbox(
                f"{namespace}-receiver",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]
        request = SimpleNamespace(
            rid="group-terminal",
            bootstrap_room=901,
            bootstrap_host="127.0.0.1",
            disagg_p_ready_notified=True,
            disagg_p_ready_transfer_started=True,
        )
        lease = object()
        owner = SimpleNamespace(
            tp_size=2,
            tp_rank=0,
            disaggregation_mode=DisaggregationMode.PREFILL,
            _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
            agentic_tp_direct_admission_active={},
            agentic_tp_direct_mailbox=None,
            disagg_prefill_inflight_queue=[request],
            _prefill_transfer_tp_background_enabled=True,
            agentic_tp_p2d_sender_mailbox=sender[0],
            agentic_tp_p2d_receiver_mailbox=receiver[0],
            agentic_p2d_host_staging_manager=None,
            agentic_host_staging_manager=None,
        )
        key = request_generation_key(request.rid, request.bootstrap_room)

        sender[0].publish_local(key, int(KVPoll.Success))
        sender[1].publish_local(key, int(KVPoll.Transferring))
        control = Scheduler._agentic_tp_prepare_admission_control(owner)
        assert control["prefill_transfer_statuses"] == [int(KVPoll.Transferring)]
        assert control["prefill_submit_keys"] == []

        sender[1].publish_local(key, int(KVPoll.Success))
        control = Scheduler._agentic_tp_prepare_admission_control(owner)
        assert control["prefill_transfer_statuses"] == [int(KVPoll.Success)]

        # A failed shard is not a fence for a peer whose sender still owns a
        # live DMA. P source pages remain represented as Transferring until
        # every shard reaches a physical terminal state.
        sender[0].publish_local(key, int(KVPoll.Failed))
        sender[1].publish_local(key, int(KVPoll.Transferring))
        control = Scheduler._agentic_tp_prepare_admission_control(owner)
        assert control["prefill_transfer_statuses"] == [int(KVPoll.Transferring)]

        sender[1].publish_local(key, int(KVPoll.Success))
        control = Scheduler._agentic_tp_prepare_admission_control(owner)
        assert control["prefill_transfer_statuses"] == [int(KVPoll.Failed)]


def test_tp_prefill_submit_failure_becomes_one_group_terminal_result():
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        namespace = f"p2d-failure-{time.time_ns()}"
        sender_mailboxes = [
            TPGroupMailbox(
                f"{namespace}-sender",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]
        receiver_mailboxes = [
            TPGroupMailbox(
                f"{namespace}-receiver",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]

        class Sender:
            def __init__(self, fail):
                self.fail = fail
                self.sent = False

            def poll(self):
                return int(
                    KVPoll.Success if self.sent else KVPoll.WaitingForInput
                )

            def init(self, _pages, _metadata_index):
                if self.fail:
                    raise RuntimeError("injected shard failure")

            def send(self, _page_indices, _state_indices):
                self.sent = True

            def fence_failed_launch(self, _error):
                return KVPoll.Failed

        schedulers = []
        requests = []
        for rank in range(2):
            request = SimpleNamespace(
                rid="submit-failure",
                bootstrap_room=789,
                metadata_buffer_index=1,
                disagg_kv_sender=Sender(fail=rank == 1),
                disagg_p_ready_notified=False,
                disagg_p_ready_transfer_started=False,
                _async_prefill_transfer_payload=(1, [rank + 1], None),
                time_stats=SimpleNamespace(
                    set_prefill_transfer_queue_entry_time=lambda: None
                ),
            )
            scheduler = SimpleNamespace(
                tp_size=2,
                tp_rank=rank,
                agentic_tp_p2d_sender_mailbox=sender_mailboxes[rank],
                agentic_tp_p2d_receiver_mailbox=receiver_mailboxes[rank],
                agentic_p2d_host_staging_manager=None,
            )
            scheduler._prefill_transfer_key = (
                SchedulerDisaggregationPrefillMixin._prefill_transfer_key
            )
            scheduler._publish_deferred_prefill_ready = (
                lambda req: setattr(req, "disagg_p_ready_notified", True)
            )
            scheduler._submit_tp_prefill_transfer = (
                lambda req, scheduler=scheduler: SchedulerDisaggregationPrefillMixin._submit_tp_prefill_transfer(
                    scheduler, req
                )
            )
            schedulers.append(scheduler)
            requests.append(request)
            sender_mailboxes[rank].publish_local(
                scheduler._prefill_transfer_key(request),
                int(KVPoll.Bootstrapping),
            )

        progress = (
            SchedulerDisaggregationPrefillMixin._prefill_transfer_progress_tp_req_once
        )
        progress(schedulers[0], requests[0])
        progress(schedulers[1], requests[1])
        progress(schedulers[0], requests[0])
        # Rank1 reports its injected submit failure but does not terminate
        # independently before TP0 publishes the group result.
        assert progress(schedulers[1], requests[1]) == int(KVPoll.Transferring)
        # The successful peer reaches its own physical terminal before TP0
        # publishes one group failure.
        assert progress(schedulers[0], requests[0]) == int(KVPoll.Failed)
        assert progress(schedulers[1], requests[1]) == int(KVPoll.Failed)


def test_nixl_sender_partial_launch_keeps_source_owned_until_handle_terminal():
    handle = object()
    states = {handle: "PROC"}

    class Agent:
        def transfer(self, submitted_handle):
            assert submitted_handle is handle
            raise RuntimeError("injected failure after post")

        def check_xfer_state(self, submitted_handle):
            return states[submitted_handle]

    manager = NixlKVManager.__new__(NixlKVManager)
    manager.agent = Agent()
    manager.transfer_infos = {}
    manager.request_status = {}

    sender = NixlKVSender.__new__(NixlKVSender)
    sender.kv_mgr = manager
    sender.bootstrap_room = 42
    sender.xfer_handles = []
    sender.has_sent = False
    sender.launch_failed = False
    sender.launch_exception = None

    with pytest.raises(RuntimeError, match="after post") as raised:
        manager._post_transfer(handle, sender.xfer_handles.append, "post failed")
    assert sender.xfer_handles == [handle]
    assert sender.fence_failed_launch(raised.value) == KVPoll.Transferring

    states[handle] = "DONE"
    assert sender.poll() == KVPoll.Failed


def test_nixl_sender_unreadable_handle_quarantines_source_pages():
    class Agent:
        def check_xfer_state(self, _handle):
            raise RuntimeError("transport status unavailable")

    sender = NixlKVSender.__new__(NixlKVSender)
    sender.kv_mgr = SimpleNamespace(
        agent=Agent(), transfer_infos={}, request_status={}
    )
    sender.bootstrap_room = 43
    sender.xfer_handles = [object()]
    sender.has_sent = True
    sender.launch_failed = True
    sender.launch_exception = RuntimeError("control failure")

    assert sender.poll() == KVPoll.Transferring


@pytest.mark.parametrize(
    ("transport_poll", "manifest_state", "expect_released"),
        (
            (KVPoll.Transferring, SnapshotState.DIRECT_LOADING, False),
            (KVPoll.Transferring, SnapshotState.CONSUMED, False),
            (KVPoll.Success, SnapshotState.DIRECT_LOADING, False),
            (KVPoll.Success, SnapshotState.CONSUMED, True),
        ),
)
def test_tp1_direct_release_waits_for_physical_nixl_completion(
    transport_poll, manifest_state, expect_released
):
    snapshot_id = "request:physical-fence"
    released = []
    cleaned = []
    popped = []
    claims = []
    manifest = SimpleNamespace(state=manifest_state)
    candidate = {
        "req": SimpleNamespace(req_pool_idx=1),
        "metadata": SimpleNamespace(
            current=SimpleNamespace(snapshot_id=snapshot_id)
        ),
        "manifest": manifest,
        "sender": SimpleNamespace(poll=lambda: transport_poll),
        "sent": True,
        "local_send_complete": False,
        "staging": False,
        "created_at": time.monotonic(),
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
    }
    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_fast_threshold=2.0,
        agentic_relay_worker=None,
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _cleanup_agentic_direct_sender=lambda value: cleaned.append(value),
        _agentic_release_early_claim=lambda value, reason: claims.append(
            (value, reason)
        ),
        _agentic_candidate_pop=lambda sid: popped.append(sid),
        _enqueue_agentic_release=lambda req, offset: released.append((req, offset)),
        _retire_candidate_for_release=lambda sid, req, offset: (
            popped.append(sid),
            released.append((req, offset)),
        ),
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )

    assert bool(released) is expect_released
    assert bool(cleaned) is expect_released
    assert bool(popped) is expect_released
    assert bool(claims) is expect_released
    assert candidate["local_send_complete"] is (transport_poll == KVPoll.Success)


def test_tp1_direct_poll_exception_quarantines_source_pages():
    snapshot_id = "request:unreadable-fence"

    def unreadable_poll():
        raise RuntimeError("injected status error")

    candidate = {
        "req": SimpleNamespace(req_pool_idx=1),
        "metadata": SimpleNamespace(
            current=SimpleNamespace(snapshot_id=snapshot_id)
        ),
        "manifest": SimpleNamespace(state=SnapshotState.CONSUMED),
        "sender": SimpleNamespace(poll=unreadable_poll),
        "sent": True,
        "local_send_complete": False,
        "staging": False,
        "created_at": time.monotonic(),
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
    }
    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_fast_threshold=2.0,
        agentic_relay_worker=None,
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda _sid, value: value is candidate,
        _agentic_direct_manifest=lambda *_args, **_kwargs: candidate["manifest"],
        _cleanup_agentic_direct_sender=lambda _value: pytest.fail(
            "unreadable transport must not be cleaned"
        ),
        _agentic_release_early_claim=lambda *_args: pytest.fail(
            "unreadable transport must retain the claim"
        ),
        _agentic_candidate_pop=lambda _sid: pytest.fail(
            "unreadable transport must retain the candidate"
        ),
        _enqueue_agentic_release=lambda *_args: pytest.fail(
            "unreadable transport must retain D KV"
        ),
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert not candidate["local_send_complete"]


def test_completed_direct_session_returned_by_p_enters_slow_without_recompute():
    snapshot_id = "request:direct-bind-retry"
    manifest = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.DIRECT_READY,
        token_count=1024,
    )
    candidate = {
        "req": SimpleNamespace(req_pool_idx=1),
        "metadata": SimpleNamespace(current=SimpleNamespace()),
        "manifest": manifest,
        "sender": SimpleNamespace(poll=lambda: KVPoll.Success),
        "sent": True,
        "local_send_complete": False,
        "staging": False,
        "created_at": time.monotonic(),
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
    }
    staged = []
    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_fast_threshold=2.0,
        agentic_relay_worker=None,
        agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda _sid, value: value is candidate,
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _agentic_try_tool_confirmation=lambda _candidate: True,
        _agentic_release_early_claim=lambda *_args: None,
        _agentic_direct_kv_usage=lambda: 0.5,
        _start_agentic_host_staging=lambda value, current: (
            staged.append((value, current)) or value.update(staging=True) or True
        ),
        _publish_agentic_route=lambda *_args, **_kwargs: True,
    )
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )

    assert candidate["local_send_complete"] is True
    assert candidate["staging"] is True
    assert staged == [(candidate, manifest)]


def test_fast_arrival_and_direct_setup_share_one_deadline(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT", "2")
    request = RequestGeneration("request-shared-direct-deadline", 7)
    snapshot_id = request.snapshot_id
    now = time.monotonic()
    manifest = SimpleNamespace(
        snapshot_id=snapshot_id,
        request=request,
        claim_id="direct:edge-claim",
        state=SnapshotState.DIRECT_LOADING,
        token_count=1024,
    )

    class Sender:
        def poll(self):
            return KVPoll.WaitingForInput

        def init(self, *_args, **_kwargs):
            pytest.fail("an expired shared deadline must not initialize Direct")

        def send(self, *_args, **_kwargs):
            pytest.fail("an expired shared deadline must not launch Direct")

    candidate = {
        "req": object(),
        "metadata": SimpleNamespace(current=SimpleNamespace()),
        "manifest": manifest,
        "sender": Sender(),
        "source_page_indices": [1],
        "sent": False,
        "staging": False,
        # Tool returned more than two seconds ago, but P claimed only now.
        # The claim must not reset the tool-return-relative deadline.
        "fast_arrival_seen": False,
        "fast_arrival_seen_at": None,
        "claimed_at": now - 0.1,
        "created_at": now - 3.0,
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
    }
    staged = []
    aborts = []
    releases = []

    marker_store = SimpleNamespace(
        publish_direct_abort=lambda current, claim_id, fence_kind: aborts.append(
            (current, claim_id, fence_kind)
        )
    )

    def observe_tool_arrival(value, _now):
        value["fast_arrival_seen"] = True
        value["fast_arrival_seen_at"] = now - 2.1
        return "arrived"

    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_fast_threshold=2.0,
        agentic_direct_setup_timeout=2.0,
        agentic_force_slow_path=False,
        agentic_relay_worker=None,
        agentic_early_claim_store=marker_store,
        agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
        _agentic_try_early_claim=observe_tool_arrival,
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _agentic_try_tool_confirmation=lambda _candidate: True,
        _agentic_release_early_claim=lambda *_args: releases.append(_args[1]),
        _agentic_direct_kv_usage=lambda: 0.5,
        _start_agentic_host_staging=lambda value, current: (
            staged.append((value, current)) or value.update(staging=True) or True
        ),
        _publish_agentic_route=lambda *_args, **_kwargs: True,
    )
    manager._agentic_publish_unstarted_direct_abort = (
        lambda candidate_value, current_manifest, **kwargs: (
            DecodeKVCacheOffloadManager._agentic_publish_unstarted_direct_abort(
                manager, candidate_value, current_manifest, **kwargs
            )
        )
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )

    # DIRECT_LOADING belongs to P.  D publishes a negative-send fence and
    # waits for P to return the claim instead of attempting an illegal Slow
    # CAS (or retrying it forever).
    assert aborts == [(request, "direct:edge-claim", "unstarted")]
    assert releases == ["direct_setup_expired"]
    assert candidate["direct_abort_claim_id"] == "direct:edge-claim"
    assert candidate["staging"] is False
    assert staged == []

    # Repeated D progress while P is acknowledging the fence is silent and
    # idempotent: no second marker, impossible Slow CAS, or fallback storm.
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert aborts == [(request, "direct:edge-claim", "unstarted")]
    assert releases == ["direct_setup_expired"]
    assert staged == []

    # P's cancellation acknowledgement is the DIRECT_LOADING->DIRECT_READY
    # transition.  The next D pass can now acquire and start Slow normally.
    manifest.state = SnapshotState.DIRECT_READY
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert candidate["staging"] is True
    assert staged == [(candidate, manifest)]


@pytest.mark.parametrize("arrival_offset", [0.2, 1.5])
def test_direct_abort_preserves_fast_identity_before_deleting_arrival(arrival_offset):
    request = RequestGeneration("abort-arrival-race", 1)
    wall, mono = time.time(), time.monotonic()
    manifest = SimpleNamespace(
        request=request, snapshot_id=request.snapshot_id, claim_id="claim",
        created_at=wall, token_count=1024,
    )
    candidate = dict(
        manifest=manifest, sent=False, created_at=mono,
        early_claim_next_poll_at=mono + 100,
    )
    events = []
    marker = {"arrived_at": wall + arrival_offset}
    manager = SimpleNamespace(
        tp_world_size=1, tp_rank=0, agentic_fast_threshold=1.0,
        agentic_direct_setup_timeout=1.0, agentic_early_claim_poll_interval=0.01,
        agentic_early_claim_store=SimpleNamespace(
            read_arrival=lambda *args, **kwargs: events.append("read") or marker,
            publish_direct_abort=lambda *args, **kwargs: events.append("abort"),
        ),
        _agentic_try_tool_confirmation=lambda _: True,
        _agentic_release_early_claim=lambda value, reason: (
            events.append("delete"), marker.clear(), value.pop("fast_arrival_seen", None)
        ),
    )
    manager._agentic_try_early_claim = lambda value, now: (
        DecodeKVCacheOffloadManager._agentic_try_early_claim(manager, value, now)
    )
    assert DecodeKVCacheOffloadManager._agentic_publish_unstarted_direct_abort(
        manager, candidate, manifest, reason="injected_init_failure"
    )
    assert events == ["read", "abort", "delete"]
    assert (candidate.get("fast_arrival_seen_at") is not None) == (arrival_offset <= 1)
    assert candidate["direct_abort_tool_confirmed"]
    assert DecodeKVCacheOffloadManager._agentic_publish_unstarted_direct_abort(
        manager, candidate, manifest, reason="retry"
    )
    assert events == ["read", "abort", "delete"]


def test_direct_abort_publish_failure_is_backed_off_and_does_not_send():
    request = RequestGeneration("direct-abort-publish-retry", 1)
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
        claim_id="direct:publish-retry",
    )
    candidate = {"fast_arrival_seen": True}
    attempts = []

    def fail_publish(*_args, **_kwargs):
        attempts.append(time.monotonic())
        raise OSError("injected tmpfs publication failure")

    manager = SimpleNamespace(
        tp_world_size=1,
        agentic_early_claim_store=SimpleNamespace(
            publish_direct_abort=fail_publish
        ),
        _agentic_try_tool_confirmation=lambda _candidate: True,
        _agentic_release_early_claim=lambda *_args: pytest.fail(
            "an unpublished fence cannot release ingress markers"
        ),
    )
    method = DecodeKVCacheOffloadManager._agentic_publish_unstarted_direct_abort

    assert method(manager, candidate, manifest, reason="test") is True
    assert len(attempts) == 1
    assert candidate["direct_abort_publish_retry_at"] > time.monotonic()
    assert candidate["direct_abort_publish_error_logged"] is True
    assert "direct_abort_claim_id" not in candidate

    # A hot D progress loop observes the retry deadline and performs no
    # additional filesystem operation or exception log.
    assert method(manager, candidate, manifest, reason="test") is True
    assert len(attempts) == 1


def test_partial_direct_send_exception_is_fenced_before_slow_fallback():
    request = RequestGeneration("direct-partial-submit", 1)
    snapshot_id = request.snapshot_id
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=snapshot_id,
        claim_id="direct:partial-submit",
        state=SnapshotState.DIRECT_LOADING,
        token_count=1024,
    )

    class Sender:
        def __init__(self):
            self.status = KVPoll.WaitingForInput
            self.xfer_handles = []
            self.fenced = []

        def poll(self):
            return self.status

        def init(self, *_args, **_kwargs):
            return None

        def send(self, *_args, **_kwargs):
            self.xfer_handles.append(object())
            raise RuntimeError("injected failure after DMA post")

        def fence_failed_launch(self, error):
            self.fenced.append(error)
            self.status = KVPoll.Transferring
            return self.status

    sender = Sender()
    candidate = {
        "req": object(),
        "metadata": SimpleNamespace(current=request),
        "manifest": manifest,
        "sender": sender,
        "source_page_indices": [1],
        "sent": False,
        "staging": False,
        "fast_arrival_seen": True,
        "fast_arrival_seen_at": time.monotonic(),
        "claimed_at": time.monotonic(),
        "created_at": time.monotonic(),
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
    }
    staged = []
    aborts = []
    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_fast_threshold=2.0,
        agentic_direct_setup_timeout=2.0,
        agentic_force_slow_path=False,
        agentic_relay_worker=None,
        agentic_early_claim_store=SimpleNamespace(
            publish_direct_abort=lambda *_args, **_kwargs: aborts.append(True)
        ),
        agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda _sid, value: value is candidate,
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _agentic_try_tool_confirmation=lambda _candidate: True,
        _agentic_release_early_claim=lambda *_args: None,
        _agentic_direct_kv_usage=lambda: 0.5,
        _start_agentic_host_staging=lambda value, current: (
            staged.append((value, current)) or value.update(staging=True) or True
        ),
        _publish_agentic_route=lambda *_args, **_kwargs: True,
    )
    manager._agentic_publish_unstarted_direct_abort = (
        lambda candidate_value, current_manifest, **kwargs: (
            DecodeKVCacheOffloadManager._agentic_publish_unstarted_direct_abort(
                manager, candidate_value, current_manifest, **kwargs
            )
        )
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert candidate["sent"] is True
    assert candidate["direct_launch_failed"] is True
    assert len(sender.xfer_handles) == 1
    assert len(sender.fenced) == 1
    assert aborts == []
    assert staged == []

    # Once every posted handle is terminal, D publishes a claim-scoped
    # no-future-write fence. P can use it even if its final notification was
    # lost, but D still retains the source and cannot enter Slow yet.
    sender.status = KVPoll.Failed
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert aborts == [True]
    assert candidate["direct_abort_claim_id"] == "direct:partial-submit"
    assert staged == []

    # P observes its failed receiver and returns the claim.  Only then can D
    # move the intact source into Shared Host.
    manifest.state = SnapshotState.DIRECT_READY
    manifest.claim_id = None
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert staged == [(candidate, manifest)]


def test_tp1_unstarted_direct_abort_returns_claim_then_releases_workset():
    request = RequestGeneration("direct-abort-unstarted", 1)
    claim_id = "direct:unstarted"
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
        state=SnapshotState.DIRECT_LOADING,
        claim_id=claim_id,
        created_at=time.time() - 1.0,
    )
    lease = SimpleNamespace(state="io_inflight")
    clears = []
    releases = []
    marker_removals = []
    lifecycle_releases = []
    abort_reads = []

    class Broker:
        @staticmethod
        def mark_io_quiesced(snapshot_id, current_lease, attempt):
            assert snapshot_id == request.snapshot_id
            assert current_lease is lease
            assert attempt == claim_id
            current_lease.state = "active"
            return True

        @staticmethod
        def request_release(snapshot_id, current_lease, **_kwargs):
            releases.append((snapshot_id, current_lease))
            current_lease.state = "releasing"

    class SnapshotStore:
        @staticmethod
        def load(current, require_ready=False):
            assert current == request
            assert require_ready is False
            return manifest

        @staticmethod
        def release_direct_claim(current, current_claim_id):
            assert current is manifest
            assert current_claim_id == claim_id
            lifecycle_releases.append(current_claim_id)
            if len(lifecycle_releases) == 1:
                raise RuntimeError("injected transient manifest failure")
            current.state = SnapshotState.DIRECT_READY
            current.claim_id = None
            return current

    def read_direct_abort(*_args, **kwargs):
        abort_reads.append(kwargs)
        return {"kind": "direct-abort", "claim_id": claim_id}

    marker_store = SimpleNamespace(
        read_direct_abort=read_direct_abort,
        remove_direct_abort=lambda current: marker_removals.append(current),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_p_workset_broker = Broker()
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_early_direct_terminal = {}
    scheduler._agentic_clear_direct_receiver = lambda *_args: clears.append(
        "metadata"
    )
    receiver = SimpleNamespace(clear=lambda: clears.append("receiver"))
    entry = AgenticEarlyDirectReceive(
        request=request,
        manifest=manifest,
        claim_id=claim_id,
        receiver=receiver,
        device_indices=None,
        started_at=time.monotonic(),
        arrived_at=time.time() - 3.0,
        workset_lease=lease,
        io_attempt=claim_id,
        transport_poll=KVPoll.WaitingForInput,
    )
    scheduler.agentic_early_direct_receives = {request.snapshot_id: entry}

    # A transient lifecycle publication failure keeps the quiesced receiver,
    # workset and marker retryable.  D still owns and pins the source.
    assert scheduler._agentic_handle_unstarted_direct_abort(
        entry,
        SnapshotStore(),
        marker_store,
        KVPoll.WaitingForInput,
        2.0,
    )
    assert lifecycle_releases == [claim_id]
    assert abort_reads[0]["max_age_seconds"] == float("inf")
    assert request.snapshot_id in scheduler.agentic_early_direct_receives
    assert clears == []
    assert releases == []
    assert marker_removals == []
    assert entry.direct_abort_retry_at > time.monotonic()
    assert entry.direct_abort_error_logged is True

    entry.direct_abort_retry_at = 0.0
    assert scheduler._agentic_handle_unstarted_direct_abort(
        entry,
        SnapshotStore(),
        marker_store,
        KVPoll.WaitingForInput,
        2.0,
    )

    assert lifecycle_releases == [claim_id, claim_id]
    assert manifest.state is SnapshotState.DIRECT_READY
    assert clears == ["receiver", "metadata"]
    assert releases == [(request.snapshot_id, lease)]
    assert marker_removals == [request]
    assert request.snapshot_id not in scheduler.agentic_early_direct_receives


def test_p_direct_abort_lookup_starts_after_deadline_and_backs_off_misses():
    request = RequestGeneration("direct-abort-negative-lookup", 1)
    reads = []
    entry = AgenticEarlyDirectReceive(
        request=request,
        manifest=SimpleNamespace(created_at=time.time()),
        claim_id="direct:negative-lookup",
        receiver=object(),
        device_indices=None,
        started_at=time.monotonic(),
        arrived_at=time.time(),
    )
    marker_store = SimpleNamespace(
        read_direct_abort=lambda *_args, **_kwargs: reads.append(_kwargs) or None
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tp_size = 1
    method = Scheduler._agentic_handle_unstarted_direct_abort

    assert not method(
        scheduler, entry, object(), marker_store, KVPoll.WaitingForInput, 2.0
    )
    assert reads == []

    entry.arrived_at = time.time() - 3.0
    assert not method(
        scheduler, entry, object(), marker_store, KVPoll.WaitingForInput, 2.0
    )
    assert len(reads) == 1
    assert not method(
        scheduler, entry, object(), marker_store, KVPoll.WaitingForInput, 2.0
    )
    assert len(reads) == 1
    assert entry.direct_abort_next_poll_at > time.monotonic()
    assert entry.direct_abort_absent_delay == 0.1


def test_direct_abort_never_releases_pages_when_transport_is_inflight():
    request = RequestGeneration("direct-abort-inflight", 1)
    claim_id = "direct:inflight"
    lease = SimpleNamespace(state="io_inflight")
    release_requests = []
    entry = AgenticEarlyDirectReceive(
        request=request,
        manifest=SimpleNamespace(created_at=time.time() - 1.0),
        claim_id=claim_id,
        receiver=SimpleNamespace(
            clear=lambda: pytest.fail("in-flight receiver must stay registered")
        ),
        device_indices=None,
        started_at=time.monotonic(),
        arrived_at=time.time() - 3.0,
        workset_lease=lease,
        io_attempt=claim_id,
        transport_poll=KVPoll.Transferring,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_p_workset_broker = SimpleNamespace(
        request_release=lambda *args, **kwargs: release_requests.append(
            (args, kwargs)
        ),
        mark_io_quiesced=lambda *_args: pytest.fail(
            "in-flight DMA is not a cancellation fence"
        ),
    )
    marker_store = SimpleNamespace(
        read_direct_abort=lambda *_args, **_kwargs: {
            "kind": "direct-abort",
            "claim_id": claim_id,
        }
    )

    assert scheduler._agentic_handle_unstarted_direct_abort(
        entry,
        SimpleNamespace(),
        marker_store,
        KVPoll.Transferring,
        2.0,
    )
    assert entry.abort_requested is True
    assert entry.abort_release_claim is True
    assert entry.abort_reason == "direct_abort_after_transfer_started"
    assert len(release_requests) == 1
    assert lease.state == "io_inflight"


def test_terminal_direct_fence_returns_claim_immediately_on_inotify_edge():
    request = RequestGeneration("direct-abort-terminal", 1)
    claim_id = "direct:terminal"
    manifest = SimpleNamespace(
        request=request,
        snapshot_id=request.snapshot_id,
        state=SnapshotState.DIRECT_LOADING,
        claim_id=claim_id,
        created_at=time.time(),
    )
    lease = SimpleNamespace(state="io_inflight")

    class Broker:
        @staticmethod
        def mark_io_quiesced(snapshot_id, current_lease, attempt):
            assert snapshot_id == request.snapshot_id
            assert current_lease is lease
            assert attempt == claim_id
            current_lease.state = "active"
            return True

        @staticmethod
        def request_release(_snapshot_id, current_lease, **_kwargs):
            current_lease.state = "releasing"

    class SnapshotStore:
        @staticmethod
        def load(current, require_ready=False):
            assert current == request
            assert require_ready is False
            return manifest

        @staticmethod
        def release_direct_claim(current, current_claim_id):
            assert current is manifest
            assert current_claim_id == claim_id
            current.state = SnapshotState.DIRECT_READY
            current.claim_id = None
            return current

    marker_path = Path("/dev/shm/direct-terminal.json")
    marker_store = SimpleNamespace(
        direct_abort_path=lambda _request: marker_path,
        read_direct_abort=lambda *_args, **_kwargs: {
            "kind": "direct-abort",
            "claim_id": claim_id,
            "fence_kind": "terminal",
        },
        remove_direct_abort=lambda _request: None,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_p_workset_broker = Broker()
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_early_direct_terminal = {}
    scheduler.agentic_direct_abort_pending_names = {marker_path.name}
    scheduler._agentic_clear_direct_receiver = lambda *_args: None
    entry = AgenticEarlyDirectReceive(
        request=request,
        manifest=manifest,
        claim_id=claim_id,
        receiver=SimpleNamespace(clear=lambda: None),
        device_indices=None,
        started_at=time.monotonic(),
        # The ordinary 2s deadline has not expired. The explicit terminal
        # event should still resolve the failed Direct immediately.
        arrived_at=time.time(),
        workset_lease=lease,
        io_attempt=claim_id,
        transport_poll=KVPoll.Transferring,
    )
    scheduler.agentic_early_direct_receives = {request.snapshot_id: entry}

    assert scheduler._agentic_handle_unstarted_direct_abort(
        entry,
        SnapshotStore(),
        marker_store,
        KVPoll.Transferring,
        2.0,
    )
    assert manifest.state is SnapshotState.DIRECT_READY
    assert request.snapshot_id not in scheduler.agentic_early_direct_receives
    assert marker_path.name not in scheduler.agentic_direct_abort_pending_names


def test_tp2_d_abort_marker_waits_for_every_source_rank_fence():
    directory = tempfile.mkdtemp(prefix="d2p-abort-d-", dir="/dev/shm")
    try:
        request = RequestGeneration("direct-abort-d-tp2", 1)
        manifest = SimpleNamespace(
            request=request,
            snapshot_id=request.snapshot_id,
            state=SnapshotState.DIRECT_LOADING,
            claim_id="direct:tp2-d",
        )
        published = []

        def manager(rank):
            return SimpleNamespace(
                tp_world_size=2,
                tp_rank=rank,
                agentic_tp_direct_abort_mailbox=TPGroupMailbox(
                    "test-d2p-abort-d",
                    tp_rank=rank,
                    tp_size=2,
                    directory=directory,
                ),
                agentic_early_claim_store=SimpleNamespace(
                    publish_direct_abort=lambda *args, **kwargs: published.append(
                        (args, kwargs)
                    )
                ),
                _agentic_try_tool_confirmation=lambda _candidate: True,
                _agentic_release_early_claim=lambda *_args: None,
            )

        rank0 = manager(0)
        rank1 = manager(1)
        candidate0 = {"sent": False}
        candidate1 = {"sent": True}
        method = DecodeKVCacheOffloadManager._agentic_publish_unstarted_direct_abort

        # Rank 0 is locally quiescent but cannot publish a group fence yet.
        assert method(
            rank0,
            candidate0,
            manifest,
            reason="deadline",
            fence_kind="unstarted",
        )
        assert published == []

        # Rank 1 reports only after its already-posted handle is terminal.
        assert method(
            rank1,
            candidate1,
            manifest,
            reason="terminal",
            fence_kind="terminal",
        )
        assert published == []

        assert method(
            rank0,
            candidate0,
            manifest,
            reason="all-ranks-terminal",
            fence_kind="unstarted",
        )
        assert len(published) == 1
        assert published[0][1]["claim_id"] == "direct:tp2-d"
        assert published[0][1]["fence_kind"] == "terminal"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_tp2_follower_observes_abort_mailbox_without_scheduler_broadcast():
    directory = tempfile.mkdtemp(prefix="d2p-abort-follower-", dir="/dev/shm")
    try:
        snapshot_id = "direct-abort-follower:1"
        rank0_mailbox = TPGroupMailbox(
            "test-d2p-abort-follower",
            tp_rank=0,
            tp_size=2,
            directory=directory,
        )
        rank1_mailbox = TPGroupMailbox(
            "test-d2p-abort-follower",
            tp_rank=1,
            tp_size=2,
            directory=directory,
        )
        rank0_mailbox.publish_local_progress(snapshot_id, 2)
        candidate = {
            "req": object(),
            "manifest": SimpleNamespace(snapshot_id=snapshot_id),
            "sender": None,
            "sent": False,
            # The native scheduler broadcast has not reached this rank and
            # local sender setup has not happened. The independent mailbox
            # still has to stop setup and close the group fence.
            "tp_command": "wait",
            "io_lock": threading.RLock(),
        }
        manager = SimpleNamespace(
            tp_world_size=2,
            tp_rank=1,
            agentic_relay_worker=None,
            agentic_tp_direct_abort_mailbox=rank1_mailbox,
            _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
            _agentic_candidate_is_live_locked=lambda sid, value: (
                sid == snapshot_id and value is candidate
            ),
        )

        DecodeKVCacheOffloadManager._check_agentic_tp_follower_progress(
            manager, progress_relay=False, progress_class="direct"
        )

        assert candidate["tp_direct_abort_requested"] is True
        assert rank1_mailbox.local_status(snapshot_id) == 2

        # After TP0 observes P's DIRECT_READY acknowledgement, its Slow
        # command authoritatively exits the local abort tombstone even if a
        # stale peer status file is still visible.
        candidate["tp_command"] = "slow"
        candidate["manifest"].state = SnapshotState.SLOW_FALLBACK
        candidate["local_prepared"] = True
        candidate["setup_committed"] = True
        candidate["setup_logged"] = True
        candidate["source_token_indices"] = [1]
        manager.agentic_host_staging_client = None
        manager._start_agentic_host_staging = (
            lambda value, _manifest: value.update(staging=True) or True
        )
        DecodeKVCacheOffloadManager._check_agentic_tp_follower_progress(
            manager, progress_relay=False, progress_class="slow"
        )
        assert candidate.get("tp_direct_abort_requested") is None
        assert candidate["staging"] is True
        assert rank1_mailbox.local_status(snapshot_id) is None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_tp2_rank0_exits_abort_after_p_returns_claim_and_starts_slow():
    directory = tempfile.mkdtemp(prefix="d2p-abort-ack-", dir="/dev/shm")
    try:
        request = RequestGeneration("direct-abort-ack-tp2", 1)
        snapshot_id = request.snapshot_id
        loading = SimpleNamespace(
            request=request,
            snapshot_id=snapshot_id,
            state=SnapshotState.DIRECT_LOADING,
            claim_id="direct:tp2-ack",
            token_count=16,
        )
        ready = SimpleNamespace(
            request=request,
            snapshot_id=snapshot_id,
            state=SnapshotState.DIRECT_READY,
            claim_id=None,
            token_count=16,
        )
        candidate = {
            "req": object(),
            "metadata": SimpleNamespace(current=request),
            "manifest": loading,
            "sender": SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
            "sent": False,
            "staging": False,
            "tp_direct_abort_requested": True,
            "direct_abort_claim_id": "direct:tp2-ack",
            "created_at": time.monotonic(),
            "claimed_at": time.monotonic(),
            "fallback_retry_at": 0.0,
            "fast_arrival_seen": True,
            "io_lock": threading.RLock(),
        }
        staged = []
        mailbox = TPGroupMailbox(
            "test-d2p-abort-ack",
            tp_rank=0,
            tp_size=2,
            directory=directory,
        )
        original_clear_group = mailbox.clear_group
        clear_attempts = []

        def flaky_clear_group(current_snapshot_id):
            clear_attempts.append(current_snapshot_id)
            if len(clear_attempts) == 1:
                raise OSError("injected TP abort cleanup failure")
            return original_clear_group(current_snapshot_id)

        mailbox.clear_group = flaky_clear_group

        def load_manifest(value, _metadata, _now, force=False):
            assert value is candidate
            if force:
                value["manifest"] = ready
            return value["manifest"]

        manager = SimpleNamespace(
            tp_world_size=2,
            tp_rank=0,
            agentic_fast_threshold=2.0,
            agentic_direct_setup_timeout=2.0,
            agentic_force_slow_path=False,
            agentic_relay_worker=None,
            agentic_early_claim_store=object(),
            agentic_tp_direct_abort_mailbox=mailbox,
            agentic_host_staging_client=object(),
            _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
            _agentic_try_final_confirmation=lambda _candidate: False,
            _agentic_candidate_is_live_locked=lambda sid, value: (
                sid == snapshot_id and value is candidate
            ),
            _agentic_direct_manifest=load_manifest,
            _agentic_try_early_claim=lambda *_args: "missing",
            _agentic_try_tool_confirmation=lambda _candidate: True,
            _agentic_release_early_claim=lambda *_args: None,
            _agentic_direct_kv_usage=lambda: 0.5,
            _start_agentic_host_staging=lambda value, current: (
                staged.append((value, current))
                or value.update(staging=True)
                or True
            ),
            _publish_agentic_route=lambda *_args, **_kwargs: True,
        )

        DecodeKVCacheOffloadManager._check_agentic_direct_progress(
            manager, progress_relay=False
        )

        assert candidate["tp_direct_abort_requested"] is True
        assert candidate["direct_abort_claim_id"] == "direct:tp2-ack"
        assert candidate["staging"] is False
        assert staged == []

        candidate["tp_abort_mailbox_retry_at"] = 0.0
        DecodeKVCacheOffloadManager._check_agentic_direct_progress(
            manager, progress_relay=False
        )

        assert candidate.get("tp_direct_abort_requested") is None
        assert candidate.get("direct_abort_claim_id") is None
        assert candidate["staging"] is True
        assert staged == [(candidate, ready)]
        assert clear_attempts == [snapshot_id, snapshot_id]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_tp2_follower_rechecks_abort_under_sender_post_lock():
    snapshot_id = "direct-abort-post-race:1"
    published = []

    class RacingMailbox:
        def __init__(self):
            self.reads = 0

        def local_status(self, _snapshot_id, rank):
            self.reads += 1
            # The loop-level scan sees no abort on either rank. The second
            # scan, immediately before sender.poll/init/send, observes TP0's
            # newly published fence.
            return 2 if self.reads >= 3 and rank == 0 else None

        @staticmethod
        def publish_local_progress(_snapshot_id, status):
            published.append(status)

    sender = SimpleNamespace(
        poll=lambda: pytest.fail("post-lock abort recheck must precede poll"),
        init=lambda *_args, **_kwargs: pytest.fail("must not initialize"),
        send=lambda *_args, **_kwargs: pytest.fail("must not post DMA"),
    )
    candidate = {
        "req": object(),
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "sender": sender,
        "source_page_indices": [1],
        "sent": False,
        "tp_command": "direct",
        "local_prepared": True,
        "setup_committed": True,
        "setup_logged": True,
        "io_lock": threading.RLock(),
    }
    manager = SimpleNamespace(
        tp_world_size=2,
        tp_rank=1,
        agentic_relay_worker=None,
        agentic_tp_direct_abort_mailbox=RacingMailbox(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
    )

    DecodeKVCacheOffloadManager._check_agentic_tp_follower_progress(
        manager, progress_relay=False, progress_class="direct"
    )

    assert candidate["sent"] is False
    assert candidate["tp_direct_abort_requested"] is True
    assert published == [2]


def test_tp2_follower_rechecks_command_epoch_under_sender_post_lock(monkeypatch):
    snapshot_id = "direct-slow-command-race:1"
    sender = SimpleNamespace(
        poll=lambda: pytest.fail("stale Direct epoch must not poll sender"),
        init=lambda *_args, **_kwargs: pytest.fail("must not initialize"),
        send=lambda *_args, **_kwargs: pytest.fail("must not post DMA"),
    )
    candidate = {
        "req": object(),
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "sender": sender,
        "source_page_indices": [1],
        "sent": False,
        "tp_command": "direct",
        "local_prepared": True,
        "setup_committed": True,
        "setup_logged": True,
        "io_lock": threading.RLock(),
    }
    mailbox = SimpleNamespace(local_status=lambda *_args: None)
    manager = SimpleNamespace(
        tp_world_size=2,
        tp_rank=1,
        agentic_relay_worker=None,
        agentic_tp_direct_abort_mailbox=mailbox,
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
    )

    def install_slow_between_checks(_self, value, _now):
        # Model _apply_tp_candidate_command winning after the loop-level
        # action read but before the sender-post critical section.
        with value["io_lock"]:
            value["tp_command"] = "slow"
            value["manifest"].state = SnapshotState.SLOW_FALLBACK
            value.pop("tp_direct_abort_requested", None)
        return True

    monkeypatch.setattr(
        DecodeKVCacheOffloadManager,
        "_progress_agentic_direct_candidate_setup",
        install_slow_between_checks,
    )

    DecodeKVCacheOffloadManager._check_agentic_tp_follower_progress(
        manager, progress_relay=False, progress_class="direct"
    )

    assert candidate["sent"] is False
    assert candidate["tp_command"] == "slow"


def test_tp_abort_mailbox_write_failures_are_backed_off_per_request():
    calls = []

    def fail_write():
        calls.append(True)
        raise OSError("injected tmpfs failure")

    candidate = {}
    d_manager = SimpleNamespace()
    d_method = DecodeKVCacheOffloadManager._agentic_tp_abort_mailbox_write
    assert not d_method(d_manager, candidate, "mailbox-backoff:1", fail_write)
    assert candidate["tp_direct_abort_requested"] is True
    assert candidate["tp_abort_mailbox_error_logged"] is True
    assert candidate["tp_abort_mailbox_retry_at"] > time.monotonic()
    assert not d_method(d_manager, candidate, "mailbox-backoff:1", fail_write)
    assert len(calls) == 1

    entry = SimpleNamespace(
        request=RequestGeneration("p-mailbox-backoff", 1)
    )
    p_scheduler = Scheduler.__new__(Scheduler)
    p_method = Scheduler._agentic_tp_abort_mailbox_write
    assert not p_method(p_scheduler, entry, fail_write)
    assert entry.tp_abort_mailbox_error_logged is True
    assert entry.tp_abort_mailbox_retry_at > time.monotonic()
    assert not p_method(p_scheduler, entry, fail_write)
    assert len(calls) == 2


def test_tp2_p_abort_returns_claim_only_after_every_destination_rank_fence():
    directory = tempfile.mkdtemp(prefix="d2p-abort-p-", dir="/dev/shm")
    try:
        request = RequestGeneration("direct-abort-p-tp2", 1)
        claim_id = "direct:tp2-p"
        manifest = SimpleNamespace(
            request=request,
            snapshot_id=request.snapshot_id,
            state=SnapshotState.DIRECT_LOADING,
            claim_id=claim_id,
            created_at=time.time(),
        )
        marker_path = Path(directory) / "group-terminal.json"
        lifecycle_releases = []
        marker_removals = []

        class SnapshotStore:
            @staticmethod
            def load(current, require_ready=False):
                assert current == request
                assert require_ready is False
                return manifest

            @staticmethod
            def release_direct_claim(current, current_claim_id):
                assert current is manifest
                assert current_claim_id == claim_id
                lifecycle_releases.append(current_claim_id)
                current.state = SnapshotState.DIRECT_READY
                current.claim_id = None
                return current

        marker_store = SimpleNamespace(
            direct_abort_path=lambda _request: marker_path,
            read_direct_abort=lambda *_args, **_kwargs: {
                "kind": "direct-abort",
                "claim_id": claim_id,
                "fence_kind": "terminal",
            },
            remove_direct_abort=lambda current: marker_removals.append(current),
        )

        def scheduler_and_entry(rank):
            lease = SimpleNamespace(state="io_inflight")

            class Broker:
                @staticmethod
                def mark_io_quiesced(_snapshot_id, current_lease, attempt):
                    assert current_lease is lease
                    assert attempt == claim_id
                    current_lease.state = "active"
                    return True

                @staticmethod
                def request_release(_snapshot_id, current_lease, **_kwargs):
                    current_lease.state = "releasing"

            scheduler = Scheduler.__new__(Scheduler)
            scheduler.tp_size = 2
            scheduler.tp_rank = rank
            scheduler.agentic_tp_direct_abort_mailbox = TPGroupMailbox(
                "test-d2p-abort-p",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            scheduler.agentic_p_workset_broker = Broker()
            scheduler.agentic_early_direct_poll_lock = threading.RLock()
            scheduler.agentic_early_direct_terminal = {}
            scheduler.agentic_direct_abort_pending_names = {marker_path.name}
            scheduler._agentic_clear_direct_receiver = lambda *_args: None
            entry = AgenticEarlyDirectReceive(
                request=request,
                manifest=manifest,
                claim_id=claim_id,
                receiver=SimpleNamespace(clear=lambda: None),
                device_indices=None,
                started_at=time.monotonic(),
                arrived_at=time.time(),
                workset_lease=lease,
                io_attempt=claim_id,
                transport_poll=KVPoll.WaitingForInput,
            )
            scheduler.agentic_early_direct_receives = {
                request.snapshot_id: entry
            }
            return scheduler, entry

        rank0, entry0 = scheduler_and_entry(0)
        rank1, entry1 = scheduler_and_entry(1)
        rank0_mailbox = rank0.agentic_tp_direct_abort_mailbox
        original_clear_group = rank0_mailbox.clear_group
        clear_attempts = []

        def flaky_clear_group(current_snapshot_id):
            clear_attempts.append(current_snapshot_id)
            if len(clear_attempts) == 1:
                raise OSError("injected P TP abort cleanup failure")
            return original_clear_group(current_snapshot_id)

        rank0_mailbox.clear_group = flaky_clear_group
        method = Scheduler._agentic_handle_unstarted_direct_abort

        # A single P rank may quiesce locally but cannot return shared state.
        assert method(
            rank1, entry1, SnapshotStore(), marker_store, KVPoll.WaitingForInput, 2.0
        )
        assert manifest.state is SnapshotState.DIRECT_LOADING
        assert lifecycle_releases == []

        # TP0 observes both destination fences and performs the sole CAS.
        assert method(
            rank0, entry0, SnapshotStore(), marker_store, KVPoll.WaitingForInput, 2.0
        )
        assert manifest.state is SnapshotState.DIRECT_READY
        assert lifecycle_releases == [claim_id]
        assert request.snapshot_id in rank0.agentic_early_direct_receives
        assert marker_removals == []

        entry0.tp_abort_mailbox_retry_at = 0.0
        assert method(
            rank0, entry0, SnapshotStore(), marker_store, KVPoll.WaitingForInput, 2.0
        )
        assert marker_removals == [request]
        assert clear_attempts == [request.snapshot_id, request.snapshot_id]

        # The follower then observes DIRECT_READY and releases its pages.
        assert method(
            rank1, entry1, SnapshotStore(), marker_store, KVPoll.WaitingForInput, 2.0
        )
        assert request.snapshot_id not in rank0.agentic_early_direct_receives
        assert request.snapshot_id not in rank1.agentic_early_direct_receives
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.parametrize("tp_world_size", [1, 2])
def test_slow_fallback_offer_retry_retains_d_kv_until_host_staging(tp_world_size):
    snapshot_id = f"request:slow-retry:tp{tp_world_size}"
    manifest = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.DIRECT_READY,
        token_count=1024,
    )
    candidate = {
        "req": object(),
        "metadata": SimpleNamespace(current=SimpleNamespace()),
        "manifest": manifest,
        "sender": SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
        "sent": False,
        "staging": False,
        "claimed_at": None,
        "created_at": time.monotonic() - 3.0,
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
    }
    attempts = []
    releases = []
    popped = []
    routes = []

    def start_host(value, current):
        attempts.append(current.state)
        if len(attempts) == 1:
            # Model begin_slow_fallback() committing ownership before the
            # first Shared-Arena offer raises.
            current.state = SnapshotState.SLOW_FALLBACK
            value["manifest"] = current
            raise RuntimeError("injected Host offer failure")
        value["staging"] = True
        return True

    manager = SimpleNamespace(
        tp_world_size=tp_world_size,
        tp_rank=0,
        agentic_fast_threshold=2.0,
        agentic_direct_setup_timeout=2.0,
        agentic_relay_worker=None,
        agentic_early_claim_store=object(),
        agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
        _agentic_try_early_claim=lambda _candidate, _now: "absent",
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _agentic_try_tool_confirmation=lambda _candidate: False,
        _agentic_release_early_claim=lambda *_args: None,
        _agentic_direct_kv_usage=lambda: 0.5,
        _start_agentic_host_staging=start_host,
        _publish_agentic_route=lambda *_args, **kwargs: (
            routes.append(kwargs) or True
        ),
        _cleanup_agentic_direct_sender=lambda *_args: pytest.fail(
            "non-durable slow fallback must retain the sender"
        ),
        _agentic_candidate_pop=lambda sid: popped.append(sid),
        _enqueue_agentic_release=lambda req, offset: releases.append((req, offset)),
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert manifest.state is SnapshotState.SLOW_FALLBACK
    assert not candidate["staging"]
    assert not releases and not popped

    candidate["fallback_retry_at"] = 0.0
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert candidate["staging"]
    assert attempts == [SnapshotState.DIRECT_READY, SnapshotState.SLOW_FALLBACK]
    assert routes and routes[-1]["route"] == "host_writing"
    assert not releases and not popped


def test_force_slow_ablation_stages_immediately_without_direct_wait():
    snapshot_id = "request:force-slow"
    manifest = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.DIRECT_READY,
        token_count=1024,
    )
    candidate = {
        "req": object(),
        "metadata": SimpleNamespace(current=SimpleNamespace()),
        "manifest": manifest,
        "sender": SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
        "sent": False,
        "staging": False,
        "claimed_at": None,
        "created_at": time.monotonic(),
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
    }
    staged = []
    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_force_slow_path=True,
        agentic_fast_threshold=2.0,
        agentic_direct_setup_timeout=2.0,
        agentic_relay_worker=None,
        agentic_early_claim_store=object(),
        agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
        _agentic_try_early_claim=lambda _candidate, _now: "absent",
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _agentic_try_tool_confirmation=lambda _candidate: False,
        _agentic_release_early_claim=lambda *_args: None,
        _agentic_direct_kv_usage=lambda: 0.5,
        _start_agentic_host_staging=lambda value, current: (
            staged.append((value, current)) or value.update(staging=True) or True
        ),
        _publish_agentic_route=lambda *_args, **_kwargs: True,
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )

    assert staged == [(candidate, manifest)]
    assert candidate["staging"] is True


@pytest.mark.parametrize(
    ("tp_world_size", "fast_arrival_seen", "expected_recompute", "expected_stage"),
    [
        (1, True, True, False),
        (2, True, True, False),
        (1, False, False, True),
    ],
)
@pytest.mark.parametrize("returned_claim", [False, True])
@pytest.mark.parametrize("adaptive_blocked", [False, True])
def test_fast_direct_failure_recompute_keeps_slow_tools_on_host(
    tp_world_size, fast_arrival_seen, expected_recompute, expected_stage,
    returned_claim, adaptive_blocked, monkeypatch,
):
    snapshot_id = f"request:fast-recompute:{fast_arrival_seen}"
    manifest = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.DIRECT_READY,
        token_count=1024,
    )
    candidate = {
        "req": object(),
        "metadata": SimpleNamespace(
            current=SimpleNamespace(snapshot_id=snapshot_id)
        ),
        "manifest": manifest,
        "sender": SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
        "sent": False,
        "staging": False,
        "claimed_at": None,
        "created_at": time.monotonic() - 2.0,
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
        "fast_arrival_seen": fast_arrival_seen,
        "fast_arrival_seen_at": (
            time.monotonic() - 1.5 if fast_arrival_seen else None
        ),
    }
    recomputes = []
    if returned_claim:
        candidate["claimed_at"] = time.monotonic() - 0.5
        # Marker cleanup drops the bool but keeps validated arrival time.
        candidate["fast_arrival_seen"] = False
        candidate["direct_abort_tool_confirmed"] = True
    staged = []
    releases = []
    cleanups = []
    manager = SimpleNamespace(
        tp_world_size=tp_world_size,
        tp_rank=0,
        agentic_force_slow_path=False,
        agentic_fast_direct_failure_recompute=True,
        agentic_fast_threshold=1.0,
        agentic_direct_setup_timeout=1.0,
        agentic_early_claim_post_timeout=0.0,
        agentic_relay_worker=None,
        agentic_early_claim_store=object(),
        agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
        _agentic_try_early_claim=lambda _candidate, _now: (
            "arrived" if fast_arrival_seen else "absent"
        ),
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _agentic_try_tool_confirmation=lambda _candidate: True,
        _agentic_release_early_claim=lambda *_args: None,
        _agentic_direct_kv_usage=lambda: 0.5,
        agentic_snapshot_store=SimpleNamespace(
            fail_direct_offer=lambda *_args, **_kwargs: SimpleNamespace(
                state=SnapshotState.FAILED
            )
        ),
        _start_agentic_host_staging=lambda value, current: (
            staged.append((value, current)) or value.update(staging=True) or True
        ),
        _publish_agentic_route=lambda *_args, **kwargs: (
            recomputes.append(snapshot_id)
            if kwargs.get("route") == "recompute"
            else None
        )
        or True,
        _cleanup_agentic_direct_sender=lambda value: cleanups.append(value),
        _retire_candidate_for_release=lambda sid, req, offset: releases.append(
            (sid, req, offset)
        ),
    )

    if adaptive_blocked:
        monkeypatch.setenv("SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE", "true")
        manager._slow_congestion_reader = SimpleNamespace(
            congested=lambda: True, sample={"q": 32}
        )
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )

    assert bool(recomputes) is expected_recompute
    assert bool(staged) is expected_stage
    if expected_recompute:
        assert releases == [(snapshot_id, candidate["req"], 0)]
        assert cleanups == [candidate]
    else:
        assert not releases and not cleanups


@pytest.mark.parametrize("sent", [False, True])
@pytest.mark.parametrize("poll_state", [KVPoll.Success, KVPoll.Failed, KVPoll.Transferring])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("adaptive", [None, False, True])
def test_returned_direct_claim_policy_respects_physical_fence(sent, poll_state, enabled, adaptive, monkeypatch):
    request = RequestGeneration("returned-direct-policy", 1)
    manifest = SimpleNamespace(
        request=request, snapshot_id=request.snapshot_id,
        state=SnapshotState.DIRECT_READY, token_count=1024,
    )
    candidate = dict(
        req=object(), metadata=SimpleNamespace(current=request), manifest=manifest,
        sender=SimpleNamespace(poll=lambda: poll_state), sent=sent,
        local_send_complete=False, staging=False, claimed_at=time.monotonic(),
        created_at=time.monotonic(), fallback_retry_at=0.0,
        io_lock=threading.RLock(), fast_arrival_seen_at=time.monotonic(),
    )
    events = []
    manager = SimpleNamespace(
        tp_world_size=1, tp_rank=0, agentic_relay_worker=None,
        agentic_fast_threshold=1000.0, agentic_fast_direct_failure_recompute=enabled,
        agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((request.snapshot_id, candidate),),
        _agentic_candidate_is_live_locked=lambda sid, value: value is candidate,
        _agentic_try_final_confirmation=lambda _: False,
        _agentic_direct_manifest=lambda *args, **kwargs: manifest,
        _agentic_try_tool_confirmation=lambda _: True,
        _agentic_direct_ready_timeout=lambda _: (1000.0, 0.5),
        _agentic_direct_kv_usage=lambda: 0.5,
        _agentic_release_early_claim=lambda *args: None,
        agentic_snapshot_store=SimpleNamespace(
            fail_direct_offer=lambda *args, **kwargs: (
                events.append("terminal") or SimpleNamespace(state=SnapshotState.FAILED)
            )
        ),
        _publish_agentic_route=lambda *args, **kwargs: (
            events.append(kwargs["route"]) or True
        ),
        _cleanup_agentic_direct_sender=lambda _: events.append("cleanup"),
        _retire_candidate_for_release=lambda *args: events.append("release"),
        _start_agentic_host_staging=lambda value, current: (
            events.append("slow") or value.update(staging=True) or True
        ),
    )
    if adaptive is not None:
        monkeypatch.setenv("SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE", "true")
        manager._slow_congestion_reader = SimpleNamespace(
            congested=lambda: adaptive, sample={"q": 32 if adaptive else 0}
        )
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(manager, progress_relay=False)
    if sent and poll_state == KVPoll.Transferring:
        assert events == []
    elif (enabled if adaptive is None else adaptive):
        assert events == ["terminal", "recompute", "cleanup", "release"]
    else:
        assert events == ["slow", "host_writing"]


@pytest.mark.parametrize(
    ("refreshed_state", "expected_release"),
    [(SnapshotState.P_RECEIVED, False), (SnapshotState.CONSUMED, True)],
)
def test_fast_direct_recompute_force_refresh_respects_p_ownership(
    refreshed_state, expected_release
):
    snapshot_id = f"request:refresh-race:{refreshed_state.value}"
    initial = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.DIRECT_READY,
        token_count=1024,
    )
    refreshed = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=refreshed_state,
        token_count=1024,
    )
    candidate = {
        "req": object(),
        "metadata": SimpleNamespace(current=SimpleNamespace()),
        "manifest": initial,
        "sender": SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
        "sent": False,
        "staging": False,
        "claimed_at": None,
        "created_at": time.monotonic() - 2.0,
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
        "fast_arrival_seen": True,
        "fast_arrival_seen_at": time.monotonic() - 1.5,
    }
    manifests = iter((initial, refreshed))
    recomputes = []
    releases = []
    cleanups = []
    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_force_slow_path=False,
        agentic_fast_direct_failure_recompute=True,
        agentic_fast_threshold=1.0,
        agentic_direct_setup_timeout=1.0,
        agentic_early_claim_post_timeout=0.0,
        agentic_relay_worker=None,
        agentic_early_claim_store=object(),
        agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
        _agentic_try_early_claim=lambda *_args: "arrived",
        _agentic_direct_manifest=lambda *_args, **_kwargs: next(manifests),
        _agentic_try_tool_confirmation=lambda _candidate: True,
        _agentic_release_early_claim=lambda *_args: None,
        _agentic_direct_kv_usage=lambda: 0.5,
        _publish_agentic_failure=lambda *_args, **_kwargs: (
            recomputes.append(snapshot_id) or True
        ),
        _start_agentic_host_staging=lambda *_args, **_kwargs: pytest.fail(
            "P-owned refreshed state must not enter Host fallback"
        ),
        _cleanup_agentic_direct_sender=lambda value: cleanups.append(value),
        _retire_candidate_for_release=lambda sid, req, offset: releases.append(
            (sid, req, offset)
        ),
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )

    assert not recomputes
    assert bool(releases) is expected_release
    assert bool(cleanups) is expected_release


def test_fast_direct_recompute_atomic_claim_loss_retains_d_source():
    """A concurrent P claim must win without D releasing or staging its KV."""

    snapshot_id = "request:fast-recompute-claim-loss"
    manifest = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.DIRECT_READY,
        token_count=1024,
    )
    candidate = {
        "req": object(),
        "metadata": SimpleNamespace(
            current=SimpleNamespace(snapshot_id=snapshot_id)
        ),
        "manifest": manifest,
        "sender": SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
        "sent": False,
        "staging": False,
        "claimed_at": None,
        "created_at": time.monotonic() - 2.0,
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
        "fast_arrival_seen": True,
        "fast_arrival_seen_at": time.monotonic() - 1.5,
    }
    routes = []
    releases = []
    cleanups = []
    manager = SimpleNamespace(
        tp_world_size=2,
        tp_rank=0,
        agentic_force_slow_path=False,
        agentic_fast_direct_failure_recompute=True,
        agentic_fast_threshold=1.0,
        agentic_direct_setup_timeout=1.0,
        agentic_early_claim_post_timeout=0.0,
        agentic_relay_worker=None,
        agentic_early_claim_store=object(),
        agentic_host_staging_client=object(),
        agentic_snapshot_store=SimpleNamespace(
            fail_direct_offer=lambda *_args, **_kwargs: None
        ),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
        _agentic_try_early_claim=lambda *_args: "arrived",
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _agentic_try_tool_confirmation=lambda _candidate: True,
        _agentic_release_early_claim=lambda *_args: None,
        _agentic_direct_kv_usage=lambda: 0.5,
        _publish_agentic_route=lambda *_args, **_kwargs: routes.append(snapshot_id),
        _start_agentic_host_staging=lambda *_args, **_kwargs: pytest.fail(
            "atomic claim loss must not enter Host fallback"
        ),
        _cleanup_agentic_direct_sender=lambda value: cleanups.append(value),
        _retire_candidate_for_release=lambda sid, req, offset: releases.append(
            (sid, req, offset)
        ),
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )

    assert routes == []
    assert releases == []
    assert cleanups == []
    assert candidate["fallback_retry_at"] > time.monotonic() - 1.0


def test_fast_direct_recompute_tp_release_uses_native_group_handoff():
    """TP recompute retirement enters the existing rank-0 release broadcast."""

    snapshot_id = "request:fast-recompute-tp:3"
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            custom_params={
                "agentic_request_id": "request:fast-recompute-tp",
                "agentic_generation": 3,
                "agentic_parent_generation": 2,
            }
        )
    )
    candidate = {"req": req}
    manager = SimpleNamespace(
        tp_world_size=2,
        agentic_direct_candidates={snapshot_id: candidate},
        _agentic_candidates_lock=threading.RLock(),
        _agentic_pending_release_lock=threading.RLock(),
        _agentic_release_ownership={},
        _agentic_tp_pending_releases={},
        _agentic_slow_active_ids={},
    )
    manager._enqueue_agentic_release = (
        lambda value, offset, **kwargs: DecodeKVCacheOffloadManager._enqueue_agentic_release(
            manager, value, offset, **kwargs
        )
    )

    retired = DecodeKVCacheOffloadManager._retire_candidate_for_release(
        manager, snapshot_id, req, 0
    )

    assert retired is candidate
    assert manager.agentic_direct_candidates == {}
    assert manager._agentic_release_ownership == {snapshot_id: (req, 0)}
    assert manager._agentic_tp_pending_releases == {snapshot_id: (req, 0)}
    assert DecodeKVCacheOffloadManager.tp_pending_release_snapshot(manager) == snapshot_id


def test_fast_direct_recompute_tp_follower_cannot_terminalize():
    """Only TP rank zero may publish FAILED or a recompute route."""

    snapshot_id = "request:fast-recompute-follower:2"
    manifest = SimpleNamespace(
        request=SimpleNamespace(snapshot_id=snapshot_id),
        state=SnapshotState.DIRECT_READY,
    )
    calls = []
    manager = SimpleNamespace(
        tp_world_size=2,
        tp_rank=1,
        agentic_snapshot_store=SimpleNamespace(
            fail_direct_offer=lambda *_args, **_kwargs: calls.append("fail")
        ),
        _publish_agentic_route=lambda *_args, **_kwargs: calls.append("route"),
        _retire_candidate_for_release=lambda *_args, **_kwargs: calls.append(
            "release"
        ),
    )
    candidate = {"fallback_retry_at": 0.0}

    assert not DecodeKVCacheOffloadManager._try_fast_direct_failure_recompute(
        manager,
        candidate,
        manifest,
        SimpleNamespace(current=manifest.request),
        time.monotonic(),
    )
    assert calls == []
    assert "fast_direct_recompute_terminalized" not in candidate


def test_fast_direct_recompute_tp_rank0_routes_before_group_release():
    """TP0 retains every source shard until recompute routing is durable."""

    snapshot_id = "request:fast-recompute-rank0:2"
    request = SimpleNamespace(snapshot_id=snapshot_id)
    manifest = SimpleNamespace(
        request=request,
        state=SnapshotState.DIRECT_READY,
    )
    failed = SimpleNamespace(request=request, state=SnapshotState.FAILED)
    events = []
    route_results = iter((False, True))
    manager = SimpleNamespace(
        tp_world_size=2,
        tp_rank=0,
        agentic_snapshot_store=SimpleNamespace(
            fail_direct_offer=lambda *_args, **_kwargs: (
                events.append("terminalize") or failed
            )
        ),
        _publish_agentic_route=lambda *_args, **_kwargs: (
            events.append("route") or next(route_results)
        ),
        _agentic_release_early_claim=lambda *_args, **_kwargs: events.append(
            "claim_cleanup"
        ),
        _cleanup_agentic_direct_sender=lambda *_args, **_kwargs: events.append(
            "sender_cleanup"
        ),
        _retire_candidate_for_release=lambda *_args, **_kwargs: events.append(
            "group_release"
        ),
    )
    candidate = {
        "req": object(),
        "created_at": time.monotonic() - 2.0,
        "fallback_retry_at": 0.0,
    }
    metadata = SimpleNamespace(current=request)

    assert not DecodeKVCacheOffloadManager._try_fast_direct_failure_recompute(
        manager, candidate, manifest, metadata, time.monotonic()
    )
    assert events == ["terminalize", "route"]
    assert candidate["fast_direct_recompute_terminalized"] is True

    candidate["fallback_retry_at"] = 0.0
    assert DecodeKVCacheOffloadManager._finish_fast_direct_recompute(
        manager, candidate, metadata, time.monotonic()
    )
    assert events == [
        "terminalize",
        "route",
        "route",
        "claim_cleanup",
        "sender_cleanup",
        "group_release",
    ]


def test_fast_direct_recompute_retries_route_before_releasing_d_kv():
    """A transient Router-marker failure must retain the sole D snapshot."""

    snapshot_id = "request:fast-recompute-route-retry"
    direct = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.DIRECT_READY,
        token_count=1024,
    )
    failed = SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.FAILED,
        token_count=1024,
    )
    candidate = {
        "req": object(),
        "metadata": SimpleNamespace(current=SimpleNamespace(snapshot_id=snapshot_id)),
        "manifest": direct,
        "sender": SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
        "sent": False,
        "staging": False,
        "claimed_at": None,
        "created_at": time.monotonic() - 2.0,
        "fallback_retry_at": 0.0,
        "io_lock": threading.RLock(),
        "fast_arrival_seen": True,
        "fast_arrival_seen_at": time.monotonic() - 1.5,
    }
    route_results = iter((False, True))
    route_attempts = []
    releases = []
    cleanups = []
    manager = SimpleNamespace(
        tp_world_size=1,
        tp_rank=0,
        agentic_force_slow_path=False,
        agentic_fast_direct_failure_recompute=True,
        agentic_fast_threshold=1.0,
        agentic_direct_setup_timeout=1.0,
        agentic_early_claim_post_timeout=0.0,
        agentic_relay_worker=None,
        agentic_early_claim_store=object(),
        agentic_host_staging_client=object(),
        agentic_snapshot_store=SimpleNamespace(
            fail_direct_offer=lambda *_args, **_kwargs: failed
        ),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda sid, value: (
            sid == snapshot_id and value is candidate
        ),
        _agentic_try_early_claim=lambda *_args: "arrived",
        _agentic_direct_manifest=lambda *_args, **_kwargs: (
            failed
            if candidate.get("fast_direct_recompute_terminalized")
            else direct
        ),
        _agentic_try_tool_confirmation=lambda _candidate: True,
        _agentic_release_early_claim=lambda *_args: None,
        _agentic_direct_kv_usage=lambda: 0.5,
        _publish_agentic_route=lambda *_args, **_kwargs: (
            route_attempts.append(snapshot_id) or next(route_results)
        ),
        _start_agentic_host_staging=lambda *_args, **_kwargs: pytest.fail(
            "recompute route retry must not enter Host fallback"
        ),
        _cleanup_agentic_direct_sender=lambda value: cleanups.append(value),
        _retire_candidate_for_release=lambda sid, req, offset: releases.append(
            (sid, req, offset)
        ),
    )

    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert route_attempts == [snapshot_id]
    assert releases == []
    assert cleanups == []
    assert candidate["fast_direct_recompute_terminalized"] is True

    candidate["fallback_retry_at"] = 0.0
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(
        manager, progress_relay=False
    )
    assert route_attempts == [snapshot_id, snapshot_id]
    assert releases == [(snapshot_id, candidate["req"], 0)]
    assert cleanups == [candidate]


def test_early_direct_physical_cap_counts_inflight_and_tp_grants_once():
    """The event-driven admission path shares the configured physical cap."""

    inflight = SimpleNamespace(
        completed_at=None,
        transport_poll=KVPoll.WaitingForInput,
    )
    completed = SimpleNamespace(
        completed_at=time.monotonic(),
        transport_poll=KVPoll.Success,
    )
    scheduler = SimpleNamespace(
        agentic_early_direct_poll_lock=threading.RLock(),
        agentic_early_direct_receives={
            "physical": inflight,
            "granted-and-physical": inflight,
            "completed": completed,
        },
        agentic_tp_direct_admission_active={
            "granted": object(),
            "granted-and-physical": object(),
            "completed": object(),
        },
    )

    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 3


def test_early_direct_admission_does_not_exceed_physical_cap(monkeypatch):
    """A 32-arrival burst remains metadata-only while all eight lanes run."""

    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "8")
    arrived_at = time.time()
    queued = []
    for index in range(32):
        request = RequestGeneration(f"cap-burst-{index}", 1)
        manifest = SimpleNamespace(
            request=request,
            state=SnapshotState.DIRECT_READY,
            created_at=arrived_at,
            token_count=1024,
        )
        queued.append(
            (
                request,
                {"arrived_at": arrived_at, "prompt_token_count": 2048},
                manifest,
            )
        )
    inflight = {
        f"active-{index}": SimpleNamespace(
            completed_at=None,
            transport_poll=KVPoll.WaitingForInput,
        )
        for index in range(8)
    }
    broker = SimpleNamespace(
        owner_is_superseded=lambda *_args, **_kwargs: False,
        cancel_unstarted=lambda *_args, **_kwargs: None,
        request=lambda *_args, **_kwargs: pytest.fail(
            "a capped arrival must not reserve P HBM"
        ),
    )
    scheduler = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        agentic_early_claim_store=object(),
        agentic_tp_direct_admission_active={},
        agentic_early_direct_admission_queue=deque(queued),
        agentic_early_direct_admission_ids={
            request.snapshot_id for request, _, _ in queued
        },
        agentic_early_direct_receives=inflight,
        agentic_early_direct_terminal={},
        agentic_p_workset_broker=broker,
    )

    Scheduler._agentic_admit_queued_direct_receives(
        scheduler,
        SimpleNamespace(load=lambda *_args, **_kwargs: pytest.fail("cached")),
        1.0,
        threading.RLock(),
    )

    assert len(scheduler.agentic_early_direct_admission_queue) == 32
    assert len(scheduler.agentic_early_direct_admission_ids) == 32
    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 8


def test_tp_follower_waits_for_rank0_grant_before_leasing_workset(monkeypatch):
    """TP followers keep a 32-arrival burst metadata-only until rank0 grants."""

    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "8")
    arrived_at = time.time()
    queued = []
    manifests = {}
    for index in range(32):
        request = RequestGeneration(f"tp-follower-cap-{index}", 1)
        manifest = SimpleNamespace(
            request=request,
            state=SnapshotState.DIRECT_LOADING,
            created_at=arrived_at,
            token_count=1024,
        )
        manifests[request.snapshot_id] = manifest
        queued.append(
            (
                request,
                {"arrived_at": arrived_at, "prompt_token_count": 2048},
                manifest,
            )
        )

    receipts = {}
    leases = {}
    requests = []

    def request_lease(snapshot_id, *_args, **_kwargs):
        requests.append(snapshot_id)
        leases[snapshot_id] = object()

    broker = SimpleNamespace(
        owner_is_superseded=lambda *_args, **_kwargs: False,
        cancel_unstarted=lambda snapshot_id, **_kwargs: leases.pop(
            snapshot_id, None
        ),
        request=request_lease,
        get=lambda snapshot_id, **_kwargs: leases.get(snapshot_id),
        request_release=lambda snapshot_id, *_args, **_kwargs: leases.pop(
            snapshot_id, None
        ),
    )
    scheduler = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        agentic_early_claim_store=object(),
        agentic_tp_direct_admission_active={},
        agentic_early_direct_admission_queue=deque(queued),
        agentic_early_direct_admission_ids={
            request.snapshot_id for request, _, _ in queued
        },
        agentic_early_direct_receives={},
        agentic_early_direct_terminal={},
        agentic_tp_direct_local_failed=set(),
        agentic_p_workset_broker=broker,
        agentic_tp_direct_mailbox=SimpleNamespace(
            receipt=lambda snapshot_id: receipts.get(snapshot_id)
        ),
    )
    store = SimpleNamespace(
        load=lambda request, require_ready=False: manifests[request.snapshot_id]
    )

    Scheduler._agentic_admit_queued_direct_receives(
        scheduler, store, 1.0, threading.RLock()
    )
    assert requests == []
    assert leases == {}
    assert len(scheduler.agentic_early_direct_admission_queue) == 32

    first_eight = [item[0].snapshot_id for item in queued[:8]]
    receipts.update({snapshot_id: 1 for snapshot_id in first_eight})
    Scheduler._agentic_admit_queued_direct_receives(
        scheduler, store, 1.0, threading.RLock()
    )
    assert requests == first_eight
    assert set(leases) == set(first_eight)
    assert len(scheduler.agentic_tp_direct_admission_active) == 8
    assert len(scheduler.agentic_early_direct_admission_queue) == 24

    # One local DMA completes while its logical TP grant remains live until
    # group bind. The physical lane is immediately reusable by exactly one
    # newly granted generation.
    completed_id = first_eight[0]
    scheduler.agentic_early_direct_receives[completed_id] = SimpleNamespace(
        completed_at=time.monotonic(),
        transport_poll=KVPoll.Success,
    )
    ninth_id = queued[8][0].snapshot_id
    receipts[ninth_id] = 1
    Scheduler._agentic_admit_queued_direct_receives(
        scheduler, store, 1.0, threading.RLock()
    )
    assert requests[-1] == ninth_id
    assert len(requests) == 9
    assert len(scheduler.agentic_early_direct_admission_queue) == 23
    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 8


def test_tp_slow_offer_uses_rank0_manifest_token_identity():
    """Follower sampling metadata must not redefine a logical TP snapshot."""

    request = RequestGeneration("tp-slow-authoritative", 1)
    authoritative_tokens = [11, 12, 13]
    manifest = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=len(authoritative_tokens),
        byte_size=0,
        state=SnapshotState.SLOW_FALLBACK,
        token_digest=token_ids_digest(authoritative_tokens),
        tp_size=2,
    )
    captured = []
    client = SimpleNamespace(
        retain_logical_hashes=False,
        arena_domain=0,
        arena_numa_node=1,
        offer=lambda **kwargs: captured.append(kwargs),
    )
    manager = SimpleNamespace(
        tp_world_size=2,
        tp_rank=1,
        agentic_host_staging_client=client,
        agentic_direct_runtime=SimpleNamespace(
            manager=SimpleNamespace(kv_args=SimpleNamespace(kv_item_lens=[64]))
        ),
        _assign_slow_host_target=lambda _candidate: None,
    )
    candidate = {
        "metadata": SimpleNamespace(current=request),
        # A follower may not own the authoritative sampled token-id view, even
        # though its KV shard covers exactly the same logical positions.
        "tokens": [91, 92, 93],
        "source_token_indices": torch.tensor([3, 4, 5], dtype=torch.int64),
        "source_page_indices": [3],
        "selected_host_numa_nodes": [0, 1],
    }

    assert DecodeKVCacheOffloadManager._start_agentic_host_staging(
        manager, candidate, manifest
    )
    assert candidate["staging"] is True
    assert captured[0]["token_count"] == len(authoritative_tokens)
    assert captured[0]["token_digest"] == manifest.token_digest


@pytest.mark.parametrize("tp_world_size", [1, 2])
def test_slow_host_selection_reserves_complete_tp_snapshot_bytes(tp_world_size):
    """A Direct manifest's zero byte_size must not reach Host routing."""

    request = RequestGeneration("tp-slow-byte-accounting", 1)
    manifest = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=8,
        byte_size=0,
        state=SnapshotState.SLOW_FALLBACK,
        token_digest=token_ids_digest(list(range(8))),
        tp_size=tp_world_size,
    )
    selected = []
    offered = []

    def assign(candidate):
        selected.append(candidate["slow_host_reservation_bytes"])
        candidate["selected_host_domain"] = 1
        candidate["selected_host_numa_nodes"] = list(range(tp_world_size))

    manager = SimpleNamespace(
        tp_world_size=tp_world_size,
        tp_rank=0,
        agentic_host_staging_client=SimpleNamespace(
            retain_logical_hashes=False,
            arena_domain=0,
            arena_numa_node=0,
            offer=lambda **kwargs: offered.append(kwargs),
        ),
        agentic_direct_runtime=SimpleNamespace(
            manager=SimpleNamespace(
                kv_args=SimpleNamespace(kv_item_lens=[64, 128])
            )
        ),
        _assign_slow_host_target=assign,
    )
    candidate = {
        "metadata": SimpleNamespace(current=request),
        "tokens": list(range(8)),
        "source_token_indices": torch.arange(8, dtype=torch.int64),
        "source_page_indices": [3, 4, 5],
    }

    assert DecodeKVCacheOffloadManager._start_agentic_host_staging(
        manager, candidate, manifest
    )
    # Each rank writes 3 * (64 + 128) bytes; Host selection reserves both
    # shards because pressure is published per logical TP domain.
    assert selected == [576 * tp_world_size]
    assert offered[0]["byte_size"] == 576
    assert offered[0]["arena_domain"] == 1


def test_slow_host_selector_rejects_unknown_snapshot_size():
    manager = SimpleNamespace()
    with pytest.raises(ValueError, match="positive complete TP snapshot size"):
        DecodeKVCacheOffloadManager._select_slow_host_domain(manager)


def test_random_routing_ablation_selects_one_deterministic_slow_host(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_ABLATION_RANDOM_ROUTING", "1")
    monkeypatch.setenv("SGLANG_PD_ABLATION_RANDOM_SEED", "2026")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_PREFILL_DOMAIN_COUNT", "2")
    candidate = {
        "manifest": SimpleNamespace(snapshot_id="request-generation-7")
    }
    manager = SimpleNamespace(
        _prefill_domain_numa_nodes=lambda domain: [domain],
    )

    DecodeKVCacheOffloadManager._assign_slow_host_target(manager, candidate)
    first = candidate["selected_host_domain"]
    assert first in {0, 1}
    assert candidate["selected_host_numa_nodes"] == [first]

    # Repeated calls are idempotent and cannot split one snapshot across P's.
    DecodeKVCacheOffloadManager._assign_slow_host_target(manager, candidate)
    assert candidate["selected_host_domain"] == first


def test_nixl_sender_records_each_posted_handle_once_and_completes():
    room = 44
    kv_handle = object()
    aux_handle = object()
    transfer_calls = []

    class Agent:
        def transfer(self, handle):
            transfer_calls.append(handle)
            return "DONE"

        def check_xfer_state(self, _handle):
            return "DONE"

    transfer = SimpleNamespace(
        room=room,
        is_dummy=lambda: False,
        dst_kv_indices=[7],
        agent_name="decode-peer",
        dst_aux_index=0,
    )
    manager = NixlKVManager.__new__(NixlKVManager)
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.agent = Agent()
    manager.transfer_infos = {room: {"decode-peer": transfer}}
    manager.request_status = {room: KVPoll.WaitingForInput}
    manager.decode_kv_args_table = {
        "decode-peer": SimpleNamespace(
            decode_tp_size=1,
            dst_kv_ptrs=[1],
            dst_aux_ptrs=[2],
            gpu_id=0,
        )
    }
    manager.is_mla_backend = False
    manager.attn_tp_size = 1
    manager.kv_args = SimpleNamespace(pp_rank=0)
    manager.enable_all_cp_ranks_for_transfer = False
    manager.is_dummy_cp_rank = False

    def send_kvcache(*_args):
        recorder = _args[-1]
        return manager._post_transfer(kv_handle, recorder, "KV post failed")

    def send_aux(*_args):
        recorder = _args[-1]
        return manager._post_transfer(aux_handle, recorder, "aux post failed")

    manager.send_kvcache = send_kvcache
    manager.send_aux = send_aux

    sender = NixlKVSender.__new__(NixlKVSender)
    sender.kv_mgr = manager
    sender.bootstrap_room = room
    sender.curr_idx = 0
    sender.num_kv_indices = 1
    sender.aux_index = 0
    sender.xfer_handles = []
    sender.has_sent = False
    sender.chunk_id = 0
    sender.launch_failed = False
    sender.launch_exception = None
    sender.send([3])

    assert sender.xfer_handles == [kv_handle, aux_handle]
    assert transfer_calls == [kv_handle, aux_handle]
    assert sender.poll() == KVPoll.Success


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
        payloads = [
            {
                "arrived_at": arrived_at + index * 0.001,
                "prompt_token_count": 2048,
            }
            for index in range(2)
        ]
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
                    agentic_p_workset_broker=SimpleNamespace(
                        owner_is_superseded=lambda *_args, **_kwargs: False,
                        request=lambda *_args, **_kwargs: None,
                        get=lambda _snapshot_id, **_kwargs: object(),
                        request_release=lambda *_args: None,
                    ),
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
    released_worksets = []
    workset = SimpleNamespace(
        lease_id=1,
        parent_tokens=64,
        allocated_tokens=128,
        parent_indices=torch.arange(64),
        parent_page_indices=[0],
        state="active",
    )
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
            agentic_p_workset_broker=SimpleNamespace(
                begin_io_attempt=lambda *_args: True,
                mark_io_inflight=lambda *_args: None,
                mark_io_quiesced=lambda *_args: True,
                cancel_io_attempt=lambda *_args: True,
                request_release=lambda snapshot_id, *_args: released_worksets.append(
                    snapshot_id
                )
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
            workset_lease=workset,
        )
        assert request.snapshot_id not in scheduler.agentic_early_direct_receives
        assert request.snapshot_id not in scheduler.agentic_tp_direct_local_failed

    assert released_worksets == [request.snapshot_id, request.snapshot_id]
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
        lease = object()
        owner = SimpleNamespace(
            tp_rank=0,
            agentic_tp_direct_mailbox=mailboxes[0],
            agentic_early_direct_poll_lock=nullcontext(),
            agentic_tp_direct_admission_active={
                request.snapshot_id: (request, arrived_at, None, 1024, lease)
            },
            agentic_early_direct_receives={},
            agentic_tp_direct_local_failed=set(),
                agentic_tp_direct_local_admitted=set(),
                agentic_p_workset_broker=SimpleNamespace(
                    request_release=lambda *_args, **_kwargs: None,
                    cancel_unstarted=lambda *_args, **_kwargs: None,
                ),
        )
        store = SimpleNamespace(
            load=lambda *_args, **_kwargs: manifest,
            release_direct_claim=fail_cleanup,
        )

        Scheduler._agentic_commit_tp_direct_groups(owner, store)

        assert mailboxes[1].receipt(request.snapshot_id) == -1


def test_tp_direct_background_start_timeout_requests_ordered_group_abort(monkeypatch):
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
        lease = object()
        owner = SimpleNamespace(
            tp_rank=0,
            agentic_tp_direct_mailbox=mailbox,
            agentic_early_direct_poll_lock=nullcontext(),
            agentic_tp_direct_admission_active={
                request.snapshot_id: (request, arrived_at, None, 1024, lease)
            },
            agentic_early_direct_receives={},
            agentic_tp_direct_local_failed=set(),
            agentic_tp_direct_local_rolled_back=set(),
            agentic_tp_direct_local_admitted=set(),
            agentic_p_workset_broker=SimpleNamespace(
                request_release=lambda *_args, **_kwargs: None,
                cancel_unstarted=lambda *_args, **_kwargs: None,
            ),
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

        assert released == []
        assert mailbox.receipt(request.snapshot_id) == -1
        assert request.snapshot_id in owner.agentic_tp_direct_local_failed


def test_tp_direct_bind_failure_returns_received_group_to_d_slow(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "p0")
    request = RequestGeneration("tp-bind-retry", 1)
    claim_id = f"direct-early-tp:p0:{request.snapshot_id}"
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        store = MooncakeSnapshotStore(AgenticNodeLocalRawStore(directory))
        offer = SnapshotManifest(
            request=request,
            page_keys=(),
            token_count=128,
            byte_size=0,
            state=SnapshotState.DIRECT_READY,
            token_digest="abc",
            direct_bootstrap_addr="127.0.0.1:45501",
            direct_room=9,
            tp_size=2,
        )
        store.publish_direct_offer(offer)
        claimed = store.claim_direct(request, claim_id)
        store.complete_direct_rank(claimed, claim_id, tp_rank=0, tp_size=2)
        received = store.complete_direct_rank(
            claimed, claim_id, tp_rank=1, tp_size=2
        )
        assert received.state is SnapshotState.P_RECEIVED

        mailbox = TPGroupMailbox(
            "tp-bind-retry", tp_rank=0, tp_size=2, directory=directory
        )
        owner = SimpleNamespace(
            tp_rank=0,
            agentic_tp_direct_mailbox=mailbox,
            agentic_early_direct_poll_lock=nullcontext(),
            agentic_tp_direct_admission_active={
                request.snapshot_id: (request, time.time(), None, 128, None)
            },
            agentic_early_direct_receives={},
            agentic_tp_direct_local_failed=set(),
            agentic_tp_direct_local_rolled_back=set(),
            agentic_p_workset_broker=SimpleNamespace(
                cancel_unstarted=lambda *_args, **_kwargs: None,
                request_release=lambda *_args, **_kwargs: None,
            ),
        )

        assert Scheduler._agentic_abort_tp_direct_grant(
            owner, request, store, reason="radix_insert_failed"
        )
        assert store.load(request, require_ready=False).state is SnapshotState.P_RECEIVED
        assert mailbox.receipt(request.snapshot_id) == -1
        assert Scheduler._agentic_abort_tp_direct_grant(
            owner,
            request,
            store,
            reason="all_ranks_rolled_back",
            rolled_back=True,
        )

        returned = store.load(request, require_ready=False)
        assert returned.state is SnapshotState.DIRECT_READY
        assert returned.claim_id is None
        assert mailbox.receipt(request.snapshot_id) == -2


def test_permanent_direct_layout_mismatch_publishes_terminal_failure():
    request = RequestGeneration("layout-mismatch", 1)
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        store = MooncakeSnapshotStore(AgenticNodeLocalRawStore(directory))
        offer = SnapshotManifest(
            request=request,
            page_keys=(),
            token_count=128,
            byte_size=0,
            state=SnapshotState.DIRECT_READY,
            token_digest="abc",
            direct_bootstrap_addr="127.0.0.1:45501",
            direct_room=9,
            tp_size=1,
            kv_layout_hash="source-layout",
        )
        store.publish_direct_offer(offer)
        owner = SimpleNamespace(
            tp_rank=0,
            tp_size=1,
            tree_cache=SimpleNamespace(is_eagle=False),
            agentic_direct_runtime=SimpleNamespace(layout_hash="other-layout"),
            agentic_p_workset_broker=SimpleNamespace(
                request_release=lambda *_args, **_kwargs: None
            ),
        )

        assert not Scheduler._agentic_start_early_direct_receive(
            owner,
            request,
            offer,
            store,
            arrived_at=time.time(),
            workset_lease=object(),
        )

        failed = store.load(request, require_ready=False)
        assert failed.state is SnapshotState.FAILED
        assert failed.failure_reason == "permanent_direct_layout_mismatch"


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
                    request.snapshot_id: (request, time.time(), None, 1024, None)
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


def test_tp1_direct_finalize_failure_retries_without_unpin_or_double_free():
    request = RequestGeneration("finalize-retry", 1)
    req = SimpleNamespace(
        rid="finalize-child", _agentic_direct_parent_pin_node=object()
    )
    lease = object()
    entry = SimpleNamespace(
        request=request,
        manifest=SimpleNamespace(token_count=8),
        prepared_req=req,
        radix_prepared=True,
        existing_tokens=4,
        device_indices=torch.arange(8),
        workset_lease=lease,
        claim_id="claim",
        arrived_at=time.time(),
    )
    attempts = []
    freed = []

    def handoff(_snapshot_id, _req, _lease):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("injected transient handoff failure")

    received = SimpleNamespace(
        state=SnapshotState.P_RECEIVED, claim_id="claim"
    )
    consumed = SimpleNamespace(state=SnapshotState.CONSUMED)
    store = SimpleNamespace(
        load=lambda *_args, **_kwargs: received,
        commit_direct_bound=lambda *_args, **_kwargs: consumed,
    )
    owner = SimpleNamespace(
        agentic_p_workset_broker=SimpleNamespace(handoff_to_req=handoff),
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda indices: freed.append(indices.clone())
        ),
        agentic_early_direct_poll_lock=nullcontext(),
        agentic_early_direct_receives={request.snapshot_id: entry},
        agentic_early_direct_terminal={},
        tree_cache=SimpleNamespace(supports_mamba=lambda: False),
        _agentic_snapshot_store=lambda: store,
    )
    owner._agentic_finalize_early_direct_bind = lambda *args, **kwargs: (
        Scheduler._agentic_finalize_early_direct_bind(owner, *args, **kwargs)
    )
    owner._agentic_admit_early_direct_bind = lambda *args, **kwargs: (
        Scheduler._agentic_admit_early_direct_bind(owner, *args, **kwargs)
    )

    assert Scheduler._agentic_try_finalize_early_direct_bind(
        owner,
        req,
        request,
        entry,
        existing_tokens=entry.existing_tokens,
        tp_size=1,
        marker_store=None,
        admit=True,
    )
    assert entry.prepared_req is req
    assert entry.radix_prepared is True
    assert entry.workset_lease is lease
    assert hasattr(req, "_agentic_direct_parent_pin_node")
    assert len(freed) == 1

    assert not Scheduler._agentic_try_finalize_early_direct_bind(
        owner,
        req,
        request,
        entry,
        existing_tokens=entry.existing_tokens,
        tp_size=1,
        marker_store=None,
        admit=True,
    )
    assert len(freed) == 1
    assert len(attempts) == 2


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
                    request.snapshot_id: (request, time.time(), None, 1024, None)
                },
            agentic_tp_direct_group_status={},
            agentic_early_direct_receives={request.snapshot_id: entry},
            agentic_tp_direct_local_failed=set(),
            agentic_tp_direct_local_admitted=set(),
                agentic_p_workset_broker=SimpleNamespace(
                    request_release=lambda *_args, **_kwargs: None,
                        install_tp_plan=lambda *_args, **_kwargs: None,
                ),
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
            "workset_plan_epoch": 1,
            "workset_allocation_plan": [],
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
            [(
                request,
                {"arrived_at": arrived_at, "prompt_token_count": 2048},
                manifest,
            )]
        ),
        agentic_early_direct_admission_ids={request.snapshot_id},
        agentic_early_direct_receives={},
        agentic_early_direct_terminal={},
            agentic_p_workset_broker=SimpleNamespace(
                owner_is_superseded=lambda *_args, **_kwargs: False,
                request=lambda *_args, **_kwargs: None,
            get=lambda _snapshot_id, **_kwargs: object(),
            request_release=lambda *_args: None,
        ),
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


def test_direct_arrival_waits_until_complete_workset_is_granted(monkeypatch):

    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    request = RequestGeneration("p-hbm-full", 1)
    arrived_at = time.time()
    manifest = SimpleNamespace(
        request=request,
        state=SnapshotState.DIRECT_READY,
        created_at=arrived_at,
        token_count=1024,
    )
    started = []
    scheduler = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        agentic_early_claim_store=object(),
        agentic_tp_direct_admission_active={},
        agentic_early_direct_admission_queue=deque(
            [(
                request,
                {"arrived_at": arrived_at, "prompt_token_count": 2048},
                manifest,
            )]
        ),
        agentic_early_direct_admission_ids={request.snapshot_id},
        agentic_early_direct_receives={},
        agentic_early_direct_terminal={},
            agentic_p_workset_broker=SimpleNamespace(
                owner_is_superseded=lambda *_args, **_kwargs: False,
                request=lambda *_args, **_kwargs: None,
            get=lambda _snapshot_id, **_kwargs: None,
            request_release=lambda *_args: None,
        ),
        server_args=SimpleNamespace(page_size=64),
        _agentic_start_early_direct_receive=lambda selected, *_args, **_kwargs: (
            started.append(selected.snapshot_id) or True
        ),
    )
    store = SimpleNamespace(load=lambda *_args, **_kwargs: manifest)

    Scheduler._agentic_admit_queued_direct_receives(
        scheduler, store, 2.0, nullcontext()
    )

    assert started == []
    assert list(scheduler.agentic_early_direct_admission_ids) == [
        request.snapshot_id
    ]
    assert len(scheduler.agentic_early_direct_admission_queue) == 1


def test_disabled_compute_ahead_does_not_double_reserve_direct_headroom(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "disabled")
    scheduler = SimpleNamespace(
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


def test_disagg_prefill_services_workset_broker_at_scheduler_boundary():
    events = []
    scheduler = SimpleNamespace(
        running_batch=SimpleNamespace(batch_is_full=True),
        waiting_queue=[],
        _agentic_service_p_workset_leases=lambda: events.append("workset"),
        process_prefill_chunk=lambda: events.append("chunk"),
        _should_throttle_p_ready_compute_ahead=lambda: False,
        get_new_batch_prefill=lambda: None,
        maybe_prepare_mlp_sync_batch=lambda batch: batch,
    )

    batch = SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run(
        scheduler
    )

    assert batch is None
    assert events == ["workset", "chunk"]
    assert scheduler.running_batch.batch_is_full is False


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
        "decode_transfer_cancel_keys": [],
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
        _agentic_release_ownership={"request:3": (req, 0)},
    )
    reserved = DecodeKVCacheOffloadManager.agentic_pending_release_token_count.fget(
        manager
    )
    assert reserved == 4160 - 2048
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


def test_tp_decode_release_never_waits_for_background_io():
    """The Decode scheduler defers, rather than blocking on an I/O lane."""

    snapshot_id = "request:5"
    req = SimpleNamespace(rid="request", req_pool_idx=9)
    io_lock = threading.RLock()
    entered = threading.Event()
    leave = threading.Event()

    def hold_io_lane():
        with io_lock:
            entered.set()
            leave.wait(timeout=5)

    holder = threading.Thread(target=hold_io_lane)
    holder.start()
    assert entered.wait(timeout=2)

    released = []
    manager = SimpleNamespace(
        _agentic_tp_pending_releases={snapshot_id: (req, 0)},
        _agentic_pending_release_lock=threading.RLock(),
        agentic_direct_candidates={
            snapshot_id: {"req": req, "io_lock": io_lock}
        },
        _agentic_candidates_lock=threading.RLock(),
        _release_finished_req=lambda value, offset: released.append(
            (value, offset)
        ),
        _cleanup_agentic_direct_sender=lambda _candidate: None,
        _agentic_release_early_claim=lambda _candidate, _reason: None,
    )

    started = time.perf_counter()
    assert not DecodeKVCacheOffloadManager.commit_tp_release(
        manager, snapshot_id
    )
    assert time.perf_counter() - started < 0.1
    assert manager._agentic_tp_pending_releases
    assert manager.agentic_direct_candidates
    assert released == []

    leave.set()
    holder.join(timeout=2)
    assert DecodeKVCacheOffloadManager.commit_tp_release(manager, snapshot_id)
    assert released == [(req, 0)]
    assert manager._agentic_tp_pending_releases == {}
    assert manager.agentic_direct_candidates == {}


def test_tp_deferred_releases_are_ordered_and_never_overwrite():
    """Two busy follower shards retain both one-shot rank-0 releases."""

    snapshots = ("request:6", "request:7")
    locks = {snapshot_id: threading.RLock() for snapshot_id in snapshots}
    leave = threading.Event()
    entered = {snapshot_id: threading.Event() for snapshot_id in snapshots}

    def hold(snapshot_id):
        with locks[snapshot_id]:
            entered[snapshot_id].set()
            leave.wait(timeout=5)

    holders = [
        threading.Thread(target=hold, args=(snapshot_id,))
        for snapshot_id in snapshots
    ]
    for holder in holders:
        holder.start()
    assert all(event.wait(timeout=2) for event in entered.values())

    reqs = {
        snapshot_id: SimpleNamespace(rid=snapshot_id, req_pool_idx=index + 1)
        for index, snapshot_id in enumerate(snapshots)
    }
    released = []
    manager = SimpleNamespace(
        _agentic_tp_pending_releases={
            snapshot_id: (reqs[snapshot_id], 0) for snapshot_id in snapshots
        },
        _agentic_pending_release_lock=threading.RLock(),
        _agentic_tp_deferred_releases={},
        agentic_direct_candidates={
            snapshot_id: {
                "req": reqs[snapshot_id],
                "io_lock": locks[snapshot_id],
            }
            for snapshot_id in snapshots
        },
        _agentic_candidates_lock=threading.RLock(),
        _release_finished_req=lambda req, _offset: released.append(req.rid),
        _cleanup_agentic_direct_sender=lambda _candidate: None,
        _agentic_release_early_claim=lambda _candidate, _reason: None,
    )

    for snapshot_id in snapshots:
        assert not DecodeKVCacheOffloadManager.commit_tp_release(
            manager, snapshot_id
        )
    assert tuple(manager._agentic_tp_deferred_releases) == snapshots

    leave.set()
    for holder in holders:
        holder.join(timeout=2)
    for snapshot_id in tuple(manager._agentic_tp_deferred_releases):
        assert DecodeKVCacheOffloadManager.commit_tp_release(manager, snapshot_id)
    assert released == list(snapshots)
    assert manager._agentic_tp_deferred_releases == {}


def test_tp_follower_skips_retired_candidate_snapshot():
    """A worker's stale map snapshot cannot touch transport after release."""

    snapshot_id = "request:8"

    class Sender:
        polls = 0

        def poll(self):
            self.polls += 1
            return KVPoll.WaitingForInput

    sender = Sender()
    candidate = {
        "tp_command": "direct",
        "io_lock": threading.RLock(),
        "sender": sender,
        "sent": False,
        "source_page_indices": [1, 2],
        "retired": True,
    }
    manager = SimpleNamespace(
        agentic_relay_worker=None,
        agentic_direct_candidates={},
        _agentic_candidates_lock=threading.RLock(),
        _agentic_candidate_items=lambda: ((snapshot_id, candidate),),
        _agentic_candidate_is_live_locked=lambda sid, value: (
            DecodeKVCacheOffloadManager._agentic_candidate_is_live_locked(
                manager, sid, value
            )
        ),
    )

    DecodeKVCacheOffloadManager._check_agentic_tp_follower_progress(
        manager, progress_relay=False, progress_class="direct"
    )
    assert sender.polls == 0


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


def test_custom_storage_only_rejects_ordinary_decode_offload():
    """The native Decode HiCache path remains baseline-only."""

    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.agentic_hostless = False
    manager.agentic_enabled = True
    manager.agentic_custom_storage_only = True
    manager.cache_controller = object()
    manager.decode_host_mem_pool = object()
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params={}),
        req_pool_idx=1,
        output_ids=[1],
    )

    assert not DecodeKVCacheOffloadManager.offload_kv_cache(manager, req)
