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
from contextlib import nullcontext
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
    AgenticPWorksetLeaseBroker,
    Scheduler,
)


def _ledger():
    fd, path = tempfile.mkstemp(prefix="sglang-agentic-tp-", dir="/dev/shm")
    os.close(fd)
    os.unlink(path)
    return SharedHostStagingLedger(path), path


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
    record = {}
    load = {
        "record": record,
        "request_generation": request,
        "workset_lease": lease,
        "io_attempt": attempt,
        "io_inflight": True,
        "io_quiesced": False,
        "launch_fence": H2DLaunchFence(
            event=event, submitted=True, armed=True
        ),
    }
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.workset_broker = broker
    manager.tree_cache = SimpleNamespace()
    manager.loads = {rid: load}
    manager.host_ready = {}
    manager._control_wakeup = SimpleNamespace(set=lambda: None)
    manager._h2d_poisoned = False
    released_host = []
    manager._release_record = lambda current: released_host.append(current)

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


def test_d2p_slow_writer_uses_independent_copy_lanes():
    class Snapshot:
        def __init__(self):
            self.started = []

        def start_backup_range_from_device(self, indices, **kwargs):
            self.started.append((indices.clone(), kwargs))
            return object(), object()

    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client._d2h_chunk_tokens = 4
    client._d2h_lanes = [
        {
            "stream": object(),
            "staging": object(),
            "host_bounce": object(),
            "snapshot_id": None,
        }
        for _ in range(2)
    ]

    def candidate(snapshot_id):
        return {
            "manifest": SimpleNamespace(snapshot_id=snapshot_id),
            "arena_write": {
                "snapshot": Snapshot(),
                "event": None,
                "copy_refs": None,
                "offset": 0,
                "chunk_end": 0,
                "gpu_elapsed_ms": 0.0,
            },
        }

    first = candidate("slow:1")
    second = candidate("slow:2")
    third = candidate("slow:3")
    indices = torch.arange(8)

    assert client._start_write_chunk(first, indices)
    assert client._start_write_chunk(second, indices)
    assert not client._start_write_chunk(third, indices)
    assert first["arena_write"]["lane_id"] != second["arena_write"]["lane_id"]
    assert {lane["snapshot_id"] for lane in client._d2h_lanes} == {
        "slow:1",
        "slow:2",
    }


def test_d2p_active_host_write_can_progress_without_ledger_poll():
    candidate = {
        "manifest": SimpleNamespace(snapshot_id="slow:local"),
        "arena_write": {
            "snapshot": object(),
            "event": None,
            "copy_refs": None,
            "offset": 0,
            "chunk_end": 0,
            "gpu_elapsed_ms": 0.0,
        },
    }
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.ledger = SimpleNamespace(
        get=lambda _snapshot_id: pytest.fail("active D2H must not poll the ledger")
    )
    client._start_write_chunk = lambda _candidate, _indices: False

    assert client.has_active_local_write(candidate)
    assert client.progress(candidate, [], local_write_only=True) == "waiting"


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
            "event": Event(),
            "copy_refs": object(),
            "offset": 0,
            "chunk_end": 1,
            "gpu_elapsed_ms": 0.0,
            "lane_id": 0,
        },
    }
    drained = []
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.tp_rank = 0
    client.tp_size = 1
    client._d2h_lanes = [
        {"snapshot_id": snapshot_id, "host_bounce": object()}
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

    assert client.progress(candidate, torch.arange(1), local_write_only=True) == expected
    assert snapshot.commits == [{"destination_start": 0, "token_count": 1}]
    assert snapshot.closed
    assert client._d2h_lanes[0]["snapshot_id"] is None
    assert bool(drained) is (not commit_succeeds)


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
    ledger = SimpleNamespace(get=lambda _snapshot_id: None)
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
        _agentic_host_workset_lease=object(),
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
    manager.owner = "p:test"
    manager.workset_broker = Broker()
    manager.ledger = SimpleNamespace(
        get=lambda snapshot_id: manager._ledger_entries_cache.get(snapshot_id)
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
            "state": HostStageState.HOST_READY.value
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


def test_tp_slow_lane_retries_transient_prepare_without_start_or_leak():
    """A TP ledger hiccup keeps one exact lease and retries before H2D."""

    request = RequestGeneration("slow-prepare-retry", 3)
    req = SimpleNamespace(rid="child-retry", origin_input_ids=[11, 22])
    lease = SimpleNamespace(lease_id="lease-retry", parent_indices=[7])
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
            return {"state": HostStageState.HOST_READY.value}

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
    lease = object()
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
    )
    manager._complete_shared_host_manifest = lambda _request: True
    manager._release_record = lambda selected: released_host.append(selected)

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


def test_tp2_slow_handoff_failure_retains_commit_context_for_retry():
    request = RequestGeneration("slow-handoff-tp2", 1)
    req = SimpleNamespace(
        rid="child-tp2",
        origin_input_ids=[11, 22],
        _agentic_host_rank_loaded=True,
        _agentic_host_rank_token_count=1,
        _agentic_host_workset_lease=object(),
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
        get=lambda _snapshot_id: {"state": HostStageState.CONSUMED.value}
    )
    manager._release_record = lambda selected: released_host.append(selected)

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
        "workset_lease": object(),
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
            get=lambda _snapshot_id: {"state": HostStageState.H2D_LOADING.value}
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


def _tp1_edge_scheduler(req, load):
    scheduler = object.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_kv_waiting_by_rid = {req.rid: (req, time.monotonic())}
    scheduler.agentic_kv_waiting_by_parent = {}
    scheduler.agentic_kv_progress_queues = {
        "fast": deque(),
        "slow": deque([req.rid]),
        "new": deque(),
    }
    scheduler.agentic_kv_progress_enqueued = {req.rid}
    scheduler.agentic_kv_retry_heap = []
    scheduler.agentic_kv_retry_deadlines = {}
    scheduler.agentic_kv_retry_sequence = 0
    scheduler.agentic_kv_waiting_queue = [(req, time.monotonic())]
    scheduler.agentic_kv_waiting_tombstones = 0
    scheduler.agentic_early_direct_completion_queue = deque()
    scheduler.agentic_early_direct_poll_lock = nullcontext()
    scheduler.agentic_host_staging_manager = SimpleNamespace(
        loads={req.rid: load}, drain_scheduler_events=lambda: ()
    )
    scheduler.agentic_p_workset_broker = SimpleNamespace(
        drain_grant_events=lambda: ()
    )
    scheduler._agentic_should_defer = lambda *_args, **_kwargs: True
    return scheduler


@pytest.mark.parametrize(
    ("load", "expects_retry"),
    [
        ({"io_complete": False, "ledger_prepare_pending": False}, False),
        ({"io_complete": False, "ledger_prepare_pending": True}, True),
        ({"io_complete": True, "ledger_prepare_pending": False}, True),
    ],
)
def test_tp1_slow_edge_retries_only_scheduler_owned_boundaries(
    monkeypatch, load, expects_retry
):
    """One H2D edge remains live through transient bind/ledger boundaries."""

    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SLOW_AGING_SECONDS", "0")
    req = SimpleNamespace(rid="slow-child", _agentic_kv_queue_class="slow")
    scheduler = _tp1_edge_scheduler(req, load)

    Scheduler._drain_agentic_kv_waiting_queue_tp1(scheduler)

    assert (req.rid in scheduler.agentic_kv_retry_deadlines) is expects_retry


def test_tp1_edge_queue_ages_slow_recovery_ahead_of_continuous_direct(
    monkeypatch,
):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "2")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SLOW_AGING_SECONDS", "0")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_NEW_AGING_SECONDS", "1000")
    fast = [
        SimpleNamespace(rid=f"fast-{index}", _agentic_kv_queue_class="fast")
        for index in range(3)
    ]
    slow = SimpleNamespace(rid="slow", _agentic_kv_queue_class="slow")
    requests = fast + [slow]
    scheduler = object.__new__(Scheduler)
    scheduler.tp_size = 1
    scheduler.agentic_kv_waiting_by_rid = {
        req.rid: (req, time.monotonic() - 10) for req in requests
    }
    scheduler.agentic_kv_waiting_by_parent = {}
    scheduler.agentic_kv_progress_queues = {
        "fast": deque(req.rid for req in fast),
        "slow": deque([slow.rid]),
        "new": deque(),
    }
    scheduler.agentic_kv_progress_enqueued = {req.rid for req in requests}
    scheduler.agentic_kv_retry_heap = []
    scheduler.agentic_kv_retry_deadlines = {}
    scheduler.agentic_kv_retry_sequence = 0
    scheduler.agentic_kv_waiting_queue = [
        scheduler.agentic_kv_waiting_by_rid[req.rid] for req in requests
    ]
    scheduler.agentic_kv_waiting_tombstones = 0
    scheduler.agentic_early_direct_completion_queue = deque()
    scheduler.agentic_early_direct_poll_lock = nullcontext()
    scheduler.agentic_host_staging_manager = None
    scheduler.agentic_p_workset_broker = SimpleNamespace(
        drain_grant_events=lambda: ()
    )
    visited = []
    scheduler._agentic_should_defer = (
        lambda req, *_args, **_kwargs: visited.append(req.rid) or True
    )

    Scheduler._drain_agentic_kv_waiting_queue_tp1(scheduler)

    assert visited == [slow.rid, fast[0].rid]


def test_prefill_priority_puts_owned_worksets_before_unrunnable_fast_fallbacks():
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


def test_tp_direct_control_round_trip_preserves_local_workset_lease():
    request = RequestGeneration("lease-round-trip", 1)
    lease = object()
    snapshot_id = request.snapshot_id
    control = {
        Scheduler._AGENTIC_TP_CONTROL_KEY: True,
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
            get=lambda *_args, **_kwargs: None
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
        "metadata": SimpleNamespace(current=SimpleNamespace()),
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
        "metadata": SimpleNamespace(current=SimpleNamespace()),
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
        agentic_early_claim_post_timeout=2.0,
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
                    request_release=lambda *_args, **_kwargs: None
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
