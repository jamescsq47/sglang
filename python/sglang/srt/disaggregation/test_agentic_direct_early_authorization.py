"""Early logical permission overlaps native allocation; it never grants pages."""

from collections import deque
from dataclasses import replace
from types import SimpleNamespace as NS
import threading
import time

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, SnapshotManifest, SnapshotState
from sglang.srt.disaggregation.agentic_tp_events import EventKey
from sglang.srt.disaggregation.agentic_tp_socket_mailbox import SocketTPGroupMailbox
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_kvcache_offload_manager import DecodeKVCacheOffloadManager as D
from sglang.srt.disaggregation.test_agentic_direct_descriptors import Allocator
from sglang.srt.disaggregation.test_agentic_direct_retirement import scheduler
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker


class Client:
    def __init__(self, rank, size, shared):
        self.rank, self.size, self.shared = rank, size, shared

    def publish_command(self, namespace, key, command, **_kw):
        self.shared["prepares"][key] = command

    def publish_receipt(self, namespace, key, status):
        self.shared["publications"].append((key, status))
        self.shared["receipts"][key] = status
        if self.shared.get("publish_error"):
            raise OSError("receipt submitted, ACK unknown")

    def receipt(self, namespace, key):
        return self.shared["receipts"].get(key)

    def report(self, namespace, key, status):
        self.shared.setdefault("reports", {})[(namespace, key, self.rank)] = status


@pytest.fixture(autouse=True)
def socket(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "1")
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")


def group(size):
    request = RequestGeneration("early-authorization", 1)
    manifest = SnapshotManifest(request, (), 8, 4, SnapshotState.DIRECT_READY,
                                token_digest="test-digest", direct_bootstrap_addr="test:1",
                                direct_room=12, tp_size=size)
    store = NS(current=manifest)
    store.load = lambda *_a, **_k: store.current
    shared = dict(prepares={}, publications=[], receipts={})
    owners, allocators, starts = [], [], []
    for rank in range(size):
        broker = AgenticPWorksetLeaseBroker(4)
        owner = scheduler(request, broker, None, size)
        owner.tp_rank = rank
        owner.agentic_tp_direct_admission_active = {}
        owner.agentic_early_claim_store = object()
        owner.agentic_tp_direct_mailbox = SocketTPGroupMailbox(
            "d2p-direct", tp_rank=rank, tp_size=size,
            client=Client(rank, size, shared),
        )
        owner.agentic_early_direct_admission_queue = deque()
        owner.agentic_early_direct_admission_ids = set()
        owner._agentic_snapshot_store = lambda: store
        owner.agentic_tp_p2d_sender_mailbox = None
        owner.agentic_tp_p2d_receiver_mailbox = None
        owner.agentic_host_staging_manager = None
        owner.agentic_tp_host_active_requests = {}
        owner.agentic_tp_host_active_since_by_snapshot = {}
        owner.agentic_tp_host_group_statuses = {}
        owner.agentic_kv_waiting_queue = []
        def start(req, manifest, _store, *, workset_lease, _rank=rank, _broker=broker, **_kw):
            assert _broker.get(req.snapshot_id) is workset_lease
            assert _broker.begin_io_attempt(req.snapshot_id, workset_lease, "physical")
            starts.append((_rank, workset_lease))
            return False
        owner._agentic_start_early_direct_receive = start
        owners.append(owner)
        allocators.append(Allocator(64))
    payload = {"arrived_at": time.time(), "prompt_token_count": 12}
    owners[0].agentic_early_direct_admission_queue.append((request, payload, manifest))
    owners[0].agentic_early_direct_admission_ids.add(request.snapshot_id)
    return NS(request=request, manifest=manifest, store=store, shared=shared,
              owners=owners, allocators=allocators, starts=starts, payload=payload)


def admit(g):
    root = g.owners[0]
    root._agentic_admit_queued_direct_receives(g.store, 1, root.agentic_early_direct_poll_lock)


@pytest.mark.parametrize("size", [2, 8])
def test_receipt_precedes_allocation_and_native_active_none_carries_exact_attempt(size):
    g = group(size)
    root, sid = g.owners[0], g.request.snapshot_id
    admit(g)
    identity = EventKey(sid, "direct-room:12")
    assert g.shared["publications"] == [(identity, 1)]
    assert identity in g.shared["prepares"]
    assert root.agentic_tp_direct_admission_active[sid][4] is None
    assert root._agentic_early_direct_slots_used() == 1  # active + intent count once
    assert not root.agentic_early_direct_admission_queue
    root._agentic_progress_tp_direct_grants(g.store)
    assert not g.starts and all(a.available_size() == 64 for a in g.allocators)
    command = root._agentic_tp_prepare_admission_control()
    assert command["direct_commands"][0]["control_attempt"] == "direct-room:12"
    for owner in g.owners:
        owner._agentic_tp_consume_admission_control([command])
        assert owner.agentic_tp_direct_admission_active[sid][4] is None
    # A follower may finish its native allocation before rank0; permission is
    # logical and neither rank can claim before its own physical commit.
    follower = g.owners[-1]
    follower.agentic_p_workset_broker.service(g.allocators[-1])
    follower._agentic_progress_tp_direct_grants(g.store)
    assert [rank for rank, _ in g.starts] == [size - 1]
    root._agentic_progress_tp_direct_grants(g.store)
    assert len(g.starts) == 1
    root.agentic_p_workset_broker.service(g.allocators[0])
    root._agentic_progress_tp_direct_grants(g.store)
    assert [rank for rank, _ in g.starts] == [size - 1, 0]
    assert len(g.shared["publications"]) == 1


@pytest.mark.parametrize("size", [2, 8])
def test_pregrant_cancel_and_late_native_plan_never_claim(size):
    g = group(size)
    root, sid = g.owners[0], g.request.snapshot_id
    admit(g)
    command = root._agentic_tp_prepare_admission_control()
    g.store.current = replace(g.manifest, state=SnapshotState.SLOW_FALLBACK)
    root._agentic_progress_tp_direct_grants(g.store)
    assert sid in root.agentic_tp_direct_local_failed
    assert not g.starts
    # Frozen allocation still installs identically; cancellation cannot free
    # it early or authorize a late claim when it finally materializes.
    for owner, allocator in zip(g.owners, g.allocators):
        owner._agentic_tp_consume_admission_control([command])
        owner.agentic_p_workset_broker.service(allocator)
        owner._agentic_progress_tp_direct_grants(g.store)
        assert sid in owner.agentic_tp_direct_local_failed
        assert allocator.available_size() == 52
        assert not owner.agentic_p_workset_broker.begin_io_attempt(
            sid, owner.agentic_p_workset_broker.get(sid), "late")
    assert not g.starts


@pytest.mark.parametrize("publish_error", [False, True])
def test_negative_or_unknown_permission_retains_owner_and_never_republishes_one(publish_error):
    g = group(2)
    root, sid = g.owners[0], g.request.snapshot_id
    key = EventKey(sid, "direct-room:12")
    if publish_error:
        g.shared["publish_error"] = True
    else:
        g.shared["receipts"][key] = -1
    admit(g)
    assert sid in root.agentic_tp_direct_admission_active
    assert sid in root.agentic_p_workset_broker._intents
    assert sid in root.agentic_tp_direct_local_failed
    assert root._agentic_early_direct_slots_used() == 1
    g.shared["receipts"][key] = -1
    root.agentic_early_direct_admission_queue.append((g.request, g.payload, g.manifest))
    root.agentic_early_direct_admission_ids.add(sid)
    admit(g)
    root._agentic_progress_tp_direct_grants(g.store)
    assert g.shared["publications"] == ([(key, 1)] if publish_error else [])
    assert not g.starts
    # The already-existing native abort command cancels this pregrant owner.
    command = root._agentic_tp_prepare_admission_control()
    assert command["direct_commands"][0]["action"] == "abort"


def test_changed_wire_room_never_reuses_cached_prepare_permission():
    g = group(2)
    g.store.current = replace(g.manifest, direct_room=13)
    admit(g)
    assert not g.shared["publications"]
    assert not g.owners[0].agentic_tp_direct_admission_active
    assert g.owners[0]._agentic_early_direct_slots_used() == 0


@pytest.mark.parametrize("allocated", [False, True])
def test_old_permission_cannot_claim_replacement_offer_before_or_after_grant(allocated):
    g = group(2)
    root, sid = g.owners[0], g.request.snapshot_id
    admit(g)
    if allocated:
        command = root._agentic_tp_prepare_admission_control()
        root._agentic_tp_consume_admission_control([command])
        root.agentic_p_workset_broker.service(g.allocators[0])
    g.store.current = replace(g.manifest, direct_room=13)
    root._agentic_progress_tp_direct_grants(g.store)
    assert not g.starts and sid in root.agentic_tp_direct_local_failed
    assert root.agentic_tp_direct_mailbox._key(sid) == EventKey(sid, "direct-room:12")
    assert g.allocators[0].available_size() == (52 if allocated else 64)


@pytest.mark.parametrize("size", [2, 8])
def test_follower_posted_before_leader_allocation_cancel_keeps_all_rank_fence(size):
    g = group(size)
    root, follower, sid = g.owners[0], g.owners[-1], g.request.snapshot_id
    admit(g)
    command = root._agentic_tp_prepare_admission_control()
    for owner in g.owners:
        owner._agentic_tp_consume_admission_control([command])
    follower.agentic_p_workset_broker.service(g.allocators[-1])
    follower._agentic_progress_tp_direct_grants(g.store)
    lease = follower.agentic_p_workset_broker.get(sid)
    follower.agentic_p_workset_broker.mark_io_inflight(sid, lease, "physical")
    assert root.agentic_p_workset_broker.get(sid) is None
    # Cancel before rank0's native allocation. Existing native abort must not
    # interpret any peer's active(None) record as an all-rank no-I/O proof.
    key = EventKey(sid, "direct-room:12")
    g.shared["receipts"][key] = -1
    command = root._agentic_tp_prepare_admission_control()
    for owner, allocator in zip(g.owners, g.allocators):
        owner._agentic_tp_consume_admission_control([command])
        owner.agentic_p_workset_broker.service(allocator)
    # The next native envelope freezes the owner-scoped cancellation created
    # by the preceding abort, including ranks with already-posted destinations.
    command = root._agentic_tp_prepare_admission_control()
    for owner in g.owners:
        owner._agentic_tp_consume_admission_control([command])
    assert not follower.agentic_p_workset_broker.tp_retire_ready(sid)
    assert lease.state == "release_pending"
    assert not follower.agentic_p_workset_broker.commit_tp_retire(sid)
    assert all(a.available_size() == 52 for a in g.allocators)
    assert len(g.starts) == 1
    assert follower.agentic_p_workset_broker.mark_io_quiesced(sid, lease, "physical")
    assert all(o.agentic_p_workset_broker.tp_retire_ready(sid) for o in g.owners)
    # Only the all-rank proof lets the existing native epoch commit retirement.
    root.agentic_tp_workset_retire_group_statuses = {sid: 1}
    command = root._agentic_tp_prepare_admission_control()
    assert command["workset_retire_commands"][0]["action"] == "commit"
    for owner, allocator in zip(g.owners, g.allocators):
        owner._agentic_tp_consume_admission_control([command])
        owner.agentic_p_workset_broker.service(allocator)
        assert allocator.available_size() == 64


@pytest.mark.parametrize("partial,report_error", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("socket_enabled", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_send_reports_same_pass_and_report_failure_never_repeats_dma(
    monkeypatch, partial, report_error, socket_enabled, rank
):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test" if socket_enabled else "")
    sends, posts, aborts = [], [], []
    sid = "send-posted:1"
    def send(*_a, **_kw):
        sends.append(1)
        if partial:
            raise RuntimeError("DMA posted before send raised")
    def report(key, status):
        posts.append((key, status))
        if report_error and len(posts) == 1:
            raise OSError("report ACK unknown")
    sender = NS(poll=lambda: KVPoll.WaitingForInput, init=lambda *_a, **_k: None,
                send=send, fence_failed_launch=lambda _e: KVPoll.Transferring)
    request = RequestGeneration("send-posted", 1)
    manifest = NS(snapshot_id=sid, request=request, state=SnapshotState.DIRECT_LOADING,
                  claim_id="physical", token_count=8)
    candidate = dict(req=object(), manifest=manifest, sender=sender,
                     source_page_indices=[1], sent=False, tp_command="direct",
                     local_prepared=True, setup_committed=True, setup_logged=True,
                     offer_published=True, route_published=True,
                     io_lock=threading.RLock(), metadata=NS(current=request),
                     staging=False, claimed_at=time.monotonic(),
                     fast_arrival_seen_at=time.monotonic(), created_at=time.monotonic())
    owner = NS(tp_world_size=8, tp_rank=rank, agentic_relay_worker=None,
               agentic_tp_direct_setup_mailbox=NS(publish_local_progress=report,
                                                 group_status=lambda *_a: None),
               agentic_tp_direct_abort_mailbox=NS(
                   local_status=lambda *_a: None,
                   publish_local_progress=lambda sid, status: aborts.append(status)),
               _agentic_candidate_items=lambda: ((sid, candidate),),
               _agentic_candidate_is_live_locked=lambda _sid, c: c is candidate,
               agentic_direct_setup_timeout=1.0, agentic_fast_threshold=2.0,
               agentic_force_slow_path=False, agentic_early_claim_store=object(),
               _agentic_try_final_confirmation=lambda _c: False,
               _agentic_direct_manifest=lambda *_a, **_kw: manifest,
               _agentic_try_tool_confirmation=lambda _c: True,
               _agentic_direct_kv_usage=lambda: 0.5)
    progress = D._check_agentic_direct_progress if rank == 0 else D._check_agentic_tp_follower_progress
    progress(owner, progress_relay=False)
    assert sends == [1] and candidate["sent"]
    if partial:
        assert not posts and not candidate.get("direct_send_posted")
        if rank == 0:
            assert candidate["direct_launch_failed"]
        else:
            assert candidate["tp_direct_abort_requested"] and aborts == [1]
        return
    assert candidate["direct_send_posted"] and not candidate.get("tp_direct_abort_requested")
    assert len(posts) == int(socket_enabled)
    if not socket_enabled and report_error:
        # Legacy still raises from its original next-pass report call.
        with pytest.raises(OSError):
            progress(owner, progress_relay=False)
    progress(owner, progress_relay=False)
    assert sends == [1] and candidate["direct_send_posted_reported"]
    assert not candidate.get("tp_direct_abort_requested")
