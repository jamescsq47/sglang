"""Socket Direct cancellation/deadline regressions, with no CUDA or files."""

import time
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, SnapshotState
from sglang.srt.disaggregation.agentic_tp_events import EventKey
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_kvcache_offload_manager import DecodeKVCacheOffloadManager as D
from sglang.srt.managers.scheduler import Scheduler


class Reports:
    def __init__(self, shared, rank, size):
        self.shared, self.rank, self.size = shared, rank, size

    def publish_local_progress(self, key, value):
        self.shared[self.rank] = value

    def group_status(self, key):
        assert self.rank == 0
        return min(self.shared.values()) if len(self.shared) == self.size else None


@pytest.mark.parametrize("size", [2, 8])
def test_setup_deadline_waits_for_every_successful_send_not_dma(size):
    posted, aborts = {}, {}
    manifest = NS(snapshot_id="req:0", state=SnapshotState.DIRECT_LOADING)
    owners = [NS(tp_rank=r, agentic_direct_setup_timeout=1.,
                 agentic_tp_direct_setup_mailbox=Reports(posted, r, size),
                 agentic_tp_direct_abort_mailbox=Reports(aborts, r, size))
              for r in range(size)]
    candidates = [dict(manifest=manifest, fast_arrival_seen_at=10., sent=True,
                       direct_send_posted=True) for _ in range(size)]
    for r in range(size):
        D._agentic_progress_direct_setup(owners[r], candidates[r], 10.5)
    # Source DMA can still be in flight long after the link deadline.
    D._agentic_progress_direct_setup(owners[0], candidates[0], 30.)
    assert candidates[0]["direct_group_started"] and not aborts
    # Missing follower must not disappear just because rank0 sent its shard.
    posted.pop(size - 1)
    candidates[0].pop("direct_group_started")
    D._agentic_progress_direct_setup(owners[0], candidates[0], 31.)
    assert candidates[0]["tp_direct_abort_requested"] and aborts == {0: 1}


def test_partial_failed_post_is_not_a_successful_setup_ack():
    reports = {}
    owner = NS(tp_rank=1, agentic_tp_direct_setup_mailbox=Reports(reports, 1, 8))
    candidate = dict(manifest=NS(snapshot_id="partial:0"), sent=True)
    D._agentic_progress_direct_setup(owner, candidate, 100.)
    assert reports == {}  # sent=True conservatively means possibly posted


@pytest.mark.parametrize("size", [2, 8])
def test_source_abort_joins_native_rollback_without_second_barrier(monkeypatch, size):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp-test")
    request = RequestGeneration("cancel", 0)
    manifest = NS(state=SnapshotState.DIRECT_LOADING, claim_id="exact")
    store = NS(load=lambda *a, **k: manifest)
    entries = []
    owners = []
    for rank in range(size):
        entry = NS(request=request, claim_id="exact", io_quiesced=True,
                   direct_abort_marker_seen=True, direct_abort_fence_kind="terminal",
                   transport_poll=KVPoll.WaitingForInput)
        owner = NS(tp_rank=rank, tp_size=size, agentic_tp_direct_local_failed=set())
        # No second abort mailbox exists. Every rank must converge anyway.
        assert not Scheduler._agentic_handle_unstarted_direct_abort(
            owner, entry, store, object(), KVPoll.WaitingForInput, 1.)
        assert entry.transport_poll == KVPoll.Failed
        assert request.snapshot_id in owner.agentic_tp_direct_local_failed
        entries.append(entry)
        owners.append(owner)
    # Repeated fence processing must not swallow generic terminal cleanup.
    assert not Scheduler._agentic_handle_unstarted_direct_abort(
        owners[0], entries[0], store, object(), KVPoll.Failed, 1.)


def test_source_fence_cannot_release_a_different_attempt(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp-test")
    entry = NS(request=RequestGeneration("stale", 0), claim_id="old",
               io_quiesced=False, direct_abort_marker_seen=True,
               direct_abort_fence_kind="terminal", transport_poll=KVPoll.WaitingForInput)
    owner = NS(tp_rank=0, tp_size=8, agentic_tp_direct_local_failed=set())
    store = NS(load=lambda *a, **k: NS(state=SnapshotState.DIRECT_LOADING, claim_id="new"))
    assert not Scheduler._agentic_handle_unstarted_direct_abort(
        owner, entry, store, object(), KVPoll.WaitingForInput, 1.)
    assert not entry.io_quiesced and not owner.agentic_tp_direct_local_failed


def test_p_does_not_apply_link_deadline_to_pending_dma(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp-test")
    request = RequestGeneration("long-dma", 0)
    started = []
    owner = NS(tp_rank=0, agentic_early_direct_poll_lock=nullcontext(),
               agentic_tp_direct_admission_active={request.snapshot_id: (request, time.time()-20, None, 1024, object())},
               agentic_tp_direct_mailbox=NS(receipt=lambda key: 1),
               agentic_early_direct_receives={}, agentic_tp_direct_local_failed=set(),
               agentic_tp_direct_local_admitted=set(),
               _agentic_tp_start_direct_shard=lambda *a, **k: started.append(a))
    Scheduler._agentic_progress_tp_direct_grants(owner, object())
    assert started and not owner.agentic_tp_direct_local_failed


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("case", ["missing_rank0", "completed_rank0", "stale", "committed", "no_fence"])
def test_source_abort_is_observed_by_group_not_only_live_receivers(monkeypatch, size, case):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp-test")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "p0")
    request = RequestGeneration("partial-complete", 0)
    claim = f"direct-early-tp:p0:{request.snapshot_id}"
    aborts, starts = [], []
    current = NS(state=SnapshotState.P_RECEIVED if case == "committed"
                 else SnapshotState.DIRECT_LOADING,
                 claim_id="other" if case == "stale" else claim, created_at=10.)
    entry = NS(completed_at=time.monotonic()) if case == "completed_rank0" else None
    def read_abort(req, **kwargs):
        assert kwargs['claim_id'] == claim
        assert kwargs['not_before'] == 0.  # Router and D clocks may differ.
        return None if case == "no_fence" else {'fence_kind': 'terminal', 'arrived_at': 50.}
    owner = NS(tp_rank=0, tp_size=size,
               agentic_early_direct_poll_lock=nullcontext(),
               agentic_tp_direct_admission_active={request.snapshot_id: (request, time.time()-20, None, 1024, object())},
               agentic_tp_direct_mailbox=NS(receipt=lambda key: 1),
               agentic_early_claim_store=NS(read_direct_abort=read_abort),
               agentic_early_direct_receives={} if entry is None else {request.snapshot_id: entry},
               agentic_tp_direct_local_failed=set(), agentic_tp_direct_local_admitted=set(),
               _agentic_abort_tp_direct_grant=lambda *a, **k: aborts.append(k),
               _agentic_tp_start_direct_shard=lambda *a, **k: starts.append(a))
    Scheduler._agentic_progress_tp_direct_grants(owner, NS(load=lambda *a, **k: current))
    if case in {"missing_rank0", "completed_rank0"}:
        assert len(aborts) == 1 and not starts
        assert aborts[0]['reason'] == 'source_group_abort_fenced'
    else:
        assert not aborts


def test_failed_authoritative_claim_release_keeps_grant_for_retry(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "p0")
    request = RequestGeneration("release-retry", 0)
    decisions = []
    def unavailable(*args):
        raise OSError("broker temporarily unavailable")
    owner = NS(agentic_early_direct_poll_lock=nullcontext(),
               agentic_tp_direct_admission_active={request.snapshot_id: (request, time.time(), None, 1024, object())},
               agentic_tp_direct_mailbox=NS(publish_receipt=lambda *a: decisions.append(a)))
    store = NS(load=lambda *a, **k: NS(state=SnapshotState.DIRECT_LOADING,
                claim_id=f"direct-early-tp:p0:{request.snapshot_id}"),
               release_direct_claim=unavailable)
    assert not Scheduler._agentic_abort_tp_direct_grant(
        owner, request, store, reason="all_ranks_rolled_back", rolled_back=True)
    assert not decisions and request.snapshot_id in owner.agentic_tp_direct_admission_active


def test_source_abort_overrides_success_in_same_progress_tick(monkeypatch):
    from sglang.srt.managers.scheduler import AgenticEarlyDirectReceive
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp-test")
    request = RequestGeneration("success-abort-race", 0)
    entry = AgenticEarlyDirectReceive(
        request=request, manifest=NS(), claim_id="exact", receiver=NS(),
        device_indices=None, started_at=time.monotonic(), arrived_at=time.time(),
        transport_poll=KVPoll.Success,
    )
    entry.io_quiesced = True
    failures = []
    def fenced(*args):
        entry.transport_poll = KVPoll.Failed
        return False
    owner = NS(
        tp_size=8, agentic_early_claim_store=object(), agentic_direct_runtime=object(),
        agentic_early_direct_receives={request.snapshot_id: entry},
        agentic_early_direct_poll_lock=nullcontext(), agentic_early_direct_terminal={},
        _agentic_snapshot_store=lambda: object(),
        _agentic_collect_direct_arrivals=lambda *a: None,
        _agentic_collect_direct_abort_events=lambda: None,
        _agentic_admit_queued_direct_receives=lambda *a: None,
        _agentic_progress_tp_direct_grants=lambda *a: None,
        _agentic_handle_unstarted_direct_abort=fenced,
        _agentic_mark_tp_direct_failed=lambda *a, **k: failures.append(k["reason"]),
        _agentic_commit_tp_direct_groups=lambda *a: None,
    )
    Scheduler._agentic_poll_early_direct_receives_once(owner)
    assert failures == ["transfer_failed_or_timeout"]
    assert entry.completed_at is None


@pytest.mark.parametrize("never_started_rank", [None, 0, 7])
def test_partial_tp8_abort_retires_grant_and_returns_direct_slot(monkeypatch, never_started_rank):
    import threading
    from sglang.srt.managers.scheduler import AgenticEarlyDirectReceive, AgenticPWorksetLeaseBroker, DisaggregationMode
    from sglang.srt.disaggregation.test_agentic_direct_descriptors import Allocator

    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp-test")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ENGINE_ID", "p0")
    request = RequestGeneration("partial-grant", 0)
    claim = f"direct-early-tp:p0:{request.snapshot_id}"
    manifest = NS(state=SnapshotState.DIRECT_LOADING, claim_id=claim, created_at=10.)
    rolled_back = set()
    receipt = [1]
    releases = []

    class Mailbox:
        def __init__(self, rank): self.rank = rank
        def receipt(self, sid): return receipt[0] if sid == request.snapshot_id else 1
        def publish_receipt(self, sid, status): receipt[0] = status
        def _key(self, sid): return EventKey(sid, "direct-room:12")
        def publish_local_progress(self, *a): pass
        def publish_local_rollback_complete(self, sid): rolled_back.add(self.rank)
        def rollback_group_complete(self, sid): return len(rolled_back) == 8
        def group_status(self, sid): return None
        def clear_local(self, sid): pass
        def clear_local_rollback(self, sid): pass
        def clear_group(self, sid): pass
        def clear_group_rollback(self, sid): pass

    def release_claim(current, exact):
        assert len(rolled_back) == 8 and exact == claim
        releases.append(exact)
        current.state = SnapshotState.DIRECT_READY
        current.claim_id = None
        return current

    store = NS(load=lambda *a, **k: manifest, release_direct_claim=release_claim)
    owners = []
    for rank in range(8):
        owner = Scheduler.__new__(Scheduler)
        owner.tp_rank, owner.tp_size = rank, 8
        owner.disaggregation_mode = DisaggregationMode.PREFILL
        owner.agentic_tp_direct_mailbox = Mailbox(rank)
        owner.agentic_tp_direct_group_status = {}
        owner.agentic_early_direct_poll_lock = nullcontext()
        owner.agentic_early_direct_terminal = {}
        owner.agentic_tp_direct_local_failed = set()
        owner.agentic_tp_direct_local_admitted = set()
        owner.agentic_tp_direct_local_rolled_back = set()
        owner.agentic_tp_host_local_admitted = set()
        owner.agentic_tp_host_active = None
        owner.agentic_tp_host_active_since = 0.
        owner.agentic_tp_host_command_visible = False
        owner.agentic_tp_host_group_status = 0
        owner.agentic_host_staging_manager = None
        owner.agentic_early_direct_progress_thread = threading.current_thread()
        owner._agentic_snapshot_store = lambda: store
        owner._agentic_tp_start_direct_shard = lambda *a, **k: None
        owner.agentic_early_claim_store = NS(read_direct_abort=lambda req, **k:
            {'fence_kind': 'terminal', 'arrived_at': 50.} if req == request else None)
        owner._agentic_clear_direct_receiver = lambda *a: None
        owner._agentic_rollback_prepared_direct_bind = lambda entry: None
        broker = owner.agentic_p_workset_broker = AgenticPWorksetLeaseBroker(4)
        broker.request(request.snapshot_id, 4, 8,
                       owner=broker.direct_owner(request.snapshot_id))
        broker.service(Allocator(64))
        lease = broker.get(request.snapshot_id)  # physical fence is established below
        owner.agentic_tp_direct_admission_active = {
            request.snapshot_id: (request, time.time(), None, 1024, lease),
            **{f"other:{n}": (RequestGeneration("other", n), time.time(), None, 1, None)
               for n in range(3)},
        }
        entry = AgenticEarlyDirectReceive(
            request=request, manifest=manifest, claim_id=claim,
            receiver=NS(clear=lambda: None), device_indices=None,
            started_at=time.monotonic(), arrived_at=time.time(), workset_lease=lease,
            io_quiesced=True, transport_poll=KVPoll.Failed if rank else KVPoll.WaitingForInput,
        )
        entry.direct_abort_marker_seen = True
        entry.direct_abort_fence_kind = "terminal"
        if never_started_rank == 0:
            # r2: seven completed receives bypassed receiver-poll cancellation
            # while rank0 had no receiver at all. Only the group grant can
            # observe the source fence and initiate the first abort receipt.
            entry.completed_at = time.monotonic()
            entry.transport_poll = KVPoll.Success
        owner.agentic_early_direct_receives = (
            {} if rank == never_started_rank else {request.snapshot_id: entry}
        )
        owners.append(owner)

    def command(action):
        return {Scheduler._AGENTIC_TP_CONTROL_KEY: True, "workset_plan_epoch": 1,
                "direct_commands": [dict(snapshot=request.snapshot_id,
                  request_id=request.request_id, generation=0, action=action)],
                "host_commands": [], "prefill_transfer_keys": []}

    root = owners[0]
    Scheduler._agentic_progress_tp_direct_grants(root, store)
    assert receipt[0] == -1 and not releases
    # The source fence initiates rollback, not page/claim release. Every rank
    # still has to execute the native command and acknowledge its own fence.
    for owner in owners[1:]:
        Scheduler._agentic_tp_consume_admission_control(owner, [command("abort")])
    assert rolled_back == set(range(1, 8)) and not releases
    assert Scheduler._agentic_early_direct_slots_used(root) == 4
    entry = root.agentic_early_direct_receives.get(request.snapshot_id)
    if entry is not None:
        assert not Scheduler._agentic_handle_unstarted_direct_abort(
            root, entry, store, object(), KVPoll.WaitingForInput, 1.)
    Scheduler._agentic_tp_consume_admission_control(root, [command("abort")])
    Scheduler._agentic_commit_tp_direct_groups(root, store)
    assert receipt[0] == -2 and releases == [claim]
    for owner in owners:
        Scheduler._agentic_tp_consume_admission_control(owner, [command("clear")])
        assert request.snapshot_id not in owner.agentic_early_direct_receives
        assert request.snapshot_id not in owner.agentic_tp_direct_admission_active
        assert Scheduler._agentic_early_direct_slots_used(owner) == 3
        # The fifth request can acquire a slot without enlarging the cap.
