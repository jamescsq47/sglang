"""Direct-retention ablation: deadlines never authorize releasing the parent."""
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace as NS
import threading
import time

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, SnapshotState
from sglang.srt.disaggregation.decode_kvcache_offload_manager import DecodeKVCacheOffloadManager as Manager
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.scheduler import Scheduler


def candidate_fixture(state, *, sent=False, poll=KVPoll.WaitingForInput):
    request = RequestGeneration("strict-wait", 3)
    manifest = NS(request=request, snapshot_id=request.snapshot_id, state=state,
                  created_at=time.time()-100, claim_id="claim", token_count=128)
    events = []
    candidate = dict(req=NS(req_pool_idx=1), metadata=NS(current=request), manifest=manifest,
                     sender=NS(poll=lambda: poll, init=lambda *a, **k: events.append("init"),
                               send=lambda *a, **k: events.append("send")),
                     sent=sent, staging=False, claimed_at=None, created_at=time.monotonic()-100,
                     fast_arrival_seen_at=time.monotonic()-90, source_page_indices=[1, 2],
                     fallback_retry_at=0, io_lock=threading.RLock())
    manager = NS(tp_world_size=1, tp_rank=0, agentic_direct_wait_only=True,
                 agentic_fast_threshold=float("inf"), agentic_direct_setup_timeout=float("inf"),
                 agentic_relay_worker=None, agentic_host_staging_client=None,
                 agentic_early_claim_store=object(),
                 _agentic_candidate_items=lambda: [(request.snapshot_id, candidate)],
                 _agentic_candidate_is_live_locked=lambda *a: True,
                 _agentic_try_final_confirmation=lambda *a: False,
                 _agentic_direct_manifest=lambda *a, **k: manifest,
                 _agentic_try_early_claim=lambda *a: "arrived",
                 _agentic_release_early_claim=lambda *a: events.append("release_claim"),
                 _retire_candidate_for_release=lambda *a: events.append("release_source"),
                 _cleanup_agentic_direct_sender=lambda *a: events.append("cleanup"))
    return manager, candidate, events


def test_late_tool_retains_source_without_claim():
    m, c, events = candidate_fixture(SnapshotState.DIRECT_READY)
    m._agentic_try_early_claim = lambda *a: "absent"
    Manager._check_agentic_direct_progress(m, progress_relay=False)
    assert not events and not c["staging"] and not c["sent"]


def test_delayed_claim_can_still_start_then_release_on_consumed():
    m, c, events = candidate_fixture(SnapshotState.DIRECT_LOADING)
    Manager._check_agentic_direct_progress(m, progress_relay=False)
    assert events == ["init", "send"] and c["sent"]
    c["manifest"].state = SnapshotState.CONSUMED
    c["sender"].poll = lambda: KVPoll.Transferring
    Manager._check_agentic_direct_progress(m, progress_relay=False)
    assert events == ["init", "send"]  # logical ACK is not a physical fence
    c["sender"].poll = lambda: KVPoll.Success
    Manager._check_agentic_direct_progress(m, progress_relay=False)
    assert events[-3:] == ["cleanup", "release_claim", "release_source"]


@pytest.mark.parametrize("state", [SnapshotState.DIRECT_READY, SnapshotState.FAILED])
def test_returned_or_failed_attempt_is_not_recomputed_or_freed(state, caplog):
    m, c, events = candidate_fixture(state, sent=True, poll=KVPoll.Success)
    Manager._check_agentic_direct_progress(m, progress_relay=False)
    assert not events and c["direct_wait_failure_logged"]
    assert "strict_direct_failed" in caplog.text


def test_late_arrival_is_accepted_with_original_identity():
    m, c, _ = candidate_fixture(SnapshotState.DIRECT_READY)
    c["fast_arrival_seen_at"] = None
    m.agentic_early_claim_poll_interval = .05
    m.agentic_early_claim_store = NS(read_arrival=lambda *a, **k: {
        "arrived_at": c["manifest"].created_at+60})
    assert Manager._agentic_try_early_claim(m, c, time.monotonic()) == "arrived"
    assert c["fast_arrival_seen"]


def test_tp8_incomplete_setup_waits_but_all_posts_commit():
    m, c, events = candidate_fixture(SnapshotState.DIRECT_LOADING)
    m.tp_world_size = 8
    statuses = [0]*8
    m.agentic_tp_direct_setup_mailbox = NS(group_status=lambda sid: min(statuses))
    m.agentic_tp_direct_abort_mailbox = NS(publish_local_progress=lambda *a: events.append("abort"))
    Manager._agentic_progress_direct_setup(m, c, time.monotonic())
    assert not events and not c.get("tp_direct_abort_requested")
    statuses[:] = [1]*8
    Manager._agentic_progress_direct_setup(m, c, time.monotonic())
    assert c["direct_group_started"] and not events


def test_old_queued_arrival_survives_capacity_wait_then_starts(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    req = RequestGeneration("old-arrival", 2)
    arrival = time.time()-600
    manifest = NS(request=req, state=SnapshotState.DIRECT_READY, created_at=arrival, token_count=128)
    events, credit = [], [None]
    scheduler = NS(tp_size=1, tp_rank=0, agentic_early_claim_store=object(),
        agentic_tp_direct_admission_active={}, agentic_early_direct_receives={},
        agentic_early_direct_terminal={},
        agentic_early_direct_admission_queue=deque([(req, {"arrived_at":arrival,"prompt_token_count":256},manifest)]),
        agentic_early_direct_admission_ids={req.snapshot_id}, server_args=NS(page_size=64),
        agentic_p_workset_broker=NS(owner_is_superseded=lambda *a, **k:False,
            request=lambda *a, **k:None, get=lambda *a, **k:credit[0],
            cancel_unstarted=lambda *a, **k:pytest.fail("expired a live parent")),
        _agentic_start_early_direct_receive=lambda *a, **k:events.append("start") or True)
    store = NS(load=lambda *a, **k:manifest)
    Scheduler._agentic_admit_queued_direct_receives(scheduler, store, float("inf"), nullcontext())
    assert not events and len(scheduler.agentic_early_direct_admission_queue)==1
    credit[0] = object()
    Scheduler._agentic_admit_queued_direct_receives(scheduler, store, float("inf"), nullcontext())
    assert events == ["start"] and not scheduler.agentic_early_direct_admission_queue


@pytest.mark.parametrize("strict", [False, True])
def test_failed_manifest_does_not_authorize_strict_tp_recompute(monkeypatch, strict):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_WAIT_ONLY", str(strict).lower())
    parent = RequestGeneration("failed-parent", 2)
    scheduler = NS(_agentic_snapshot_store=lambda:NS(load=lambda *a, **k:
        NS(state=SnapshotState.FAILED, failure_reason="injected_failure")))
    result = Scheduler._agentic_no_host_terminal_reason(scheduler, parent)
    assert result == (None if strict else "injected_failure")
