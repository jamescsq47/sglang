"""CPU fault tests for multi-node P→D control/receiver lifetime isolation."""
import queue
import threading
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.base import KVPoll
import sglang.srt.disaggregation.decode as decode_module
from sglang.srt.disaggregation.decode import DecodeTransferQueue, DecodePreallocQueue
from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostLoadManager, AgenticPToDHostReceiver, HostStageState,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler


def forbidden(*args, **kwargs):
    raise AssertionError("scheduler/receiver-lock performed remote I/O")


def transfer_queue(mailbox):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q._async_progress_enabled = True
    q._async_tp_control_enabled = True
    q._async_poll_lock = threading.Lock()
    q.enable_staging = False
    q.tp_rank = 0
    dr = SimpleNamespace(req=SimpleNamespace(rid="r", bootstrap_room=7),
                         kv_receiver=SimpleNamespace(poll=lambda: int(KVPoll.Success)))
    q.queue = [dr]
    q.scheduler = SimpleNamespace(tp_size=2, agentic_tp_p2d_receiver_mailbox=mailbox)
    return q, dr


@pytest.mark.parametrize("blocked_operation", ["publish_local", "publish_receipt"])
def test_blocked_nfs_does_not_hold_receiver_lock_or_expose_terminal(blocked_operation):
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(3)

    mailbox = SimpleNamespace(publish_local=lambda *a: None,
                              publish_receipt=lambda *a: None,
                              transfer_group_status=lambda *a: (int(KVPoll.Success), False))
    setattr(mailbox, blocked_operation, blocked)
    q, dr = transfer_queue(mailbox)
    thread = threading.Thread(target=q.background_progress, daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        assert q._async_poll_lock.acquire(blocking=False)
        q._async_poll_lock.release()
        assert q.cached_tp_transfer_results()[("r", 7)] == (None, False)
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert q.cached_tp_transfer_results()[("r", 7)] == (int(KVPoll.Success), False)
    # P may clear the shared files before D metadata commit is ready. Do not
    # downgrade the durable terminal or resurrect those files on a later poll.
    dr._async_transfer_poll = None
    mailbox.publish_local = forbidden
    mailbox.transfer_group_status = forbidden
    q.background_progress()
    assert q.cached_tp_transfer_results()[("r", 7)][0] == int(KVPoll.Success)


def test_failed_receipt_publication_cannot_release_source():
    mailbox = SimpleNamespace(publish_local=lambda *a: None,
                              publish_receipt=forbidden,
                              transfer_group_status=lambda *a: (int(KVPoll.Success), False))
    q, _ = transfer_queue(mailbox)
    with pytest.raises(AssertionError):
        q.background_progress()
    assert q.cached_tp_transfer_results()[("r", 7)] == (None, False)


def test_group_failure_waits_for_peer_dma_and_scheduler_reads_only_cache():
    mailbox = SimpleNamespace(publish_local=lambda *a: None,
                              publish_receipt=forbidden,
                              transfer_group_status=lambda *a: (int(KVPoll.Transferring), True))
    q, _ = transfer_queue(mailbox)
    q.background_progress()
    s = SimpleNamespace(tp_size=2, tp_rank=0, disaggregation_mode=DisaggregationMode.DECODE,
                        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
                        disagg_decode_transfer_queue=q, agentic_tp_p2d_receiver_mailbox=mailbox)
    mailbox.transfer_group_status = forbidden
    control = Scheduler._agentic_tp_prepare_admission_control(s)
    assert control["decode_transfer_statuses"] == [int(KVPoll.Transferring)]
    assert control["decode_transfer_cancel_keys"] == [("r", 7)]


def test_unbounded_tp_decode_admission_does_not_reapply_transfer_count_cap():
    transfers = [
        SimpleNamespace(req=SimpleNamespace(rid=f"inflight-{i}", bootstrap_room=i))
        for i in range(8)
    ]
    pending = [
        SimpleNamespace(req=SimpleNamespace(rid=f"ready-{i}", bootstrap_room=100 + i),
                        waiting_for_input=True)
        for i in range(2)
    ]
    transfer_queue = SimpleNamespace(
        queue=transfers, _async_progress_enabled=True,
        _async_tp_control_enabled=True, cached_tp_transfer_results=lambda: {},
    )
    prealloc_queue = SimpleNamespace(
        queue=pending, max_transfer_inflight=0, _async_metadata_pending_count=0,
        p_ready_dir="", _requires_p_ready=lambda req: False,
    )
    scheduler = SimpleNamespace(
        tp_size=8, tp_rank=0, disaggregation_mode=DisaggregationMode.DECODE,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
        disagg_decode_transfer_queue=transfer_queue,
        disagg_decode_prealloc_queue=prealloc_queue,
        agentic_tp_p2d_receiver_mailbox=SimpleNamespace(transfer_group_status=forbidden),
        agentic_tp_p2d_admission_mailbox=SimpleNamespace(
            group_status=lambda key: int(KVPoll.Success)
        ),
    )
    control = Scheduler._agentic_tp_prepare_admission_control(scheduler)
    assert control["decode_admit_keys"] == [("ready-0", 100), ("ready-1", 101)]
    prealloc_queue.max_transfer_inflight = 8
    control = Scheduler._agentic_tp_prepare_admission_control(scheduler)
    assert control["decode_admit_keys"] == []


def test_prefill_background_control_does_not_duplicate_shared_reads():
    request = SimpleNamespace(rid="r", bootstrap_room=7, disagg_p_ready_notified=False)
    s = SimpleNamespace(tp_size=2, tp_rank=0, disaggregation_mode=DisaggregationMode.PREFILL,
                        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
                        _prefill_transfer_tp_background_enabled=True,
                        agentic_tp_direct_admission_active={}, agentic_early_direct_receives={},
                        disagg_prefill_inflight_queue=[request],
                        agentic_tp_p2d_sender_mailbox=SimpleNamespace(
                            group_status=lambda key: int(KVPoll.Bootstrapping),
                            transfer_group_status=lambda key: (int(KVPoll.Transferring), False)),
                        agentic_tp_p2d_receiver_mailbox=SimpleNamespace(receipt=forbidden),
                        agentic_p2d_host_staging_manager=SimpleNamespace(group_claimed=forbidden),
                        _publish_deferred_prefill_ready=forbidden,
                        agentic_host_staging_manager=None)
    control = Scheduler._agentic_tp_prepare_admission_control(s)
    assert control["prefill_transfer_statuses"] == [int(KVPoll.Transferring)]


def host_manager(ledger):
    m = AgenticPToDHostLoadManager.__new__(AgenticPToDHostLoadManager)
    m._async_host_control = True
    m.ledger = ledger
    m._work = queue.SimpleQueue()
    m._completion_lock = threading.RLock()
    m._pending_aborts = {}
    m._group_pending = {}
    m._group_wakeup = threading.Event()
    m._stop = threading.Event()
    m._dma_quarantine = []
    m._dma_poisoned = False
    m.tp_rank, m.tp_size, m.numa_node, m.chunk_tokens = 0, 2, 0, 1024
    m._remote_bridge = SimpleNamespace(load=forbidden)
    return m


def test_host_submit_and_abort_do_not_read_ledger_or_release_queued_pages():
    calls = []
    ledger = SimpleNamespace(get=forbidden)
    m = host_manager(ledger)
    receiver = AgenticPToDHostReceiver(m, "generation")
    receiver.bind([1, 2])
    receiver.bind([1, 2])
    assert m._work.qsize() == 1
    receiver.abort()
    assert receiver.poll() == int(KVPoll.Transferring)
    assert not receiver._terminal  # queued is not physically drained
    ledger.get = lambda key: {"p_owner": "P"}
    ledger.request_host_load_failure = lambda *a, **kw: calls.append("intent")
    ledger.mark_host_load_rank_drained = lambda *a, **kw: calls.append("drained")
    m._progress_abort_requests_once()
    assert calls == ["intent"]
    assert not m._pending_aborts
    # Only the original load worker's physical completion may permit release.
    m._group_pending[receiver.snapshot_id] = object()
    receiver.abort()
    m._progress_abort_requests_once()
    assert calls[-2:] == ["intent", "drained"]


def test_host_prepare_uses_original_grant_and_claim_once():
    grant = {"kind": "shared_host_extent", "tp_rank": 0, "arena_numa_node": 0,
             "token_count": 2, "byte_size": 16}
    calls = []
    ledger = SimpleNamespace(
        get=lambda key: {"p_owner": "P", "state": HostStageState.HOST_READY.value,
                         "grants": [grant]},
        begin_host_load_rank=lambda *a, **kw: calls.append((a, kw)) or True)
    m = host_manager(ledger)
    r = AgenticPToDHostReceiver(m, "generation")
    r.bind([1, 2])
    assert calls == []
    assert m._prepare_load(*m._work.get())
    assert len(calls) == 1 and r._grant is grant and r._owner == "P"
    assert r.poll() == int(KVPoll.Transferring)


def test_abort_before_remote_prepare_drains_without_dma(monkeypatch):
    calls = []
    ledger = SimpleNamespace(
        get=lambda key: {"p_owner": "P", "state": HostStageState.ABORTING.value,
                         "h2d_abort_started": True},
        mark_host_load_rank_drained=lambda *a, **kw: calls.append("drained"))
    m = host_manager(ledger)
    r = AgenticPToDHostReceiver(m, "generation")
    r.bind([1, 2])
    m._work.put(None)
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    m._worker(0, SimpleNamespace(device="cuda:0"), None, ())
    assert calls == ["drained"]
    assert r.poll() == int(KVPoll.Failed)


@pytest.mark.parametrize("failure", ["bad_grant", "claim_rejected"])
def test_async_prepare_failure_keeps_target_until_group_failure(failure, monkeypatch):
    calls = []
    entry = {
        "p_owner": "P", "state": HostStageState.HOST_READY.value,
        "grants": [{"kind": "shared_host_extent", "tp_rank": 0,
                    "arena_numa_node": 0, "token_count": 2, "byte_size": 16}],
    }
    if failure == "bad_grant":
        entry["grants"] = []

    def fail(snapshot, owner, **kw):
        assert owner == "P"
        calls.append("abort_intent")
        entry.update(state=HostStageState.ABORTING.value, h2d_abort_started=True)

    def drain(snapshot, owner, **kw):
        assert owner == "P"
        calls.append("rank_drained")

    ledger = SimpleNamespace(get=lambda key: entry,
                              begin_host_load_rank=lambda *a, **kw: False,
                              request_host_load_failure=fail,
                              mark_host_load_rank_drained=drain)
    m = host_manager(ledger)
    r = AgenticPToDHostReceiver(m, "generation")
    r.bind([1, 2])
    m._work.put(None)
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    m._worker(0, SimpleNamespace(device="cuda:0"), None, ())
    assert calls == ["abort_intent", "rank_drained"]
    assert r.poll() == int(KVPoll.Transferring)
    assert "generation" in m._group_pending
    # Completion waits for group failure, not merely local preparation error.
    entry["state"] = HostStageState.FAILED.value
    m._progress_group_completions_once()
    assert r.poll() == int(KVPoll.Failed)
    assert not m._group_pending


def test_blocked_host_prepare_does_not_hold_receiver_state_lock(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    drained = []

    def get(key):
        entered.set()
        assert release.wait(3)
        return {"p_owner": "P", "state": HostStageState.ABORTING.value,
                "h2d_abort_started": True}

    m = host_manager(SimpleNamespace(get=get, mark_host_load_rank_drained=lambda *a, **kw: drained.append(1)))
    r = AgenticPToDHostReceiver(m, "generation")
    r.bind([1, 2])
    m._work.put(None)
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    thread = threading.Thread(target=m._worker,
        args=(0, SimpleNamespace(device="cuda:0"), None, ()), daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        assert r._state_lock.acquire(blocking=False)
        try:
            assert r.poll() == int(KVPoll.Transferring)
            r.abort()
            assert not r._terminal
        finally:
            r._state_lock.release()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert drained == [1]
    assert r.poll() == int(KVPoll.Failed)


def test_metadata_failure_after_host_bind_keeps_target_in_transfer_queue(monkeypatch):
    m = host_manager(SimpleNamespace(get=forbidden))
    r = AgenticPToDHostReceiver(m, "generation")
    r.bind([1, 2])
    dr = SimpleNamespace(kv_receiver=r, req=object(), metadata_buffer_index=4)
    done = queue.SimpleQueue()
    done.put((dr, OSError("ready marker write failed after bind")))
    q = SimpleNamespace(
        _async_metadata_done=done, _async_metadata_count_lock=threading.Lock(),
        _async_metadata_pending_count=1,
        req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=forbidden),
        scheduler=SimpleNamespace(stream_output=forbidden), tree_cache=None,
    )
    monkeypatch.setattr(decode_module, "prepare_abort", lambda *a, **kw: None)
    monkeypatch.setattr(decode_module, "release_kv_cache", forbidden)
    ready, failed = DecodePreallocQueue._drain_background_metadata(q)
    assert ready == [dr] and failed == []
    assert dr.metadata_buffer_index == 4
    assert r.abort_pending and not r._terminal
    assert r.poll() == int(KVPoll.Transferring)
