"""Late application final ACKs must reclaim Host, never a live DMA/workset."""
import threading
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticPHostStagingManager, HostStageState, SharedHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="agentic-final-test-", dir="/dev/shm") as path:
        yield Path(path)


def setup(tmp_path, *, tp_size=1, state="host_ready", count=1):
    ledger = SharedHostStagingLedger(str(tmp_path / "host.json"))
    store = AgenticEarlyClaimStore(str(tmp_path / "claims"))
    requests = [RequestGeneration(f"request-{i}", 3) for i in range(count)]
    for request in requests:
        ledger.offer(dict(snapshot_id=request.snapshot_id, token_count=64,
                          byte_size=4096, tp_size=tp_size))

        def ready(entries):
            entries[request.snapshot_id].update(p_owner="p:test", state=state,
                                                updated_at=time.time())
            return True, True

        ledger._mutate(ready, event_snapshot_id=request.snapshot_id)
    managers = []
    for rank in range(tp_size):
        m = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        m.owner = "p:test"
        m.tp_rank, m.tp_size = rank, tp_size
        m.ledger = ledger
        m._final_host_claim_store = store
        m._state_lock = threading.RLock()
        m.host_ready = {q.snapshot_id: dict(offer=dict(token_count=64, byte_size=4096))
                        for q in requests}
        m.active = {}
        m._host_eviction_local_released = set()
        m._host_eviction_count = m._host_eviction_tokens = m._host_eviction_bytes = 0
        m.released, m.notifications = [], []
        m._release_record = lambda record, m=m: m.released.append(record) or True
        m._notify_scheduler = lambda *args, m=m: m.notifications.append(args)
        managers.append(m)
    return ledger, store, requests, managers


def entries(ledger):
    return ledger.snapshot_entries(force_refresh=True)


def test_late_final_releases_once_without_recompute(tmp_path):
    ledger, store, (request,), (m,) = setup(tmp_path)
    m._progress_final_host_cleanup(entries(ledger))
    assert not m.released
    store.publish_final(request)
    m._progress_final_host_cleanup(entries(ledger))
    for _ in range(3):
        m._progress_final_host_cleanup(entries(ledger))
        m._progress_host_evictions(entries(ledger))
    assert ledger.get(request.snapshot_id)["state"] == "consumed"
    assert len(m.released) == 1
    assert not m.host_ready and not m.notifications
    assert m._host_eviction_count == 0
    assert not m._host_eviction_local_released


@pytest.mark.parametrize("field,value", [
    ("recovery_owner", "p:remote"), ("recovery_claims", {"0": {"phase": "pinned"}}),
    ("loading_ranks", [0]), ("h2d_prepared_ranks", [0]),
    ("loader_acks", [0]), ("binder_acks", [0]),
])
def test_final_cannot_bypass_recovery_fences(tmp_path, field, value):
    ledger, store, (request,), (m,) = setup(tmp_path)
    def pin(data):
        data[request.snapshot_id][field] = value
        return True, True
    ledger._mutate(pin, event_snapshot_id=request.snapshot_id)
    store.publish_final(request)
    m._progress_final_host_cleanup(entries(ledger))
    assert not m.released
    assert ledger.get(request.snapshot_id)["state"] == "host_ready"


@pytest.mark.parametrize("state", ["offered", "host_reserved", "host_writing", "aborting", "h2d_loading", "hbm_ready"])
def test_final_waits_for_durable_and_never_interrupts_dma(tmp_path, state):
    ledger, store, (request,), (m,) = setup(tmp_path, state=state)
    store.publish_final(request)
    m._progress_final_host_cleanup(entries(ledger))
    assert not m.released
    assert ledger.get(request.snapshot_id)["state"] == state
    def durable(data):
        data[request.snapshot_id]["state"] = "host_ready"
        return True, True
    ledger._mutate(durable, event_snapshot_id=request.snapshot_id)
    m._progress_final_host_cleanup(entries(ledger))
    assert len(m.released) == 1


def test_tp2_final_terminal_only_after_both_physical_releases(tmp_path):
    ledger, store, (request,), (m0, m1) = setup(tmp_path, tp_size=2)
    store.publish_final(request)
    m1._progress_final_host_cleanup(entries(ledger))
    assert not m1.released
    m0._progress_final_host_cleanup(entries(ledger))
    assert ledger.get(request.snapshot_id)["state"] == "evicting"
    assert len(m0.released) == 1 and not m1.released
    m0._progress_host_evictions(entries(ledger))
    assert ledger.get(request.snapshot_id)["state"] == "evicting"
    m1._progress_host_evictions(entries(ledger))
    assert ledger.get(request.snapshot_id)["state"] == "consumed"
    assert len(m0.released) == len(m1.released) == 1
    assert not m0.notifications and not m1.notifications


def test_release_failure_retains_record_and_retries(tmp_path):
    ledger, store, (request,), (m,) = setup(tmp_path)
    store.publish_final(request)
    m._release_record = lambda record: False
    m._progress_final_host_cleanup(entries(ledger))
    assert request.snapshot_id in m.host_ready
    assert ledger.get(request.snapshot_id)["state"] == "evicting"
    m._release_record = lambda record: m.released.append(record) or True
    m._progress_host_evictions(entries(ledger))
    assert len(m.released) == 1 and ledger.get(request.snapshot_id)["state"] == "consumed"


def test_ack_failure_does_not_double_free(tmp_path, monkeypatch):
    ledger, store, (request,), (m,) = setup(tmp_path)
    original = ledger.complete_host_eviction_rank
    def fail(*args, **kwargs):
        raise OSError("ACK unavailable")
    monkeypatch.setattr(ledger, "complete_host_eviction_rank", fail)
    store.publish_final(request)
    with pytest.raises(OSError):
        m._progress_final_host_cleanup(entries(ledger))
    assert len(m.released) == 1 and ledger.get(request.snapshot_id)["state"] == "evicting"
    monkeypatch.setattr(ledger, "complete_host_eviction_rank", original)
    m._progress_host_evictions(entries(ledger))
    assert len(m.released) == 1 and ledger.get(request.snapshot_id)["state"] == "consumed"


def test_marker_generation_kind_and_broker_are_respected(tmp_path):
    ledger, store, (request,), (m,) = setup(tmp_path)
    store.publish_final(RequestGeneration(request.request_id, request.generation - 1))
    m._progress_final_host_cleanup(entries(ledger))
    assert not m.released
    store._publish(store.final_path(request), request, "tool")
    m._progress_final_host_cleanup(entries(ledger))
    assert not m.released
    store.publish_final(request)
    m.workset_broker = SimpleNamespace(eviction_blocker=lambda sid: "live lease")
    m._progress_final_host_cleanup(entries(ledger))
    assert not m.released
    m.workset_broker.eviction_blocker = lambda sid: None
    m._progress_final_host_cleanup(entries(ledger))
    assert len(m.released) == 1


def test_final_and_pressure_cannot_retarget_each_other(tmp_path):
    ledger, store, (request,), (m,) = setup(tmp_path)
    assert ledger.begin_host_eviction(request.snapshot_id, m.owner, tp_size=1, reason="pressure")
    assert not ledger.begin_host_eviction(request.snapshot_id, m.owner, tp_size=1,
                                           reason="application_final", terminal_on_release=True)
    m._progress_host_evictions(entries(ledger))
    assert ledger.get(request.snapshot_id)["state"] == "recompute_required"
    assert m._host_eviction_count == 1


def test_bounded_scan_eventually_reclaims_all_and_does_not_require_pressure(tmp_path):
    ledger, store, requests, (m,) = setup(tmp_path, count=150)
    for request in requests:
        store.publish_final(request)
    m._progress_final_host_cleanup(entries(ledger))
    assert len(m.released) == 128
    m._progress_final_host_cleanup(entries(ledger))
    assert len(m.released) == 150
    assert all(e['state'] == 'consumed' for e in entries(ledger).values())
    assert not m.notifications and m._host_eviction_count == 0


def test_final_before_host_offer_is_not_rejected(tmp_path):
    ledger, store, (request,), (m,) = setup(tmp_path)
    marker = store.publish_final(request)
    def delayed_offer(data):
        data[request.snapshot_id].update(
            created_at=marker['arrived_at'] + 10,
            tool_started_at=marker['arrived_at'] - 1,
        )
        return True, True
    ledger._mutate(delayed_offer, event_snapshot_id=request.snapshot_id)
    m._progress_final_host_cleanup(entries(ledger))
    assert len(m.released) == 1


def test_active_assignment_then_expiration(tmp_path):
    ledger, store, (request,), (m,) = setup(tmp_path)
    store.publish_final(request)
    def assign(data):
        data[request.snapshot_id].update(recovery_domain=0,
                                          recovery_assignment_expires_at=time.time()+60)
        return True, True
    ledger._mutate(assign, event_snapshot_id=request.snapshot_id)
    m._progress_final_host_cleanup(entries(ledger))
    assert not m.released
    def expire(data):
        data[request.snapshot_id]['recovery_assignment_expires_at'] = 0
        return True, True
    ledger._mutate(expire, event_snapshot_id=request.snapshot_id)
    m._progress_final_host_cleanup(entries(ledger))
    assert len(m.released) == 1


def test_no_control_directory_is_noop(tmp_path, monkeypatch):
    ledger, store, (request,), (m,) = setup(tmp_path)
    del m._final_host_claim_store
    monkeypatch.delenv('SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR', raising=False)
    monkeypatch.delenv('SGLANG_PD_P_READY_DIR', raising=False)
    m._progress_final_host_cleanup(entries(ledger))
    assert not m.released and not hasattr(m, '_final_host_claim_store')
