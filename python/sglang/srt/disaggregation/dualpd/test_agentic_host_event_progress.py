"""CPU regressions for bounded, opt-in TP1 Host-event refill."""
import queue
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager as Manager
from sglang.srt.disaggregation.test_agentic_h2d_decoupling import (
    manager, prepared_manager_and_scheduler,
)
from sglang.srt.managers.scheduler import Scheduler


def enable(m):
    m.h2d_event_progress = True
    m._scheduler_events = queue.SimpleQueue()


def test_event_drain_bounded_and_preserves_order():
    m = manager(); enable(m)
    for i in range(70):
        m._notify_scheduler('host_ready', str(i))
    assert m.drain_scheduler_events(64) == tuple(('host_ready', str(i)) for i in range(64))
    assert m.drain_scheduler_events() == tuple(('host_ready', str(i)) for i in range(64,70))
    assert m.drain_scheduler_events(0) == ()


@pytest.mark.parametrize('tp,decoupled,enabled', [(2,True,True),(1,False,True),(1,True,False)])
def test_default_dense_or_tp_does_not_touch_queues(tp,decoupled,enabled):
    s = object.__new__(Scheduler); s.tp_size = tp
    s.agentic_host_staging_manager = NS(h2d_decoupled=decoupled,h2d_event_progress=enabled)
    s._agentic_refill_host_events()


def test_empty_waiters_and_duplicate_old_events_are_harmless():
    s = object.__new__(Scheduler); s.tp_size = 1
    m = manager(); enable(m); s.agentic_host_staging_manager = m
    s.agentic_kv_waiting_queue = []
    m._notify_scheduler('hbm_ready','gone:0'); m._notify_scheduler('hbm_ready','gone:0')
    s._agentic_refill_host_events()
    assert m.drain_scheduler_events() == ()


def test_real_gate_and_broker_progress_in_same_boundary_without_stealing_fifth_slot(monkeypatch):
    m,s,requests = prepared_manager_and_scheduler(monkeypatch); enable(m)
    s._agentic_should_defer = lambda r,*a,allow_start_io=True: m.gate_request(
        r,r.parent,allow_prepare=allow_start_io,allow_start=allow_start_io)
    m._notify_scheduler('host_ready','0:0')
    s._agentic_refill_host_events()
    assert set(m.loads) == {'0','1','2','3'}
    assert all(x['start_allowed'] for x in m.loads.values())
    assert '4:0' not in m.h2d_selected_snapshots()
    assert len(s.agentic_kv_waiting_queue) == 5


def test_extra_pass_cannot_reset_native_admission_budget(monkeypatch):
    m,s,requests = prepared_manager_and_scheduler(monkeypatch); enable(m)
    # Remove fixture-owned lanes: all requests are now ordinary metadata.
    for sid in list(m.h2d_selected_snapshots()):m._release_h2d_lane(sid)
    m.workset_broker._intents.clear()
    s._agentic_admission_used_this_tick = 8
    s._agentic_should_defer = lambda *a,**k: pytest.fail('exhausted budget admitted fresh work')
    m._notify_scheduler('host_ready','0:0')
    s._agentic_refill_host_events()
    assert not m.loads
    assert s._agentic_admission_used_this_tick == 8


def test_existing_owned_progress_does_not_need_an_event(monkeypatch):
    m,s,requests = prepared_manager_and_scheduler(monkeypatch); enable(m)
    s._agentic_service_p_workset_leases()
    assert set(m.loads) == {'0','1','2','3'}  # level-trigger backstop retained


def test_fallback_false_does_not_get_owned_completion_priority(monkeypatch):
    m,s,requests = prepared_manager_and_scheduler(monkeypatch); enable(m)
    s.waiting_queue = []
    s._agentic_publish_p_scheduled = lambda r: None
    s._add_request_to_queue = s.waiting_queue.append
    m.gate_request = lambda *a,**k: False
    s._agentic_progress_granted_slow(owned_only=True)
    assert not s.waiting_queue
    assert len(s.agentic_kv_waiting_queue) == 5


def test_one_refill_does_not_spin_on_new_events():
    s=object.__new__(Scheduler);s.tp_size=1
    m=manager();enable(m);s.agentic_host_staging_manager=m
    s.agentic_kv_waiting_queue=[(NS(),0)]
    calls=[]
    def drain(**kwargs):
        assert kwargs == {'continue_tick':True}
        calls.append('fifo');m._notify_scheduler('host_ready','later:0')
    s._drain_agentic_kv_waiting_queue=drain
    s.token_to_kv_pool_allocator=object()
    s.agentic_p_workset_broker=NS(service=lambda *a,**k:calls.append(('service',k['reserve_tokens'])))
    s._agentic_progress_granted_slow=lambda **k:calls.append(('progress',k['owned_only']))
    m._notify_scheduler('hbm_ready','ready:0')
    s._agentic_refill_host_events(reserve_tokens=64)
    assert calls == ['fifo',('service',64),('progress',True)]
    assert m.drain_scheduler_events() == (('host_ready','later:0'),)


def test_safe_boundary_fallback_cannot_bypass_exhausted_budget(monkeypatch):
    m,s,requests = prepared_manager_and_scheduler(monkeypatch); enable(m)
    s.waiting_queue = []
    s._agentic_admission_used_this_tick = 8
    s._agentic_publish_p_scheduled = lambda r: None
    s._add_request_to_queue = s.waiting_queue.append
    def terminal_fallback(req, parent, **kwargs):
        req._agentic_kv_gate_complete = True
        m._release_h2d_lane(parent.snapshot_id)
        return False
    m.gate_request = terminal_fallback
    s._agentic_should_defer = lambda *a,**k: pytest.fail('fallback bypassed budget')
    m._notify_scheduler('hbm_ready','0:0')
    s._agentic_service_p_workset_leases()
    assert not s.waiting_queue
    assert len(s.agentic_kv_waiting_queue) == 5
