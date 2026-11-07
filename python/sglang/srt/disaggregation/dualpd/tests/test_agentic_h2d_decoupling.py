"""CPU ownership and scheduler regressions for opt-in TP1 hybrid H2D staging."""

import ast
import inspect
import textwrap
import threading
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager as Manager
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.agentic_kv_lifecycle import token_ids_digest
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers import scheduler as sched_module
from sglang.srt.managers.scheduler import Scheduler, AgenticPWorksetLeaseBroker


def manager(enabled=True, lanes=1):
    m = object.__new__(Manager)
    m.h2d_decoupled = enabled
    m.max_h2d_inflight = lanes
    m._state_lock = threading.RLock()
    m._control_wakeup = threading.Event()
    m._h2d_lane_reservations = {}
    m._h2d_resident_reservations = set()
    m.loads = {}
    return m


def completed_load(m, name):
    parent = RequestGeneration(name, 0)
    assert m._reserve_h2d_lane(parent.snapshot_id) is not None
    load = dict(rid=name, request_generation=parent, h2d_copy_complete=True,
                io_quiesced=True, event=None, copy_refs=None,
                record=object(), workset_lease=object())
    m.loads[name] = load
    return load


@pytest.mark.parametrize('missing', ['h2d_copy_complete', 'io_quiesced', 'event', 'copy_refs', 'prefetch_future'])
def test_cannot_recycle_before_composite_fence_and_cpu_worker_retire(missing):
    m = manager()
    load = completed_load(m, 'old')
    load[missing] = False if missing in {'h2d_copy_complete', 'io_quiesced'} else object()
    m._release_quiesced_h2d_lane(load)
    assert m.h2d_physical_occupancy() == 1
    assert not load.get('transport_lane_released')


def test_recycle_preserves_host_and_workset_and_late_cancel_cannot_free_new_lane():
    m = manager()
    old = completed_load(m, 'old')
    host, lease = old['record'], old['workset_lease']
    m._release_quiesced_h2d_lane(old)
    m._release_quiesced_h2d_lane(old)
    assert m.h2d_physical_occupancy() == 0
    assert m.h2d_selected_snapshots() == {'old:0'}
    new = completed_load(m, 'new')
    # Original handoff/abort cleanup is keyed by snapshot, not physical lane id.
    m._release_h2d_lane('old:0')
    assert m._h2d_lane_reservations == {'new:0': 0}
    assert new['record'] is not host
    assert old['record'] is host and old['workset_lease'] is lease
    assert m.h2d_selected_snapshots() == {'new:0'}


def test_resident_stage_bounded_even_when_all_dmas_finish_but_binding_stalls():
    m = manager(lanes=2)
    for i in range(4):
        load = completed_load(m, str(i))
        m._release_quiesced_h2d_lane(load)
    assert m.h2d_physical_occupancy() == 0
    assert m._reserve_h2d_lane('overflow:0') is None
    m._release_h2d_lane('0:0')
    assert m._reserve_h2d_lane('overflow:0') == 0


def test_disabled_keeps_lane_until_original_handoff():
    m = manager(enabled=False)
    load = completed_load(m, 'old')
    m._release_quiesced_h2d_lane(load)
    assert m.h2d_physical_occupancy() == 1
    assert m._reserve_h2d_lane('new:0') is None


def test_real_progress_recycles_only_final_composite_event_even_if_ack_fails():
    m = manager()
    load = completed_load(m, 'old')
    load.update(h2d_copy_complete=False, io_quiesced=False, start_allowed=True,
                device_indices=[1], chunk_end=1, offset=0, io_attempt='attempt',
                gpu_elapsed_ms=0.0, record={'snapshot': object()})
    done = [False]
    load['event'] = NS(query=lambda: done[0], synchronize=lambda: None)
    m._h2d_poisoned = False
    m.tp_size, m.tp_rank, m.owner = 1, 0, 'p'
    m.ledger = NS(get=lambda _: {'state': 'h2d_loading'},
                  complete_d2p_host_load_rank=lambda *a, **k: False)
    m.workset_broker = NS(mark_io_quiesced=lambda *a: True)
    m._progress_h2d_loads()
    assert m.h2d_physical_occupancy() == 1
    done[0] = True
    m._progress_h2d_loads()
    assert m.h2d_physical_occupancy() == 0
    assert load['io_quiesced'] and not load.get('io_complete')
    assert 'old:0' in m.h2d_selected_snapshots()
    assert m.loads['old'] is load  # Failed ACK cannot discard Host/workset.


def test_same_boundary_services_only_already_selected_and_does_not_duplicate(monkeypatch):
    m = manager()
    m._h2d_resident_reservations = {'selected:0', 'ready:0'}
    s = object.__new__(Scheduler)
    s.agentic_host_staging_manager = m
    fresh, selected, ready = [NS(rid=k, parent=RequestGeneration(k, 0))
                              for k in ('fresh', 'selected', 'ready')]
    s.agentic_kv_waiting_queue = [(fresh, 0), (selected, 1), (ready, 2)]
    s.waiting_queue = []
    s._agentic_publish_p_scheduled = lambda req: None
    s._add_request_to_queue = s.waiting_queue.append
    monkeypatch.setattr(sched_module.AgenticRequestMetadata, 'from_req', lambda r: NS(parent=r.parent))
    calls = []
    def gate(req, parent):
        calls.append(req.rid)
        if req is ready:
            m._release_h2d_lane(parent.snapshot_id)
            return False
        return True  # Already selected grant starts, still in I/O.
    m.gate_request = gate
    s._agentic_progress_granted_slow()
    assert calls == ['selected', 'ready']
    assert s.waiting_queue == [ready]
    s._agentic_progress_granted_slow()
    assert s.waiting_queue == [ready]
    assert [r.rid for r, _ in s.agentic_kv_waiting_queue] == ['fresh', 'selected']


def test_default_scheduler_path_no_additional_work():
    s = object.__new__(Scheduler)
    s.agentic_host_staging_manager = manager(enabled=False)
    s._agentic_progress_granted_slow()  # No queue required/accessed.


@pytest.mark.parametrize('error', [sched_module.SnapshotNotReadyError, sched_module.SnapshotLifecycleError])
def test_transient_gate_failure_preserves_waiter_and_progresses_other_selected(monkeypatch, error):
    m = manager()
    m._h2d_resident_reservations = {'retry:0', 'ready:0'}
    retry, ready = [NS(rid=k, parent=RequestGeneration(k, 0)) for k in ('retry', 'ready')]
    s = object.__new__(Scheduler)
    s.agentic_host_staging_manager = m
    s.agentic_kv_waiting_queue = [(retry, 0), (ready, 1)]
    s.waiting_queue = []
    s._agentic_publish_p_scheduled = lambda req: None
    s._add_request_to_queue = s.waiting_queue.append
    monkeypatch.setattr(sched_module.AgenticRequestMetadata, 'from_req', lambda r: NS(parent=r.parent))
    def gate(req, parent):
        if req is retry:
            raise error('retryable ledger race')
        m._release_h2d_lane(parent.snapshot_id)
        return False
    m.gate_request = gate
    s._agentic_progress_granted_slow()
    assert s.agentic_kv_waiting_queue == [(retry, 0)]
    assert s.waiting_queue == [ready]
    assert m.h2d_selected_snapshots() == {'retry:0'}


def test_mode_initialization_excludes_dense_tp_and_non_request_owned(monkeypatch):
    # Execute only the pure configuration expression; no CUDA manager init.
    init = ast.parse(textwrap.dedent(inspect.getsource(Manager.__init__)))
    assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == 'h2d_decoupled' for t in n.targets))
    expression = compile(ast.Expression(assignment.value), '<mode>', 'eval')
    import os
    monkeypatch.setenv('SGLANG_AGENTIC_KV_P_H2D_DECOUPLED', 'true')
    for tp, hybrid, owned in [(2, True, True), (1, False, True), (1, True, False), (1, True, True)]:
        assert eval(expression, dict(os=os, self=NS(tp_size=tp),
                    tree_cache=NS(supports_mamba=lambda: hybrid),
                    request_owned_mamba_enabled=lambda: owned)) == (tp == 1 and hybrid and owned)


def test_both_slow_and_direct_preserve_index_readiness_sync(monkeypatch):
    class Alloc:
        def alloc(self, n): return torch.arange(n)
        def free(self, x): pass
    calls = []
    original = sched_module.kv_to_page_indices
    monkeypatch.setattr(sched_module, 'kv_to_page_indices', lambda *a: calls.append(True) or original(*a))
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.request('slow:0', 4, 8, owner=broker.slow_owner('slow:0', 'rid'))
    broker.request('direct:0', 4, 8, owner=broker.direct_owner('direct:0'))
    broker.service(Alloc())
    assert len(calls) == 2
    assert broker.get('slow:0').parent_page_indices.size == 1
    assert broker.get('direct:0').parent_page_indices.size == 1


def prepared_manager_and_scheduler(monkeypatch):
    class Alloc:
        def alloc(self, n): return torch.arange(n)
        def free(self, x): pass
    m = manager(lanes=4)
    m.active, m.aborting, m.host_ready = {}, {}, {}
    m._ledger_entries_cache = {}
    m.tp_rank, m.tp_size, m.arena_domain, m.owner = 0, 1, 0, 'p:test'
    m.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
    m.ledger = NS(get=lambda sid: m._ledger_entries_cache.get(sid),
                  claim_d2p_recovery_rank=lambda *a, **k: True,
                  attach_d2p_recovery_lease_rank=lambda *a, **k: True,
                  begin_host_load_rank=lambda *a, **k: True)
    requests = []
    for i in range(5):
        parent = RequestGeneration(str(i), 0)
        req = NS(rid=str(i), parent=parent, origin_input_ids=[11, 22], _agentic_kv_queue_class='slow')
        requests.append(req)
        m.host_ready[parent.snapshot_id] = dict(snapshot=NS(_materialized=object()),
            offer=dict(token_count=1, token_digest=token_ids_digest([11]), byte_size=128), loading=False)
        m._ledger_entries_cache[parent.snapshot_id] = dict(state='host_ready', p_owner=m.owner)
        assert m.gate_request(req, parent) is True
    assert m.h2d_physical_occupancy() == 4 and not m.loads
    s = object.__new__(Scheduler)
    s.agentic_host_staging_manager = m
    s.agentic_p_workset_broker = m.workset_broker
    s.token_to_kv_pool_allocator = Alloc()
    s.agentic_kv_waiting_queue = [(r, float(i)) for i, r in enumerate(requests)]
    s.tp_size = 1
    s.running_batch = NS(batch_is_full=True)
    s.process_prefill_chunk = lambda: None
    s._should_throttle_p_ready_compute_ahead = lambda: False
    s.get_new_batch_prefill = lambda: None
    s.maybe_prepare_mlp_sync_batch = lambda batch: batch
    monkeypatch.setattr(sched_module.AgenticRequestMetadata, 'from_req', lambda r: NS(parent=r.parent))
    return m, s, requests


def test_real_pd_entry_allocates_and_starts_four_preselected_without_another_tick(monkeypatch):
    m, s, requests = prepared_manager_and_scheduler(monkeypatch)
    # Invoke the actual disaggregated entry, real broker and real gate_request.
    # R10's incorrect generic-only hook fails here with 0 loads/4 stuck lanes.
    SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run(s)
    assert set(m.loads) == {'0', '1', '2', '3'}
    assert all(load['start_allowed'] for load in m.loads.values())
    assert m.h2d_physical_occupancy() == 4
    assert '4:0' not in m.h2d_selected_snapshots()
    assert len(s.agentic_kv_waiting_queue) == 5


def test_full_lane_cap_does_not_block_existing_prestart_grants_in_normal_drain(monkeypatch):
    m, s, requests = prepared_manager_and_scheduler(monkeypatch)
    # Even before the safe-boundary shortcut, selected intents must not be
    # mistaken for NEW I/O. A free new-slot count of zero cannot block owners.
    m.workset_broker.service(s.token_to_kv_pool_allocator)
    s._agentic_should_defer = lambda r, *a, allow_start_io=True: m.gate_request(
        r, r.parent, allow_prepare=allow_start_io, allow_start=allow_start_io)
    Scheduler._drain_agentic_kv_waiting_queue(s)
    assert set(m.loads) == {'0', '1', '2', '3'}
    assert '4:0' not in m.h2d_selected_snapshots()
