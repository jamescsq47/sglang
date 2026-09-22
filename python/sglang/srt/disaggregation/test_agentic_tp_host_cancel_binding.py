"""HTTP cancellation must retire a bound/handed, not-yet-admitted workset."""
import threading
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    CUSTOM_GENERATION, CUSTOM_PARENT_GENERATION, CUSTOM_REQUEST_ID, RequestGeneration,
)
from sglang.srt.disaggregation.test_agentic_direct_descriptors import Allocator
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker, Scheduler


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("handed", [False, True])
def test_http_cancel_binding_retires_suffix_once_at_native_tp_boundary(tp_size, handed):
    parent = RequestGeneration("cancel-bound", 1)
    sid = parent.snapshot_id
    ranks = []
    for rank in range(tp_size):
        allocator = Allocator(32)
        broker = AgenticPWorksetLeaseBroker(4)
        owner = broker.slow_owner(sid, "r")
        broker.request(sid, 8, 12, owner=owner)
        broker.prepare_tp_control(1)
        broker.service(allocator)
        lease = broker.get(sid)
        assert broker.begin_bind(sid, lease)
        broker.commit_parent_bound(sid, lease)
        pin_calls, parent_releases = [], []
        def release_parent(req, *, committed_len, _defer_if_blocked, a=allocator,
                           indices=lease.parent_indices, calls=parent_releases):
            assert committed_len == 8 and not _defer_if_blocked
            calls.append(committed_len)
            a.free(indices)
        tree = NS(dec_lock_ref=pin_calls.append, release_agentic_request_cache=release_parent,
                  supports_mamba=lambda: False)
        manager = object.__new__(AgenticPHostStagingManager)
        manager.tp_size, manager.tp_rank = tp_size, rank
        manager.owner = "p"
        manager.ledger = NS(is_event_control=True)
        manager._control_wakeup = threading.Event()
        manager.workset_broker, manager.tree_cache = broker, tree
        manager._h2d_lane_reservations = {sid: 0}
        manager._h2d_resident_reservations = {sid}
        req = NS(rid="r", req_pool_idx=None, origin_input_ids=list(range(12)),
                 sampling_params=NS(custom_params={
            CUSTOM_REQUEST_ID: parent.request_id, CUSTOM_GENERATION: 2,
            CUSTOM_PARENT_GENERATION: 1}),
            _agentic_host_workset_lease=lease, _agentic_host_rank_loaded=True,
            _agentic_host_rank_token_count=8, _agentic_kv_host_pin_node="parent-pin",
            _agentic_host_remote_read_epoch=4)
        if handed:
            broker.handoff_to_req(sid, req, lease)
            assert req._agentic_workset_backed
            assert lease.state == "handed"
        scheduler = NS(tp_size=tp_size, tree_cache=tree,
                       agentic_host_staging_manager=manager,
                       agentic_p_workset_broker=broker,
                       agentic_tp_host_active_requests={sid: parent})
        Scheduler._agentic_abort_cleanup(scheduler, req)
        assert pin_calls == ["parent-pin"] and parent_releases == [8]
        assert lease.state == "retire_ready"
        assert allocator.available_size() == 28  # suffix still group-owned
        assert not manager._h2d_lane_reservations
        assert not manager._h2d_resident_reservations
        assert sid in manager._pending_host_abort_requests  # worker, not scheduler RPC
        assert not hasattr(req, "_agentic_host_workset_lease")
        assert not hasattr(req, "_agentic_p_workset_lease")
        Scheduler._agentic_abort_cleanup(scheduler, req)
        assert parent_releases == [8]
        broker.service(allocator)
        assert allocator.available_size() == 28
        ranks.append((broker, allocator))
    assert all(broker.tp_retire_ready(sid) for broker, _ in ranks)
    for broker, allocator in ranks:
        assert broker.commit_tp_retire(sid)
        broker.service(allocator)
        assert broker.get(sid) is None
        assert allocator.available_size() == 32
        assert allocator.free_indices.unique().numel() == 32
