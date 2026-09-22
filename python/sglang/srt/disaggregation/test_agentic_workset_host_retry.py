"""Actual Host preparation/retry boundary with the socket workset authority."""
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation.agentic_host_control import InMemoryHostStagingLedger
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, token_ids_digest
from sglang.srt.disaggregation.test_agentic_host_control import ready, SID, OWNER
from sglang.srt.disaggregation.test_agentic_host_async_prepare import fixture
from sglang.srt.disaggregation.test_agentic_workset_broker import cluster, admit
from sglang.srt.disaggregation.test_agentic_workset_runtime import eventually


@pytest.mark.parametrize("size", [2, 8])
def test_actual_host_retry_rearms_capacity_waiter_only_after_all_rank_host_fence(cluster, size):
    authority, brokers, runtimes, _ = cluster(size, pages=5)
    # Occupied by another live request; the restore fits in principle but must
    # stay a capacity intent with no physical lease until the blocker departs.
    admit(brokers, runtimes, sid="blocker", parent=0, prompt=20, owner="fresh:blocker")
    host = InMemoryHostStagingLedger()
    ready(host, size)
    parent = RequestGeneration("request", 0)
    owner = brokers[0].slow_owner(SID, "child")
    managers = []
    for rank, broker in enumerate(brokers):
        m, _, _ = fixture(tp_size=size)
        m.tp_rank, m.owner, m.workset_broker = rank, OWNER, broker
        m.ledger, m._ledger_entries_cache = host, None
        m.host_ready = {SID: dict(snapshot=NS(_materialized=object()), loading=False,
            network_host=True, offer=dict(token_count=16, byte_size=128,
                                         token_digest=token_ids_digest(list(range(16)))))}
        m._remote_host_bridge = NS(cancel_unstarted=lambda *args, **kwargs: True)
        req = NS(rid="child", origin_input_ids=list(range(20)))
        assert m._prepare_host_restore(req, parent) is True
        assert broker._restore_epochs[(SID, owner)] == 1
        assert broker.get(SID) is None and SID in broker._intents
        managers.append((m, req))
    previous = brokers[0]._requests[(SID, owner)]["future"]
    assert not previous.done() and authority.counts.free_pages == 0
    for broker in brokers:
        assert broker.cancel_unstarted(SID, owner=owner)
    eventually(previous.done)
    eventually(lambda: not brokers[0].owner_has_unretired_work(SID, owner=owner))
    assert (SID, owner) in brokers[-1]._pending_closed_owners
    assert host.request_d2p_retry(SID, OWNER, reason="test_prestart_retry")
    frozen = host.get(SID)
    for m, req in managers[:-1]:
        assert m._complete_prestart_remote_retry(req, parent, frozen)
    assert host.get(SID)["state"] == "retry_pending"
    # No last-peer drain yet: old claim/epoch cannot open the closed waiter.
    for m, req in managers:
        assert m._prepare_host_restore(req, parent) is True
        assert SID not in m.workset_broker._intents
        assert m.workset_broker._restore_epochs[(SID, owner)] == 1
    last, req = managers[-1]
    assert last._complete_prestart_remote_retry(req, parent, frozen)
    assert host.get(SID)["state"] == "host_ready"
    for m, req in managers:
        assert m._prepare_host_restore(req, parent) is True
        broker = m.workset_broker
        assert broker._restore_epochs[(SID, owner)] == 2
        assert (SID, owner) not in broker._pending_closed_owners
        assert SID in broker._intents and broker.get(SID) is None
    replacement = brokers[0]._requests[(SID, owner)]["future"]
    assert replacement is not previous and not replacement.done()
    assert authority.counts.free_pages == 0  # Retry never invented space.
    for broker in brokers:
        broker.cancel_unstarted(SID, owner=owner)
    eventually(replacement.done)
