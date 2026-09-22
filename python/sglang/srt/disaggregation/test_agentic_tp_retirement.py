"""Ordered workset history reclamation; generic attempt tombstones unchanged."""

import json

import pytest

from sglang.srt.disaggregation.agentic_tp_events import (
    EventKey, TPEventClient, TPEventServer, _RetiredVersions,
)


BASE = "workset-grant"


def identities(version, incarnation="pool"):
    snapshot, attempt = f"request-{version}:0", f"attempt-{version}"
    fields = [incarnation, attempt, version]
    grant = [BASE, snapshot, json.dumps(fields)]
    final = [BASE + ":decisions", snapshot, json.dumps(fields + ["decision", version * 3])]
    plan = dict(incarnation=incarnation, snapshot_id=snapshot, attempt_id=attempt,
                version=version, sequence=version * 3 - 2)
    free = dict(plan, operation="free", sequence=version * 3)
    return grant, final, plan, free


def state(size=2):
    return dict(size=size, entries={}, workset_entries={}, retired_worksets={})


def install(server, target, version, ack_last=True):
    grant, final, plan, free = identities(version)
    for identity, value in ((grant, plan), (final, free)):
        server._apply(target, 0, dict(op="command", identity=identity, value=value, command_id=1))
        for rank in range(target["size"]):
            if identity == final and rank == target["size"] - 1 and not ack_last:
                continue
            server._apply(target, rank, dict(op="command_ack", identity=identity, command_id=1))
    for suffix, identity in ((":prepared", grant), (":fenced", final)):
        key = [BASE + suffix, *identity[1:]]
        server._apply(target, 0, dict(op="report", identity=key, status=1))
    return dict(op="retire_workset", identity=grant, final_identity=final)


def test_intervals_preserve_live_holes_and_compact():
    retired = _RetiredVersions()
    for version in (3, 1, 5):
        retired.add(version)
    assert retired.ranges == [(1, 1), (3, 3), (5, 5)]
    assert not retired.contains(2) and not retired.contains(4)
    retired.add(2)
    retired.add(4)
    retired.add(3)
    assert retired.ranges == [(1, 5)]
    assert retired.contains(1) and retired.contains(5) and not retired.contains(6)


def test_more_than_default_historical_entry_limit_with_small_live_bound():
    server = TPEventServer("run", "token", max_entries=8)
    target = state()
    try:
        for version in range(1, 25002):  # 100004 records, above old run-long cap.
            message = install(server, target, version)
            server._retire_workset(target, 0, message)
            assert server._entry_count == 0
        assert not target["entries"] and not target["workset_entries"]
        assert target["retired_worksets"][(BASE, "pool")].ranges == [(1, 25001)]
        old = identities(1)[0]
        assert server._apply(target, 1, dict(op="report", identity=old, status=99)) is None
        assert server._entry_count == 0
    finally:
        server.close()


def test_retirement_requires_last_rank_free_ack_and_preserves_other_live_lease():
    server = TPEventServer("run", "token", max_entries=16)
    target = state(size=8)
    try:
        old = install(server, target, 1, ack_last=False)
        current = install(server, target, 2)
        with pytest.raises(ValueError, match="all-rank exact FREE"):
            server._retire_workset(target, 0, old)
        assert server._entry_count == 8
        server._retire_workset(target, 0, current)
        assert server._entry_count == 4
        assert tuple(old["identity"]) in target["entries"]
        server._apply(target, 7, dict(op="command_ack", identity=old["final_identity"], command_id=1))
        server._retire_workset(target, 0, old)
        assert target["retired_worksets"][(BASE, "pool")].ranges == [(1, 2)]
        assert server._entry_count == 0
    finally:
        server.close()


def test_retirement_rejects_wrong_attempt_follower_and_observed_namespace():
    server = TPEventServer("run", "token")
    target = state()
    try:
        message = install(server, target, 1)
        with pytest.raises(ValueError, match="rank zero"):
            server._retire_workset(target, 1, message)
        bad = dict(message, identity=[BASE, "other:0", message["identity"][2]])
        with pytest.raises(ValueError, match="exact final decision"):
            server._retire_workset(target, 0, bad)
        server._receipt_observers[("P", BASE)] = frozenset({"D"})
        with pytest.raises(ValueError, match="observed receipt"):
            server._retire_workset(target, 0, message)
        assert server._entry_count == 4
    finally:
        server.close()


def test_client_and_server_prune_history_and_refuse_late_reports():
    server = TPEventServer("run", "token", max_entries=8)
    clients = [TPEventClient(server.address, run_id="run", token="token", group="P", rank=r, size=2)
               for r in range(2)]
    try:
        for client in clients:
            client.wait_ready()
            client.subscribe_updates(BASE)
            client.subscribe_updates(BASE + ":decisions")
        leader = clients[0]
        for version in range(1, 20):
            grant, final, plan, free = identities(version)
            grant_key, final_key = EventKey(*grant[1:]), EventKey(*final[1:])
            for namespace, key, wire in ((BASE, grant_key, plan), (BASE + ":decisions", final_key, free)):
                leader.publish_command(namespace, key, wire, command_id=1)
                leader.flush()
                for client in clients:
                    with client._condition:
                        assert client._condition.wait_for(lambda: client.command(namespace, key) is not None, 5)
                    client.next_command(namespace, key)
                    client.ack_command(namespace, key, 1)
                    client.flush()
                with leader._condition:
                    assert leader._condition.wait_for(lambda: leader.command_complete(namespace, key), 5)
            for client in clients:
                client.report(BASE + ":prepared", grant_key, 1)
                client.flush()
            leader.retire_workset(BASE, grant_key, final_key)
            leader.flush()
            for client in clients:
                with client._condition:
                    assert client._condition.wait_for(lambda: client._is_retired_workset(grant), 5)
                client.report(BASE, grant_key, 99)
                client.publish_receipt(BASE, grant_key, 1) if client.rank == 0 else None
                client.flush()
                assert not client._entries and not client._reported
                assert not client._command_ids and not client._delivered_commands
                assert not client._published_receipts and not client._workset_cache_keys
                assert client._command_inbox_size == client._update_inbox_size == 0
            assert server._entry_count == 0
        assert clients[0]._retired_worksets[(BASE, "pool")].ranges == [(1, 19)]
        stats = server.stats()
        assert stats["entry_count"] == stats["observed_count"] == 0
        assert stats["max_entries"] == 8
        assert stats["groups"]["P"] == {
            "entries": 0, "workset_versions": 0, "retired_intervals": 1,
            "retired_versions": 19, "failed": None,
        }
    finally:
        for client in clients:
            client.close()
        server.close()
