import pytest

from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.agentic_tp_events import (
    EventKey,
    TPEventClient,
    TPEventServer,
)
from sglang.srt.disaggregation.agentic_tp_socket_mailbox import SocketTPGroupMailbox
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    RequestGeneration,
    SnapshotManifest,
    SnapshotState,
)
from sglang.srt.disaggregation.agentic_tp_control import bind_direct_wire_mailboxes


def client(server, rank, group="P"):
    value = TPEventClient(
        server.address, run_id="run", token="secret", group=group, rank=rank, size=2
    )
    value.wait_ready()
    return value


def wait(value, predicate):
    with value._condition:
        assert value._condition.wait_for(predicate, 3)


def test_factory_uses_socket_without_touching_mailbox_directory(tmp_path, monkeypatch):
    import sglang.srt.disaggregation.agentic_tp_socket_mailbox as module

    server = TPEventServer("run", "secret")
    c = client(server, 0)
    monkeypatch.setenv("SGLANG_AGENTIC_TP_EVENT_ENDPOINT", "selected")
    monkeypatch.setattr(module, "get_tp_event_client", lambda rank, size: c)
    forbidden = tmp_path / "must-not-create"
    try:
        mailbox = TPGroupMailbox(
            "p2d-sender", tp_rank=0, tp_size=2, directory=str(forbidden)
        )
        assert isinstance(mailbox, SocketTPGroupMailbox)
        assert not forbidden.exists()
        mailbox.publish_local("request@123", int(KVPoll.WaitingForInput))
        c.flush()
        assert mailbox.local_status("request@123") == int(KVPoll.WaitingForInput)
    finally:
        c.close()
        server.close()


def test_retryable_snapshot_requires_explicit_identity_and_no_rebinding(monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_MULTINODE_ROLE", raising=False)
    server = TPEventServer("run", "secret")
    c = client(server, 0)
    mailbox = SocketTPGroupMailbox("d2p-host:0", tp_rank=0, tp_size=2, client=c)
    try:
        with pytest.raises(ValueError, match="explicit"):
            mailbox.publish_local("snapshot:3", 1)
        first = EventKey("snapshot:3", "claim:1")
        second = EventKey("snapshot:3", "claim:2")
        mailbox.bind_identity("legacy-callback", first)
        mailbox.publish_local("legacy-callback", 1)
        c.flush()
        mailbox.publish_local("legacy-callback", -1)
        c.flush()
        assert mailbox.local_status(first) == -1
        mailbox.publish_local("legacy-callback", 5)  # cleanup after failed I/O
        c.flush()
        assert mailbox.local_status(first) == 5
        with pytest.raises(ValueError, match="rebind"):
            mailbox.bind_identity("legacy-callback", second)
        mailbox.publish_local(second, 2)
        c.flush()
        assert mailbox.local_status(first) == 5
        assert mailbox.local_status(second) == 2
        mailbox.clear_group(first)
        c.flush()
        mailbox.publish_local(first, 4)
        c.flush()
        assert mailbox.local_status(first) is None
        assert mailbox.local_status(second) == 2
    finally:
        c.close()
        server.close()


def test_socket_mailbox_transfer_and_rollback_keep_all_shard_fences(monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_MULTINODE_ROLE", raising=False)
    server = TPEventServer("run", "secret")
    clients = [client(server, rank) for rank in range(2)]
    mailboxes = [
        SocketTPGroupMailbox("p2d-sender", tp_rank=rank, tp_size=2, client=c)
        for rank, c in enumerate(clients)
    ]
    key = "request@123"
    try:
        mailboxes[0].publish_local(key, KVPoll.Failed)
        mailboxes[1].publish_local(key, KVPoll.Transferring)
        for c in clients:
            c.flush()
        wait(
            clients[0],
            lambda: mailboxes[0].transfer_group_status(key)
            == (int(KVPoll.Transferring), True),
        )
        mailboxes[1].publish_local(key, KVPoll.Success)
        clients[1].flush()
        wait(
            clients[0],
            lambda: mailboxes[0].transfer_group_status(key)
            == (int(KVPoll.Failed), True),
        )
        for mailbox in mailboxes:
            mailbox.publish_local_rollback_complete(key)
        for c in clients:
            c.flush()
        wait(clients[0], lambda: mailboxes[0].rollback_group_complete(key))
        mailboxes[0].clear_group_rollback(key)
        clients[0].flush()
        assert not mailboxes[0].rollback_group_complete(key)
        assert mailboxes[0].transfer_group_status(key) == (int(KVPoll.Failed), True)
        with pytest.raises(ValueError, match="rank zero"):
            mailboxes[1].group_status(key)
    finally:
        for c in clients:
            c.close()
        server.close()


def test_prepare_is_one_pushed_command_with_exact_generation_attempt():
    from collections import deque
    import threading
    from types import SimpleNamespace
    from sglang.srt.managers.scheduler import Scheduler

    server = TPEventServer("run", "secret")
    clients = [client(server, rank) for rank in range(2)]
    boxes = [
        SocketTPGroupMailbox("d2p-direct", tp_rank=rank, tp_size=2, client=c)
        for rank, c in enumerate(clients)
    ]
    request = RequestGeneration("req", 2)
    manifest = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=8,
        byte_size=0,
        state=SnapshotState.DIRECT_READY,
        direct_room=123,
        direct_bootstrap_addr="127.0.0.1:1",
        token_digest="digest",
        kv_layout_hash="layout",
        tp_size=2,
    )
    try:
        bind_direct_wire_mailboxes(manifest, boxes[0])
        boxes[0].publish_prepare(request, {"prompt_token_count": 12}, manifest)
        clients[0].flush()
        key = EventKey(request.snapshot_id, "direct-room:123")
        wait(
            clients[1],
            lambda: clients[1].command("d2p-direct:prepare", key) is not None,
        )
        follower = SimpleNamespace(
            tp_rank=1,
            agentic_tp_direct_mailbox=boxes[1],
            agentic_early_direct_arrival_watcher=None,
            agentic_early_direct_admission_ids=set(),
            agentic_early_direct_receives={},
            agentic_early_direct_terminal={},
            agentic_early_direct_admission_queue=deque(),
        )
        Scheduler._agentic_collect_direct_arrivals(follower, threading.RLock())
        assert len(follower.agentic_early_direct_admission_queue) == 1
        queued_request, queued_payload, queued_manifest = (
            follower.agentic_early_direct_admission_queue[0]
        )
        assert queued_request == request and queued_manifest == manifest
        assert queued_payload == {"prompt_token_count": 12}
        assert boxes[1].drain_prepares() == []
        for box in boxes[:1]:
            commands = box.drain_prepares()
            assert len(commands) == 1
            identity, command_id, body = commands[0]
            assert identity == key and command_id == 1
            assert body["op"] == "PREPARE" and body["generation"] == 2
            box.ack_prepare(identity, command_id)
            assert box.drain_prepares() == []
        for c in clients:
            c.flush()
        wait(clients[0], lambda: clients[0].command_complete("d2p-direct:prepare", key))
        boxes[0].publish_prepare(request, {"prompt_token_count": 12}, manifest)
        clients[0].flush()
        assert boxes[1].drain_prepares() == []
    finally:
        for c in clients:
            c.close()
        server.close()
