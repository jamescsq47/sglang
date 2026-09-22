"""Group-local progress must not touch NFS; cross-engine receipts still must."""
from pathlib import Path

import pytest

from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.base import KVPoll


LOCAL = (
    "d2p-direct", "d2p-direct-abort-p", "d2p-host:0",
    "p-workset-retire", "p2d-sender", "p2d-cleanup", "p2d-admission",
)


def mailbox(tmp_path, namespace, engine="p0", rank=0, **kwargs):
    return TPGroupMailbox(namespace, tp_rank=rank, tp_size=2,
                          directory=str(tmp_path / "shared"),
                          group_local_directory=str(tmp_path / engine), **kwargs)


@pytest.mark.parametrize("namespace", LOCAL)
def test_group_reports_never_access_shared_fs(tmp_path, monkeypatch, namespace):
    original = Path.stat

    def stat(path, *args, **kwargs):
        if path.is_relative_to(tmp_path / "shared"):
            raise AssertionError("group-local report touched the shared filesystem")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    leader = mailbox(tmp_path, namespace)
    follower = mailbox(tmp_path, namespace, rank=1)
    other_engine = mailbox(tmp_path, namespace, engine="p1")
    leader.publish_local_progress("generation:1", 1)
    assert leader.group_status("generation:1") is None
    follower.publish_local_progress("generation:1", 2)
    assert leader.group_status("generation:1") == 1
    assert other_engine.group_status("generation:1") is None
    follower.publish_local_progress("generation:1", -1)
    follower.publish_local_progress("generation:1", 3)
    assert leader.group_status("generation:1") == -1
    leader.clear_group("generation:1")
    assert follower.group_status("generation:1") is None


def test_cross_engine_receiver_receipt_and_cleanup_stay_shared(tmp_path):
    p = mailbox(tmp_path, "p2d-receiver", engine="p")
    d0 = mailbox(tmp_path, "p2d-receiver", engine="d")
    d1 = mailbox(tmp_path, "p2d-receiver", engine="d", rank=1)
    assert p.directory == d0.directory == d1.directory
    d0.publish_local("generation:1", int(KVPoll.Success))
    d1.publish_local("generation:1", int(KVPoll.Success))
    assert d0.group_status("generation:1") == int(KVPoll.Success)
    d0.publish_receipt("generation:1", int(KVPoll.Success))
    assert p.receipt("generation:1") == int(KVPoll.Success)
    p.clear_group("generation:1")
    assert d0.receipt("generation:1") is None
    assert d1.local_status("generation:1") is None


def test_unknown_namespace_remains_shared(tmp_path):
    assert mailbox(tmp_path, "future-protocol").directory.parent == tmp_path / "shared"


def test_multi_host_tp_rejects_local_mailbox(tmp_path):
    with pytest.raises(ValueError, match="entire TP group"):
        mailbox(tmp_path, "p2d-sender", nnodes=2)


def test_no_override_preserves_existing_directory(tmp_path):
    old = TPGroupMailbox("p2d-sender", tp_rank=0, tp_size=2, directory=str(tmp_path))
    assert old.directory.parent == tmp_path
