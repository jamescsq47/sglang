from collections import defaultdict
from types import SimpleNamespace
import threading

import pytest

from sglang.srt.disaggregation.agentic_host_restore_controller import AgenticHostRestoreController
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration


class Bus:
    def __init__(self, size):
        self.size = size
        self.inboxes = defaultdict(list)
        self.commands, self.acks, self.reports = {}, defaultdict(set), defaultdict(dict)

    def client(self, rank):
        bus = self
        class Client:
            def publish_command(self, ns, key, payload, command_id):
                assert rank == 0
                if (ns, key) in bus.commands:
                    assert len(bus.acks[ns, key]) == bus.size
                bus.commands[ns, key] = command_id
                bus.acks[ns, key] = set()
                for target in range(bus.size):
                    bus.inboxes[target, ns].append((key, command_id, dict(payload)))

            def drain_commands(self, ns):
                result = bus.inboxes[rank, ns]
                bus.inboxes[rank, ns] = []
                return result

            def ack_command(self, ns, key, command_id):
                assert bus.commands[ns, key] == command_id
                bus.acks[ns, key].add(rank)

            def command_complete(self, ns, key):
                return len(bus.acks[ns, key]) == bus.size

            def clear(self, ns, key):
                bus.reports.pop((ns, key), None)
                bus.commands.pop((ns, key), None)
                bus.acks.pop((ns, key), None)

            def report_state(self, ns, key, status):
                bus.reports[ns, key][rank] = status

            def group_status(self, ns, key):
                values = bus.reports[ns, key]
                return min(values.values()) if len(values) == bus.size else None
        return Client()


def cluster(size=2, lanes=2):
    bus = Bus(size)
    controllers, managers = [], []
    for rank in range(size):
        client = bus.client(rank)
        manager = SimpleNamespace(
            workset_broker=SimpleNamespace(controller_mode=True),
            ledger=SimpleNamespace(is_event_control=True),
            max_h2d_inflight=lanes, h2d_lane_overlap=True,
            _control_wakeup=threading.Event(), ready=set(), terminal={}, queued=[],
            aborts=[], quiescent=False, commits=[],
        )
        manager.snapshot_ready = lambda parent, m=manager: parent.snapshot_id in m.ready
        manager.terminal_restore_reason = lambda parent, m=manager: m.terminal.get(parent.snapshot_id)
        manager.register_tp_host_progress = lambda *args: None
        manager._queue_host_prepare = lambda view, parent, m=manager: m.queued.append((view, parent))
        manager.abort_request = lambda rid, parent, m=manager: m.aborts.append((rid, parent))
        manager.tp_host_control_quiescent = lambda *args, m=manager: m.quiescent
        def commit(parent, m=manager):
            m.commits.append(parent)
            return True
        manager._complete_shared_host_manifest = commit
        mailbox = SimpleNamespace(client=client, tp_rank=rank, tp_size=size, namespace="host")
        mailbox.publish_local = lambda key, status, c=client: c.report_state("host", key, status)
        mailbox.group_status = lambda key, c=client: c.group_status("host", key)
        mailbox.local_status = lambda key, r=rank: bus.reports["host", key].get(r)
        controllers.append(AgenticHostRestoreController(manager, mailbox))
        managers.append(manager)
    return bus, controllers, managers


def observe(controllers, managers, name="r", generation=0):
    parent = RequestGeneration(name, generation)
    req = SimpleNamespace(rid=name + "-next", origin_input_ids=[1, 2, 3])
    for controller, manager in zip(controllers, managers):
        controller.observe(req, parent)
        manager.ready.add(parent.snapshot_id)
    return parent, req


def tick(controllers):
    for controller in controllers:
        controller.progress()


@pytest.mark.parametrize("size", [1, 2, 8])
def test_prepare_and_start_need_no_scheduler_tick(size):
    bus, controllers, managers = cluster(size)
    parent, req = observe(controllers, managers)
    req.origin_input_ids[0] = 99
    tick(controllers)
    tick(controllers)
    assert all(m.queued[0][0].origin_input_ids == (1, 2, 3) for m in managers)
    assert controllers[0].native_commands() == []
    key = controllers[0]._jobs[parent.snapshot_id]["key"]
    for rank in range(size):
        bus.reports["host", key][rank] = 2
    controllers[0].progress()
    assert controllers[0].native_commands()[0]["action"] == "bind"


def test_capacity_rejection_retries_without_scheduler_selection():
    _, controllers, managers = cluster()
    observe(controllers, managers)
    for _ in range(4):
        tick(controllers)
    assert all(len(m.queued) >= 2 for m in managers)
    assert controllers[0].native_commands() == []


def test_missing_follower_ingress_blocks_only_its_prepare_not_other_snapshots():
    _, controllers, managers = cluster()
    parent = RequestGeneration("late", 0)
    req = SimpleNamespace(rid="late-next", origin_input_ids=[1])
    controllers[0].observe(req, parent)
    managers[0].ready.add(parent.snapshot_id)
    observe(controllers, managers, "other")
    tick(controllers)
    tick(controllers)
    assert not any(p == parent for _, p in managers[1].queued)
    assert any(p.request_id == "other" for _, p in managers[1].queued)
    controllers[1].observe(req, parent)
    managers[1].ready.add(parent.snapshot_id)
    tick(controllers)
    assert any(p == parent for _, p in managers[1].queued)


def test_only_lane_bounded_ready_host_snapshots_are_selected():
    _, controllers, managers = cluster(lanes=1)
    first, _ = observe(controllers, managers, "first")
    observe(controllers, managers, "second")
    tick(controllers)
    assert list(controllers[0]._jobs) == [first.snapshot_id]
    assert controllers[0].native_commands() == []


def test_native_clear_is_retained_until_all_rank_execution_acks():
    bus, controllers, managers = cluster()
    parent, _ = observe(controllers, managers)
    tick(controllers)
    tick(controllers)
    key = controllers[0]._jobs[parent.snapshot_id]["key"]
    for rank in range(2):
        bus.reports["host", key][rank] = 5
    controllers[0].progress()
    assert controllers[0].native_commands()[0]["action"] == "clear"
    controllers[0].native_cleared(parent.snapshot_id, key.attempt_id)
    bus.reports.pop(("host", key))  # Existing native CLEAR hides phase mailbox.
    tick(controllers)
    assert controllers[0].native_commands()[0]["action"] == "clear"
    controllers[1].native_cleared(parent.snapshot_id, key.attempt_id)
    tick(controllers)
    tick(controllers)
    assert all(not c.owns(parent.snapshot_id) for c in controllers)
    assert not controllers[0].has_pending()


def test_cancel_cannot_reuse_old_success_as_quiescent_proof():
    bus, controllers, managers = cluster()
    parent, req = observe(controllers, managers)
    tick(controllers)
    tick(controllers)
    key = controllers[0]._jobs[parent.snapshot_id]["key"]
    for rank, controller in enumerate(controllers):
        bus.reports["host", key][rank] = 5
        controller.cancel(parent.snapshot_id, req.rid)
    tick(controllers)
    assert controllers[0].native_commands()[0]["action"] == "abort"
    managers[0].quiescent = True
    tick(controllers)
    assert controllers[0].native_commands()[0]["action"] == "abort"
    managers[1].quiescent = True
    tick(controllers)
    tick(controllers)
    assert controllers[0].native_commands()[0]["action"] == "clear"


def test_eviction_before_selection_stays_native_terminal_owned():
    _, controllers, managers = cluster()
    parent, req = observe(controllers, managers)
    managers[0].terminal[parent.snapshot_id] = "shared_host_evicted"
    tick(controllers)
    assert not controllers[0].owns(parent.snapshot_id)
    assert not managers[0].queued
    for controller in controllers:
        controller.forget_waiter(parent.snapshot_id, req.rid)
    assert not controllers[0].has_pending()


def test_eviction_after_selection_uses_same_tp_terminal_attempt():
    _, controllers, managers = cluster()
    parent, _ = observe(controllers, managers)
    tick(controllers)
    managers[0].terminal[parent.snapshot_id] = "shared_host_evicted"
    tick(controllers)
    command = controllers[0].native_commands()[0]
    assert command["action"] == "terminal_prepare"
    assert command["terminal"]["reason"] == "shared_host_evicted"


def test_cancel_wrong_retry_rid_does_not_cancel_live_attempt():
    _, controllers, managers = cluster()
    parent, _ = observe(controllers, managers)
    tick(controllers)
    controllers[0].cancel(parent.snapshot_id, "another-attempt")
    assert not controllers[0]._cancelled


@pytest.mark.parametrize("already_published", [False, True])
@pytest.mark.parametrize("size", [2, 8])
def test_cancel_before_follower_prepare_does_not_leave_tombstone(already_published, size):
    _, controllers, managers = cluster(size)
    parent, req = observe(controllers, managers)
    if already_published:
        controllers[0].progress()  # Follower has not consumed PREPARE yet.
    for controller, manager in zip(controllers, managers):
        controller.cancel(parent.snapshot_id, req.rid)
        manager.quiescent = True
    tick(controllers)
    tick(controllers)
    tick(controllers)
    command = controllers[0].native_commands()[0]
    assert command["action"] == "clear"
    assert not any(m.queued for m in managers)
    for controller in controllers:
        controller.native_cleared(parent.snapshot_id, command["control_attempt"])
    tick(controllers)
    tick(controllers)
    assert all(not c.has_pending() for c in controllers)
    assert all(not c._cancelled for c in controllers)


def test_pipeline_depth_override(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TP_HOST_PIPELINE_DEPTH", "1")
    _, controllers, _ = cluster(lanes=4)
    assert all(c.depth == 1 for c in controllers)


def test_native_terminal_takes_unselected_ingress_before_abort():
    _, controllers, managers = cluster(8)
    parent, req = observe(controllers, managers)
    for controller in controllers:
        assert controller.take_native_terminal(parent.snapshot_id, req.rid)
        controller.cancel(parent.snapshot_id, req.rid)
    tick(controllers)
    assert all(not c.has_pending() for c in controllers)
    assert not any(m.queued for m in managers)


def test_native_terminal_cannot_rename_active_background_attempt():
    _, controllers, managers = cluster()
    parent, req = observe(controllers, managers)
    tick(controllers)
    tick(controllers)
    assert all(not c.take_native_terminal(parent.snapshot_id, req.rid) for c in controllers)


def test_native_terminal_removes_unselected_cancel_intent():
    _, controllers, managers = cluster()
    parent, req = observe(controllers, managers)
    for controller in controllers:
        controller.cancel(parent.snapshot_id, req.rid)
        assert controller.take_native_terminal(parent.snapshot_id, req.rid)
    tick(controllers)
    assert all(not c.has_pending() for c in controllers)


@pytest.mark.parametrize("size", [2, 8])
def test_copied_worksets_do_not_hold_copy_credit_while_scheduler_stalls(size):
    bus, controllers, managers = cluster(size, lanes=1)
    parents = [observe(controllers, managers, str(i))[0] for i in range(12)]
    for i, parent in enumerate(parents):
        tick(controllers)
        tick(controllers)
        key = controllers[0]._jobs[parent.snapshot_id]["key"]
        # A slow last shard still owns the logical copy slot.
        for rank in range(size - 1):
            bus.reports["host", key][rank] = 2
        controllers[0].progress()
        assert len(controllers[0]._jobs) == i + 1
        bus.reports["host", key][size - 1] = 2
    tick(controllers)
    # No native scheduler BIND/ADMIT/CLEAR has happened. All old exact jobs
    # remain protected, and only their metadata reaches the completion queue.
    assert len(controllers[0]._jobs) == 12
    assert len(controllers[0].native_commands()) == 12
    assert all(c["action"] == "bind" for c in controllers[0].native_commands())
    assert all(c.owns(p.snapshot_id) for c in controllers for p in parents)


def test_pending_clear_does_not_reclaim_a_finished_copy_slot():
    bus, controllers, managers = cluster(lanes=1)
    parent, _ = observe(controllers, managers, "old")
    tick(controllers)
    tick(controllers)
    key = controllers[0]._jobs[parent.snapshot_id]["key"]
    for rank in range(2):
        bus.reports["host", key][rank] = 5
    controllers[0].progress()  # Issues RETIRE; native CLEAR has not run.
    bus.reports.pop(("host", key))
    new, _ = observe(controllers, managers, "new")
    controllers[0].progress()
    assert controllers[0].owns(new.snapshot_id)
    assert controllers[0].owns(parent.snapshot_id)


def test_failed_or_cancelling_copy_needs_all_rank_quiescence():
    bus, controllers, managers = cluster(lanes=1)
    parent, req = observe(controllers, managers, "old")
    tick(controllers)
    tick(controllers)
    key = controllers[0]._jobs[parent.snapshot_id]["key"]
    new, _ = observe(controllers, managers, "new")
    for rank in range(2):
        bus.reports["host", key][rank] = -1
    controllers[0].progress()
    assert not controllers[0].owns(new.snapshot_id)
    for c in controllers:
        c.cancel(parent.snapshot_id, req.rid)
    managers[0].quiescent = True
    tick(controllers)
    assert not controllers[0].owns(new.snapshot_id)
    managers[1].quiescent = True
    tick(controllers)
    tick(controllers)
    assert controllers[0].owns(new.snapshot_id)


def test_same_key_retry_requires_copy_credit_again():
    bus, controllers, managers = cluster(lanes=1)
    old, _ = observe(controllers, managers, "retry")
    tick(controllers)
    tick(controllers)
    key = controllers[0]._jobs[old.snapshot_id]["key"]
    for rank in range(2):
        bus.reports["host", key][rank] = 2
    controllers[0].progress()
    # A new read epoch under the same control identity invalidates the old
    # completion. Never latch an old success as permission for another DMA.
    bus.reports["host", key][1] = 0
    new, _ = observe(controllers, managers, "next")
    controllers[0].progress()
    assert not controllers[0].owns(new.snapshot_id)
    bus.reports["host", key][1] = 2
    controllers[0].progress()
    assert controllers[0].owns(new.snapshot_id)
