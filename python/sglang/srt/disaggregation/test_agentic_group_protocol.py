from __future__ import annotations

import builtins
import os
import threading
import time

import pytest

from sglang.srt.disaggregation.agentic_group_protocol import (
    AttemptOutcome,
    GenerationKey,
    LinkCapacityEdge,
    LinkApplicationFinal,
    GenerationTerminal,
    GroupDisconnectedError,
    GroupLifecycleCoordinator,
    LinkLifecycleCoordinator,
    LinkFailure,
    LinkIntent,
    LinkReadiness,
    LinkDisconnected,
    LinkParticipant,
    LinkRankAck,
    GroupNotReadyError,
    GroupProtocolError,
    Owner,
    RankAck,
    RankPhase,
    ReadinessPhase,
    StaleAttemptError,
    TCPGroupRelayServer,
    TCPRankAgent,
    send_application_final,
    _encode_frame,
    TCPRankZeroServer,
)


def test_external_application_final_reaches_fixed_link_coordinator():
    links = {
        "global": {
            "coordinator": {"endpoint_group": "p0", "rank": 0},
            "endpoints": [
                {"endpoint_group": "p0", "role": "prefill", "size": 1}
            ],
        }
    }
    relay = TCPGroupRelayServer(run_id="run", token="secret", links=links)
    agent = TCPRankAgent(
        relay.address,
        run_id="run",
        group_id="global",
        token="secret",
        rank=0,
        tp_size=1,
        endpoint_group="p0",
        endpoint_role="prefill",
    )
    try:
        assert relay.wait_connected("global", timeout=2)
        key = GenerationKey("run", "request", 7)
        send_application_final(
            relay.address,
            run_id="run",
            group_id="global",
            token="secret",
            key=key,
        )
        event = agent.receive_event()
        assert event == LinkApplicationFinal("global", key)
    finally:
        agent.close()
        relay.close()


def _ack(command, rank, phase, *, ok=True, detail=""):
    return RankAck(
        key=command.key,
        attempt=command.attempt,
        command_seq=command.command_seq,
        rank=rank,
        phase=phase,
        lease_id=command.lease_id,
        ok=ok,
        detail=detail,
    )


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_tp_group_commit_release_and_short_attempt_retirement(tp_size):
    coordinator = GroupLifecycleCoordinator("run", "p", tp_size)
    key = GenerationKey("run", "request", 3)
    prepare = coordinator.begin_attempt(
        key,
        source_owner=Owner.D_GPU,
        target_owner=Owner.PREFILL_READY,
        lease_id="p-workset-1",
        required_commit_phase=RankPhase.RELEASED,
    )

    # Rank reports may arrive in any cross-rank order.
    for rank in reversed(range(tp_size)):
        assert coordinator.apply_ack(_ack(prepare, rank, RankPhase.PREPARED))
    start = coordinator.issue_start(key, prepare.attempt)
    for rank in range(tp_size):
        coordinator.apply_ack(_ack(start, rank, RankPhase.DMA_SUBMITTED))
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(_ack(start, rank, RankPhase.DMA_DONE))

    release = coordinator.issue_release(key, prepare.attempt)
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(_ack(release, rank, RankPhase.BOUND))
    handoff = coordinator.issue_handoff(key, prepare.attempt)
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(_ack(handoff, rank, RankPhase.RELEASED))
    activate = coordinator.issue_activate(key, prepare.attempt)
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(_ack(activate, rank, RankPhase.STAGED))
    scheduler_activate = coordinator.issue_scheduler_activate(key, prepare.attempt)
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(
            _ack(scheduler_activate, rank, RankPhase.ACTIVATION_ARMED)
        )
    publish = coordinator.issue_publish_activation(key, prepare.attempt)
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(_ack(publish, rank, RankPhase.ACTIVATION_READY))
    ticket = coordinator.issue_activation_ticket(key, prepare.attempt)
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(_ack(ticket, rank, RankPhase.ACTIVATED))
    committed = coordinator.commit(key, prepare.attempt)
    assert committed.owner is Owner.PREFILL_READY
    assert coordinator.outcome(key, prepare.attempt) is AttemptOutcome.COMMITTED
    finalize = coordinator.issue_finalize(key, prepare.attempt)
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(_ack(finalize, rank, RankPhase.FINALIZED))
    coordinator.retire(key, prepare.attempt)
    assert coordinator.active_attempt(key) is None
    assert coordinator.record(key) == committed

    # The next attempt is monotonic. No old ACK history was retained, yet the
    # ledger's attempt number is sufficient to reject a delayed callback.
    next_prepare = coordinator.begin_attempt(
        key,
        source_owner=Owner.PREFILL_READY,
        target_owner=Owner.P_HOST,
        lease_id="p-host-2",
    )
    assert next_prepare.attempt == prepare.attempt + 1
    with pytest.raises(StaleAttemptError):
        coordinator.apply_ack(_ack(handoff, 0, RankPhase.RELEASED))


def test_duplicate_ack_is_idempotent_and_changed_duplicate_is_rejected():
    coordinator = GroupLifecycleCoordinator("run", "d", 2)
    key = GenerationKey("run", "r", 0)
    command = coordinator.begin_attempt(
        key,
        source_owner=Owner.D_GPU,
        target_owner=Owner.D_HOST,
        lease_id="host-1",
    )
    ack = _ack(command, 0, RankPhase.PREPARED)
    assert coordinator.apply_ack(ack)
    assert not coordinator.apply_ack(ack)
    with pytest.raises(GroupProtocolError, match="changed"):
        coordinator.apply_ack(
            _ack(command, 0, RankPhase.PREPARED, detail="different")
        )


def test_in_rank_phase_jump_is_rejected_but_cross_rank_reordering_is_valid():
    coordinator = GroupLifecycleCoordinator("run", "d", 2)
    key = GenerationKey("run", "r", 1)
    prepare = coordinator.begin_attempt(
        key,
        source_owner=Owner.D_GPU,
        target_owner=Owner.P_GPU,
        lease_id="direct-1",
    )
    coordinator.apply_ack(_ack(prepare, 1, RankPhase.PREPARED))
    with pytest.raises(GroupNotReadyError):
        coordinator.issue_start(key, prepare.attempt)
    coordinator.apply_ack(_ack(prepare, 0, RankPhase.PREPARED))
    start = coordinator.issue_start(key, prepare.attempt)
    with pytest.raises(GroupProtocolError, match="jumped"):
        coordinator.apply_ack(_ack(start, 0, RankPhase.DMA_DONE))

    coordinator.apply_ack(_ack(start, 1, RankPhase.DMA_SUBMITTED))
    coordinator.apply_ack(_ack(start, 0, RankPhase.DMA_SUBMITTED))
    coordinator.apply_ack(_ack(start, 0, RankPhase.DMA_DONE))
    coordinator.apply_ack(_ack(start, 1, RankPhase.DMA_DONE))
    assert coordinator.group_reached(key, prepare.attempt, RankPhase.DMA_DONE)


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_abort_waits_for_every_rank_drained_fence(tp_size):
    coordinator = GroupLifecycleCoordinator("run", "d", tp_size)
    key = GenerationKey("run", "r", 2)
    prepare = coordinator.begin_attempt(
        key,
        source_owner=Owner.D_GPU,
        target_owner=Owner.P_GPU,
        lease_id="direct-2",
    )
    for rank in range(tp_size):
        coordinator.apply_ack(_ack(prepare, rank, RankPhase.PREPARED))
    start = coordinator.issue_start(key, prepare.attempt)
    for rank in range(tp_size):
        coordinator.apply_ack(_ack(start, rank, RankPhase.DMA_SUBMITTED))

    cancel = coordinator.request_abort(key, prepare.attempt, "direct failed")
    for rank in range(tp_size - 1):
        coordinator.apply_ack(
            _ack(
                cancel,
                rank,
                RankPhase.FAILED_DRAINED,
                ok=False,
                detail="local DMA drained",
            )
        )
    with pytest.raises(GroupNotReadyError):
        coordinator.complete_abort(key, prepare.attempt)

    coordinator.apply_ack(
        _ack(
            cancel,
            tp_size - 1,
            RankPhase.FAILED_DRAINED,
            ok=False,
            detail="local DMA drained",
        )
    )
    record = coordinator.complete_abort(key, prepare.attempt)
    assert record.owner is Owner.D_GPU
    assert coordinator.outcome(key, prepare.attempt) is AttemptOutcome.ABORTED
    finalize = coordinator.issue_abort_finalize(key, prepare.attempt)
    for rank in reversed(range(tp_size)):
        coordinator.apply_ack(
            _ack(
                finalize,
                rank,
                RankPhase.ABORTED,
                detail="local rollback finalized",
            )
        )
    coordinator.retire(key, prepare.attempt)


def test_disconnect_is_fail_closed_and_retains_source_owner():
    coordinator = GroupLifecycleCoordinator("run", "d", 2)
    key = GenerationKey("run", "r", 4)
    prepare = coordinator.begin_attempt(
        key,
        source_owner=Owner.D_GPU,
        target_owner=Owner.P_GPU,
        lease_id="direct-4",
    )
    for rank in range(2):
        coordinator.apply_ack(_ack(prepare, rank, RankPhase.PREPARED))
    start = coordinator.issue_start(key, prepare.attempt)
    for rank in range(2):
        coordinator.apply_ack(_ack(start, rank, RankPhase.DMA_SUBMITTED))
    coordinator.apply_ack(_ack(start, 0, RankPhase.DMA_DONE))
    coordinator.mark_disconnected(1)

    assert coordinator.record(key).owner is Owner.D_GPU
    assert coordinator.outcome(key, prepare.attempt) is AttemptOutcome.ABORTING
    with pytest.raises(GroupDisconnectedError):
        coordinator.commit(key, prepare.attempt)
    with pytest.raises(GroupDisconnectedError):
        coordinator.complete_abort(key, prepare.attempt)
    with pytest.raises(GroupNotReadyError):
        coordinator.retire(key, prepare.attempt)


def test_terminal_generation_rejects_new_attempts():
    coordinator = GroupLifecycleCoordinator("run", "p", 1)
    key = GenerationKey("run", "r", 5)
    coordinator.register(key, Owner.P_GPU)
    record = coordinator.mark_terminal(key, GenerationTerminal.FINAL)
    assert record.owner is Owner.NONE
    with pytest.raises(GroupProtocolError, match="terminal"):
        coordinator.begin_attempt(
            key,
            source_owner=Owner.NONE,
            target_owner=Owner.D_GPU,
            lease_id="impossible",
        )


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_tcp_rank0_push_and_rank_ack_for_tp_sizes(tp_size):
    coordinator = GroupLifecycleCoordinator("run", "p", tp_size)
    server = TCPRankZeroServer(coordinator, token="secret")
    agents = [
        TCPRankAgent(
            server.address,
            run_id="run",
            group_id="p",
            token="secret",
            rank=rank,
            tp_size=tp_size,
        )
        for rank in range(tp_size)
    ]
    try:
        assert server.wait_connected(timeout=2)
        key = GenerationKey("run", "tcp", tp_size)
        prepare = coordinator.begin_attempt(
            key,
            source_owner=Owner.P_GPU,
            target_owner=Owner.DECODE_READY,
            lease_id=f"tcp-{tp_size}",
            payload={"direction": "p2d-direct"},
        )
        server.broadcast(prepare)
        received = [agent.receive() for agent in agents]
        assert received == [prepare] * tp_size
        for agent, command in reversed(list(zip(agents, received))):
            agent.acknowledge(command, RankPhase.PREPARED)
        assert coordinator.wait_for(
            lambda: coordinator.group_reached(
                key, prepare.attempt, RankPhase.PREPARED
            ),
            timeout=2,
        )

        start = coordinator.issue_start(key, prepare.attempt)
        server.broadcast(start)
        received = [agent.receive() for agent in agents]
        for agent, command in zip(agents, received):
            agent.acknowledge(command, RankPhase.DMA_SUBMITTED)
            agent.acknowledge(command, RankPhase.DMA_DONE)
        assert coordinator.wait_for(
            lambda: coordinator.group_reached(
                key, prepare.attempt, RankPhase.DMA_DONE
            ),
            timeout=2,
        )
        assert coordinator.commit(key, prepare.attempt).owner is Owner.DECODE_READY
    finally:
        for agent in agents:
            agent.close()
        server.close()


def test_tcp_disconnect_marks_active_attempt_fail_closed():
    coordinator = GroupLifecycleCoordinator("run", "d", 2)
    server = TCPRankZeroServer(coordinator, token="secret")
    agents = [
        TCPRankAgent(
            server.address,
            run_id="run",
            group_id="d",
            token="secret",
            rank=rank,
            tp_size=2,
        )
        for rank in range(2)
    ]
    try:
        assert server.wait_connected(timeout=2)
        key = GenerationKey("run", "disconnect", 0)
        prepare = coordinator.begin_attempt(
            key,
            source_owner=Owner.D_GPU,
            target_owner=Owner.P_GPU,
            lease_id="disconnect-lease",
        )
        server.broadcast(prepare)
        commands = [agent.receive() for agent in agents]
        for agent, command in zip(agents, commands):
            agent.acknowledge(command, RankPhase.PREPARED)
        assert coordinator.wait_for(
            lambda: coordinator.group_reached(
                key, prepare.attempt, RankPhase.PREPARED
            ),
            timeout=2,
        )
        agents[1].close()
        assert coordinator.wait_for(
            lambda: coordinator.outcome(key, prepare.attempt)
            is AttemptOutcome.ABORTING,
            timeout=2,
        )
        assert coordinator.record(key).owner is Owner.D_GPU
        with pytest.raises(GroupDisconnectedError):
            coordinator.commit(key, prepare.attempt)
    finally:
        agents[0].close()
        server.close()


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_standalone_relay_carries_rank0_commands_and_all_rank_acks(tp_size):
    coordinator = GroupLifecycleCoordinator("run", "p", tp_size)
    relay = TCPGroupRelayServer(
        run_id="run", token="secret", groups={"p": tp_size}
    )
    agents = [
        TCPRankAgent(
            relay.address,
            run_id="run",
            group_id="p",
            token="secret",
            rank=rank,
            tp_size=tp_size,
        )
        for rank in range(tp_size)
    ]
    try:
        assert relay.wait_connected("p", timeout=2)
        key = GenerationKey("run", "relay", tp_size)
        command = coordinator.begin_attempt(
            key,
            source_owner=Owner.D_GPU,
            target_owner=Owner.P_GPU,
            lease_id="relay-lease",
        )
        agents[0].publish(command)
        received = [agent.receive_event() for agent in agents]
        assert received == [command] * tp_size
        for agent, rank_command in zip(agents, received):
            agent.acknowledge(rank_command, RankPhase.PREPARED)
        for _ in range(tp_size):
            event = agents[0].receive_event()
            assert isinstance(event, RankAck)
            coordinator.apply_ack(event)
        assert coordinator.group_reached(
            key, command.attempt, RankPhase.PREPARED
        )
    finally:
        for agent in agents:
            agent.close()
        relay.close()


def test_rank_agent_from_env_needs_no_control_root(monkeypatch):
    relay = TCPGroupRelayServer(run_id="run", token="secret", groups={"p": 1})
    host, port = relay.address
    values = {
        "ENDPOINT": f"tcp://{host}:{port}",
        "RUN_ID": "run",
        "TOKEN": "secret",
        "GROUP_ID": "p",
        "RANK": "0",
        "SIZE": "1",
    }
    for name, value in values.items():
        monkeypatch.setenv("SGLANG_AGENTIC_GROUP_" + name, value)
    monkeypatch.delenv("SGLANG_AGENTIC_MULTINODE_CONTROL_ROOT", raising=False)
    agent = TCPRankAgent.from_env()
    try:
        assert relay.wait_connected("p", timeout=2)
        assert agent.group_id == "p"
        assert agent.rank == 0
    finally:
        agent.close()
        relay.close()


def test_same_parent_environment_creates_eight_distinct_ranks(monkeypatch):
    relay = TCPGroupRelayServer(run_id="run", token="secret", groups={"p": 8})
    host, port = relay.address
    values = {
        "ENDPOINT": f"tcp://{host}:{port}", "RUN_ID": "run",
        "TOKEN": "secret", "GROUP_ID": "p", "SIZE": "8",
    }
    for name, value in values.items():
        monkeypatch.setenv("SGLANG_AGENTIC_GROUP_" + name, value)
    monkeypatch.delenv("SGLANG_AGENTIC_GROUP_RANK", raising=False)
    agents = [TCPRankAgent.from_env(rank=rank, tp_size=8) for rank in range(8)]
    try:
        assert relay.wait_connected("p", timeout=2)
        assert [agent.rank for agent in agents] == list(range(8))
    finally:
        for agent in agents:
            agent.close()
        relay.close()


def test_link_attempt_commits_only_after_source_and_target_tp8_fences():
    link_id = "p0--d0"
    links = {
        link_id: {
            "coordinator": {"endpoint_group": "p0", "rank": 0},
            "endpoints": [
                {"endpoint_group": "p0", "role": "prefill", "size": 8},
                {"endpoint_group": "d0", "role": "decode", "size": 8},
            ],
        }
    }
    relay = TCPGroupRelayServer(run_id="run", token="secret", links=links)
    participants = [
        LinkParticipant(role, group, rank)
        for role, group in (("prefill", "p0"), ("decode", "d0"))
        for rank in range(8)
    ]
    coordinator = LinkLifecycleCoordinator("run", link_id, participants)
    agents = {
        participant: TCPRankAgent(
            relay.address, run_id="run", group_id=link_id, token="secret",
            rank=participant.rank, tp_size=8,
            endpoint_group=participant.endpoint_group,
            endpoint_role=participant.role,
        )
        for participant in participants
    }
    try:
        assert relay.wait_connected(link_id, timeout=2)
        key = GenerationKey("run", "cross-endpoint", 0)
        prepare = coordinator.begin_attempt(
            key, source_owner=Owner.D_GPU, target_owner=Owner.P_GPU,
            lease_id="link-lease",
        )
        agents[LinkParticipant("prefill", "p0", 0)].publish(prepare)
        received = {participant: agent.receive() for participant, agent in agents.items()}
        assert set(received) == set(participants)
        for participant, agent in agents.items():
            agent.acknowledge(received[participant], RankPhase.PREPARED)
        for _ in participants:
            event = agents[LinkParticipant("prefill", "p0", 0)].receive_event()
            assert isinstance(event, LinkRankAck)
            coordinator.apply_ack(event)
        start = coordinator.issue_start(key, prepare.attempt)
        agents[LinkParticipant("prefill", "p0", 0)].publish(start)
        started = {participant: agent.receive() for participant, agent in agents.items()}
        # DMA_SUBMITTED and DMA_DONE are separate physical fences.
        for phase in (RankPhase.DMA_SUBMITTED, RankPhase.DMA_DONE):
            for participant, agent in agents.items():
                agent.acknowledge(started[participant], phase)
            for _ in participants:
                coordinator.apply_ack(
                    agents[LinkParticipant("prefill", "p0", 0)].receive_event()
                )
        record = coordinator.commit(key, prepare.attempt)
        assert record.owner is Owner.P_GPU
    finally:
        for agent in agents.values():
            agent.close()
        relay.close()


def test_global_fabric_command_targets_only_selected_tp2_groups():
    link_id = "global"
    endpoint_specs = [
        (role, f"{prefix}{index}", 2)
        for role, prefix in (("prefill", "p"), ("decode", "d"))
        for index in range(4)
    ]
    links = {
        link_id: {
            "coordinator": {"endpoint_group": "p0", "rank": 0},
            "endpoints": [
                {"endpoint_group": group, "role": role, "size": size}
                for role, group, size in endpoint_specs
            ],
        }
    }
    participants = [
        LinkParticipant(role, group, rank)
        for role, group, size in endpoint_specs
        for rank in range(size)
    ]
    relay = TCPGroupRelayServer(run_id="run", token="secret", links=links)
    agents = {
        participant: TCPRankAgent(
            relay.address,
            run_id="run",
            group_id=link_id,
            token="secret",
            rank=participant.rank,
            tp_size=2,
            endpoint_group=participant.endpoint_group,
            endpoint_role=participant.role,
        )
        for participant in participants
    }
    coordinator = LinkLifecycleCoordinator("run", link_id, participants)
    selected = tuple(
        participant
        for participant in participants
        if participant.endpoint_group in {"d1", "p2"}
    )
    try:
        assert relay.wait_connected(link_id, timeout=2)
        key = GenerationKey("run", "late-bind", 1)
        command = coordinator.begin_attempt(
            key,
            source_owner=Owner.D_GPU,
            target_owner=Owner.P_GPU,
            lease_id="d1-to-p2",
            endpoint_groups=("d1", "p2"),
        )
        agents[LinkParticipant("prefill", "p0", 0)].publish(command)
        received = {
            participant: agents[participant].receive() for participant in selected
        }
        assert set(received) == set(selected)
        assert coordinator.participants_for(key, command.attempt) == selected
        for participant, value in received.items():
            agents[participant].acknowledge(value, RankPhase.PREPARED)
        for _ in selected:
            coordinator.apply_ack(
                agents[LinkParticipant("prefill", "p0", 0)].receive_event()
            )
        assert coordinator.group_reached(key, command.attempt, RankPhase.PREPARED)
        with pytest.raises(GroupProtocolError, match="selected attempt"):
            coordinator.apply_ack(
                LinkRankAck(
                    LinkParticipant("decode", "d0", 0),
                    _ack(command, 0, RankPhase.PREPARED),
                )
            )
    finally:
        for agent in agents.values():
            agent.close()
        relay.close()


def test_noncoordinator_rank0_intent_is_forwarded_once_and_stale_is_rejected():
    links = {
        "p--d": {
            "coordinator": {"endpoint_group": "p", "rank": 0},
            "endpoints": [
                {"endpoint_group": "p", "role": "prefill", "size": 2},
                {"endpoint_group": "d", "role": "decode", "size": 2},
            ],
        }
    }
    relay = TCPGroupRelayServer(run_id="run", token="secret", links=links)
    agents = {
        (group, rank): TCPRankAgent(
            relay.address, run_id="run", group_id="p--d", token="secret",
            rank=rank, tp_size=2, endpoint_group=group,
            endpoint_role="prefill" if group == "p" else "decode",
        )
        for group in ("p", "d") for rank in range(2)
    }
    key = GenerationKey("run", "intent", 0)
    try:
        assert relay.wait_connected("p--d", timeout=2)
        with pytest.raises(GroupProtocolError):
            agents[("d", 1)].propose_intent(key, 1, "d2p_direct")
        agents[("d", 0)].propose_intent(key, 1, "d2p_direct", {"tokens": 9})
        event = agents[("p", 0)].receive_event()
        assert isinstance(event, LinkIntent)
        assert event.participant == LinkParticipant("decode", "d", 0)
        assert event.proposal_seq == 1 and event.payload["tokens"] == 9
        # Exact retry is consumed idempotently; the next sequence remains the
        # next event visible to the coordinator.
        agents[("d", 0)].propose_intent(key, 1, "d2p_direct", {"tokens": 9})
        agents[("d", 0)].propose_intent(key, 2, "d2p_slow")
        event = agents[("p", 0)].receive_event()
        assert isinstance(event, LinkIntent) and event.proposal_seq == 2
        agents[("d", 0)].propose_intent(key, 1, "d2p_direct", {"tokens": 9})
        assert relay.wait_for_error(timeout=2)
        assert any(isinstance(error, StaleAttemptError) for error in relay.errors)
    finally:
        for agent in agents.values():
            agent.close()
        relay.close()


def test_tp8_decode_capacity_edge_is_typed_idempotent_and_reconnectable():
    links = {
        "p--d": {
            "coordinator": {"endpoint_group": "p", "rank": 0},
            "endpoints": [
                {"endpoint_group": "p", "role": "prefill", "size": 8},
                {"endpoint_group": "d", "role": "decode", "size": 8},
            ],
        }
    }
    relay = TCPGroupRelayServer(run_id="run", token="secret", links=links)
    agents = {
        (group, rank): TCPRankAgent(
            relay.address,
            run_id="run",
            group_id="p--d",
            token="secret",
            rank=rank,
            tp_size=8,
            endpoint_group=group,
            endpoint_role="prefill" if group == "p" else "decode",
        )
        for group in ("p", "d")
        for rank in range(8)
    }
    p0 = agents[("p", 0)]
    d0 = agents[("d", 0)]
    try:
        assert relay.wait_connected("p--d", timeout=5)
        with pytest.raises(GroupProtocolError):
            agents[("d", 1)].send_capacity_edge(128)

        assert d0.send_capacity_edge(4096) == 1
        edge = p0.receive_event()
        assert edge == LinkCapacityEdge(
            "p--d", LinkParticipant("decode", "d", 0),
            d0._capacity_session_id, 1, 4096,
        )

        # An exact TCP retransmission is consumed by the relay and creates no
        # second policy edge.  The next sequence is the next visible event.
        with d0._send_lock:
            d0._socket.sendall(
                _encode_frame(
                    {
                        "type": "capacity_edge",
                        "session_id": d0._capacity_session_id,
                        "edge_seq": 1,
                        "available_tokens": 4096,
                    }
                )
            )
        assert d0.send_capacity_edge(8192) == 2
        edge = p0.receive_event()
        assert isinstance(edge, LinkCapacityEdge)
        assert edge.edge_seq == 2 and edge.available_tokens == 8192

        # Sequence state is scoped to the live TCP session.  A replacement D0
        # may restart at one; the disconnect itself remains visible/fail-closed
        # to a production lifecycle runtime.
        d0.close()
        disconnected = p0.receive_event()
        assert isinstance(disconnected, LinkDisconnected)
        deadline = time.monotonic() + 2
        while relay.wait_connected("p--d", timeout=0.01):
            assert time.monotonic() < deadline
        replacement = TCPRankAgent(
            relay.address,
            run_id="run",
            group_id="p--d",
            token="secret",
            rank=0,
            tp_size=8,
            endpoint_group="d",
            endpoint_role="decode",
        )
        agents[("d", 0)] = replacement
        assert relay.wait_connected("p--d", timeout=2)
        assert replacement.send_capacity_edge(2048) == 1
        edge = p0.receive_event()
        assert isinstance(edge, LinkCapacityEdge)
        assert edge.edge_seq == 1 and edge.session_id != d0._capacity_session_id
    finally:
        for agent in agents.values():
            agent.close()
        relay.close()


def test_any_link_rank_failure_is_forwarded_to_coordinator_before_cancel():
    links = {
        "p--d": {
            "coordinator": {"endpoint_group": "p", "rank": 0},
            "endpoints": [
                {"endpoint_group": "p", "role": "prefill", "size": 2},
                {"endpoint_group": "d", "role": "decode", "size": 2},
            ],
        }
    }
    relay = TCPGroupRelayServer(run_id="run", token="secret", links=links)
    agents = {
        (group, rank): TCPRankAgent(
            relay.address,
            run_id="run",
            group_id="p--d",
            token="secret",
            rank=rank,
            tp_size=2,
            endpoint_group=group,
            endpoint_role="prefill" if group == "p" else "decode",
        )
        for group in ("p", "d")
        for rank in range(2)
    }
    key = GenerationKey("run", "failure", 0)
    coordinator = LinkLifecycleCoordinator(
        "run",
        "p--d",
        [
            LinkParticipant(role, group, rank)
            for role, group in (("prefill", "p"), ("decode", "d"))
            for rank in range(2)
        ],
    )
    command = coordinator.begin_attempt(
        key,
        source_owner=Owner.D_GPU,
        target_owner=Owner.P_GPU,
        lease_id="failure-lease",
    )
    try:
        assert relay.wait_connected("p--d", timeout=2)
        agents[("d", 1)].report_failure(
            key, command.attempt, command.lease_id, "local DMA submit failed"
        )
        event = agents[("p", 0)].receive_event()
        assert event == LinkFailure(
            "p--d",
            LinkParticipant("decode", "d", 1),
            key,
            command.attempt,
            "failure-lease",
            "local DMA submit failed",
        )
        # A failure report does not mutate ownership or fabricate a drained
        # fence. The coordinator must explicitly issue CANCEL next.
        assert coordinator.record(key).owner is Owner.D_GPU
        assert coordinator.outcome(key, command.attempt) is AttemptOutcome.ACTIVE
        cancel = coordinator.request_abort(key, command.attempt, event.detail)
        assert cancel.kind.value == "cancel"
    finally:
        for agent in agents.values():
            agent.close()
        relay.close()


def test_failed_drained_cannot_be_reported_as_success():
    with pytest.raises(ValueError, match="failed ACK"):
        RankAck(
            key=GenerationKey("run", "bad-fence", 0),
            attempt=1,
            command_seq=1,
            rank=0,
            phase=RankPhase.FAILED_DRAINED,
            lease_id="lease",
            ok=True,
        )


def test_rank_result_is_json_only_and_exclusive_to_dma_done():
    values = dict(
        key=GenerationKey("run", "result", 0),
        attempt=1,
        command_seq=2,
        rank=0,
        lease_id="lease",
    )
    with pytest.raises(ValueError, match="only valid on DMA_DONE"):
        RankAck(
            **values,
            phase=RankPhase.PREPARED,
            result={"host_shard": {"export_id": "x"}},
        )
    with pytest.raises(TypeError):
        RankAck(
            **values,
            phase=RankPhase.DMA_DONE,
            result={"host_shard": object()},
        )
    payload = {"host_shard": {"export_id": "x"}}
    ack = RankAck(**values, phase=RankPhase.DMA_DONE, result=payload)
    payload["host_shard"]["export_id"] = "changed"
    assert ack.result["host_shard"]["export_id"] == "x"


def test_link_disconnect_is_fail_closed_for_whole_cross_endpoint_attempt():
    participants = [
        LinkParticipant(role, group, rank)
        for role, group in (("prefill", "p"), ("decode", "d"))
        for rank in range(2)
    ]
    coordinator = LinkLifecycleCoordinator("run", "p--d", participants)
    key = GenerationKey("run", "disconnect", 0)
    command = coordinator.begin_attempt(
        key, source_owner=Owner.D_GPU, target_owner=Owner.P_GPU,
        lease_id="lease",
    )
    coordinator.apply_event(
        LinkDisconnected("p--d", LinkParticipant("prefill", "p", 1))
    )
    assert coordinator.record(key).owner is Owner.D_GPU
    with pytest.raises(GroupDisconnectedError):
        coordinator.commit(key, command.attempt)


def test_runtime_protocol_uses_no_filesystem_or_directory_polling(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("group protocol touched the filesystem")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)

    coordinator = GroupLifecycleCoordinator("run", "p", 1)
    key = GenerationKey("run", "memory-only", 0)
    prepare = coordinator.begin_attempt(
        key,
        source_owner=Owner.P_GPU,
        target_owner=Owner.D_GPU,
        lease_id="memory-only-lease",
    )
    coordinator.apply_ack(_ack(prepare, 0, RankPhase.PREPARED))
    start = coordinator.issue_start(key, prepare.attempt)
    coordinator.apply_ack(_ack(start, 0, RankPhase.DMA_SUBMITTED))
    coordinator.apply_ack(_ack(start, 0, RankPhase.DMA_DONE))
    assert coordinator.commit(key, prepare.attempt).owner is Owner.D_GPU


def test_tcp_commands_are_immutable_across_sender_mutation():
    payload = {"indices": [1, 2, 3]}
    coordinator = GroupLifecycleCoordinator("run", "p", 1)
    key = GenerationKey("run", "immutable", 0)
    command = coordinator.begin_attempt(
        key,
        source_owner=Owner.P_GPU,
        target_owner=Owner.D_GPU,
        lease_id="lease",
        payload=payload,
    )
    payload["indices"].append(4)
    assert command.payload["indices"] == [1, 2, 3]
    with pytest.raises(TypeError):
        command.payload["new"] = True


def test_wait_condition_is_event_driven():
    coordinator = GroupLifecycleCoordinator("run", "p", 1)
    event = threading.Event()

    def publish():
        event.wait(timeout=1)
        coordinator.register(GenerationKey("run", "wake", 0), Owner.P_GPU)

    thread = threading.Thread(target=publish)
    thread.start()
    event.set()
    assert coordinator.wait_for(
        lambda: coordinator.record(GenerationKey("run", "wake", 0)) is not None,
        timeout=1,
    )
    thread.join(timeout=1)
