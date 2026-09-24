"""Minimal in-memory TP group protocol for request-generation KV movement.

The protocol deliberately separates logical group decisions from physical I/O:

* TP rank zero is the only process that creates attempts and commands.
* Every rank executes the same immutable command for its local shard.
* Rank acknowledgements carry physical fence progress, never routing choices.
* Ownership changes only after every rank reaches the required fence.
* A cancelled transfer remains live until every rank reports a drained fence.

The TCP classes below are only a push transport for commands and ACKs.  They do
not use files, shared directories, polling, expiry, or a persistent event log.
An unexpected disconnect is fail-closed: the current owner is retained and an
active attempt cannot commit or retire until the missing rank is known drained.
"""

from __future__ import annotations

import copy
import argparse
import json
import os
import signal
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional


_MAX_FRAME_BYTES = 4 * 1024 * 1024


class GroupProtocolError(RuntimeError):
    """Base error for invalid ownership or TP group progress."""


class StaleAttemptError(GroupProtocolError):
    """A delayed message refers to an attempt older than the active one."""


class GroupNotReadyError(GroupProtocolError):
    """The requested group transition lacks a complete physical fence."""


class GroupDisconnectedError(GroupProtocolError):
    """At least one rank disconnected, so ownership must remain unchanged."""


@dataclass(frozen=True, slots=True)
class GenerationKey:
    run_id: str
    request_id: str
    generation: int

    def __post_init__(self) -> None:
        if not self.run_id or not self.request_id:
            raise ValueError("run_id and request_id must be non-empty")
        if int(self.generation) < 0:
            raise ValueError("generation must be non-negative")
        object.__setattr__(self, "generation", int(self.generation))

    @property
    def snapshot_id(self) -> str:
        return f"{self.request_id}:{int(self.generation)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "generation": int(self.generation),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationKey":
        return cls(
            run_id=str(value["run_id"]),
            request_id=str(value["request_id"]),
            generation=int(value["generation"]),
        )


class Owner(str, Enum):
    D_GPU = "d_gpu"
    D_HOST = "d_host"
    P_GPU = "p_gpu"
    P_HOST = "p_host"
    PREFILL_READY = "prefill_ready"
    DECODE_READY = "decode_ready"
    NONE = "none"


class GenerationTerminal(str, Enum):
    OPEN = "open"
    FINAL = "final"
    EVICTED = "evicted"
    FAILED = "failed"


class AttemptOutcome(str, Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"


class CommandKind(str, Enum):
    PREPARE = "prepare"
    START = "start"
    CANCEL = "cancel"
    RELEASE = "release"
    HANDOFF = "handoff"
    ACTIVATE = "activate"
    SCHEDULER_ACTIVATE = "scheduler_activate"
    PUBLISH_ACTIVATION = "publish_activation"
    ISSUE_ACTIVATION_TICKET = "issue_activation_ticket"
    FINALIZE = "finalize"
    ABORT_FINALIZE = "abort_finalize"


class RankPhase(IntEnum):
    NONE = 0
    PREPARED = 1
    DMA_SUBMITTED = 2
    DMA_DONE = 3
    BOUND = 4
    RELEASED = 5
    STAGED = 6
    ACTIVATION_ARMED = 7
    ACTIVATION_READY = 8
    ACTIVATED = 9
    FINALIZED = 10
    # This is a terminal physical fence on the abort branch, not an ordered
    # successor of RELEASED.  It is intentionally outside normal comparisons.
    FAILED_DRAINED = 100
    ABORTED = 101


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """The complete long-lived group ledger for one generation."""

    key: GenerationKey
    attempt: int
    owner: Owner
    terminal: GenerationTerminal = GenerationTerminal.OPEN


def _frozen_payload(payload: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    value = {} if payload is None else copy.deepcopy(dict(payload))
    # Validate that messages never retain Python/CUDA objects by accident.
    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class GroupCommand:
    key: GenerationKey
    attempt: int
    command_seq: int
    kind: CommandKind
    source_owner: Owner
    target_owner: Owner
    lease_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.attempt) < 1 or int(self.command_seq) < 1:
            raise ValueError("attempt and command_seq must be positive")
        if not self.lease_id:
            raise ValueError("lease_id must be non-empty")
        object.__setattr__(self, "attempt", int(self.attempt))
        object.__setattr__(self, "command_seq", int(self.command_seq))
        object.__setattr__(self, "payload", _frozen_payload(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "command",
            "key": self.key.to_dict(),
            "attempt": int(self.attempt),
            "command_seq": int(self.command_seq),
            "kind": self.kind.value,
            "source_owner": self.source_owner.value,
            "target_owner": self.target_owner.value,
            "lease_id": self.lease_id,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroupCommand":
        if value.get("type") != "command":
            raise ValueError("frame is not a group command")
        return cls(
            key=GenerationKey.from_dict(value["key"]),
            attempt=int(value["attempt"]),
            command_seq=int(value["command_seq"]),
            kind=CommandKind(value["kind"]),
            source_owner=Owner(value["source_owner"]),
            target_owner=Owner(value["target_owner"]),
            lease_id=str(value["lease_id"]),
            payload=value.get("payload") or {},
        )


@dataclass(frozen=True, slots=True)
class RankAck:
    key: GenerationKey
    attempt: int
    command_seq: int
    rank: int
    phase: RankPhase
    lease_id: str
    ok: bool = True
    detail: str = ""
    result: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.attempt) < 1 or int(self.command_seq) < 1:
            raise ValueError("attempt and command_seq must be positive")
        if int(self.rank) < 0:
            raise ValueError("rank must be non-negative")
        if not self.lease_id:
            raise ValueError("lease_id must be non-empty")
        if bool(self.ok) != (self.phase is not RankPhase.FAILED_DRAINED):
            raise ValueError(
                "a failed ACK must prove its local fence is drained, and a "
                "drained-failure ACK cannot report success"
            )
        if self.result and self.phase not in {
            RankPhase.DMA_DONE,
            RankPhase.FAILED_DRAINED,
        }:
            raise ValueError(
                "rank-local result is only valid on DMA_DONE/FAILED_DRAINED"
            )
        object.__setattr__(self, "attempt", int(self.attempt))
        object.__setattr__(self, "command_seq", int(self.command_seq))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "result", _frozen_payload(self.result))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ack",
            "key": self.key.to_dict(),
            "attempt": int(self.attempt),
            "command_seq": int(self.command_seq),
            "rank": int(self.rank),
            "phase": int(self.phase),
            "lease_id": self.lease_id,
            "ok": bool(self.ok),
            "detail": self.detail,
            "result": dict(self.result),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RankAck":
        if value.get("type") != "ack":
            raise ValueError("frame is not a rank ACK")
        return cls(
            key=GenerationKey.from_dict(value["key"]),
            attempt=int(value["attempt"]),
            command_seq=int(value["command_seq"]),
            rank=int(value["rank"]),
            phase=RankPhase(int(value["phase"])),
            lease_id=str(value["lease_id"]),
            ok=bool(value.get("ok", True)),
            detail=str(value.get("detail", "")),
            result=value.get("result") or {},
        )


@dataclass(frozen=True, slots=True)
class RankDisconnected:
    group_id: str
    rank: int


@dataclass(frozen=True, slots=True)
class LinkParticipant:
    """One physical shard endpoint in a cross-engine TP transaction."""

    role: str
    endpoint_group: str
    rank: int

    def __post_init__(self) -> None:
        if not self.role or not self.endpoint_group or int(self.rank) < 0:
            raise ValueError("participant role/group and non-negative rank required")
        object.__setattr__(self, "rank", int(self.rank))


@dataclass(frozen=True, slots=True)
class LinkRankAck:
    """A physical fence ACK with an unambiguous endpoint identity."""

    participant: LinkParticipant
    ack: RankAck


@dataclass(frozen=True, slots=True)
class LinkDisconnected:
    link_id: str
    participant: LinkParticipant


@dataclass(frozen=True, slots=True)
class LinkIntent:
    """Endpoint-rank0 request for the link coordinator to make a decision."""

    link_id: str
    participant: LinkParticipant
    key: GenerationKey
    proposal_seq: int
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.link_id or int(self.proposal_seq) < 1 or not self.kind:
            raise ValueError("link, positive proposal sequence and kind required")
        if self.participant.rank != 0:
            raise ValueError("only an endpoint rank zero may propose an intent")
        object.__setattr__(self, "proposal_seq", int(self.proposal_seq))
        object.__setattr__(self, "payload", _frozen_payload(self.payload))


@dataclass(frozen=True, slots=True)
class LinkCapacityEdge:
    """Ephemeral D-capacity notification forwarded to the link coordinator.

    This is a wake-up hint, not request-generation state: it never enters the
    lifecycle ledger and may be coalesced.  ``session_id`` lets a freshly
    connected D rank zero restart its sequence without accepting stale frames
    from the previous TCP session.
    """

    link_id: str
    participant: LinkParticipant
    session_id: str
    edge_seq: int
    available_tokens: int

    def __post_init__(self) -> None:
        if not self.link_id or not self.session_id or int(self.edge_seq) < 1:
            raise ValueError("link, session and positive capacity sequence required")
        if self.participant.rank != 0 or self.participant.role != "decode":
            raise ValueError("only Decode endpoint rank zero may report capacity")
        if int(self.available_tokens) < 0:
            raise ValueError("available capacity must be non-negative")
        object.__setattr__(self, "edge_seq", int(self.edge_seq))
        object.__setattr__(self, "available_tokens", int(self.available_tokens))


class ReadinessPhase(str, Enum):
    """Ephemeral prerequisites for publishing a group transfer attempt."""

    REGISTERED = "registered"
    SOURCE_READY = "source_ready"


@dataclass(frozen=True, slots=True)
class LinkReadiness:
    """One rank-local readiness edge forwarded to the link coordinator.

    Readiness is deliberately outside the request-generation lifecycle ledger.
    It is a transient admission barrier: rank zero may publish PREPARE only
    after every participant required by the selected plan reported its local
    Req/source state.  Exact duplicates are harmless.
    """

    link_id: str
    participant: LinkParticipant
    key: GenerationKey
    phase: ReadinessPhase

    def __post_init__(self) -> None:
        if not self.link_id:
            raise ValueError("link ID is required")


@dataclass(frozen=True, slots=True)
class LinkFailure:
    """A rank-local failure report forwarded to the fixed link coordinator.

    This is deliberately not an ACK: a failed rank has not yet established a
    drained physical fence.  Rank zero must first publish CANCEL; every rank
    then acknowledges FAILED_DRAINED for that CANCEL command.
    """

    link_id: str
    participant: LinkParticipant
    key: GenerationKey
    attempt: int
    lease_id: str
    detail: str

    def __post_init__(self) -> None:
        if not self.link_id or int(self.attempt) < 1 or not self.lease_id:
            raise ValueError("link, positive attempt and lease ID required")
        if not self.detail:
            raise ValueError("failure detail must be non-empty")
        object.__setattr__(self, "attempt", int(self.attempt))


class TCPGroupRelayServer:
    """Standalone in-memory relay for one or more logical TP groups.

    The relay has no lifecycle authority.  It accepts commands only from a
    group's rank zero, pushes each command to every rank, and forwards rank
    ACKs or disconnect notices only to rank zero.  The rank-zero process owns
    :class:`GroupLifecycleCoordinator` and applies those events locally.
    """

    def __init__(
        self,
        *,
        run_id: str,
        token: str,
        groups: Optional[Mapping[str, int]] = None,
        links: Optional[Mapping[str, Mapping[str, Any]]] = None,
        address: tuple[str, int] = ("127.0.0.1", 0),
    ) -> None:
        normalized = {str(name): int(size) for name, size in (groups or {}).items()}
        if (
            not run_id
            or not token
            or (not normalized and not links)
            or any(not name or size < 1 for name, size in normalized.items())
        ):
            raise ValueError("run/token and non-empty positive TP groups are required")
        self.run_id = str(run_id)
        self.token = str(token)
        self.groups = MappingProxyType(normalized)
        link_specs: dict[str, dict[str, Any]] = {}
        for name, size in normalized.items():
            link_specs[name] = {
                "coordinator": (name, 0),
                "members": tuple(
                    LinkParticipant("group", name, rank) for rank in range(size)
                ),
                "sizes": {name: size},
            }
        for link_id, raw in (links or {}).items():
            if not link_id or not isinstance(raw, Mapping):
                raise ValueError("invalid link specification")
            endpoint_specs = raw.get("endpoints")
            coordinator = raw.get("coordinator")
            if not isinstance(endpoint_specs, list) or not isinstance(coordinator, Mapping):
                raise ValueError("link requires endpoints and coordinator")
            members: list[LinkParticipant] = []
            sizes: dict[str, int] = {}
            for endpoint in endpoint_specs:
                endpoint_group = str(endpoint["endpoint_group"])
                role = str(endpoint["role"])
                size = int(endpoint["size"])
                if not endpoint_group or not role or size < 1 or endpoint_group in sizes:
                    raise ValueError("invalid or duplicate link endpoint")
                sizes[endpoint_group] = size
                members.extend(
                    LinkParticipant(role, endpoint_group, rank)
                    for rank in range(size)
                )
            coordinator_id = (str(coordinator["endpoint_group"]), int(coordinator["rank"]))
            if not any(
                member.endpoint_group == coordinator_id[0]
                and member.rank == coordinator_id[1]
                for member in members
            ):
                raise ValueError("link coordinator is not a member")
            link_specs[str(link_id)] = {
                "coordinator": coordinator_id,
                "members": tuple(members),
                "sizes": sizes,
            }
        self.links = MappingProxyType(link_specs)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(address)
        self._socket.listen(sum(len(spec["members"]) for spec in link_specs.values()))
        self.address = self._socket.getsockname()
        self._sessions: dict[tuple[str, str, int], socket.socket] = {}
        self._send_locks: dict[tuple[str, str, int], threading.Lock] = {}
        self._intent_state: dict[tuple[str, str], tuple[int, tuple[Any, ...]]] = {}
        self._capacity_state: dict[
            tuple[str, str], tuple[str, int, int]
        ] = {}
        self._condition = threading.Condition()
        self._closed = False
        self._errors: list[BaseException] = []
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="agentic-group-relay-accept",
            daemon=True,
        )
        self._accept_thread.start()

    @property
    def errors(self) -> tuple[BaseException, ...]:
        with self._condition:
            return tuple(self._errors)

    def wait_for_error(self, timeout: Optional[float] = None) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: bool(self._errors), timeout)

    def _accept_loop(self) -> None:
        while True:
            try:
                connection, _ = self._socket.accept()
            except OSError:
                return
            _configure_socket(connection)
            threading.Thread(
                target=self._session_loop,
                args=(connection,),
                name="agentic-group-relay-session",
                daemon=True,
            ).start()

    def _send(self, identity: tuple[str, str, int], frame: Mapping[str, Any]) -> None:
        with self._condition:
            connection = self._sessions.get(identity)
            lock = self._send_locks.get(identity)
        if connection is None or lock is None:
            raise GroupDisconnectedError(
                f"link {identity[0]} endpoint {identity[1]} rank {identity[2]} is disconnected"
            )
        encoded = _encode_frame(frame)
        with lock:
            connection.sendall(encoded)

    def _broadcast(self, group_id: str, command: Mapping[str, Any]) -> None:
        spec = self.links[group_id]
        with self._condition:
            missing = [
                member
                for member in spec["members"]
                if (group_id, member.endpoint_group, member.rank)
                not in self._sessions
            ]
        if missing:
            raise GroupDisconnectedError(
                f"link {group_id} lacks participants {missing}"
            )
        for member in spec["members"]:
            self._send((group_id, member.endpoint_group, member.rank), command)

    def _notify_rank0_disconnect(
        self, group_id: str, endpoint_group: str, role: str, rank: int
    ) -> None:
        coordinator_group, coordinator_rank = self.links[group_id]["coordinator"]
        try:
            frame = {"type": "rank_disconnected", "group_id": group_id, "rank": rank}
            if group_id not in self.groups:
                frame.update(endpoint_group=endpoint_group, endpoint_role=role)
            self._send(
                (group_id, coordinator_group, coordinator_rank),
                frame,
            )
        except (OSError, GroupDisconnectedError):
            pass

    def _session_loop(self, connection: socket.socket) -> None:
        identity: Optional[tuple[str, str, int]] = None
        role = ""
        try:
            hello = _recv_frame(connection)
            if hello.get("type") != "hello":
                raise ValueError("first group protocol frame must be HELLO")
            group_id = str(hello.get("group_id", ""))
            endpoint_group = str(hello.get("endpoint_group") or group_id)
            role = str(hello.get("endpoint_role") or "group")
            spec = self.links.get(group_id)
            endpoint_size = None if spec is None else spec["sizes"].get(endpoint_group)
            if (
                hello.get("run_id") != self.run_id
                or hello.get("token") != self.token
                or spec is None
                or endpoint_size is None
                or int(hello.get("tp_size", -1)) != endpoint_size
            ):
                raise ValueError("rank HELLO does not match this relay")
            rank = int(hello["rank"])
            participant = LinkParticipant(role, endpoint_group, rank)
            if participant not in spec["members"]:
                raise ValueError("HELLO rank is outside its TP group")
            identity = (group_id, endpoint_group, rank)
            with self._condition:
                if identity in self._sessions:
                    raise ValueError("duplicate live rank session")
                self._sessions[identity] = connection
                self._send_locks[identity] = threading.Lock()
                self._condition.notify_all()
            connection.sendall(_encode_frame(
                {"type": "hello_ack", "rank": rank,
                 "endpoint_group": endpoint_group, "endpoint_role": role}
            ))
            while True:
                frame = _recv_frame(connection)
                frame_type = frame.get("type")
                if frame_type == "publish":
                    if (endpoint_group, rank) != spec["coordinator"]:
                        raise ValueError("only the link coordinator may publish commands")
                    command = GroupCommand.from_dict(frame["command"])
                    if command.key.run_id != self.run_id:
                        raise ValueError("command belongs to another run")
                    self._broadcast(group_id, command.to_dict())
                elif frame_type == "intent":
                    if rank != 0:
                        raise ValueError("only an endpoint rank zero may propose")
                    proposal_seq = int(frame["proposal_seq"])
                    key = GenerationKey.from_dict(frame["key"])
                    if key.run_id != self.run_id or proposal_seq < 1:
                        raise ValueError("invalid intent run or sequence")
                    signature = (
                        key,
                        str(frame["kind"]),
                        json.dumps(
                            frame.get("payload") or {}, sort_keys=True,
                            separators=(",", ":"), allow_nan=False,
                        ),
                    )
                    state_key = (group_id, endpoint_group)
                    with self._condition:
                        previous = self._intent_state.get(state_key)
                        if previous is not None and proposal_seq < previous[0]:
                            raise StaleAttemptError("stale endpoint intent")
                        if previous is not None and proposal_seq == previous[0]:
                            if signature != previous[1]:
                                raise GroupProtocolError(
                                    "duplicate intent changed its payload"
                                )
                            # Exact retransmission is idempotent: the original
                            # was already placed on the coordinator's TCP stream.
                            continue
                        self._intent_state[state_key] = (proposal_seq, signature)
                    coordinator_group, coordinator_rank = spec["coordinator"]
                    self._send(
                        (group_id, coordinator_group, coordinator_rank),
                        {
                            "type": "intent", "group_id": group_id,
                            "endpoint_group": endpoint_group,
                            "endpoint_role": role, "rank": rank,
                            "key": key.to_dict(), "proposal_seq": proposal_seq,
                            "kind": str(frame["kind"]),
                            "payload": frame.get("payload") or {},
                        },
                    )
                elif frame_type == "capacity_edge":
                    if rank != 0 or role != "decode":
                        raise ValueError(
                            "only Decode endpoint rank zero may report capacity"
                        )
                    session_id = str(frame.get("session_id", ""))
                    edge_seq = int(frame.get("edge_seq", 0))
                    available_tokens = int(frame.get("available_tokens", -1))
                    if not session_id or edge_seq < 1 or available_tokens < 0:
                        raise ValueError("invalid capacity edge")
                    state_key = (group_id, endpoint_group)
                    with self._condition:
                        previous = self._capacity_state.get(state_key)
                        if previous is not None and previous[0] == session_id:
                            if edge_seq < previous[1]:
                                raise StaleAttemptError("stale capacity edge")
                            if edge_seq == previous[1]:
                                if available_tokens != previous[2]:
                                    raise GroupProtocolError(
                                        "duplicate capacity edge changed its payload"
                                    )
                                continue
                        self._capacity_state[state_key] = (
                            session_id,
                            edge_seq,
                            available_tokens,
                        )
                    coordinator_group, coordinator_rank = spec["coordinator"]
                    self._send(
                        (group_id, coordinator_group, coordinator_rank),
                        {
                            "type": "capacity_edge",
                            "group_id": group_id,
                            "endpoint_group": endpoint_group,
                            "endpoint_role": role,
                            "rank": rank,
                            "session_id": session_id,
                            "edge_seq": edge_seq,
                            "available_tokens": available_tokens,
                        },
                    )
                elif frame_type == "readiness":
                    key = GenerationKey.from_dict(frame["key"])
                    phase = ReadinessPhase(str(frame["phase"]))
                    if key.run_id != self.run_id:
                        raise ValueError("readiness belongs to another run")
                    coordinator_group, coordinator_rank = spec["coordinator"]
                    self._send(
                        (group_id, coordinator_group, coordinator_rank),
                        {
                            "type": "readiness",
                            "group_id": group_id,
                            "endpoint_group": endpoint_group,
                            "endpoint_role": role,
                            "rank": rank,
                            "key": key.to_dict(),
                            "phase": phase.value,
                        },
                    )
                elif frame_type == "ack":
                    ack = RankAck.from_dict(frame)
                    if ack.rank != rank or ack.key.run_id != self.run_id:
                        raise ValueError("rank ACK identity mismatch")
                    forwarded = ack.to_dict()
                    if group_id not in self.groups:
                        forwarded.update(
                            endpoint_group=endpoint_group, endpoint_role=role
                        )
                    coordinator_group, coordinator_rank = spec["coordinator"]
                    self._send((group_id, coordinator_group, coordinator_rank), forwarded)
                elif frame_type == "failure":
                    key = GenerationKey.from_dict(frame["key"])
                    attempt = int(frame["attempt"])
                    lease_id = str(frame["lease_id"])
                    detail = str(frame["detail"])
                    if (
                        key.run_id != self.run_id
                        or attempt < 1
                        or not lease_id
                        or not detail
                    ):
                        raise ValueError("invalid rank-local failure")
                    coordinator_group, coordinator_rank = spec["coordinator"]
                    self._send(
                        (group_id, coordinator_group, coordinator_rank),
                        {
                            "type": "failure",
                            "group_id": group_id,
                            "endpoint_group": endpoint_group,
                            "endpoint_role": role,
                            "rank": rank,
                            "key": key.to_dict(),
                            "attempt": attempt,
                            "lease_id": lease_id,
                            "detail": detail,
                        },
                    )
                else:
                    raise ValueError("unsupported group relay frame")
        except (EOFError, OSError) as exc:
            if identity is not None and not self._closed:
                self._notify_rank0_disconnect(identity[0], identity[1], role, identity[2])
                with self._condition:
                    self._errors.append(exc)
                    self._condition.notify_all()
        except BaseException as exc:
            if identity is not None:
                self._notify_rank0_disconnect(identity[0], identity[1], role, identity[2])
            with self._condition:
                self._errors.append(exc)
                self._condition.notify_all()
        finally:
            with self._condition:
                if identity is not None and self._sessions.get(identity) is connection:
                    self._sessions.pop(identity, None)
                    self._send_locks.pop(identity, None)
                    if identity[2] == 0:
                        self._capacity_state.pop((identity[0], identity[1]), None)
                self._condition.notify_all()
            try:
                connection.close()
            except OSError:
                pass

    def wait_connected(
        self, group_id: Optional[str] = None, timeout: Optional[float] = None
    ) -> bool:
        if group_id is not None and group_id not in self.links:
            raise ValueError("unknown TP group")

        def ready() -> bool:
            selected = self.links if group_id is None else {group_id: self.links[group_id]}
            return all(
                (name, member.endpoint_group, member.rank) in self._sessions
                for name, spec in selected.items()
                for member in spec["members"]
            )

        with self._condition:
            return self._condition.wait_for(ready, timeout=timeout)

    def close(self) -> None:
        self._closed = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        with self._condition:
            sessions = tuple(self._sessions.values())
        for connection in sessions:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self._accept_thread.join(timeout=1)


@dataclass(slots=True)
class _Attempt:
    key: GenerationKey
    attempt: int
    source_owner: Owner
    target_owner: Owner
    lease_id: str
    tp_size: int
    required_commit_phase: RankPhase
    command_seq: int = 0
    last_command: Optional[CommandKind] = None
    outcome: AttemptOutcome = AttemptOutcome.ACTIVE
    rank_phases: dict[int, RankPhase] = field(default_factory=dict)
    ack_signatures: dict[tuple[int, RankPhase], tuple[Any, ...]] = field(
        default_factory=dict
    )
    disconnected: set[int] = field(default_factory=set)
    abort_reason: str = ""

    def command(
        self, kind: CommandKind, payload: Optional[Mapping[str, Any]] = None
    ) -> GroupCommand:
        self.command_seq += 1
        self.last_command = kind
        return GroupCommand(
            key=self.key,
            attempt=self.attempt,
            command_seq=self.command_seq,
            kind=kind,
            source_owner=self.source_owner,
            target_owner=self.target_owner,
            lease_id=self.lease_id,
            payload=payload or {},
        )


class GroupLifecycleCoordinator:
    """Rank-zero logical authority for one TP group.

    Active rank progress is intentionally short lived.  After retirement only
    :class:`GenerationRecord` remains, which is enough to reject old attempts
    and identify the current logical owner without retaining ACK history.
    """

    def __init__(self, run_id: str, group_id: str, tp_size: int):
        if not run_id or not group_id or int(tp_size) < 1:
            raise ValueError("run_id, group_id and positive tp_size are required")
        self.run_id = str(run_id)
        self.group_id = str(group_id)
        self.tp_size = int(tp_size)
        self._records: dict[GenerationKey, GenerationRecord] = {}
        self._active: dict[GenerationKey, _Attempt] = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)

    def register(self, key: GenerationKey, owner: Owner) -> GenerationRecord:
        if key.run_id != self.run_id:
            raise ValueError("generation belongs to another run")
        with self._changed:
            current = self._records.get(key)
            if current is not None:
                if current.owner is not owner:
                    raise GroupProtocolError("generation already has a different owner")
                return current
            record = GenerationRecord(key=key, attempt=0, owner=owner)
            self._records[key] = record
            self._changed.notify_all()
            return record

    def record(self, key: GenerationKey) -> Optional[GenerationRecord]:
        with self._lock:
            return self._records.get(key)

    def active_attempt(self, key: GenerationKey) -> Optional[int]:
        with self._lock:
            attempt = self._active.get(key)
            return None if attempt is None else attempt.attempt

    def begin_attempt(
        self,
        key: GenerationKey,
        *,
        source_owner: Owner,
        target_owner: Owner,
        lease_id: str,
        required_commit_phase: RankPhase = RankPhase.DMA_DONE,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        if required_commit_phase not in {
            RankPhase.DMA_DONE,
            RankPhase.BOUND,
            RankPhase.RELEASED,
        }:
            raise ValueError("commit phase must be DMA_DONE, BOUND or RELEASED")
        with self._changed:
            record = self._records.get(key)
            if record is None:
                record = self.register(key, source_owner)
            if record.terminal is not GenerationTerminal.OPEN:
                raise GroupProtocolError("terminal generation cannot start an attempt")
            if record.owner is not source_owner:
                raise GroupProtocolError("attempt source does not own the generation")
            if key in self._active:
                raise GroupProtocolError("generation already has an active attempt")
            next_attempt = record.attempt + 1
            attempt = _Attempt(
                key=key,
                attempt=next_attempt,
                source_owner=source_owner,
                target_owner=target_owner,
                lease_id=str(lease_id),
                tp_size=self.tp_size,
                required_commit_phase=required_commit_phase,
            )
            self._records[key] = GenerationRecord(
                key=key,
                attempt=next_attempt,
                owner=record.owner,
                terminal=record.terminal,
            )
            self._active[key] = attempt
            command = attempt.command(CommandKind.PREPARE, payload)
            self._changed.notify_all()
            return command

    def _require_active(self, key: GenerationKey, attempt_id: int) -> _Attempt:
        record = self._records.get(key)
        if record is None:
            raise StaleAttemptError("unknown generation")
        active = self._active.get(key)
        if active is None or int(attempt_id) != active.attempt:
            if int(attempt_id) <= record.attempt:
                raise StaleAttemptError("attempt is retired or superseded")
            raise GroupProtocolError("attempt is newer than rank-zero authority")
        return active

    def issue_start(
        self, key: GenerationKey, attempt_id: int, payload: Optional[Mapping[str, Any]] = None
    ) -> GroupCommand:
        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ACTIVE:
                raise GroupProtocolError("only an active attempt may start DMA")
            if not self._all_at_least(attempt, RankPhase.PREPARED):
                raise GroupNotReadyError("not every rank is prepared")
            command = attempt.command(CommandKind.START, payload)
            self._changed.notify_all()
            return command

    def request_abort(
        self, key: GenerationKey, attempt_id: int, reason: str
    ) -> GroupCommand:
        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is AttemptOutcome.COMMITTED:
                raise GroupProtocolError("a committed attempt cannot be cancelled")
            if attempt.outcome is AttemptOutcome.ABORTED:
                raise GroupProtocolError("attempt is already aborted")
            attempt.outcome = AttemptOutcome.ABORTING
            attempt.abort_reason = str(reason)
            command = attempt.command(CommandKind.CANCEL, {"reason": str(reason)})
            self._changed.notify_all()
            return command

    def issue_abort_finalize(
        self,
        key: GenerationKey,
        attempt_id: int,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        """Finalize rollback only after every participant proved local drain."""

        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ABORTED:
                raise GroupNotReadyError("abort finalization requires drained abort")
            if not all(
                attempt.rank_phases.get(rank) is RankPhase.FAILED_DRAINED
                for rank in range(self.tp_size)
            ):
                raise GroupNotReadyError("not every rank drained its local transfer")
            command = attempt.command(CommandKind.ABORT_FINALIZE, payload)
            self._changed.notify_all()
            return command

    def issue_release(
        self,
        key: GenerationKey,
        attempt_id: int,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ACTIVE:
                raise GroupNotReadyError("handoff preparation requires an active attempt")
            if not self._all_at_least(attempt, RankPhase.DMA_DONE):
                raise GroupNotReadyError("handoff preparation requires every DMA fence")
            command = attempt.command(CommandKind.RELEASE, payload)
            self._changed.notify_all()
            return command

    def issue_handoff(
        self,
        key: GenerationKey,
        attempt_id: int,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        """Stage activation after every local bind without publishing ready.

        ``RELEASE`` prepares each rank-local handoff but deliberately exposes
        nothing.  Only after every rank reports ``BOUND`` may rank zero send
        this command.  Participants acknowledge receipt as ``RELEASED`` but
        still expose no ready queue/source release until ``ACTIVATE``.
        """

        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ACTIVE:
                raise GroupNotReadyError("handoff staging requires an active attempt")
            if not self._all_at_least(attempt, RankPhase.BOUND):
                raise GroupNotReadyError("not every rank prepared its handoff")
            command = attempt.command(CommandKind.HANDOFF, payload)
            self._changed.notify_all()
            return command

    def issue_activate(
        self,
        key: GenerationKey,
        attempt_id: int,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        """Stage local activation only after every participant bound locally."""

        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ACTIVE:
                raise GroupNotReadyError("activation staging requires an active attempt")
            if not self._all_at_least(attempt, RankPhase.RELEASED):
                raise GroupNotReadyError("not every rank staged activation")
            command = attempt.command(CommandKind.ACTIVATE, payload)
            self._changed.notify_all()
            return command

    def issue_scheduler_activate(
        self,
        key: GenerationKey,
        attempt_id: int,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        """Create the endpoint-TP scheduler ticket after all ranks are staged."""

        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ACTIVE:
                raise GroupNotReadyError("scheduler activation requires an active attempt")
            if not self._all_at_least(attempt, RankPhase.STAGED):
                raise GroupNotReadyError("not every rank staged scheduler activation")
            command = attempt.command(CommandKind.SCHEDULER_ACTIVATE, payload)
            self._changed.notify_all()
            return command

    def issue_publish_activation(
        self,
        key: GenerationKey,
        attempt_id: int,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        """Publish a scheduler ticket only after every rank armed the attempt."""

        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ACTIVE:
                raise GroupNotReadyError("activation publication requires an active attempt")
            if not self._all_at_least(attempt, RankPhase.ACTIVATION_ARMED):
                raise GroupNotReadyError("not every rank armed scheduler activation")
            command = attempt.command(CommandKind.PUBLISH_ACTIVATION, payload)
            self._changed.notify_all()
            return command

    def issue_activation_ticket(
        self,
        key: GenerationKey,
        attempt_id: int,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        """Release TP0's ticket after every rank can accept its broadcast."""

        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ACTIVE:
                raise GroupNotReadyError("ticket issue requires an active attempt")
            if not self._all_at_least(attempt, RankPhase.ACTIVATION_READY):
                raise GroupNotReadyError("not every rank is activation-ready")
            command = attempt.command(CommandKind.ISSUE_ACTIVATION_TICKET, payload)
            self._changed.notify_all()
            return command

    def issue_finalize(
        self,
        key: GenerationKey,
        attempt_id: int,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> GroupCommand:
        """Release the old owner only after target scheduler adoption succeeded."""

        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.COMMITTED:
                raise GroupNotReadyError("finalization requires group commit")
            if not self._all_at_least(attempt, RankPhase.ACTIVATED):
                raise GroupNotReadyError("not every rank passed scheduler activation")
            command = attempt.command(CommandKind.FINALIZE, payload)
            self._changed.notify_all()
            return command

    @staticmethod
    def _all_at_least(attempt: _Attempt, phase: RankPhase) -> bool:
        for rank in range(attempt.tp_size):
            observed = attempt.rank_phases.get(rank, RankPhase.NONE)
            if observed is RankPhase.FAILED_DRAINED or observed < phase:
                return False
        return True

    def group_reached(
        self, key: GenerationKey, attempt_id: int, phase: RankPhase
    ) -> bool:
        with self._lock:
            attempt = self._require_active(key, attempt_id)
            if phase is RankPhase.FAILED_DRAINED:
                return all(
                    attempt.rank_phases.get(rank) is RankPhase.FAILED_DRAINED
                    for rank in range(self.tp_size)
                )
            return self._all_at_least(attempt, phase)

    def apply_ack(self, ack: RankAck) -> bool:
        """Apply one ACK; return ``False`` only for an exact duplicate."""

        with self._changed:
            attempt = self._require_active(ack.key, ack.attempt)
            if not 0 <= ack.rank < self.tp_size:
                raise GroupProtocolError("ACK rank is outside this TP group")
            if ack.lease_id != attempt.lease_id:
                raise GroupProtocolError("ACK lease does not match the attempt")
            if ack.command_seq > attempt.command_seq:
                raise GroupProtocolError("ACK refers to an unissued command")
            expected_command = {
                RankPhase.PREPARED: CommandKind.PREPARE,
                RankPhase.DMA_SUBMITTED: CommandKind.START,
                RankPhase.DMA_DONE: CommandKind.START,
                RankPhase.BOUND: CommandKind.RELEASE,
                RankPhase.RELEASED: CommandKind.HANDOFF,
                RankPhase.STAGED: CommandKind.ACTIVATE,
                RankPhase.ACTIVATION_ARMED: CommandKind.SCHEDULER_ACTIVATE,
                RankPhase.ACTIVATION_READY: CommandKind.PUBLISH_ACTIVATION,
                RankPhase.ACTIVATED: CommandKind.ISSUE_ACTIVATION_TICKET,
                RankPhase.FINALIZED: CommandKind.FINALIZE,
                RankPhase.FAILED_DRAINED: CommandKind.CANCEL,
                RankPhase.ABORTED: CommandKind.ABORT_FINALIZE,
            }[ack.phase]
            if (
                attempt.last_command is expected_command
                and ack.command_seq != attempt.command_seq
            ):
                raise GroupProtocolError("ACK command sequence is stale")
            signature_key = (ack.rank, ack.phase)
            signature = (
                ack.command_seq,
                ack.lease_id,
                bool(ack.ok),
                ack.detail,
                json.dumps(
                    dict(ack.result),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
            previous_signature = attempt.ack_signatures.get(signature_key)
            if previous_signature is not None:
                if previous_signature != signature:
                    raise GroupProtocolError("duplicate ACK changed its payload")
                return False

            previous = attempt.rank_phases.get(ack.rank, RankPhase.NONE)
            if ack.phase is RankPhase.FAILED_DRAINED:
                if attempt.outcome is not AttemptOutcome.ABORTING:
                    raise GroupProtocolError("drained failure requires rank-zero CANCEL")
            elif ack.phase is RankPhase.ABORTED:
                if attempt.outcome is not AttemptOutcome.ABORTED:
                    raise GroupProtocolError(
                        "abort finalization requires the all-rank drain barrier"
                    )
                previous = attempt.rank_phases.get(ack.rank, RankPhase.NONE)
                if previous is not RankPhase.FAILED_DRAINED:
                    raise GroupProtocolError(
                        "rank finalized abort before its drained fence"
                    )
            else:
                if (
                    attempt.outcome is AttemptOutcome.ABORTING
                    and ack.command_seq < attempt.command_seq
                ):
                    # PREPARE/START ACKs already in another rank's TCP stream
                    # may arrive after rank zero publishes CANCEL.  They cannot
                    # advance or commit the attempt; cancellation completion is
                    # proven exclusively by the later FAILED_DRAINED ACK.
                    return False
                if attempt.outcome is not AttemptOutcome.ACTIVE and not (
                    attempt.outcome is AttemptOutcome.COMMITTED
                    and ack.phase in {RankPhase.FINALIZED}
                ):
                    raise GroupProtocolError("normal progress after cancellation is forbidden")
                expected = {
                    RankPhase.PREPARED: RankPhase.NONE,
                    RankPhase.DMA_SUBMITTED: RankPhase.PREPARED,
                    RankPhase.DMA_DONE: RankPhase.DMA_SUBMITTED,
                    RankPhase.BOUND: RankPhase.DMA_DONE,
                }.get(ack.phase)
                if ack.phase is RankPhase.RELEASED:
                    if previous is not RankPhase.BOUND:
                        raise GroupProtocolError(
                            "rank staged handoff before its local bind"
                        )
                elif ack.phase is RankPhase.STAGED:
                    if previous is not RankPhase.RELEASED:
                        raise GroupProtocolError(
                            "rank staged before the all-rank release barrier"
                        )
                elif ack.phase is RankPhase.ACTIVATED:
                    if previous is not RankPhase.ACTIVATION_READY:
                        raise GroupProtocolError(
                            "rank activated before the all-rank ready barrier"
                        )
                elif ack.phase is RankPhase.ACTIVATION_READY:
                    if previous is not RankPhase.ACTIVATION_ARMED:
                        raise GroupProtocolError(
                            "rank became activation-ready before arming"
                        )
                elif ack.phase is RankPhase.ACTIVATION_ARMED:
                    if previous is not RankPhase.STAGED:
                        raise GroupProtocolError(
                            "rank armed activation before its local staged lease"
                        )
                elif ack.phase is RankPhase.FINALIZED:
                    if attempt.outcome is not AttemptOutcome.COMMITTED:
                        raise GroupProtocolError("finalize ACK arrived before group commit")
                    if previous is not RankPhase.ACTIVATED:
                        raise GroupProtocolError(
                            "rank finalized before scheduler activation"
                        )
                elif expected is None or previous is not expected:
                    raise GroupProtocolError(
                        f"rank {ack.rank} phase jumped from {previous.name} "
                        f"to {ack.phase.name}"
                    )
            attempt.rank_phases[ack.rank] = ack.phase
            attempt.ack_signatures[signature_key] = signature
            self._changed.notify_all()
            return True

    def commit(self, key: GenerationKey, attempt_id: int) -> GenerationRecord:
        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.disconnected:
                raise GroupDisconnectedError("rank disconnected before commit")
            if attempt.outcome is not AttemptOutcome.ACTIVE:
                raise GroupProtocolError("attempt is not commit eligible")
            if not self._all_at_least(attempt, attempt.required_commit_phase):
                raise GroupNotReadyError("not every rank reached the commit fence")
            current = self._records[key]
            record = GenerationRecord(
                key=key,
                attempt=attempt.attempt,
                owner=attempt.target_owner,
                terminal=current.terminal,
            )
            self._records[key] = record
            attempt.outcome = AttemptOutcome.COMMITTED
            self._changed.notify_all()
            return record

    def complete_abort(self, key: GenerationKey, attempt_id: int) -> GenerationRecord:
        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is not AttemptOutcome.ABORTING:
                raise GroupProtocolError("attempt is not aborting")
            if attempt.disconnected:
                raise GroupDisconnectedError("disconnected rank has no drained fence")
            if not all(
                attempt.rank_phases.get(rank) is RankPhase.FAILED_DRAINED
                for rank in range(self.tp_size)
            ):
                raise GroupNotReadyError("not every rank drained its local transfer")
            attempt.outcome = AttemptOutcome.ABORTED
            self._changed.notify_all()
            return self._records[key]

    def retire(self, key: GenerationKey, attempt_id: int) -> None:
        with self._changed:
            attempt = self._require_active(key, attempt_id)
            if attempt.outcome is AttemptOutcome.COMMITTED:
                ready = all(
                    attempt.rank_phases.get(rank) is RankPhase.FINALIZED
                    for rank in range(self.tp_size)
                )
            elif attempt.outcome is AttemptOutcome.ABORTED:
                ready = all(
                    attempt.rank_phases.get(rank) is RankPhase.ABORTED
                    for rank in range(self.tp_size)
                )
            else:
                ready = False
            if not ready:
                raise GroupNotReadyError("attempt still owns physical resources")
            del self._active[key]
            self._changed.notify_all()

    def mark_terminal(
        self, key: GenerationKey, terminal: GenerationTerminal
    ) -> GenerationRecord:
        if terminal is GenerationTerminal.OPEN:
            raise ValueError("OPEN is not a terminal outcome")
        with self._changed:
            if key in self._active:
                raise GroupNotReadyError("active attempt must retire before terminal state")
            current = self._records[key]
            record = GenerationRecord(
                key=key,
                attempt=current.attempt,
                owner=Owner.NONE,
                terminal=terminal,
            )
            self._records[key] = record
            self._changed.notify_all()
            return record

    def mark_disconnected(self, rank: int) -> None:
        if not 0 <= int(rank) < self.tp_size:
            raise ValueError("rank is outside this TP group")
        with self._changed:
            for attempt in self._active.values():
                attempt.disconnected.add(int(rank))
                if attempt.outcome is AttemptOutcome.ACTIVE:
                    attempt.outcome = AttemptOutcome.ABORTING
                    attempt.abort_reason = "rank_disconnected"
            self._changed.notify_all()

    def apply_event(self, event: RankAck | RankDisconnected) -> bool:
        """Apply one event received by rank zero from the standalone relay."""

        if isinstance(event, RankAck):
            return self.apply_ack(event)
        if event.group_id != self.group_id:
            raise GroupProtocolError("disconnect notice belongs to another TP group")
        self.mark_disconnected(event.rank)
        return True

    def outcome(self, key: GenerationKey, attempt_id: int) -> AttemptOutcome:
        with self._lock:
            return self._require_active(key, attempt_id).outcome

    def wait_for(
        self, predicate: Callable[[], bool], timeout: Optional[float] = None
    ) -> bool:
        with self._changed:
            return self._changed.wait_for(predicate, timeout=timeout)


class LinkLifecycleCoordinator:
    """One ownership transaction spanning source and target TP groups.

    This is deliberately one coordinator/attempt, not two group commits.  The
    wrapped coordinator sees every endpoint shard as one participant slot, so
    commit, abort and release require a complete cross-endpoint fence.
    """

    def __init__(
        self,
        run_id: str,
        link_id: str,
        participants: list[LinkParticipant] | tuple[LinkParticipant, ...],
    ) -> None:
        ordered = tuple(participants)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("link participants must be non-empty and unique")
        self.run_id = str(run_id)
        self.link_id = str(link_id)
        self.participants = ordered
        self._slots = {participant: slot for slot, participant in enumerate(ordered)}
        self._coordinator = GroupLifecycleCoordinator(
            run_id, link_id, len(ordered)
        )

    @classmethod
    def from_endpoint_sizes(
        cls,
        run_id: str,
        link_id: str,
        endpoints: list[tuple[str, str, int]],
    ) -> "LinkLifecycleCoordinator":
        participants = [
            LinkParticipant(role, endpoint_group, rank)
            for role, endpoint_group, size in endpoints
            for rank in range(int(size))
        ]
        return cls(run_id, link_id, participants)

    def __getattr__(self, name: str) -> Any:
        # Lifecycle creation/commit/retire APIs are exactly the proven group
        # implementation; only ACK/disconnect identity needs translation.
        return getattr(self._coordinator, name)

    def apply_ack(self, event: LinkRankAck) -> bool:
        try:
            slot = self._slots[event.participant]
        except KeyError as exc:
            raise GroupProtocolError("ACK participant is outside this link") from exc
        ack = event.ack
        return self._coordinator.apply_ack(
            RankAck(
                key=ack.key,
                attempt=ack.attempt,
                command_seq=ack.command_seq,
                rank=slot,
                phase=ack.phase,
                lease_id=ack.lease_id,
                ok=ack.ok,
                detail=ack.detail,
                result=ack.result,
            )
        )

    def apply_event(self, event: LinkRankAck | LinkDisconnected) -> bool:
        if isinstance(event, LinkRankAck):
            return self.apply_ack(event)
        if event.link_id != self.link_id:
            raise GroupProtocolError("disconnect belongs to another link")
        try:
            slot = self._slots[event.participant]
        except KeyError as exc:
            raise GroupProtocolError("disconnect participant is outside this link") from exc
        self._coordinator.mark_disconnected(slot)
        return True


def _encode_frame(value: Mapping[str, Any]) -> bytes:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if not 0 < len(payload) <= _MAX_FRAME_BYTES:
        raise ValueError("invalid group protocol frame size")
    return struct.pack("!I", len(payload)) + payload


def _read_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise EOFError("group protocol TCP session closed")
        result.extend(chunk)
    return bytes(result)


def _recv_frame(sock: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _read_exact(sock, 4))[0]
    if not 0 < size <= _MAX_FRAME_BYTES:
        raise ValueError("invalid group protocol frame size")
    value = json.loads(_read_exact(sock, size))
    if not isinstance(value, dict):
        raise ValueError("group protocol frame must be an object")
    return value


def _configure_socket(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


class TCPRankZeroServer:
    """Push-only TCP transport owned by TP rank zero.

    All ranks, including rank zero's local RankAgent, connect through the same
    interface.  This keeps TP=1 and TP>1 behavior identical and makes the
    transport straightforward to test without model collectives.
    """

    def __init__(
        self,
        coordinator: GroupLifecycleCoordinator,
        *,
        token: str,
        address: tuple[str, int] = ("127.0.0.1", 0),
    ) -> None:
        if not token:
            raise ValueError("a non-empty session token is required")
        self.coordinator = coordinator
        self.token = str(token)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(address)
        self._socket.listen(coordinator.tp_size)
        self.address = self._socket.getsockname()
        self._sessions: dict[int, socket.socket] = {}
        self._session_locks: dict[int, threading.Lock] = {}
        self._errors: list[BaseException] = []
        self._condition = threading.Condition()
        self._closed = False
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="agentic-group-accept", daemon=True
        )
        self._accept_thread.start()

    @property
    def errors(self) -> tuple[BaseException, ...]:
        with self._condition:
            return tuple(self._errors)

    def _accept_loop(self) -> None:
        while True:
            try:
                connection, _ = self._socket.accept()
            except OSError:
                return
            _configure_socket(connection)
            threading.Thread(
                target=self._session_loop,
                args=(connection,),
                name="agentic-group-session",
                daemon=True,
            ).start()

    def _session_loop(self, connection: socket.socket) -> None:
        rank: Optional[int] = None
        try:
            hello = _recv_frame(connection)
            if hello.get("type") != "hello":
                raise ValueError("first group protocol frame must be HELLO")
            if (
                hello.get("run_id") != self.coordinator.run_id
                or hello.get("group_id") != self.coordinator.group_id
                or hello.get("token") != self.token
                or int(hello.get("tp_size", -1)) != self.coordinator.tp_size
            ):
                raise ValueError("rank HELLO does not match this TP group")
            rank = int(hello["rank"])
            if not 0 <= rank < self.coordinator.tp_size:
                raise ValueError("HELLO rank is outside this TP group")
            with self._condition:
                if rank in self._sessions:
                    raise ValueError("duplicate live rank session")
                self._sessions[rank] = connection
                self._session_locks[rank] = threading.Lock()
                self._condition.notify_all()
            connection.sendall(
                _encode_frame(
                    {
                        "type": "hello_ack",
                        "rank": rank,
                        "endpoint_group": str(
                            hello.get("endpoint_group") or self.coordinator.group_id
                        ),
                        "endpoint_role": str(hello.get("endpoint_role") or "group"),
                    }
                )
            )
            while True:
                frame = _recv_frame(connection)
                ack = RankAck.from_dict(frame)
                if ack.rank != rank:
                    raise ValueError("rank cannot ACK for another shard")
                self.coordinator.apply_ack(ack)
        except (EOFError, OSError) as exc:
            if rank is not None and not self._closed:
                self.coordinator.mark_disconnected(rank)
                with self._condition:
                    self._errors.append(exc)
                    self._condition.notify_all()
        except BaseException as exc:
            if rank is not None:
                self.coordinator.mark_disconnected(rank)
            with self._condition:
                self._errors.append(exc)
                self._condition.notify_all()
        finally:
            with self._condition:
                if rank is not None and self._sessions.get(rank) is connection:
                    self._sessions.pop(rank, None)
                    self._session_locks.pop(rank, None)
                self._condition.notify_all()
            try:
                connection.close()
            except OSError:
                pass

    def wait_connected(self, timeout: Optional[float] = None) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self._sessions) == self.coordinator.tp_size,
                timeout=timeout,
            )

    def broadcast(self, command: GroupCommand) -> None:
        if command.key.run_id != self.coordinator.run_id:
            raise ValueError("command belongs to another run")
        with self._condition:
            if len(self._sessions) != self.coordinator.tp_size:
                raise GroupDisconnectedError("not every TP rank is connected")
            sessions = tuple(
                (rank, self._sessions[rank], self._session_locks[rank])
                for rank in range(self.coordinator.tp_size)
            )
        frame = _encode_frame(command.to_dict())
        for rank, connection, lock in sessions:
            try:
                with lock:
                    connection.sendall(frame)
            except OSError:
                self.coordinator.mark_disconnected(rank)
                raise GroupDisconnectedError(
                    f"rank {rank} disconnected while publishing a command"
                )

    def close(self) -> None:
        self._closed = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        with self._condition:
            sessions = tuple(self._sessions.values())
        for connection in sessions:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self._accept_thread.join(timeout=1)


class TCPRankAgent:
    """Blocking rank-side command channel for a dedicated control thread."""

    def __init__(
        self,
        address: tuple[str, int],
        *,
        run_id: str,
        group_id: str,
        token: str,
        rank: int,
        tp_size: int,
        endpoint_group: Optional[str] = None,
        endpoint_role: str = "group",
        timeout: float = 5.0,
    ) -> None:
        self.run_id = str(run_id)
        self.group_id = str(group_id)
        self.rank = int(rank)
        self.tp_size = int(tp_size)
        self.endpoint_group = str(endpoint_group or group_id)
        self.endpoint_role = str(endpoint_role)
        self._capacity_session_id = uuid.uuid4().hex
        self._capacity_sequence = 0
        self._socket = socket.create_connection(address, timeout=timeout)
        self._socket.settimeout(None)
        _configure_socket(self._socket)
        self._send_lock = threading.Lock()
        self._socket.sendall(
            _encode_frame(
                {
                    "type": "hello",
                    "run_id": self.run_id,
                    "group_id": self.group_id,
                    "token": str(token),
                    "rank": self.rank,
                    "tp_size": self.tp_size,
                    "endpoint_group": self.endpoint_group,
                    "endpoint_role": self.endpoint_role,
                }
            )
        )
        reply = _recv_frame(self._socket)
        expected_reply = {
            "type": "hello_ack",
            "rank": self.rank,
            "endpoint_group": self.endpoint_group,
            "endpoint_role": self.endpoint_role,
        }
        if reply != expected_reply:
            self._socket.close()
            raise GroupProtocolError("rank HELLO was rejected")

    @classmethod
    def from_env(
        cls,
        *,
        rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        prefix: str = "SGLANG_AGENTIC_GROUP_",
    ) -> "TCPRankAgent":
        """Create a rank session without any shared control-root setting.

        ``sglang.launch_server`` starts all TP scheduler children from one
        parent environment, so callers should pass the scheduler's real
        ``tp_rank`` explicitly.  Reading ``RANK`` remains available only for
        launchers that genuinely create one process environment per rank.
        """

        def required(name: str) -> str:
            value = os.getenv(prefix + name, "").strip()
            if not value:
                raise ValueError(f"{prefix + name} is required")
            return value

        endpoint = required("ENDPOINT").removeprefix("tcp://")
        host, port = endpoint.rsplit(":", 1)
        configured_size = int(required("SIZE"))
        if tp_size is not None and int(tp_size) != configured_size:
            raise ValueError("runtime TP size does not match group control SIZE")
        selected_rank = int(required("RANK")) if rank is None else int(rank)
        return cls(
            (host, int(port)),
            run_id=required("RUN_ID"),
            group_id=required("GROUP_ID"),
            token=required("TOKEN"),
            rank=selected_rank,
            tp_size=configured_size,
            endpoint_group=os.getenv(prefix + "ENDPOINT_GROUP") or required("GROUP_ID"),
            endpoint_role=os.getenv(prefix + "ENDPOINT_ROLE", "group"),
        )

    def receive(self) -> GroupCommand:
        return GroupCommand.from_dict(_recv_frame(self._socket))

    def receive_event(
        self,
    ) -> GroupCommand | RankAck | RankDisconnected | LinkRankAck | LinkDisconnected | LinkIntent | LinkCapacityEdge | LinkReadiness | LinkFailure:
        """Receive a relay event; rank zero gets ACKs and disconnect notices."""

        frame = _recv_frame(self._socket)
        frame_type = frame.get("type")
        if frame_type == "command":
            return GroupCommand.from_dict(frame)
        if frame_type == "ack":
            if self.rank != 0:
                raise GroupProtocolError("only rank zero may receive group ACKs")
            ack = RankAck.from_dict(frame)
            if "endpoint_group" in frame:
                return LinkRankAck(
                    LinkParticipant(
                        str(frame["endpoint_role"]),
                        str(frame["endpoint_group"]),
                        ack.rank,
                    ),
                    ack,
                )
            return ack
        if frame_type == "rank_disconnected":
            if self.rank != 0:
                raise GroupProtocolError(
                    "only rank zero may receive disconnect notices"
                )
            if "endpoint_group" in frame:
                return LinkDisconnected(
                    link_id=str(frame["group_id"]),
                    participant=LinkParticipant(
                        str(frame["endpoint_role"]),
                        str(frame["endpoint_group"]),
                        int(frame["rank"]),
                    ),
                )
            return RankDisconnected(group_id=str(frame["group_id"]), rank=int(frame["rank"]))
        if frame_type == "intent":
            if self.rank != 0:
                raise GroupProtocolError("only endpoint rank zero may receive an intent")
            return LinkIntent(
                link_id=str(frame["group_id"]),
                participant=LinkParticipant(
                    str(frame["endpoint_role"]),
                    str(frame["endpoint_group"]),
                    int(frame["rank"]),
                ),
                key=GenerationKey.from_dict(frame["key"]),
                proposal_seq=int(frame["proposal_seq"]),
                kind=str(frame["kind"]),
                payload=frame.get("payload") or {},
            )
        if frame_type == "capacity_edge":
            if self.rank != 0:
                raise GroupProtocolError(
                    "only link rank zero may receive a capacity edge"
                )
            return LinkCapacityEdge(
                link_id=str(frame["group_id"]),
                participant=LinkParticipant(
                    str(frame["endpoint_role"]),
                    str(frame["endpoint_group"]),
                    int(frame["rank"]),
                ),
                session_id=str(frame["session_id"]),
                edge_seq=int(frame["edge_seq"]),
                available_tokens=int(frame["available_tokens"]),
            )
        if frame_type == "readiness":
            if self.rank != 0:
                raise GroupProtocolError(
                    "only link rank zero may receive a readiness edge"
                )
            return LinkReadiness(
                link_id=str(frame["group_id"]),
                participant=LinkParticipant(
                    str(frame["endpoint_role"]),
                    str(frame["endpoint_group"]),
                    int(frame["rank"]),
                ),
                key=GenerationKey.from_dict(frame["key"]),
                phase=ReadinessPhase(str(frame["phase"])),
            )
        if frame_type == "failure":
            if self.rank != 0:
                raise GroupProtocolError("only link rank zero may receive a failure")
            return LinkFailure(
                link_id=str(frame["group_id"]),
                participant=LinkParticipant(
                    str(frame["endpoint_role"]),
                    str(frame["endpoint_group"]),
                    int(frame["rank"]),
                ),
                key=GenerationKey.from_dict(frame["key"]),
                attempt=int(frame["attempt"]),
                lease_id=str(frame["lease_id"]),
                detail=str(frame["detail"]),
            )
        raise GroupProtocolError("unsupported relay event")

    def propose_intent(
        self,
        key: GenerationKey,
        proposal_seq: int,
        kind: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Ask the fixed link coordinator to decide; never changes ownership."""

        if self.rank != 0:
            raise GroupProtocolError("only an endpoint rank zero may propose")
        if key.run_id != self.run_id or int(proposal_seq) < 1 or not kind:
            raise ValueError("invalid intent")
        frame = {
            "type": "intent", "key": key.to_dict(),
            "proposal_seq": int(proposal_seq), "kind": str(kind),
            "payload": dict(_frozen_payload(payload)),
        }
        with self._send_lock:
            self._socket.sendall(_encode_frame(frame))

    def publish(self, command: GroupCommand) -> None:
        if self.rank != 0:
            raise GroupProtocolError("only TP rank zero may publish commands")
        if command.key.run_id != self.run_id:
            raise ValueError("command belongs to another run")
        with self._send_lock:
            self._socket.sendall(
                _encode_frame({"type": "publish", "command": command.to_dict()})
            )

    def send_capacity_edge(self, available_tokens: int) -> int:
        """Wake P0 after D capacity becomes available; exact duplicates coalesce."""

        if self.rank != 0 or self.endpoint_role != "decode":
            raise GroupProtocolError(
                "only Decode endpoint rank zero may report capacity"
            )
        available_tokens = int(available_tokens)
        if available_tokens < 0:
            raise ValueError("available capacity must be non-negative")
        with self._send_lock:
            self._capacity_sequence += 1
            edge_seq = self._capacity_sequence
            self._socket.sendall(
                _encode_frame(
                    {
                        "type": "capacity_edge",
                        "session_id": self._capacity_session_id,
                        "edge_seq": edge_seq,
                        "available_tokens": available_tokens,
                    }
                )
            )
        return edge_seq

    def report_readiness(
        self, key: GenerationKey, phase: ReadinessPhase
    ) -> None:
        """Report rank-local state without creating lifecycle ledger state."""

        if key.run_id != self.run_id:
            raise ValueError("readiness belongs to another run")
        phase = ReadinessPhase(phase)
        with self._send_lock:
            self._socket.sendall(
                _encode_frame(
                    {
                        "type": "readiness",
                        "key": key.to_dict(),
                        "phase": phase.value,
                    }
                )
            )

    def report_failure(
        self,
        key: GenerationKey,
        attempt: int,
        lease_id: str,
        detail: str,
    ) -> None:
        """Report a local error without pretending its DMA is drained."""

        if key.run_id != self.run_id or int(attempt) < 1:
            raise ValueError("invalid failure attempt")
        if not lease_id or not detail:
            raise ValueError("failure lease ID and detail are required")
        with self._send_lock:
            self._socket.sendall(
                _encode_frame(
                    {
                        "type": "failure",
                        "key": key.to_dict(),
                        "attempt": int(attempt),
                        "lease_id": str(lease_id),
                        "detail": str(detail),
                    }
                )
            )

    def acknowledge(
        self,
        command: GroupCommand,
        phase: RankPhase,
        *,
        ok: bool = True,
        detail: str = "",
        result: Optional[Mapping[str, Any]] = None,
    ) -> None:
        ack = RankAck(
            key=command.key,
            attempt=command.attempt,
            command_seq=command.command_seq,
            rank=self.rank,
            phase=phase,
            lease_id=command.lease_id,
            ok=ok,
            detail=detail,
            result=result or {},
        )
        self.send_ack(ack)

    def send_ack(self, ack: RankAck) -> None:
        """Send an already constructed local ACK after identity validation."""

        if ack.key.run_id != self.run_id or ack.rank != self.rank:
            raise ValueError("rank ACK identity does not match this session")
        with self._send_lock:
            self._socket.sendall(_encode_frame(ack.to_dict()))

    def close(self) -> None:
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()


def _parse_listen(value: str) -> tuple[str, int]:
    host, port = str(value).removeprefix("tcp://").rsplit(":", 1)
    if not host:
        raise ValueError("listen host must be explicit")
    return host, int(port)


def _parse_groups(value: str) -> dict[str, int]:
    groups = json.loads(value)
    if not isinstance(groups, dict):
        raise ValueError("--groups must be a JSON object mapping group to TP size")
    return {str(name): int(size) for name, size in groups.items()}


def _parse_links(value: str) -> dict[str, Mapping[str, Any]]:
    links = json.loads(value)
    if not isinstance(links, dict):
        raise ValueError("--links must be a JSON object mapping link IDs to specs")
    return {str(name): spec for name, spec in links.items()}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the in-memory agentic TP group relay"
    )
    parser.add_argument("--listen", required=True, help="HOST:PORT")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--token", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--groups", help='legacy single-endpoint mapping, e.g. {"p0":8}'
    )
    selection.add_argument(
        "--links",
        help=('cross-endpoint JSON; each link has coordinator and endpoint specs'),
    )
    args = parser.parse_args(argv)
    server = TCPGroupRelayServer(
        run_id=args.run_id,
        token=args.token,
        groups=_parse_groups(args.groups) if args.groups else None,
        links=_parse_links(args.links) if args.links else None,
        address=_parse_listen(args.listen),
    )
    stop = threading.Event()

    def request_stop(_signum, _frame):
        stop.set()

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_stop)
    host, port = server.address
    print(
        json.dumps(
            {
                "endpoint": f"tcp://{host}:{port}",
                "run_id": args.run_id,
                "groups": dict(server.groups),
                "links": {
                    link_id: {
                        "coordinator": list(spec["coordinator"]),
                        "participants": len(spec["members"]),
                    }
                    for link_id, spec in server.links.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        stop.wait()
    finally:
        server.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


__all__ = [
    "AttemptOutcome",
    "CommandKind",
    "GenerationKey",
    "GenerationRecord",
    "GenerationTerminal",
    "GroupCommand",
    "GroupDisconnectedError",
    "GroupLifecycleCoordinator",
    "LinkLifecycleCoordinator",
    "LinkCapacityEdge",
    "LinkParticipant",
    "LinkRankAck",
    "LinkDisconnected",
    "LinkFailure",
    "LinkIntent",
    "LinkReadiness",
    "GroupNotReadyError",
    "GroupProtocolError",
    "Owner",
    "RankAck",
    "RankDisconnected",
    "RankPhase",
    "ReadinessPhase",
    "StaleAttemptError",
    "TCPGroupRelayServer",
    "TCPRankAgent",
    "TCPRankZeroServer",
]


if __name__ == "__main__":
    raise SystemExit(main())
