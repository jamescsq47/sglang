"""In-memory, run-scoped metadata transactions for the event control broker.

This is an authoritative state core, not a scheduler-side network client. It
does not move KV, grant a DMA fence, or decide when an owner may release memory.
The existing lifecycle code must establish those facts before mutating records.
There are deliberately no files, expiry timers, distributed locks, or polling
threads here. A broker publishes committed changes to connected subscribers.
"""

from __future__ import annotations

import copy
import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional


class ControlRecordError(RuntimeError):
    pass


class ControlCapacityError(ControlRecordError):
    """Fail closed rather than dropping a live ownership/tombstone record."""


class ControlReplayError(ControlRecordError):
    """A request was reordered or retried with a different payload."""


@dataclass(frozen=True)
class RecordKey:
    namespace: str
    snapshot_id: str
    attempt_id: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (
            self.namespace, self.snapshot_id, self.attempt_id
        )):
            raise ValueError("namespace, generation and explicit attempt are required")

    @classmethod
    def ownership(cls, snapshot_id: str):
        """One exclusion fence across ALL attempts of a generation.

        The actual claim/attempt ID belongs in Record.owner/value. Giving each
        attempt a separate ownership key would allow Direct and Slow to both
        acquire the same snapshot. Attempt-specific keys are only for reports,
        descriptors and receipts, never for this generation-wide exclusion.
        """
        return cls("snapshot-owner", snapshot_id, "generation-lifetime")


@dataclass(frozen=True)
class Record:
    revision: int
    owner: str
    value: Any
    deleted: bool = False


@dataclass(frozen=True)
class RecordChange:
    sequence: int
    key: RecordKey
    record: Record


@dataclass(frozen=True)
class MutationResult:
    applied: bool
    record: Optional[Record]


@dataclass(frozen=True)
class Mutation:
    key: RecordKey
    expected_revision: int
    expected_owner: Optional[str]
    owner: str
    value: Any = None
    delete: bool = False


class ControlRecords:
    """Atomic version/owner CAS with push cursors and exact retry receipts.

    Each client lane submits strictly sequential requests, waiting for its ACK
    before submitting the next request on that lane. Independent IO lanes use
    distinct client IDs. Retrying the last operation returns its original
    result, including when its record has since been changed by another owner.
    Older/out-of-order requests fail instead of silently applying again.

    A deleted record keeps its revision (ABA protection). There is no TTL-based
    deletion of ownership or retry receipts. The broker is run-scoped and must
    be stopped only after the GPU/IO processes have safely stopped.
    """

    def __init__(self, run_id: str, *, max_records: int = 100_000,
                 max_clients: int = 4096, event_capacity: int = 8192):
        if not run_id or min(max_records, max_clients, event_capacity) < 1:
            raise ValueError("run and positive capacities are required")
        self.run_id = run_id
        self.max_records = max_records
        self.max_clients = max_clients
        self._records: dict[RecordKey, Record] = {}
        self._receipts: dict[str, tuple[int, str, MutationResult]] = {}
        self._sequence = 0
        self._events: deque[RecordChange] = deque(maxlen=event_capacity)
        self._condition = threading.Condition()
        self._closed = False

    def _require_run(self, run_id):
        if self._closed:
            raise ControlRecordError("control service stopped; ownership is retained")
        if run_id != self.run_id:
            raise ControlRecordError("control message belongs to another run")

    @staticmethod
    def _signature(mutation: Mutation) -> str:
        # JSON both validates metadata (no arbitrary Python/CUDA objects) and
        # canonically identifies a retried operation. Disallow NaN/Infinity.
        return json.dumps([
            mutation.key.namespace, mutation.key.snapshot_id,
            mutation.key.attempt_id, mutation.expected_revision,
            mutation.expected_owner, mutation.owner,
            mutation.value, mutation.delete,
        ], sort_keys=True, separators=(",", ":"), allow_nan=False)

    def mutate(self, run_id: str, client_id: str, sequence: int,
               mutation: Mutation) -> MutationResult:
        if not client_id or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("client identity and positive sequence are required")
        if (not isinstance(mutation.expected_revision, int)
                or mutation.expected_revision < 0):
            raise ValueError("expected revision must be non-negative")
        if not isinstance(mutation.owner, str) or not mutation.owner:
            raise ValueError("an explicit record owner is required")
        if mutation.expected_revision and not mutation.expected_owner:
            raise ValueError("updates and deletion require exact previous owner")
        if mutation.delete and mutation.expected_revision == 0:
            raise ValueError("cannot delete a record without its revision")
        # Copy before taking the lock: a caller must not retain mutable aliases
        # into broker state or mutate a retry signature after its first submit.
        mutation = copy.deepcopy(mutation)
        signature = self._signature(mutation)
        with self._condition:
            self._require_run(run_id)
            previous = self._receipts.get(client_id)
            if previous is not None:
                last_sequence, last_signature, last_result = previous
                if sequence == last_sequence:
                    if signature != last_signature:
                        raise ControlReplayError("same sequence, different operation")
                    return copy.deepcopy(last_result)
                if sequence != last_sequence + 1:
                    raise ControlReplayError("non-sequential client operation")
            else:
                if sequence != 1:
                    raise ControlReplayError("new client must start at sequence one")
                if len(self._receipts) >= self.max_clients:
                    raise ControlCapacityError("client receipt capacity exhausted")

            current = self._records.get(mutation.key)
            revision = 0 if current is None else current.revision
            matches = revision == mutation.expected_revision and (
                current is None or current.owner == mutation.expected_owner
            )
            if matches:
                if current is None and len(self._records) >= self.max_records:
                    raise ControlCapacityError("record capacity exhausted")
                self._sequence += 1
                current = Record(
                    revision=self._sequence, owner=mutation.owner,
                    value=None if mutation.delete else mutation.value,
                    deleted=mutation.delete,
                )
                self._records[mutation.key] = current
                self._events.append(RecordChange(self._sequence, mutation.key, current))
                self._condition.notify_all()
            result = MutationResult(matches, current)
            self._receipts[client_id] = (sequence, signature, result)
            return copy.deepcopy(result)

    def get(self, run_id: str, key: RecordKey) -> Optional[Record]:
        with self._condition:
            self._require_run(run_id)
            return copy.deepcopy(self._records.get(key))

    def snapshot(self, run_id: str) -> tuple[int, dict[RecordKey, Record]]:
        """One atomic startup/reconnection view; not a periodic hot-path scan."""
        with self._condition:
            self._require_run(run_id)
            return self._sequence, copy.deepcopy(self._records)

    def changes(self, run_id: str, after: int, *, timeout: Optional[float] = None):
        """Wait for a commit notification, not a filesystem/polling interval.

        Return ``None`` on cursor overflow, requiring an explicit atomic resync.
        Consumers must not treat an overflow as an empty successful update.
        A timeout is only an observation timeout, never a lease/DMA timeout.
        """
        with self._condition:
            self._require_run(run_id)
            if not isinstance(after, int) or not 0 <= after <= self._sequence:
                raise ValueError("invalid event cursor")
            self._condition.wait_for(
                lambda: self._closed or self._sequence > after, timeout=timeout
            )
            self._require_run(run_id)
            if self._events and after < self._events[0].sequence - 1:
                return None
            return copy.deepcopy(tuple(event for event in self._events
                                       if event.sequence > after))

    def close(self):
        """Wake subscribers; do not infer that GPU memory is safe to release."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
