"""Typed, file-free metadata transactions for remote Host READs.

KV bytes and CUDA/NIXL fences never enter this service. The caller supplies a
completion only after observing its physical fence. Source retirement seals a
generation against new readers atomically, before deregistration starts; the
service never holds a distributed lock across DMA or source cleanup.
"""

from copy import deepcopy
from dataclasses import asdict
import os
import threading

from sglang.srt.disaggregation.agentic_remote_host import (
    HostShard,
    ReadReceipt,
    group_ack,
)


class RemoteHostControlState:
    """Run-owned broker state; records/tombstones are retained until shutdown."""

    METHODS = frozenset(
        {
            "export",
            "descriptor",
            "shards",
            "receipts",
            "claim_attempt",
            "reserve_destination",
            "receipt",
            "cancel_unstarted",
            "prepare_cleanup",
            "complete_cleanup",
        }
    )

    def __init__(self, *, max_generations=100_000, ledger_lookup=None):
        if type(max_generations) is not int or max_generations < 1:
            raise ValueError("positive generation capacity required")
        self._lock = threading.Lock()
        self._entries = {}
        self._max_generations = max_generations
        self._ledger_lookup = ledger_lookup

    def handle(self, method, payload):
        if method not in self.METHODS:
            raise ValueError("unknown remote Host control operation")
        payload = deepcopy(payload)
        direction, sid = payload.pop("direction"), payload.pop("snapshot_id")
        if direction not in {"d2p", "p2d"} or not isinstance(sid, str) or not sid:
            raise ValueError("explicit Host direction and snapshot required")
        with self._lock:
            key = (direction, sid)
            entry = self._entries.get(key)
            if entry is None and method == "export":
                if len(self._entries) >= self._max_generations:
                    raise RuntimeError("remote Host control capacity exhausted")
                # Publish only after validation succeeds; invalid exports must
                # not leave a partial source record or consume capacity.
                entry = {
                    "exports": {},
                    "attempt": None,
                    "attempts": {},
                    "sealed": None,
                    "released": [],
                }
                result = self._export(entry, sid, **payload)
                self._entries[key] = entry
            elif entry is None:
                if method in {"descriptor", "shards", "receipts", "prepare_cleanup"}:
                    return None
                raise ValueError("remote Host source does not exist")
            else:
                if method == "prepare_cleanup" and entry["sealed"] is None:
                    if self._ledger_lookup is None:
                        raise RuntimeError(
                            "remote Host cleanup requires authoritative ledger lookup"
                        )
                    # The RPC broker serializes this read with Host ledger
                    # mutations. A stale source-side FAILED copy must never
                    # manufacture no-READ proofs for a newer recovery epoch.
                    current = self._ledger_lookup(direction, sid)
                    if current is None:
                        return None
                    payload["terminal_entry"] = deepcopy(current)
                result = getattr(self, "_" + method)(entry, sid, **payload)
            return deepcopy(result)

    @staticmethod
    def _rank(entry, rank):
        if type(rank) is not int or not 0 <= rank < entry["size"]:
            raise ValueError("invalid remote Host rank")

    def _export(self, entry, sid, record):
        shard = HostShard.from_dict(record["shard"])
        if (
            shard.snapshot_id != sid
            or type(shard.tp_rank) is not int
            or type(shard.tp_size) is not int
        ):
            raise ValueError("invalid source shard identity")
        if not record.get("node_id") or not record.get("engine_id"):
            raise ValueError("source node/engine required")
        expected = (
            shard.tp_size,
            shard.layout,
            shard.token_count,
            record["node_id"],
            record["engine_id"],
        )
        if entry["exports"]:
            if entry["identity"] != expected:
                raise ValueError("source TP exports disagree")
        previous = entry["exports"].get(str(shard.tp_rank))
        if previous is not None:
            if previous != record:
                raise ValueError("source export changed within generation")
            return previous
        if entry["sealed"] is not None:
            raise ValueError("source generation is retired")
        entry["identity"], entry["size"] = expected, shard.tp_size
        entry["exports"][str(shard.tp_rank)] = record
        return record

    def _descriptor(self, entry, sid, rank):
        self._rank(entry, rank)
        return entry["exports"].get(str(rank))

    @staticmethod
    def _shards(entry, sid):
        exports = entry["exports"]
        if set(exports) != {str(rank) for rank in range(entry["size"])}:
            return None
        return [exports[str(rank)]["shard"] for rank in range(entry["size"])]

    def _receipts(self, entry, sid, attempt_id):
        attempt = entry["attempts"].get(attempt_id)
        if attempt is None or set(attempt["receipts"]) != {
            str(rank) for rank in range(entry["size"])
        }:
            return None
        return [attempt["receipts"][str(rank)] for rank in range(entry["size"])]

    def _claim_attempt(self, entry, sid, attempt_id, engine_id):
        if not isinstance(attempt_id, str) or not attempt_id or not engine_id:
            raise ValueError("read attempt/engine required")
        wanted = {"attempt_id": attempt_id, "engine_id": engine_id}
        current = entry["attempt"]
        if current == wanted:
            return wanted
        if entry["sealed"] is not None:
            raise ValueError("source generation is retired")
        # Attempts cannot be replayed after another epoch took over, even if
        # all old reads had drained. Their retained receipts are tombstones.
        if attempt_id in entry["attempts"]:
            raise ValueError("stale or foreign remote Host attempt")
        if current is not None:
            shards = self._shards(entry, sid)
            receipts = self._receipts(entry, sid, current["attempt_id"])
            if shards is None or receipts is None:
                raise ValueError("previous remote Host attempt is not fully fenced")
            group_ack(
                [HostShard.from_dict(s) for s in shards],
                [ReadReceipt(**r) for r in receipts],
                cancelled=True,
            )
        entry["attempt"] = wanted
        entry["attempts"][attempt_id] = {
            "engine_id": engine_id,
            "destinations": {},
            "receipts": {},
        }
        return wanted

    def _current(self, entry, attempt_id, engine_id):
        if entry["sealed"] is not None:
            raise ValueError("source generation is retired")
        if entry["attempt"] != {"attempt_id": attempt_id, "engine_id": engine_id}:
            raise ValueError("remote Host read does not own current attempt")
        return entry["attempts"][attempt_id]

    def _reserve_destination(self, entry, sid, rank, attempt_id, engine_id, signature):
        self._rank(entry, rank)
        digest = signature.get("spans_digest", "")
        if (
            not isinstance(signature.get("incarnation"), str)
            or not signature["incarnation"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(signature.get("span_count")) is not int
            or signature["span_count"] < 1
            or "spans" in signature
        ):
            raise ValueError("invalid bounded destination identity")
        attempt = entry["attempts"].get(attempt_id)
        if attempt is None or attempt["engine_id"] != engine_id:
            raise ValueError("destination belongs to unknown/foreign attempt")
        source = entry["exports"].get(str(rank))
        if (
            source is None
            or signature.get("shard") != source["shard"]
            or signature.get("engine") != engine_id
        ):
            raise ValueError("destination does not match source/receiver")
        previous = attempt["destinations"].get(str(rank))
        if previous is not None and previous != signature:
            raise ValueError(
                "remote Host attempt changed destination/process incarnation"
            )
        receipt = attempt["receipts"].get(str(rank))
        if receipt is not None:
            return receipt
        self._current(entry, attempt_id, engine_id)
        attempt["destinations"][str(rank)] = signature
        return None

    def _receipt(self, entry, sid, engine_id, receipt):
        value = ReadReceipt(**receipt)
        self._rank(entry, value.tp_rank)
        source = entry["exports"].get(str(value.tp_rank))
        if source is None:
            raise ValueError("receipt has no source shard")
        shard = HostShard.from_dict(source["shard"])
        if (
            value.snapshot_id,
            value.export_id,
            value.tp_size,
            value.layout,
            value.token_count,
        ) != (
            sid,
            shard.export_id,
            shard.tp_size,
            shard.layout,
            shard.token_count,
        ) or value.outcome not in {
            "loaded",
            "drained",
        }:
            raise ValueError("receipt identity/fence mismatch")
        attempt = entry["attempts"].get(value.read_id)
        if attempt is None or attempt["engine_id"] != engine_id:
            raise ValueError("receipt belongs to unknown/foreign attempt")
        previous = attempt["receipts"].get(str(value.tp_rank))
        if previous is not None:
            if previous != receipt:
                raise ValueError("physical remote Host receipt changed")
            return previous
        self._current(entry, value.read_id, engine_id)
        if (
            value.outcome == "loaded"
            and str(value.tp_rank) not in attempt["destinations"]
        ):
            raise ValueError("loaded receipt without destination")
        attempt["receipts"][str(value.tp_rank)] = receipt
        return receipt

    def _cancel_unstarted(self, entry, sid, rank, attempt_id, engine_id):
        self._rank(entry, rank)
        previous_attempt = entry["attempts"].get(attempt_id)
        if previous_attempt is not None and previous_attempt["engine_id"] == engine_id:
            previous = previous_attempt["receipts"].get(str(rank))
            if previous is not None:
                return previous
        if str(rank) not in entry["exports"]:
            raise ValueError("cancelled Host snapshot has no source descriptor")
        self._claim_attempt(entry, sid, attempt_id, engine_id)
        attempt = self._current(entry, attempt_id, engine_id)
        previous = attempt["receipts"].get(str(rank))
        if previous is not None:
            return previous
        if str(rank) in attempt["destinations"]:
            raise ValueError("remote Host worker already acquired its destination")
        shard = HostShard.from_dict(entry["exports"][str(rank)]["shard"])
        receipt = asdict(
            ReadReceipt(
                sid,
                shard.export_id,
                rank,
                entry["size"],
                shard.layout,
                shard.token_count,
                attempt_id,
                "drained",
            )
        )
        attempt["receipts"][str(rank)] = receipt
        return receipt

    def _prepare_cleanup(self, entry, sid, rank, export_id, terminal_entry):
        self._rank(entry, rank)
        source = entry["exports"].get(str(rank))
        if source is None or source["shard"]["export_id"] != export_id:
            raise ValueError("cleanup belongs to another export")
        if entry["sealed"] is not None:
            return entry["sealed"]
        state = terminal_entry.get("state")
        success = state == "consumed"
        if state not in {
            "consumed",
            "failed",
            "rejected",
            "evicting",
            "recompute_required",
        }:
            return None
        current = entry["attempt"]
        shards, receipts = None, None
        if current is not None:
            shards = self._shards(entry, sid)
            if shards is None:
                return None
            attempt = entry["attempts"][current["attempt_id"]]
            # Preserve the original no-READ proof: the lifecycle has declared
            # FAILED and the missing physical rank drained without destination.
            records = dict(attempt["receipts"])
            drained = set(terminal_entry.get("loader_drained_ranks", []))
            for source_shard in shards:
                shard = HostShard.from_dict(source_shard)
                key = str(shard.tp_rank)
                if (
                    state == "failed"
                    and shard.tp_rank in drained
                    and key not in records
                    and key not in attempt["destinations"]
                ):
                    records[key] = asdict(
                        ReadReceipt(
                            sid,
                            shard.export_id,
                            shard.tp_rank,
                            entry["size"],
                            shard.layout,
                            shard.token_count,
                            current["attempt_id"],
                            "drained",
                        )
                    )
            if set(records) != {str(r) for r in range(entry["size"])}:
                return None
            receipts = [records[str(r)] for r in range(entry["size"])]
            group_ack(
                [HostShard.from_dict(s) for s in shards],
                [ReadReceipt(**r) for r in receipts],
                cancelled=not success,
            )
        elif success and (
            terminal_entry.get("loader_acks")
            or terminal_entry.get("loading_ranks")
            or terminal_entry.get("recovery_claims")
        ):
            return None
        # Seal before returning. New attempt/destination claims now fail even
        # while source deregistration runs asynchronously on another process.
        entry["sealed"] = {
            "attempt": current,
            "shards": shards,
            "receipts": receipts,
            "success": success,
        }
        return entry["sealed"]

    def _complete_cleanup(self, entry, sid, rank, export_id):
        self._rank(entry, rank)
        source = entry["exports"].get(str(rank))
        if (
            entry["sealed"] is None
            or source is None
            or source["shard"]["export_id"] != export_id
        ):
            raise ValueError("cleanup has no sealed matching export")
        if rank not in entry["released"]:
            entry["released"].append(rank)
        return True


class RemoteHostControlClient:
    """Typed adapter; synchronous RPC is restricted to existing Host workers."""

    def __init__(self, client, direction):
        self.client, self.direction = client, direction

    def call(self, method, snapshot_id, **payload):
        return self.client.call(
            "remote_host",
            "call",
            method,
            {
                "direction": self.direction,
                "snapshot_id": snapshot_id,
                **payload,
            },
        )


def create_remote_host_control_client(direction):
    if not os.environ.get("SGLANG_AGENTIC_CONTROL_ENDPOINT"):
        return None
    from sglang.srt.disaggregation.agentic_control_rpc import get_control_client

    return RemoteHostControlClient(get_control_client(), direction)
