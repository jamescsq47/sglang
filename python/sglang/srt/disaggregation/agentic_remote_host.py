"""Opt-in network data plane for source-local Host snapshots.

This module does not replace the existing single-node mmap path. A caller first
finishes source-local D2H, pins the Arena extent in the authoritative lifecycle,
then exports it. The receiver must already own its complete workset and registered
VRAM. Only metadata travels over the control plane; NIXL READ moves DRAM directly
to remote VRAM. Registration, start and poll belong to background I/O workers.

TP rank0 collects all shard receipts before releasing ANY source shard. A timeout
is never a fence: failed/ambiguous posts retain their handle and destination lease
until NIXL cancellation succeeds. Source storage remains pinned until a complete
group ACK. Engine wiring and GPU/RDMA validation are separate integration gates.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class HostShard:
    snapshot_id: str
    export_id: str
    tp_rank: int
    tp_size: int
    layout: str
    token_count: int
    address: int
    byte_size: int
    metadata_b64: str
    peer_id: str = ""

    def validate(self) -> None:
        if not self.snapshot_id or not self.export_id or not self.layout:
            raise ValueError("snapshot, export epoch and layout are required")
        if self.token_count <= 0:
            raise ValueError("invalid snapshot token count")
        if len(self.layout) != 64 or any(c not in "0123456789abcdef" for c in self.layout):
            raise ValueError("layout must be a canonical SHA256 fingerprint")
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid TP shard")
        if self.address <= 0 or self.byte_size <= 0:
            raise ValueError("empty/invalid Host extent")
        base64.b64decode(self.metadata_b64, validate=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> HostShard:
        shard = cls(**value)
        shard.validate()
        return shard


@dataclass(frozen=True)
class ReadReceipt:
    snapshot_id: str
    export_id: str
    tp_rank: int
    tp_size: int
    layout: str
    token_count: int
    read_id: str
    outcome: str = "loaded"


def layout_fingerprint(layout: dict) -> str:
    """Hash agreed model/KV schema (dtype, layers, heads, page size, state layout).

    Callers must derive the descriptor from the actual pools, not the model name
    alone. Token count is separate; physical addresses/rank IDs do not belong here.
    """
    if not layout:
        raise ValueError("empty KV layout")
    return hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def group_ack(shards: Sequence[HostShard], receipts: Sequence[ReadReceipt], *,
              cancelled: bool = False) -> tuple:
    """Validate a complete TP handoff; partial, duplicate or stale ACKs fail."""
    if not shards:
        raise ValueError("empty TP group")
    first = shards[0]
    expected = (first.snapshot_id, first.tp_size, first.layout, first.token_count)
    by_rank = {}
    for shard in shards:
        shard.validate()
        if (shard.snapshot_id, shard.tp_size, shard.layout, shard.token_count) != expected:
            raise ValueError("mixed snapshot/layout/TP group")
        if shard.tp_rank in by_rank:
            raise ValueError("duplicate TP shard")
        by_rank[shard.tp_rank] = shard
    if set(by_rank) != set(range(first.tp_size)):
        raise ValueError("incomplete TP shard group")
    acked = {}
    attempt_id = None
    for receipt in receipts:
        if (receipt.snapshot_id, receipt.tp_size, receipt.layout, receipt.token_count) != expected:
            raise ValueError("receipt belongs to another generation/layout")
        allowed = {"loaded", "drained"} if cancelled else {"loaded"}
        if receipt.outcome not in allowed:
            raise ValueError("cancelled/unfinished read cannot commit a handoff")
        shard = by_rank.get(receipt.tp_rank)
        if shard is None or receipt.export_id != shard.export_id or not receipt.read_id:
            raise ValueError("stale or invalid receipt")
        if attempt_id is not None and receipt.read_id != attempt_id:
            raise ValueError("mixed TP recovery attempts")
        attempt_id = receipt.read_id
        if receipt.tp_rank in acked:
            raise ValueError("duplicate TP receipt")
        acked[receipt.tp_rank] = receipt
    if set(acked) != set(by_rank):
        raise ValueError("all TP reads must finish before Host release")
    return tuple(acked[rank] for rank in range(first.tp_size))


def mha_read_spans(
    *, layer_bases: Sequence[int], token_indices: Sequence[int], bytes_per_token: int
) -> list[tuple[int, int, int]]:
    """Map [K/V,layer,token,...] Host data into paged device KV, coalescing runs.

    layer_bases must be [all K layers, all V layers], matching SharedMHAHostSnapshot.
    Returns (Host-relative offset, device address, bytes). Hybrid/Mamba callers
    must supply their complete layout separately, never pass only attention KV.
    """
    indices = tuple(int(index) for index in token_indices)
    if not layer_bases or len(layer_bases) % 2 or not indices or bytes_per_token <= 0:
        raise ValueError("invalid MHA layout")
    if min(indices) < 0 or len(set(indices)) != len(indices):
        raise ValueError("device token slots must be nonnegative and unique")
    spans = []
    for layer, base in enumerate(layer_bases):
        if int(base) <= 0:
            raise ValueError("invalid device layer base")
        start = 0
        while start < len(indices):
            end = start + 1
            while end < len(indices) and indices[end] == indices[end - 1] + 1:
                end += 1
            spans.append(((layer * len(indices) + start) * bytes_per_token,
                          int(base) + indices[start] * bytes_per_token,
                          (end - start) * bytes_per_token))
            start = end
    return spans


class RemoteHostExport:
    """One source rank's pinned, durable extent. No automatic expiry/free."""

    def __init__(self, transport, shard, registration, keepalive):
        self.transport = transport
        self.shard = shard
        self.registration = registration
        # Holding the Python object does NOT replace a ledger eviction pin.
        self.keepalive = keepalive
        self.reader_id = None
        self.closed = False

    def claim(self, read_id: str) -> HostShard:
        with self.transport.lock:
            if self.closed or not read_id:
                raise RuntimeError("export is closed or read lease is empty")
            if self.reader_id not in (None, read_id):
                raise RuntimeError("Host export already has a reader")
            self.reader_id = read_id
            return self.shard

    def release_after_group_ack(self, shards, receipts, *, committed_read_id: str) -> bool:
        """Release only with a group ownership commit, not a bare DMA receipt.

        committed_read_id must come from the authoritative lifecycle after all
        destination ranks accept/bind the workset. Caller releases Arena extent
        only after this returns successfully. Control messages must be trusted.
        """
        ack = group_ack(shards, receipts)
        if not committed_read_id or any(item.read_id != committed_read_id for item in ack):
            raise ValueError("missing/mismatched destination ownership commit")
        with self.transport.lock:
            own = ack[self.shard.tp_rank]
            if (next(shard for shard in shards if shard.tp_rank == self.shard.tp_rank)
                    != self.shard or own.read_id != self.reader_id):
                raise ValueError("ACK does not match this export/read lease")
            if self.closed:
                return False
            self.transport.agent.deregister_memory(
                self.registration, backends=self.transport.backends
            )
            self.closed = True
            self.keepalive = None
            self.transport.exports.pop(self.shard.export_id, None)
            return True

    def unclaim_after_group_cancel(self, shards, receipts) -> bool:
        """All rank reads fenced: permit a new recovery attempt, retain Host."""
        ack = group_ack(shards, receipts, cancelled=True)
        with self.transport.lock:
            own = ack[self.shard.tp_rank]
            if (self.closed or next(shard for shard in shards
                if shard.tp_rank == self.shard.tp_rank) != self.shard):
                raise ValueError("cancellation targets a stale export")
            if self.reader_id is None:
                return False
            if own.read_id != self.reader_id:
                raise ValueError("cancellation targets another read lease")
            self.reader_id = None
            return True


class RemoteHostRead:
    def __init__(self, transport, shard, read_id, handle):
        self.transport = transport
        self.shard = shard
        self.read_id = read_id
        self.handle = handle
        self.post_error = None
        self.state = "PREPARED"
        self.receipt = None
        self.drain_receipt = None

    def start(self) -> None:
        with self.transport.lock:
            if self.state != "PREPARED":
                raise RuntimeError("read has already been posted")
            # Publish ambiguous ownership BEFORE post: post may submit then raise.
            self.state = "PROC"
            try:
                self.state = self.transport.agent.transfer(self.handle)
            except Exception as exc:
                self.post_error = exc
                self.state = "UNKNOWN"

    def poll(self) -> ReadReceipt | None:
        """Non-blocking progress. ERR is not permission to free either side."""
        with self.transport.lock:
            if self.receipt is not None:
                return self.receipt
            if self.state == "DRAINED":
                raise RuntimeError("cancelled read cannot produce a receipt")
            if self.state == "PREPARED":
                return None
            self.state = self.transport.agent.check_xfer_state(self.handle)
            if self.state != "DONE":
                return None
            self.transport.agent.release_xfer_handle(self.handle)
            self.handle = None
            shard = self.shard
            self.receipt = ReadReceipt(shard.snapshot_id, shard.export_id,
                shard.tp_rank, shard.tp_size, shard.layout, shard.token_count, self.read_id)
            return self.receipt

    def drain_failure(self) -> ReadReceipt:
        """Cancellation is a fence ONLY when NIXL release succeeds.

        Keeps source export pinned. Caller must coordinate TP cancellation and
        source unclaim separately; no implicit retry or extent release occurs.
        """
        with self.transport.lock:
            if self.receipt is not None:
                return self.receipt
            if self.drain_receipt is not None:
                return self.drain_receipt
            self.transport.agent.release_xfer_handle(self.handle)
            self.handle = None
            self.state = "DRAINED"
            shard = self.shard
            self.drain_receipt = ReadReceipt(shard.snapshot_id, shard.export_id,
                shard.tp_rank, shard.tp_size, shard.layout, shard.token_count,
                self.read_id, "drained")
            return self.drain_receipt


class RemoteHostTransport:
    """Thin adapter around an existing NIXL agent; does not create CUDA contexts.

    The agent must have UCX enabled and final VRAM registered. All users of a
    shared agent must use the same lifecycle lock. For isolation, use an agent
    dedicated to remote Host I/O. No HTTP/NFS data fallback is provided.
    """

    def __init__(self, agent: Any, *, lock=None, backends=("UCX",), max_reads=128,
                 max_peer_imports=256):
        if int(max_reads) < 1 or int(max_peer_imports) < 1:
            raise ValueError("read and peer import limits must be positive")
        self.agent = agent
        self.lock = lock if lock is not None else threading.RLock()
        self._peer_condition = threading.Condition(self.lock)
        self.backends = list(backends)
        # Registry owns live mappings even if an integration drops its wrapper.
        self.exports = {}
        self.quarantined = {}
        self.max_reads = int(max_reads)
        self.max_peer_imports = int(max_peer_imports)
        # Retain handles across caller exceptions/GC. Completed entries also
        # suppress duplicate posts until the authoritative attempt is retired.
        self.reads = {}
        # Loading partial metadata also establishes the UCX connection. Keep
        # that connection across reads: disconnect/reconnect for every idle
        # gap eventually exhausts mlx5 DevX QP creation resources.  Metadata
        # for old, deregistered extents is reclaimed by a bounded rollover,
        # but only after every READ using this peer is physically fenced.
        self._remote_peers = {}
        self._imported_exports = {}
        self._imported_ranges = {}

    def _active_peers(self):
        return {getattr(record[1], "peer", None) for record in self.reads.values()}

    def _drop_peer(self, peer_id):
        record = self._remote_peers.pop(peer_id)
        self.agent.remove_remote_agent(record["peer"])
        for export_id, imported_peer_id in tuple(self._imported_exports.items()):
            if imported_peer_id == peer_id:
                self._imported_exports.pop(export_id)
                self._imported_ranges.pop(export_id, None)

    def _rollover_saturated_peers(self, *, force=False):
        active = self._active_peers()
        for peer_id, record in tuple(self._remote_peers.items()):
            if record["peer"] in active or (not force and record["imports"] < self.max_peer_imports):
                continue
            self._drop_peer(peer_id)

    def _peer_range_conflicts(self, peer_id, shard):
        start, end = int(shard.address), int(shard.address) + int(shard.byte_size)
        return any(
            imported_peer_id == peer_id and left < end and start < right
            for export_id, imported_peer_id in self._imported_exports.items()
            for left, right in (self._imported_ranges[export_id],)
        )

    def close_remote_peers(self) -> None:
        """Disconnect idle peers explicitly at engine teardown.

        Process exit remains the final fallback.  A live READ is never treated
        as fenced merely because shutdown was requested.
        """
        with self.lock:
            if self.reads:
                raise RuntimeError("cannot close remote peers with live READ handles")
            self._rollover_saturated_peers(force=True)

    def retire_read(self, export_id: str, read_id: str) -> bool:
        """Forget a physically fenced attempt after control-plane retirement.

        Caller must first make replay of this exact attempt impossible (group
        commit/cancel acknowledged). This method does not release a workset.
        """
        key = (export_id, read_id)
        with self.lock:
            record = self.reads.get(key)
            if record is None:
                self._rollover_saturated_peers()
                return False
            pending = record[1]
            if pending.handle is not None or (
                pending.receipt is None and pending.drain_receipt is None
            ):
                raise RuntimeError("cannot retire an unfenced READ handle")
            self.reads.pop(key)
            self._rollover_saturated_peers()
            self._peer_condition.notify_all()
            return True

    def export(self, *, snapshot_id, tp_rank, tp_size, layout, token_count, address,
               byte_size, keepalive) -> RemoteHostExport:
        """Call ONLY after local D2H fence + authoritative eviction pin."""
        shard = HostShard(str(snapshot_id), uuid.uuid4().hex, int(tp_rank),
                          int(tp_size), str(layout), int(token_count), int(address), int(byte_size), "",
                          str(getattr(self.agent, "name", "")))
        shard.validate()
        if keepalive is None:
            raise ValueError("a live source mapping reference is required")
        with self.lock:
            registration = None
            try:
                registration = self.agent.register_memory(
                    [(shard.address, shard.byte_size, 0, "")], "DRAM", backends=self.backends
                )
                metadata = self.agent.get_partial_agent_metadata(
                    registration, inc_conn_info=True, backends=self.backends
                )
            except Exception as error:
                # Registration itself may partially succeed before raising. A
                # failed cleanup must never drop the last mapping reference.
                record = {"registration": registration, "keepalive": keepalive,
                          "error": repr(error)}
                self.quarantined[shard.export_id] = record
                if registration is not None:
                    try:
                        self.agent.deregister_memory(registration, backends=self.backends)
                    except Exception as cleanup_error:
                        record["cleanup_error"] = repr(cleanup_error)
                    else:
                        self.quarantined.pop(shard.export_id)
                raise
            shard = HostShard(**dict(shard.to_dict(), metadata_b64=
                                    base64.b64encode(metadata).decode("ascii")))
            exported = RemoteHostExport(self, shard, registration, keepalive)
            self.exports[shard.export_id] = exported
            return exported

    def retry_unpublished_cleanup(self) -> int:
        """Retry only known registrations never published to remote readers.

        Unknown/partial registrations stay quarantined until process teardown.
        This does not free the authoritative Arena lease; integration owns that.
        """
        cleaned = 0
        with self.lock:
            for export_id, record in tuple(self.quarantined.items()):
                if record["registration"] is None:
                    continue
                self.agent.deregister_memory(record["registration"], backends=self.backends)
                self.quarantined.pop(export_id)
                cleaned += 1
        return cleaned

    def prepare_read(self, shard: HostShard, *, read_id: str, tp_rank: int,
                     tp_size: int, layout: str, gpu_id: int,
                     spans: Sequence[tuple[int, int, int]], payload_ranges=None) -> RemoteHostRead:
        """Prepare after source claim and full destination workset allocation.

        Spans must cover the entire exported shard exactly once in source order.
        The existing destination VRAM pool registration remains caller-owned.
        """
        shard.validate()
        if (shard.tp_rank, shard.tp_size, shard.layout) != (tp_rank, tp_size, layout):
            raise ValueError("TP shard/layout mismatch")
        if not read_id or gpu_id < 0:
            raise ValueError("missing read lease or invalid GPU")
        # Hybrid extents contain declared alignment padding between Attention
        # and state. All payload bytes (not padding) must be covered once.
        # The engine derives ranges from the fingerprinted local pool layout.
        ranges = tuple(payload_ranges) if payload_ranges is not None else ((0, shard.byte_size),)
        previous_end = 0
        for start, size in ranges:
            if start < previous_end or size <= 0 or start + size > shard.byte_size:
                raise ValueError("invalid payload ranges")
            previous_end = start + size
        if not ranges or ranges[0][0] != 0 or previous_end != shard.byte_size:
            raise ValueError("payload ranges must cover both ends of the shard")
        range_index, offset = 0, 0
        local, remote, destinations = [], [], []
        for source_offset, target_address, length in spans:
            if range_index < len(ranges) and offset == sum(ranges[range_index]):
                range_index += 1
                if range_index < len(ranges):
                    offset = ranges[range_index][0]
            if source_offset != offset or target_address <= 0 or length <= 0:
                raise ValueError("incomplete or invalid shard span")
            if range_index >= len(ranges) or offset + length > sum(ranges[range_index]):
                raise ValueError("span crosses payload/padding boundary")
            offset += length
            local.append((int(target_address), int(length), int(gpu_id)))
            remote.append((shard.address + int(source_offset), int(length), 0))
            destinations.append((target_address, target_address + length))
        if offset != shard.byte_size:
            raise ValueError("must transfer the complete shard")
        ordered = sorted(destinations)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("overlapping destination spans")
        with self.lock:
            key = (shard.export_id, str(read_id))
            signature = (shard, int(gpu_id), tuple(local), tuple(remote))
            existing = self.reads.get(key)
            if existing is not None:
                if existing[0] != signature:
                    raise ValueError("duplicate read attempt changed destination/shard")
                return existing[1]
            if len(self.reads) >= self.max_reads:
                raise RuntimeError("remote Host READ registry capacity reached")
            peer_id = shard.peer_id or hashlib.sha256(
                base64.b64decode(shard.metadata_b64)
            ).hexdigest()
            peer_record = self._remote_peers.get(peer_id)
            imported_peer_id = self._imported_exports.get(shard.export_id)
            if imported_peer_id is not None and imported_peer_id != peer_id:
                raise RuntimeError("remote Host export changed its source peer")
            while (imported_peer_id is None and peer_record is not None and (
                peer_record["imports"] >= self.max_peer_imports
                or self._peer_range_conflicts(peer_id, shard)
            )):
                # Drain one bounded peer generation without failing a recovery
                # attempt. NIXL cannot import a new registration that overlaps
                # stale metadata for a recycled Host-arena address, so address
                # reuse also starts a fresh generation. This runs only on Host
                # I/O workers; retiring reads signal the condition and never
                # need this worker's scheduler.
                if peer_record["peer"] in self._active_peers():
                    self._peer_condition.wait()
                    peer_record = self._remote_peers.get(peer_id)
                    continue
                self._drop_peer(peer_id)
                peer_record = None
                imported_peer_id = None
            if imported_peer_id is None:
                peer = self.agent.add_remote_agent(base64.b64decode(shard.metadata_b64))
                if peer_record is None:
                    peer_record = self._remote_peers[peer_id] = {
                        "peer": peer, "imports": 0,
                    }
                elif peer_record["peer"] != peer:
                    raise RuntimeError("remote Host peer identity changed")
                peer_record["imports"] += 1
                self._imported_exports[shard.export_id] = peer_id
                self._imported_ranges[shard.export_id] = (
                    int(shard.address), int(shard.address) + int(shard.byte_size)
                )
            else:
                if peer_record is None:
                    raise RuntimeError("remote Host peer cache lost an imported export")
                peer = peer_record["peer"]
            try:
                local_desc = self.agent.get_xfer_descs(local, "VRAM")
                remote_desc = self.agent.get_xfer_descs(remote, "DRAM")
                handle = self.agent.initialize_xfer(
                    "READ", local_desc, remote_desc, peer, backends=self.backends
                )
                if handle is None:
                    raise RuntimeError("NIXL returned no READ handle")
            except Exception:
                # No transfer is posted by initialize. Do not retain imported
                # source registrations past the bounded peer generation. The
                # same export retry reuses its already imported metadata.
                self._rollover_saturated_peers()
                raise
            pending = RemoteHostRead(self, shard, str(read_id), handle)
            pending.peer = peer
            self.reads[key] = (signature, pending)
            return pending
