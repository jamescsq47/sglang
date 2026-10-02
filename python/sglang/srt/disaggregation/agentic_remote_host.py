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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ReadReceipt":
        receipt = cls(**value)
        if (
            not receipt.snapshot_id
            or not receipt.export_id
            or not receipt.read_id
            or receipt.tp_size < 1
            or not 0 <= receipt.tp_rank < receipt.tp_size
            or receipt.token_count <= 0
            or receipt.outcome not in {"loaded", "drained"}
        ):
            raise ValueError("invalid remote Host read receipt")
        return receipt


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
    runs = []
    start = 0
    while start < len(indices):
        end = start + 1
        while end < len(indices) and indices[end] == indices[end - 1] + 1:
            end += 1
        runs.append((start, end, indices[start]))
        start = end

    spans = []
    layer_stride = len(indices) * bytes_per_token
    for layer, base in enumerate(layer_bases):
        base = int(base)
        if base <= 0:
            raise ValueError("invalid device layer base")
        host_layer_offset = layer * layer_stride
        for start, end, first_index in runs:
            spans.append(
                (
                    host_layer_offset + start * bytes_per_token,
                    base + first_index * bytes_per_token,
                    (end - start) * bytes_per_token,
                )
            )
    return spans


def mha_page_read_spans(
    *,
    layer_bases: Sequence[int],
    page_indices: Sequence[int],
    page_size: int,
    token_count: int,
    bytes_per_token: int,
) -> list[tuple[int, int, int]]:
    """Build MHA READ descriptors from an immutable CPU page plan."""

    pages = tuple(int(index) for index in page_indices)
    page_size, token_count = int(page_size), int(token_count)
    if (
        not layer_bases
        or len(layer_bases) % 2
        or page_size <= 0
        or token_count <= 0
        or token_count % page_size
        or bytes_per_token <= 0
        or len(pages) != token_count // page_size
    ):
        raise ValueError("invalid paged MHA layout")
    if min(pages) < 0 or len(set(pages)) != len(pages):
        raise ValueError("device pages must be nonnegative and unique")

    runs = []
    start = 0
    while start < len(pages):
        end = start + 1
        while end < len(pages) and pages[end] == pages[end - 1] + 1:
            end += 1
        runs.append((start, end, pages[start]))
        start = end

    page_bytes = page_size * bytes_per_token
    layer_stride = token_count * bytes_per_token
    spans = []
    for layer, base in enumerate(layer_bases):
        base = int(base)
        if base <= 0:
            raise ValueError("invalid device layer base")
        host_layer_offset = layer * layer_stride
        for start, end, first_page in runs:
            spans.append(
                (
                    host_layer_offset + start * page_bytes,
                    base + first_page * page_bytes,
                    (end - start) * page_bytes,
                )
            )
    return spans


class RemoteHostExport:
    """One source rank's pinned, durable extent. No automatic expiry/free."""

    def __init__(self, transport, shard, registration, keepalive, *, owns_registration=True):
        self.transport = transport
        self.shard = shard
        self.registration = registration
        self.owns_registration = bool(owns_registration)
        # Holding the Python object does NOT replace a ledger eviction pin.
        self.keepalive = keepalive
        self.reader_id = None
        self.eviction_id = None
        self.closed = False

    def claim(self, read_id: str) -> HostShard:
        with self.transport.lock:
            if self.closed or self.eviction_id is not None or not read_id:
                raise RuntimeError("export is closed or read lease is empty")
            if self.reader_id not in (None, read_id):
                raise RuntimeError("Host export already has a reader")
            self.reader_id = read_id
            return self.shard

    def reserve_eviction(self, eviction_id: str) -> bool:
        """Fence a still-unclaimed export against future recovery readers."""

        with self.transport.lock:
            if self.closed or self.reader_id is not None or not eviction_id:
                return False
            if self.eviction_id not in (None, eviction_id):
                return False
            self.eviction_id = eviction_id
            return True

    def cancel_eviction(self, eviction_id: str) -> bool:
        with self.transport.lock:
            if self.closed:
                return False
            if self.eviction_id != eviction_id:
                return False
            self.eviction_id = None
            return True

    def finish_eviction(self, eviction_id: str) -> bool:
        """Deregister after the all-rank eviction prepare barrier."""

        with self.transport.lock:
            if self.closed or self.reader_id is not None:
                return False
            if self.eviction_id != eviction_id:
                return False
            if self.owns_registration:
                self.transport.agent.deregister_memory(
                    self.registration, backends=self.transport.backends
                )
            self.closed = True
            self.keepalive = None
            self.transport.exports.pop(self.shard.export_id, None)
            return True

    def discard_unclaimed(self) -> bool:
        """Deregister a durable export only if no remote reader can touch it."""

        with self.transport.lock:
            if self.closed:
                return True
            if self.reader_id is not None or self.eviction_id is not None:
                return False
            if self.owns_registration:
                self.transport.agent.deregister_memory(
                    self.registration, backends=self.transport.backends
                )
            self.closed = True
            self.keepalive = None
            self.transport.exports.pop(self.shard.export_id, None)
            return True

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
            if self.owns_registration:
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

    def __init__(self, agent: Any, *, lock=None, backends=("UCX",), max_reads=128):
        if int(max_reads) < 1:
            raise ValueError("max_reads must be positive")
        self.agent = agent
        self.lock = lock if lock is not None else threading.RLock()
        self.backends = list(backends)
        # Registry owns live mappings even if an integration drops its wrapper.
        self.exports = {}
        self.quarantined = {}
        self.max_reads = int(max_reads)
        # Retain handles across caller exceptions/GC. Completed entries also
        # suppress duplicate posts until the authoritative attempt is retired.
        self.reads = {}
        self._remote_peers = set()
        self._shared_registrations = []

    def register_shared_arena(self, address: int, byte_size: int, keepalive: Any):
        """Publish one stable registration for a recyclable Host arena.

        A per-snapshot registration cannot safely reuse the same virtual
        address: NIXL retains remote metadata by address and rejects a new
        rkey at that address. The arena registration lives with this transport.
        """
        if address <= 0 or byte_size <= 0 or keepalive is None:
            raise ValueError("invalid shared Host arena registration")
        with self.lock:
            registration = self.agent.register_memory(
                [(int(address), int(byte_size), 0, "")], "DRAM", backends=self.backends
            )
            try:
                metadata = self.agent.get_partial_agent_metadata(
                    registration, inc_conn_info=True, backends=self.backends
                )
            except Exception:
                self.agent.deregister_memory(registration, backends=self.backends)
                raise
            self._shared_registrations.append((registration, keepalive))
            return registration, metadata

    def retire_read(self, export_id: str, read_id: str) -> bool:
        """Forget a physically fenced attempt after control-plane retirement.

        Caller must first make replay of this exact attempt impossible (group
        commit/cancel acknowledged). This method does not release a workset.
        """
        key = (export_id, read_id)
        with self.lock:
            record = self.reads.get(key)
            if record is None:
                return False
            pending = record[1]
            if pending.handle is not None or (
                pending.receipt is None and pending.drain_receipt is None
            ):
                raise RuntimeError("cannot retire an unfenced READ handle")
            self.reads.pop(key)
            # A TP rank reuses the same source agent for many Host snapshots.
            # NIXL's remove_remote_agent disconnects that agent, not merely
            # this snapshot's registration; repeatedly disconnecting during
            # concurrent READs can invalidate another transfer. Keep the
            # peer connection until this dedicated I/O agent exits.
            return True

    def export(self, *, snapshot_id, tp_rank, tp_size, layout, token_count, address,
               byte_size, keepalive, shared_registration=None,
               shared_metadata=None) -> RemoteHostExport:
        """Call ONLY after local D2H fence + authoritative eviction pin."""
        shard = HostShard(
            str(snapshot_id),
            uuid.uuid4().hex,
            int(tp_rank),
            int(tp_size),
            str(layout),
            int(token_count),
            int(address),
            int(byte_size),
            "",
            str(getattr(self.agent, "name", "")),
        )
        shard.validate()
        if keepalive is None:
            raise ValueError("a live source mapping reference is required")
        with self.lock:
            if (shared_registration is None) != (shared_metadata is None):
                raise ValueError("shared registration and metadata must be paired")
            if shared_registration is not None:
                registration, metadata = shared_registration, shared_metadata
            else:
                registration = None
                try:
                    registration = self.agent.register_memory(
                        [(shard.address, shard.byte_size, 0, "")], "DRAM", backends=self.backends
                    )
                    metadata = self.agent.get_partial_agent_metadata(
                        registration, inc_conn_info=True, backends=self.backends
                    )
                except Exception as error:
                    # Registration itself may partially succeed before raising.
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
            exported = RemoteHostExport(
                self, shard, registration, keepalive,
                owns_registration=shared_registration is None,
            )
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

    def discard_unpublished_keepalive(self, keepalive: Any) -> bool:
        """Fence failed exports before their source mapping may be reclaimed.

        ``export`` can fail after DRAM registration succeeded.  Such a mapping
        is retained in ``quarantined`` so its backing extent must remain live
        until deregistration has really completed.  This method deliberately
        considers only records backed by the supplied mapping; an unrelated
        quarantined export cannot block cleanup of this snapshot.
        """

        if keepalive is None:
            raise ValueError("a live source mapping reference is required")
        with self.lock:
            matches = [
                (export_id, record)
                for export_id, record in self.quarantined.items()
                if record.get("keepalive") is keepalive
            ]
            for export_id, record in matches:
                registration = record.get("registration")
                if registration is None:
                    # A partially-created registration has no safe explicit
                    # fence.  Retain the mapping until process teardown.
                    return False
                try:
                    self.agent.deregister_memory(
                        registration, backends=self.backends
                    )
                except Exception as error:
                    record["cleanup_error"] = repr(error)
                    return False
                self.quarantined.pop(export_id, None)
            return True

    def prepare_read(self, shard: HostShard, *, read_id: str, tp_rank: int,
                     tp_size: int, layout: str, gpu_id: int,
                     spans: Sequence[tuple[int, int, int]],
                     payload_ranges: Sequence[tuple[int, int]] | None = None,
                     ) -> RemoteHostRead:
        """Prepare after source claim and full destination workset allocation.

        Spans must cover the entire exported shard exactly once in source order.
        The existing destination VRAM pool registration remains caller-owned.
        """
        shard.validate()
        if (shard.tp_rank, shard.tp_size, shard.layout) != (tp_rank, tp_size, layout):
            raise ValueError("TP shard/layout mismatch")
        if not read_id or gpu_id < 0:
            raise ValueError("missing read lease or invalid GPU")
        ranges = tuple(
            (int(start), int(length))
            for start, length in (
                payload_ranges
                if payload_ranges is not None
                else ((0, shard.byte_size),)
            )
        )
        if not ranges:
            raise ValueError("empty payload ranges")
        previous_end = 0
        for start, length in ranges:
            if start < previous_end or length <= 0 or start + length > shard.byte_size:
                raise ValueError("invalid payload ranges")
            previous_end = start + length
        if ranges[0][0] != 0 or previous_end != shard.byte_size:
            raise ValueError("payload ranges must cover both ends of the shard")

        range_index = 0
        offset = ranges[0][0]
        local, remote, destinations = [], [], []
        for source_offset, target_address, length in spans:
            while (
                range_index < len(ranges)
                and offset == ranges[range_index][0] + ranges[range_index][1]
            ):
                range_index += 1
                if range_index < len(ranges):
                    offset = ranges[range_index][0]
            if source_offset != offset or target_address <= 0 or length <= 0:
                raise ValueError("incomplete or invalid shard span")
            if (
                range_index >= len(ranges)
                or offset + length
                > ranges[range_index][0] + ranges[range_index][1]
            ):
                raise ValueError("span crosses payload range or padding")
            offset += length
            local.append((int(target_address), int(length), int(gpu_id)))
            remote.append((shard.address + int(source_offset), int(length), 0))
            destinations.append((target_address, target_address + length))
        if range_index != len(ranges) - 1 or offset != shard.byte_size:
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
            peer = self.agent.add_remote_agent(base64.b64decode(shard.metadata_b64))
            self._remote_peers.add(peer)
            try:
                local_desc = self.agent.get_xfer_descs(local, "VRAM")
                remote_desc = self.agent.get_xfer_descs(remote, "DRAM")
                handle = self.agent.initialize_xfer(
                    "READ", local_desc, remote_desc, peer, backends=self.backends
                )
                if handle is None:
                    raise RuntimeError("NIXL returned no READ handle")
            except Exception:
                # No transfer is posted by initialize. Keep the shared peer
                # connection live for other READs even if this attempt fails.
                raise
            pending = RemoteHostRead(self, shard, str(read_id), handle)
            pending.peer = peer
            self.reads[key] = (signature, pending)
            return pending
