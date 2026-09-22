from __future__ import annotations

"""Small rank-mailbox primitives for agentic TP control.

The model scheduler already broadcasts one Python control record from TP rank
zero to every follower.  Agentic KV transport therefore needs only the reverse
direction: each physical rank reports completion of the command it was given.
With SGLANG_AGENTIC_TP_EVENT_ENDPOINT, the factory selects persistent socket
reports and cached coordinator receipts: no mailbox files or directory scans.
The legacy single-node file backend remains available when no endpoint is
configured. Neither backend introduces model collectives or CUDA operations.
"""

import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from sglang.srt.disaggregation.base import KVPoll


def bind_direct_wire_mailboxes(manifest, *mailboxes):
    """Retain the existing immutable NIXL wire room as the TP Direct attempt.

    File backends remain untouched. Socket callers must never silently rebind
    a snapshot to a different room; that requires explicit captured EventKeys.
    """
    selected = [
        mailbox for mailbox in mailboxes
        if callable(getattr(mailbox, "bind_identity", None))
    ]
    if not selected:
        return
    from sglang.srt.disaggregation.agentic_tp_events import EventKey

    if manifest.direct_room is None:
        raise ValueError("Direct TP control requires the exact NIXL wire room")
    identity = EventKey(manifest.snapshot_id, f"direct-room:{manifest.direct_room}")
    for mailbox in selected:
        mailbox.bind_identity(manifest.snapshot_id, identity)


def tp_command_identity(mailbox, snapshot_id, attempt=None):
    """A rank-zero command's immutable attempt; file callers are unchanged."""
    if not callable(getattr(mailbox, "bind_identity", None)):
        return snapshot_id
    from sglang.srt.disaggregation.agentic_tp_events import EventKey

    if not attempt:
        raise ValueError("socket TP command is missing rank-zero attempt identity")
    return EventKey(str(snapshot_id), str(attempt))


class TPGroupMailbox:
    """Rank-local reports and one rank-zero logical receipt.

    A key is a request-generation identity (snapshot id or ``rid@room``), never
    a bare request id.  Followers may only publish their local status.  Rank
    zero reduces local reports, decides the logical transition, and carries
    that decision on the scheduler's existing native TP broadcast. Endpoint
    mode dispatches to SocketTPGroupMailbox before touching any file path.
    """

    def __new__(cls, *args, **kwargs):
        if cls is TPGroupMailbox and os.getenv("SGLANG_AGENTIC_TP_EVENT_ENDPOINT"):
            from sglang.srt.disaggregation.agentic_tp_socket_mailbox import SocketTPGroupMailbox

            return SocketTPGroupMailbox(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        namespace: str,
        *,
        tp_rank: int,
        tp_size: int,
        directory: Optional[str] = None,
        group_local_directory: Optional[str] = None,
        nnodes: int = 1,
    ) -> None:
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid TP rank/size")
        root = directory or os.getenv("SGLANG_PD_P_READY_DIR", "/dev/shm")
        # Only intra-engine reports can use node-local tmpfs. In particular,
        # p2d-receiver is deliberately NOT here: D writes its receipt, P reads
        # and clears it. Unknown/new namespaces remain shared by default.
        if group_local_directory:
            if int(nnodes) != 1:
                raise ValueError("local TP mailboxes require the entire TP group on one node")
            if namespace in {
                "d2p-direct", "d2p-direct-abort-p", "p-workset-retire",
                "p2d-sender", "p2d-cleanup", "p2d-admission",
            } or namespace.startswith("d2p-host:"):
                root = group_local_directory
        digest = hashlib.sha256(str(namespace).encode("utf-8")).hexdigest()[:16]
        self.directory = Path(root) / f"tp-control-{digest}"
        self.directory.mkdir(parents=True, exist_ok=True)
        # Status changes are sparse (prepared -> transferring -> complete),
        # while progress loops run every few milliseconds.  Avoid replacing
        # the same tmpfs file on every loop and avoid reparsing unchanged peer
        # files.  The filesystem remains authoritative across processes.
        self._published: dict[Path, int] = {}
        self._read_cache: dict[Path, tuple[int, int, int, int]] = {}
        self._cache_lock = threading.RLock()

    @staticmethod
    def _digest(key: object) -> str:
        return hashlib.sha256(str(key).encode("utf-8")).hexdigest()

    def _rank_path(self, key: object, rank: int) -> Path:
        return self.directory / f"{self._digest(key)}.rank-{int(rank)}"

    def _receipt_path(self, key: object) -> Path:
        return self.directory / f"{self._digest(key)}.receipt"

    @staticmethod
    def _atomic_write(path: Path, status: int) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.write(fd, f"{int(status)}\n".encode("ascii"))
            os.close(fd)
            fd = -1
            os.replace(temporary, path)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _read_uncached(path: Path) -> Optional[int]:
        try:
            return int(path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, OSError, ValueError):
            return None

    def _read(self, path: Path) -> Optional[int]:
        with self._cache_lock:
            try:
                stat = path.stat()
            except OSError:
                self._read_cache.pop(path, None)
                return None
            cached = self._read_cache.get(path)
            signature = (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))
            if cached is not None and cached[:3] == signature:
                return cached[3]
            status = self._read_uncached(path)
            if status is not None:
                self._read_cache[path] = (*signature, int(status))
            return status

    def publish_local(self, key: object, status: int) -> None:
        path = self._rank_path(key, self.tp_rank)
        status = int(status)
        with self._cache_lock:
            if self._published.get(path) == status and path.exists():
                return
            self._atomic_write(path, status)
            self._published[path] = status
            self._read_cache.pop(path, None)

    def publish_local_progress(self, key: object, status: int) -> None:
        """Publish a terminal-safe monotonic progress state.

        Non-negative states may only advance.  A negative failure is terminal
        and cannot be overwritten by a stale background success snapshot.
        This is intentionally separate from ``publish_local`` because KVPoll
        users encode failure as zero and do not share this ordering contract.
        """

        path = self._rank_path(key, self.tp_rank)
        status = int(status)
        with self._cache_lock:
            current = self._published.get(path)
            if current is None:
                current = self._read(path)
            if current is not None and (
                current < 0 or (status >= 0 and status <= current)
            ):
                return
            self._atomic_write(path, status)
            self._published[path] = status
            self._read_cache.pop(path, None)

    def local_status(self, key: object, rank: Optional[int] = None) -> Optional[int]:
        return self._read(
            self._rank_path(key, self.tp_rank if rank is None else int(rank))
        )

    def group_status(self, key: object) -> Optional[int]:
        """Return the minimum rank status once every shard has reported."""

        statuses = [self.local_status(key, rank) for rank in range(self.tp_size)]
        if any(status is None for status in statuses):
            return None
        return min(int(status) for status in statuses if status is not None)

    def transfer_group_status(self, key: object) -> tuple[Optional[int], bool]:
        """Reduce a KV transfer without turning peer failure into a fake fence.

        Returns ``(status, cancel_requested)``.  One failed rank requests
        cancellation, but group failure is terminal only after every rank is
        physically terminal.  Until then the logical transfer remains in
        ``Transferring`` so no peer's destination pages can be recycled.
        """

        statuses = [self.local_status(key, rank) for rank in range(self.tp_size)]
        if any(status is None for status in statuses):
            return None, False
        observed = [int(status) for status in statuses if status is not None]
        cancel_requested = KVPoll.Failed in observed
        terminal = {int(KVPoll.Failed), int(KVPoll.Success)}
        if cancel_requested:
            if all(status in terminal for status in observed):
                return int(KVPoll.Failed), True
            return int(KVPoll.Transferring), True
        if all(status == int(KVPoll.Success) for status in observed):
            return int(KVPoll.Success), False
        return min(observed), False

    def publish_receipt(self, key: object, status: int) -> None:
        if self.tp_rank != 0:
            raise RuntimeError("only TP rank zero may publish a logical receipt")
        path = self._receipt_path(key)
        status = int(status)
        with self._cache_lock:
            if self._published.get(path) == status and path.exists():
                return
            self._atomic_write(path, status)
            self._published[path] = status
            self._read_cache.pop(path, None)

    def receipt(self, key: object) -> Optional[int]:
        return self._read(self._receipt_path(key))

    def publish_local_rollback_complete(self, key: object) -> None:
        """ACK physical rollback independently of sticky transfer failure.

        Only the Direct cleanup owner calls this after its DMA fence and
        local rollback.  It must never turn a failed transfer into success.
        """
        self.publish_local_progress(("direct-rollback", key), 1)

    def rollback_group_complete(self, key: object) -> bool:
        return self.group_status(("direct-rollback", key)) == 1

    def clear_local_rollback(self, key: object) -> None:
        self.clear_local(("direct-rollback", key))

    def clear_group_rollback(self, key: object) -> None:
        self.clear_group(("direct-rollback", key))

    def clear_local(self, key: object) -> None:
        path = self._rank_path(key, self.tp_rank)
        with self._cache_lock:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            self._published.pop(path, None)
            self._read_cache.pop(path, None)

    def clear_group(self, key: object) -> None:
        """Remove one completed generation; callers must already hold TP0 authority."""

        if self.tp_rank != 0:
            raise RuntimeError("only TP rank zero may clear a logical generation")
        with self._cache_lock:
            for rank in range(self.tp_size):
                path = self._rank_path(key, rank)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                self._published.pop(path, None)
                self._read_cache.pop(path, None)
            receipt = self._receipt_path(key)
            try:
                receipt.unlink()
            except FileNotFoundError:
                pass
            self._published.pop(receipt, None)
            self._read_cache.pop(receipt, None)
