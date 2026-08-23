from __future__ import annotations

"""Small rank-mailbox primitives for agentic TP control.

The model scheduler already broadcasts one Python control record from TP rank
zero to every follower.  Agentic KV transport therefore needs only the reverse
direction: each physical rank reports completion of the command it was given.
This module implements that report path with exact, run-scoped files in
``/dev/shm``.  It deliberately performs no directory scans and no distributed
collectives, so CUDA/NCCL model execution cannot be reordered by transport
progress.
"""

import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional


class TPGroupMailbox:
    """Rank-local reports and one rank-zero logical receipt.

    A key is a request-generation identity (snapshot id or ``rid@room``), never
    a bare request id.  Followers may only publish their local status.  Rank
    zero reads all rank files, decides the logical transition, and carries that
    decision on the scheduler's existing native TP broadcast.
    """

    def __init__(
        self,
        namespace: str,
        *,
        tp_rank: int,
        tp_size: int,
        directory: Optional[str] = None,
    ) -> None:
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid TP rank/size")
        root = directory or os.getenv("SGLANG_PD_P_READY_DIR", "/dev/shm")
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
