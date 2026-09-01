"""Same-node admission control for large agentic Mooncake PUTs.

Mooncake Store exposes one shared data plane to all decode writers and the
prefill reader.  With many decode workers, unbounded ``batch_put_from`` calls
can occupy every Store worker and cause a latency-sensitive P-side GET to sit
behind an entire wave of snapshots.  This module limits only large PUT calls;
GETs and manifest operations remain unconstrained.

The implementation deliberately uses robust ``flock`` token files instead of
adding another service.  A process crash closes its file descriptor and
therefore releases the token automatically.  ``/dev/shm`` also keeps this
coordination off NFS.  This matches agentic KV V1's existing same-node scope.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator

logger = logging.getLogger(__name__)


class AgenticMooncakePutAdmission:
    """A crash-safe, process-shared semaphore for large Mooncake PUT calls."""

    def __init__(
        self,
        *,
        max_concurrent_puts: int,
        min_bytes: int,
        base_dir: str,
        store_identity: str,
    ) -> None:
        if max_concurrent_puts < 1:
            raise ValueError("max_concurrent_puts must be at least 1")
        if min_bytes < 0:
            raise ValueError("min_bytes must be non-negative")

        digest = hashlib.blake2b(
            store_identity.encode("utf-8"), digest_size=12
        ).hexdigest()
        self.token_dir = (
            Path(base_dir) / f"uid-{os.getuid()}" / f"store-{digest}"
        )
        self.token_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.token_paths = tuple(
            self.token_dir / f"put-{index}.lock"
            for index in range(max_concurrent_puts)
        )
        self.min_bytes = min_bytes
        self._cursor = os.getpid() % max_concurrent_puts
        self._cursor_lock = Lock()

    def _candidate_indices(self) -> tuple[int, ...]:
        with self._cursor_lock:
            start = self._cursor
            self._cursor = (self._cursor + 1) % len(self.token_paths)
        return tuple(
            (start + offset) % len(self.token_paths)
            for offset in range(len(self.token_paths))
        )

    @contextmanager
    def admit(self, byte_count: int) -> Iterator[float]:
        """Yield PUT admission wait seconds; small operations pass through."""

        if byte_count < self.min_bytes:
            yield 0.0
            return

        started_at = time.monotonic()
        last_warning_at = started_at
        acquired_fd = None
        acquired_path = None
        while acquired_fd is None:
            for index in self._candidate_indices():
                path = self.token_paths[index]
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                acquired_fd = fd
                acquired_path = path
                break

            if acquired_fd is None:
                now = time.monotonic()
                if now - last_warning_at >= 30.0:
                    logger.warning(
                        "Large Mooncake PUT has waited %.1fs for admission in %s",
                        now - started_at,
                        self.token_dir,
                    )
                    last_warning_at = now
                time.sleep(0.002)

        wait_seconds = time.monotonic() - started_at
        if wait_seconds >= 0.1:
            logger.debug(
                "Large Mooncake PUT admitted after %.3fs via %s",
                wait_seconds,
                acquired_path,
            )
        try:
            yield wait_seconds
        finally:
            fcntl.flock(acquired_fd, fcntl.LOCK_UN)
            os.close(acquired_fd)
