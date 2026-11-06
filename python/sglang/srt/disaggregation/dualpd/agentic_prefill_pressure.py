"""Atomic Shared-Host reservations for D->P slow-path placement."""

from __future__ import annotations

import fcntl
import json
import os
import time
from typing import Any, Iterable


class SharedPrefillPressureReservations:
    """Bridge stale pressure samples while D workers choose a Host arena.

    The Router publishes relatively expensive physical/load measurements.
    Decode workers consume that snapshot without blocking Decode, but several
    workers can otherwise select the same arena before the next publication.
    This tiny tmpfs ledger makes selection plus byte charging one flock
    transaction. Entries expire after physical Host pressure has appeared in a
    later Router sample. P-HBM routing is intentionally a separate decision.
    """

    # V2 reservations are byte-sized.  V1 used token_count under the same
    # path; accepting it would silently undercharge Host capacity.
    VERSION = 2

    def __init__(self, path: str, *, ttl_seconds: float = 5.0):
        if not path:
            raise ValueError("Prefill reservation path is required")
        directory = os.path.dirname(path) or "."
        if directory != "/dev/shm" and not directory.startswith("/dev/shm/"):
            raise ValueError("Prefill reservations must reside in /dev/shm")
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.abspath(path)
        self.ttl_seconds = max(0.5, float(ttl_seconds))

    @classmethod
    def _read(cls, file_obj) -> dict[str, Any]:
        file_obj.seek(0)
        raw = file_obj.read()
        if not raw:
            return {"version": cls.VERSION, "reservations": {}}
        payload = json.loads(raw)
        if payload.get("version") != cls.VERSION:
            # Reservations are short-lived shadows rather than physical
            # ownership.  Clearing an incompatible schema is safe and avoids
            # treating V1 token counts as V2 bytes after a service restart.
            return {
                "version": cls.VERSION,
                "reservations": {},
                "_reset_incompatible": True,
            }
        payload.setdefault("reservations", {})
        return payload

    @staticmethod
    def _write(file_obj, payload: dict[str, Any]) -> None:
        payload.pop("_reset_incompatible", None)
        file_obj.seek(0)
        json.dump(payload, file_obj, separators=(",", ":"), sort_keys=True)
        file_obj.truncate()
        file_obj.flush()

    @staticmethod
    def _prune(payload: dict[str, Any], now: float) -> None:
        reservations = payload.setdefault("reservations", {})
        for snapshot_id, value in tuple(reservations.items()):
            if float(value.get("expires_at", 0.0)) <= now:
                reservations.pop(snapshot_id, None)

    def select_and_reserve(
        self,
        snapshot_id: str,
        byte_size: int,
        domains: Iterable[dict[str, Any]],
    ) -> int:
        """Choose the Host arena with most remaining bytes and charge it."""

        snapshot_id = str(snapshot_id)
        byte_size = max(1, int(byte_size))
        now = time.time()
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            payload = self._read(file_obj)
            self._prune(payload, now)
            reservations = payload["reservations"]
            existing = reservations.get(snapshot_id)
            if existing is not None:
                domain = int(existing["domain"])
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
                return domain

            reserved_bytes: dict[int, int] = {}
            for value in reservations.values():
                domain = int(value["domain"])
                reserved_bytes[domain] = reserved_bytes.get(domain, 0) + int(
                    value.get("byte_size", 0)
                )

            remaining: list[tuple[int, int]] = []
            for item in domains:
                domain = int(item["domain"])
                arena_capacity = int(item.get("arena_capacity_bytes", 0))
                if arena_capacity <= 0:
                    continue
                remaining.append(
                    (
                        arena_capacity
                        - int(item.get("arena_used_bytes", 0))
                        - reserved_bytes.get(domain, 0),
                        domain,
                    )
                )
            if not remaining:
                raise ValueError("empty Prefill pressure snapshot")
            # Stable domain-id tie-break keeps the decision deterministic.
            _, selected = max(remaining, key=lambda item: (item[0], -item[1]))
            reservations[snapshot_id] = {
                "domain": int(selected),
                "byte_size": byte_size,
                "created_at": now,
                "expires_at": now + self.ttl_seconds,
            }
            self._write(file_obj, payload)
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
            return int(selected)

    def totals(self) -> dict[int, tuple[int, int]]:
        """Return live byte/request reservations grouped by Host arena."""

        now = time.time()
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            payload = self._read(file_obj)
            before = len(payload["reservations"])
            reset_incompatible = bool(payload.pop("_reset_incompatible", False))
            self._prune(payload, now)
            if reset_incompatible or len(payload["reservations"]) != before:
                self._write(file_obj, payload)
            totals: dict[int, tuple[int, int]] = {}
            for value in payload["reservations"].values():
                domain = int(value["domain"])
                byte_count, requests = totals.get(domain, (0, 0))
                totals[domain] = (
                    byte_count + int(value.get("byte_size", 0)),
                    requests + 1,
                )
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
            return totals
