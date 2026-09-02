"""Atomic cross-process reservations for D->P slow-path routing."""

from __future__ import annotations

import fcntl
import json
import os
import time
from typing import Any, Iterable


class SharedPrefillPressureReservations:
    """Bridge stale pressure samples while independent D workers choose P.

    The Router publishes relatively expensive physical/load measurements.
    Decode workers consume that snapshot without blocking Decode, but several
    workers can otherwise select the same P before the next publication.  This
    tiny tmpfs ledger makes selection plus token/request charging one flock
    transaction.  Entries expire after physical Host/HBM pressure has had time
    to appear in a later Router sample.
    """

    VERSION = 1

    def __init__(self, path: str, *, ttl_seconds: float = 5.0):
        if not path:
            raise ValueError("Prefill reservation path is required")
        directory = os.path.dirname(path) or "."
        if directory != "/dev/shm" and not directory.startswith("/dev/shm/"):
            raise ValueError("Prefill reservations must reside in /dev/shm")
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.abspath(path)
        self.ttl_seconds = max(0.5, float(ttl_seconds))

    @staticmethod
    def _read(file_obj) -> dict[str, Any]:
        file_obj.seek(0)
        raw = file_obj.read()
        if not raw:
            return {"version": 1, "reservations": {}}
        payload = json.loads(raw)
        if payload.get("version") != 1:
            raise ValueError("unsupported Prefill reservation version")
        payload.setdefault("reservations", {})
        return payload

    @staticmethod
    def _write(file_obj, payload: dict[str, Any]) -> None:
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
        token_count: int,
        domains: Iterable[dict[str, Any]],
    ) -> int:
        """Choose the least-pressure P and atomically charge this generation."""

        snapshot_id = str(snapshot_id)
        token_count = max(1, int(token_count))
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

            reserved_tokens: dict[int, int] = {}
            reserved_requests: dict[int, int] = {}
            for value in reservations.values():
                domain = int(value["domain"])
                reserved_tokens[domain] = reserved_tokens.get(domain, 0) + int(
                    value.get("token_count", 0)
                )
                reserved_requests[domain] = reserved_requests.get(domain, 0) + 1

            scored: list[tuple[float, int]] = []
            for item in domains:
                domain = int(item["domain"])
                hbm_capacity = max(1, int(item.get("hbm_capacity_tokens", 0)))
                arena_capacity = max(1, int(item.get("arena_capacity_bytes", 0)))
                p2d_arena_capacity = max(
                    1, int(item.get("p2d_arena_capacity_bytes", arena_capacity))
                )
                token_pressure = (
                    max(0, int(item.get("pending_tokens", 0)))
                    + max(0, int(item.get("p2d_inflight_tokens", 0)))
                    + max(0, int(item.get("p2d_host_tokens", 0)))
                    + reserved_tokens.get(domain, 0)
                ) / hbm_capacity
                hbm_pressure = max(
                    0, int(item.get("hbm_used_tokens", 0))
                ) / hbm_capacity
                host_pressure = 2.0 * max(
                    0, int(item.get("arena_used_bytes", 0))
                ) / arena_capacity
                delivery_pressure = 2.0 * max(
                    0, int(item.get("p2d_host_bytes", 0))
                ) / p2d_arena_capacity
                request_pressure = 0.01 * (
                    max(0, int(item.get("pending_requests", 0)))
                    + max(0, int(item.get("scheduler_waiting", 0)))
                    + max(0, int(item.get("p2d_inflight_requests", 0)))
                    + max(0, int(item.get("p2d_host_requests", 0)))
                    + reserved_requests.get(domain, 0)
                )
                scored.append(
                    (
                        token_pressure
                        + hbm_pressure
                        + host_pressure
                        + delivery_pressure
                        + request_pressure,
                        domain,
                    )
                )
            if not scored:
                raise ValueError("empty Prefill pressure snapshot")
            _, selected = min(scored)
            reservations[snapshot_id] = {
                "domain": int(selected),
                "token_count": token_count,
                "created_at": now,
                "expires_at": now + self.ttl_seconds,
            }
            self._write(file_obj, payload)
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
            return int(selected)

    def totals(self) -> dict[int, tuple[int, int]]:
        """Return live token/request reservations grouped by logical P."""

        now = time.time()
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            payload = self._read(file_obj)
            before = len(payload["reservations"])
            self._prune(payload, now)
            if len(payload["reservations"]) != before:
                self._write(file_obj, payload)
            totals: dict[int, tuple[int, int]] = {}
            for value in payload["reservations"].values():
                domain = int(value["domain"])
                tokens, requests = totals.get(domain, (0, 0))
                totals[domain] = (
                    tokens + int(value.get("token_count", 0)),
                    requests + 1,
                )
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
            return totals
