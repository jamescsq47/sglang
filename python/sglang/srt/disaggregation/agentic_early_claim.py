"""Node-local early-arrival markers for the agentic D-to-P direct path.

The HTTP router publishes a tiny marker as soon as a later agent turn arrives.
Decode workers use the marker only to distinguish a fast tool return from a
slow one.  It allocates no P HBM and carries no global capacity/credit policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration


_VERSION = 1


class AgenticEarlyClaimStore:
    def __init__(self, directory: str):
        if not directory:
            raise ValueError("early-claim directory must be non-empty")
        self.directory = Path(directory)
        self.marker_directory = self.directory / "arrivals"
        self.final_directory = self.directory / "finals"
        self.tool_directory = self.directory / "tool-valid"
        self.marker_directory.mkdir(parents=True, exist_ok=True)
        self.final_directory.mkdir(parents=True, exist_ok=True)
        self.tool_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(request: RequestGeneration) -> str:
        return hashlib.sha256(request.snapshot_id.encode("utf-8")).hexdigest()

    def marker_path(self, request: RequestGeneration) -> Path:
        return self.marker_directory / f"{self._digest(request)}.json"

    def final_path(self, request: RequestGeneration) -> Path:
        return self.final_directory / f"{self._digest(request)}.json"

    def tool_path(self, request: RequestGeneration) -> Path:
        return self.tool_directory / f"{self._digest(request)}.json"

    def producer_path(self, request: RequestGeneration) -> Path:
        # Keep producer tombstones at the top level so the run-script's
        # bounded /dev/shm cleanup removes them without a recursive scan.
        return self.directory / f"producer-{self._digest(request)}"

    def claim_generation_producer(self, request: RequestGeneration) -> bool:
        """Elect exactly one D producer for a request-generation.

        Long model calls can outlive an HTTP client's retry timeout.  A retry
        may then be routed to a different D and finish concurrently with the
        original.  Retain this tiny O_EXCL tombstone for the run so only the
        first D may publish or mutate the generation's KV lifecycle.
        """

        path = self.producer_path(request)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
        finally:
            os.close(fd)
        return True

    @staticmethod
    def _publish(path: Path, request: RequestGeneration, kind: str) -> dict[str, Any]:
        now = time.time()
        payload = {
            "version": _VERSION,
            "kind": kind,
            "snapshot_id": request.snapshot_id,
            # Keep the structured identity in addition to snapshot_id so the
            # P worker can start a reverse transfer before the tokenized Req
            # reaches its scheduler.  Parsing snapshot_id would be ambiguous
            # when an application request id itself contains a colon.
            "request_id": request.request_id,
            "generation": request.generation,
            "arrived_at": now,
            "publisher_pid": os.getpid(),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as file_obj:
                file_obj.write(data)
                file_obj.flush()
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return payload

    def publish_arrival(self, request: RequestGeneration) -> dict[str, Any]:
        return self._publish(self.marker_path(request), request, "arrival")

    def publish_final(self, request: RequestGeneration) -> dict[str, Any]:
        """Confirm that the application consumed this output as terminal."""

        return self._publish(self.final_path(request), request, "final")

    def publish_tool(self, request: RequestGeneration) -> dict[str, Any]:
        """Confirm that the application parser accepted a real tool call."""

        return self._publish(self.tool_path(request), request, "tool")

    @staticmethod
    def _read(
        path: Path,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        try:
            payload = json.loads(path.read_bytes())
            arrived_at = float(payload["arrived_at"])
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        now = time.time()
        if (
            payload.get("version") != _VERSION
            or payload.get("snapshot_id") != request.snapshot_id
            or arrived_at + max_age_seconds < now
            or arrived_at + 0.05 < not_before
        ):
            return None
        return payload

    def read_arrival(
        self,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        return self._read(
            self.marker_path(request),
            request,
            not_before=not_before,
            max_age_seconds=max_age_seconds,
        )

    def iter_arrivals(
        self, *, max_age_seconds: float
    ) -> list[tuple[RequestGeneration, dict[str, Any]]]:
        """Return valid arrival markers without consuming them.

        Decode removes the marker after Direct completion or slow fallback.
        P therefore only observes markers here; consuming one in P could race
        with Decode's fast-tool-window check.
        """

        now = time.time()
        arrivals: list[tuple[RequestGeneration, dict[str, Any]]] = []
        try:
            paths = tuple(self.marker_directory.glob("*.json"))
        except OSError:
            return arrivals
        for path in paths:
            try:
                payload = json.loads(path.read_bytes())
                request = RequestGeneration(
                    str(payload["request_id"]), int(payload["generation"])
                )
                arrived_at = float(payload["arrived_at"])
            except (
                FileNotFoundError,
                OSError,
                TypeError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ):
                continue
            if (
                payload.get("version") != _VERSION
                or payload.get("kind") != "arrival"
                or payload.get("snapshot_id") != request.snapshot_id
                or arrived_at + max_age_seconds < now
            ):
                continue
            arrivals.append((request, payload))
        arrivals.sort(key=lambda item: float(item[1]["arrived_at"]))
        return arrivals

    def read_final(
        self,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        return self._read(
            self.final_path(request),
            request,
            not_before=not_before,
            max_age_seconds=max_age_seconds,
        )

    def read_tool(
        self,
        request: RequestGeneration,
        *,
        not_before: float,
        max_age_seconds: float,
    ) -> Optional[dict[str, Any]]:
        return self._read(
            self.tool_path(request),
            request,
            not_before=not_before,
            max_age_seconds=max_age_seconds,
        )

    def remove_arrival(self, request: RequestGeneration) -> None:
        """Remove only the ingress marker; no capacity ledger is involved."""

        try:
            self.marker_path(request).unlink(missing_ok=True)
        except OSError:
            pass

    def remove_final(self, request: RequestGeneration) -> None:
        try:
            self.final_path(request).unlink(missing_ok=True)
        except OSError:
            pass

    def remove_tool(self, request: RequestGeneration) -> None:
        try:
            self.tool_path(request).unlink(missing_ok=True)
        except OSError:
            pass
