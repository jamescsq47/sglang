from __future__ import annotations

"""Asynchronous Prefill-HBM -> Host -> Decode-HBM staging.

This is the reverse direction of :mod:`agentic_host_staging`.  It is used only
after Prefill has completed and the late-binding router cannot immediately
find Decode capacity.  The control ledger and byte arena are deliberately
separate from D->P snapshots: pressure in one direction must never consume the
other direction's credit.

GPU allocation and release remain scheduler-owned.  Background workers only
copy immutable, scheduler-pinned pages and publish completion state.
"""

import logging
import os
import queue
import threading
import time
from typing import Any, Optional

import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    HostStageState,
    LayerFirstD2HStaging,
    PinnedMHAHostBounce,
    SharedHostSnapshotArena,
    SharedHostStagingLedger,
    SharedMHAHostSnapshot,
)
from sglang.srt.disaggregation.base import KVPoll

logger = logging.getLogger(__name__)

P2D_CUSTOM_SNAPSHOT_ID = "agentic_p2d_host_snapshot_id"
P2D_CUSTOM_PREFILL_DOMAIN = "agentic_p2d_prefill_domain"


def p2d_snapshot_id(bootstrap_room: int) -> str:
    """Return a collision-free id in the direction-specific ledger."""

    return f"p2d:{int(bootstrap_room)}"


def p2d_snapshot_from_req(req) -> Optional[str]:
    params = getattr(getattr(req, "sampling_params", None), "custom_params", None)
    if not isinstance(params, dict):
        return None
    value = params.get(P2D_CUSTOM_SNAPSHOT_ID)
    return None if value in (None, "") else str(value)


def _prefill_metadata(req) -> dict[str, Any]:
    """Serialize the small non-KV result normally carried by NIXL metadata."""

    if getattr(req, "return_logprob", False):
        raise ValueError("P->D Host staging V1 does not support return_logprob")
    if not req.output_ids:
        raise ValueError("Prefill result has no sampled output token")
    return {
        "output_id": int(req.output_ids[0]),
        "cached_tokens": int(getattr(req, "cached_tokens", 0) or 0),
        "cached_tokens_device": int(
            getattr(req, "cached_tokens_device", 0) or 0
        ),
        "cached_tokens_host": int(getattr(req, "cached_tokens_host", 0) or 0),
        "cached_tokens_storage": int(
            getattr(req, "cached_tokens_storage", 0) or 0
        ),
    }


class AgenticPToDHostStagingManager:
    """P-side producer for complete request-generation Host snapshots."""

    def __init__(
        self,
        *,
        ledger: SharedHostStagingLedger,
        device_pool,
        page_size: int,
        arena_directory: str,
        arena_capacity_bytes: int,
        prefill_domain: int,
        numa_node: int,
        hard_watermark: float = 0.90,
    ):
        if not (0.0 < hard_watermark <= 1.0):
            raise ValueError("P->D Host hard watermark must be in (0, 1]")
        self.ledger = ledger
        self.device_pool = device_pool
        self.page_size = int(page_size)
        self.prefill_domain = int(prefill_domain)
        self.numa_node = int(numa_node)
        self.hard_watermark = float(hard_watermark)
        self.owner = f"p2d-p:{os.getpid()}"
        self.arena = SharedHostSnapshotArena(
            arena_directory, int(arena_capacity_bytes)
        )
        self.chunk_tokens = max(
            self.page_size,
            int(os.getenv("SGLANG_AGENTIC_KV_P2D_D2H_CHUNK_TOKENS", "512")),
        )
        self.chunk_tokens = max(
            self.page_size,
            self.chunk_tokens // self.page_size * self.page_size,
        )
        self._stream = torch.cuda.Stream(device=torch.cuda.current_device(), priority=0)
        self._staging = LayerFirstD2HStaging(self.device_pool, self.chunk_tokens)
        self._bounce = PinnedMHAHostBounce(self.device_pool, self.chunk_tokens)
        self._work: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._active: dict[str, dict[str, Any]] = {}
        self._results: dict[str, int] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._pending_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"agentic-p2d-spill-{os.getpid()}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Agentic P->D Host staging enabled directory=%s capacity_gib=%.1f "
            "P=%d numa=%d chunk_tokens=%d",
            self.arena.directory,
            self.arena.capacity_bytes / (1024**3),
            self.prefill_domain,
            self.numa_node,
            self.chunk_tokens,
        )

    def _byte_size(self, token_count: int) -> int:
        return (
            2
            * int(token_count)
            * int(self.device_pool.layer_num)
            * int(self.device_pool.head_num)
            * int(self.device_pool.head_dim)
            * int(self.device_pool.store_dtype.itemsize)
        )

    def _targets_this_p(self, entry: dict[str, Any]) -> bool:
        return int(entry.get("prefill_domain", -1)) == self.prefill_domain

    def has_offer(self, req) -> bool:
        snapshot_id = p2d_snapshot_id(req.bootstrap_room)
        entry = self.ledger.get(snapshot_id)
        return bool(
            entry is not None
            and entry.get("state") == HostStageState.OFFERED.value
            and self._targets_this_p(entry)
        )

    def try_submit(self, req, source_indices) -> bool:
        """Atomically claim one Router offer and enqueue immutable D2H work.

        This method is called by the P scheduler.  It performs no Host copy and
        no CUDA synchronization.
        """

        snapshot_id = p2d_snapshot_id(req.bootstrap_room)
        with self._lock:
            if snapshot_id in self._active or snapshot_id in self._results:
                return True
        entry = self.ledger.get(snapshot_id)
        if (
            entry is None
            or entry.get("state") != HostStageState.OFFERED.value
            or not self._targets_this_p(entry)
        ):
            return False
        token_count = len(req.origin_input_ids)
        if token_count <= 0:
            # Normal NIXL remains the correctness fallback for shapes the V1
            # pageable arena cannot represent atomically.
            return False
        if len(source_indices) != token_count:
            raise RuntimeError("P->D staging source token count mismatch")
        byte_size = self._byte_size(token_count)
        with self._lock:
            if (
                self.arena.used_bytes + self._pending_bytes + byte_size
                > int(self.arena.capacity_bytes * self.hard_watermark)
            ):
                return False
        claimed = self.ledger.claim(snapshot_id, self.owner)
        if claimed is None:
            return False
        try:
            record = {
                "source_indices": source_indices,
                "token_count": token_count,
                "byte_size": byte_size,
                "prefill_metadata": _prefill_metadata(req),
                "started_at": time.monotonic(),
            }
            with self._lock:
                self._pending_bytes += byte_size
                self._active[snapshot_id] = record
            req._agentic_p2d_host_snapshot_id = snapshot_id
            self._work.put((snapshot_id, record))
            logger.info(
                "AgenticKV p2d_host_d2h_queued snapshot=%s tokens=%d bytes=%d "
                "P=%d numa=%d",
                snapshot_id,
                token_count,
                byte_size,
                self.prefill_domain,
                self.numa_node,
            )
            return True
        except Exception as exc:
            self.ledger.transition(
                snapshot_id,
                HostStageState.FAILED,
                owner=self.owner,
                reason=f"p2d_submit_failed:{exc}",
            )
            logger.exception("Failed to submit P->D Host staging for %s", snapshot_id)
            return False

    def is_active(self, req) -> bool:
        if getattr(req, "_agentic_p2d_host_terminal", False):
            return True
        snapshot_id = getattr(req, "_agentic_p2d_host_snapshot_id", None)
        if snapshot_id is None:
            return False
        with self._lock:
            return snapshot_id in self._active or snapshot_id in self._results

    def poll(self, req) -> Optional[int]:
        snapshot_id = getattr(req, "_agentic_p2d_host_snapshot_id", None)
        if snapshot_id is None:
            return None
        with self._lock:
            result = self._results.get(snapshot_id)
            active = snapshot_id in self._active
        if result == int(KVPoll.Success):
            return result
        if result == int(KVPoll.Failed):
            # The Router observes FAILED and resumes the normal direct path.
            # Keep the already-computed P snapshot alive for that retry.
            with self._lock:
                self._results.pop(snapshot_id, None)
            delattr(req, "_agentic_p2d_host_snapshot_id")
            return None
        return int(KVPoll.Transferring) if active else None

    def mark_scheduler_consumed(self, req) -> None:
        snapshot_id = getattr(req, "_agentic_p2d_host_snapshot_id", None)
        if snapshot_id is None:
            return
        with self._lock:
            self._results.pop(snapshot_id, None)
        # The independent NIXL consumer may still hold this request in its
        # round-robin queue.  Publish a terminal level-trigger so it removes
        # the request instead of resurrecting the unused direct sender.
        req._agentic_p2d_host_terminal = True

    def close(self) -> None:
        """Stop the background worker before its tmpfs control files vanish."""

        self._stop.set()
        self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot_id, record = self._work.get(timeout=0.1)
            except queue.Empty:
                self._cleanup_consumed()
                continue
            try:
                snapshot = self.arena.create(
                    snapshot_id,
                    int(record["token_count"]),
                    self.device_pool,
                    int(record["byte_size"]),
                )
                with self._lock:
                    self._pending_bytes = max(
                        0, self._pending_bytes - int(record["byte_size"])
                    )
                    record["snapshot"] = snapshot
                    self._records[snapshot_id] = record
                grant = {
                    "kind": "shared_host_extent",
                    "arena_path": snapshot.path,
                    "byte_size": int(record["byte_size"]),
                    "token_count": int(record["token_count"]),
                    "prefill_domain": self.prefill_domain,
                    "arena_numa_node": self.numa_node,
                    "prefill_metadata": record["prefill_metadata"],
                }
                if not self.ledger.publish_grants(
                    snapshot_id, self.owner, [grant]
                ):
                    raise RuntimeError("P->D grant publication was rejected")
                snapshot = record["snapshot"].materialize()
                source_indices = record["source_indices"]
                token_count = int(record["token_count"])
                for start in range(0, token_count, self.chunk_tokens):
                    end = min(start + self.chunk_tokens, token_count)
                    event, _ = snapshot.start_backup_range_from_device(
                        source_indices[start:end],
                        destination_start=start,
                        stream=self._stream,
                        staging=self._staging,
                        host_bounce=self._bounce,
                    )
                    event.synchronize()
                    snapshot.commit_backup_range_from_bounce(
                        self._bounce,
                        destination_start=start,
                        token_count=end - start,
                    )
                if not self.ledger.ack_chunk(snapshot_id, self.owner, 0):
                    raise RuntimeError("P->D Host write ACK was rejected")
                if not self.ledger.mark_host_ready(snapshot_id, self.owner, 1):
                    raise RuntimeError("P->D HOST_READY publication was rejected")
                elapsed = time.monotonic() - float(record["started_at"])
                with self._lock:
                    self._active.pop(snapshot_id, None)
                    self._results[snapshot_id] = int(KVPoll.Success)
                logger.info(
                    "AgenticKV p2d_host_d2h_complete snapshot=%s tokens=%d "
                    "elapsed_ms=%.3f gib_per_s=%.3f",
                    snapshot_id,
                    token_count,
                    elapsed * 1000.0,
                    int(record["byte_size"]) / max(elapsed, 1e-9) / (1024**3),
                )
            except Exception as exc:
                with self._lock:
                    if "snapshot" not in record:
                        self._pending_bytes = max(
                            0, self._pending_bytes - int(record["byte_size"])
                        )
                self.ledger.transition(
                    snapshot_id,
                    HostStageState.FAILED,
                    owner=self.owner,
                    reason=f"p2d_d2h_failed:{exc}",
                )
                with self._lock:
                    self._active.pop(snapshot_id, None)
                    self._results[snapshot_id] = int(KVPoll.Failed)
                logger.exception("P->D Host D2H failed for %s", snapshot_id)
            finally:
                record["source_indices"] = None
            self._cleanup_consumed()

    def _cleanup_consumed(self) -> None:
        with self._lock:
            records = list(self._records.items())
        for snapshot_id, record in records:
            entry = self.ledger.get(snapshot_id)
            if entry is None or entry.get("state") not in {
                HostStageState.CONSUMED.value,
                HostStageState.FAILED.value,
                HostStageState.REJECTED.value,
            }:
                continue
            # Never unlink while this process still has an active D2H event.
            with self._lock:
                if snapshot_id in self._active:
                    continue
                current = self._records.pop(snapshot_id, None)
            if current is not None:
                self.arena.release(current["snapshot"])
                logger.info(
                    "AgenticKV p2d_host_release snapshot=%s state=%s",
                    snapshot_id,
                    entry.get("state"),
                )


class AgenticPToDHostLoadManager:
    """D-side serialized, asynchronous Host->HBM loader."""

    def __init__(
        self,
        *,
        ledger: SharedHostStagingLedger,
        device_pool,
        page_size: int,
        decode_domain: int,
        numa_node: int,
    ):
        self.ledger = ledger
        self.device_pool = device_pool
        self.page_size = int(page_size)
        self.decode_domain = int(decode_domain)
        self.numa_node = int(numa_node)
        self.chunk_tokens = max(
            self.page_size,
            int(os.getenv("SGLANG_AGENTIC_KV_P2D_H2D_CHUNK_TOKENS", "1024")),
        )
        self.chunk_tokens = max(
            self.page_size,
            self.chunk_tokens // self.page_size * self.page_size,
        )
        self._stream = torch.cuda.Stream(device=torch.cuda.current_device(), priority=0)
        self._staging = LayerFirstD2HStaging(self.device_pool, self.chunk_tokens)
        self._bounce = PinnedMHAHostBounce(self.device_pool, self.chunk_tokens)
        self._work: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"agentic-p2d-load-{os.getpid()}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._work.put(None)
        self._thread.join(timeout=2.0)

    def submit(self, receiver: "AgenticPToDHostReceiver", device_indices) -> None:
        if receiver._submitted:
            return
        entry = self.ledger.get(receiver.snapshot_id)
        if entry is None or entry.get("state") != HostStageState.HOST_READY.value:
            raise RuntimeError("P->D Host snapshot is not ready")
        grants = entry.get("grants", [])
        if len(grants) != 1 or grants[0].get("kind") != "shared_host_extent":
            raise RuntimeError("P->D Host snapshot has no complete extent")
        grant = grants[0]
        if int(grant.get("arena_numa_node", -1)) != self.numa_node:
            raise RuntimeError(
                "P->D slow path crossed NUMA: "
                f"arena={grant.get('arena_numa_node')} D={self.numa_node}"
            )
        if int(grant["token_count"]) != len(device_indices):
            raise RuntimeError("P->D Host destination token count mismatch")
        owner = entry.get("p_owner")
        if not self.ledger.transition(
            receiver.snapshot_id, HostStageState.H2D_LOADING, owner=owner
        ):
            raise RuntimeError("P->D H2D ownership transition was rejected")
        receiver._submitted = True
        receiver._poll = int(KVPoll.Transferring)
        receiver._grant = grant
        receiver._owner = owner
        self._work.put((receiver, device_indices))

    def _worker(self) -> None:
        while not self._stop.is_set():
            work = self._work.get()
            if work is None:
                break
            receiver, device_indices = work
            started_at = time.monotonic()
            snapshot = None
            try:
                grant = receiver._grant
                snapshot = SharedMHAHostSnapshot(
                    path=str(grant["arena_path"]),
                    token_count=int(grant["token_count"]),
                    device_pool=self.device_pool,
                    byte_size=int(grant["byte_size"]),
                    create=False,
                )
                for start in range(0, len(device_indices), self.chunk_tokens):
                    end = min(start + self.chunk_tokens, len(device_indices))
                    event, _ = snapshot.start_load_range_to_device(
                        device_indices[start:end],
                        self._stream,
                        source_start=start,
                        staging=self._staging,
                        host_bounce=self._bounce,
                    )
                    event.synchronize()
                if not self.ledger.transition(
                    receiver.snapshot_id,
                    HostStageState.CONSUMED,
                    owner=receiver._owner,
                ):
                    raise RuntimeError("P->D CONSUMED publication was rejected")
                receiver._poll = int(KVPoll.Success)
                elapsed = time.monotonic() - started_at
                logger.info(
                    "AgenticKV p2d_host_h2d_complete snapshot=%s tokens=%d "
                    "elapsed_ms=%.3f gib_per_s=%.3f D_domain=%d numa=%d",
                    receiver.snapshot_id,
                    len(device_indices),
                    elapsed * 1000.0,
                    int(grant["byte_size"]) / max(elapsed, 1e-9) / (1024**3),
                    self.decode_domain,
                    self.numa_node,
                )
            except Exception as exc:
                try:
                    self.ledger.transition(
                        receiver.snapshot_id,
                        HostStageState.FAILED,
                        owner=receiver._owner,
                        reason=f"p2d_h2d_failed:{exc}",
                    )
                except Exception:
                    logger.exception("Failed to publish P->D H2D failure")
                receiver._error = exc
                receiver._poll = int(KVPoll.Failed)
                logger.exception("P->D Host H2D failed for %s", receiver.snapshot_id)
            finally:
                if snapshot is not None:
                    snapshot.close(unlink=False)


class AgenticPToDHostReceiver:
    """Small receiver adapter consumed by DecodeTransferQueue."""

    require_staging = False

    def __init__(self, manager: AgenticPToDHostLoadManager, snapshot_id: str):
        self.manager = manager
        self.snapshot_id = str(snapshot_id)
        self._poll = int(KVPoll.WaitingForInput)
        self._submitted = False
        self._grant: Optional[dict[str, Any]] = None
        self._owner: Optional[str] = None
        self._error: Optional[BaseException] = None

    def init(self, _prefill_dp_rank: int) -> None:
        return

    def bind(self, device_indices) -> None:
        self.manager.submit(self, device_indices)

    def poll(self):
        return self._poll

    def failure_exception(self):
        if self._error is not None:
            raise self._error

    def clear(self) -> None:
        return

    def abort(self) -> None:
        self._poll = int(KVPoll.Failed)

    def commit_req(self, req) -> None:
        if self._grant is None:
            raise RuntimeError("P->D Host metadata is missing")
        metadata = self._grant.get("prefill_metadata") or {}
        req.output_ids.append(int(metadata["output_id"]))
        req.cached_tokens = int(metadata.get("cached_tokens", 0))
        req.cached_tokens_device = int(metadata.get("cached_tokens_device", 0))
        req.cached_tokens_host = int(metadata.get("cached_tokens_host", 0))
        req.cached_tokens_storage = int(metadata.get("cached_tokens_storage", 0))
