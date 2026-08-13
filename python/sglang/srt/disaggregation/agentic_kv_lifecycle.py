from __future__ import annotations

import json
import hashlib
import base64
import fcntl
import math
import os
import threading
import time
import unicodedata
import zlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence


MANIFEST_VERSION = 1
MANIFEST_PREFIX = "sglang:agentic-kv:v1:manifest:"
CUSTOM_REQUEST_ID = "agentic_request_id"
CUSTOM_GENERATION = "agentic_generation"
CUSTOM_PARENT_GENERATION = "agentic_parent_generation"
CUSTOM_TOOL_TYPE = "agentic_tool_type"
CUSTOM_TOOL_SUFFIXES = "agentic_tool_suffix_token_ids"
CUSTOM_TERMINAL_MARKERS = "agentic_terminal_marker_token_ids"
CUSTOM_TOOL_SUFFIX_STRINGS = "agentic_tool_suffix_strings"
CUSTOM_TERMINAL_MARKER_STRINGS = "agentic_terminal_marker_strings"
EXTRA_KEY_ENVELOPE_PREFIX = "agentic-v1e:"
_AGENTIC_WIRE_KEYS = frozenset(
    {
        CUSTOM_REQUEST_ID,
        CUSTOM_GENERATION,
        CUSTOM_PARENT_GENERATION,
        CUSTOM_TOOL_TYPE,
        CUSTOM_TOOL_SUFFIXES,
        CUSTOM_TERMINAL_MARKERS,
        CUSTOM_TOOL_SUFFIX_STRINGS,
        CUSTOM_TERMINAL_MARKER_STRINGS,
    }
)


def _decode_envelope_component(value: str, *, max_bytes: int) -> bytes:
    if not value or len(value) > max_bytes * 2:
        raise ValueError("invalid agentic envelope component size")
    padding = "=" * (-len(value) % 4)
    decoded = base64.b64decode(
        value + padding, altchars=b"-_", validate=True
    )
    if len(decoded) > max_bytes:
        raise ValueError("agentic envelope component is too large")
    return decoded


def _discard_shared_ledger_snapshot(snapshot_id: str) -> None:
    """Best-effort removal from the node-local admission ledger.

    P is a different process from every D writer, so a successful P-side
    load-then-delete cannot update a D worker's in-memory index directly.  The
    shared ledger is the authority for cross-D admission; remove the terminal
    generation under the same file lock as reserve/commit so capacity is
    released immediately rather than at the next D reconciliation.
    """

    path = os.getenv("SGLANG_AGENTIC_KV_LEDGER_PATH", "")
    directory = os.path.dirname(path)
    if not path or (
        directory != "/dev/shm" and not directory.startswith("/dev/shm/")
    ):
        return
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return
    except OSError:
        return
    try:
        with os.fdopen(fd, "r+", encoding="utf-8") as file_obj:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            try:
                file_obj.seek(0)
                raw = file_obj.read()
                ledger = json.loads(raw) if raw else {}
                if ledger.get("version") != 1:
                    return
                removed = False
                for section in ("reservations", "residents"):
                    entries = ledger.get(section)
                    if isinstance(entries, dict) and snapshot_id in entries:
                        entries.pop(snapshot_id, None)
                        removed = True
                terminals = ledger.setdefault("terminals", {})
                if isinstance(terminals, dict):
                    terminals[snapshot_id] = time.time()
                    removed = True
                if removed:
                    file_obj.seek(0)
                    json.dump(
                        ledger, file_obj, separators=(",", ":"), sort_keys=True
                    )
                    file_obj.truncate()
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError, json.JSONDecodeError):
        # Physical storage and the manifest are authoritative.  A later D
        # reserve reconciles stale ledger entries if this advisory cleanup is
        # interrupted or observes a transient file error.
        return


def unpack_agentic_extra_key(extra_key: Any) -> Optional[tuple[str, dict[str, Any]]]:
    """Validate a router-safe envelope and return its stable radix key."""

    if not isinstance(extra_key, str) or not extra_key.startswith(
        EXTRA_KEY_ENVELOPE_PREFIX
    ):
        return None
    if len(extra_key) > 32768:
        raise ValueError("agentic extra_key envelope is too large")
    encoded = extra_key[len(EXTRA_KEY_ENVELOPE_PREFIX) :]
    try:
        stable_raw, params_raw = encoded.split(":", 1)
        stable_key = _decode_envelope_component(
            stable_raw, max_bytes=4096
        ).decode("utf-8")
        params = json.loads(
            _decode_envelope_component(params_raw, max_bytes=16384).decode("utf-8")
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid agentic extra_key envelope") from exc
    if not isinstance(params, dict) or not set(params).issubset(_AGENTIC_WIRE_KEYS):
        raise ValueError("agentic envelope contains unsupported fields")
    metadata = AgenticRequestMetadata.from_custom_params(params)
    expected_key = (
        f"agentic-v1:{metadata.request_id}:g{metadata.generation}"
        if metadata is not None
        else None
    )
    # Read legacy trajectory-only envelopes so an in-flight request submitted
    # by an older client fails soft during a rolling update.  New clients
    # always emit generation-scoped keys to isolate P cache cleanup races.
    legacy_key = (
        f"agentic-v1:{metadata.request_id}" if metadata is not None else None
    )
    if metadata is None or stable_key not in {expected_key, legacy_key}:
        raise ValueError("agentic envelope request id mismatch")
    return stable_key, params


class SnapshotState(str, Enum):
    """Logical ownership state for one request-generation KV snapshot."""

    D_GPU = "d_gpu"
    DIRECT_READY = "direct_ready"
    DIRECT_LOADING = "direct_loading"
    SLOW_FALLBACK = "slow_fallback"
    OFFLOADING = "offloading"
    MOONCAKE_READY = "mooncake_ready"
    P_LOADING = "p_loading"
    P_HOST = "p_host"
    P_GPU = "p_gpu"
    TO_DECODE = "to_decode"
    CONSUMED = "consumed"
    DELETE_PENDING = "delete_pending"
    EVICTED = "evicted"
    FINAL = "final"
    FAILED = "failed"


class AgenticOutputKind(str, Enum):
    """Application-independent classification of one finished model call."""

    TOOL = "tool"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"


_ALLOWED_TRANSITIONS: Mapping[SnapshotState, frozenset[SnapshotState]] = {
    SnapshotState.D_GPU: frozenset(
        {
            SnapshotState.DIRECT_READY,
            SnapshotState.OFFLOADING,
            SnapshotState.P_GPU,
            SnapshotState.FINAL,
            SnapshotState.FAILED,
        }
    ),
    SnapshotState.DIRECT_READY: frozenset(
        {
            SnapshotState.DIRECT_LOADING,
            SnapshotState.SLOW_FALLBACK,
            SnapshotState.FINAL,
            SnapshotState.FAILED,
        }
    ),
    SnapshotState.DIRECT_LOADING: frozenset(
        {
            SnapshotState.DIRECT_READY,
            SnapshotState.SLOW_FALLBACK,
            SnapshotState.CONSUMED,
            SnapshotState.FAILED,
        }
    ),
    SnapshotState.SLOW_FALLBACK: frozenset(
        {
            SnapshotState.OFFLOADING,
            SnapshotState.CONSUMED,
            SnapshotState.FAILED,
        }
    ),
    SnapshotState.OFFLOADING: frozenset(
        {SnapshotState.MOONCAKE_READY, SnapshotState.FAILED}
    ),
    SnapshotState.MOONCAKE_READY: frozenset(
        {
            SnapshotState.P_LOADING,
            SnapshotState.DELETE_PENDING,
            SnapshotState.EVICTED,
            SnapshotState.FAILED,
        }
    ),
    SnapshotState.P_LOADING: frozenset(
        {
            SnapshotState.P_HOST,
            SnapshotState.DELETE_PENDING,
            SnapshotState.FAILED,
        }
    ),
    SnapshotState.P_HOST: frozenset(
        {
            SnapshotState.P_GPU,
            SnapshotState.DELETE_PENDING,
            SnapshotState.FAILED,
        }
    ),
    SnapshotState.P_GPU: frozenset(
        {
            SnapshotState.TO_DECODE,
            SnapshotState.DELETE_PENDING,
            SnapshotState.FAILED,
        }
    ),
    SnapshotState.TO_DECODE: frozenset(
        {SnapshotState.CONSUMED, SnapshotState.FAILED}
    ),
    SnapshotState.DELETE_PENDING: frozenset(
        {SnapshotState.CONSUMED, SnapshotState.EVICTED, SnapshotState.FAILED}
    ),
    SnapshotState.CONSUMED: frozenset(),
    SnapshotState.EVICTED: frozenset(),
    SnapshotState.FINAL: frozenset(),
    SnapshotState.FAILED: frozenset(),
}


class SnapshotLifecycleError(RuntimeError):
    pass


class SnapshotNotReadyError(SnapshotLifecycleError):
    pass


class SnapshotDeletePendingError(SnapshotLifecycleError):
    pass


@dataclass(frozen=True, slots=True)
class RequestGeneration:
    request_id: str
    generation: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")

    @property
    def snapshot_id(self) -> str:
        return f"{self.request_id}:{self.generation}"

    @property
    def storage_id(self) -> str:
        request_digest = hashlib.sha256(self.request_id.encode()).hexdigest()[:32]
        return f"{request_digest}:{self.generation}"

    @property
    def manifest_key(self) -> str:
        return f"{MANIFEST_PREFIX}{self.storage_id}"

    @property
    def claim_key(self) -> str:
        return f"{self.manifest_key}:claim"


@dataclass(frozen=True, slots=True)
class AgenticRequestMetadata:
    """Minimal per-call metadata needed by the request-generation pipeline.

    The stable request id identifies one agent trajectory.  ``generation`` is
    the model-call index within that trajectory.  A later P call reads the D
    snapshot produced by ``parent_generation``.  Tool suffixes let D avoid
    publishing terminal answers without importing an application parser.
    """

    request_id: str
    generation: int
    parent_generation: Optional[int] = None
    tool_type: Optional[str] = None
    tool_suffix_token_ids: tuple[tuple[int, ...], ...] = ()
    terminal_marker_token_ids: tuple[tuple[int, ...], ...] = ()
    tool_suffix_strings: tuple[str, ...] = ()
    terminal_marker_strings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        RequestGeneration(self.request_id, self.generation)
        if self.parent_generation is not None:
            if self.parent_generation < 0:
                raise ValueError("parent_generation must be non-negative")
            if self.parent_generation >= self.generation:
                raise ValueError("parent_generation must precede generation")
        if any(not suffix for suffix in self.tool_suffix_token_ids):
            raise ValueError("tool suffixes must not be empty")
        if any(not marker for marker in self.terminal_marker_token_ids):
            raise ValueError("terminal markers must not be empty")
        if any(not suffix for suffix in self.tool_suffix_strings):
            raise ValueError("tool suffix strings must not be empty")
        if any(not marker for marker in self.terminal_marker_strings):
            raise ValueError("terminal marker strings must not be empty")

    @property
    def current(self) -> RequestGeneration:
        return RequestGeneration(self.request_id, self.generation)

    @property
    def parent(self) -> Optional[RequestGeneration]:
        if self.parent_generation is None:
            return None
        return RequestGeneration(self.request_id, self.parent_generation)

    def classify_output(
        self, output_ids: Sequence[int], tokenizer: Any = None
    ) -> AgenticOutputKind:
        """Classify a finished call without assuming a fixed BPE boundary.

        Text markers are canonical: they survive special-token aliases,
        context-dependent BPE merges, constrained decoding, and tokenizer
        changes across datasets.  Token ids remain a cheap compatibility path.

        A response matching neither marker class is ``UNKNOWN``.  This method
        only recognizes wire markers; the serving caller owns the policy.  In
        particular, the reverse-KV path treats UNKNOWN as terminal so ordinary
        answers and malformed/no-tool output never enter Host storage.
        """

        if tokenizer is not None and (
            self.tool_suffix_strings or self.terminal_marker_strings
        ):
            try:
                decoded = tokenizer.decode(
                    list(output_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                decoded = unicodedata.normalize("NFC", decoded)
                if any(
                    unicodedata.normalize("NFC", marker) in decoded
                    for marker in self.terminal_marker_strings
                ):
                    return AgenticOutputKind.TERMINAL
                stripped = decoded.rstrip()
                if any(
                    stripped.endswith(unicodedata.normalize("NFC", suffix))
                    for suffix in self.tool_suffix_strings
                ):
                    return AgenticOutputKind.TOOL
            except Exception:
                # A custom tokenizer should not disable lifecycle handling;
                # fall through to the legacy token-id representation.
                pass
        for marker in self.terminal_marker_token_ids:
            marker_len = len(marker)
            if any(
                tuple(output_ids[offset : offset + marker_len]) == marker
                for offset in range(len(output_ids) - marker_len + 1)
            ):
                return AgenticOutputKind.TERMINAL
        for suffix in self.tool_suffix_token_ids:
            if len(output_ids) >= len(suffix) and tuple(output_ids[-len(suffix) :]) == suffix:
                return AgenticOutputKind.TOOL
        return AgenticOutputKind.UNKNOWN

    def is_tool_output(self, output_ids: Sequence[int], tokenizer: Any = None) -> bool:
        """Compatibility predicate for callers that only need tool/non-tool."""

        return self.classify_output(output_ids, tokenizer) is AgenticOutputKind.TOOL

    @classmethod
    def from_custom_params(
        cls, custom_params: Optional[Mapping[str, Any]]
    ) -> Optional["AgenticRequestMetadata"]:
        if not isinstance(custom_params, Mapping):
            return None
        request_id = custom_params.get(CUSTOM_REQUEST_ID)
        generation = custom_params.get(CUSTOM_GENERATION)
        if request_id is None or generation is None:
            return None

        raw_suffixes = custom_params.get(CUSTOM_TOOL_SUFFIXES) or ()
        # Accept one suffix as [1, 2] and multiple suffixes as [[1, 2], [3]].
        if raw_suffixes and isinstance(raw_suffixes[0], int):
            raw_suffixes = (raw_suffixes,)
        suffixes = tuple(tuple(int(token) for token in suffix) for suffix in raw_suffixes)
        raw_terminal_markers = custom_params.get(CUSTOM_TERMINAL_MARKERS) or ()
        if raw_terminal_markers and isinstance(raw_terminal_markers[0], int):
            raw_terminal_markers = (raw_terminal_markers,)
        terminal_markers = tuple(
            tuple(int(token) for token in marker)
            for marker in raw_terminal_markers
        )
        raw_suffix_strings = custom_params.get(CUSTOM_TOOL_SUFFIX_STRINGS) or ()
        if isinstance(raw_suffix_strings, str):
            raw_suffix_strings = (raw_suffix_strings,)
        suffix_strings = tuple(str(marker) for marker in raw_suffix_strings)
        raw_terminal_strings = (
            custom_params.get(CUSTOM_TERMINAL_MARKER_STRINGS) or ()
        )
        if isinstance(raw_terminal_strings, str):
            raw_terminal_strings = (raw_terminal_strings,)
        terminal_strings = tuple(str(marker) for marker in raw_terminal_strings)
        parent = custom_params.get(CUSTOM_PARENT_GENERATION)
        return cls(
            request_id=str(request_id),
            generation=int(generation),
            parent_generation=None if parent is None else int(parent),
            tool_type=(
                None
                if custom_params.get(CUSTOM_TOOL_TYPE) is None
                else str(custom_params[CUSTOM_TOOL_TYPE])
            ),
            tool_suffix_token_ids=suffixes,
            terminal_marker_token_ids=terminal_markers,
            tool_suffix_strings=suffix_strings,
            terminal_marker_strings=terminal_strings,
        )

    @classmethod
    def from_req(cls, req: Any) -> Optional["AgenticRequestMetadata"]:
        sampling_params = getattr(req, "sampling_params", None)
        return cls.from_custom_params(getattr(sampling_params, "custom_params", None))


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Small logical manifest; KV bytes remain physically page-based."""

    request: RequestGeneration
    page_keys: tuple[str, ...]
    token_count: int
    byte_size: int
    state: SnapshotState
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tool_type: Optional[str] = None
    tool_started_at: Optional[float] = None
    claim_id: Optional[str] = None
    deletion_target: Optional[SnapshotState] = None
    terminal_at: Optional[float] = None
    failure_reason: Optional[str] = None
    token_digest: Optional[str] = None
    direct_bootstrap_addr: Optional[str] = None
    direct_room: Optional[int] = None
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.version != MANIFEST_VERSION:
            raise ValueError(
                f"unsupported manifest version {self.version}; expected {MANIFEST_VERSION}"
            )
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if self.byte_size < 0:
            raise ValueError("byte_size must be non-negative")
        if not self.page_keys and self.state not in {
            SnapshotState.FAILED,
            SnapshotState.FINAL,
            SnapshotState.DIRECT_READY,
            SnapshotState.DIRECT_LOADING,
            SnapshotState.SLOW_FALLBACK,
            SnapshotState.CONSUMED,
        }:
            raise ValueError("page_keys must contain the complete snapshot")
        if any(not key for key in self.page_keys):
            raise ValueError("page_keys must not contain empty keys")
        if len(set(self.page_keys)) != len(self.page_keys):
            raise ValueError("page_keys must be unique physical objects")
        if self.tool_started_at is not None and not math.isfinite(self.tool_started_at):
            raise ValueError("tool_started_at must be finite")
        if self.deletion_target not in {
            None,
            SnapshotState.CONSUMED,
            SnapshotState.EVICTED,
        }:
            raise ValueError("deletion_target must be CONSUMED or EVICTED")
        if self.state is SnapshotState.DELETE_PENDING and self.deletion_target is None:
            raise ValueError("DELETE_PENDING requires deletion_target")
        if self.state in {SnapshotState.DIRECT_READY, SnapshotState.DIRECT_LOADING}:
            if not self.direct_bootstrap_addr or self.direct_room is None:
                raise ValueError("direct snapshot requires bootstrap address and room")
            if not self.token_digest:
                raise ValueError("direct snapshot requires token digest")

    @property
    def snapshot_id(self) -> str:
        return self.request.snapshot_id

    @property
    def manifest_key(self) -> str:
        return self.request.manifest_key

    def transition(self, target: SnapshotState, now: Optional[float] = None):
        if target == self.state:
            return self
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise SnapshotLifecycleError(
                f"invalid snapshot transition {self.state.value} -> {target.value} "
                f"for {self.snapshot_id}"
            )
        return replace(self, state=target, updated_at=time.time() if now is None else now)

    def eviction_cost(
        self,
        now: float,
        expected_tool_seconds: Optional[float],
    ) -> float:
        """Cost requested by the design: KV_Size * expected remaining tool time."""

        if self.state is not SnapshotState.MOONCAKE_READY:
            return float("-inf")
        if expected_tool_seconds is None:
            expected_tool_seconds = 0.0
        elapsed = 0.0
        if self.tool_started_at is not None:
            elapsed = max(0.0, now - self.tool_started_at)
        remaining = max(float(expected_tool_seconds) - elapsed, 0.0)
        return float(self.byte_size) * remaining

    def to_bytes(self) -> bytes:
        body = {
            "version": self.version,
            "request_id": self.request.request_id,
            "generation": self.request.generation,
            "page_keys": self.page_keys,
            "token_count": self.token_count,
            "byte_size": self.byte_size,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_type": self.tool_type,
            "tool_started_at": self.tool_started_at,
            "claim_id": self.claim_id,
            "deletion_target": (
                None if self.deletion_target is None else self.deletion_target.value
            ),
            "terminal_at": self.terminal_at,
            "failure_reason": self.failure_reason,
            "token_digest": self.token_digest,
            "direct_bootstrap_addr": self.direct_bootstrap_addr,
            "direct_room": self.direct_room,
        }
        raw = json.dumps(body, separators=(",", ":"), ensure_ascii=True).encode()
        return zlib.compress(raw, level=1)

    @classmethod
    def from_bytes(cls, value: bytes) -> "SnapshotManifest":
        body = json.loads(zlib.decompress(value))
        return cls(
            request=RequestGeneration(body["request_id"], int(body["generation"])),
            page_keys=tuple(body["page_keys"]),
            token_count=int(body["token_count"]),
            byte_size=int(body["byte_size"]),
            state=SnapshotState(body["state"]),
            created_at=float(body["created_at"]),
            updated_at=float(body["updated_at"]),
            tool_type=body.get("tool_type"),
            tool_started_at=body.get("tool_started_at"),
            claim_id=body.get("claim_id"),
            deletion_target=(
                None
                if body.get("deletion_target") is None
                else SnapshotState(body["deletion_target"])
            ),
            terminal_at=body.get("terminal_at"),
            failure_reason=body.get("failure_reason"),
            token_digest=body.get("token_digest"),
            direct_bootstrap_addr=body.get("direct_bootstrap_addr"),
            direct_room=body.get("direct_room"),
            version=int(body["version"]),
        )


class MooncakeRawStore(Protocol):
    def put(self, key: str, value: bytes) -> int: ...

    def upsert(self, key: str, value: bytes) -> int: ...

    def get(self, key: str) -> bytes: ...

    def is_exist(self, key: str) -> int: ...

    def batch_is_exist(self, keys: list[str]) -> list[int]: ...

    def batch_remove(self, keys: list[str], force: bool = False) -> list[int]: ...

    def remove(self, key: str, force: bool = False) -> int: ...


@dataclass(frozen=True, slots=True)
class SnapshotDeleteResult:
    snapshot_id: str
    removed: bool
    remaining_keys: tuple[str, ...] = ()
    remove_codes: tuple[int, ...] = ()


class MooncakeSnapshotStore:
    """Logical request-level operations over Mooncake's page-level objects.

    A manifest is the visibility/commit marker.  Deletion first changes it to
    DELETE_PENDING, so a partial physical batch removal is never exposed as a
    valid snapshot.  The manifest itself is removed last.
    """

    def __init__(self, store: MooncakeRawStore):
        self.store = store

    @staticmethod
    def _local_claim_path(request: RequestGeneration) -> Optional[str]:
        """Return the node-local fence shared by P and every D worker.

        Mooncake's create-if-absent claim protects the storage object, but a
        claim-key PUT and a manifest UPSERT are two independent transactions.
        In addition, an ambiguous UPSERT can still be completing in the master
        after the client has returned ``ILLEGAL_CLIENT``.  The V1 direct path
        is node-local already, so an O_EXCL file in the run's P-ready tmpfs is
        the authoritative cross-process owner fence for Direct versus slow
        fallback.  It contains only a short owner id; no KV data is copied.
        """

        directory = os.getenv("SGLANG_PD_P_READY_DIR", "")
        if not directory:
            return None
        digest = hashlib.sha256(request.snapshot_id.encode("utf-8")).hexdigest()
        return os.path.join(directory, f"lifecycle-claim-{digest}")

    @classmethod
    def _acquire_local_claim(
        cls, request: RequestGeneration, claim_id: str
    ) -> bool:
        path = cls._local_claim_path(request)
        if path is None:
            # Lightweight tests and non-node-local storage configurations keep
            # the original Mooncake-only behavior.
            return True
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        try:
            os.write(fd, claim_id.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return True

    @classmethod
    def _release_local_claim(
        cls, request: RequestGeneration, claim_id: Optional[str]
    ) -> None:
        path = cls._local_claim_path(request)
        if path is None:
            return
        try:
            with open(path, "rb") as file_obj:
                owner = file_obj.read(1024).decode("utf-8")
        except FileNotFoundError:
            return
        except (OSError, UnicodeDecodeError):
            return
        if claim_id is not None and owner != claim_id:
            return
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @classmethod
    def _local_claim_owner(cls, request: RequestGeneration) -> Optional[str]:
        path = cls._local_claim_path(request)
        if path is None:
            return None
        try:
            with open(path, "rb") as file_obj:
                return file_obj.read(1024).decode("utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError):
            return None

    @classmethod
    def _require_local_claim_owner(
        cls, request: RequestGeneration, claim_id: str
    ) -> None:
        """Fence every claimed transition by the caller's actual owner id."""

        path = cls._local_claim_path(request)
        if path is None:
            return
        owner = cls._local_claim_owner(request)
        if owner != claim_id:
            raise SnapshotNotReadyError(
                f"snapshot {request.snapshot_id} claim is owned by {owner!r}, "
                f"not {claim_id!r}"
            )

    def _update_claimed_transition(
        self,
        manifest: SnapshotManifest,
        *,
        expected_states: Iterable[SnapshotState],
        owner_claim_id: Optional[str] = None,
        max_attempts: int = 6,
    ) -> None:
        """Retry an ambiguous Mooncake upsert while holding the claim object.

        Mooncake can transiently return ``ILLEGAL_CLIENT`` (-601) from
        ``UpsertEnd`` when several clients finish small metadata operations at
        once.  The return is ambiguous: the new bytes may already be visible,
        or the old state may still be present.  Blindly retrying can overwrite
        a transition made by another owner, so this helper is deliberately
        restricted to call sites that already own the per-generation claim.

        After -601, read the manifest back.  An observed target state makes the
        operation idempotently successful; an unexpected state means ownership
        was lost and must never be overwritten.  Only a missing/read-stale or
        still-expected predecessor is retried.
        """

        expected = frozenset(expected_states)
        if not expected:
            raise ValueError("expected_states must not be empty")
        owner_claim_id = owner_claim_id or manifest.claim_id
        if not owner_claim_id:
            raise SnapshotLifecycleError(
                f"claimed transition for {manifest.snapshot_id} has no owner"
            )
        self._require_local_claim_owner(manifest.request, owner_claim_id)
        retry_delay = 0.01
        last_code = 0
        for attempt in range(max_attempts):
            last_code = self.store.upsert(
                manifest.manifest_key, manifest.to_bytes()
            )
            if last_code == 0:
                return
            if last_code != -601:
                break

            # Do not immediately launch another UPSERT.  Mooncake may still
            # be completing the previous PutEnd after returning -601.  Polling
            # its result under the node-local owner fence avoids self-overlap
            # and prevents a subsequent owner from racing an unfinished PUT.
            settle_deadline = time.monotonic() + 0.25
            while time.monotonic() < settle_deadline:
                time.sleep(retry_delay)
                observed = self.load(manifest.request, require_ready=False)
                if (
                    observed is not None
                    and observed.state is manifest.state
                    and observed.claim_id == manifest.claim_id
                ):
                    return
                if observed is not None and observed.state not in expected:
                    raise SnapshotNotReadyError(
                        f"snapshot {manifest.snapshot_id} advanced to "
                        f"{observed.state.value} while updating "
                        f"{manifest.state.value}"
                    )
            if attempt + 1 < max_attempts:
                retry_delay = min(retry_delay * 2.0, 0.08)

        raise SnapshotLifecycleError(
            f"failed to update claimed manifest {manifest.snapshot_id}: "
            f"code={last_code}"
        )

    def begin_publish(self, manifest: SnapshotManifest) -> None:
        """Publish an invisible OFFLOADING record before writing KV pages.

        Recording the complete intended page list first makes a producer crash
        recoverable: a GC can find and remove pages belonging to an abandoned
        OFFLOADING generation instead of leaving unreachable objects forever.
        """

        if manifest.state is not SnapshotState.OFFLOADING:
            raise SnapshotLifecycleError("begin_publish requires OFFLOADING state")
        ret = self.store.put(manifest.manifest_key, manifest.to_bytes())
        if ret != 0:
            raise SnapshotLifecycleError(
                f"failed to begin manifest {manifest.snapshot_id}: code={ret}"
            )

    def publish_direct_offer(self, manifest: SnapshotManifest) -> None:
        if manifest.state is not SnapshotState.DIRECT_READY:
            raise SnapshotLifecycleError("direct offer requires DIRECT_READY state")
        ret = self.store.put(manifest.manifest_key, manifest.to_bytes())
        if ret != 0:
            raise SnapshotLifecycleError(
                f"failed to publish direct offer {manifest.snapshot_id}: code={ret}"
            )

    def claim_direct(
        self, request: RequestGeneration, claim_id: str
    ) -> SnapshotManifest:
        if not claim_id:
            raise ValueError("claim_id must be non-empty")
        if not self._acquire_local_claim(request, claim_id):
            raise SnapshotNotReadyError(
                f"direct snapshot {request.snapshot_id} is locally claimed"
            )
        claim_code = self.store.put(request.claim_key, f"direct:{claim_id}".encode())
        if claim_code != 0:
            self._release_local_claim(request, claim_id)
            raise SnapshotNotReadyError(
                f"direct snapshot {request.snapshot_id} is already claimed"
            )
        try:
            manifest = self.load(request, require_ready=False)
            if manifest is None or manifest.state is not SnapshotState.DIRECT_READY:
                state = "missing" if manifest is None else manifest.state.value
                raise SnapshotNotReadyError(
                    f"direct snapshot {request.snapshot_id} is {state}"
                )
            claimed = replace(manifest, claim_id=claim_id).transition(
                SnapshotState.DIRECT_LOADING
            )
            self._update_claimed_transition(
                claimed, expected_states=(SnapshotState.DIRECT_READY,)
            )
            return claimed
        except Exception:
            self.store.remove(request.claim_key, force=False)
            self._release_local_claim(request, claim_id)
            raise

    def complete_direct(
        self, manifest: SnapshotManifest, claim_id: str
    ) -> SnapshotManifest:
        if (
            manifest.state is not SnapshotState.DIRECT_LOADING
            or manifest.claim_id != claim_id
        ):
            raise SnapshotLifecycleError(
                f"invalid direct completion for {manifest.snapshot_id}"
            )
        terminal = replace(
            manifest.transition(SnapshotState.CONSUMED),
            terminal_at=time.time(),
        )
        self._update_claimed_transition(
            terminal,
            expected_states=(SnapshotState.DIRECT_LOADING,),
            owner_claim_id=claim_id,
        )
        self.store.remove(terminal.request.claim_key, force=False)
        self._release_local_claim(terminal.request, claim_id)
        _discard_shared_ledger_snapshot(terminal.snapshot_id)
        return terminal

    def release_direct_claim(
        self, manifest: SnapshotManifest, claim_id: str
    ) -> SnapshotManifest:
        if (
            manifest.state is not SnapshotState.DIRECT_LOADING
            or manifest.claim_id != claim_id
        ):
            raise SnapshotLifecycleError(
                f"invalid direct claim release for {manifest.snapshot_id}"
            )
        ready = replace(manifest, claim_id=None).transition(
            SnapshotState.DIRECT_READY
        )
        self._update_claimed_transition(
            ready,
            expected_states=(SnapshotState.DIRECT_LOADING,),
            owner_claim_id=claim_id,
        )
        self.store.remove(manifest.request.claim_key, force=False)
        self._release_local_claim(manifest.request, claim_id)
        return ready

    def begin_slow_fallback(
        self, manifest: SnapshotManifest, owner_id: str = "decode"
    ) -> Optional[SnapshotManifest]:
        """Transfer ownership from Direct to the complete slow-path pipeline.

        The direct claim object is the per-generation mutex.  At high
        concurrency P may be claiming DIRECT_READY at the exact instant D's
        fast-tool timer expires.  Removing P's claim and concurrently
        upserting the manifest caused Mooncake ``ILLEGAL_CLIENT`` errors and
        allowed D to overwrite DIRECT_LOADING.

        The fallback claim deliberately remains live after this method returns.
        Mooncake metadata operations on the claim key and manifest key do not
        form a cross-key transaction: releasing the claim immediately after the
        manifest upsert allowed a late P to acquire the claim, read a stale
        DIRECT_READY manifest, and overwrite SLOW_FALLBACK.  The slow-path
        owner therefore keeps the claim until either the complete Shared-Host
        snapshot reaches P GPU, or a Mooncake publish reaches MOONCAKE_READY.
        ``None`` means P owns the direct claim and D must keep the candidate
        alive and poll again.
        """

        claim_id = f"fallback:{owner_id}"
        if not self._acquire_local_claim(manifest.request, claim_id):
            return None
        if self.store.put(manifest.request.claim_key, claim_id.encode()) != 0:
            self._release_local_claim(manifest.request, claim_id)
            return None
        keep_claim = False
        try:
            # Mooncake GET may briefly return empty immediately after the
            # successful offer PUT.  Once D owns the exclusive fallback claim,
            # its locally retained offer is a safe fallback version: no P can
            # begin or complete a direct transition until this claim is
            # released.  Prefer a visible newer value when one is available.
            current = self.load(manifest.request, require_ready=False) or manifest
            # DIRECT_LOADING always belongs to a P receiver.  Even if an
            # earlier ambiguous metadata call temporarily made its Mooncake
            # claim invisible, D must not overwrite that in-flight state.
            # P's release settles it back to DIRECT_READY; D retries then.
            if current.state is SnapshotState.DIRECT_LOADING:
                return None
            if current.state is not SnapshotState.DIRECT_READY:
                raise SnapshotLifecycleError(
                    f"cannot fall back to slow Put from {current.state.value}"
                )
            fallback = replace(current, claim_id=claim_id).transition(
                SnapshotState.SLOW_FALLBACK
            )
            self._update_claimed_transition(
                fallback,
                expected_states=(SnapshotState.DIRECT_READY,),
                owner_claim_id=claim_id,
            )
            keep_claim = True
            return fallback
        finally:
            if not keep_claim:
                self.store.remove(manifest.request.claim_key, force=False)
                self._release_local_claim(manifest.request, claim_id)

    @staticmethod
    def _is_fallback_claim(claim_id: Optional[str]) -> bool:
        return bool(claim_id and claim_id.startswith("fallback:"))

    def complete_slow_fallback(
        self, manifest: SnapshotManifest
    ) -> SnapshotManifest:
        """Acknowledge complete Shared-Host→P-GPU recovery.

        The P GPU copy is authoritative before this transition.  CONSUMED is
        the same terminal ownership state used by a successful Direct receive;
        only after publishing it may the persistent fallback claim be removed.
        """

        if (
            manifest.state is not SnapshotState.SLOW_FALLBACK
            or not self._is_fallback_claim(manifest.claim_id)
        ):
            raise SnapshotLifecycleError(
                f"invalid slow fallback completion for {manifest.snapshot_id}"
            )
        terminal = replace(
            manifest.transition(SnapshotState.CONSUMED),
            claim_id=None,
            terminal_at=time.time(),
        )
        self._update_claimed_transition(
            terminal,
            expected_states=(SnapshotState.SLOW_FALLBACK,),
            owner_claim_id=manifest.claim_id,
        )
        self.store.remove(manifest.request.claim_key, force=False)
        self._release_local_claim(manifest.request, manifest.claim_id)
        return terminal

    def publish_failure(
        self,
        request: RequestGeneration,
        *,
        reason: str,
        tool_type: Optional[str] = None,
    ) -> SnapshotManifest:
        marker = SnapshotManifest(
            request=request,
            page_keys=(),
            token_count=0,
            byte_size=0,
            state=SnapshotState.FAILED,
            tool_type=tool_type,
            failure_reason=reason[:256],
        )
        code = self.store.put(marker.manifest_key, marker.to_bytes())
        if code != 0:
            existing = self.load(request, require_ready=False)
            if existing is not None:
                return existing
            raise SnapshotLifecycleError(
                f"failed to publish failure marker {marker.snapshot_id}: code={code}"
            )
        return marker

    def mark_failed(
        self,
        manifest: SnapshotManifest,
        *,
        reason: str,
        owner_claim_id: Optional[str] = None,
    ) -> SnapshotManifest:
        old_claim_id = manifest.claim_id
        failed = replace(
            manifest, failure_reason=reason[:256], claim_id=None
        ).transition(SnapshotState.FAILED)
        if old_claim_id:
            if owner_claim_id != old_claim_id:
                raise SnapshotNotReadyError(
                    f"cannot fail claimed snapshot {manifest.snapshot_id}: "
                    "caller does not own its lifecycle claim"
                )
            self._update_claimed_transition(
                failed,
                expected_states=(manifest.state,),
                owner_claim_id=old_claim_id,
            )
        else:
            self.update(failed)
        self.store.remove(manifest.request.claim_key, force=False)
        self._release_local_claim(manifest.request, old_claim_id)
        return failed

    def _terminalize_direct_offer(
        self,
        manifest: SnapshotManifest,
        *,
        state: SnapshotState,
        owner_id: str,
        reason: Optional[str] = None,
    ) -> Optional[SnapshotManifest]:
        """Atomically retire an unclaimed Direct offer.

        D may learn from the application parser that a provisional tool-looking
        result is terminal or invalid at the same time P tries to claim it.
        The same local+Mooncake fence used by Direct/fallback arbitration must
        decide that race.  A ``None`` result means P won; D must leave the
        snapshot and its KV untouched until P completes or releases its claim.
        """

        if state not in {SnapshotState.FINAL, SnapshotState.FAILED}:
            raise ValueError("direct offer may only be retired as FINAL or FAILED")
        claim_id = f"terminal:{owner_id}"
        if not self._acquire_local_claim(manifest.request, claim_id):
            return None
        if self.store.put(manifest.request.claim_key, claim_id.encode()) != 0:
            self._release_local_claim(manifest.request, claim_id)
            return None
        try:
            current = self.load(manifest.request, require_ready=False) or manifest
            if current.state is not SnapshotState.DIRECT_READY:
                return None
            terminal = replace(
                current.transition(state),
                claim_id=None,
                failure_reason=(reason[:256] if reason else current.failure_reason),
                terminal_at=time.time(),
            )
            self._update_claimed_transition(
                terminal,
                expected_states=(SnapshotState.DIRECT_READY,),
                owner_claim_id=claim_id,
            )
            return terminal
        finally:
            self.store.remove(manifest.request.claim_key, force=False)
            self._release_local_claim(manifest.request, claim_id)

    def finalize_direct_offer(
        self, manifest: SnapshotManifest, *, owner_id: str
    ) -> Optional[SnapshotManifest]:
        return self._terminalize_direct_offer(
            manifest, state=SnapshotState.FINAL, owner_id=owner_id
        )

    def fail_direct_offer(
        self, manifest: SnapshotManifest, *, owner_id: str, reason: str
    ) -> Optional[SnapshotManifest]:
        return self._terminalize_direct_offer(
            manifest,
            state=SnapshotState.FAILED,
            owner_id=owner_id,
            reason=reason,
        )

    def continue_slow_publish(self, manifest: SnapshotManifest) -> None:
        """Move an owned fallback snapshot from SLOW_FALLBACK to OFFLOADING."""

        if (
            manifest.state is not SnapshotState.OFFLOADING
            or not self._is_fallback_claim(manifest.claim_id)
        ):
            raise SnapshotLifecycleError(
                "continue_slow_publish requires an owned OFFLOADING manifest"
            )
        self._update_claimed_transition(
            manifest,
            expected_states=(SnapshotState.SLOW_FALLBACK,),
            owner_claim_id=manifest.claim_id,
        )

    def rollback_slow_publish(
        self,
        offloading: SnapshotManifest,
        fallback: SnapshotManifest,
    ) -> None:
        """Restore an owned spill placeholder after an incomplete page Put."""

        if offloading.request != fallback.request:
            raise SnapshotLifecycleError("spill rollback request mismatch")
        if (
            offloading.state is not SnapshotState.OFFLOADING
            or fallback.state is not SnapshotState.SLOW_FALLBACK
            or not self._is_fallback_claim(fallback.claim_id)
            or offloading.claim_id != fallback.claim_id
        ):
            raise SnapshotLifecycleError("invalid owned spill rollback")
        self._update_claimed_transition(
            fallback,
            expected_states=(SnapshotState.OFFLOADING,),
            owner_claim_id=fallback.claim_id,
        )

    def commit_publish(self, request: RequestGeneration) -> SnapshotManifest:
        """Make a snapshot visible only after every physical page exists."""

        manifest = self.load(request, require_ready=False)
        if manifest is None:
            raise SnapshotLifecycleError(f"missing OFFLOADING manifest {request.snapshot_id}")
        if manifest.state is not SnapshotState.OFFLOADING:
            raise SnapshotLifecycleError(
                f"commit_publish requires OFFLOADING, got {manifest.state.value}"
            )
        exists = self.store.batch_is_exist(list(manifest.page_keys))
        missing = [key for key, result in zip(manifest.page_keys, exists) if result != 1]
        if missing:
            raise SnapshotLifecycleError(
                f"cannot publish incomplete snapshot {manifest.snapshot_id}; "
                f"{len(missing)} physical objects are missing"
            )
        fallback_claim = self._is_fallback_claim(manifest.claim_id)
        ready = manifest.transition(SnapshotState.MOONCAKE_READY)
        if fallback_claim:
            ready = replace(ready, claim_id=None)
            self._update_claimed_transition(
                ready,
                expected_states=(SnapshotState.OFFLOADING,),
                owner_claim_id=manifest.claim_id,
            )
            self.store.remove(manifest.request.claim_key, force=False)
            self._release_local_claim(manifest.request, manifest.claim_id)
        else:
            self.update(ready)
        return ready

    def fail_publish(self, manifest: SnapshotManifest) -> SnapshotDeleteResult:
        """Hide an incomplete publish and remove every page that did arrive."""

        fallback_claim = self._is_fallback_claim(manifest.claim_id)
        if manifest.state is SnapshotState.OFFLOADING:
            failed = replace(
                manifest.transition(SnapshotState.FAILED), claim_id=None
            )
            if fallback_claim:
                self._update_claimed_transition(
                    failed,
                    expected_states=(SnapshotState.OFFLOADING,),
                    owner_claim_id=manifest.claim_id,
                )
                self.store.remove(manifest.request.claim_key, force=False)
                self._release_local_claim(manifest.request, manifest.claim_id)
            else:
                self.update(failed)
        elif manifest.state is SnapshotState.FAILED:
            failed = manifest
        else:
            raise SnapshotLifecycleError(
                f"cannot fail publish from {manifest.state.value}"
            )
        codes = tuple(self.store.batch_remove(list(failed.page_keys), force=False))
        remaining = tuple(
            key
            for key, exists in zip(
                failed.page_keys,
                self.store.batch_is_exist(list(failed.page_keys)),
            )
            if exists == 1
        )
        return SnapshotDeleteResult(
            snapshot_id=failed.snapshot_id,
            removed=not remaining,
            remaining_keys=remaining,
            remove_codes=codes,
        )

    def load(self, request: RequestGeneration, require_ready: bool = True):
        value = self.store.get(request.manifest_key)
        if not value:
            return None
        manifest = SnapshotManifest.from_bytes(value)
        if require_ready and manifest.state is not SnapshotState.MOONCAKE_READY:
            raise SnapshotNotReadyError(
                f"snapshot {manifest.snapshot_id} is {manifest.state.value}"
            )
        return manifest

    def update(self, manifest: SnapshotManifest) -> None:
        ret = self.store.upsert(manifest.manifest_key, manifest.to_bytes())
        if ret != 0:
            raise SnapshotLifecycleError(
                f"failed to update manifest {manifest.snapshot_id}: code={ret}"
            )

    def claim_for_load(
        self, request: RequestGeneration, claim_id: str
    ) -> SnapshotManifest:
        """Claim the single-consumer V1 snapshot before issuing batch_get.

        Mooncake has no metadata CAS, but ``put`` is create-if-absent.  A tiny
        per-generation claim object therefore serializes P-load and eviction.
        The storage lease additionally protects pages once GET starts.
        """

        if not claim_id:
            raise ValueError("claim_id must be non-empty")
        claim_value = f"load:{claim_id}".encode()
        claim_code = self.store.put(request.claim_key, claim_value)
        if claim_code != 0:
            raise SnapshotNotReadyError(
                f"snapshot {request.snapshot_id} is already claimed"
            )
        try:
            manifest = self.load(request, require_ready=True)
            claimed = replace(manifest, claim_id=claim_id).transition(
                SnapshotState.P_LOADING
            )
            self.update(claimed)
        except Exception:
            self.store.remove(request.claim_key, force=False)
            raise
        observed = self.load(request, require_ready=False)
        if (
            observed is None
            or observed.state is not SnapshotState.P_LOADING
            or observed.claim_id != claim_id
        ):
            self.store.remove(request.claim_key, force=False)
            raise SnapshotLifecycleError(
                f"lost load claim for {request.snapshot_id}: {claim_id}"
            )
        return observed

    def mark_p_host(self, manifest: SnapshotManifest, claim_id: str) -> SnapshotManifest:
        if manifest.state is not SnapshotState.P_LOADING or manifest.claim_id != claim_id:
            raise SnapshotLifecycleError(f"invalid P host ACK for {manifest.snapshot_id}")
        updated = manifest.transition(SnapshotState.P_HOST)
        self.update(updated)
        return updated

    def abandon_load(
        self, manifest: SnapshotManifest, claim_id: str
    ) -> SnapshotDeleteResult:
        """Delete a claimed snapshot after a partial GET or request abort.

        A request can be aborted after the complete L3->Host read has been
        acknowledged but before Host->GPU finishes, so both P_LOADING and
        P_HOST are valid abandonment points.  The claim is retained until all
        physical pages are gone, preventing a second P consumer from racing
        the cleanup.
        """

        if manifest.state not in {
            SnapshotState.P_LOADING,
            SnapshotState.P_HOST,
        } or manifest.claim_id != claim_id:
            raise SnapshotLifecycleError(
                f"invalid P load abandonment for {manifest.snapshot_id}"
            )
        tombstone = replace(
            manifest, deletion_target=SnapshotState.EVICTED
        ).transition(SnapshotState.DELETE_PENDING)
        self.update(tombstone)
        return self.delete_snapshot(
            tombstone, final_state=SnapshotState.EVICTED
        )

    def mark_p_gpu(self, manifest: SnapshotManifest, claim_id: str) -> SnapshotManifest:
        if manifest.state is not SnapshotState.P_HOST or manifest.claim_id != claim_id:
            raise SnapshotLifecycleError(f"invalid P GPU ACK for {manifest.snapshot_id}")
        updated = manifest.transition(SnapshotState.P_GPU)
        self.update(updated)
        return updated

    def delete_snapshot(
        self,
        manifest: SnapshotManifest,
        *,
        final_state: SnapshotState,
        update_shared_ledger: bool = True,
    ) -> SnapshotDeleteResult:
        if final_state not in {SnapshotState.CONSUMED, SnapshotState.EVICTED}:
            raise ValueError("final_state must be CONSUMED or EVICTED")
        if manifest.state is not SnapshotState.DELETE_PENDING:
            required = (
                SnapshotState.P_GPU
                if final_state is SnapshotState.CONSUMED
                else SnapshotState.MOONCAKE_READY
            )
            if manifest.state is not required:
                raise SnapshotLifecycleError(
                    f"cannot delete {manifest.snapshot_id} as {final_state.value} "
                    f"from {manifest.state.value}; expected {required.value}"
                )

            if final_state is SnapshotState.EVICTED:
                claim_code = self.store.put(
                    manifest.request.claim_key, b"evict"
                )
                if claim_code != 0:
                    raise SnapshotNotReadyError(
                        f"snapshot {manifest.snapshot_id} was claimed before eviction"
                    )

        if (
            manifest.state is SnapshotState.DELETE_PENDING
            and manifest.deletion_target is not final_state
        ):
            raise SnapshotLifecycleError(
                f"delete retry target mismatch for {manifest.snapshot_id}: "
                f"stored={manifest.deletion_target}, requested={final_state.value}"
            )
        tombstone = (
            manifest
            if manifest.state is SnapshotState.DELETE_PENDING
            else replace(manifest, deletion_target=final_state).transition(
                SnapshotState.DELETE_PENDING
            )
        )
        self.update(tombstone)

        # P_GPU is acknowledged only after the complete L3->Host->GPU load has
        # finished.  At that point this single-consumer snapshot has no valid
        # reader, while Mooncake's batch_get_into can retain soft leases past
        # completion on the TCP path.  Force-removing CONSUMED pages is thus
        # both safe and necessary to implement load-then-delete.  Eviction and
        # recovery remain lease-safe because they may race a live GET.
        force_remove = final_state is SnapshotState.CONSUMED
        codes = tuple(
            self.store.batch_remove(
                list(tombstone.page_keys), force=force_remove
            )
        )
        remaining = tuple(
            key
            for key, exists in zip(
                tombstone.page_keys,
                self.store.batch_is_exist(list(tombstone.page_keys)),
            )
            if exists == 1
        )
        if remaining:
            return SnapshotDeleteResult(
                snapshot_id=tombstone.snapshot_id,
                removed=False,
                remaining_keys=remaining,
                remove_codes=codes,
            )

        terminal = replace(
            tombstone.transition(final_state),
            terminal_at=time.time(),
        )
        self.update(terminal)
        self.store.remove(terminal.request.claim_key, force=False)
        if update_shared_ledger:
            _discard_shared_ledger_snapshot(terminal.snapshot_id)
        # Keep the tiny terminal manifest as an idempotency tombstone.  It is
        # removed later by gc_terminal_manifest after the retry/ABA window.
        return SnapshotDeleteResult(
            snapshot_id=terminal.snapshot_id,
            removed=True,
            remove_codes=codes,
        )

    def recover_stale(self, manifest: SnapshotManifest) -> SnapshotDeleteResult:
        """Best-effort recovery for an abandoned request-generation state.

        The caller decides staleness from ``updated_at``.  This method never
        force-removes pages: an active Mooncake GET therefore leaves a
        DELETE_PENDING tombstone that the normal retry loop can finish.
        """

        if manifest.state is SnapshotState.OFFLOADING:
            return self.fail_publish(
                replace(
                    manifest, failure_reason="stale_offloading_recovered"
                )
            )
        if manifest.state in {SnapshotState.P_LOADING, SnapshotState.P_HOST}:
            if not manifest.claim_id:
                raise SnapshotLifecycleError(
                    f"stale claimed snapshot has no claim id: {manifest.snapshot_id}"
                )
            return self.abandon_load(manifest, manifest.claim_id)
        if manifest.state is SnapshotState.P_GPU:
            return self.delete_snapshot(
                manifest, final_state=SnapshotState.CONSUMED
            )
        if manifest.state is SnapshotState.DELETE_PENDING:
            if manifest.deletion_target is None:
                raise SnapshotLifecycleError(
                    f"stale delete has no target: {manifest.snapshot_id}"
                )
            return self.delete_snapshot(
                manifest, final_state=manifest.deletion_target
            )
        raise SnapshotLifecycleError(
            f"state {manifest.state.value} has no stale recovery action"
        )

    def gc_terminal_manifest(
        self,
        manifest: SnapshotManifest,
        *,
        retention_seconds: float,
        now: Optional[float] = None,
    ) -> bool:
        if manifest.state not in {SnapshotState.CONSUMED, SnapshotState.EVICTED}:
            return False
        if manifest.terminal_at is None:
            return False
        now = time.time() if now is None else now
        if now - manifest.terminal_at < max(0.0, retention_seconds):
            return False
        # A process can die after publishing a terminal state but before
        # removing its ownership claim.  Terminal GC cleans both tiny metadata
        # objects so a crash cannot leak per-generation fences indefinitely.
        self.store.remove(manifest.request.claim_key, force=False)
        code = self.store.remove(manifest.manifest_key, force=False)
        return code == 0 or self.store.is_exist(manifest.manifest_key) != 1


class SnapshotIndex:
    """Thread-safe local index used for complete-snapshot eviction selection."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._manifests: dict[str, SnapshotManifest] = {}

    def upsert(self, manifest: SnapshotManifest) -> None:
        with self._lock:
            self._manifests[manifest.snapshot_id] = manifest

    def discard(self, snapshot_id: str) -> None:
        with self._lock:
            self._manifests.pop(snapshot_id, None)

    @property
    def byte_size(self) -> int:
        with self._lock:
            return sum(
                manifest.byte_size
                for manifest in self._manifests.values()
                if manifest.state is SnapshotState.MOONCAKE_READY
            )

    @property
    def resident_byte_size(self) -> int:
        """Physical bytes still resident, including claimed/deleting snapshots."""

        with self._lock:
            return sum(
                manifest.byte_size
                for manifest in self._manifests.values()
                if manifest.state
                not in {
                    SnapshotState.CONSUMED,
                    SnapshotState.EVICTED,
                    SnapshotState.FINAL,
                }
            )

    def manifests(self) -> tuple[SnapshotManifest, ...]:
        with self._lock:
            return tuple(self._manifests.values())

    def select_evictions(
        self,
        bytes_to_free: int,
        *,
        now: Optional[float] = None,
        expected_tool_seconds: Optional[Mapping[str, float]] = None,
    ) -> list[SnapshotManifest]:
        if bytes_to_free <= 0:
            return []
        now = time.time() if now is None else now
        expected_tool_seconds = expected_tool_seconds or {}
        with self._lock:
            candidates = [
                manifest
                for manifest in self._manifests.values()
                if manifest.state is SnapshotState.MOONCAKE_READY
            ]
        candidates.sort(
            key=lambda manifest: (
                manifest.eviction_cost(
                    now, expected_tool_seconds.get(manifest.tool_type or "")
                ),
                manifest.byte_size,
                -manifest.created_at,
            ),
            reverse=True,
        )
        selected = []
        selected_bytes = 0
        for manifest in candidates:
            selected.append(manifest)
            selected_bytes += manifest.byte_size
            if selected_bytes >= bytes_to_free:
                break
        return selected


class SnapshotEvictionController:
    """Per-D request-level admission and high-watermark eviction.

    A multi-D deployment gives each D a shard of the shared Mooncake budget.
    This avoids distributed CAS/candidate coordination in V1 while keeping the
    aggregate bound deterministic when routing is balanced.
    """

    def __init__(
        self,
        snapshot_store: MooncakeSnapshotStore,
        *,
        capacity_bytes: int,
        high_watermark: float = 0.90,
        expected_tool_seconds: Optional[Mapping[str, float]] = None,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if not 0.0 < high_watermark <= 1.0:
            raise ValueError("high_watermark must be in (0, 1]")
        self.snapshot_store = snapshot_store
        self.capacity_bytes = capacity_bytes
        self.high_watermark = high_watermark
        self.expected_tool_seconds = dict(expected_tool_seconds or {})
        self.index = SnapshotIndex()
        self._reserved_bytes = 0
        self._pending_deletes: dict[str, RequestGeneration] = {}
        self._lock = threading.RLock()

    @property
    def byte_limit(self) -> int:
        return int(self.capacity_bytes * self.high_watermark)

    def _reconcile(self) -> None:
        for manifest in self.index.manifests():
            observed = self.snapshot_store.load(
                manifest.request, require_ready=False
            )
            if observed is None or observed.state in {
                SnapshotState.CONSUMED,
                SnapshotState.EVICTED,
                SnapshotState.FINAL,
            }:
                self.index.discard(manifest.snapshot_id)
            else:
                self.index.upsert(observed)

    def _retry_pending_deletes(self) -> None:
        """Finish non-forced whole-snapshot eviction after GET leases drain."""

        for snapshot_id, request in list(self._pending_deletes.items()):
            observed = self.snapshot_store.load(request, require_ready=False)
            if observed is None or observed.state is SnapshotState.EVICTED:
                self._pending_deletes.pop(snapshot_id, None)
                self.index.discard(snapshot_id)
                continue
            if (
                observed.state is not SnapshotState.DELETE_PENDING
                or observed.deletion_target is not SnapshotState.EVICTED
            ):
                continue
            try:
                result = self.snapshot_store.delete_snapshot(
                    observed, final_state=SnapshotState.EVICTED
                )
            except SnapshotLifecycleError:
                continue
            if result.removed:
                self._pending_deletes.pop(snapshot_id, None)
                self.index.discard(snapshot_id)
            else:
                self.index.upsert(
                    self.snapshot_store.load(request, require_ready=False)
                    or observed
                )

    def reserve(self, incoming_bytes: int) -> bool:
        """Evict whole READY snapshots if needed, then reserve a pending Put."""

        if incoming_bytes <= 0 or incoming_bytes > self.byte_limit:
            return False
        with self._lock:
            self._retry_pending_deletes()
            self._reconcile()
            projected = (
                self.index.resident_byte_size
                + self._reserved_bytes
                + incoming_bytes
            )
            bytes_to_free = max(0, projected - self.byte_limit)
            if bytes_to_free:
                candidates = self.index.select_evictions(
                    bytes_to_free,
                    expected_tool_seconds=self.expected_tool_seconds,
                )
                for candidate in candidates:
                    observed = self.snapshot_store.load(
                        candidate.request, require_ready=False
                    )
                    if (
                        observed is None
                        or observed.state is not SnapshotState.MOONCAKE_READY
                    ):
                        self.index.discard(candidate.snapshot_id)
                        continue
                    try:
                        result = self.snapshot_store.delete_snapshot(
                            observed, final_state=SnapshotState.EVICTED
                        )
                    except SnapshotNotReadyError:
                        self.index.discard(candidate.snapshot_id)
                        continue
                    if result.removed:
                        self.index.discard(candidate.snapshot_id)
                    else:
                        self._pending_deletes[candidate.snapshot_id] = (
                            candidate.request
                        )
                        observed = self.snapshot_store.load(
                            candidate.request, require_ready=False
                        )
                        if observed is not None:
                            self.index.upsert(observed)
                projected = (
                    self.index.resident_byte_size
                    + self._reserved_bytes
                    + incoming_bytes
                )
                if projected > self.byte_limit:
                    return False
            self._reserved_bytes += incoming_bytes
            return True

    def commit(self, manifest: SnapshotManifest) -> None:
        if manifest.state is not SnapshotState.MOONCAKE_READY:
            raise SnapshotLifecycleError("only READY snapshots can be committed")
        with self._lock:
            self._reserved_bytes = max(0, self._reserved_bytes - manifest.byte_size)
            self.index.upsert(manifest)

    def cancel(self, byte_size: int) -> None:
        with self._lock:
            self._reserved_bytes = max(0, self._reserved_bytes - max(0, byte_size))


class SharedSnapshotEvictionController:
    """Cross-process admission for all D writers on one node.

    Mooncake does not expose a transactional request-level capacity API.  A
    tiny ledger in ``/dev/shm`` therefore serializes agentic reservations and
    complete-snapshot eviction across independent D scheduler processes.  KV
    bytes remain in Mooncake; the ledger only stores compressed manifests and
    pending reservation metadata.

    The accounting intentionally covers the agentic namespace only.  Native
    HiCache objects and foreign writers remain protected by Mooncake's own
    physical high-watermark policy.
    """

    LEDGER_VERSION = 1

    def __init__(
        self,
        snapshot_store: MooncakeSnapshotStore,
        *,
        ledger_path: str,
        capacity_bytes: int,
        high_watermark: float = 0.90,
        expected_tool_seconds: Optional[Mapping[str, float]] = None,
        reservation_ttl_seconds: float = 300.0,
    ) -> None:
        if not ledger_path or not os.path.isabs(ledger_path):
            raise ValueError("ledger_path must be a non-empty absolute path")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if not 0.0 < high_watermark <= 1.0:
            raise ValueError("high_watermark must be in (0, 1]")
        self.snapshot_store = snapshot_store
        self.ledger_path = ledger_path
        self.capacity_bytes = capacity_bytes
        self.high_watermark = high_watermark
        self.expected_tool_seconds = dict(expected_tool_seconds or {})
        self.reservation_ttl_seconds = max(1.0, reservation_ttl_seconds)
        directory = os.path.dirname(ledger_path)
        if not directory.startswith("/dev/shm/") and directory != "/dev/shm":
            raise ValueError("shared agentic ledger must live under /dev/shm")
        os.makedirs(directory, mode=0o700, exist_ok=True)

    @property
    def byte_limit(self) -> int:
        return int(self.capacity_bytes * self.high_watermark)

    @staticmethod
    def _empty_ledger() -> dict[str, Any]:
        return {
            "version": SharedSnapshotEvictionController.LEDGER_VERSION,
            "reservations": {},
            "residents": {},
            # A P can consume between D's manifest commit and ledger commit.
            # This tiny tombstone prevents the later D step from resurrecting
            # already-freed capacity accounting.
            "terminals": {},
        }

    @staticmethod
    def _encode_manifest(manifest: SnapshotManifest) -> str:
        return base64.b64encode(manifest.to_bytes()).decode("ascii")

    @staticmethod
    def _decode_manifest(value: str) -> SnapshotManifest:
        return SnapshotManifest.from_bytes(base64.b64decode(value.encode("ascii")))

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _open_locked(self):
        fd = os.open(self.ledger_path, os.O_RDWR | os.O_CREAT, 0o600)
        file_obj = os.fdopen(fd, "r+", encoding="utf-8")
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
        return file_obj

    def _read_locked(self, file_obj) -> dict[str, Any]:
        file_obj.seek(0)
        raw = file_obj.read()
        if not raw:
            return self._empty_ledger()
        try:
            ledger = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SnapshotLifecycleError(
                f"corrupt agentic KV ledger {self.ledger_path}"
            ) from exc
        if ledger.get("version") != self.LEDGER_VERSION:
            raise SnapshotLifecycleError(
                f"unsupported agentic KV ledger version {ledger.get('version')}"
            )
        ledger.setdefault("reservations", {})
        ledger.setdefault("residents", {})
        ledger.setdefault("terminals", {})
        return ledger

    @staticmethod
    def _write_locked(file_obj, ledger: Mapping[str, Any]) -> None:
        file_obj.seek(0)
        json.dump(ledger, file_obj, separators=(",", ":"), sort_keys=True)
        file_obj.truncate()
        file_obj.flush()
        os.fsync(file_obj.fileno())

    def _reconcile_locked(self, ledger: dict[str, Any], now: float) -> None:
        terminal_retention = max(600.0, self.reservation_ttl_seconds * 2)
        for snapshot_id, terminal_at in list(ledger["terminals"].items()):
            try:
                expired = now - float(terminal_at) >= terminal_retention
            except (TypeError, ValueError):
                expired = True
            if expired:
                ledger["terminals"].pop(snapshot_id, None)
        reservations = ledger["reservations"]
        for snapshot_id, reservation in list(reservations.items()):
            age = now - float(reservation.get("created_at", 0.0))
            owner_pid = int(reservation.get("owner_pid", -1))
            if age < self.reservation_ttl_seconds and self._pid_alive(owner_pid):
                continue
            try:
                intended = self._decode_manifest(reservation["manifest"])
                observed = self.snapshot_store.load(
                    intended.request, require_ready=False
                )
                if observed is not None and observed.state is SnapshotState.OFFLOADING:
                    self.snapshot_store.recover_stale(observed)
                    observed = self.snapshot_store.load(
                        intended.request, require_ready=False
                    )
                if observed is not None and observed.state not in {
                    SnapshotState.CONSUMED,
                    SnapshotState.EVICTED,
                    SnapshotState.FINAL,
                    SnapshotState.FAILED,
                }:
                    ledger["residents"][snapshot_id] = self._encode_manifest(
                        observed
                    )
            except Exception:
                # Preserve the reservation on metadata-store failure.  It is
                # safer to temporarily reject work than under-account bytes.
                continue
            reservations.pop(snapshot_id, None)

        residents = ledger["residents"]
        for snapshot_id, encoded in list(residents.items()):
            try:
                recorded = self._decode_manifest(encoded)
                observed = self.snapshot_store.load(
                    recorded.request, require_ready=False
                )
            except Exception:
                # A transient metadata-store error must not under-account and
                # admit more bytes, so retain the last known resident record.
                continue
            if observed is None or observed.state in {
                SnapshotState.CONSUMED,
                SnapshotState.EVICTED,
                SnapshotState.FINAL,
                SnapshotState.FAILED,
            }:
                residents.pop(snapshot_id, None)
                continue
            if observed.state is SnapshotState.DELETE_PENDING:
                try:
                    result = self.snapshot_store.delete_snapshot(
                        observed,
                        final_state=(
                            observed.deletion_target or SnapshotState.EVICTED
                        ),
                        update_shared_ledger=False,
                    )
                except Exception:
                    result = None
                if result is not None and result.removed:
                    residents.pop(snapshot_id, None)
                    continue
            residents[snapshot_id] = self._encode_manifest(observed)

    @staticmethod
    def _reservation_bytes(ledger: Mapping[str, Any]) -> int:
        return sum(
            max(0, int(item.get("byte_size", 0)))
            for item in ledger["reservations"].values()
        )

    def _resident_manifests(
        self, ledger: Mapping[str, Any]
    ) -> list[SnapshotManifest]:
        result = []
        for snapshot_id, encoded in ledger["residents"].items():
            try:
                result.append(self._decode_manifest(encoded))
            except Exception as exc:
                # Never turn corrupt accounting into free capacity.
                raise SnapshotLifecycleError(
                    f"corrupt resident manifest in shared ledger: {snapshot_id}"
                ) from exc
        return result

    def reserve(self, manifest: SnapshotManifest) -> bool:
        """Atomically reserve one snapshot and evict whole READY snapshots."""

        incoming_bytes = manifest.byte_size
        if incoming_bytes <= 0 or incoming_bytes > self.byte_limit:
            return False
        file_obj = self._open_locked()
        try:
            ledger = self._read_locked(file_obj)
            now = time.time()
            self._reconcile_locked(ledger, now)
            if manifest.snapshot_id in ledger["reservations"]:
                return True

            residents = self._resident_manifests(ledger)
            resident_bytes = sum(
                item.byte_size
                for item in residents
                if item.state
                not in {
                    SnapshotState.CONSUMED,
                    SnapshotState.EVICTED,
                    SnapshotState.FINAL,
                    SnapshotState.FAILED,
                }
            )
            projected = (
                resident_bytes
                + self._reservation_bytes(ledger)
                + incoming_bytes
            )
            bytes_to_free = max(0, projected - self.byte_limit)
            if bytes_to_free:
                candidates = [
                    item
                    for item in residents
                    if item.state is SnapshotState.MOONCAKE_READY
                ]
                candidates.sort(
                    key=lambda item: (
                        item.eviction_cost(
                            now,
                            self.expected_tool_seconds.get(item.tool_type or ""),
                        ),
                        item.byte_size,
                        -item.created_at,
                    ),
                    reverse=True,
                )
                freed = 0
                for candidate in candidates:
                    try:
                        observed = self.snapshot_store.load(
                            candidate.request, require_ready=False
                        )
                        if (
                            observed is None
                            or observed.state is not SnapshotState.MOONCAKE_READY
                        ):
                            ledger["residents"].pop(candidate.snapshot_id, None)
                            continue
                        result = self.snapshot_store.delete_snapshot(
                            observed,
                            final_state=SnapshotState.EVICTED,
                            update_shared_ledger=False,
                        )
                        latest = self.snapshot_store.load(
                            candidate.request, require_ready=False
                        )
                        if result.removed:
                            ledger["residents"].pop(candidate.snapshot_id, None)
                            freed += candidate.byte_size
                        elif latest is not None:
                            ledger["residents"][candidate.snapshot_id] = (
                                self._encode_manifest(latest)
                            )
                    except SnapshotNotReadyError:
                        continue
                    if freed >= bytes_to_free:
                        break

                residents = self._resident_manifests(ledger)
                projected = (
                    sum(
                        item.byte_size
                        for item in residents
                        if item.state
                        not in {
                            SnapshotState.CONSUMED,
                            SnapshotState.EVICTED,
                            SnapshotState.FINAL,
                            SnapshotState.FAILED,
                        }
                    )
                    + self._reservation_bytes(ledger)
                    + incoming_bytes
                )
                if projected > self.byte_limit:
                    self._write_locked(file_obj, ledger)
                    return False

            ledger["reservations"][manifest.snapshot_id] = {
                "byte_size": incoming_bytes,
                "owner_pid": os.getpid(),
                "created_at": now,
                "manifest": self._encode_manifest(manifest),
            }
            self._write_locked(file_obj, ledger)
            return True
        finally:
            file_obj.close()

    def commit(self, manifest: SnapshotManifest) -> None:
        if manifest.state is not SnapshotState.MOONCAKE_READY:
            raise SnapshotLifecycleError("only READY snapshots can be committed")
        file_obj = self._open_locked()
        try:
            ledger = self._read_locked(file_obj)
            self._reconcile_locked(ledger, time.time())
            ledger["reservations"].pop(manifest.snapshot_id, None)
            # commit_publish makes READY visible before this ledger commit.
            # A fast P can consume and delete the snapshot in that window.
            # Recheck while holding the ledger lock: if P already reached a
            # terminal state, do not resurrect stale capacity accounting.
            observed = self.snapshot_store.load(
                manifest.request, require_ready=False
            )
            if manifest.snapshot_id in ledger["terminals"]:
                ledger["residents"].pop(manifest.snapshot_id, None)
            elif (
                observed is not None
                and observed.state is SnapshotState.MOONCAKE_READY
            ):
                ledger["residents"][manifest.snapshot_id] = self._encode_manifest(
                    observed
                )
            else:
                ledger["residents"].pop(manifest.snapshot_id, None)
            self._write_locked(file_obj, ledger)
        finally:
            file_obj.close()

    def cancel(self, manifest: SnapshotManifest) -> None:
        file_obj = self._open_locked()
        try:
            ledger = self._read_locked(file_obj)
            ledger["reservations"].pop(manifest.snapshot_id, None)
            self._write_locked(file_obj, ledger)
        finally:
            file_obj.close()


def expand_mha_page_keys(
    logical_page_keys: Iterable[str], suffixes: Sequence[str]
) -> tuple[str, ...]:
    """Expand logical page hashes into the physical K/V object keys."""

    result = []
    for key in logical_page_keys:
        for suffix in suffixes:
            result.append(f"{key}_{suffix}_k")
            result.append(f"{key}_{suffix}_v")
    return tuple(result)


def namespace_page_keys(
    request: RequestGeneration, logical_page_keys: Iterable[str]
) -> tuple[str, ...]:
    """Make physical page ownership generation-unique to prevent ABA deletes."""

    prefix = page_namespace(request)
    return tuple(f"{prefix}{key}" for key in logical_page_keys)


def page_namespace(request: RequestGeneration) -> str:
    """Prefix applied to every logical page key in a snapshot generation."""

    return f"sglang:agentic-kv:v1:page:{request.storage_id}:"


def token_ids_digest(token_ids: Sequence[int]) -> str:
    """Stable compact guard against direct-transfer prompt serialization drift."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()
