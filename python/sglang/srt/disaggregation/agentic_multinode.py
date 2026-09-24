"""Configuration boundary for the event-driven multi-node agentic PD path.

V2 has no shared control directory. Request-generation ownership is
coordinated over a long-lived TCP session and KV bytes travel by NIXL or
through source-local DRAM. Files may still hold logs/checkpoints; they never
participate in request progress.

The feature is opt-in, so the established single-node ``pd_mamba`` path is
unchanged unless ``SGLANG_AGENTIC_MULTINODE_ENABLED=1`` is set.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable


PREFIX = "SGLANG_AGENTIC_MULTINODE_"
GROUP_PREFIX = "SGLANG_AGENTIC_GROUP_"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _enabled(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in _TRUE | _FALSE:
        raise ValueError(f"{key} must be a boolean, got {value!r}")
    return normalized in _TRUE


def _required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required for multi-node configuration")
    return value


def _identity(env: Mapping[str, str], key: str) -> str:
    value = _required(env, key)
    if not _IDENTITY.fullmatch(value):
        raise ValueError(f"{key} is not a safe identifier")
    return value


def _integer(env: Mapping[str, str], key: str, default: Optional[int] = None) -> int:
    raw = env.get(key)
    if (raw is None or raw == "") and default is not None:
        return int(default)
    try:
        return int(_required(env, key))
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _tcp_endpoint(value: str) -> tuple[str, int]:
    endpoint = value.removeprefix("tcp://")
    try:
        host, raw_port = endpoint.rsplit(":", 1)
        port = int(raw_port)
        address = ipaddress.ip_address(host)
    except (ValueError, TypeError) as exc:
        raise ValueError("group endpoint must be tcp://IP:PORT") from exc
    if (
        address.is_unspecified
        or address.is_multicast
        or address.is_link_local
        or not 1 <= port <= 65535
    ):
        raise ValueError("group endpoint must be a reachable unicast IP and port")
    return str(address), port


@dataclass(frozen=True, slots=True)
class MultiNodeConfig:
    run_id: str
    node_id: str
    engine_id: str
    group_id: str
    endpoint_group: str
    peer_group: str
    peer_role: str
    coordinator_group: str
    role: str
    host_ip: str
    tp_size: int
    peer_tp_size: int
    control_endpoint: str
    control_token: str

    @property
    def endpoint(self) -> tuple[str, int]:
        return _tcp_endpoint(self.control_endpoint)

    def validate_server_args(self, server_args) -> None:
        if self.role == "router":
            raise ValueError("router identity cannot start a model worker")
        expected = {
            "tp_size": self.tp_size,
            "dp_size": 1,
            "pp_size": 1,
            "nnodes": 1,
        }
        for name, value in expected.items():
            actual = getattr(server_args, name, None)
            if actual is None or int(actual) != value:
                raise ValueError(
                    f"server_args.{name}={actual!r}; multi-node V2 requires {value}"
                )
        actual_role = getattr(server_args, "disaggregation_mode", None)
        actual_role = getattr(actual_role, "value", actual_role)
        if actual_role != self.role:
            raise ValueError("server disaggregation_mode disagrees with ROLE")
        if getattr(server_args, "disaggregation_transfer_backend", None) != "nixl":
            raise ValueError("multi-node Direct requires NIXL")
        if bool(getattr(server_args, "enable_dp_attention", False)):
            raise ValueError("multi-node V2 does not support DP attention")
        if getattr(server_args, "speculative_algorithm", None):
            raise ValueError("multi-node V2 does not transfer speculative state")
        if bool(getattr(server_args, "enable_hierarchical_cache", False)):
            raise ValueError("custom Host staging cannot enable native HiCache")
        if getattr(server_args, "hicache_storage_backend", None):
            raise ValueError("custom Host staging cannot enable native storage")
        if bool(
            getattr(
                server_args,
                "disaggregation_decode_enable_offload_kvcache",
                False,
            )
        ):
            raise ValueError("multi-node V2 does not use native Decode offload")

    def validate_kv_pool(self, kv_pool) -> None:
        """Accept MHA and the explicitly supported Qwen3.5 Attention+GDN pool."""
        pool = getattr(kv_pool, "full_kv_pool", kv_pool)
        if not all(hasattr(pool, name) for name in ("k_buffer", "v_buffer")):
            raise ValueError("multi-node V2 requires an MHA attention component")
        if hasattr(kv_pool, "mamba_pool"):
            cache = getattr(kv_pool.mamba_pool, "mamba_cache", None)
            if cache is None or not hasattr(cache, "conv") or not hasattr(
                cache, "temporal"
            ):
                raise ValueError("hybrid pool lacks request-owned GDN state")


@runtime_checkable
class AgenticMultinodeRuntime(Protocol):
    """Scheduler-facing boundary of the V2 control/data runtime.

    The scheduler never polls transports or legacy lifecycle stores.  It only
    submits an immutable request to this non-blocking boundary, advances work
    that is already signalled ready, and consumes the role-specific memory
    bridge.  A concrete transport runtime is intentionally supplied outside
    this configuration module.
    """

    p_memory_bridge: Any
    d_memory_bridge: Any

    def submit_request(self, req: Any, *, is_retracted: bool = False) -> None:
        ...

    def progress_nonblocking(self) -> None:
        ...

    def take_activation_ticket(self, timeout: Optional[float] = None) -> Any:
        ...

    def activate_staged(self, ticket: Any) -> None:
        ...

    def confirm_scheduler_adopted(self, ticket: Any) -> None:
        ...

    def close(self) -> None:
        ...


def create_agentic_multinode_runtime(
    scheduler: Any,
    config: MultiNodeConfig,
    *,
    draft_token_to_kv_pool: Any = None,
    draft_model_config: Any = None,
) -> AgenticMultinodeRuntime:
    """Create the opt-in V2 runtime without falling back to V1 machinery.

    Keeping the import lazy makes the scheduler shell independently testable
    and prevents disabled ``pd_mamba`` launches from importing experimental
    multi-node transport code.  The implementation module is the only place
    allowed to compose memory authorities with physical transfer queues.
    """

    try:
        from sglang.srt.disaggregation.agentic_multinode_composite import (
            create_runtime,
        )
    except ImportError as exc:
        raise RuntimeError(
            "multi-node V2 is enabled but its runtime implementation is not "
            "available"
        ) from exc
    runtime = create_runtime(
        scheduler,
        config,
        draft_token_to_kv_pool=draft_token_to_kv_pool,
        draft_model_config=draft_model_config,
    )
    if not isinstance(runtime, AgenticMultinodeRuntime):
        raise TypeError("multi-node runtime does not implement the scheduler contract")
    return runtime


def load_multinode_config(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[MultiNodeConfig]:
    env = os.environ if environ is None else environ
    if not _enabled(env.get(PREFIX + "ENABLED", "0"), PREFIX + "ENABLED"):
        return None

    run_id = _identity(env, GROUP_PREFIX + "RUN_ID")
    node_id = _identity(env, PREFIX + "NODE_ID")
    engine_id = _identity(env, PREFIX + "ENGINE_ID")
    group_id = _identity(env, GROUP_PREFIX + "GROUP_ID")
    role = _required(env, PREFIX + "ROLE")
    if role not in {"prefill", "decode", "router"}:
        raise ValueError("ROLE must be prefill, decode or router")
    host_ip = _required(env, PREFIX + "HOST_IP")
    try:
        host_address = ipaddress.ip_address(host_ip)
    except ValueError as exc:
        raise ValueError("HOST_IP must be an explicit IP") from exc
    if (
        host_address.is_unspecified
        or host_address.is_loopback
        or host_address.is_multicast
        or host_address.is_link_local
    ):
        raise ValueError("HOST_IP must be peer reachable")

    tp_size = _integer(env, GROUP_PREFIX + "SIZE")
    peer_tp_size = _integer(env, PREFIX + "PEER_TP_SIZE")
    if not 1 <= tp_size <= 8 or peer_tp_size != tp_size:
        raise ValueError("matching P/D TP sizes in [1, 8] are required")
    for suffix in ("DP_SIZE", "PP_SIZE", "ENGINE_NNODES"):
        if _integer(env, PREFIX + suffix, 1) != 1:
            raise ValueError(f"{PREFIX + suffix} must be 1")

    endpoint = _required(env, GROUP_PREFIX + "ENDPOINT")
    _tcp_endpoint(endpoint)
    token = _required(env, GROUP_PREFIX + "TOKEN")
    if len(token) < 16:
        raise ValueError("group control token must contain at least 16 characters")

    endpoint_group = _identity(env, GROUP_PREFIX + "ENDPOINT_GROUP")
    endpoint_role = _required(env, GROUP_PREFIX + "ENDPOINT_ROLE")
    if endpoint_group != engine_id:
        raise ValueError("control endpoint group must equal this engine ID")
    if endpoint_role != role:
        raise ValueError("control endpoint role must equal this engine role")
    peer_group = _identity(env, GROUP_PREFIX + "PEER_GROUP")
    peer_role = _required(env, GROUP_PREFIX + "PEER_ROLE")
    if peer_group == endpoint_group or peer_role not in {"prefill", "decode"}:
        raise ValueError("control peer identity is invalid")
    if {endpoint_role, peer_role} != {"prefill", "decode"}:
        raise ValueError("one Prefill and one Decode endpoint are required")
    coordinator_group = _identity(env, GROUP_PREFIX + "COORDINATOR_GROUP")
    if coordinator_group not in {endpoint_group, peer_group}:
        raise ValueError("control coordinator must be one link endpoint")

    forbidden = (
        PREFIX + "CONTROL_ROOT",
        "SGLANG_AGENTIC_KV_LEDGER_PATH",
        "SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH",
        "SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH",
        "SGLANG_AGENTIC_KV_METADATA_DIR",
        "SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR",
        "SGLANG_PD_P_READY_DIR",
    )
    configured = [key for key in forbidden if env.get(key, "").strip()]
    if configured:
        raise ValueError(
            "multi-node V2 forbids filesystem control settings: "
            + ", ".join(configured)
        )

    agreements = {
        "SGLANG_HOST_IP": host_ip,
        "HOST_IP": host_ip,
        "SGLANG_AGENTIC_KV_ENGINE_ID": engine_id,
        "SGLANG_AGENTIC_KV_TP_SIZE": str(tp_size),
    }
    for key, expected in agreements.items():
        actual = env.get(key)
        if actual not in {None, ""} and actual != expected:
            raise ValueError(f"{key}={actual!r} disagrees with {expected!r}")

    return MultiNodeConfig(
        run_id=run_id,
        node_id=node_id,
        engine_id=engine_id,
        group_id=group_id,
        endpoint_group=endpoint_group,
        peer_group=peer_group,
        peer_role=peer_role,
        coordinator_group=coordinator_group,
        role=role,
        host_ip=host_ip,
        tp_size=tp_size,
        peer_tp_size=peer_tp_size,
        control_endpoint=endpoint,
        control_token=token,
    )


def validate_multinode_runtime(server_args, kv_pool=None) -> Optional[MultiNodeConfig]:
    config = load_multinode_config()
    if config is None:
        return None
    config.validate_server_args(server_args)
    # The V2 runtime has its own memory authority and four data queues.  The
    # legacy CUSTOM_STORAGE/HOST_STAGING/D_HOSTLESS switches instantiate the
    # old filesystem-ledger implementation and are deliberately not activation
    # requirements here.  Agent metadata itself remains mandatory.
    key = "SGLANG_AGENTIC_KV_LIFECYCLE"
    if not _enabled(os.getenv(key, "0"), key):
        raise ValueError(f"multi-node full method requires {key}=true")
    if os.getenv(PREFIX + "HOST_BACKEND", "") != "memfd":
        raise ValueError("Host data must use source-local memfd DRAM")
    if kv_pool is not None:
        config.validate_kv_pool(kv_pool)
    return config


def control_poll_interval(
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    """Compatibility hook for the old single-node watcher.

    V2 never turns a local file watcher into a shared-filesystem poller.  The
    function remains importable so disabled/single-node ``pd_mamba`` modules
    keep their original edge-triggered behavior.
    """
    return None


def source_host_placement(
    tp_size: int, recovery_domain: int
) -> Optional[tuple[int, list[int]]]:
    """Return source-local NUMA placement; never a remote-node domain."""
    config = load_multinode_config()
    if config is None:
        return None
    if config.role != "decode" or int(tp_size) != config.tp_size:
        raise ValueError("D-to-P Host placement requires this Decode TP group")
    raw = os.getenv("SGLANG_AGENTIC_KV_TP_NUMA_NODES", "").strip()
    nodes = (
        [int(value.strip()) for value in raw.split(",")]
        if raw
        else [-1] * int(tp_size)
    )
    if len(nodes) != int(tp_size) or any(node < -1 for node in nodes):
        raise ValueError("NUMA vector must contain one entry per TP rank")
    return int(recovery_domain), nodes


def capabilities() -> dict:
    return {
        "architecture": "event_driven_v2",
        "control": "tcp_in_memory_rank0_authority",
        "runtime_filesystem_control": False,
        "host_placement": "source_local_memfd_dram",
        "paths": ["d2p_direct", "d2host2p", "p2d_direct", "p2host2d"],
        "tp_scope": "matching_1_to_8",
        "hybrid_state": "qwen35_attention_gdn_complete_snapshot",
        "hardware_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true", required=True)
    parser.parse_args()
    config = load_multinode_config()
    print(
        json.dumps(
            {
                "enabled": config is not None,
                "configuration": asdict(config) if config else None,
                "capabilities": capabilities(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
