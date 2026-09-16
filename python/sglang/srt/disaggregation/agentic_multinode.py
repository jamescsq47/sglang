"""Opt-in deployment validation for the experimental cross-host agentic path.

This module performs no I/O, registration, transfer, or scheduler mutation. A
valid configuration is not evidence that the transport is integrated or that a
shared filesystem implements the required cross-host ownership primitives.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Optional


PREFIX = "SGLANG_AGENTIC_MULTINODE_"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _enabled(value: str, key: str) -> bool:
    value = value.strip().lower()
    if value not in _TRUE | _FALSE:
        raise ValueError(f"{key} must be a boolean, got {value!r}")
    return value in _TRUE


def _required(env: Mapping[str, str], suffix: str) -> str:
    key = PREFIX + suffix
    value = env.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required for multi-node configuration")
    return value


def _identity(env: Mapping[str, str], suffix: str) -> str:
    value = _required(env, suffix)
    if not _IDENTITY.fullmatch(value):
        raise ValueError(f"{PREFIX + suffix} is not a safe run/node/engine identifier")
    return value


def _integer(env: Mapping[str, str], suffix: str, default: Optional[int] = None) -> int:
    raw = env.get(PREFIX + suffix)
    if raw is None and default is not None:
        return default
    try:
        return int(_required(env, suffix))
    except ValueError as exc:
        raise ValueError(f"{PREFIX + suffix} must be an integer") from exc


@dataclass(frozen=True)
class MultiNodeConfig:
    run_id: str
    node_id: str
    engine_id: str
    role: str
    host_ip: str
    tp_size: int
    peer_tp_size: int
    control_root: str
    dp_size: int = 1
    pp_size: int = 1
    engine_nnodes: int = 1
    control_poll_interval: float = 0.1

    @property
    def control_directory(self) -> str:
        """One shared run namespace; identical on P, D and Router."""
        return os.path.join(self.control_root, self.run_id)

    def validate_server_args(self, server_args) -> None:
        """Check actual engine arguments explicitly, without altering them."""
        if self.role == "router":
            raise ValueError("multi-node router identity cannot start a model worker")
        expected = {"tp_size": self.tp_size, "dp_size": 1, "pp_size": 1, "nnodes": 1}
        for name, value in expected.items():
            actual = getattr(server_args, name, None)
            if actual is None or int(actual) != value:
                raise ValueError(f"server_args.{name}={actual!r}; multi-node V1 requires {value}")
        actual_role = getattr(server_args, "disaggregation_mode", None)
        actual_role = getattr(actual_role, "value", actual_role)
        if actual_role != self.role:
            raise ValueError("server disaggregation_mode disagrees with multi-node ROLE")
        if getattr(server_args, "disaggregation_transfer_backend", None) != "nixl":
            raise ValueError("multi-node Direct requires the NIXL transfer backend")
        if bool(getattr(server_args, "enable_dp_attention", False)):
            raise ValueError("multi-node V1 does not support DP attention")
        if getattr(server_args, "speculative_algorithm", None):
            raise ValueError("multi-node V1 does not transfer speculative draft state")
        if bool(getattr(server_args, "enable_hierarchical_cache", False)):
            raise ValueError("multi-node custom Host staging cannot enable native HiCache")
        if getattr(server_args, "hicache_storage_backend", None):
            raise ValueError("multi-node custom Host staging cannot enable native storage")
        if bool(getattr(server_args, "disaggregation_decode_enable_offload_kvcache", False)):
            raise ValueError("multi-node uses custom lifecycle offload, not native Decode offload")

    def validate_kv_pool(self, kv_pool) -> None:
        """Reject unsupported layouts before starting any agentic I/O workers.

        A hybrid snapshot is not just attention KV. Until the network adapter
        covers its recurrent/conv checkpoint, do not silently transfer a subset.
        """
        pool = getattr(kv_pool, "full_kv_pool", kv_pool)
        if hasattr(kv_pool, "mamba_pool") or hasattr(pool, "mamba_pool"):
            raise ValueError("multi-node V1 supports MHA KV only; Mamba state transport is not integrated")
        if pool is not kv_pool or not all(hasattr(pool, name) for name in ("k_buffer", "v_buffer")):
            raise ValueError("multi-node V1 requires the native MHA KV pool, not MLA/SWA/hybrid layouts")


def validate_multinode_runtime(server_args, kv_pool=None) -> Optional[MultiNodeConfig]:
    """Startup-only guard. Disabled mode never inspects local model/layout flags."""
    config = load_multinode_config()
    if config is not None:
        config.validate_server_args(server_args)
        for key in (
            "SGLANG_AGENTIC_KV_LIFECYCLE",
            "SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY",
            "SGLANG_AGENTIC_KV_HOST_STAGING",
            "SGLANG_AGENTIC_KV_P2D_HOST_STAGING",
        ):
            if not _enabled(os.getenv(key, "0"), key):
                raise ValueError(f"multi-node full method requires {key}=true")
        if config.role == "decode" and not _enabled(
            os.getenv("SGLANG_AGENTIC_KV_D_HOSTLESS", "0"), "SGLANG_AGENTIC_KV_D_HOSTLESS"
        ):
            raise ValueError("multi-node Decode requires custom D_HOSTLESS, not native offload storage")
        if os.getenv("SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND", "") != "memfd":
            raise ValueError("multi-node Host data must use source-local memfd DRAM")
        shared_root = os.path.realpath(config.control_directory)
        for key in (
            "SGLANG_AGENTIC_KV_LEDGER_PATH", "SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH",
            "SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH", "SGLANG_AGENTIC_KV_METADATA_DIR",
            "SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR",
        ):
            value = os.getenv(key, "")
            if not value or not os.path.isabs(value) or os.path.commonpath(
                [shared_root, os.path.realpath(value)]
            ) != shared_root:
                raise ValueError(f"{key} must be inside the shared run control directory")
        if kv_pool is not None:
            config.validate_kv_pool(kv_pool)
    return config


def load_multinode_config(environ: Optional[Mapping[str, str]] = None) -> Optional[MultiNodeConfig]:
    """Validate opt-in settings; disabled returns None and preserves old behavior.

    The caller must explicitly invoke this function. It does not patch imports,
    set environment variables, or switch the engine to a different backend.
    """
    env = os.environ if environ is None else environ
    if not _enabled(env.get(PREFIX + "ENABLED", "0"), PREFIX + "ENABLED"):
        return None
    run_id, node_id, engine_id = (_identity(env, key) for key in ("RUN_ID", "NODE_ID", "ENGINE_ID"))
    role = _required(env, "ROLE")
    if role not in {"prefill", "decode", "router"}:
        raise ValueError("multi-node ROLE must be prefill, decode or router")
    host_ip = _required(env, "HOST_IP")
    try:
        address = ipaddress.ip_address(host_ip)
    except ValueError as exc:
        raise ValueError("multi-node HOST_IP must be an explicit reachable IP, not an interface name") from exc
    if address.is_unspecified or address.is_loopback or address.is_multicast or address.is_link_local:
        raise ValueError("multi-node HOST_IP must be a peer-reachable unicast address")
    if address.version != 4:
        raise ValueError("multi-node V1 requires IPv4: reverse-bootstrap address formatting is not IPv6-audited")
    # Private cluster addresses are valid; actual reachability is a remote test.
    tp_size = _integer(env, "TP_SIZE")
    peer_tp_size = _integer(env, "PEER_TP_SIZE")
    if not 1 <= tp_size <= 8 or peer_tp_size != tp_size:
        raise ValueError("multi-node V1 requires matching P/D TP sizes in [1, 8]")
    for name in ("DP_SIZE", "PP_SIZE", "ENGINE_NNODES"):
        if _integer(env, name, 1) != 1:
            raise ValueError(f"{PREFIX + name} must be 1: each whole TP group stays on one host")
    control_root = os.path.normpath(_required(env, "CONTROL_ROOT"))
    if not os.path.isabs(control_root) or control_root == "/":
        raise ValueError("CONTROL_ROOT must be a dedicated absolute shared-filesystem directory")
    # realpath also catches an existing alias to /dev/shm. This is a negative
    # guard only: a path outside these trees is NOT proof of cross-host sharing.
    resolved = os.path.realpath(control_root)
    if any(resolved == path or resolved.startswith(path + "/") for path in ("/dev", "/proc", "/sys", "/run")):
        raise ValueError("CONTROL_ROOT must be shared across hosts, not local tmpfs/device state")
    try:
        poll_interval = float(env.get(PREFIX + "CONTROL_POLL_INTERVAL", "0.1"))
    except ValueError as exc:
        raise ValueError("CONTROL_POLL_INTERVAL must be numeric") from exc
    if not math.isfinite(poll_interval) or not 0.05 <= poll_interval <= 1.0:
        raise ValueError("CONTROL_POLL_INTERVAL must be between 0.05 and 1.0 seconds")
    config = MultiNodeConfig(run_id, node_id, engine_id, role, host_ip, tp_size, peer_tp_size, control_root,
                             control_poll_interval=poll_interval)
    agreements = {
        "SGLANG_HOST_IP": host_ip,
        "HOST_IP": host_ip,
        "SGLANG_AGENTIC_KV_ENGINE_ID": engine_id,
        "SGLANG_AGENTIC_KV_TP_SIZE": str(tp_size),
        "SGLANG_PD_P_READY_DIR": config.control_directory,
    }
    for key, expected in agreements.items():
        actual = env.get(key)
        if actual not in {None, ""} and actual != expected:
            raise ValueError(f"{key}={actual!r} disagrees with multi-node configuration ({expected!r})")
    # These opt-in optimizations assume node-local notifications/preparation.
    # Do not enable them implicitly in an experimental remote deployment.
    for key in (
        "SGLANG_AGENTIC_KV_P_HOST_ASYNC_PREPARE",
        "SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS",
        "SGLANG_AGENTIC_KV_NUMA_HOST_POOL",
        "SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER",
        "SGLANG_PD_ABLATION_P2D_PREBIND",
    ):
        if _enabled(env.get(key, "0"), key):
            raise ValueError(f"{key} is outside the multi-node V1 scope")
    return config


def control_poll_interval(environ: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """Remote filesystem changes need bounded polling, not inotify alone."""
    config = load_multinode_config(environ)
    return None if config is None else config.control_poll_interval


def source_host_placement(tp_size: int, recovery_domain: int) -> Optional[tuple[int, list[int]]]:
    """Keep a recovery hint but place slow shards on source-local NUMA nodes.

    Node-local NUMA IDs must never be interpreted as a remote P's memory domain.
    Returning None leaves the single-node dynamic Host router entirely intact.
    """
    config = load_multinode_config()
    if config is None:
        return None
    if config.role != "decode" or int(tp_size) != config.tp_size:
        raise ValueError("D->P source Host placement requires the configured Decode TP group")
    raw = os.getenv("SGLANG_AGENTIC_KV_TP_NUMA_NODES", "").strip()
    nodes = [int(value.strip()) for value in raw.split(",")] if raw else [-1] * int(tp_size)
    if len(nodes) != int(tp_size) or any(node < -1 for node in nodes):
        raise ValueError("source Host NUMA vector must contain one valid entry per TP rank")
    return int(recovery_domain), nodes


def capabilities() -> dict:
    """Machine-readable stage boundary, not a promise of transport performance."""
    return {
        "integrated": True,
        "engine_integration": True,
        "hardware_verified": False,
        "status": "experimental_remote_smoke_required",
        "features": [
            "source_local_host_rdma", "shared_control_polling", "tp_shard_atomicity",
            "remote_host_fence_release", "p2d_d2p_integration",
        ],
        "direct_transport": "existing_nixl_network_capable_remote_unverified",
        "host_transport": "source_local_dram_to_remote_hbm_nixl_ucx",
        "shared_control": "posix_bridge_requires_two_host_validation",
        "tp_scope": "matching_1_to_8_each_group_within_one_host",
        "safe_to_launch_full_pipeline": False,
        "experimental_smoke_launch_enabled": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true", required=True)
    parser.parse_args()
    config = load_multinode_config()
    print(json.dumps({"enabled": config is not None,
                      "configuration": asdict(config) if config else None,
                      "capabilities": capabilities(),
                      "validation_only": True,
                      "runtime_or_rdma_verified": False}, indent=2))


if __name__ == "__main__":
    main()
