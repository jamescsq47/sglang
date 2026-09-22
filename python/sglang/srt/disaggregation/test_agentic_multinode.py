"""CPU-only deployment guard tests; no CUDA, SSH, NFS, or RDMA initialization."""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest


def _module():
    path = pathlib.Path(__file__).with_name("agentic_multinode.py")
    spec = importlib.util.spec_from_file_location("agentic_multinode_unit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m = _module()


def environment(**changes):
    values = {"ENABLED": "1", "RUN_ID": "r-001", "NODE_ID": "node-a",
              "ENGINE_ID": "p-a", "ROLE": "prefill", "HOST_IP": "10.20.1.2",
              "TP_SIZE": "8", "PEER_TP_SIZE": "8", "CONTROL_ROOT": "/shared/dualpd-control"}
    values.update(changes)
    return {m.PREFIX + key: value for key, value in values.items()}


def test_disabled_keeps_original_behavior():
    assert m.load_multinode_config({}) is None
    assert m.control_poll_interval({}) is None
    assert m.load_multinode_config(environment(ENABLED="0", TP_SIZE="nonsense")) is None


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_matching_tp_and_distinct_ids(tp_size):
    config = m.load_multinode_config(environment(TP_SIZE=str(tp_size), PEER_TP_SIZE=str(tp_size)))
    assert config.tp_size == tp_size
    assert config.control_directory == "/shared/dualpd-control/r-001"
    assert config.node_id == "node-a"
    assert m.control_poll_interval(environment()) == 0.1


@pytest.mark.parametrize("changes", [
    {"TP_SIZE": "0"}, {"TP_SIZE": "16", "PEER_TP_SIZE": "16"},
    {"PEER_TP_SIZE": "4"}, {"DP_SIZE": "2"}, {"PP_SIZE": "2"},
    {"ENGINE_NNODES": "2"}, {"ROLE": "unknown"}, {"RUN_ID": "../other"},
    {"NODE_ID": ""}, {"ENGINE_ID": "a/b"}, {"ENABLED": "maybe"},
    {"HOST_IP": "127.0.0.1"}, {"HOST_IP": "0.0.0.0"},
    {"HOST_IP": "169.254.1.1"}, {"HOST_IP": "224.0.0.1"},
    {"HOST_IP": "my-node"}, {"HOST_IP": "2001:db8::1"},
    {"CONTROL_ROOT": "/dev/shm/abc"}, {"CONTROL_ROOT": "/run/abc"},
    {"CONTROL_ROOT": "/"}, {"CONTROL_ROOT": "relative"},
    {"CONTROL_POLL_INTERVAL": "0"}, {"CONTROL_POLL_INTERVAL": "nan"},
    {"CONTROL_POLL_INTERVAL": "inf"}, {"CONTROL_POLL_INTERVAL": "2"},
])
def test_invalid_deployment_refused(changes):
    with pytest.raises(ValueError):
        m.load_multinode_config(environment(**changes))


@pytest.mark.parametrize("key,value", [
    ("SGLANG_HOST_IP", "10.20.1.3"), ("HOST_IP", "10.20.1.3"),
    ("SGLANG_AGENTIC_KV_ENGINE_ID", "other-engine"),
    ("SGLANG_AGENTIC_KV_TP_SIZE", "2"),
    ("SGLANG_PD_P_READY_DIR", "/dev/shm/local"),
    ("SGLANG_AGENTIC_KV_P_HOST_ASYNC_PREPARE", "1"),
    ("SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS", "1"),
    ("SGLANG_AGENTIC_KV_NUMA_HOST_POOL", "1"),
    ("SGLANG_PD_ABLATION_P2D_PREBIND", "1"),
])
def test_disagreement_and_unsupported_flags_refused(key, value):
    env = environment()
    env[key] = value
    with pytest.raises(ValueError):
        m.load_multinode_config(env)


def test_existing_local_alias_is_refused(tmp_path):
    alias = tmp_path / "looks-shared"
    alias.symlink_to("/dev/shm", target_is_directory=True)
    with pytest.raises(ValueError):
        m.load_multinode_config(environment(CONTROL_ROOT=str(alias)))


def test_agreements_do_not_mutate_environment():
    env = environment()
    env.update(SGLANG_HOST_IP="10.20.1.2", SGLANG_AGENTIC_KV_ENGINE_ID="p-a",
               SGLANG_AGENTIC_KV_TP_SIZE="8", SGLANG_PD_P_READY_DIR="/shared/dualpd-control/r-001")
    original = dict(env)
    m.load_multinode_config(env)
    assert env == original


def test_real_server_args_must_match():
    config = m.load_multinode_config(environment())
    args = SimpleNamespace(tp_size=8, dp_size=1, pp_size=1, nnodes=1,
                           disaggregation_mode="prefill", disaggregation_transfer_backend="nixl")
    config.validate_server_args(args)
    args.tp_size = 4
    with pytest.raises(ValueError):
        config.validate_server_args(args)
    args.tp_size = 8
    args.disaggregation_mode = "decode"
    with pytest.raises(ValueError):
        config.validate_server_args(args)


def test_capabilities_separate_integration_from_hardware_acceptance():
    caps = m.capabilities()
    assert caps["integrated"] is True
    assert caps["engine_integration"] is True
    assert caps["hardware_verified"] is False
    assert caps["safe_to_launch_full_pipeline"] is False
    assert caps["experimental_smoke_launch_enabled"] is True
    assert set(caps["features"]) == {
        "source_local_host_rdma", "shared_control_polling", "tp_shard_atomicity",
        "remote_host_fence_release", "p2d_d2p_integration",
    }


@pytest.mark.parametrize("flag,value", [
    ("speculative_algorithm", "EAGLE"),
    ("enable_hierarchical_cache", True),
    ("hicache_storage_backend", "mooncake"),
    ("disaggregation_decode_enable_offload_kvcache", True),
])
def test_native_storage_and_draft_state_refused(flag, value):
    config = m.load_multinode_config(environment())
    args = SimpleNamespace(tp_size=8, dp_size=1, pp_size=1, nnodes=1,
                           disaggregation_mode="prefill", disaggregation_transfer_backend="nixl")
    setattr(args, flag, value)
    with pytest.raises(ValueError):
        config.validate_server_args(args)


def test_pool_validation_prevents_partial_hybrid_snapshot():
    config = m.load_multinode_config(environment())
    config.validate_kv_pool(SimpleNamespace(k_buffer=[], v_buffer=[]))
    for pool in (SimpleNamespace(mamba_pool=object(), k_buffer=[], v_buffer=[]),
                 SimpleNamespace(kv_buffer=[]),
                 SimpleNamespace(full_kv_pool=SimpleNamespace(k_buffer=[], v_buffer=[]))):
        with pytest.raises(ValueError):
            config.validate_kv_pool(pool)


def test_runtime_guard_disabled_does_not_change_single_node(monkeypatch):
    monkeypatch.delenv(m.PREFIX + "ENABLED", raising=False)
    assert m.validate_multinode_runtime(object(), object()) is None


def test_runtime_guard_checks_enabled_engine(monkeypatch):
    for key, value in environment().items():
        monkeypatch.setenv(key, value)
    for suffix in ("LIFECYCLE", "CUSTOM_STORAGE_ONLY", "HOST_STAGING", "P2D_HOST_STAGING"):
        monkeypatch.setenv("SGLANG_AGENTIC_KV_" + suffix, "true")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND", "memfd")
    for suffix in ("LEDGER_PATH", "STAGING_LEDGER_PATH", "P2D_STAGING_LEDGER_PATH", "METADATA_DIR", "EARLY_CLAIM_DIR"):
        monkeypatch.setenv("SGLANG_AGENTIC_KV_" + suffix, "/shared/dualpd-control/r-001/" + suffix)
    args = SimpleNamespace(tp_size=8, dp_size=1, pp_size=1, nnodes=1,
                           disaggregation_mode="prefill", disaggregation_transfer_backend="nixl")
    config = m.validate_multinode_runtime(args, SimpleNamespace(k_buffer=[], v_buffer=[]))
    assert config.tp_size == 8
    monkeypatch.setenv("SGLANG_AGENTIC_KV_HOST_STAGING", "false")
    assert m.validate_multinode_runtime(args).tp_size == 8
    monkeypatch.setenv("SGLANG_AGENTIC_KV_HOST_STAGING", "true")
    with pytest.raises(ValueError):
        m.validate_multinode_runtime(args, SimpleNamespace(mamba_pool=object()))
    monkeypatch.setenv("SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH", "/dev/shm/wrong-run")
    with pytest.raises(ValueError, match="shared run control"):
        m.validate_multinode_runtime(args)


def test_router_can_watch_shared_control_but_not_start_engine():
    config = m.load_multinode_config(environment(ROLE="router", ENGINE_ID="router-001"))
    assert config.control_poll_interval == 0.1
    with pytest.raises(ValueError, match="router identity"):
        config.validate_server_args(object())


def test_source_host_placement_preserves_single_node(monkeypatch):
    monkeypatch.delenv(m.PREFIX + "ENABLED", raising=False)
    assert m.source_host_placement(8, 7) is None


def test_source_host_placement_uses_d_numa_not_p(monkeypatch):
    for key, value in environment(ROLE="decode", ENGINE_ID="d-node").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TP_NUMA_NODES", "0,0,0,0,1,1,1,1")
    assert m.source_host_placement(8, 7) == (7, [0, 0, 0, 0, 1, 1, 1, 1])
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TP_NUMA_NODES", "0,1")
    with pytest.raises(ValueError, match="NUMA vector"):
        m.source_host_placement(8, 7)


def test_decode_fallback_does_not_query_remote_p_host_capacity(monkeypatch):
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import DecodeKVCacheOffloadManager

    for key, value in environment(ROLE="decode", ENGINE_ID="d-node").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TP_NUMA_NODES", "0,0,0,0,1,1,1,1")
    manager = SimpleNamespace(tp_world_size=8, agentic_host_staging_client=SimpleNamespace(arena_domain=0))
    candidate = {}
    DecodeKVCacheOffloadManager._assign_slow_host_target(manager, candidate)
    assert candidate == {"selected_host_domain": 0, "selected_host_numa_nodes": [0, 0, 0, 0, 1, 1, 1, 1]}
    # Replaying a rank0 decision cannot re-route an already offered snapshot.
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TP_NUMA_NODES", "bad")
    DecodeKVCacheOffloadManager._assign_slow_host_target(manager, candidate)
