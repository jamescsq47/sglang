"""CPU-only tests for the V2 deployment boundary."""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest


def _module():
    path = pathlib.Path(__file__).with_name("agentic_multinode.py")
    spec = importlib.util.spec_from_file_location("agentic_multinode_v2_unit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m = _module()


def environment(**changes):
    values = {
        m.PREFIX + "ENABLED": "1",
        m.PREFIX + "NODE_ID": "node-a",
        m.PREFIX + "ENGINE_ID": "p-a",
        m.PREFIX + "ROLE": "prefill",
        m.PREFIX + "HOST_IP": "10.20.1.2",
        m.PREFIX + "PEER_TP_SIZE": "8",
        m.GROUP_PREFIX + "RUN_ID": "r-001",
        m.GROUP_PREFIX + "GROUP_ID": "p0",
        m.GROUP_PREFIX + "ENDPOINT_GROUP": "p-a",
        m.GROUP_PREFIX + "ENDPOINT_ROLE": "prefill",
        m.GROUP_PREFIX + "PEER_GROUP": "d-a",
        m.GROUP_PREFIX + "PEER_ROLE": "decode",
        m.GROUP_PREFIX + "COORDINATOR_GROUP": "p-a",
        m.GROUP_PREFIX + "SIZE": "8",
        m.GROUP_PREFIX + "ENDPOINT": "tcp://10.20.1.1:17391",
        m.GROUP_PREFIX + "TOKEN": "0123456789abcdef",
    }
    values.update(changes)
    return values


def test_disabled_does_not_inspect_v2_settings():
    assert m.load_multinode_config({}) is None
    assert m.load_multinode_config({m.PREFIX + "ENABLED": "0"}) is None


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_tcp_configuration(tp_size):
    config = m.load_multinode_config(
        environment(
            **{
                m.GROUP_PREFIX + "SIZE": str(tp_size),
                m.PREFIX + "PEER_TP_SIZE": str(tp_size),
            }
        )
    )
    assert config.tp_size == tp_size
    assert config.endpoint == ("10.20.1.1", 17391)
    assert config.group_id == "p0"


@pytest.mark.parametrize(
    "changes",
    [
        {"SGLANG_AGENTIC_GROUP_SIZE": "0"},
        {"SGLANG_AGENTIC_GROUP_SIZE": "16", m.PREFIX + "PEER_TP_SIZE": "16"},
        {m.PREFIX + "PEER_TP_SIZE": "4"},
        {m.PREFIX + "DP_SIZE": "2"},
        {m.PREFIX + "ROLE": "unknown"},
        {m.GROUP_PREFIX + "RUN_ID": "../bad"},
        {m.PREFIX + "HOST_IP": "127.0.0.1"},
        {m.GROUP_PREFIX + "ENDPOINT": "tcp://0.0.0.0:9"},
        {m.GROUP_PREFIX + "ENDPOINT": "not-an-endpoint"},
        {m.GROUP_PREFIX + "TOKEN": "short"},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        m.load_multinode_config(environment(**changes))


def test_endpoint_identity_must_match_engine_and_role():
    with pytest.raises(ValueError, match="endpoint group"):
        m.load_multinode_config(
            environment(**{m.GROUP_PREFIX + "ENDPOINT_GROUP": "another-p"})
        )
    with pytest.raises(ValueError, match="endpoint role"):
        m.load_multinode_config(
            environment(**{m.GROUP_PREFIX + "ENDPOINT_ROLE": "decode"})
        )


@pytest.mark.parametrize(
    "legacy_key",
    [
        m.PREFIX + "CONTROL_ROOT",
        "SGLANG_AGENTIC_KV_LEDGER_PATH",
        "SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH",
        "SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH",
        "SGLANG_AGENTIC_KV_METADATA_DIR",
        "SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR",
        "SGLANG_PD_P_READY_DIR",
    ],
)
def test_runtime_control_files_are_forbidden(legacy_key):
    with pytest.raises(ValueError, match="filesystem control"):
        m.load_multinode_config(environment(**{legacy_key: "/shared/control"}))


def test_real_server_args_and_native_cache_are_checked():
    config = m.load_multinode_config(environment())
    args = SimpleNamespace(
        tp_size=8,
        dp_size=1,
        pp_size=1,
        nnodes=1,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="nixl",
        enable_hierarchical_cache=False,
        hicache_storage_backend=None,
        disaggregation_decode_enable_offload_kvcache=False,
        enable_dp_attention=False,
        speculative_algorithm=None,
    )
    config.validate_server_args(args)
    args.enable_hierarchical_cache = True
    with pytest.raises(ValueError, match="HiCache"):
        config.validate_server_args(args)


def test_complete_mha_and_hybrid_pool_validation():
    config = m.load_multinode_config(environment())
    config.validate_kv_pool(SimpleNamespace(k_buffer=[], v_buffer=[]))
    hybrid = SimpleNamespace(
        full_kv_pool=SimpleNamespace(k_buffer=[], v_buffer=[]),
        mamba_pool=SimpleNamespace(
            mamba_cache=SimpleNamespace(conv=[], temporal=object())
        ),
    )
    config.validate_kv_pool(hybrid)
    with pytest.raises(ValueError):
        config.validate_kv_pool(SimpleNamespace(kv_buffer=[]))
    with pytest.raises(ValueError):
        config.validate_kv_pool(
            SimpleNamespace(
                full_kv_pool=SimpleNamespace(k_buffer=[], v_buffer=[]),
                mamba_pool=SimpleNamespace(mamba_cache=object()),
            )
        )


def test_capabilities_state_no_runtime_filesystem():
    caps = m.capabilities()
    assert caps["control"] == "tcp_in_memory_rank0_authority"
    assert caps["runtime_filesystem_control"] is False
    assert set(caps["paths"]) == {
        "d2p_direct",
        "d2host2p",
        "p2d_direct",
        "p2host2d",
    }


def test_source_host_placement_uses_source_numa(monkeypatch):
    for key, value in environment(
        **{
            m.PREFIX + "ROLE": "decode",
            m.PREFIX + "ENGINE_ID": "d0",
            m.GROUP_PREFIX + "GROUP_ID": "d0",
            m.GROUP_PREFIX + "ENDPOINT_GROUP": "d0",
            m.GROUP_PREFIX + "ENDPOINT_ROLE": "decode",
            m.GROUP_PREFIX + "PEER_GROUP": "p-a",
            m.GROUP_PREFIX + "PEER_ROLE": "prefill",
            m.GROUP_PREFIX + "COORDINATOR_GROUP": "p-a",
        }
    ).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(
        "SGLANG_AGENTIC_KV_TP_NUMA_NODES", "0,0,0,0,1,1,1,1"
    )
    assert m.source_host_placement(8, 3) == (
        3,
        [0, 0, 0, 0, 1, 1, 1, 1],
    )


def test_load_is_pure():
    env = environment()
    before = dict(env)
    m.load_multinode_config(env)
    assert env == before
