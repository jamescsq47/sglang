"""Exercise the real model projection path without loading model weights.

Qwen3.5-27B has V/K=3 and takes the unfused split-view path. Its GDN
Prefill kernel requires contiguous A/B rows (upstream issue #22311).
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def model_methods():
    path = Path(__file__).parents[1] / "models/qwen3_5.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Qwen3_5GatedDeltaNet")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {"forward", "fix_query_key_value_ordering"}]
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
    env = dict(torch=torch, _is_cpu=False, _is_amx_available=False,
               triton=SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env


def projected_ab(batch, tp, value_heads=48):
    env = model_methods()
    nk, nv, dim = 16 // tp, value_heads // tp, 128
    # Values identify each token/head unambiguously; padding between the
    # split views must never be mistaken for the next token's gating values.
    ba = torch.arange(batch*nv*2, dtype=torch.float32).reshape(batch, nv*2) / 100
    qkvz = torch.zeros(batch, (nk+nv)*dim*2)
    owner = SimpleNamespace(num_v_heads=value_heads, num_k_heads=16,
        attn_tp_size=tp, head_k_dim=dim, head_v_dim=dim,
        key_dim=16*dim, value_dim=value_heads*dim,
        _forward_input_proj=lambda _: (qkvz, ba),
        norm=lambda x, _: x, out_proj=lambda x: (x, None))
    owner.fix_query_key_value_ordering = lambda x, y: env[
        "fix_query_key_value_ordering"](owner, x, y)
    captured = {}

    def attention(_, *, mixed_qkv, a, b):
        captured.update(a=a, b=b)
        return torch.zeros(batch, nv, dim)

    def fused_split(*args):
        captured["fastpath"] = True
        return (torch.zeros(batch, (2*nk+nv)*dim),
                torch.zeros(batch, nv, dim),
                ba[:, :nv].contiguous(), ba[:, nv:].contiguous())

    owner.attn = attention
    env["fused_qkvzba_split_reshape_cat_contiguous"] = fused_split
    env["forward"](owner, torch.empty(0), None)
    return captured, ba


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 2, 17])
def test_27b_fallback_passes_correct_contiguous_ab(tp, batch):
    captured, ba = projected_ab(batch, tp)
    assert not captured.get("fastpath")
    nv = 48 // tp
    for name, expected in zip(("b", "a"), ba.split([nv, nv], dim=-1)):
        value = captured[name]
        assert value.is_contiguous(), (name, value.shape, value.stride())
        # The production Prefill kernel uses row*NUM_HEADS, not stride(0).
        read_by_kernel = value.as_strided(value.shape, (nv, 1))
        torch.testing.assert_close(read_by_kernel, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp", [1, 2])
def test_9b_keeps_existing_fused_projection_path(tp):
    captured, _ = projected_ab(17, tp, value_heads=32)
    assert captured["fastpath"]


@pytest.mark.skipif(os.getenv("PD_GDN_CUDA_PROBE") != "1", reason="explicit GPU probe only")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_cuda_gating_with_27b_tp2_projection(dtype):
    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating

    captured, ba = projected_ab(17, 2)
    ba = ba.to(device="cuda", dtype=dtype)
    b_ref, a_ref = (v.contiguous() for v in ba.split([24, 24], dim=-1))
    a = captured["a"].to(device="cuda", dtype=dtype)
    b = captured["b"].to(device="cuda", dtype=dtype)
    log_a = torch.linspace(-1, 1, 24, device="cuda")
    bias = torch.linspace(-0.5, 0.5, 24, device="cuda")
    expected = fused_gdn_gating(log_a, a_ref, b_ref, bias)
    actual = fused_gdn_gating(log_a, a, b, bias)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    reference_g = -log_a.exp() * torch.nn.functional.softplus(a.float() + bias)
    reference_beta = b.float().sigmoid().to(dtype).float()
    torch.testing.assert_close(actual[0][0], reference_g, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual[1][0], reference_beta, rtol=2e-6, atol=2e-6)
    unsafe_b, unsafe_a = ba.split([24, 24], dim=-1)
    unsafe = fused_gdn_gating(log_a, unsafe_a, unsafe_b, bias)
    errors = [float((x-y).abs().max()) for x, y in zip(unsafe, expected)]
    print("old split-view max errors", dtype, errors)
    assert errors[0] > 0.1 and errors[1] > 0.01
