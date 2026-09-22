#!/usr/bin/env bash
set -euo pipefail

# CPU-only preflight for the P->D background retirement boundary. No model or
# server is started; use the matching environment on the execution node.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PD_MULTI_NODE_PYTHON:-python3}"

cd "$repo_dir"
"$python_bin" -m py_compile \
  python/sglang/srt/disaggregation/agentic_workset_ledger.py \
  python/sglang/srt/disaggregation/agentic_workset_native.py \
  python/sglang/srt/disaggregation/agentic_workset_broker.py \
  python/sglang/srt/disaggregation/prefill.py \
  python/sglang/srt/managers/scheduler.py
"$python_bin" -m pytest -q \
  python/sglang/srt/disaggregation/test_agentic_p2d_native_release.py \
  python/sglang/srt/disaggregation/test_agentic_tp_p2d_release.py \
  python/sglang/srt/disaggregation/test_agentic_tp_p2d_control.py \
  python/sglang/srt/disaggregation/test_agentic_workset_native.py \
  python/sglang/srt/disaggregation/test_agentic_workset_runtime.py \
  python/sglang/srt/disaggregation/test_agentic_workset_abort.py
