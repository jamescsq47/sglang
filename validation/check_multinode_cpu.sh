#!/usr/bin/env bash
# CPU-only regression gate; never creates a CUDA context or launches workers.
set -euo pipefail
dualpd_sglang_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dualpd_slime_root="${DUALPD_SLIME_ROOT:-${dualpd_sglang_root}/../slime}"
dualpd_python="${DUALPD_PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=''
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export SGLANG_AGENTIC_MULTINODE_ENABLED=0
export PYTHONPATH="${dualpd_sglang_root}/python:${dualpd_slime_root}/examples/pd:${dualpd_slime_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${dualpd_sglang_root}"
"${dualpd_python}" -m pytest -q \
  python/sglang/srt/disaggregation/test_agentic_multinode*.py \
  python/sglang/srt/disaggregation/test_agentic_minimax_layout.py \
  python/sglang/srt/disaggregation/test_agentic_remote_host*.py \
  python/sglang/srt/disaggregation/test_agentic_kv_lifecycle.py \
  python/sglang/srt/disaggregation/test_agentic_tp*.py \
  python/sglang/srt/disaggregation/test_agentic_host_async_prepare.py \
  python/sglang/srt/disaggregation/test_agentic_host_event_progress.py \
  python/sglang/srt/disaggregation/test_agentic_final_host_cleanup.py \
  python/sglang/srt/disaggregation/test_agentic_slow_congestion.py \
  "${dualpd_slime_root}/examples/pd/tests/test_agentic_early_claim.py" \
  "${dualpd_slime_root}/examples/pd/tests/test_p2d_host_staging.py" \
  "${dualpd_slime_root}/examples/pd/tests/test_late_binding_router.py"
"${dualpd_python}" -m unittest discover -s "${dualpd_slime_root}/tools/dualpd" -p 'test_*.py'
bash -n "${dualpd_slime_root}/tools/dualpd/multinode.sh"
