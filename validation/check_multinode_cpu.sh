#!/usr/bin/env bash
# CPU-only regression gate; never creates a CUDA context or launches workers.
set -euo pipefail
dualpd_sglang_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dualpd_slime_root="${DUALPD_SLIME_ROOT:-${dualpd_sglang_root}/../slime}"
dualpd_python="${DUALPD_PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=''
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export SGLANG_AGENTIC_MULTINODE_ENABLED=0
unset SGLANG_AGENTIC_CONTROL_ENDPOINT SGLANG_AGENTIC_TP_EVENT_ENDPOINT
export PYTHONPATH="${dualpd_sglang_root}/python:${dualpd_slime_root}/examples/pd:${dualpd_slime_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${dualpd_sglang_root}"
"${dualpd_python}" -m pytest -q \
  python/sglang/srt/disaggregation/test_agentic_multinode*.py \
  python/sglang/srt/disaggregation/test_agentic_minimax_layout.py \
  python/sglang/srt/disaggregation/test_agentic_direct_descriptors.py \
  python/sglang/srt/disaggregation/test_agentic_direct_control.py \
  python/sglang/srt/disaggregation/test_agentic_direct_retirement.py \
  python/sglang/srt/disaggregation/test_agentic_direct_cancel_slots.py \
  python/sglang/srt/disaggregation/test_agentic_direct_early_authorization.py \
  python/sglang/srt/disaggregation/test_agentic_direct_transport_credit.py \
  python/sglang/srt/disaggregation/test_agentic_workset*.py \
  python/sglang/srt/disaggregation/test_agentic_remote_host*.py \
  python/sglang/srt/disaggregation/test_agentic_remote_hybrid.py \
  python/sglang/srt/disaggregation/test_agentic_kv_lifecycle.py \
  python/sglang/srt/disaggregation/test_agentic_tp*.py \
  python/sglang/srt/disaggregation/test_p2d_host_worker_device.py \
  python/sglang/srt/disaggregation/test_agentic_control*.py \
  python/sglang/srt/disaggregation/test_agentic_broker_integration.py \
  python/sglang/srt/disaggregation/test_agentic_lifecycle_control.py \
  python/sglang/srt/disaggregation/test_agentic_host_rpc.py \
  python/sglang/srt/disaggregation/test_agentic_host_control.py \
  python/sglang/srt/disaggregation/test_agentic_host_restore_controller.py \
  python/sglang/srt/disaggregation/test_agentic_host_controller_scheduler.py \
  python/sglang/srt/disaggregation/test_agentic_host_async_prepare.py \
  python/sglang/srt/disaggregation/test_agentic_host_terminal_admission.py \
  python/sglang/srt/disaggregation/test_agentic_host_event_progress.py \
  python/sglang/srt/disaggregation/test_agentic_final_host_cleanup.py \
  python/sglang/srt/disaggregation/test_agentic_slow_congestion.py \
  python/sglang/srt/disaggregation/test_agentic_startup_prewarm.py \
  "${dualpd_slime_root}/examples/pd/tests/test_agentic_early_claim.py" \
  "${dualpd_slime_root}/examples/pd/tests/test_agentic_host_staging.py" \
  "${dualpd_slime_root}/examples/pd/tests/test_p2d_host_staging.py" \
  "${dualpd_slime_root}/examples/pd/tests/test_late_binding_router.py" \
  "${dualpd_slime_root}/examples/pd/tests/test_host_eviction_route.py" \
  "${dualpd_slime_root}/examples/pd/tests/test_socket_ready_router.py" \
  "${dualpd_slime_root}/examples/pd/tests/test_socket_router_cancellation.py"
"${dualpd_python}" -m unittest discover -s "${dualpd_slime_root}/tools/dualpd" -p 'test_*.py'
bash -n "${dualpd_slime_root}/tools/dualpd/multinode.sh"
