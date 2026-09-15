#!/usr/bin/env bash
# Isolated TP1 2P:2D transport smoke; NOT a serving performance benchmark.
set -euo pipefail
ENGINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PD_ROOT=/homes/siqic/slime/examples/pd
export RUN_DIR="${RUN_DIR:?Set a fresh RUN_DIR}"
[[ ! -e "${RUN_DIR}/ready" ]] || { echo 'RUN_DIR already used' >&2; exit 2; }
mkdir -p "${RUN_DIR}/logs"
export PD_ENV_BIN=/homes/siqic/anaconda3/envs/pd_mamba_baseline/bin
export PATH="${PD_ENV_BIN}:${PATH}"
export SGLANG_OVERLAY_ROOT="${ENGINE_ROOT}/python"
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${PD_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export NIXL_PLUGIN_DIR=/tmp/pd-runtime/nixl-132-pr1987-plugin
export PD_RUN_QWEN_SCRIPT="${ENGINE_ROOT}/validation/run_pd_servers.sh"
control_dir="$(mktemp -d /dev/shm/pd-host-event-smoke.XXXXXX)"
export PD_P_READY_DIR="${control_dir}/ready"
mkdir -p "${PD_P_READY_DIR}"
ln -s "${PD_P_READY_DIR}" "${RUN_DIR}/ready"
export SGLANG_AGENTIC_KV_LEDGER_PATH="${control_dir}/ledger.json"
export SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH="${control_dir}/host.json"
export SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH="${control_dir}/p2d-host.json"
export SGLANG_AGENTIC_KV_METADATA_DIR="${PD_P_READY_DIR}/snapshot-metadata"
unset SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_DIR
export LOCAL_GPUS='' LOCAL_PORTS='' LOCAL_ROUTER_PORT=''
export SGLANG_PD_ABLATION_P2D_PREBIND=false
export SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=false
export MODEL_PATH=/homes/siqic/Qwen3.5-9B
export PREFILL_GPUS='0 6' DECODE_GPUS='1 7'
export PREFILL_GPU_GROUPS='0;6' DECODE_GPU_GROUPS='1;7'
export PREFILL_TP_SIZE=1 DECODE_TP_SIZE=1
export PREFILL_PORTS='29400 29420' DECODE_PORTS='29401 29421'
export BOOTSTRAP_PORT=29402 BOOTSTRAP_PORTS='29402 29422'
export ROUTER_PORT=29403 ROUTER_PROMETHEUS_PORT=29404
export AGENTIC_DIRECT_BASE_PORT=29500
export PD_PREFILL_NCCL_PORT_BASE=29510 PD_DECODE_NCCL_PORT_BASE=29512
export PD_LATE_BINDING=1 PD_LATE_BIND_NUMA_DOMAINS=1
export SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS=1
export SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1
export PD_SERVE_ONLY=1 PD_SKIP_SEARCH=1 PD_PARALLEL_BOOTSTRAP=1
export MEM_FRACTION_STATIC=0.80 DECODE_MEM_FRACTION_STATICS='0.80 0.80'
export MAX_CONTEXT_LENGTH=32768 MAX_RESPONSE_LENGTH=256
export PREFILL_CHUNKED_PREFILL_SIZE=8192 PREFILL_MAX_PREFILL_TOKENS=8192
export PD_PAGE_SIZE=64 MAMBA_TRACK_INTERVAL=64 MAMBA_FULL_MEMORY_RATIO=0.5
export PD_DETERMINISTIC_INFERENCE=1 PD_SERVER_RANDOM_SEED=2026
export SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT=true
export SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED=true
export SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION=true
export SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH=false SGLANG_AGENTIC_KV_DEBUG_DIGEST=0
export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB=16
export SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB=8
export SGLANG_AGENTIC_KV_P2D_HOST_STAGING=true
export SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB=48
export SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB=8
export SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA=1
export SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER=1
export SGLANG_AGENTIC_KV_REGISTER_PREWARM_DIR="${control_dir}/host-register-prewarm"
export SGLANG_AGENTIC_KV_REGISTER_PREWARM_TIMEOUT_SECONDS=600
export SGLANG_AGENTIC_KV_RELAY_ENABLED=false
export SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD=1
export SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT=1
export SGLANG_AGENTIC_KV_EARLY_CLAIM_POST_TIMEOUT=1
export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=false
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE=false
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH=32
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW=32
export SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT=4
export SGLANG_AGENTIC_KV_P_H2D_DECOUPLED=true
export SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS="${SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS:-true}"
export SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS=4096
export SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS=4096
export SGLANG_AGENTIC_KV_D2H_ACTIVE_SNAPSHOTS=4
export SGLANG_PD_LATE_BIND_MAX_PREFILL_INFLIGHT=32
export SGLANG_PD_P_READY_BACKPRESSURE_MODE=disabled
export SGLANG_AGENTIC_KV_LIFECYCLE=true
export PD_RAW_REQUEST_LOG_DIR="${RUN_DIR}/raw"
pipeline_pid=''
client_pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${client_pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  for pid in "${client_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  if [[ -n "${pipeline_pid}" ]]; then
    kill -TERM "${pipeline_pid}" 2>/dev/null || true
    # Let the existing supervisor perform dependency-ordered shutdown/reaping.
    wait "${pipeline_pid}" 2>/dev/null || true
  fi
  cp -a "${control_dir}" "${RUN_DIR}/control-final"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
env | LC_ALL=C sort | rg '^(SGLANG_|PD_|PREFILL_|DECODE_|MODEL_PATH=|MEM_FRACTION_STATIC=|MAMBA_|MAX_CONTEXT_LENGTH=|NIXL_PLUGIN_DIR=)' > "${RUN_DIR}/launch-environment.txt"
bash "${PD_ROOT}/scripts/new_method/internal/run_agentic_pipeline.sh" > "${RUN_DIR}/services.log" 2>&1 &
pipeline_pid=$!
deadline=$((SECONDS + 900))
until rg -q 'PD services ready;' "${RUN_DIR}/services.log"; do
  kill -0 "${pipeline_pid}" || { echo 'service startup failed' >&2; exit 1; }
  (( SECONDS < deadline )) || { echo 'startup deadline exceeded' >&2; exit 1; }
  sleep 1
done
for port in 29400 29420 29401 29421; do
  curl --fail --max-time 10 -s "http://127.0.0.1:${port}/metrics" > "${RUN_DIR}/metrics-before-${port}.txt"
done
for ((i=0; i<${TEST_CLIENTS:-8}; i++)); do
  CUDA_VISIBLE_DEVICES='' python "${ENGINE_ROOT}/validation/check_multiturn.py" \
    --url http://127.0.0.1:29403 --run-dir "${RUN_DIR}" --long-output \
    --archive-repeats 128 > "${RUN_DIR}/logs/client-${i}.log" 2>&1 &
  client_pids+=("$!")
done
status=0
for pid in "${client_pids[@]}"; do wait "${pid}" || status=1; done
client_pids=()
python - "${RUN_DIR}" "${TEST_CLIENTS:-8}" <<'PY' || status=1
import json, sys
from pathlib import Path
files = list(Path(sys.argv[1]).glob('multiturn-*.json'))
rows = [r for f in files for r in json.loads(f.read_text())]
expected = int(sys.argv[2]) * 6
assert len(files) == int(sys.argv[2]) and len(rows) == expected, (len(files), len(rows))
assert all(r.get('exact_output_match') is True for r in rows), 'reference token mismatch/incomplete'
print(f'Exact token comparison passed: {expected}/{expected}', flush=True)
PY
sleep 10
for port in 29400 29420 29401 29421; do
  curl --fail --max-time 10 -s "http://127.0.0.1:${port}/metrics" > "${RUN_DIR}/metrics-after-${port}.txt"
done
cp -a "${control_dir}" "${RUN_DIR}/control-before-stop"
echo "Transport clients finished, status=${status}; stopping owned services"
exit "${status}"
