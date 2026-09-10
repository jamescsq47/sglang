#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="/homes/siqic/slime/examples/pd/scripts/new_method/internal"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd -- "${PD_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(dirname -- "${REPO_ROOT}")"
cd "${PD_DIR}"

# Use the node-local, byte-identical model replica by default.  Keeping model
# reads off /homes avoids the shared NFS thundering herd when 1P+3D start.
MODEL_PATH="${MODEL_PATH:-/dataset/model/qwen3/Qwen3-8B}"
MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-}"
MODEL_TOOL_CALL_PARSER="${MODEL_TOOL_CALL_PARSER:-}"
# Opt-in raw request/response capture, isolated per engine and experiment.
PD_RAW_REQUEST_LOG_DIR="${PD_RAW_REQUEST_LOG_DIR:-}"
MATH_DATA="${MATH_DATA:-${WORKSPACE_ROOT}/data/dapo-math-17k/dapo-math-17k.jsonl}"
QA_DATA="${QA_DATA:-${WORKSPACE_ROOT}/data/browsecomp/bc_train.jsonl}"
WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-}"
PREFILL_GPU="${PREFILL_GPU:-0}"
PREFILL_GPUS="${PREFILL_GPUS:-${PREFILL_GPU}}"
PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
DECODE_GPU="${DECODE_GPU:-1}"
DECODE_GPUS="${DECODE_GPUS:-${DECODE_GPU}}"
DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-1}"
SEARCH_GPU="${SEARCH_GPU:-2}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
PREFILL_PORTS="${PREFILL_PORTS:-${PREFILL_PORT}}"
DECODE_PORT="${DECODE_PORT:-30001}"
DECODE_PORTS="${DECODE_PORTS:-${DECODE_PORT}}"
LOCAL_GPUS="${LOCAL_GPUS:-}"
LOCAL_PORTS="${LOCAL_PORTS:-}"
LOCAL_ROUTER_PORT="${LOCAL_ROUTER_PORT:-}"
ROUTER_PORT="${ROUTER_PORT:-30002}"
ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-29000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-${BOOTSTRAP_PORT}}"
SEARCH_PORT="${SEARCH_PORT:-8000}"
PD_SKIP_SEARCH="${PD_SKIP_SEARCH:-0}"
SEARCH_START_AFTER_MODELS="${SEARCH_START_AFTER_MODELS:-false}"
SEARCH_SERVER_EMBEDDING_CACHE="${SEARCH_SERVER_EMBEDDING_CACHE:-${REPO_ROOT}/examples/artifacts/search/corpus_embeddings.pkl}"
ARRIVAL_RATE="${ARRIVAL_RATE:-0.05}"
ARRIVAL_RATES="${ARRIVAL_RATES:-${ARRIVAL_RATE}}"
ARRIVAL_DISTRIBUTION="${ARRIVAL_DISTRIBUTION:-poisson}"
DISPATCH_POLICY="${DISPATCH_POLICY:-random}"
SCHEDULE_FILE="${SCHEDULE_FILE:-}"
PRESERVE_SOURCE_ORDER="${PRESERVE_SOURCE_ORDER:-false}"
DYNAMIC_LOOKBACK_SECONDS="${DYNAMIC_LOOKBACK_SECONDS:-12}"
DYNAMIC_RECENT_SECONDS="${DYNAMIC_RECENT_SECONDS:-10}"
DYNAMIC_HISTORY_START_SECONDS="${DYNAMIC_HISTORY_START_SECONDS:-20}"
DYNAMIC_HISTORY_END_SECONDS="${DYNAMIC_HISTORY_END_SECONDS:-60}"
DYNAMIC_PREFILL_CAPACITY_TPS="${DYNAMIC_PREFILL_CAPACITY_TPS:-9000}"
DYNAMIC_DECODE_CAPACITY_TPS="${DYNAMIC_DECODE_CAPACITY_TPS:-1100}"
DYNAMIC_DECODE_TARGET_ACTIVE="${DYNAMIC_DECODE_TARGET_ACTIVE:-30}"
DYNAMIC_HYSTERESIS="${DYNAMIC_HYSTERESIS:-0.12}"
DYNAMIC_MAX_IMBALANCE="${DYNAMIC_MAX_IMBALANCE:-8}"
DYNAMIC_MAX_CONSECUTIVE="${DYNAMIC_MAX_CONSECUTIVE:-3}"
REQUESTS="${REQUESTS:-20}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
MAX_INFLIGHT="${MAX_INFLIGHT:-16}"
MATH_RATIO="${MATH_RATIO:-0.5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
METRICS_INTERVAL="${METRICS_INTERVAL:-2}"
SEED="${SEED:-2026}"
MEASUREMENT_DURATION_SECONDS="${MEASUREMENT_DURATION_SECONDS:-}"
CLOSED_LOOP="${CLOSED_LOOP:-0}"
CLOSED_LOOP_WARMUP_MIN_SECONDS="${CLOSED_LOOP_WARMUP_MIN_SECONDS:-300}"
CLOSED_LOOP_WARMUP_COMPLETIONS="${CLOSED_LOOP_WARMUP_COMPLETIONS:-128}"
CLOSED_LOOP_RECENT_SECONDS="${CLOSED_LOOP_RECENT_SECONDS:-120}"
CLOSED_LOOP_MAX_WARMUP_SECONDS="${CLOSED_LOOP_MAX_WARMUP_SECONDS:-1800}"
CLOSED_LOOP_MEASUREMENT_SECONDS="${CLOSED_LOOP_MEASUREMENT_SECONDS:-300}"
ROUTER_HEALTH_TIMEOUT_SECS="${ROUTER_HEALTH_TIMEOUT_SECS:-60}"
ROUTER_HEALTH_FAILURE_THRESHOLD="${ROUTER_HEALTH_FAILURE_THRESHOLD:-10}"
MAX_EXISTING_GPU_MEMORY_MB="${MAX_EXISTING_GPU_MEMORY_MB:-1024}"
# Fair-comparison default: PD workers use the same per-physical-GPU static
# memory fraction as the corresponding colocated run. Topology wrappers must
# still override GPUs that host another service.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
PREFILL_CHUNKED_PREFILL_SIZE="${PREFILL_CHUNKED_PREFILL_SIZE:-8192}"
PREFILL_MAX_PREFILL_TOKENS="${PREFILL_MAX_PREFILL_TOKENS:-8192}"
DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-40960}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-36864}"
PD_MAX_TRANSFER_INFLIGHT="${PD_MAX_TRANSFER_INFLIGHT:-0}"
PD_P_READY_DIR="${PD_P_READY_DIR:-}"
PD_LATE_BINDING="${PD_LATE_BINDING:-0}"
PD_LATE_BIND_READY_TIMEOUT_S="${PD_LATE_BIND_READY_TIMEOUT_S:-600}"
PD_LATE_BIND_RESERVATION_TIMEOUT_S="${PD_LATE_BIND_RESERVATION_TIMEOUT_S:-120}"
PD_LATE_BIND_DECODE_HEADROOM_TOKENS="${PD_LATE_BIND_DECODE_HEADROOM_TOKENS:-512}"
PD_LATE_BIND_WAIT_FOR_FEASIBLE="${PD_LATE_BIND_WAIT_FOR_FEASIBLE:-1}"
PD_LATE_BIND_LOAD_CACHE_TTL_S="${PD_LATE_BIND_LOAD_CACHE_TTL_S:-0.05}"
PD_LATE_BIND_NO_CAPACITY_POLL_S="${PD_LATE_BIND_NO_CAPACITY_POLL_S:-0.01}"
PD_LATE_BIND_SOFT_RESERVATION_DELAY_S="${PD_LATE_BIND_SOFT_RESERVATION_DELAY_S:-30.0}"
PD_LATE_BIND_SOFT_RESERVATION_MIN_TOKENS="${PD_LATE_BIND_SOFT_RESERVATION_MIN_TOKENS:-20000}"
PD_LATE_BIND_SOFT_RESERVATION_FORCE_AFTER_S="${PD_LATE_BIND_SOFT_RESERVATION_FORCE_AFTER_S:-120.0}"
# SGLang 0.5.10 in the pd environment currently returns 500 from /v1/loads
# because of its included-router middleware. /get_load is scheduler-native and
# contains physical plus queued token demand, so use it until that bug is fixed.
PD_LATE_BIND_FORCE_LEGACY_LOADS="${PD_LATE_BIND_FORCE_LEGACY_LOADS:-1}"
PD_HICACHE_STORAGE_BACKEND="${PD_HICACHE_STORAGE_BACKEND:-}"
if [[ -n "${PD_HICACHE_STORAGE_BACKEND}" ]]; then
  echo "new_method does not use a native HiCache/Mooncake storage backend" >&2
  exit 2
fi
case "${SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY:-}" in
  1|true|TRUE|yes|YES|on|ON) ;;
  *)
    echo "new_method refuses stock HiCache/Mooncake policy; " \
      "SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY must be true" >&2
    exit 2
    ;;
esac
PD_ENABLE_DECODE_OFFLOAD_KVCACHE="${PD_ENABLE_DECODE_OFFLOAD_KVCACHE:-0}"
PD_HICACHE_SIZE_GB="${PD_HICACHE_SIZE_GB:-0}"
PD_PREFILL_HICACHE_SIZE_GB="${PD_PREFILL_HICACHE_SIZE_GB:-${PD_HICACHE_SIZE_GB}}"
PD_DECODE_HICACHE_SIZE_GB="${PD_DECODE_HICACHE_SIZE_GB:-${PD_HICACHE_SIZE_GB}}"
PD_PREFILL_HICACHE_WRITE_POLICY="${PD_PREFILL_HICACHE_WRITE_POLICY:-write_through}"
PD_HICACHE_STORAGE_PREFETCH_POLICY="${PD_HICACHE_STORAGE_PREFETCH_POLICY:-best_effort}"
PD_HICACHE_PREFETCH_THRESHOLD="${PD_HICACHE_PREFETCH_THRESHOLD:-256}"
PD_HICACHE_STORAGE_EXTRA_CONFIG="${PD_HICACHE_STORAGE_EXTRA_CONFIG:-}"
PD_HICACHE_MEM_LAYOUT="${PD_HICACHE_MEM_LAYOUT:-layer_first}"
PD_PAGE_SIZE="${PD_PAGE_SIZE:-1}"
PD_CORRECTNESS_CAPTURE_OUTPUT="${PD_CORRECTNESS_CAPTURE_OUTPUT:-}"
PD_CORRECTNESS_CAPTURE_LABEL="${PD_CORRECTNESS_CAPTURE_LABEL:-pd-capture}"
PD_SERVE_ONLY="${PD_SERVE_ONLY:-0}"
PD_DETERMINISTIC_INFERENCE="${PD_DETERMINISTIC_INFERENCE:-0}"
PD_SERVER_RANDOM_SEED="${PD_SERVER_RANDOM_SEED:-2026}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-}"
# Keep reverse NIXL listeners out of Linux's usual ephemeral range.
# The legacy offset remains available when explicitly supplied.
AGENTIC_DIRECT_BASE_PORT="${AGENTIC_DIRECT_BASE_PORT:-61000}"
AGENTIC_DIRECT_PORT_OFFSET="${AGENTIC_DIRECT_PORT_OFFSET:-}"
INFERENCE_ENTRY="${INFERENCE_ENTRY:-${PD_DIR}/inference.py}"
[[ -f "${INFERENCE_ENTRY}" ]] || {
  echo "missing inference entry: ${INFERENCE_ENTRY}" >&2
  exit 1
}
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs/${RUN_ID}}"
PD_HICACHE_STORAGE_DIR="${PD_HICACHE_STORAGE_DIR:-${RUN_DIR}/hicache}"

mkdir -p "${RUN_DIR}/logs"
if [[ -n "${PD_HICACHE_STORAGE_BACKEND}" ]]; then
  mkdir -p "${PD_HICACHE_STORAGE_DIR}"
fi
export PYTHONPATH="${PD_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"
export LOCAL_SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}"
# Router backend connections are pooled.  Keep SGLang's server-side lifetime
# comfortably above the Router's 30-second client idle lifetime so a saturated
# Router never reuses a socket that Uvicorn has already reaped.
export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-120}"

if [[ "${PD_ENABLE_DECODE_OFFLOAD_KVCACHE}" == "1" ]]; then
  echo "new_method permanently forbids native Decode KV offload" >&2
  exit 1
fi
if [[ "${PD_LATE_BINDING}" == "1" && -z "${PD_P_READY_DIR}" ]]; then
  echo "PD_LATE_BINDING=1 requires a shared PD_P_READY_DIR" >&2
  exit 1
fi

prefill_hicache_args=()
decode_hicache_args=()
deterministic_args=()
mamba_args=()
if [[ "${MODEL_PATH}" == *"Qwen3.5"* ]]; then
  mamba_args+=(--mamba-scheduler-strategy extra_buffer)
fi
model_parser_args=()
if [[ "${PD_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  deterministic_args+=(
    --enable-deterministic-inference
    --attention-backend triton
    --random-seed "${PD_SERVER_RANDOM_SEED}"
  )
fi
if [[ -n "${MAMBA_TRACK_INTERVAL}" ]]; then
  mamba_args+=(--mamba-track-interval "${MAMBA_TRACK_INTERVAL}")
fi
if [[ -n "${MODEL_REASONING_PARSER}" ]]; then
  model_parser_args+=(--reasoning-parser "${MODEL_REASONING_PARSER}")
fi
if [[ -n "${MODEL_TOOL_CALL_PARSER}" ]]; then
  model_parser_args+=(--tool-call-parser "${MODEL_TOOL_CALL_PARSER}")
fi
storage_extra_config="${PD_HICACHE_STORAGE_EXTRA_CONFIG}"
if [[ -z "${storage_extra_config}" && "${PD_HICACHE_STORAGE_BACKEND}" == "file" ]]; then
  storage_extra_config="{\"prefetch_threshold\":${PD_HICACHE_PREFETCH_THRESHOLD}}"
fi
hicache_metadata_args=(
  --pd-hicache-storage-backend "${PD_HICACHE_STORAGE_BACKEND}"
  --pd-hicache-storage-dir "${PD_HICACHE_STORAGE_DIR}"
  --pd-hicache-storage-prefetch-policy "${PD_HICACHE_STORAGE_PREFETCH_POLICY}"
  --pd-hicache-prefetch-threshold "${PD_HICACHE_PREFETCH_THRESHOLD}"
)
if [[ -n "${PD_HICACHE_STORAGE_BACKEND}" ]]; then
  prefill_hicache_args+=(
    --hicache-mem-layout "${PD_HICACHE_MEM_LAYOUT}"
    --hicache-storage-backend "${PD_HICACHE_STORAGE_BACKEND}"
  )
  if [[ -n "${storage_extra_config}" ]]; then
    prefill_hicache_args+=(--hicache-storage-backend-extra-config "${storage_extra_config}")
  fi
  if (( PD_PREFILL_HICACHE_SIZE_GB > 0 )); then
    prefill_hicache_args+=(--hicache-size "${PD_PREFILL_HICACHE_SIZE_GB}")
  fi
fi
if [[ -n "${PD_HICACHE_STORAGE_BACKEND}" ]]; then
  decode_hicache_args+=(
    --hicache-mem-layout "${PD_HICACHE_MEM_LAYOUT}"
    --hicache-storage-backend "${PD_HICACHE_STORAGE_BACKEND}"
  )
  if [[ -n "${storage_extra_config}" ]]; then
    decode_hicache_args+=(--hicache-storage-backend-extra-config "${storage_extra_config}")
  fi
fi

pids=()
frontend_pids=()
cleanup_started=0

process_group_has_live_members() {
  local pgid="$1"
  # Inspect every task, not only thread-group leaders.  A native service can
  # briefly have a zombie leader while worker threads still own sockets or a
  # CUDA context; process-only `ps -eo` would incorrectly declare it gone.
  ps -eLo pgid=,stat= | awk -v target="${pgid}" '
    $1 == target && $2 !~ /^Z/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

cleanup() {
  local pid index alive task_id
  local frontend_graceful_seconds="${PD_FRONTEND_GRACEFUL_SHUTDOWN_SECONDS:-10}"
  local graceful_seconds="${PD_SERVICE_GRACEFUL_SHUTDOWN_SECONDS:-120}"
  local kill_wait_seconds="${PD_SERVICE_KILL_WAIT_SECONDS:-30}"
  local final_kill_wait_seconds="${PD_SERVICE_FINAL_KILL_WAIT_SECONDS:-60}"
  local deadline
  (( cleanup_started == 1 )) && return
  cleanup_started=1
  trap - INT TERM

  # Stop the HTTP frontends before their P/D dependencies.  A fixed-duration
  # closed-loop run intentionally ends with live agents; terminating P/D at
  # the same instant makes the Router report those expected cancellations as
  # ServerDisconnectedError and then wait for its full graceful timeout.
  # Keeping P/D alive through this short frontend drain preserves the current
  # snapshot owner until the client-facing request has either unwound or the
  # Router itself is retired.
  for pid in "${frontend_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
  done
  deadline=$((SECONDS + frontend_graceful_seconds))
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${frontend_pids[@]}"; do
      process_group_has_live_members "${pid}" && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for pid in "${frontend_pids[@]}"; do
    if process_group_has_live_members "${pid}"; then
      echo "Frontend process group ${pid} exceeded ${frontend_graceful_seconds}s drain; stopping it before P/D" >&2
      kill -KILL -- "-${pid}" 2>/dev/null || true
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done

  # With every frontend gone, no new requests or transfers can be created.
  # Address the whole setsid process group even when its launch_server parent
  # has already exited but a scheduler child remains.
  for ((index=${#pids[@]} - 1; index >= 0; index--)); do
    pid="${pids[index]}"
    kill -TERM -- "-${pid}" 2>/dev/null || true
    # Some Python entrypoints call setsid/daemonize internally after the shell
    # records their launcher PID.  Signal the exact leader as well as the
    # original process group so a reparented Router cannot survive cleanup.
    kill -TERM "${pid}" 2>/dev/null || true
  done
  deadline=$((SECONDS + graceful_seconds))
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${pids[@]}"; do
      process_group_has_live_members "${pid}" && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done

  # SIGKILL is a last resort, not the normal 15-second shutdown path.  Emit
  # enough state to diagnose any uninterruptible CUDA/UVM thread before using
  # it, then wait for the complete process group rather than only its leader.
  for pid in "${pids[@]}"; do
    if process_group_has_live_members "${pid}"; then
      echo "Process group ${pid} exceeded ${graceful_seconds}s graceful shutdown" >&2
      ps -eo pid=,ppid=,pgid=,stat=,wchan:32=,cmd= \
        | awk -v target="${pid}" '$3 == target { print }' >&2 || true
      kill -KILL -- "-${pid}" 2>/dev/null || true
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  deadline=$((SECONDS + kill_wait_seconds))
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${pids[@]}"; do
      process_group_has_live_members "${pid}" && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done

  # A CUDA scheduler can need several seconds to unwind after SIGKILL even
  # after launch_server has already exited and reparented it to PID 1.  Do not
  # return from the experiment while such a child still owns a GPU.  Repeat
  # SIGKILL against both the exact process group and its enumerated members,
  # then give the kernel a separate final reap window.
  deadline=$((SECONDS + final_kill_wait_seconds))
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${pids[@]}"; do
      if process_group_has_live_members "${pid}"; then
        alive=1
        kill -KILL -- "-${pid}" 2>/dev/null || true
        kill -KILL "${pid}" 2>/dev/null || true
        # `pid=` is the thread-group ID even under `ps -L`.  Use `lwp=` so a
        # scheduler whose leader is already a zombie but whose CUDA worker
        # thread is still alive is explicitly signalled as well.  This exact
        # state otherwise leaves an orphaned CUDA context owned by PID 1.
        while read -r task_id; do
          [[ -n "${task_id}" ]] && kill -KILL "${task_id}" 2>/dev/null || true
        done < <(ps -eLo lwp=,pgid= | awk -v target="${pid}" '$2 == target { print $1 }')
      fi
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

port_is_open() {
  python - "$1" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

check_gpu_idle() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( used > MAX_EXISTING_GPU_MEMORY_MB )); then
    echo "GPU ${gpu} already uses ${used} MiB; limit is ${MAX_EXISTING_GPU_MEMORY_MB} MiB" >&2
    echo "refusing to interfere with an existing workload" >&2
    exit 1
  fi
}

gpu_numa_node() {
  local gpu="$1"
  local bus
  local node
  # Query this GPU's PCI function directly instead of parsing the complete
  # topology matrix.  On large NVSwitch hosts ``nvidia-smi topo -m`` can block
  # indefinitely while the ordinary per-GPU NVML query and sysfs remain
  # healthy.  NUMA affinity is a PCI-device property, so sysfs is also the
  # authoritative and cheaper source here.
  bus="$(
    timeout 5s nvidia-smi --id="${gpu}" --query-gpu=pci.bus_id \
      --format=csv,noheader,nounits 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -d ' '
  )" || true
  # NVML commonly prints an eight-digit PCI domain, while Linux sysfs uses
  # four digits for the same function (00000000:07:00.0 -> 0000:07:00.0).
  bus="${bus#0000}"
  if [[ -n "${bus}" && -r "/sys/bus/pci/devices/${bus}/numa_node" ]]; then
    node="$(<"/sys/bus/pci/devices/${bus}/numa_node")"
  else
    node=""
  fi
  if [[ "${node}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${node}"
  else
    printf '0\n'
  fi
}

wait_http() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local timeout="${4:-600}"
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited during startup; see ${RUN_DIR}/logs/${name}.log" >&2
      return 1
    fi
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "${name} ready: ${url}"
      return 0
    fi
    sleep 2
  done
  echo "${name} did not become ready within ${timeout}s" >&2
  return 1
}

wait_registered_host_prewarm() {
  case "${SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER:-0}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    *) return 0 ;;
  esac
  local root="${SGLANG_AGENTIC_KV_REGISTER_PREWARM_DIR:?missing Host prewarm directory}"
  local timeout="${SGLANG_AGENTIC_KV_REGISTER_PREWARM_TIMEOUT_SECONDS:-1800}"
  local expected="$(((${#prefill_gpu_groups[@]} + ${#decode_gpu_groups[@]}) * PREFILL_TP_SIZE))"
  local complete_count failed_count pid
  local deadline=$((SECONDS + timeout))
  mkdir -p "${root}/arenas" "${root}/complete" "${root}/failed"
  : >"${root}/start.tmp"
  mv -f "${root}/start.tmp" "${root}/start"
  echo "Host registration prewarm started; waiting for ${expected} P/D CUDA contexts"
  while (( SECONDS < deadline )); do
    failed_count="$(find "${root}/failed" -maxdepth 1 -type f -name '*.json' | wc -l)"
    if (( failed_count > 0 )); then
      echo "Host registration prewarm failed:" >&2
      find "${root}/failed" -maxdepth 1 -type f -name '*.json' -print -exec cat {} \; >&2
      return 1
    fi
    complete_count="$(find "${root}/complete" -maxdepth 1 -type f -name '*.json' | wc -l)"
    if (( complete_count == expected )); then
      python - "${root}" "${RUN_DIR}/host_register_prewarm.json" "${expected}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted((root / "complete").glob("*.json"))
]
report = {
    "status": "complete",
    "participant_count": len(records),
    "expected_participant_count": int(sys.argv[3]),
    "registered_bytes_sum_across_cuda_contexts": sum(
        int(item["registered_bytes"]) for item in records
    ),
    "max_elapsed_seconds": max(
        (float(item["elapsed_seconds"]) for item in records), default=0.0
    ),
    "participants": records,
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
      echo "Host registration prewarm complete for all ${expected} P/D CUDA contexts"
      return 0
    fi
    if (( complete_count > expected )); then
      echo "Host registration prewarm produced too many completion records: ${complete_count}/${expected}" >&2
      return 1
    fi
    for pid in "${prefill_pids[@]}" "${decode_pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "A P/D service exited during Host registration prewarm" >&2
        return 1
      fi
    done
    sleep 2
  done
  echo "Host registration prewarm timed out after ${timeout}s" >&2
  find "${root}" -maxdepth 2 -type f -print >&2
  return 1
}

read -r -a prefill_ports <<<"${PREFILL_PORTS}"
read -r -a bootstrap_ports <<<"${BOOTSTRAP_PORTS}"
read -r -a decode_ports <<<"${DECODE_PORTS}"
read -r -a decode_mem_fraction_statics <<<"${DECODE_MEM_FRACTION_STATICS}"
read -r -a local_gpus <<<"${LOCAL_GPUS}"
read -r -a local_ports <<<"${LOCAL_PORTS}"
decode_ports_csv="$(IFS=,; echo "${decode_ports[*]}")"
prefill_ports_csv="$(IFS=,; echo "${prefill_ports[*]}")"
local_ports_csv="$(IFS=,; echo "${local_ports[*]}")"

# A semicolon separates logical engines and a comma separates TP ranks inside
# one engine.  Legacy whitespace-delimited *_GPUS remains byte-for-byte
# compatible and is converted to singleton TP=1 groups only when the new
# variables are absent.
if [[ -n "${PREFILL_GPU_GROUPS}" ]]; then
  IFS=';' read -r -a prefill_gpu_groups <<<"${PREFILL_GPU_GROUPS}"
else
  read -r -a legacy_prefill_gpus <<<"${PREFILL_GPUS}"
  prefill_gpu_groups=("${legacy_prefill_gpus[@]}")
fi
if [[ -n "${DECODE_GPU_GROUPS}" ]]; then
  IFS=';' read -r -a decode_gpu_groups <<<"${DECODE_GPU_GROUPS}"
else
  read -r -a legacy_decode_gpus <<<"${DECODE_GPUS}"
  decode_gpu_groups=("${legacy_decode_gpus[@]}")
fi

validate_gpu_groups() {
  local role="$1" tp_size="$2"
  shift 2
  local group gpu
  local -a ranks
  [[ "${tp_size}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${role}_TP_SIZE must be a positive integer" >&2
    exit 1
  }
  for group in "$@"; do
    IFS=',' read -r -a ranks <<<"${group}"
    if (( ${#ranks[@]} != tp_size )); then
      echo "${role} GPU group '${group}' has ${#ranks[@]} ranks; expected ${tp_size}" >&2
      exit 1
    fi
    for gpu in "${ranks[@]}"; do
      [[ "${gpu}" =~ ^[0-9]+$ ]] || {
        echo "invalid GPU '${gpu}' in ${role} group '${group}'" >&2
        exit 1
      }
    done
  done
}

validate_gpu_groups PREFILL "${PREFILL_TP_SIZE}" "${prefill_gpu_groups[@]}"
validate_gpu_groups DECODE "${DECODE_TP_SIZE}" "${decode_gpu_groups[@]}"
if (( PREFILL_TP_SIZE != DECODE_TP_SIZE )); then
  echo "Agentic TP currently requires equal PREFILL_TP_SIZE and DECODE_TP_SIZE" >&2
  exit 1
fi
export SGLANG_AGENTIC_KV_PREFILL_DOMAIN_COUNT="${#prefill_gpu_groups[@]}"
export SGLANG_AGENTIC_KV_TP_SIZE="${PREFILL_TP_SIZE}"
if (( ${#decode_gpu_groups[@]} != ${#decode_ports[@]} )); then
  echo "DECODE GPU groups and DECODE_PORTS must contain the same number of entries" >&2
  exit 1
fi
if (( ${#prefill_gpu_groups[@]} != ${#prefill_ports[@]} || ${#prefill_gpu_groups[@]} != ${#bootstrap_ports[@]} )); then
  echo "PREFILL GPU groups, PREFILL_PORTS and BOOTSTRAP_PORTS must have equal lengths" >&2
  exit 1
fi
if (( ${#decode_gpu_groups[@]} % ${#prefill_gpu_groups[@]} != 0 )); then
  echo "DECODE GPU groups must split evenly across logical P workers" >&2
  exit 1
fi
if (( ${#decode_mem_fraction_statics[@]} == 0 )); then
  for _ in "${decode_gpu_groups[@]}"; do
    decode_mem_fraction_statics+=("${MEM_FRACTION_STATIC}")
  done
elif (( ${#decode_mem_fraction_statics[@]} != ${#decode_gpu_groups[@]} )); then
  echo "DECODE_MEM_FRACTION_STATICS must be empty or match logical DECODE GPU groups" >&2
  exit 1
fi
if (( ${#local_gpus[@]} != ${#local_ports[@]} )); then
  echo "LOCAL_GPUS and LOCAL_PORTS must contain the same number of entries" >&2
  exit 1
fi
if (( ${#local_gpus[@]} > 0 )) && [[ -z "${LOCAL_ROUTER_PORT}" ]]; then
  echo "LOCAL_ROUTER_PORT is required when LOCAL_GPUS is non-empty" >&2
  exit 1
fi

if [[ "${PD_TOPOLOGY_VALIDATE_ONLY:-0}" == "1" ]]; then
  printf 'prefill_tp=%s groups=%s\n' "${PREFILL_TP_SIZE}" "${prefill_gpu_groups[*]}"
  printf 'decode_tp=%s groups=%s\n' "${DECODE_TP_SIZE}" "${decode_gpu_groups[*]}"
  printf 'decode_mem_fraction_statics=%s\n' "${decode_mem_fraction_statics[*]}"
  exit 0
fi

agentic_direct_ports=()
for index in "${!decode_ports[@]}"; do
  if [[ -n "${AGENTIC_DIRECT_PORT_OFFSET}" ]]; then
    direct_port="$((decode_ports[index] + AGENTIC_DIRECT_PORT_OFFSET))"
  else
    direct_port="$((AGENTIC_DIRECT_BASE_PORT + index))"
  fi
  if (( direct_port < 1 || direct_port > 65535 )); then
    echo "invalid agentic reverse NIXL port ${direct_port}" >&2
    exit 1
  fi
  agentic_direct_ports+=("${direct_port}")
done

ports_to_check=("${prefill_ports[@]}" "${decode_ports[@]}" "${ROUTER_PORT}" "${bootstrap_ports[@]}" "${local_ports[@]}")
[[ "${PD_SKIP_SEARCH}" != "1" ]] && ports_to_check+=("${SEARCH_PORT}")
if [[ "${SGLANG_AGENTIC_KV_LIFECYCLE:-false}" == "true" ]] \
  && [[ "${SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD:-0}" != "0" ]]; then
  for port in "${agentic_direct_ports[@]}"; do
    ports_to_check+=("${port}")
  done
fi
[[ -n "${LOCAL_ROUTER_PORT}" ]] && ports_to_check+=("${LOCAL_ROUTER_PORT}")
for port in "${ports_to_check[@]}"; do
  if port_is_open "${port}"; then
    echo "port ${port} is already in use; refusing to disturb an existing service" >&2
    exit 1
  fi
done
model_gpus=()
for group in "${prefill_gpu_groups[@]}" "${decode_gpu_groups[@]}"; do
  IFS=',' read -r -a group_ranks <<<"${group}"
  model_gpus+=("${group_ranks[@]}")
done
declare -A seen_gpus=()
for gpu in "${model_gpus[@]}" "${local_gpus[@]}"; do
  if [[ -n "${seen_gpus[${gpu}]:-}" ]]; then
    echo "GPU ${gpu} appears in more than one model group" >&2
    exit 1
  fi
  seen_gpus["${gpu}"]=1
  check_gpu_idle "${gpu}"
done
# SEARCH_GPU is intentionally allowed to share a Decode rank.
if [[ "${PD_SKIP_SEARCH}" != "1" ]]; then
  check_gpu_idle "${SEARCH_GPU}"
fi

local_pids=()
for index in "${!local_gpus[@]}"; do
  setsid env CUDA_VISIBLE_DEVICES="${local_gpus[$index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 0.0.0.0 --port "${local_ports[$index]}" \
      --context-length 40960 \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --enable-metrics \
    >"${RUN_DIR}/logs/local-${index}.log" 2>&1 &
  local_pids+=("$!")
  pids+=("$!")
done

start_search_server() {
  setsid env CUDA_VISIBLE_DEVICES="${SEARCH_GPU}" SEARCH_SERVER_GPU_IDS=0 \
    SEARCH_SERVER_EMBEDDING_CACHE="${SEARCH_SERVER_EMBEDDING_CACHE}" \
    python "${PD_DIR}/search_server.py" \
      --model Qwen/Qwen3-Embedding-8B \
      --corpus Tevatron/browsecomp-plus-corpus \
      --corpus-embedding-dataset miaolu3/browsecomp-plus \
      --host 0.0.0.0 --port "${SEARCH_PORT}" \
    >"${RUN_DIR}/logs/search.log" 2>&1 &
  search_pid=$!
  pids+=("${search_pid}")
  wait_http search "http://127.0.0.1:${SEARCH_PORT}/health" "${search_pid}" 1200
}

if [[ "${PD_SKIP_SEARCH}" != "1" && "${SEARCH_START_AFTER_MODELS}" != "true" ]]; then
  start_search_server
fi

prefill_numa_vectors=()
prefill_pids=()
for index in "${!prefill_gpu_groups[@]}"; do
  prefill_group="${prefill_gpu_groups[$index]}"
  IFS=',' read -r -a prefill_group_gpus <<<"${prefill_group}"
  prefill_group_numas=()
  for gpu in "${prefill_group_gpus[@]}"; do
    prefill_group_numas+=("$(gpu_numa_node "${gpu}")")
  done
  prefill_numa_csv="$(IFS=,; echo "${prefill_group_numas[*]}")"
  prefill_numa_vectors+=("${prefill_numa_csv}")
  prefill_numa="${prefill_group_numas[0]}"
  prefill_launch=(setsid)
  prefill_raw_log_args=()
  if [[ -n "${PD_RAW_REQUEST_LOG_DIR}" ]]; then
    mkdir -p "${PD_RAW_REQUEST_LOG_DIR}/prefill-${index}"
    prefill_raw_log_args=(--log-requests --log-requests-level 3
      --log-requests-format json --log-requests-target "${PD_RAW_REQUEST_LOG_DIR}/prefill-${index}")
  fi
  if (( PREFILL_TP_SIZE == 1 )) && [[ "${SGLANG_AGENTIC_KV_HOST_STAGING:-false}" == "true" ]] && command -v numactl >/dev/null 2>&1; then
    prefill_launch+=(numactl --cpunodebind="${prefill_numa}" --membind="${prefill_numa}")
  fi
  "${prefill_launch[@]}" env CUDA_VISIBLE_DEVICES="${prefill_group}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    SGLANG_AGENTIC_KV_ENGINE_ID="prefill-${index}" \
    SGLANG_AGENTIC_KV_PREFILL_DOMAIN="${index}" \
    SGLANG_AGENTIC_KV_ARENA_NUMA_NODE="${prefill_numa}" \
    SGLANG_AGENTIC_KV_TP_NUMA_NODES="${prefill_numa_csv}" \
    SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR="${SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR}/p-${index}-numa-${prefill_numa}" \
    SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_DIR="${SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_DIR:-/dev/shm/sglang-agentic-p2d-disabled}/p-${index}-numa-${prefill_numa}" \
    SGLANG_PD_P_READY_DIR="${PD_P_READY_DIR}" \
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${PD_HICACHE_STORAGE_DIR}" \
    python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" --host 0.0.0.0 --port "${prefill_ports[$index]}" \
      --tp-size "${PREFILL_TP_SIZE}" --numa-node "${prefill_group_numas[@]}" \
      --context-length "${MAX_CONTEXT_LENGTH}" --page-size "${PD_PAGE_SIZE}" \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" --enable-metrics \
      --skip-server-warmup \
      --chunked-prefill-size "${PREFILL_CHUNKED_PREFILL_SIZE}" \
      --max-prefill-tokens "${PREFILL_MAX_PREFILL_TOKENS}" \
      "${mamba_args[@]}" \
      "${model_parser_args[@]}" \
      "${prefill_raw_log_args[@]}" \
      --uvicorn-access-log-exclude-prefixes /get_load /metrics /health \
      "${deterministic_args[@]}" \
      --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
      --disaggregation-bootstrap-port "${bootstrap_ports[$index]}" \
      "${prefill_hicache_args[@]}" \
    >"${RUN_DIR}/logs/prefill-${index}.log" 2>&1 &
  prefill_pid="$!"
  prefill_pids+=("${prefill_pid}")
  pids+=("${prefill_pid}")
  # A disaggregated P's /health probe performs a real Prefill and waits for a
  # Decode consumer.  During bootstrap no D or Router exists yet, so using it
  # here serializes startup behind an impossible transfer.  /model_info proves
  # the HTTP/model process is ready without creating a P-ready generation.
  wait_http "prefill-${index}" "http://127.0.0.1:${prefill_ports[$index]}/model_info" "${prefill_pids[$index]}"
  # The HTTP server can become healthy even when its auxiliary KV bootstrap
  # listener failed to bind.  That produces a deceptive half-alive P: every
  # later P->D handshake retries forever behind the first FIFO generation.
  # Treat any listener bind failure during startup as fatal.
  if grep -Eq 'Server error:.*address already in use|error while attempting to bind.*address already in use' \
      "${RUN_DIR}/logs/prefill-${index}.log"; then
    echo "prefill-${index} KV bootstrap listener failed; see ${RUN_DIR}/logs/prefill-${index}.log" >&2
    exit 1
  fi
done

prefill_numa_domains="$(IFS=';'; echo "${prefill_numa_vectors[*]}")"
export SGLANG_AGENTIC_KV_PREFILL_TP_NUMA_DOMAINS="${prefill_numa_domains}"

decode_pids=()
for index in "${!decode_gpu_groups[@]}"; do
  decode_group="${decode_gpu_groups[$index]}"
  IFS=',' read -r -a decode_group_gpus <<<"${decode_group}"
  decode_group_numas=()
  for gpu in "${decode_group_gpus[@]}"; do
    decode_group_numas+=("$(gpu_numa_node "${gpu}")")
  done
  decode_numa_csv="$(IFS=,; echo "${decode_group_numas[*]}")"
  decode_numa="${decode_group_numas[0]}"
  decode_launch=(setsid)
  decode_raw_log_args=()
  if [[ -n "${PD_RAW_REQUEST_LOG_DIR}" ]]; then
    mkdir -p "${PD_RAW_REQUEST_LOG_DIR}/decode-${index}"
    decode_raw_log_args=(--log-requests --log-requests-level 3
      --log-requests-format json --log-requests-target "${PD_RAW_REQUEST_LOG_DIR}/decode-${index}")
  fi
  if (( DECODE_TP_SIZE == 1 )) && [[ "${SGLANG_AGENTIC_KV_HOST_STAGING:-false}" == "true" ]] && command -v numactl >/dev/null 2>&1; then
    decode_launch+=(numactl --cpunodebind="${decode_numa}" --membind="${decode_numa}")
  fi
  domain="$((index / (${#decode_gpu_groups[@]} / ${#prefill_gpu_groups[@]})))"
  arena_numa_csv="${prefill_numa_vectors[$domain]}"
  IFS=',' read -r -a arena_group_numas <<<"${arena_numa_csv}"
  arena_numa="${arena_group_numas[0]}"
  "${decode_launch[@]}" env CUDA_VISIBLE_DEVICES="${decode_group}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    SGLANG_AGENTIC_KV_ENGINE_ID="decode-${index}" \
    SGLANG_AGENTIC_KV_PREFILL_DOMAIN="${domain}" \
    SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT="${agentic_direct_ports[$index]}" \
    SGLANG_AGENTIC_KV_RELAY_ID="decode-${index}-gpus-${decode_group//,/-}" \
    SGLANG_AGENTIC_KV_GPU_NUMA_NODE="${decode_numa}" \
    SGLANG_AGENTIC_KV_ARENA_NUMA_NODE="${arena_numa}" \
    SGLANG_AGENTIC_KV_TP_NUMA_NODES="${decode_numa_csv}" \
    SGLANG_AGENTIC_KV_TP_ARENA_NUMA_NODES="${arena_numa_csv}" \
    SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR="${SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR}/p-${domain}-numa-${arena_numa}" \
    SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_DIR="${SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_DIR:-/dev/shm/sglang-agentic-p2d-disabled}/p-${domain}-numa-${arena_numa}" \
    SGLANG_PD_MAX_TRANSFER_INFLIGHT="${PD_MAX_TRANSFER_INFLIGHT}" \
    SGLANG_PD_P_READY_DIR="${PD_P_READY_DIR}" \
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${PD_HICACHE_STORAGE_DIR}" \
    python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 0.0.0.0 --port "${decode_ports[$index]}" \
      --tp-size "${DECODE_TP_SIZE}" --numa-node "${decode_group_numas[@]}" \
      --context-length "${MAX_CONTEXT_LENGTH}" \
      --page-size "${PD_PAGE_SIZE}" \
      --mem-fraction-static "${decode_mem_fraction_statics[$index]}" \
      "${mamba_args[@]}" \
      "${model_parser_args[@]}" \
      "${decode_raw_log_args[@]}" \
      --enable-metrics --skip-server-warmup \
      --uvicorn-access-log-exclude-prefixes /get_load /metrics /health \
      "${deterministic_args[@]}" \
      --disaggregation-mode decode \
      --disaggregation-transfer-backend nixl \
      "${decode_hicache_args[@]}" \
    >"${RUN_DIR}/logs/decode-${index}.log" 2>&1 &
  decode_pids+=("$!")
  pids+=("$!")
  # Keep model bootstrap non-generative for the same reason as Prefill.  The
  # Router health check below validates the complete P/D serving topology once
  # every worker is present.
  wait_http "decode-${index}" "http://127.0.0.1:${decode_ports[$index]}/model_info" "${decode_pids[$index]}"
done

wait_registered_host_prewarm

if [[ "${PD_SKIP_SEARCH}" != "1" && "${SEARCH_START_AFTER_MODELS}" == "true" ]]; then
  start_search_server
fi

for index in "${!local_ports[@]}"; do
  wait_http "local-${index}" "http://127.0.0.1:${local_ports[$index]}/health" "${local_pids[$index]}"
done

# Prefill's startup probe is a real disaggregated request and can leave a
# room-0 P-ready marker before the late-binding Router exists.  That marker is
# not workload traffic, but its ready_sequence would remain the oldest FIFO
# entry forever.  Remove only markers whose payload identifies the built-in
# health probe; ordinary request markers are never touched.
if [[ "${PD_LATE_BINDING}" == "1" ]]; then
  python - "${PD_P_READY_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for ready in root.glob("*.ready"):
    try:
        payload = json.loads(ready.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        continue
    if not str(payload.get("rid", "")).startswith("HEALTH_CHECK_"):
        continue
    stem = ready.name.removesuffix(".ready")
    for suffix in ("ready", "accepted", "scheduled"):
        (root / f"{stem}.{suffix}").unlink(missing_ok=True)
PY
fi

router_decode_args=()
router_prefill_args=()
for index in "${!prefill_ports[@]}"; do
  router_prefill_args+=(--prefill "http://127.0.0.1:${prefill_ports[$index]}" "${bootstrap_ports[$index]}")
done
for port in "${decode_ports[@]}"; do
  router_decode_args+=(--decode "http://127.0.0.1:${port}")
done
router_entry=(python -m sglang_router.launch_router)
router_policy_args=()
if [[ "${PD_LATE_BINDING}" == "1" ]]; then
  router_entry=(python "${PD_LATE_BIND_ROUTER_ENTRY:-${PD_DIR}/launch_late_binding_router.py}")
  router_policy_args=(--policy random)
fi
setsid env \
  SGLANG_PD_P_READY_DIR="${PD_P_READY_DIR}" \
  SGLANG_PD_LATE_BIND_READY_TIMEOUT_S="${PD_LATE_BIND_READY_TIMEOUT_S}" \
  SGLANG_PD_LATE_BIND_RESERVATION_TIMEOUT_S="${PD_LATE_BIND_RESERVATION_TIMEOUT_S}" \
  SGLANG_PD_LATE_BIND_DECODE_HEADROOM_TOKENS="${PD_LATE_BIND_DECODE_HEADROOM_TOKENS}" \
  SGLANG_PD_LATE_BIND_WAIT_FOR_FEASIBLE="${PD_LATE_BIND_WAIT_FOR_FEASIBLE}" \
  SGLANG_PD_LATE_BIND_LOAD_CACHE_TTL_S="${PD_LATE_BIND_LOAD_CACHE_TTL_S}" \
  SGLANG_PD_LATE_BIND_NO_CAPACITY_POLL_S="${PD_LATE_BIND_NO_CAPACITY_POLL_S}" \
  SGLANG_PD_LATE_BIND_SOFT_RESERVATION_DELAY_S="${PD_LATE_BIND_SOFT_RESERVATION_DELAY_S}" \
  SGLANG_PD_LATE_BIND_SOFT_RESERVATION_MIN_TOKENS="${PD_LATE_BIND_SOFT_RESERVATION_MIN_TOKENS}" \
  SGLANG_PD_LATE_BIND_SOFT_RESERVATION_FORCE_AFTER_S="${PD_LATE_BIND_SOFT_RESERVATION_FORCE_AFTER_S}" \
  SGLANG_PD_LATE_BIND_FORCE_LEGACY_LOADS="${PD_LATE_BIND_FORCE_LEGACY_LOADS}" \
  SGLANG_PD_LATE_BIND_NUMA_DOMAINS="${PD_LATE_BIND_NUMA_DOMAINS:-0}" \
  "${router_entry[@]}" \
  --pd-disaggregation \
  "${router_prefill_args[@]}" \
  "${router_decode_args[@]}" \
  "${router_policy_args[@]}" \
  --health-check-timeout-secs "${ROUTER_HEALTH_TIMEOUT_SECS}" \
  --health-failure-threshold "${ROUTER_HEALTH_FAILURE_THRESHOLD}" \
  --prometheus-port "${ROUTER_PROMETHEUS_PORT}" \
  --host 0.0.0.0 --port "${ROUTER_PORT}" \
  >"${RUN_DIR}/logs/router.log" 2>&1 &
router_pid=$!
pids+=("${router_pid}")
frontend_pids+=("${router_pid}")

if (( ${#local_ports[@]} > 0 )); then
  local_worker_args=()
  for port in "${local_ports[@]}"; do
    local_worker_args+=("http://127.0.0.1:${port}")
  done
  setsid python -m sglang_router.launch_router \
    --worker-urls "${local_worker_args[@]}" \
    --policy cache_aware \
    --host 0.0.0.0 --port "${LOCAL_ROUTER_PORT}" \
    >"${RUN_DIR}/logs/local-router.log" 2>&1 &
  local_router_pid=$!
  pids+=("${local_router_pid}")
  frontend_pids+=("${local_router_pid}")
fi

wait_http router "http://127.0.0.1:${ROUTER_PORT}/health" "${router_pid}" 120
if [[ "${PD_LATE_BINDING}" == "1" ]] \
  && ! grep -q "Late-binding PD enabled" "${RUN_DIR}/logs/router.log"; then
  echo "Late-binding router health check passed, but its activation marker is missing" >&2
  exit 1
fi
if (( ${#local_ports[@]} > 0 )); then
  wait_http local-router "http://127.0.0.1:${LOCAL_ROUTER_PORT}/health" "${local_router_pid}" 120
fi
if [[ "${PD_SERVE_ONLY}" == "1" ]]; then
  echo "PD services ready; PD_SERVE_ONLY=1, waiting for external requests"
  while true; do
    sleep 30
  done
fi
if [[ -n "${PD_CORRECTNESS_CAPTURE_OUTPUT}" ]]; then
  env CUDA_VISIBLE_DEVICES="" python "${PD_DIR}/scripts/tools/compare_decode_offload_outputs.py" \
    --url "http://127.0.0.1:${ROUTER_PORT}" \
    --model "${MODEL_PATH}" \
    --label "${PD_CORRECTNESS_CAPTURE_LABEL}" \
    --output "${PD_CORRECTNESS_CAPTURE_OUTPUT}" \
    --sequential
  echo "PD correctness capture complete: ${PD_CORRECTNESS_CAPTURE_OUTPUT}"
  exit 0
fi

read -r -a rates <<<"${ARRIVAL_RATES}"
summaries=()
for rate in "${rates[@]}"; do
  rate_dir="${RUN_DIR}"
  if (( ${#rates[@]} > 1 )); then
    rate_dir="${RUN_DIR}/rate-${rate}"
  fi
  duration_args=()
  if [[ -n "${MEASUREMENT_DURATION_SECONDS}" ]]; then
    duration_args+=(--measurement-duration-seconds "${MEASUREMENT_DURATION_SECONDS}")
  fi
  schedule_args=()
  if [[ -n "${SCHEDULE_FILE}" ]]; then
    schedule_args+=(--schedule-file "${SCHEDULE_FILE}")
  fi
  if [[ "${PRESERVE_SOURCE_ORDER}" == "true" ]]; then
    schedule_args+=(--preserve-source-order)
  fi
  closed_loop_args=()
  if [[ "${CLOSED_LOOP}" == "1" ]]; then
    closed_loop_args+=(
      --closed-loop
      --closed-loop-warmup-min-seconds "${CLOSED_LOOP_WARMUP_MIN_SECONDS}"
      --closed-loop-warmup-completions "${CLOSED_LOOP_WARMUP_COMPLETIONS}"
      --closed-loop-recent-seconds "${CLOSED_LOOP_RECENT_SECONDS}"
      --closed-loop-max-warmup-seconds "${CLOSED_LOOP_MAX_WARMUP_SECONDS}"
      --closed-loop-measurement-seconds "${CLOSED_LOOP_MEASUREMENT_SECONDS}"
    )
  fi
  local_args=()
  if (( ${#local_ports[@]} > 0 )); then
    local_args+=(--retool-local-router-port "${LOCAL_ROUTER_PORT}" --local-ports "${local_ports_csv}")
  fi
  workload_args=()
  if [[ -n "${WORKLOAD_CONFIG}" ]]; then
    workload_args+=(--workload-config "${WORKLOAD_CONFIG}")
  else
    workload_args+=(--math-data "${MATH_DATA}" --qa-data "${QA_DATA}" --math-ratio "${MATH_RATIO}")
  fi
  env CUDA_VISIBLE_DEVICES="" python "${INFERENCE_ENTRY}" \
    --model "${MODEL_PATH}" \
    "${workload_args[@]}" \
    --router-port "${ROUTER_PORT}" \
    --router-request-timeout-seconds "${ROUTER_REQUEST_TIMEOUT_SECONDS:-3600}" \
    --prefill-port "${prefill_ports[0]}" \
    --prefill-ports "${prefill_ports_csv}" \
    --decode-port "${DECODE_PORT}" \
    --decode-ports "${decode_ports_csv}" \
    --pd-max-transfer-inflight "${PD_MAX_TRANSFER_INFLIGHT}" \
    --pd-p-ready-dir "${PD_P_READY_DIR}" \
    "${hicache_metadata_args[@]}" \
    "${local_args[@]}" \
    --request-rate "${rate}" \
    --arrival-distribution "${ARRIVAL_DISTRIBUTION}" \
    --dispatch-policy "${DISPATCH_POLICY}" \
    "${schedule_args[@]}" \
    --dynamic-lookback-seconds "${DYNAMIC_LOOKBACK_SECONDS}" \
    --dynamic-recent-seconds "${DYNAMIC_RECENT_SECONDS}" \
    --dynamic-history-start-seconds "${DYNAMIC_HISTORY_START_SECONDS}" \
    --dynamic-history-end-seconds "${DYNAMIC_HISTORY_END_SECONDS}" \
    --dynamic-prefill-capacity-tps "${DYNAMIC_PREFILL_CAPACITY_TPS}" \
    --dynamic-decode-capacity-tps "${DYNAMIC_DECODE_CAPACITY_TPS}" \
    --dynamic-decode-target-active "${DYNAMIC_DECODE_TARGET_ACTIVE}" \
    --dynamic-hysteresis "${DYNAMIC_HYSTERESIS}" \
    --dynamic-max-imbalance "${DYNAMIC_MAX_IMBALANCE}" \
    --dynamic-max-consecutive "${DYNAMIC_MAX_CONSECUTIVE}" \
    --requests "${REQUESTS}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --max-inflight "${MAX_INFLIGHT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-context-length "${MAX_CONTEXT_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --metrics-interval "${METRICS_INTERVAL}" \
    --seed "${SEED}" \
    "${closed_loop_args[@]}" \
    "${duration_args[@]}" \
    --output-dir "${rate_dir}"
  summaries+=("${rate_dir}/summary.json")
done

if (( ${#rates[@]} > 1 )); then
  python "${PD_DIR}/scripts/tools/select_rate.py" "${summaries[@]}" --output "${RUN_DIR}/rate_sweep_summary.json"
fi

echo "PD experiment complete: ${RUN_DIR}"
