#!/usr/bin/env bash
# CPU-only two-host control protocol test. No model/CUDA/NCCL, files or GPU jobs
# are started on the peer. The current protocol source is sent over SSH stdin.
set -euo pipefail
dualpd_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=''
exec "${DUALPD_PYTHON:-python3}" \
  "${dualpd_root}/validation/tp_events_two_node.py" "$@"
