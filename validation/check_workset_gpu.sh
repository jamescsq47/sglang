#!/usr/bin/env bash
# Bounded controller/CUDA ownership test, NOT a model/throughput benchmark.
set -euo pipefail
validation_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${validation_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "${DUALPD_PYTHON:-/homes/siqic/anaconda3/envs/pd_multi_node/bin/python}" \
  "${validation_root}/validation/workset_gpu.py" "$@"
