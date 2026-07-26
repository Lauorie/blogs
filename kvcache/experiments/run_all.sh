#!/bin/bash
# Run all KV cache experiments sequentially; logs to results/run.log.
set -uo pipefail
cd "$(dirname "$0")"
PY=/root/miniconda3/bin/python
mkdir -p results
{
  echo "=== run started $(date -u +%FT%TZ) ==="
  for exp in e1_equivalence e2_latency e3_ttft e4_memory; do
    echo "--- $exp ---"
    $PY "$exp.py" || echo "!!! $exp FAILED with exit $?"
  done
  $PY -m pip freeze > results/pip_freeze.txt
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > results/gpu.txt
  echo "=== run finished $(date -u +%FT%TZ) ==="
} 2>&1 | tee results/run.log
