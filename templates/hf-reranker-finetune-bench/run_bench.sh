#!/usr/bin/env bash
set -euo pipefail

date
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; skipping GPU info."
fi

python3 /app/bench_hf_reranker_finetune.py "$@"
