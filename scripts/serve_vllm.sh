#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=${1:-./merged-model}
"${PYTHON:-python}" -m jev_lite.preflight "vLLM 서빙"
vllm serve "$MODEL_DIR" \
  --served-model-name qwen-jev-custom \
  --dtype half --max-model-len 1024 --max-num-seqs 8 \
  --gpu-memory-utilization 0.65 --enforce-eager \
  --max-logprobs 5 --host 127.0.0.1 --port 8000
