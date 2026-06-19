#!/usr/bin/env bash
set -e

MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 8000