#!/usr/bin/env bash
# Serve the full-quality Qwen2.5-VL-72B on the 7xL40S server (336GB total VRAM).
# 72B in bf16 (~145GB) shards cleanly across 4 GPUs with tensor parallelism.
set -euo pipefail
# Model weights + the vLLM env live under $SAFE_FILES (shared models disk).
# Override SAFE_FILES / VENV / HF_HOME to relocate.
SAFE_FILES="${SAFE_FILES:-/mnt/disk2/SAFE_files}"
VENV="${VENV:-$SAFE_FILES/safe-vllm}"
VLLM="$VENV/bin/vllm"
export PATH="$VENV/bin:$PATH"
export HF_HOME="${HF_HOME:-$SAFE_FILES/hf_cache}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL="${MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-4}"               # tensor-parallel GPUs (4 of the 7 L40S)
MAX_LEN="${MAX_LEN:-32768}"

echo "[serve] model=$MODEL port=$PORT tp=$TP max_len=$MAX_LEN"
exec "$VLLM" serve "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --limit-mm-per-prompt '{"image": 3}'
