#!/usr/bin/env bash
# Serve a small Qwen2.5-VL on this machine's single 12GB GPU (RTX 3080 Ti) for SMOKE TESTING.
# The desktop already uses ~2.7GB of the 12GB, so we keep gpu-memory-utilization modest and
# use the AWQ (4-bit) 3B model. For full quality use serve_server.sh on the L40S box.
set -euo pipefail
ENV_NAME="${ENV_NAME:-safe-vllm}"
CONDA_BASE="$(conda info --base)"
VLLM="$CONDA_BASE/envs/$ENV_NAME/bin/vllm"
# put the env's bin on PATH so JIT helpers (ninja/nvcc) resolve, and avoid flashinfer's
# runtime JIT-compiled sampler (needs ninja) by using vLLM's native PyTorch sampler.
export PATH="$CONDA_BASE/envs/$ENV_NAME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
GMU="${GMU:-0.60}"     # fraction of TOTAL 12GB -> ~7.2GB, leaves room for Xorg/browser

echo "[serve] model=$MODEL port=$PORT max_len=$MAX_LEN gpu_util=$GMU"
exec "$VLLM" serve "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image": 3}' \
  --dtype float16
