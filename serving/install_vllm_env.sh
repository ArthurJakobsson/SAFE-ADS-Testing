#!/usr/bin/env bash
# Creates a dedicated conda env for serving Qwen2.5-VL via vLLM (OpenAI-compatible API).
# Base conda is Python 3.13 which vLLM does not support, so we use a fresh py3.11 env.
set -euo pipefail

ENV_NAME="${ENV_NAME:-safe-vllm}"
CONDA_BASE="$(conda info --base)"
PY_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

echo "[install] creating conda env '$ENV_NAME' (python 3.11)"
conda create -y -n "$ENV_NAME" python=3.11

echo "[install] upgrading pip"
"$PIP_BIN" install --upgrade pip wheel

echo "[install] installing vllm + openai client (this pulls torch+cuda, may take a while)"
# vllm pins a compatible torch/transformers; transformers must be >=4.49 for Qwen2.5-VL,
# which recent vllm releases satisfy automatically.
"$PIP_BIN" install "vllm>=0.7.2" "openai>=1.40" "huggingface_hub[hf_transfer]"

echo "[install] vllm version:"
"$PY_BIN" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "[install] DONE. Serve with: bash serving/serve_local.sh"
