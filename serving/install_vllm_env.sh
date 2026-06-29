#!/usr/bin/env bash
# Build the dedicated vLLM serving env under $SAFE_FILES (shared models disk), next to the
# model weights. We use a virtualenv rather than a conda env: on this box conda.anaconda.org is
# behind an SSL-inspecting proxy so `conda create` fails, while PyPI is reachable.
set -euo pipefail

SAFE_FILES="${SAFE_FILES:-/mnt/disk2/SAFE_files}"
VENV="${VENV:-$SAFE_FILES/safe-vllm}"
PY="${PY:-python3}"                 # base interpreter (vLLM needs Python <3.13)
PIP_BIN="$VENV/bin/pip"
PY_BIN="$VENV/bin/python"

mkdir -p "$SAFE_FILES"
echo "[install] creating venv at $VENV (python: $("$PY" --version 2>&1))"
if "$PY" -m venv "$VENV" 2>/dev/null; then
  :
else
  echo "[install] base python lacks ensurepip; falling back to virtualenv"
  "$PY" -m pip install --user -q virtualenv
  "$PY" -m virtualenv "$VENV"
fi

echo "[install] upgrading pip/wheel/setuptools"
"$PIP_BIN" install --upgrade pip wheel setuptools

echo "[install] installing latest vllm + transformers + clients (pulls torch+cuda, slow)"
# Qwen3-VL needs a recent vLLM/transformers, so install the latest releases.
"$PIP_BIN" install -U vllm transformers "huggingface_hub" hf_transfer "openai>=1.40"

echo "[install] versions:"
"$PY_BIN" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

echo "[install] DONE. Serve with: bash serving/serve_server.sh"
