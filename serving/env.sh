# Source this before running the SAFE pipeline against the served vLLM model.
#   source serving/env.sh
# The repo's OpenAI() clients auto-read OPENAI_BASE_URL + OPENAI_API_KEY, so this is the
# entire client-side swap from GPT-4o to the self-hosted Qwen3-VL.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8001/v1}"
# Model name to pass as --gpt to the SAFE stages (must match the server's --served-model-name).
export SAFE_MODEL="${SAFE_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
# Shared models disk: model weights + the vLLM env live here; the server resolves weights from it.
export SAFE_FILES="${SAFE_FILES:-/mnt/disk2/SAFE_files}"
export HF_HOME="${HF_HOME:-$SAFE_FILES/hf_cache}"
echo "[env] OPENAI_BASE_URL=$OPENAI_BASE_URL  SAFE_MODEL=$SAFE_MODEL  HF_HOME=$HF_HOME"
