# Source this before running the SAFE pipeline against a local/served vLLM model.
#   source serving/env.sh
# The repo's OpenAI() clients auto-read OPENAI_BASE_URL + OPENAI_API_KEY, so this is the
# entire client-side swap from GPT-4o to a self-hosted Qwen2.5-VL.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
# Model name to pass as --gpt to the SAFE stages (must match the server's --served-model-name).
export SAFE_MODEL="${SAFE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}"
echo "[env] OPENAI_BASE_URL=$OPENAI_BASE_URL  SAFE_MODEL=$SAFE_MODEL"
