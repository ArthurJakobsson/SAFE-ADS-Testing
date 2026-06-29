#!/usr/bin/env bash
# Safely stop the vLLM server. NOTE: do NOT `pkill -f vllm` from a shell whose own command
# line contains "vllm" — it matches and kills itself. pgrep excludes its own PID, so this is safe.
pids="$(pgrep -f 'bin/vllm serve' || true)"
if [ -z "$pids" ]; then echo "[stop] no vllm serve process found"; exit 0; fi
echo "[stop] killing vllm serve pids: $pids"
kill $pids 2>/dev/null || true
sleep 2
# force any stragglers
pids="$(pgrep -f 'bin/vllm serve' || true)"
[ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
echo "[stop] done"
