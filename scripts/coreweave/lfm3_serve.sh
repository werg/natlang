#!/bin/bash
# LFM3 serving on a CoreWeave GPU sandbox with the prebuilt llama.cpp-neuralese binary (bootstrap step 4b exports
# NATLANG_LLAMA_SERVER; nothing is compiled). Job templates: scripts/coreweave/jobs/lfm3/*.job.json.
#
#   lfm3_serve.sh check OUT [GGUF_URL SHA256]   boot-to-ready timing, one tool-call request, host facts, matmul probe
#   lfm3_serve.sh eval  OUT SPEC [GGUF_URL SHA256]   scripts/held_task_budget_probe.py over SPEC (a task_eval JSON)
#   lfm3_serve.sh actor OUT [GGUF_URL SHA256]   serve on 0.0.0.0:$LFM3_PORT (the job's HTTPS service) until stopped
#
# GGUF_URL defaults to the official LiquidAI/LFM3-1.7B-GGUF BF16 at the pinned revision. A GGUF already in the bundle
# is named with LFM3_GGUF (absolute path); its sha256 is still checked. Server flags: LFM3_SERVER_ARGS
# (default: 16 slots, 64k context each, all layers on the GPU).
set -euo pipefail
MODE=$1; OUT=$2; shift 2
SPEC=; if [ "$MODE" = eval ]; then SPEC=$1; shift; fi
URL=${1:-https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-BF16.gguf}
SHA=${2:-39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68}
PORT=${LFM3_PORT:-8097}; HOST=127.0.0.1; [ "$MODE" = actor ] && HOST=0.0.0.0 && PORT=${LFM3_PORT:-8080}
SERVER=${NATLANG_LLAMA_SERVER:?no prebuilt llama-server: add --prebuilt llama-cpp-neuralese-sm120 to the bundle}
mkdir -p "$OUT"
t0=$(date +%s.%N)
el() { python3 -c "print(round($(date +%s.%N) - $t0, 1))"; }
facts() {
  { echo "== at $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap --format=csv
    echo "nproc $(nproc)"; echo "cgroup memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"; free -g
    lscpu | grep -E 'Model name|^CPU\(s\)'; df -h "${NATLANG_CW_WORK:-/work}"; } >> "$OUT/host.txt" 2>&1
}
facts
GGUF=${LFM3_GGUF:-${NATLANG_CW_WORK:-/work}/models/$(basename "$URL")}
if [ ! -f "$GGUF" ]; then
  mkdir -p "$(dirname "$GGUF")"; curl -fsSL --retry 4 -o "$GGUF.partial" "$URL"; mv "$GGUF.partial" "$GGUF"
fi
echo "$SHA  $GGUF" | sha256sum -c - > /dev/null
download_s=$(el)
read -r -a ARGS <<< "${LFM3_SERVER_ARGS:--np 16 -c 1048576}"
if [ "$MODE" = actor ]; then
  echo "{\"mode\":\"actor\",\"download_seconds\":$download_s}" > "$OUT/timings.json"
  exec "$SERVER" -m "$GGUF" --jinja -ngl 999 "${ARGS[@]}" --host "$HOST" --port "$PORT" 2>&1 | tee "$OUT/llama-server.log"
fi
"$SERVER" -m "$GGUF" --jinja -ngl 999 "${ARGS[@]}" --host "$HOST" --port "$PORT" > "$OUT/llama-server.log" 2>&1 &
pid=$!; trap 'kill $pid 2>/dev/null || true' EXIT
until curl -fs "http://$HOST:$PORT/health" > /dev/null; do
  kill -0 $pid 2>/dev/null || { echo '{"error":"llama-server exited"}' > "$OUT/timings.json"; tail -40 "$OUT/llama-server.log"; exit 1; }
  sleep 0.5
done
ready_s=$(el)
facts
curl -fs "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is 17*23? Use the eval tool."}],"tools":[{"type":"function","function":{"name":"eval","description":"Evaluate a JavaScript expression","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}],"temperature":0,"max_tokens":2048}' \
  > "$OUT/toolcall.json"
toolcall_s=$(el)
python3 - "$OUT/toolcall.json" <<'PY'
import json, sys
message = json.load(open(sys.argv[1]))['choices'][0]['message']
calls = message.get('tool_calls') or []
assert calls and calls[0]['function']['name'] == 'eval', f'no eval tool call: {message}'
print(json.dumps({'tool_call': calls[0]['function']}))
PY
boot=$(python3 -c "import os,time; print(round(time.time() - os.stat('/proc/1').st_ctime, 1))" 2>/dev/null || echo null)
echo "{\"mode\":\"$MODE\",\"download_seconds\":$download_s,\"server_ready_seconds\":$ready_s,\"toolcall_done_seconds\":$toolcall_s,\"since_container_start_seconds\":$boot,\"server_args\":\"${ARGS[*]}\"}" \
  > "$OUT/timings.json"
cat "$OUT/timings.json"
if [ "$MODE" = check ]; then
  probe="$NATLANG_REPO/training/neuralese/scripts/probe_lowp_matmul.py"
  if [ -f "$probe" ]; then python3 "$probe" > "$OUT/lowp-matmul.jsonl" 2> "$OUT/lowp-matmul.err" || true; fi
elif [ "$MODE" = eval ]; then
  python3 scripts/held_task_budget_probe.py --spec "$SPEC" --server "http://$HOST:$PORT" --out "$OUT/probe" \
    --case-seconds 7200 --max-model-requests 1000 --max-turns 1000 --workers "${LFM3_EVAL_WORKERS:-8}" > "$OUT/probe.log" 2>&1
fi
facts
