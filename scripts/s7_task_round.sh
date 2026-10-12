#!/bin/bash
# usage: scripts/s7_task_round.sh RUN_DIR [SHARDS]  (env: RENDER=full|program-header, PORT, SNAP, DIRECT=export dir)
# S7 smoke round (plans/neuralese/S7_RL.md §8): serve LFM2.5-350M on GPU, run the group queue through the collector in SHARDS
# parallel queue runners, merge journals, score.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=$1; N=${2:-1}
SNAP=${SNAP:-$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f}
PY=$ROOT/.venv-neuralese/bin/python
export PYTHONPATH=$ROOT/training/neuralese HF_HUB_OFFLINE=1
if [ -n "${DIRECT:-}" ]; then  # a direct-SFT export serves with its own declared rendering unless RENDER overrides
  MODEL=(--direct-artifact "$DIRECT" --direct-base-snapshot "$SNAP" ${RENDER:+--prompt-rendering $RENDER})
else
  MODEL=(--base "$SNAP" --prompt-rendering "${RENDER:-full}")
fi
$PY -m natlang_neuralese.serve "${MODEL[@]}" --device cuda --dtype bfloat16 --port ${PORT:-8097} > $R/server.log 2>&1 &
SP=$!
for i in $(seq 1 150); do curl -sf http://127.0.0.1:${PORT:-8097}/v1/models >/dev/null && break; sleep 2; done
mkdir -p $R/shards
python3 - "$R" "$N" <<'PY'
import json, sys
r, n = sys.argv[1], int(sys.argv[2])
rows = [l for l in open(f"{r}/queue.jsonl") if l.strip()]
for s in range(n):
    open(f"{r}/shards/queue-{s}.jsonl", "w").write("".join(rows[s::n]))
PY
pids=()
for s in $(seq 0 $((N-1))); do
  python3 $ROOT/scripts/run_bonsai_queue.py $R/shards/queue-$s.jsonl $R/shards/journal-$s.jsonl --runtime $ROOT/ts-host \
    --server http://127.0.0.1:${PORT:-8097} --chat-completions-url http://127.0.0.1:${PORT:-8097}/v1/chat/completions --model-id lfm \
    --model-concurrency 2 > $R/shards/queue-$s.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
cat $R/shards/journal-*.jsonl > $R/journal.jsonl
$PY -m natlang_neuralese.rl.task_rollouts score --run-dir $R > $R/score.log 2>&1
kill $SP; wait $SP
