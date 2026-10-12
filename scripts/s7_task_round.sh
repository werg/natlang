#!/bin/bash
# usage: scripts/s7_task_round.sh RUN_DIR [SHARDS]  (env: RENDER, PORT, SNAP, DIRECT=export dir,
#   SERVE_IMAGE + RECEIPT + CONTAINER: serve DIRECT in a container at its source-proof paths)
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
if [ -n "${SERVE_IMAGE:-}" ]; then
  # A direct export pins its source proof to the exporting machine's paths (hf_hub_cache, launch receipt). Serve it in
  # a container that mounts the local HF cache and the exact receipt (RECEIPT, hash-checked by the loader) at those
  # paths; the container is named CONTAINER and must be declared to the ledger unit (--container).
  PROOF=$(python3 - "$DIRECT/manifest.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))["base_binding"]["source_proof"]
print(p["hf_hub_cache"], p["launch_receipt_path"], p["snapshot_path"])
PY
)
  read -r CACHE_AT RECEIPT_AT SNAP_AT <<<"$PROOF"
  docker rm -f "$CONTAINER" >/dev/null 2>&1
  docker run --rm --name "$CONTAINER" --gpus all --network host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONPATH=$ROOT/training/neuralese -e HF_HUB_OFFLINE=1 \
    -v $ROOT:$ROOT:ro -v "$(dirname "$(dirname "$(dirname "$SNAP")")")":"$CACHE_AT":ro -v "$RECEIPT":"$RECEIPT_AT":ro \
    -v "$DIRECT":"$DIRECT":ro -w $ROOT "$SERVE_IMAGE" python -m natlang_neuralese.serve \
    --direct-artifact "$DIRECT" --direct-base-snapshot "$SNAP_AT" ${RENDER:+--prompt-rendering $RENDER} \
    --device cuda --dtype bfloat16 --port ${PORT:-8097} > $R/server.log 2>&1 &
else
  $PY -m natlang_neuralese.serve "${MODEL[@]}" --device cuda --dtype bfloat16 --port ${PORT:-8097} > $R/server.log 2>&1 &
fi
SP=$!
for i in $(seq 1 150); do curl -sf http://127.0.0.1:${PORT:-8097}/v1/models >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:${PORT:-8097}/v1/models >/dev/null || { echo "server did not come up; see $R/server.log" >&2; kill $SP; exit 1; }
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
[ -n "${SERVE_IMAGE:-}" ] && docker stop "$CONTAINER" >/dev/null 2>&1
