#!/bin/bash
# usage: scripts/s7_task_round.sh RUN_DIR [SHARDS]  (env: RENDER, PORT, SNAP, DIRECT=export dir,
#   SERVE_IMAGE + RECEIPT + CONTAINER: serve DIRECT in a container at its source-proof paths;
#   UPDATE_FROM=<scored run dir>: one GRPO round instead of a rollout round; RESTORE_ADAPTER=<adapter.nzb>: sample with it;
#   CONTROL_DIR=<prepared run dir>: concurrent no-adapter arm; PROBE_ROLLOUTS=<rollouts.jsonl>: decode tok/s probe)
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
URL=http://127.0.0.1:${PORT:-8097}
stop_server() { kill $SP 2>/dev/null; wait $SP 2>/dev/null; [ -n "${SERVE_IMAGE:-}" ] && docker stop "$CONTAINER" >/dev/null 2>&1; true; }
if [ -n "${UPDATE_FROM:-}" ]; then
  # One GRPO round on a scored run (S7 §5): zero adapter (policy == reference), the run's rollouts as behaviour samples,
  # natlang.rollout/1 records from the collector trajectories, adapter saved to $R/policy-r1/adapter.nzb.
  $PY -m natlang_neuralese.rl.task_rollouts score --run-dir "$UPDATE_FROM" --emit-rollouts $R/rollouts.jsonl > $R/emit.log 2>&1
  $PY -m natlang_neuralese.rl.grpo init --server $URL --kind ${ADAPTER_KIND:-xs} --rank ${ADAPTER_RANK:-8} --out $R/policy-r0 > $R/init.log 2>&1
  A0=$(python3 -c "import json;print(json.load(open('$R/policy-r0/adapter.json'))['adapter'])")
  $PY -m natlang_neuralese.rl.grpo round --server $URL --rollouts $R/rollouts.jsonl --adapter $A0 --out $R/policy-r1 \
    --lr ${LR:-1e-3} --beta ${BETA:-0.02} --steps ${STEPS:-1} > $R/round.log 2>&1
  status=$?; stop_server; exit $status
fi
CHAT=()
if [ -n "${RESTORE_ADAPTER:-}" ]; then  # sample with a saved adapter policy (re-score after an update)
  AID=$($PY -c "from natlang_neuralese.rl.grpo import Server; from pathlib import Path; print(Server('$URL').restore_block(Path('$RESTORE_ADAPTER')))")
  echo "{\"x_natlang_adapters\": [{\"id\": \"$AID\", \"scale\": 1}]}" > $R/chat-config.json
  CHAT=(--chat-request-config $R/chat-config.json)
fi
if [ -n "${PROBE_ROLLOUTS:-}" ] && [ -n "${AID:-}" ]; then  # decode tok/s with and without the adapter, alternating
  python3 $ROOT/scripts/s7_decode_probe.py $URL "$PROBE_ROLLOUTS" "$AID" $R/decode-probe.json > $R/decode-probe.log 2>&1
fi
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
    --model-concurrency 2 "${CHAT[@]}" > $R/shards/queue-$s.log 2>&1 &
  pids+=($!)
done
if [ -n "${CONTROL_DIR:-}" ]; then
  # Matched control arm without the adapter (same rows and seeds), run concurrently on the same server so both arms
  # see the same GPU contention.
  mkdir -p $CONTROL_DIR/shards
  python3 - "$CONTROL_DIR" "$N" <<'PY'
import sys
r, n = sys.argv[1], int(sys.argv[2])
rows = [l for l in open(f"{r}/queue.jsonl") if l.strip()]
for s in range(n):
    open(f"{r}/shards/queue-{s}.jsonl", "w").write("".join(rows[s::n]))
PY
  for s in $(seq 0 $((N-1))); do
    python3 $ROOT/scripts/run_bonsai_queue.py $CONTROL_DIR/shards/queue-$s.jsonl $CONTROL_DIR/shards/journal-$s.jsonl \
      --runtime $ROOT/ts-host --server $URL --chat-completions-url $URL/v1/chat/completions --model-id lfm \
      --model-concurrency 2 > $CONTROL_DIR/shards/queue-$s.log 2>&1 &
    pids+=($!)
  done
fi
wait "${pids[@]}"
cat $R/shards/journal-*.jsonl > $R/journal.jsonl
$PY -m natlang_neuralese.rl.task_rollouts score --run-dir $R > $R/score.log 2>&1
if [ -n "${CONTROL_DIR:-}" ]; then
  cat $CONTROL_DIR/shards/journal-*.jsonl > $CONTROL_DIR/journal.jsonl
  $PY -m natlang_neuralese.rl.task_rollouts score --run-dir $CONTROL_DIR > $CONTROL_DIR/score.log 2>&1
fi
stop_server
