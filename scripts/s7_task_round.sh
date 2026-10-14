#!/bin/bash
# usage: scripts/s7_task_round.sh RUN_DIR [SHARDS]  (env: RENDER, PORT, SNAP, DIRECT=export dir,
#   SERVE_IMAGE + RECEIPT + CONTAINER: serve DIRECT in a container at its source-proof paths;
#   TRAINING_CHECKPOINT + TRAINING_HEADS + pinned SHA-256 values + SERVE_IMAGE + CONTAINER:
#     serve a direct-stage training checkpoint through the in-run evaluator's exact restore path;
#   UPDATE_FROM=<scored run dir>: one GRPO round instead of a rollout round; RESTORE_ADAPTER=<adapter.nzb>: sample with it;
#   CONTROL_DIR=<prepared run dir>: concurrent no-adapter arm; PROBE_ROLLOUTS=<rollouts.jsonl>: decode tok/s probe)
# S7 smoke round (plans/neuralese/S7_RL.md §8): serve LFM2.5-350M on GPU, run the group queue through the collector in SHARDS
# parallel queue runners, merge journals, score.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=$1; N=${2:-1}
SNAP=${SNAP:-$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f}
PY=${NEURALESE_PYTHON:-$ROOT/.venv-neuralese/bin/python}
if [ ! -x "$PY" ]; then PY=python3; fi
export PYTHONPATH=$ROOT/training/neuralese HF_HUB_OFFLINE=1
if [ -n "${DIRECT:-}" ]; then  # a direct-SFT export serves with its own declared rendering unless RENDER overrides
  MODEL=(--direct-artifact "$DIRECT" --direct-base-snapshot "$SNAP" ${RENDER:+--prompt-rendering $RENDER})
else
  MODEL=(--base "$SNAP" --prompt-rendering "${RENDER:-full}")
fi
if [ -n "${TRAINING_CHECKPOINT:-}" ]; then
  : "${TRAINING_HEADS:?TRAINING_HEADS is required with TRAINING_CHECKPOINT}"
  : "${TRAINING_CHECKPOINT_SHA256:?TRAINING_CHECKPOINT_SHA256 is required}"
  : "${TRAINING_HEADS_SHA256:?TRAINING_HEADS_SHA256 is required}"
  : "${SERVE_IMAGE:?SERVE_IMAGE is required with TRAINING_CHECKPOINT}"
  : "${CONTAINER:?CONTAINER is required with TRAINING_CHECKPOINT}"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run --rm --name "$CONTAINER" --gpus all --network host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONPATH=$ROOT/training/neuralese -e HF_HUB_OFFLINE=1 \
    -v /home/werg:/home/werg:ro -v /srv/storage:/srv/storage:ro -w "$ROOT" "$SERVE_IMAGE" \
    python scripts/neuralese_data/serve_direct_checkpoint.py \
      --checkpoint "$TRAINING_CHECKPOINT" --checkpoint-sha256 "$TRAINING_CHECKPOINT_SHA256" \
      --heads "$TRAINING_HEADS" --heads-sha256 "$TRAINING_HEADS_SHA256" \
      --prompt-rendering "${RENDER:-full}" --device cuda --port "${PORT:-8097}" > "$R/server.log" 2>&1 &
elif [ -n "${SERVE_IMAGE:-}" ] && [ -z "${DIRECT:-}" ]; then
  # Keep the matched base and trained-checkpoint arms on the same image,
  # tokenizer snapshot, dtype, prompt rendering, and HTTP collector surface.
  : "${CONTAINER:?CONTAINER is required with SERVE_IMAGE}"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run --rm --name "$CONTAINER" --gpus all --network host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONPATH=$ROOT/training/neuralese -e HF_HUB_OFFLINE=1 \
    -v /home/werg:/home/werg:ro -v /srv/storage:/srv/storage:ro -w "$ROOT" "$SERVE_IMAGE" \
    python -m natlang_neuralese.serve --base "$SNAP" --prompt-rendering "${RENDER:-full}" \
      --device cuda --dtype bfloat16 --port "${PORT:-8097}" > "$R/server.log" 2>&1 &
elif [ -n "${SERVE_IMAGE:-}" ]; then
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
stop_server() {
  if [ -n "${SERVE_IMAGE:-}" ]; then
    # The docker CLI is the background process here. SIGTERM followed by wait
    # can hang while the container is still serving, retaining its GPU claim.
    docker stop --timeout 2 "$CONTAINER" >/dev/null 2>&1 || true
  else
    kill "$SP" 2>/dev/null || true
  fi
  wait "$SP" 2>/dev/null || true
}
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
collector_status=0
for pid in "${pids[@]}"; do
  wait "$pid" || collector_status=1
done
if [ "$collector_status" -ne 0 ]; then
  # A case that exhausts its declared model-request budget still has an exact
  # terminal journal record and is a scored rollout. A host wall-clock limit
  # also has an exact terminal record, but reward_of leaves it unscored because
  # the limit depends on serving load. Preserve both; fail closed for missing
  # terminal records, transport errors and accounting failures.
  python3 - "$R" <<'PY' || {
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
entries = [json.loads(line) for line in (r / "queue.jsonl").read_text().splitlines() if line.strip()]
finishes = {}
for path in sorted((r / "shards").glob("journal-*.jsonl")):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "finish":
            if row["key"] in finishes:
                raise SystemExit(f"duplicate rollout finish: {row['key']}")
            finishes[row["key"]] = row
keys = {entry["key"] for entry in entries}
if set(finishes) != keys:
    raise SystemExit(f"rollout finish coverage mismatch: missing={sorted(keys-set(finishes))}, extra={sorted(set(finishes)-keys)}")
for key, row in finishes.items():
    if row.get("status") in {"complete", "complete_with_skips"}:
        continue
    if row.get("status") == "incomplete" and row.get("resource_limit_reason") == "model_request_budget":
        continue
    if (row.get("status") == "timeout" and row.get("resource_limit_reason") == "case_wall_budget" and
            row.get("output_accounting", {}).get("exact_result_rows") == 0):
        continue
    raise SystemExit(f"rollout collector failed: {key}: {row.get('status')}: {row.get('resource_limit_reason')}")
print("all rollout collectors terminated with exact results or declared resource limits")
PY
    echo "one or more rollout collectors failed; inspect shard logs" >&2
    stop_server
    exit 1
  }
fi
cat "$R"/shards/journal-*.jsonl > "$R/journal.jsonl" || {
  echo "rollout journals could not be merged" >&2
  stop_server
  exit 1
}
$PY -m natlang_neuralese.rl.task_rollouts score --run-dir $R > $R/score.log 2>&1 || {
  echo "rollout scoring failed; inspect $R/score.log" >&2
  stop_server
  exit 1
}
if [ -n "${CONTROL_DIR:-}" ]; then
  cat "$CONTROL_DIR"/shards/journal-*.jsonl > "$CONTROL_DIR/journal.jsonl" || {
    echo "control rollout journals could not be merged" >&2
    stop_server
    exit 1
  }
  $PY -m natlang_neuralese.rl.task_rollouts score --run-dir $CONTROL_DIR > $CONTROL_DIR/score.log 2>&1 || {
    echo "control rollout scoring failed; inspect $CONTROL_DIR/score.log" >&2
    stop_server
    exit 1
  }
fi
stop_server
