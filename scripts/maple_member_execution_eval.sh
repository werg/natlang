#!/usr/bin/env bash
# MAPLE_NESTED §4a / §9: protected execution evaluation of trained family members, as deployed. Same 24-case packet,
# runner, CPU llama.cpp server settings and chat template as the official Maple baseline
# (runs/maple-preview-evaluation-20261005/execution-v5: 12 complete successes of 24). CPU only; each server goes
# through the memory ledger.
#   scripts/maple_member_execution_eval.sh STATE OUT_DIR "24x64 24x32 8x16" [PORT]
set -euo pipefail
STATE=$(realpath "${1:?nested-state.pt}")
OUT=$(realpath -m "${2:?out dir}")
MEMBERS=${3:-"full 24x64 24x32 8x16"}
PORT=${4:-18095}
ROOT=/home/werg/natlang
BASE=$ROOT/runs/maple-preview-evaluation-20261005
BIN=/home/werg/llama.cpp-neuralese/build-cpu/bin
SERVER=$BASE/dgx-neuralese-binaries/llama-server  # the server build of the official baseline (b11402-5d999c0c5)
FULL=/home/werg/natlang-model-evaluation/models/maple-preview-GGUF/maple-preview-TQ2_0-head-F16.gguf
PY=$ROOT/.venv-neuralese/bin/python
mkdir -p "$OUT"
for member in $MEMBERS; do
  model=$OUT/$member-q.gguf
  if [ ! -f "$model" ]; then
    (cd "$ROOT/training/neuralese" && "$PY" -m natlang_neuralese.maple.export_member --gguf "$FULL" --state "$STATE" \
      --member "$member" --out "$OUT/$member.gguf")
    "$BIN/llama-quantize" --allow-requantize --token-embedding-type q4_k --output-tensor-type q6_k \
      "$OUT/$member.gguf" "$model" TQ2_0 > "$OUT/$member.quantize.log" 2>&1
    rm "$OUT/$member.gguf"
  fi
  [ -d "$OUT/execution-$member" ] && { echo "$member: already evaluated"; continue; }
  unit=natlang-maple-member-server-${member/x/-}
  python3 "$ROOT/scripts/memory_ledger.py" run --unit "$unit" --budget-gb 14 --class experiment --wait 3600 \
    --workdir "$ROOT" -- "$SERVER" -m "$model" --host 127.0.0.1 --port "$PORT" \
    --alias "maple-member-$member" --jinja --chat-template-file "$BASE/runtime-dgx-chat-template.jinja" \
    -c 16384 -np 1 -t 8 -tb 16 -ngl 0
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do sleep 5; done
  plan=$OUT/plan-$member.json
  "$PY" - "$BASE/evaluation-plan-v5.json" "$plan" "$model" "$member" "$STATE" "$PORT" "$OUT" <<'EOF'
import hashlib, json, sys
source, target, model, member, state, port, out = sys.argv[1:]
plan = json.load(open(source))
h = hashlib.sha256()
with open(model, "rb") as f:
    for block in iter(lambda: f.read(8 << 20), b""):
        h.update(block)
plan.update(output=f"{out}/execution-{member}", endpoint=f"http://127.0.0.1:{port}", model=f"maple-member-{member}",
            model_path=model, model_sha256=h.hexdigest(),
            server_properties_receipt=f"{out}/execution-{member}-server-properties.json")
plan["candidate_identity"] = {**plan["candidate_identity"], "variant": f"nested member {member}",
                              "training_applied": True, "nested_state": state,
                              "formats": "TQ2_0 body, Q4_K embedding, Q6_K head"}
json.dump(plan, open(target, "w"), indent=1)
EOF
  sha=$(sha256sum "$plan" | cut -d' ' -f1)
  node "$BASE/runner-v5/evaluate-interpreter-candidate.mjs" "$plan" --execute "$sha" > "$OUT/execution-$member.log" 2>&1 || true
  systemctl --user stop "$unit"
  "$PY" - "$OUT" "$member" <<'EOF'
import collections, json, sys
out, member = sys.argv[1:]
rows = [json.loads(l) for l in open(f"{out}/execution-{member}/case-results.jsonl")]
counts = collections.Counter(r["disposition"] for r in rows)
record = {"member": member, "cases": len(rows), **counts}
print(json.dumps(record))
open(f"{out}/results.jsonl", "a").write(json.dumps(record) + "\n")
EOF
done
