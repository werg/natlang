#!/usr/bin/env bash
# Convert and evaluate a completed LFM 8B LoRA run. This never starts corpus generation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN="${1:?usage: evaluate_lfm_teacher_adapter.sh RUN_DIR [LABEL] [CHECKPOINT_DIR]}"
LABEL="${2:-$(basename "$RUN")}"
CHECKPOINT="${3:-$RUN/checkpoint}"
STATE="$CHECKPOINT/state.json"
ADAPTER="$CHECKPOINT/weights"
GGUF="$CHECKPOINT/adapter-f16.gguf"
RESULTS="runs/evaluations/$LABEL"
PROBE="$RESULTS/applications.json"
TURN_MANIFEST="$RESULTS/turns.manifest.json"
TURN_RESULTS="$RESULTS/turns.json"
READINESS="$RESULTS/readiness.json"
BASE_CONFIG="runs/lfm25-8b-base-config"
BASE_GGUF="candidates/lfm25-8b-a1b/LFM2.5-8B-A1B-Q4_K_M.gguf"

python - "$STATE" "${3:+snapshot}" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    raise SystemExit(f"missing training state: {p}")
s = json.loads(p.read_text())
snapshot = len(sys.argv) > 2 and sys.argv[2] == "snapshot"
if snapshot and s.get("step", 0) < 1:
    raise SystemExit(f"empty snapshot: {p}")
if not snapshot and (s.get("step", 0) < s.get("corpus", {}).get("steps", 10**18) or "heldout_after" not in s):
    raise SystemExit(f"training is not complete: step={s.get('step')}, heldout_after={s.get('heldout_after')}")
PY

mkdir -p "$RESULTS"
if [ ! -f "$GGUF" ]; then
  docker run --rm -v "$ROOT:/work" -w /work \
    -e PYTHONPATH=/work/vendor/llama.cpp/gguf-py --entrypoint python natlang-train \
    vendor/llama.cpp/convert_lora_to_gguf.py "$ADAPTER" --base "$BASE_CONFIG" \
    --outfile "$GGUF" --outtype f16
fi

cleanup() {
  docker stop -t 20 natlang-llama >/dev/null 2>&1 || true
  wait "${server_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT
NATLANG_LORA="$GGUF" scripts/serve.sh "$BASE_GGUF" 8080 4 32768 &
server_pid=$!
for _ in $(seq 1 90); do
  curl -fsS http://127.0.0.1:8080/health >/dev/null && break
  sleep 1
done
curl -fsS http://127.0.0.1:8080/health >/dev/null

.venv/bin/python scripts/application_probe.py --n 3 --workers 4 --seconds 120 \
  --validation-feedback caller --write-constraints runtime --out "$PROBE"
.venv/bin/python scripts/eval_turns.py data/ref-v8-failures.jsonl --per-cell 2 \
  --manifest "$TURN_MANIFEST" --out "$TURN_RESULTS" --model-label "$LABEL"

python - "$PROBE" "$TURN_RESULTS" "$READINESS" "$LABEL" <<'PY'
import json, pathlib, sys
from collections import Counter
probe_path, turns_path, out_path, label = map(pathlib.Path, sys.argv[1:])
probe = json.loads(probe_path.read_text())
turns = json.loads(turns_path.read_text())
rows = probe["rows"]
correct = sum(bool(row["correct"]) for row in rows)
by_family = Counter()
totals = Counter()
for row in rows:
    totals[row["family"]] += 1
    by_family[row["family"]] += bool(row["correct"])
ready = correct >= 7 and all(by_family[name] >= 2 for name in totals)
result = {
    "schema": "natlang.teacher_readiness/1",
    "model": str(label),
    "ready_for_generation": ready,
    "whole_program": {"correct": correct, "total": len(rows),
                      "correct_by_family": dict(by_family), "total_by_family": dict(totals)},
    "diagnostic_turns": turns["summary"],
    "application_probe": str(probe_path),
    "turn_results": str(turns_path),
}
out_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
