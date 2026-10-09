#!/bin/sh
# Harness-bench records from SWE-rebench OpenHands trajectories (plans/neuralese/HARNESS_BENCH.md):
# agent surface -> normalize + join base commits -> replay in pi's tools -> native training records.
# Usage: harness_bench_build.sh OUT [LIMIT] [OFFSET]   (run it under scripts/memory_ledger.py run)
# Resumable: surface.json and prepared.jsonl are written under a temporary name and renamed when complete, so a stage
# whose output exists is done and skipped; replay keeps the trajectories replayed.jsonl holds and appends the rest.
set -eu
OUT=$1; LIMIT=${2:-0}; OFFSET=${3:-0}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY="$ROOT/.venv-neuralese/bin/python"
CORPORA="$ROOT/data/neuralese/corpora"
WORKSPACE=/workspace/project
mkdir -p "$OUT"
export PATH="$HOME/.local/bin:$PATH"
if [ ! -e "$OUT/surface.json" ]; then
  node "$ROOT/ts-host/bin/natlang.mjs" run "$ROOT/applications/pi" -- surface --cwd "$WORKSPACE" --companion > "$OUT/surface.json.tmp"
  mv "$OUT/surface.json.tmp" "$OUT/surface.json"
fi
cd "$ROOT/training/neuralese"
if [ ! -e "$OUT/prepared.jsonl" ]; then
  "$PY" -m natlang_neuralese.harness_bench.prepare --trajectories "$CORPORA/nebius-swe-rebench-openhands-trajectories-20261009-v1/trajectories.parquet" \
    --tasks "$CORPORA/nebius-swe-rebench-tasks-20261009-v1" --workspace "$WORKSPACE" --resolved-only \
    --limit "$LIMIT" --offset "$OFFSET" --out "$OUT/prepared.jsonl.tmp"
  mv "$OUT/prepared.jsonl.tmp" "$OUT/prepared.jsonl"
fi
node "$ROOT/ts-host/bin/natlang.mjs" run "$ROOT/applications/pi" -- replay --in "$OUT/prepared.jsonl" --out "$OUT/replayed.jsonl" \
  2>> "$OUT/replay.log"
"$PY" -m natlang_neuralese.harness_bench.records --replayed "$OUT/replayed.jsonl" --surface "$OUT/surface.json" \
  --corpus nebius-swe-rebench-openhands-trajectories-20261009-v1 --out "$OUT"
