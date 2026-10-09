#!/bin/sh
# Harness-bench records from SWE-rebench OpenHands trajectories (plans/neuralese/HARNESS_BENCH.md):
# agent surface -> normalize + join base commits -> replay in pi's tools -> native training records.
# Usage: harness_bench_build.sh OUT [LIMIT] [OFFSET]   (run it under scripts/memory_ledger.py run)
set -eu
OUT=$1; LIMIT=${2:-0}; OFFSET=${3:-0}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY="$ROOT/.venv-neuralese/bin/python"
CORPORA="$ROOT/data/neuralese/corpora"
WORKSPACE=/workspace/project
mkdir -p "$OUT"
export PATH="$HOME/.local/bin:$PATH"
node "$ROOT/ts-host/bin/natlang.mjs" run "$ROOT/applications/pi" -- surface --cwd "$WORKSPACE" --companion > "$OUT/surface.json"
cd "$ROOT/training/neuralese"
"$PY" -m natlang_neuralese.harness_bench.prepare --trajectories "$CORPORA/nebius-swe-rebench-openhands-trajectories-20261009-v1/trajectories.parquet" \
  --tasks "$CORPORA/nebius-swe-rebench-tasks-20261009-v1" --workspace "$WORKSPACE" --resolved-only \
  --limit "$LIMIT" --offset "$OFFSET" --out "$OUT/prepared.jsonl"
node "$ROOT/ts-host/bin/natlang.mjs" run "$ROOT/applications/pi" -- replay --in "$OUT/prepared.jsonl" --out "$OUT/replayed.jsonl" \
  2> "$OUT/replay.log"
"$PY" -m natlang_neuralese.harness_bench.records --replayed "$OUT/replayed.jsonl" --surface "$OUT/surface.json" \
  --corpus nebius-swe-rebench-openhands-trajectories-20261009-v1 --out "$OUT"
