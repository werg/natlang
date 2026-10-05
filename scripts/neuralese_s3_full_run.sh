#!/bin/sh
# The S3 full run (plans/neuralese/S3_FULL_RUN_PLAN.md): phases A–F on the S1 final subset (scripts/neuralese_s3_subset.py),
# max length 64 at 16 tokens per vector, the harness at every phase boundary, resumable (rerun with the same RUN).
# Needs the GPU to itself: launch only after the owner has paused the teacher campaign, through the ledger:
#   python3 scripts/memory_ledger.py run --unit natlang-s3-full --budget-gb 48 --class experiment --wait 3600 \
#     --workdir /home/werg/natlang -- scripts/neuralese_s3_full_run.sh [RUN] [extra pilot options, e.g. --stop-after-phase A]
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:-$REPO/runs/neuralese-s3-full-20261005}
[ $# -gt 0 ] && shift
RECORDS=${NATLANG_S3_RECORDS:-/home/werg/data/neuralese-s3-full/records}
TRAIN=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(','.join(f for f,v in sorted(m['families'].items()) if v['train']))" "$RECORDS/manifest.json")
EVAL=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(','.join(f for f,v in sorted(m['families'].items()) if v['eval']))" "$RECORDS/manifest.json")
mkdir -p "$RUN"
cd "$REPO/training/neuralese"
exec "$REPO/.venv-neuralese/bin/python" -m natlang_neuralese.train.pilot --out "$RUN" --records "$RECORDS" --stream \
  --families "$TRAIN" --eval-families "$EVAL" --eval-per-family 64 \
  --phase-steps A=3000,B=3000,C=4000,D=30000,E=4000,F=6000 --harness-phases A,B,C,D,E,F --checkpoint-every 500 \
  --max-length 64 --tokens-per-vector 16 --span-lengths 8,12,16,24,32 --stop-source final --stop-exploration 0.5 \
  --optimizer muon --fail-on-shortcut --memory-gb 44 --seed 0 "$@" >> "$RUN.log" 2>&1
