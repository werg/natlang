#!/usr/bin/env bash
# Build the S1 sample port records end to end: convert, validate, dedup, protected scan, closure.
# Usage: scripts/neuralese_sample_build.sh [OUT_ROOT] [PYTHON]
# PYTHON must have pyarrow (for example /home/werg/bgkit/.venv/bin/python).
set -euo pipefail
OUT_ROOT=${1:-/mnt/external/natlang-development-data/data/neuralese}
PY=${2:-/home/werg/bgkit/.venv/bin/python}
TOOL="$(dirname "$0")/neuralese_port_records.py"
TAG=samples-$(date +%Y%m%d)
S="$OUT_ROOT/port-records/$TAG"

nice -n 10 "$PY" "$TOOL" convert-bgkit --limit 3000 --out "$S"
nice -n 10 "$PY" "$TOOL" convert-schnitzel --limit 2500 --out "$S"
nice -n 10 "$PY" "$TOOL" convert-swe --limit 1500 --windows 2 --out "$S"
nice -n 10 "$PY" "$TOOL" convert-turns --limit 300 --out "$S"
nice -n 10 "$PY" "$TOOL" convert-agents --limit 300 --out "$S"
nice -n 10 "$PY" "$TOOL" protected --output "$OUT_ROOT/protected/bgkit-benchmarks.protected.json"
nice -n 10 "$PY" "$TOOL" validate --schema --report "$S.validate.json" "$S" > /dev/null
nice -n 10 "$PY" "$TOOL" dedup "$S" --out "$S-dedup"
rm -rf "$S-closed"
nice -n 10 "$PY" "$TOOL" close "$S" --links "$S-dedup/dedup.links.jsonl" --drop "$S-dedup/dedup.drop.json" \
  --protected "$OUT_ROOT/protected/bgkit-benchmarks.protected.json" --out "$S-closed"
nice -n 10 "$PY" "$TOOL" validate --report "$S-closed.validate.json" "$S-closed" > /dev/null
echo "sample build complete: $S-closed"
