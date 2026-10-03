#!/usr/bin/env bash
# Build the S1 sample port records end to end: convert, validate, dedup, protected scan, closure.
# Usage: scripts/neuralese_sample_build.sh [OUT_ROOT] [PYTHON] [TAG]
# PYTHON must have pyarrow (for example /home/werg/bgkit/.venv/bin/python).
set -euo pipefail
OUT_ROOT=${1:-/mnt/external/natlang-development-data/data/neuralese}
PY=${2:-/home/werg/bgkit/.venv/bin/python}
TOOL="$(dirname "$0")/neuralese_port_records.py"
TAG=${3:-samples-$(date -u +%Y%m%dT%H%M%S%N)}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid build tag: use one directory name starting with a letter or digit." >&2
  exit 2
fi
S="$OUT_ROOT/port-records/$TAG"
PROTECTED="$S/protected.json"
for target in "$S" "$S-dedup" "$S-closed" "$S.validate.json" "$S-closed.validate.json"; do
  if [[ -e "$target" || -L "$target" ]]; then
    echo "Build artifact already exists: $target. Choose a new tag." >&2
    exit 2
  fi
done
mkdir -p "$OUT_ROOT/port-records"
mkdir "$S"

nice -n 10 "$PY" "$TOOL" convert-bgkit --limit 3000 --out "$S"
nice -n 10 "$PY" "$TOOL" convert-schnitzel --limit 2500 --out "$S"
nice -n 10 "$PY" "$TOOL" convert-swe --limit 1500 --windows 2 --out "$S"
nice -n 10 "$PY" "$TOOL" convert-turns --limit 300 --out "$S"
nice -n 10 "$PY" "$TOOL" convert-agents --limit 300 --out "$S"
nice -n 10 "$PY" "$TOOL" protected --output "$PROTECTED"
nice -n 10 "$PY" "$TOOL" validate --schema --report "$S.validate.json" "$S" > /dev/null
nice -n 10 "$PY" "$TOOL" dedup "$S" --out "$S-dedup"
nice -n 10 "$PY" "$TOOL" close "$S" --links "$S-dedup/dedup.links.jsonl" --drop "$S-dedup/dedup.drop.json" \
  --protected "$PROTECTED" --out "$S-closed"
nice -n 10 "$PY" "$TOOL" validate --report "$S-closed.validate.json" "$S-closed" > /dev/null
echo "sample build complete: $S-closed"
