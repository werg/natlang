#!/usr/bin/env bash
# Build the full S1 port-record set: convert every migrated family into raw/, then one streaming
# finalize pass (exact and near dedup, split closure, protected scan, validation, manifest) into final/.
# Usage: scripts/neuralese_full_build.sh [OUT_ROOT] [PYTHON] [TAG]
# PYTHON must have pyarrow (for example /home/werg/bgkit/.venv/bin/python). Runs at low CPU and IO priority.
set -euo pipefail
OUT_ROOT=${1:-/mnt/external/natlang-development-data/data/neuralese}
PY=${2:-/home/werg/bgkit/.venv/bin/python}
TAG=${3:-full-$(date -u +%Y%m%dT%H%M%S%N)}
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid build tag: use one directory name starting with a letter or digit." >&2
  exit 2
fi
TOOL="$(dirname "$0")/neuralese_port_records.py"
S="$OUT_ROOT/port-records/$TAG"
PROTECTED="$S/protected.json"
run() { echo "== $(date -Is) $*"; ionice -c 3 nice -n 15 "$PY" "$TOOL" "$@"; }

mkdir -p "$OUT_ROOT/port-records"
# Atomic reservation: preserve completed and interrupted builds alike.
if ! mkdir "$S"; then
  echo "Build directory already exists or cannot be created: $S. Choose a new tag." >&2
  exit 2
fi
mkdir "$S/raw"
run protected --output "$PROTECTED"
run convert-bgkit --out "$S/raw"
run convert-schnitzel --out "$S/raw"
run convert-turns --out "$S/raw"
run convert-swe --windows 2 --out "$S/raw"
run convert-agents --out "$S/raw"
run finalize "$S/raw" --protected "$PROTECTED" --out "$S/final"
echo "full build complete: $S/final"
