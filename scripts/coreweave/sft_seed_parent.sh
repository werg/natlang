#!/bin/bash
# Fresh production SFT prefix in worker 0, followed by a verified full-state coordinator parent.
# Usage: sft_seed_parent.sh FRESH_OUT ARGS.json PARENT.pt [STOP_AT=50]
set -euo pipefail
OUT=${1:?fresh output directory required}
ARGS=${2:?trainer arguments JSON required}
PARENT=${3:?full parent path required}
STOP_AT=${4:-50}
python3 - "$OUT" "$ARGS" "$PARENT" "$STOP_AT" <<'PY'
import json, os, sys
from pathlib import Path
out, args, parent = map(Path, sys.argv[1:4]); stop = int(sys.argv[4])
if stop < 1 or out.exists() or parent.exists():
    raise ValueError('fresh seed output and absent parent are required; STOP_AT must be positive')
argv = json.loads(args.read_text())
if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
    raise ValueError('trainer arguments must be a JSON list of strings')
for flag in ('--out', '--continue-from', '--diloco'):
    if any(x == flag or x.startswith(flag + '=') for x in argv):
        raise ValueError('fresh seed cannot declare ' + flag)
def option(name, default):
    values = [x.split('=', 1)[1] for x in argv if x.startswith(name + '=')]
    values += [argv[i+1] for i, x in enumerate(argv[:-1]) if x == name]
    if len(values) > 1: raise ValueError('duplicate ' + name)
    return values[0] if values else default
if option('--base', '') != '@@BASE@@':
    raise ValueError('seed must use the pinned LFM3 base marker @@BASE@@')
if float(option('--epochs', '1')) != 1.0:
    raise ValueError('seed must use the one-epoch production recipe')
interval = int(option('--checkpoint-every', '0'))
if interval < 1 or interval > stop:
    raise ValueError('seed needs checkpoint-every between 1 and STOP_AT')
PY
child=''
cleanup() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
CHECK_PHASE_B=0 "${NATLANG_SFT_CHECK_SCRIPT:-scripts/coreweave/lfm3_sft_check.sh}" "$OUT" "$ARGS" "$STOP_AT" &
child=$!
wait "$child"
child=''
python3 scripts/coreweave/sft_seed_parent_verify.py "$OUT" "$PARENT" "$STOP_AT" "$ARGS"
