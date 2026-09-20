#!/usr/bin/env bash
# Train or resume, export the merged checkpoint, then promote a browser Q4_K_M artifact.
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo 'usage: train_and_publish_browser.sh DATA RUN NAME STEPS [BASE_MODEL]' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA="$1"; RUN="$2"; NAME="$3"; STEPS="$4"
if [ "$#" -eq 5 ]; then
  python scripts/train_lora.py "$DATA" "$RUN" --steps "$STEPS" --model "$5"
else
  python scripts/train_lora.py "$DATA" "$RUN" --steps "$STEPS"
fi
node scripts/publish_browser_model.mjs --run "$RUN" --name "$NAME" --quant Q4_K_M
