#!/usr/bin/env bash
# Docker GPU training followed by host-side GGUF conversion and default publication.
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo 'usage: train_and_publish_browser_docker.sh DATA RUN NAME STEPS [BASE_MODEL]' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA="$1"; RUN="$2"; NAME="$3"; STEPS="$4"
if [ "$#" -eq 5 ]; then
  docker run --rm --gpus all -v "$ROOT:/work" -e HF_HOME=/work/models/hf natlang-train \
    python scripts/train_lora.py "$DATA" "$RUN" --steps "$STEPS" --model "$5"
else
  docker run --rm --gpus all -v "$ROOT:/work" -e HF_HOME=/work/models/hf natlang-train \
    python scripts/train_lora.py "$DATA" "$RUN" --steps "$STEPS"
fi
node scripts/publish_browser_model.mjs --run "$RUN" --name "$NAME" --quant Q4_K_M
