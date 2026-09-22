#!/usr/bin/env bash
# Action-focused correction pass from the last sound teacher adapter, followed by readiness evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${1:-data/lfm25-v8-weakness-live-action-balanced-v2.train.jsonl}"
START_ADAPTER="${2:-runs/lfm25-teacher-v8-phase2-from0273/checkpoint/weights}"
OUT="${3:-runs/lfm25-teacher-v8-live-action-balanced-v2}"
LABEL="${4:-lfm25-8b-live-action-balanced-v2}"

docker rm -f natlang-llama natlang-train-action >/dev/null 2>&1 || true
docker run --rm --gpus all -v "$ROOT:/work" -w /work -e HF_HOME=/work/models/hf \
  --name natlang-train-action natlang-train python scripts/train_lora.py \
  "$DATA" "$OUT" \
  --model models/candidates/lfm25-8b-a1b-nf4 --load-in-4bit --unsloth-lfm-experts \
  --accum 8 --microbatch 1 --batch-tokens 6144 --max-len 6144 --rank 32 \
  --init-adapter "$START_ADAPTER" --steps 1200 --lr 3e-5 --holdout 400 \
  --save-every 50 --snapshot-every 100 \
  --token-cache data/tokens-lfm25-v8-live-action-balanced-v2.sqlite

exec scripts/evaluate_lfm_teacher_adapter.sh "$OUT" "$LABEL"
