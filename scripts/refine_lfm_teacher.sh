#!/usr/bin/env bash
# Weakness-focused curriculum training, program-biased teacher polish, and readiness gate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_ADAPTER="${1:-runs/lfm25-teacher-v8-phase2-from0273/checkpoint/weights}"
HARD_OUT="${2:-runs/lfm25-teacher-v8-weakness-mix}"
POLISH_OUT="${3:-runs/lfm25-teacher-v8-weakness-program-polish}"
LABEL="${4:-lfm25-8b-weakness-program-polish}"
BASE="models/candidates/lfm25-8b-a1b-nf4"
COMMON=(--model "$BASE" --load-in-4bit --unsloth-lfm-experts
        --accum 8 --microbatch 1 --batch-tokens 6144 --max-len 6144
        --rank 32)

docker rm -f natlang-train-hard >/dev/null 2>&1 || true
docker run --rm --gpus all -v "$ROOT:/work" -w /work -e HF_HOME=/work/models/hf \
  --name natlang-train-hard natlang-train python scripts/train_lora.py \
  data/lfm25-v8-weakness-mix-v1.train.jsonl "$HARD_OUT" \
  "${COMMON[@]}" --init-adapter "$START_ADAPTER" --steps 1200 --lr 5e-5 \
  --holdout 400 --save-every 50 --snapshot-every 100 \
  --token-cache data/tokens-lfm25-v8-weakness-mix-v1.sqlite

docker rm -f natlang-train-polish >/dev/null 2>&1 || true
docker run --rm --gpus all -v "$ROOT:/work" -w /work -e HF_HOME=/work/models/hf \
  --name natlang-train-polish natlang-train python scripts/train_lora.py \
  data/lfm25-teacher-program-polish-v1.train.jsonl "$POLISH_OUT" \
  "${COMMON[@]}" --init-adapter "$HARD_OUT/checkpoint/weights" --epochs 2 --lr 2e-5 \
  --holdout 0 --save-every 25 --snapshot-every 25 \
  --token-cache data/tokens-lfm25-v8-weakness-program-polish.sqlite

exec scripts/evaluate_lfm_teacher_adapter.sh "$POLISH_OUT" "$LABEL"
