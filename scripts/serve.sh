#!/usr/bin/env bash
# Serve a GGUF model on the GPU with llama.cpp's official CUDA image.
# Needs Docker with the NVIDIA runtime; no system CUDA toolkit, no particular glibc.
# Usage: scripts/serve.sh [MODEL.gguf in models/] [PORT] [SLOTS] [CTX]  stop: docker stop natlang-llama
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-LFM2.5-350M-Q8_0.gguf}"
PORT="${2:-8080}"
SLOTS="${3:-${NATLANG_SLOTS:-4}}"
CTX="${4:-${NATLANG_CTX:-32768}}"
CACHE_RAM="${NATLANG_CACHE_RAM:-$((SLOTS * 256))}"
IMAGE="${NATLANG_LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp@sha256:971dccd9ce8f64a81f26800298e37ee34cf111d483d82e2e5ae426a9ceb87f36}"
[ -f "$ROOT/models/$MODEL" ] || { echo "missing models/$MODEL (see README)"; exit 1; }
# The chat template embedded in LiquidAI's GGUF is a reduced one: it drops `tool_calls` from history
# and has no tool-call tokens. If the official template is present, use it instead.
MODEL_STEM="$(basename "${MODEL%%-Q*}")"
TEMPLATE="${NATLANG_TEMPLATE:-$ROOT/models/templates/${MODEL_STEM}.jinja}"
EXTRA=()
[ -f "$TEMPLATE" ] && EXTRA=(--jinja --chat-template-file "/models/templates/$(basename "$TEMPLATE")")
DRAFT_MODEL="${NATLANG_DRAFT_MODEL:-}"
DRAFT=()
if [ -n "$DRAFT_MODEL" ]; then
  [ -f "$ROOT/models/$DRAFT_MODEL" ] || { echo "missing models/$DRAFT_MODEL"; exit 1; }
  DRAFT=(-md "/models/$DRAFT_MODEL" --spec-type draft-dspark --spec-draft-n-max 10 --spec-draft-n-min 0)
fi
docker rm -f natlang-llama >/dev/null 2>&1 || true
exec docker run --rm --name natlang-llama --gpus all \
  -v "$ROOT/models:/models:ro" -p "127.0.0.1:$PORT:8080" \
  "$IMAGE" \
  -m "/models/$MODEL" --host 0.0.0.0 --port 8080 --parallel "$SLOTS" -c "$CTX" -ngl 99 --cache-ram "$CACHE_RAM" --metrics --no-webui "${DRAFT[@]}" "${EXTRA[@]}"
