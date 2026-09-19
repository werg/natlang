#!/usr/bin/env bash
# Serve a GGUF model on the GPU with llama.cpp's official CUDA image.
# Needs Docker with the NVIDIA runtime; no system CUDA toolkit, no particular glibc.
# Usage: scripts/serve.sh [MODEL.gguf in models/] [PORT]      stop: docker stop natlang-llama
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-LFM2.5-350M-Q8_0.gguf}"
PORT="${2:-8080}"
[ -f "$ROOT/models/$MODEL" ] || { echo "missing models/$MODEL (see README)"; exit 1; }
# The chat template embedded in LiquidAI's GGUF is a reduced one: it drops `tool_calls` from history
# and has no tool-call tokens. If the official template is present, use it instead.
TEMPLATE="$ROOT/models/templates/${MODEL%%-Q*}.jinja"
EXTRA=()
[ -f "$TEMPLATE" ] && EXTRA=(--jinja --chat-template-file "/models/templates/$(basename "$TEMPLATE")")
docker rm -f natlang-llama >/dev/null 2>&1 || true
exec docker run --rm --name natlang-llama --gpus all \
  -v "$ROOT/models:/models:ro" -p "127.0.0.1:$PORT:8080" \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m "/models/$MODEL" --host 0.0.0.0 --port 8080 --parallel 4 -c 32768 -ngl 99 --no-webui "${EXTRA[@]}"
