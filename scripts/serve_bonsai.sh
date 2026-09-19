#!/usr/bin/env bash
# Serve Prism ML's Ternary Bonsai 2 27B as a teacher, following the lab's own instructions
# (github.com/PrismML-Eng/Bonsai-demo): their llama.cpp FORK (mainline cannot read PTQ1_0), their
# official chat template (not the one embedded in the GGUF), q4_0 KV cache, a capped thinking budget.
# The fork's prebuilt binaries lack the CUDA and OpenMP runtimes, so they run inside a small image built from
# docker/prism.Dockerfile (docker build -t natlang-prism-runtime -f docker/prism.Dockerfile docker).
# Usage: scripts/serve_bonsai.sh [PORT] [CTX] [NGL]          stop: docker stop natlang-bonsai
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8081}"; CTX="${2:-8192}"; NGL="${3:-99}"
MODEL="Ternary-Bonsai-2-27B-PTQ1_0.gguf"
[ -f "$ROOT/models/$MODEL" ] || { echo "missing models/$MODEL"; exit 1; }
[ -x "$ROOT/vendor/prism/bin/llama-server" ] || { echo "missing vendor/prism/bin (see README)"; exit 1; }
docker rm -f natlang-bonsai >/dev/null 2>&1 || true
exec docker run --rm --name natlang-bonsai --gpus all \
  -v "$ROOT/vendor/prism/bin:/prism:ro" -v "$ROOT/models:/models:ro" -e LD_LIBRARY_PATH=/prism \
  -p "127.0.0.1:$PORT:8080" natlang-prism-runtime \
  /prism/llama-server -m "/models/$MODEL" --host 0.0.0.0 --port 8080 \
    -ngl "$NGL" -fa on -c "$CTX" --cache-type-k q4_0 --cache-type-v q4_0 -np 1 \
    --jinja --chat-template-file /models/templates/Ternary-Bonsai-2-27B.jinja \
    --temp 1.0 --top-p 0.95 --top-k 20 --reasoning-budget 1024 --no-webui
