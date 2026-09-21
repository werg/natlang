#!/usr/bin/env bash
# Serve Prism ML's Ternary Bonsai 2 27B as a teacher, following the lab's own instructions
# (github.com/PrismML-Eng/Bonsai-demo): their llama.cpp FORK (mainline cannot read PTQ1_0), their
# official chat template (not the one embedded in the GGUF), q4_0 KV cache, a capped thinking budget.
# The fork's prebuilt binaries lack the CUDA and OpenMP runtimes, so they run inside a small image built from
# docker/prism.Dockerfile (docker build -t natlang-prism-runtime -f docker/prism.Dockerfile docker).
# Usage: scripts/serve_bonsai.sh [PORT] [CTX] [NGL] [SLOTS]  stop: docker stop natlang-bonsai
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8081}"; CTX="${2:-32768}"; NGL="${3:-99}"; SLOTS="${4:-${BONSAI_SLOTS:-1}}"
# A long Bonsai context occupies roughly 0.6-0.9 GiB in the host prompt cache.
# Agent programs alternate between root and nested invocations, so one entry
# per slot thrashes even with only two workers.  Keep about two contexts per
# slot; BONSAI_CACHE_RAM remains available for memory-constrained machines.
if [ "$SLOTS" -gt 1 ]; then DEFAULT_CACHE_RAM=$((SLOTS * 1536)); else DEFAULT_CACHE_RAM=1536; fi
CACHE_RAM="${BONSAI_CACHE_RAM:-$DEFAULT_CACHE_RAM}"
MODEL="Ternary-Bonsai-2-27B-PTQ1_0.gguf"
[ -f "$ROOT/models/$MODEL" ] || { echo "missing models/$MODEL"; exit 1; }
[ -x "$ROOT/vendor/prism/bin/llama-server" ] || { echo "missing vendor/prism/bin (see README)"; exit 1; }
docker rm -f natlang-bonsai >/dev/null 2>&1 || true
# --no-mmap: weights go straight to the GPU instead of staying mapped in host RAM; the memory cap protects the desktop.
exec docker run --rm --name natlang-bonsai --gpus all --memory "${BONSAI_MEM:-5g}" --memory-swap "${BONSAI_MEM:-5g}" \
  -v "$ROOT/vendor/prism/bin:/prism:ro" -v "$ROOT/models:/models:ro" -e LD_LIBRARY_PATH=/prism \
  -p "127.0.0.1:$PORT:8080" natlang-prism-runtime \
  /prism/llama-server -m "/models/$MODEL" --host 0.0.0.0 --port 8080 \
    -ngl "$NGL" -fa on -c "$CTX" --cache-type-k q4_0 --cache-type-v q4_0 -np "$SLOTS" --no-mmap --cache-ram "$CACHE_RAM" --metrics \
    --jinja --chat-template-file /models/templates/Ternary-Bonsai-2-27B.jinja \
    --temp 1.0 --top-p 0.95 --top-k 20 --reasoning-budget 1024 --no-webui
