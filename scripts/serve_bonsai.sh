#!/usr/bin/env bash
# Serve Prism ML's Ternary Bonsai 2 27B as a teacher, following the lab's own instructions
# (github.com/PrismML-Eng/Bonsai-demo): their llama.cpp FORK (mainline cannot read PTQ1_0), their
# official chat template (not the one embedded in the GGUF), q4_0 KV cache, a capped thinking budget.
# The fork's prebuilt binaries lack the CUDA and OpenMP runtimes, so they run inside a small image built from
# docker/prism.Dockerfile (docker build -t natlang-prism-runtime -f docker/prism.Dockerfile docker).
# Usage: scripts/serve_bonsai.sh [PORT] [CTX] [NGL] [SLOTS]  (collectors: --workers to match SLOTS)  stop: docker stop natlang-bonsai
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Defaults: 6 slots sharing a 52k-token KV buffer, the measured fit for the 8 GB RTX 4060 (each slot holds about
# 145 MB of SSM state; 8 slots do not fit). About 1.6x one slot's decode throughput, more per collected case.
PORT="${1:-8081}"; CTX="${2:-53248}"; NGL="${3:-99}"; SLOTS="${4:-${BONSAI_SLOTS:-6}}"
# A long Bonsai context occupies roughly 0.6-0.9 GiB in the host prompt cache.
# Agent programs alternate between root and nested invocations, so one entry
# per slot thrashes even with only two workers.  Keep about two contexts per
# slot; BONSAI_CACHE_RAM remains available for memory-constrained machines.
# The server process itself holds about 3 GiB of host memory besides this cache, and the container is killed at
# BONSAI_MEM, so the cache stays at least 5 GiB below the cap (measured with 6 slots: a 4 GiB cache under 7g was killed).
if [ "$SLOTS" -gt 1 ]; then DEFAULT_CACHE_RAM=$((SLOTS * 1536)); DEFAULT_MEM=8; else DEFAULT_CACHE_RAM=1536; DEFAULT_MEM=5; fi
MEM="${BONSAI_MEM:-${DEFAULT_MEM}g}"
MAX_CACHE_RAM=$(( (${MEM%g} - 5) * 1024 )); [ "$MAX_CACHE_RAM" -lt 1024 ] && MAX_CACHE_RAM=1024
[ "$DEFAULT_CACHE_RAM" -gt "$MAX_CACHE_RAM" ] && DEFAULT_CACHE_RAM=$MAX_CACHE_RAM
CACHE_RAM="${BONSAI_CACHE_RAM:-$DEFAULT_CACHE_RAM}"
# Several slots share one KV buffer, so a long call can use more than an even share of the context.
if [ "$SLOTS" -gt 1 ]; then KV_UNIFIED=--kv-unified; else KV_UNIFIED=; fi
MODEL="Ternary-Bonsai-2-27B-PTQ1_0.gguf"
[ -f "$ROOT/models/$MODEL" ] || { echo "missing models/$MODEL"; exit 1; }
[ -x "$ROOT/vendor/prism/bin/llama-server" ] || { echo "missing vendor/prism/bin (see README)"; exit 1; }
CONTAINER="${BONSAI_SERVER_NAME:-natlang-bonsai}"
if [ -z "${BONSAI_SERVER_NAME:-}" ]; then
  docker rm -f natlang-bonsai >/dev/null 2>&1 || true
fi
# --no-mmap: weights go straight to the GPU instead of staying mapped in host RAM; the memory cap protects the desktop.
exec docker run --rm --name "$CONTAINER" --gpus all --memory "$MEM" --memory-swap "$MEM" \
  -v "$ROOT/vendor/prism/bin:/prism:ro" -v "$ROOT/models:/models:ro" -e LD_LIBRARY_PATH=/prism \
  -p "127.0.0.1:$PORT:8080" natlang-prism-runtime \
  /prism/llama-server -m "/models/$MODEL" --host 0.0.0.0 --port 8080 \
    -ngl "$NGL" -fa on -c "$CTX" --cache-type-k q4_0 --cache-type-v q4_0 -np "$SLOTS" $KV_UNIFIED --no-mmap --cache-ram "$CACHE_RAM" --metrics \
    --jinja --chat-template-file /models/templates/Ternary-Bonsai-2-27B.jinja \
    --temp 1.0 --top-p 0.95 --top-k 20 --reasoning-budget 1024 --no-webui
