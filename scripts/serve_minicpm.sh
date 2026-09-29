#!/usr/bin/env bash
# Isolated Sharp-MiniCPM5 server. CPU by default; --gpu after Bonsai is safely paused.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:---cpu}"
case "$MODE" in --cpu) DEVICE=(-ngl 0); LIMIT=(--memory 3g --cpus 2);; --gpu) DEVICE=(-ngl 99); LIMIT=(--gpus all --memory 4g);; *) echo 'Usage: scripts/serve_minicpm.sh [--cpu|--gpu]' >&2; exit 2;; esac
MODEL="candidates/sharp-minicpm5-2b/Sharp-MiniCPM5-2B-Q4_K_XL.gguf"
[ -f "$ROOT/models/$MODEL" ] || { echo "missing verified model: $MODEL" >&2; exit 1; }
if [ "$MODE" = --gpu ] && docker ps --format '{{.Names}}' | grep -qx natlang-bonsai; then
  echo 'Bonsai is running. Pause generation at a case boundary and stop its server before GPU evaluation.' >&2
  exit 1
fi
exec docker run --rm --name "${NATLANG_MINICPM_SERVER_NAME:-natlang-minicpm}" "${LIMIT[@]}" \
  -v "$ROOT/models:/models:ro" -p "127.0.0.1:${NATLANG_MINICPM_PORT:-8082}:8080" \
  ghcr.io/ggml-org/llama.cpp@sha256:971dccd9ce8f64a81f26800298e37ee34cf111d483d82e2e5ae426a9ceb87f36 \
  -m "/models/$MODEL" --host 0.0.0.0 --port 8080 "${DEVICE[@]}" \
  -c "${NATLANG_MINICPM_CTX:-4096}" -np 1 -t 2 --cache-ram 0 \
  -ctk q8_0 -ctv q8_0 -fa on --jinja \
  --chat-template-file /models/templates/Sharp-MiniCPM5-2B.jinja \
  --chat-template-kwargs '{"terse":false,"enable_thinking":true}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --metrics --no-webui
