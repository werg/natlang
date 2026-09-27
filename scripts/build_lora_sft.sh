#!/usr/bin/env bash
# Build one model's LoRA training set from admitted teacher runs, rendered with that model's own chat template.
# Usage: scripts/build_lora_sft.sh lfm|ling|spark OUT_DIR RESULTS.jsonl [MORE.jsonl ...]
# Runs are admitted with --require-technique (a correct run that skipped a required technique teaches the wrong
# thing), materialized into one training decision per model turn, and rendered by a CPU-only llama-server loaded
# with the model's GGUF and template, so prompts and completions are byte-for-byte what the model sees when served.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# NATLANG_TS_HOST picks the ts-host whose scripts and dist run the build (default: this checkout's), so a build can run
# while a collector uses this checkout's dist.
TS_HOST="${NATLANG_TS_HOST:-$ROOT/ts-host}"
MODEL="${1:?model: lfm, ling or spark}"; OUT="${2:?output directory}"; shift 2
[ "$#" -gt 0 ] || { echo "no results files given"; exit 2; }
# REASONING_END closes the template's reasoning: reasoning no model wrote (curriculum references) is masked through it.
REASONING_END=()
case "$MODEL" in
  lfm)  GGUF=candidates/lfm25-8b-a1b/LFM2.5-8B-A1B-Q4_K_M.gguf; TEMPLATE=LFM2.5-8B-A1B-stateless.jinja; END='<|im_end|>' ;;
  ling) GGUF=candidates/ling3-tiny/Ling-3.0-tiny-Q4_K_M.gguf; TEMPLATE=Ling-3.0-tiny.jinja; END='<|role_end|>'
    REASONING_END=(--reasoning-end '</think>') ;;
  spark) GGUF=candidates/sharp-spark-x25-4b/Sharp-Spark-X2.5-4B-Q4_K_XL.gguf; TEMPLATE=Sharp-Spark-X2.5-4B.jinja
    END='<｜end▁of▁sentence｜>'; REASONING_END=(--reasoning-end '</think>') ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
IMAGE="$(grep -o 'ghcr.io/ggml-org/llama.cpp@sha256:[0-9a-f]*' "$ROOT/scripts/serve.sh")"
PORT="${NATLANG_TEMPLATE_PORT:-8083}"; NAME="natlang-template-$MODEL"
# The template's options, as scripts/serve.sh passes them when serving.
KWARGS="$ROOT/models/templates/${TEMPLATE%.jinja}.kwargs.json"
KWARG_ARGS=(); [ -f "$KWARGS" ] && KWARG_ARGS=(--chat-template-kwargs "$(cat "$KWARGS")")
mkdir -p "$OUT"
node "$TS_HOST/scripts/inline-curriculum/admit.mjs" "$@" --ledger "$OUT/admission.jsonl" \
  --admitted "$OUT/admitted.jsonl" --require-technique | tail -2 | head -1
node --max-old-space-size=6000 "$TS_HOST/scripts/materialize-native-teacher.mjs" "$OUT/admitted.jsonl" "$OUT/turns.jsonl" --replace

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" -v "$ROOT/models:/models:ro" -p "127.0.0.1:$PORT:8080" "$IMAGE" \
  -m "/models/$GGUF" --host 0.0.0.0 --port 8080 -ngl 0 -c 512 -np 1 --no-warmup --jinja \
  --chat-template-file "/models/templates/$TEMPLATE" "${KWARG_ARGS[@]}" --no-webui >/dev/null
trap 'docker stop "$NAME" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 90); do curl -fsS "127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
node --max-old-space-size=4000 "$TS_HOST/scripts/export-native-sft.mjs" "$OUT/turns.jsonl" "$OUT/sft.jsonl" \
  --server "http://127.0.0.1:$PORT" --workers 4 --end-token "$END" "${REASONING_END[@]}"
