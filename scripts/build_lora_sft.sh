#!/usr/bin/env bash
# Build one model's LoRA training set from admitted teacher runs, rendered with that model's own chat template.
# Usage: scripts/build_lora_sft.sh lfm|ling|spark OUT_DIR RESULTS.jsonl [MORE.jsonl ...]
# NATLANG_HANDOFFS=RUNS.jsonl[,...] adds the preference pairs of handoff runs (build-handoffs.mjs) to preferences.jsonl.
# Ready source-backed, recovered directory-expansion and retired WorkflowEvals static manifests are included automatically.
# sft.jsonl is rendered evidence; use the staged recipe for final token, dedup and holdout auditing.
# NATLANG_STATIC_BUNDLE=MANIFEST[,MANIFEST...] selects ready bundles; =off disables it.
# Correct direct and delegated runs are both admitted, materialized into one training decision per model turn, and rendered by a CPU-only llama-server loaded
# with the model's GGUF and template, so prompts and completions are byte-for-byte what the model sees when served.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# NATLANG_TS_HOST picks the ts-host whose scripts and dist run the build (default: this checkout's), so a build can run
# while a collector uses this checkout's dist.
TS_HOST="${NATLANG_TS_HOST:-$ROOT/ts-host}"
MODEL="${1:?model: lfm, ling or spark}"; OUT="${2:?output directory}"; shift 2
# Discover historical/current completed teacher jobs by default; immutable
# snapshots exclude eval splits and apply current admission, with a reasons ledger.
if [ "${NATLANG_GENERATED_RESULTS:-auto}" != off ]; then
  GENERATED_RESULT="$(node "$TS_HOST/scripts/snapshot-generated-training.mjs" --repo "$ROOT")"
  set -- "$@" "$GENERATED_RESULT"
fi
for REFERENCE_RESULT in "$ROOT/runs/inline-curriculum/ref-v1.results.jsonl" "$ROOT/runs/inline-curriculum/ref-composed-v1.results.jsonl"; do
  [ -f "$REFERENCE_RESULT" ] || continue
  REFERENCE_PRESENT=0
  for INPUT_RESULT in "$@"; do
    [ "$(realpath "$INPUT_RESULT")" = "$REFERENCE_RESULT" ] && REFERENCE_PRESENT=1
  done
  [ "$REFERENCE_PRESENT" = 1 ] || set -- "$@" "$REFERENCE_RESULT"
done
STATIC_MANIFESTS=("$ROOT/data/teacher/source-backed/static.manifest.json" "$ROOT/data/teacher/recovered/static.manifest.json" "$ROOT/data/teacher/directory-expansion/static.manifest.json" "$ROOT/data/teacher/workflowevals/static.manifest.json")
if [ -n "${NATLANG_STATIC_BUNDLE:-}" ]; then
  IFS=',' read -r -a STATIC_MANIFESTS <<< "$NATLANG_STATIC_BUNDLE"
fi
for STATIC_MANIFEST in "${STATIC_MANIFESTS[@]}"; do
  [ "$STATIC_MANIFEST" = off ] && continue
  STATIC_OPTIONAL=(); [ -z "${NATLANG_STATIC_BUNDLE:-}" ] && STATIC_OPTIONAL=(--optional)
  STATIC_RESULT="$(node "$TS_HOST/scripts/inline-curriculum/static-bundle-input.mjs" "$STATIC_MANIFEST" "${STATIC_OPTIONAL[@]}")"
  if [ -n "$STATIC_RESULT" ]; then
    STATIC_ALREADY_PRESENT=0
    for INPUT_RESULT in "$@"; do
      [ "$(realpath "$INPUT_RESULT")" = "$STATIC_RESULT" ] && STATIC_ALREADY_PRESENT=1
    done
    if [ "$STATIC_ALREADY_PRESENT" = 0 ]; then set -- "$@" "$STATIC_RESULT"; fi
    echo "Including validated static source results: $STATIC_RESULT"
  fi
done
[ "$#" -gt 0 ] || { echo "no results files or ready static bundle given"; exit 2; }
INVENTORY_ARGS=()
for INPUT_RESULT in "$@"; do INVENTORY_ARGS+=(--input "$INPUT_RESULT"); done
for STATIC_MANIFEST in "${STATIC_MANIFESTS[@]}"; do
  [ "$STATIC_MANIFEST" = off ] || INVENTORY_ARGS+=(--input "$STATIC_MANIFEST")
done
if [ -n "${NATLANG_STATIC_BUNDLE:-}" ] || [ "${NATLANG_GENERATED_RESULTS:-auto}" = off ]; then
  INVENTORY_ARGS+=(--allow-input-override)
fi
python3 "$ROOT/scripts/inventory_training_data.py" --repo "$ROOT" "${INVENTORY_ARGS[@]}"
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
  --admitted "$OUT/admitted.jsonl" | tail -2 | head -1
# NATLANG_DIRECT_ANSWERS=1 trains answers given without reasoning towards them (a child's judgment, a scripted verdict),
# for a small student that answers directly.
DIRECT=(); [ "${NATLANG_DIRECT_ANSWERS:-}" = 1 ] && DIRECT=(--direct-answers)
node --max-old-space-size=6000 "$TS_HOST/scripts/materialize-native-teacher.mjs" "$OUT/admitted.jsonl" "$OUT/turns.jsonl" --replace "${DIRECT[@]}"
# Corrected variants: where a call failed and then fixed it, the same run with the fix made first, so the student
# learns the right action as well as recovery. Only the fix is trained in a variant.
node --max-old-space-size=6000 "$TS_HOST/scripts/inline-curriculum/corrections.mjs" "$OUT/admitted.jsonl" "$OUT/turns.jsonl" \
  "$OUT/corrected.jsonl" --workers 6
node --max-old-space-size=6000 "$TS_HOST/scripts/materialize-native-teacher.mjs" "$OUT/corrected.jsonl" "$OUT/corrected.turns.jsonl" \
  --replace "${DIRECT[@]}"
cat "$OUT/corrected.turns.jsonl" >> "$OUT/turns.jsonl" && rm "$OUT/corrected.turns.jsonl"
# Preference pairs for a DPO stage after SFT (scripts/train_dpo.py): each corrected fix over the failed attempt it
# replaces, and each handoff teacher's decision over the failed one it was handed.
HANDOFFS=(); [ -n "${NATLANG_HANDOFFS:-}" ] && HANDOFFS=(--handoffs "$NATLANG_HANDOFFS")
node --max-old-space-size=6000 "$TS_HOST/scripts/build-preference-pairs.mjs" "$OUT/pairs.jsonl" \
  --variants "$OUT/corrected.jsonl" --parents "$OUT/admitted.jsonl" "${HANDOFFS[@]}" --workers 6

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" -v "$ROOT/models:/models:ro" -p "127.0.0.1:$PORT:8080" "$IMAGE" \
  -m "/models/$GGUF" --host 0.0.0.0 --port 8080 -ngl 0 -c 512 -np 1 --no-warmup --jinja \
  --chat-template-file "/models/templates/$TEMPLATE" "${KWARG_ARGS[@]}" --no-webui >/dev/null
trap 'docker stop "$NAME" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 90); do curl -fsS "127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
node --max-old-space-size=4000 "$TS_HOST/scripts/export-native-sft.mjs" "$OUT/turns.jsonl" "$OUT/sft.jsonl" \
  --server "http://127.0.0.1:$PORT" --workers 4 --end-token "$END" "${REASONING_END[@]}"
node "$TS_HOST/scripts/export-preference-pairs.mjs" "$OUT/pairs.jsonl" "$OUT/preferences.jsonl" \
  --server "http://127.0.0.1:$PORT" --workers 4 --end-token "$END" "${REASONING_END[@]}"
