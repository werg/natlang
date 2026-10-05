#!/usr/bin/env bash
# MAPLE_NESTED §4a: evaluate trained family members as deployed models. For each member of a nested-state.pt
# checkpoint (and the trained full model), export to GGUF exactly as trained, re-quantize embedding (Q4_K) and head
# (Q6_K), and measure size, llama.cpp perplexity on the same natlang text as the untrained slices
# (/home/werg/data/maple-slices/text.txt, results.jsonl there), and CPU speed.
#   scripts/maple_member_eval.sh STATE OUT_DIR "full 24x32 24x64 8x16"
set -euo pipefail
STATE=${1:?nested-state.pt}
OUT=${2:?out dir}
MEMBERS=${3:-"full 24x32 24x64 8x16"}
ROOT=/home/werg/natlang
BIN=/home/werg/llama.cpp-neuralese/build-cpu/bin
FULL=/home/werg/natlang-model-evaluation/models/maple-preview-GGUF/maple-preview-TQ2_0-head-F16.gguf
TEXT=/home/werg/data/maple-slices/text.txt
PY=$ROOT/.venv-neuralese/bin/python
THREADS=${THREADS:-16}
mkdir -p "$OUT"
for member in $MEMBERS; do
  (cd "$ROOT/training/neuralese" && "$PY" -m natlang_neuralese.maple.export_member --gguf "$FULL" --state "$STATE" \
    --member "$member" --out "$OUT/$member.gguf")
  "$BIN/llama-quantize" --allow-requantize --token-embedding-type q4_k --output-tensor-type q6_k \
    "$OUT/$member.gguf" "$OUT/$member-q.gguf" TQ2_0 > "$OUT/$member.quantize.log" 2>&1
  rm "$OUT/$member.gguf"
  ppl=$("$BIN/llama-perplexity" -m "$OUT/$member-q.gguf" -f "$TEXT" -c 1024 -b 1024 --chunks 12 -t "$THREADS" 2>&1 \
    | tee "$OUT/$member.perplexity.log" | grep -o "Final estimate: PPL = [0-9.]*" | grep -o "[0-9.]*$" || true)
  "$BIN/llama-bench" -m "$OUT/$member-q.gguf" -t "$THREADS" -p 512 -n 128 -o json > "$OUT/$member.bench.json" 2> /dev/null || true
  speed=$("$PY" -c "import json;r=json.load(open('$OUT/$member.bench.json'));print(round([x for x in r if x['n_gen']>0][0]['avg_ts'],1))" 2>/dev/null || echo NA)
  printf '{"member": "%s", "state": "%s", "bytes": %s, "ppl": "%s", "gen_tok_s": "%s"}\n' "$member" "$STATE" \
    "$(stat -c %s "$OUT/$member-q.gguf")" "$ppl" "$speed" | tee -a "$OUT/results.jsonl"
done
