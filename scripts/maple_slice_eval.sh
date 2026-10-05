#!/usr/bin/env bash
# MAPLE_NESTED N1 (deployed side): cut nested slices of the official Maple GGUF, re-quantize embedding (Q4_K) and
# head (Q6_K), and measure file size, llama.cpp perplexity on fixed natlang text, and CPU speed.
#   scripts/maple_slice_eval.sh OUT_DIR "32 48 64 128"
set -euo pipefail
OUT=${1:?out dir}
SIZES=${2:-"32 48 64 128"}
ROOT=/home/werg/natlang
BIN=/home/werg/llama.cpp-neuralese/build-cpu/bin
FULL=/home/werg/natlang-model-evaluation/models/maple-preview-GGUF/maple-preview-TQ2_0-head-F16.gguf
ORDER=$ROOT/runs/maple-nested-20261005/n0-v1/expert-order.pt
PY=$ROOT/.venv-neuralese/bin/python
THREADS=${THREADS:-16}
mkdir -p "$OUT"
TEXT=$OUT/text.txt
if [ ! -s "$TEXT" ]; then
  "$PY" - "$TEXT" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open("/home/werg/natlang/runs/maple-joint-20261005/qwen3-render-v1.jsonl")][-40:]
open(sys.argv[1], "w").write("\n\n".join(r["prompt"][-3000:] + r["completion"] for r in rows))
EOF
fi
measure() {  # name gguf
  local name=$1 gguf=$2
  local ppl
  ppl=$("$BIN/llama-perplexity" -m "$gguf" -f "$TEXT" -c 1024 -b 1024 --chunks 12 -t "$THREADS" 2>&1 | tee "$OUT/$name.perplexity.log" | grep -o "Final estimate: PPL = [0-9.]*" | grep -o "[0-9.]*$" || true)
  "$BIN/llama-bench" -m "$gguf" -t "$THREADS" -p 512 -n 128 -o json > "$OUT/$name.bench.json" 2> "$OUT/$name.bench.log" || true
  local tg pp
  tg=$("$PY" -c "import json;r=json.load(open('$OUT/$name.bench.json'));print(round([x for x in r if x['n_gen']>0][0]['avg_ts'],1))" 2>/dev/null || echo NA)
  pp=$("$PY" -c "import json;r=json.load(open('$OUT/$name.bench.json'));print(round([x for x in r if x['n_prompt']>0][0]['avg_ts'],1))" 2>/dev/null || echo NA)
  printf '{"name": "%s", "bytes": %s, "ppl": "%s", "gen_tok_s": "%s", "prompt_tok_s": "%s"}\n' "$name" "$(stat -c %s "$gguf")" "$ppl" "$tg" "$pp" | tee -a "$OUT/results.jsonl"
}
measure full-256-f16tables "$FULL"
for n in $SIZES; do
  "$PY" "$ROOT/scripts/maple_slice_gguf.py" --gguf "$FULL" --order "$ORDER" --experts "$n" --out "$OUT/slice-$n.gguf"
  "$BIN/llama-quantize" --allow-requantize --token-embedding-type q4_k --output-tensor-type q6_k \
    "$OUT/slice-$n.gguf" "$OUT/slice-$n-q4emb-q6head.gguf" TQ2_0 > "$OUT/quantize-$n.log" 2>&1
  rm "$OUT/slice-$n.gguf"
  measure "slice-$n" "$OUT/slice-$n-q4emb-q6head.gguf"
done
