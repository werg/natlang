#!/usr/bin/env bash
# Convert a merged Hugging Face checkpoint (scripts/train_lora.py) to GGUF.
# Usage: scripts/to_gguf.sh runs/lora-v5/merged models/natlang-350M-v5-Q8_0.gguf [Q8_0|Q4_K_M|Q5_K_M|Q6_K]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$1"; DST="$2"; QUANT="${3:-Q8_0}"
case "$QUANT" in Q8_0|Q4_K_M|Q5_K_M|Q6_K) ;; *) echo "Unsupported quantization: $QUANT" >&2; exit 2 ;; esac
if [ ! -f "$ROOT/vendor/llama.cpp/convert_hf_to_gguf.py" ]; then          # the converter only: a 3 MB sparse checkout
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp "$ROOT/vendor/llama.cpp"
  git -C "$ROOT/vendor/llama.cpp" sparse-checkout set --no-cone /convert_hf_to_gguf.py /gguf-py /conversion /requirements
fi
if [ "$QUANT" = Q8_0 ]; then
  docker run --rm -v "$ROOT:/work" -e PYTHONPATH=/work/vendor/llama.cpp/gguf-py --entrypoint bash natlang-train -o pipefail -c \
    "python vendor/llama.cpp/convert_hf_to_gguf.py '$SRC' --outfile '$DST' --outtype q8_0 2>&1 | tail -3"
else
  TMP="${DST%.gguf}-F16-intermediate.gguf"
  trap 'rm -f "$ROOT/$TMP"' EXIT
  docker run --rm -v "$ROOT:/work" -e PYTHONPATH=/work/vendor/llama.cpp/gguf-py --entrypoint bash natlang-train -o pipefail -c \
    "python vendor/llama.cpp/convert_hf_to_gguf.py '$SRC' --outfile '$TMP' --outtype f16 2>&1 | tail -3"
  llama-quantize "$ROOT/$TMP" "$ROOT/$DST" "$QUANT"
fi
# serve.sh looks for models/templates/<name before -Q>.jinja: the tuned model keeps the official template
NAME="$(basename "$DST")"; cp -n "$ROOT/models/templates/LFM2.5-350M.jinja" "$ROOT/models/templates/${NAME%%-Q*}.jinja"
