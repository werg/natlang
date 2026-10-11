#!/bin/bash
# Mellum2.1 tokenizer twin of the 4962-row Neuralese foundation text input (LFM2.5 tokenizer).
# Run under the DGX memory ledger, e.g.:
#   python3 scripts/memory_ledger.py run --unit twin-mellum4962-<id> --budget-gb 8 -- \
#     scripts/neuralese_data/text_tokenizer_twin_mellum4962.sh OUT_DIR [--workers N] [--limit N]
# OUT_DIR must exist and be empty. CPU only, no network, inside the natlang-neuralese image.
set -euo pipefail
REPO=/home/werg/natlang
MELLUM=/home/werg/data/models/mellum21-12b-a2.5b-thinking
OUT=$(realpath "$1"); shift
mapfile -t RECORDS < "$REPO/scripts/neuralese_data/text_tokenizer_twin_mellum4962.natrecords"
exec docker run --rm --network none --user "$(id -u):$(id -g)" --memory "${MEM:-8g}" --cpus "${CPUS:-8}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= -e TOKENIZERS_PARALLELISM=false -e HOME=/tmp \
  -v "$REPO:$REPO:ro" -v "$MELLUM:$MELLUM:ro" -v "$OUT:$OUT" -w "$REPO" --entrypoint python3 natlang-neuralese:26.09 \
  scripts/neuralese_data/text_tokenizer_twin.py \
  --source-text runs/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/text-data.jsonl \
  --source-tokenizer "$REPO/models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f" \
  --twin-tokenizer "$MELLUM" \
  --renderer-package-root "$REPO/runs/neuralese-gold-text-serving-boundary-refresh-20261009-v1/publication-candidate-v3/code/renderer-snapshot-v2" \
  --native-records "${RECORDS[@]}" \
  --out-dir "$OUT" "$@"
