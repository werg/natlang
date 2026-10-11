#!/bin/bash
# Mellum2.1 tokenizer twin of one admitted LFM2.5 ordinary-text packet (free-provider packets, D8), with its
# context_only_spans re-expressed in Mellum coordinates. CPU, docker, under the DGX memory ledger:
#   python3 scripts/memory_ledger.py run --unit twin-<id> --budget-gb 4 --priority low -- \
#     scripts/neuralese_data/text_tokenizer_twin_packet.sh OUT_DIR SOURCE_TEXT NATIVE_RECORDS... [-- --workers N]
set -euo pipefail
REPO=/home/werg/natlang
MELLUM=/home/werg/data/models/mellum21-12b-a2.5b-thinking
OUT=$(realpath "$1"); SRC=$2; shift 2
RECORDS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do RECORDS+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
mkdir -p "$OUT"
exec docker run --rm --name "twin-$(basename "$OUT")" --network none --user "$(id -u):$(id -g)" --memory "${MEM:-6g}" --cpus "${CPUS:-4}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= -e TOKENIZERS_PARALLELISM=false -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:$REPO:ro" -v /mnt/external:/mnt/external:ro -v /home/werg/data:/home/werg/data:ro -v "$OUT:$OUT" -w "$REPO" \
  --entrypoint python3 natlang-neuralese:26.09 scripts/neuralese_data/text_tokenizer_twin.py \
  --source-text "$SRC" \
  --source-tokenizer "$REPO/models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f" \
  --twin-tokenizer "$MELLUM" --renderer-package-root "$REPO/training/neuralese" \
  --native-records "${RECORDS[@]}" --out-dir "$OUT" "$@"
