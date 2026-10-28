#!/bin/bash
# Tokenizer twin of one admitted LFM2.5 ordinary-text packet (free-provider packets, D8), with its context_only_spans
# re-expressed in the twin's coordinates. The twin student is TWIN (scripts/neuralese_data/student_tokenizers.py;
# default lfm3, the current line; TWIN=mellum21 was the original use). CPU, docker, under the DGX memory ledger:
#   python3 scripts/memory_ledger.py run --unit twin-<id> --budget-gb 4 --priority low -- \
#     scripts/neuralese_data/text_tokenizer_twin_packet.sh OUT_DIR SOURCE_TEXT NATIVE_RECORDS... [-- --workers N]
set -euo pipefail
REPO=/home/werg/natlang
SELF_DIR=$(cd "$(dirname "$0")" && pwd)
TWIN=${TWIN:-lfm3}
SOURCE_TOKENIZER=$(python3 "$SELF_DIR/student_tokenizers.py" path lfm25)
TWIN_TOKENIZER=$(python3 "$SELF_DIR/student_tokenizers.py" path "$TWIN")  # verifies the twin's pinned bytes
MOUNTS_TEXT=$(python3 "$SELF_DIR/student_tokenizers.py" mounts "$TWIN")
read -r -a TOKENIZER_MOUNTS <<< "$MOUNTS_TEXT"
OUT=$(realpath "$1"); SRC=$2; shift 2
RECORDS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do RECORDS+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
mkdir -p "$OUT"
exec docker run --rm --name "twin-$(basename "$OUT")" --network none --user "$(id -u):$(id -g)" --memory "${MEM:-6g}" --cpus "${CPUS:-4}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= -e TOKENIZERS_PARALLELISM=false -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:$REPO:ro" -v /mnt/external:/mnt/external:ro -v /home/werg/data:/home/werg/data:ro -v "$OUT:$OUT" \
  "${TOKENIZER_MOUNTS[@]}" -w "$REPO" \
  --entrypoint python3 natlang-neuralese:26.09 scripts/neuralese_data/text_tokenizer_twin.py \
  --source-text "$SRC" \
  --source-tokenizer "$SOURCE_TOKENIZER" \
  --twin-tokenizer "$TWIN_TOKENIZER" --renderer-package-root "$REPO/training/neuralese" \
  --native-records "${RECORDS[@]}" --out-dir "$OUT" "$@"
