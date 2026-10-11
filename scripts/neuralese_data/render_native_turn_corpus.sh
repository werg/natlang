#!/bin/bash
# Docker wrapper for render_native_turn_corpus.py (CPU only, no network, natlang-neuralese image).
# Run under the DGX memory ledger, e.g.:
#   python3 scripts/memory_ledger.py run --unit render-turns-<id> --budget-gb 6 --priority low --resumable -- \
#     scripts/neuralese_data/render_native_turn_corpus.sh OUT_DIR --corpus-id ID --source F ... [--workers N]
# The LFM2.5 and Mellum2.1 tokenizers are the pinned snapshots of the existing twin builders.
set -euo pipefail
REPO=/home/werg/natlang
LFM=$REPO/models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f
MELLUM=/home/werg/data/models/mellum21-12b-a2.5b-thinking
OUT=$(realpath "$1"); shift
mkdir -p "$OUT"
# Named container, stopped with the unit: a ledger preemption (or systemctl stop) must not leave an orphan
# writing into OUT. A relaunch with the same OUT resumes from the finished shards.
NAME="render-$(basename "$OUT")"
docker rm -f "$NAME" >/dev/null 2>&1 || true
trap 'docker stop -t 30 "$NAME" >/dev/null 2>&1' TERM INT
docker run --rm --name "$NAME" --network none --user "$(id -u):$(id -g)" --memory "${MEM:-24g}" --cpus "${CPUS:-16}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= -e TOKENIZERS_PARALLELISM=false -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:$REPO:ro" -v /mnt/external:/mnt/external:ro -v /home/werg/data:/home/werg/data:ro -v "$OUT:$OUT" \
  -w "$REPO" --entrypoint python3 natlang-neuralese:26.09 \
  scripts/neuralese_data/render_native_turn_corpus.py \
  --tokenizer lfm25="$LFM" --tokenizer mellum21="$MELLUM" --out-dir "$OUT" "$@" &
wait $!; status=$?
wait $! 2>/dev/null || true
exit $status
