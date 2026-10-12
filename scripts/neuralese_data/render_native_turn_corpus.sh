#!/bin/bash
# Docker wrapper for render_native_turn_corpus.py (CPU only, no network, natlang-neuralese image).
# Run under the DGX memory ledger, e.g.:
#   python3 scripts/memory_ledger.py run --unit render-turns-<id> --budget-gb 6 --priority low --resumable -- \
#     scripts/neuralese_data/render_native_turn_corpus.sh OUT_DIR --corpus-id ID --source F ... [--workers N]
# The LFM2.5 and Mellum2.1 tokenizers are the pinned snapshots of the existing twin builders.
set -euo pipefail
# A stop (ledger preemption, systemctl stop) is a clean resumable stop: the container is stopped and the wrapper
# exits 0, so the unit ends with Result=success and the ledger requeues it; the relaunch resumes finished shards.
# Trapped before anything else: a stop seconds after launch, before the container exists, is also a clean stop (the
# handler's `docker stop` of a missing container failed under set -e and the unit ended with status 1, 2026-10-12).
stopped=0
trap 'stopped=1; docker stop -t 30 "${NAME:-}" >/dev/null 2>&1 || true' TERM INT
REPO=/home/werg/natlang
LFM=$REPO/models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f
MELLUM=/home/werg/data/models/mellum21-12b-a2.5b-thinking
OUT=$(realpath "$1"); shift
mkdir -p "$OUT"
# Named container, stopped with the unit: a ledger preemption (or systemctl stop) must not leave an orphan
# writing into OUT. A relaunch with the same OUT resumes from the finished shards.
# The name carries a hash of OUT: parts of different corpora share basenames (parts/p01), and `docker rm -f` of a
# same-named container would kill another corpus's running render (2026-10-12).
NAME="render-$(basename "$OUT")-$(printf %s "$OUT" | sha1sum | cut -c1-10)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
ERRLOG=$(mktemp "${TMPDIR:-/tmp}/render-stderr.XXXXXX")
trap 'rm -f "$ERRLOG"' EXIT
# The image has no Node; candidate-selection admission runs scripts/verify_candidate_selection_packet.mjs (needs a
# built ts-host/dist), so the host's Node binary is mounted in.
NODE_BIN=$(readlink -f "$(command -v node || echo /home/werg/.local/bin/node)")
[ $stopped = 1 ] && exit 0  # stopped before the container started: do not start it
docker run --rm --name "$NAME" --network none --user "$(id -u):$(id -g)" --memory "${MEM:-24g}" --cpus "${CPUS:-16}" \
  -v "$NODE_BIN:/usr/local/bin/node:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= -e TOKENIZERS_PARALLELISM=false -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:$REPO:ro" -v /mnt/external:/mnt/external:ro -v /home/werg/data:/home/werg/data:ro -v "$OUT:$OUT" \
  -w "$REPO" --entrypoint python3 natlang-neuralese:26.09 \
  scripts/neuralese_data/render_native_turn_corpus.py \
  --tokenizer lfm25="$LFM" --tokenizer mellum21="$MELLUM" --out-dir "$OUT" "$@" \
  2> >(tee "$ERRLOG" >&2) &
PID=$!
wait $PID && status=0 || status=$?  # (set -e must not end the script here)
wait $PID 2>/dev/null || true
sleep 1
[ $stopped = 1 ] && exit 0
# The ledger/systemd repeats SIGTERM to every process of the unit: the docker client gives up after the third
# ("got 3 SIGTERM/SIGINTs, forcefully exiting", exit 1) even when this script was not signalled first. That is
# still a preemption, a clean resumable stop (2026-10-12).
if [ $status != 0 ] && grep -q 'SIGTERM/SIGINTs, forcefully exiting' "$ERRLOG" 2>/dev/null; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  exit 0
fi
exit $status
