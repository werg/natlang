#!/usr/bin/env bash
# A Python interpreter for check-main.sh on hosts whose Python lacks the
# training stack. Run the surrounding check under memory_ledger.py with
# --container matching NATLANG_PYTHON_CONTAINER, so Docker RAM/VRAM is charged
# to that unit and the container stops with it.
set -euo pipefail
: "${NATLANG_PYTHON_CONTAINER:?Declare this container with memory_ledger.py --container}"
: "${NATLANG_PYTHON_IMAGE:?Pin the installed training image for this check}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
args=(--rm --name "$NATLANG_PYTHON_CONTAINER" --pull never --gpus all
  --network host --ipc host --user "$(id -u):$(id -g)"
  -e HOME=/home/werg -e HF_HOME -e HF_HUB_OFFLINE -e PYTHONPATH
  -v /home/werg:/home/werg:rw -v /tmp:/tmp:rw
  -v /srv/storage:/srv/storage:ro -v /usr/bin/node:/usr/bin/node:ro
  -w "$ROOT")
if [ -d /mnt/external ]; then args+=(-v /mnt/external:/mnt/external:ro); fi
exec docker run "${args[@]}" "$NATLANG_PYTHON_IMAGE" python "$@"
