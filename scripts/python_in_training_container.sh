#!/usr/bin/env bash
# Use the Pop training image for the Python half of check-main when the host's
# general-purpose Python lacks the pinned Neuralese dependencies. Invoke only
# from a memory-ledger unit and use that same unit name for the container.
set -euo pipefail

unit=${NATLANG_PYTHON_CONTAINER_UNIT:?set to the enclosing memory-ledger unit name}
repo=$(cd "$(dirname "$0")/.." && pwd)
image=${NATLANG_PYTHON_CONTAINER_IMAGE:-natlang-neuralese:26.09-pop}

exec docker run --rm --name "$unit" --gpus all --network host \
  --user "$(id -u):$(id -g)" \
  --volume "$repo:$repo" \
  --volume /srv/storage:/srv/storage \
  --volume /tmp:/tmp \
  --volume /usr/bin/node:/usr/bin/node:ro \
  --volume /usr/bin/zstd:/usr/bin/zstd:ro \
  --workdir "$repo" \
  --env "PYTHONPATH=$repo/training/neuralese:$repo" \
  --env "HF_HOME=${HF_HOME:-/tmp/natlang-check-hf-empty}" \
  --env "NATLANG_NEURALESE_DEVICE=${NATLANG_NEURALESE_DEVICE:-cuda}" \
  --entrypoint python "$image" "$@"
