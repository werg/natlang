#!/bin/sh
# GPU regression suite for the DGX training image (natlang-neuralese:<tag>): fused MoE kernels (incl. trained block
# scales), Maple QAT, FlexAttention branch attention vs the tiled reference (a reported sm_121 Flex miscompile would
# show here), isolated-sequence replay and the QAT weight cache. Runs as the invoking user with /home/werg mounted,
# admitted through the memory ledger. Rerun after any image, Triton, CUDA or kernel change.
#   scripts/neuralese_container_tests.sh [image-tag]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=natlang-neuralese:${1:-26.09}
NAME=natlang-neuralese-test
docker rm -f $NAME >/dev/null 2>&1 || true
docker create --name $NAME --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME="$HOME" -v "$HOME:$HOME" \
  -w "$ROOT/training/neuralese" "$IMAGE" python -m pytest -q -p no:cacheprovider \
  ../../tests/neuralese/test_maple_model.py ../../tests/neuralese/test_isolated_sequence_qwen.py \
  ../../tests/neuralese/test_branch_attention.py ../../tests/neuralese/test_shared_parametrized_weights.py >/dev/null
systemctl --user reset-failed natlang-container-test 2>/dev/null || true
python3 "$ROOT/scripts/memory_ledger.py" run --unit natlang-container-test --budget-gb 6 --reserve-gb 0 --class experiment \
  --wait 1800 --workdir "$ROOT" -- docker start -a $NAME
sleep 5
while [ "$(docker inspect -f '{{.State.Running}}' $NAME)" = true ]; do sleep 5; done
docker logs $NAME 2>&1 | tail -3
exit "$(docker inspect -f '{{.State.ExitCode}}' $NAME)"
