#!/bin/sh
# vLLM-with-Neuralese parity and throughput in the DGX vLLM image. Usage: scripts/neuralese_vllm_parity.sh HEADS_CHECKPOINT [lora]
# Run through the ledger: python3 scripts/memory_ledger.py run --unit natlang-vllm-parity --budget-gb 13 --class experiment -- scripts/neuralese_vllm_parity.sh HEADS
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEPS=${NATLANG_VLLM_PYDEPS:-$HOME/.cache/natlang-vllm-pydeps}
mkdir -p "$DEPS"
exec docker run --rm --gpus all -v "$REPO":"$REPO" -v "$DEPS":/pydeps -v "$(dirname "$1")":"$(dirname "$1")" \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint sh "${VLLM_IMAGE:-vllm-node:latest}" -c "[ -d /pydeps/peft ] || pip install -q --no-deps --target /pydeps accelerate peft; PYTHONPATH=/pydeps python3 $REPO/scripts/neuralese_vllm_parity.py $1 ${2:-}"
