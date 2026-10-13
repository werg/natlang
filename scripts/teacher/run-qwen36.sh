#!/bin/sh
# Run spec of the DGX vLLM teacher: container natlang-qwen36-nvfp4-server serving nvidia/Qwen3.6-35B-A3B-NVFP4 on
# 127.0.0.1:8082. Active infrastructure, never retired (storage-policy.json keep.docker): teacher data generation,
# live measurements (plans/MODEL_FACING_CHANGES.md), the specializer and the weekly NatLang teacher window use it.
#
#   run-qwen36.sh image    pull the pinned image if `vllm-node` is missing or not the pinned image ID
#   run-qwen36.sh create   image + create the container (not started) if it does not exist
#   run-qwen36.sh start    create + `docker start -a` (the ledger unit's command; scripts/ensure_teacher.sh)
#   run-qwen36.sh check    exit 0 when the image is the pinned one and the container exists with it
#
# Image: eugr/spark-vllm nightly-20260921 (spark-vllm-docker prebuilt runner), vLLM 0.29.1rc1.dev467+g0aee727ff,
# config digest = image ID 666307b2... (plans/GENERATION_DECISIONS.md). Pulled by manifest digest, so it is immutable.
# Flags: the last deployed profile (journal of natlang-qwen36-teacher, 2026-10-07 "non-default args"): the
# 2026-10-02 receipt runs/dgx-model-comparison-20261002/nvfp4-server-command.json with 65536 context and 256 sequences.
# Starting it costs ~74 GB at peak (CUDA graph capture), ~59 GB serving: start only under the ledger (68 GB).
set -eu
NAME=natlang-qwen36-nvfp4-server
TAG=vllm-node
PULL=eugr/spark-vllm@sha256:673b4e1479a0133d6b04476bc8fbd6a39171d6c7e0cb61972bfba52f5017fbaf
IMAGE_ID=sha256:666307b207f9f583f6048111e90c5d177355a2f94d9101494c8c85c5b277471d
MODEL=/mnt/external/bgkit-data/models/hf/Qwen3.6-35B-A3B-NVFP4
CACHE=/mnt/external/natlang-vllm-qwen36

image_ok() { [ "$(docker image inspect -f '{{.Id}}' $TAG 2>/dev/null)" = "$IMAGE_ID" ]; }
container_ok() { [ "$(docker container inspect -f '{{.Image}}' $NAME 2>/dev/null)" = "$IMAGE_ID" ]; }

ensure_image() {
  image_ok && return 0
  echo "teacher: pulling pinned image $PULL as $TAG" >&2
  docker pull "$PULL"
  docker tag "$PULL" $TAG
  image_ok || { echo "teacher: $TAG is not $IMAGE_ID after pull" >&2; exit 1; }
}

ensure_container() {
  ensure_image
  if docker container inspect $NAME >/dev/null 2>&1; then
    container_ok && return 0
    echo "teacher: container $NAME exists with another image; leaving it (remove it to recreate from this spec)" >&2
    exit 1
  fi
  [ -f "$MODEL/model.safetensors.index.json" ] || { echo "teacher: weights missing at $MODEL" >&2; exit 1; }
  mkdir -p "$CACHE"
  echo "teacher: creating container $NAME from scripts/teacher/run-qwen36.sh" >&2
  docker create --name $NAME --gpus all --shm-size 8g --ipc=host --restart no \
    -p 127.0.0.1:8082:8080 \
    -v "$MODEL":/model:ro \
    -v "$CACHE":/root/.cache/vllm \
    --entrypoint /usr/local/bin/vllm \
    "$IMAGE_ID" \
    serve /model \
    --served-model-name nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --host 0.0.0.0 --port 8080 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len 65536 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.45 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --moe-backend marlin \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --chat-template-content-format string \
    --safetensors-load-strategy prefetch >/dev/null
}

case "${1:-}" in
  image) ensure_image ;;
  create) ensure_container ;;
  start) ensure_container; exec docker start -a $NAME ;;
  check) image_ok && container_ok ;;
  *) echo "usage: $0 image|create|start|check" >&2; exit 2 ;;
esac
