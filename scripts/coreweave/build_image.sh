#!/bin/bash
# Build and push the x86_64 (linux/amd64) natlang training image for CoreWeave sandboxes.
#
# CoreWeave's serverless GPU sandboxes run the RTX PRO 6000 Blackwell Server Edition (sm_120) on Intel hosts, so the
# DGX's own natlang-neuralese:26.09 (arm64) cannot run there. This builds the same Dockerfile
# (training/neuralese/docker, one dependency list in requirements.txt) for linux/amd64. NGC 26.09's PyTorch already
# lists sm_120 (measured on DGX 2026-10-12: sm_80 sm_86 sm_90 sm_100 sm_110 sm_120 compute_120).
#
# Not needed for the first jobs: the bootstrap path runs the public nvcr.io/nvidia/pytorch:26.09-py3 image and installs
# requirements.txt at start (about a minute). Build this once image pulls or pip become the start-up cost, then set
# "image" in the job to the pushed reference and NATLANG_CW_IMAGE_READY=1 in its env.
#
# Usage: NATLANG_CW_IMAGE_REPO=ghcr.io/<owner>/natlang-neuralese scripts/coreweave/build_image.sh [tag]
#   Needs docker buildx with an amd64 builder (QEMU binfmt on DGX: `docker run --privileged --rm
#   tonistiigi/binfmt --install amd64`, or build on an x86_64 host such as Pop). The registry login is the caller's
#   (`docker login ghcr.io`); a private registry also needs sandbox pull credentials (ImagePullCredentials from a
#   secret store), so a public repository is the simple path. On DGX run it under the memory ledger, priority low.
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${NATLANG_CW_IMAGE_REPO:?set NATLANG_CW_IMAGE_REPO, e.g. ghcr.io/werg/natlang-neuralese}
TAG=${1:-26.09-$(git -C "$REPO_DIR" rev-parse --short HEAD)-amd64}
docker buildx build --platform linux/amd64 \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3 \
  -t "$IMAGE:$TAG" --push "$REPO_DIR/training/neuralese/docker"
digest=$(docker buildx imagetools inspect "$IMAGE:$TAG" --format '{{json .Manifest.Digest}}' | tr -d '"')
echo "{\"image\": \"$IMAGE:$TAG\", \"digest\": \"$digest\", \"platform\": \"linux/amd64\"}"
