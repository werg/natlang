#!/bin/bash
# Build an OCI archive locally; no registry push or sandbox allocation.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
: "${NATLANG_WARM_BASE_IMAGE:?set immutable base image reference with @sha256:}"
: "${NATLANG_WARM_WHEELHOUSE:?set path to the frozen wheelhouse tar.gz}"
: "${NATLANG_WARM_WHEELHOUSE_SHA256:?set wheelhouse archive SHA256}"
: "${NATLANG_WARM_IMAGE_OUTPUT:?set output OCI archive path}"
[[ "$NATLANG_WARM_BASE_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'base must be digest-pinned' >&2; exit 1; }
printf '%s  %s\n' "$NATLANG_WARM_WHEELHOUSE_SHA256" "$NATLANG_WARM_WHEELHOUSE" | sha256sum -c -
context=$(mktemp -d); trap 'rm -rf "$context"' EXIT
cp "$repo/training/neuralese/docker/Dockerfile.warm" "$context/Dockerfile"
cp "$NATLANG_WARM_WHEELHOUSE" "$context/wheelhouse.tar.gz"
docker buildx build --platform linux/amd64 --build-arg "BASE_IMAGE=$NATLANG_WARM_BASE_IMAGE" \
  --build-arg "WHEELHOUSE_SHA256=$NATLANG_WARM_WHEELHOUSE_SHA256" \
  --output "type=oci,dest=$NATLANG_WARM_IMAGE_OUTPUT" "$context"
sha256sum "$NATLANG_WARM_IMAGE_OUTPUT"
