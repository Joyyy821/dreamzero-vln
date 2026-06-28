#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-dreamzero-cuda129-py311}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Run from repo root.
docker build \
  -f docker/dreamzero/Dockerfile \
  -t ${IMAGE_NAME}:${IMAGE_TAG} \
  .