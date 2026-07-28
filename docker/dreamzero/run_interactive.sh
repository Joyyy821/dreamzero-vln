#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-dreamzero-cuda129-py311}
IMAGE_TAG=${IMAGE_TAG:-latest}
CONTAINER_NAME=${CONTAINER_NAME:-${USER}-dreamzero-vln}

# Host paths
HOST_REPO=${HOST_REPO:-$HOME/workspace/dreamzero-vln}
HOST_DATA_ROOT=${HOST_DATA_ROOT:-/data/$USER/dreamzero}
HOST_UID=${HOST_UID:-$(id -u)}
HOST_GID=${HOST_GID:-$(id -g)}
HOST_GROUP=${HOST_GROUP:-$(id -gn)}

# Container paths
CONTAINER_REPO=/workspace/dreamzero-vln
CONTAINER_DATA=/data

mkdir -p "$HOST_DATA_ROOT/datasets"
mkdir -p "$HOST_DATA_ROOT/checkpoints"
mkdir -p "$HOST_DATA_ROOT/outputs"
mkdir -p "$HOST_DATA_ROOT/cache/home"
mkdir -p "$HOST_DATA_ROOT/cache/hf_home"
mkdir -p "$HOST_DATA_ROOT/cache/hf_hub"
mkdir -p "$HOST_DATA_ROOT/cache/hf_xet"
mkdir -p "$HOST_DATA_ROOT/cache/torch"
mkdir -p "$HOST_DATA_ROOT/cache/wandb"

docker run --rm -it \
  --name "$CONTAINER_NAME" \
  --label "owner=$USER" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HOST_UID=$HOST_UID \
  -e HOST_GID=$HOST_GID \
  -e HOST_USER=$USER \
  -e HOST_GROUP="$HOST_GROUP" \
  -e HOME=$CONTAINER_DATA/cache/home \
  -e HF_HOME=$CONTAINER_DATA/cache/hf_home \
  -e HF_HUB_CACHE=$CONTAINER_DATA/cache/hf_hub \
  -e HF_XET_CACHE=$CONTAINER_DATA/cache/hf_xet \
  -e TORCH_HOME=$CONTAINER_DATA/cache/torch \
  -e WANDB_DIR=$CONTAINER_DATA/cache/wandb \
  -e WANDB_MODE=offline \
  -e HYDRA_FULL_ERROR=1 \
  -e NO_ALBUMENTATIONS_UPDATE=1 \
  -v "$HOST_REPO":$CONTAINER_REPO \
  -v "$HOST_DATA_ROOT":$CONTAINER_DATA \
  -w $CONTAINER_REPO \
  ${IMAGE_NAME}:${IMAGE_TAG} \
  bash
