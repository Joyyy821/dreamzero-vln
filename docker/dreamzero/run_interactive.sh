#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-dreamzero-cuda129-py311}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Host paths
HOST_WORKSPACE=${HOST_WORKSPACE:-$HOME/workspace}
HOST_DATA_ROOT=${HOST_DATA_ROOT:-/data/$USER/dreamzero}

# Container paths
CONTAINER_WORKSPACE=/workspace
CONTAINER_DATA=/data

mkdir -p "$HOST_WORKSPACE"
mkdir -p "$HOST_DATA_ROOT/datasets"
mkdir -p "$HOST_DATA_ROOT/checkpoints"
mkdir -p "$HOST_DATA_ROOT/outputs"
mkdir -p "$HOST_DATA_ROOT/cache/huggingface"
mkdir -p "$HOST_DATA_ROOT/cache/torch"
mkdir -p "$HOST_DATA_ROOT/cache/wandb"

docker run --rm -it \
  --name yjiao-dreamzero-vln \
  --label owner=yangjiao \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=$CONTAINER_DATA/cache/huggingface \
  -e TORCH_HOME=$CONTAINER_DATA/cache/torch \
  -e WANDB_DIR=$CONTAINER_DATA/cache/wandb \
  -e WANDB_MODE=offline \
  -e HYDRA_FULL_ERROR=1 \
  -e DATA_ROOT=$CONTAINER_DATA/datasets \
  -e CKPT_ROOT=$CONTAINER_DATA/checkpoints \
  -e OUTPUT_ROOT=$CONTAINER_DATA/outputs \
  -v "$HOST_WORKSPACE":$CONTAINER_WORKSPACE \
  -v "$HOST_DATA_ROOT":$CONTAINER_DATA \
  -w $CONTAINER_WORKSPACE/dreamzero \
  ${IMAGE_NAME}:${IMAGE_TAG} \
  bash