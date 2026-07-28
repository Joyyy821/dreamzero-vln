#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" != "0" ]]; then
  exec "$@"
fi

HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-$HOST_UID}
HOST_USER=${HOST_USER:-dreamzero}
HOST_GROUP=${HOST_GROUP:-$HOST_USER}
HOME_DIR=${HOME:-/data/cache/home}

sanitize_name() {
  local name
  name=$(printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^[^A-Za-z_]/u&/')
  printf '%.32s' "${name:-dreamzero}"
}

CONTAINER_GROUP=$(getent group "$HOST_GID" | cut -d: -f1 || true)
if [[ -z "$CONTAINER_GROUP" ]]; then
  CONTAINER_GROUP=$(sanitize_name "$HOST_GROUP")
  if getent group "$CONTAINER_GROUP" >/dev/null; then
    CONTAINER_GROUP="dreamzero_g${HOST_GID}"
  fi
  groupadd --gid "$HOST_GID" "$CONTAINER_GROUP"
fi

CONTAINER_USER=$(getent passwd "$HOST_UID" | cut -d: -f1 || true)
if [[ -z "$CONTAINER_USER" ]]; then
  CONTAINER_USER=$(sanitize_name "$HOST_USER")
  if getent passwd "$CONTAINER_USER" >/dev/null; then
    CONTAINER_USER="dreamzero_u${HOST_UID}"
  fi
  useradd \
    --uid "$HOST_UID" \
    --gid "$HOST_GID" \
    --home-dir "$HOME_DIR" \
    --shell /bin/bash \
    --no-create-home \
    "$CONTAINER_USER"
fi

CACHE_DIRS=(
  "$HOME_DIR"
  "${HF_HOME:-/data/cache/hf_home}"
  "${HF_HUB_CACHE:-/data/cache/hf_hub}"
  "${HF_XET_CACHE:-/data/cache/hf_xet}"
  "${TORCH_HOME:-/data/cache/torch}"
  "${WANDB_DIR:-/data/cache/wandb}"
)

for cache_dir in "${CACHE_DIRS[@]}"; do
  mkdir -p "$cache_dir"
  chown "$HOST_UID:$HOST_GID" "$cache_dir"
done

export HOME="$HOME_DIR"
export USER="$CONTAINER_USER"
export LOGNAME="$CONTAINER_USER"

exec gosu "$HOST_UID:$HOST_GID" "$@"
