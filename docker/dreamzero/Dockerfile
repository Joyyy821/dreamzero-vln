# docker/dreamzero/Dockerfile
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/data/cache/hf_home
ENV HF_HUB_CACHE=/data/cache/hf_hub
ENV HF_XET_CACHE=/data/cache/hf_xet
ENV TORCH_HOME=/data/cache/torch
ENV MAX_JOBS=8
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl ca-certificates \
    gosu \
    tmux \
    build-essential ninja-build cmake pkg-config \
    ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --upgrade pip setuptools wheel packaging

WORKDIR /workspace/dreamzero-vln

COPY . .

# Install the repository and its pinned dependencies. The editable source path is
# also the path used by the runtime bind mount in run_interactive.sh.
RUN python -m pip install -e . \
    --extra-index-url https://download.pytorch.org/whl/cu129 && \
    python -m pip install h5py

# FlashAttention is often the most fragile dependency.
# Build it after torch is installed.
RUN python -m pip install --no-build-isolation flash-attn

COPY docker/dreamzero/entrypoint.sh /usr/local/bin/dreamzero-entrypoint
RUN chmod +x /usr/local/bin/dreamzero-entrypoint

ENTRYPOINT ["dreamzero-entrypoint"]
CMD ["/bin/bash"]
