# docker/dreamzero/Dockerfile
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV MAX_JOBS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl ca-certificates \
    tmux \
    build-essential ninja-build cmake pkg-config \
    ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel packaging

# PyTorch CUDA 12.9 wheels.
RUN pip install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu129

# Core training / data / video dependencies that are useful for DreamZero + Isaac data conversion.
COPY docker/dreamzero/requirements-extra.txt /tmp/requirements-extra.txt
RUN pip install -r /tmp/requirements-extra.txt

# FlashAttention is often the most fragile dependency.
# Build it after torch is installed.
RUN pip install --no-build-isolation flash-attn

WORKDIR /workspace

CMD ["/bin/bash"]