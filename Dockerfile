# syntax=docker/dockerfile:1.7
ARG CUDA_VER=12.1.1
ARG UBUNTU_VER=22.04
FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu${UBUNTU_VER}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
    PIP_PREFER_BINARY=1

# OS deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates tzdata curl git tini software-properties-common gnupg \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-distutils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Create venv on Python 3.11 and upgrade build tooling
RUN /usr/bin/python3.11 -m venv /opt/venv \
 && pip install --upgrade pip wheel setuptools

# Install Python deps (wheels only)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: -r requirements.txt

# Default to interactive shell so you can run any script
CMD ["bash"]
