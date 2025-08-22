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
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates tzdata curl git tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# venv
RUN python3 -m venv /opt/venv && pip install --upgrade pip wheel

# deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary numpy==2.2.6 && \
    pip install --only-binary=:all: -r requirements.txt

# keep runtime simple so you can run scripts yourself
CMD ["bash"]
