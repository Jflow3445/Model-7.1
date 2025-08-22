# syntax=docker/dockerfile:1.7
ARG CUDA_VER=12.1.1
ARG UBUNTU_VER=22.04
FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu${UBUNTU_VER}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates tzdata curl git tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /workspace/app

# venv + modern build tools
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel

# --- Install PyTorch (CUDA 12.1) first ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1+cu121

# --- Then the rest of your deps ---
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# App files and boot scripts (unchanged)
COPY app/ ./app/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint && \
    mkdir -p /workspace/config /workspace/data && \
    chown -R appuser:appuser /workspace

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl -fsS http://localhost:8000/health || exit 1

USER appuser
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint"]
CMD ["python","-m","app"]
