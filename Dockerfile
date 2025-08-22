# syntax=docker/dockerfile:1.7
ARG CUDA_VER=12.1.1
ARG UBUNTU_VER=22.04
FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu${UBUNTU_VER}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates tzdata curl git tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root
RUN useradd -m -u 1000 appuser
WORKDIR /workspace/app

# Venv + deps
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip wheel
# you committed requirements-lock.txt; use it as requirements
COPY requirements-lock.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copy source (relies on .dockerignore to exclude logs/venvs/data/.git)
COPY . .

# Entry + defaults (skip missing app.env.example)
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint && \
    mkdir -p /workspace/config /workspace/data && \
    chown -R appuser:appuser /workspace

# (No healthcheck since this isn’t a web app)
USER appuser
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint"]
CMD ["python","main.py"]
