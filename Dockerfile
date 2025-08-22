# syntax=docker/dockerfile:1.7
ARG CUDA_VER=12.1.1
ARG UBUNTU_VER=22.04
FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu${UBUNTU_VER}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/workspace"

# Minimal OS deps + tini
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates tzdata curl git tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 appuser

# Work at repo root inside the image
WORKDIR /workspace

# Isolated venv
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip wheel
# before: pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel && \
    pip install --prefer-binary numpy==2.2.6 && \
    pip install --prefer-binary -r requirements.txt

# Install deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Bring in the source tree (root → /workspace)
COPY . .

# Entrypoint and optional defaults (only keep this COPY if the file exists in the repo)
# COPY docker/entrypoint.sh /usr/local/bin/entrypoint
# COPY docker/config/app.env.example /opt/defaults/app.env
# If you don't ship a default env, just do:
COPY docker/entrypoint.sh /usr/local/bin/entrypoint

RUN chmod +x /usr/local/bin/entrypoint && \
    mkdir -p /workspace/config /workspace/data && \
    chown -R appuser:appuser /workspace

# Optional healthcheck — disable if you don't have a /health endpoint
# HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
#   CMD curl -fsS http://localhost:8000/health || exit 1

USER appuser
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint"]
# Default to an interactive shell so you can run any script
CMD ["bash"]
