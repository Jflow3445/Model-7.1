# syntax=docker/dockerfile:1.7
ARG CUDA_VER=12.1.1
ARG UBUNTU_VER=22.04
FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu${UBUNTU_VER}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PIP_DEFAULT_TIMEOUT=120

# OS deps (+ build tools for wheels that need compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates tzdata curl git tini \
      build-essential gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user and working dir
RUN useradd -m -u 1000 appuser
WORKDIR /workspace

# Venv + latest pip/wheel
RUN python3 -m venv /opt/venv && pip install --upgrade pip wheel

# === requirements ===
# If you have "requirements-lock.txt", either rename it to requirements.txt in git,
# or copy it as requirements.txt here. Pick ONE approach. This example expects requirements.txt.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Entrypoints & app files
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint \
    && mkdir -p /workspace/config /workspace/data \
    && chown -R appuser:appuser /workspace

# Copy your source tree last (better cache usage for pip step)
COPY . .

USER appuser
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint"]
CMD ["bash"]   # <— default to a shell so you can run any script
