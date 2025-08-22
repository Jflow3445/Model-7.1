#!/usr/bin/env bash
set -Eeuo pipefail

: "${CONFIG_DIR:=/workspace/config}"
: "${STARTUP_CMD:=}"

# First-run: materialize defaults onto the (persistent) volume
if [ ! -f "$CONFIG_DIR/app.env" ] && [ -f "/opt/defaults/app.env" ]; then
  mkdir -p "$CONFIG_DIR"
  cp /opt/defaults/app.env "$CONFIG_DIR/app.env"
fi

# Load env file if present (never bake secrets into image)
if [ -f "$CONFIG_DIR/app.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$CONFIG_DIR/app.env"
  set +a
fi

# Allow RunPod to override the command without rebuilding the image
if [ -n "$STARTUP_CMD" ]; then
  exec bash -lc "$STARTUP_CMD"
fi

exec "$@"
