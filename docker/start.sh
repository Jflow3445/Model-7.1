#!/usr/bin/env bash
set -euo pipefail

# Where to persist (RunPod volume mounts here)
VOL="${RUNPOD_VOL:-/runpod-volume}"
APP="/workspace"
LOGS="${VOL}/logs"
CKPTS="${VOL}/checkpoints"
mkdir -p "$LOGS" "$CKPTS"

echo "[boot] Python: $(python -V)"
echo "[boot] CUDA visible: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Torch perf knobs (safe on Ampere/ADA/Hopper)
python - <<'PY'
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
print("[boot] Enabled cuDNN benchmark and TF32 matmul precision")
PY

# Select task (override via env TASK=main|retrain_historical|retrain_ticks|weekly)
TASK="${TASK:-weekly}"

# Common args (you can tack on EXTRA_ARGS from RunPod UI)
COMMON_ARGS=( --log-dir "$LOGS" --save-dir "$CKPTS" )
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=( ${EXTRA_ARGS} )
fi

# Auto-resume policy
if [[ -f "${CKPTS}/ppo_retrained_finetuned.zip" ]]; then
  RESUME="--resume ${CKPTS}/ppo_retrained_finetuned.zip"
elif [[ -f "${CKPTS}/ppo_retrained.zip" ]]; then
  RESUME="--resume ${CKPTS}/ppo_retrained.zip"
else
  RESUME=""
fi

case "$TASK" in
  main)
    echo "[run] main.py ${COMMON_ARGS[*]} $RESUME"
    exec python -m main ${COMMON_ARGS[@]} $RESUME
    ;;
  retrain_historical)
    echo "[run] retrain_historical.py (Optuna mandatory) ${COMMON_ARGS[*]}"
    exec python -m retrain_historical ${COMMON_ARGS[@]}
    ;;
  retrain_ticks)
    echo "[run] retrain_from_ticks.py ${COMMON_ARGS[*]} $RESUME"
    exec python -m retrain_from_ticks ${COMMON_ARGS[@]} $RESUME
    ;;
  weekly|*)
    echo "[run] weekly pipeline via run_weekly_retraining.py ${COMMON_ARGS[*]}"
    exec python -m run_weekly_retraining ${COMMON_ARGS[@]}
    ;;
esac
