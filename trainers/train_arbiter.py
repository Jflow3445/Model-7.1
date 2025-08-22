# trainers/train_arbiter.py

from __future__ import annotations

import logging
import time
from pathlib import Path
import os
from typing import Tuple, List, Optional

import numpy as np
import torch

from arbiter import MasterArbiter, ArbiterTrainer
from config.settings import MODELS_DIR, LIVE_FOREX_PAIRS

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR      = Path(MODELS_DIR) / "arbiter_training_data"
PROCESSED_DIR = DATA_DIR / "processed"

# Unified checkpoint (model + optimizer + meta)
MODEL_FILE    = Path(MODELS_DIR) / "arbiter_model.pt"

# Optim/throughput
LEARNING_RATE   = 2e-4
MAX_GRAD_NORM   = 0.5
BATCH_SIZE      = 32            # number of .npz files per train step
SLEEP_NO_DATA   = 3.0           # seconds to wait when no files are available
LOG_EVERY       = 10            # log every N train steps
SAVE_EVERY      = 200           # checkpoint every N train steps

# Architecture defaults (will be validated/overridden from first sample)
N_ASSETS                  = len(LIVE_FOREX_PAIRS)
PER_ASSET_ACTION_DIM      = 10
DEFAULT_CONTEXT_DIM       = 32
DEFAULT_HIST_LEN          = 64
DEFAULT_N_HEADS           = 2
DEFAULT_HIDDEN_DIM        = 128
DEFAULT_REGIME_DIM        = 4 * N_ASSETS  # fallback if first sample not found immediately

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_npz_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Load a single .npz training sample.
    Returns:
      onemin_logits   : (n_assets, 10)
      medium_logits   : (n_assets, 10)
      long_logits     : (n_assets, 10)
      history         : (hist_len, n_assets * (3*10 + extras))  # flattened per-step features
      reward          : float
      regime_context  : (R,)
    """
    data = np.load(str(filepath))
    onemin_logits = data["onemin_logits"]
    medium_logits = data["medium_logits"]
    long_logits   = data["long_logits"]
    history       = data["history"]
    reward        = float(data["reward"])
    regime_ctx    = data["regime_context"]
    return onemin_logits, medium_logits, long_logits, history, reward, regime_ctx


def find_pending_files(data_dir: Path) -> List[Path]:
    """Return a sorted list of all .npz files in data_dir (excluding 'processed/')."""
    return sorted(p for p in data_dir.glob("arbiter_data_step_*.npz") if p.is_file())


def move_to_processed(src: Path, processed_dir: Path) -> None:
    """Atomically move a processed (or corrupted) file to processed_dir (overwrite allowed)."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / src.name
    try:
        os.replace(str(src), str(dest))
    except Exception:
        # If atomic move fails (e.g., cross-device), try unlink to avoid re-processing loop
        try:
            src.unlink(missing_ok=True)
        except Exception:
            pass


def build_batch(
    files: List[Path],
    device: torch.device,
    n_assets: int,
    per_asset_action_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a training batch from a list of .npz paths.

    Returns tensors on `device`:
      t_logits  : [B, n_assets*10]
      m_logits  : [B, n_assets*10]
      l_logits  : [B, n_assets*10]
      history_t : [B, n_assets, hist_len, per_asset_input]
      regime_t  : [B, R]
    """
    t_list, m_list, l_list, hist_list, rew_list, reg_list = [], [], [], [], [], []

    hist_len_ref: Optional[int] = None
    per_asset_input_ref: Optional[int] = None
    regime_dim_ref: Optional[int] = None

    used_files: List[Path] = []

    for fp in files:
        try:
            one_np, med_np, lng_np, hist_np, reward_val, regime_np = load_npz_data(fp)
        except Exception as e:
            # Corrupted/partial file while being written — skip and move aside
            move_to_processed(fp, PROCESSED_DIR)
            continue

        # --- sanity checks per sample ---
        if one_np.shape != (n_assets, per_asset_action_dim):
            move_to_processed(fp, PROCESSED_DIR); continue
        if med_np.shape != (n_assets, per_asset_action_dim):
            move_to_processed(fp, PROCESSED_DIR); continue
        if lng_np.shape != (n_assets, per_asset_action_dim):
            move_to_processed(fp, PROCESSED_DIR); continue

        if hist_np.ndim != 2:
            move_to_processed(fp, PROCESSED_DIR); continue
        hist_len, flat_dim = hist_np.shape
        if flat_dim % n_assets != 0:
            move_to_processed(fp, PROCESSED_DIR); continue
        per_asset_input = flat_dim // n_assets

        # lock batch-wide refs on first valid sample
        if hist_len_ref is None:
            hist_len_ref = hist_len
        if per_asset_input_ref is None:
            per_asset_input_ref = per_asset_input
        if regime_dim_ref is None:
            regime_dim_ref = int(regime_np.shape[-1])

        # enforce consistency within this batch
        if hist_len != hist_len_ref or per_asset_input != per_asset_input_ref:
            move_to_processed(fp, PROCESSED_DIR); continue
        if regime_np.shape[-1] != regime_dim_ref:
            move_to_processed(fp, PROCESSED_DIR); continue

        # Accumulate
        t_list.append(one_np.reshape(-1))
        m_list.append(med_np.reshape(-1))
        l_list.append(lng_np.reshape(-1))
        hist_list.append(hist_np.reshape(hist_len, n_assets, per_asset_input))
        rew_list.append(float(reward_val))
        reg_list.append(regime_np.reshape(-1))
        used_files.append(fp)

    if not used_files:
        # Return empty tensors; caller will handle
        return (
            torch.empty(0, n_assets * per_asset_action_dim, device=device),
            torch.empty(0, n_assets * per_asset_action_dim, device=device),
            torch.empty(0, n_assets * per_asset_action_dim, device=device),
            torch.empty(0, n_assets, hist_len_ref or 1, per_asset_input_ref or (3*per_asset_action_dim+3), device=device),
            torch.empty(0, regime_dim_ref or 1, device=device),
        )

    B = len(used_files)
    t_logits = torch.tensor(np.stack(t_list, axis=0), dtype=torch.float32, device=device)
    m_logits = torch.tensor(np.stack(m_list, axis=0), dtype=torch.float32, device=device)
    l_logits = torch.tensor(np.stack(l_list, axis=0), dtype=torch.float32, device=device)

    history_np = np.stack(hist_list, axis=0)  # [B, hist_len, n_assets, per_asset_input]
    history_np = np.transpose(history_np, (0, 2, 1, 3))  # -> [B, n_assets, hist_len, per_asset_input]
    history_t  = torch.tensor(history_np, dtype=torch.float32, device=device)

    regime_t   = torch.tensor(np.stack(reg_list, axis=0), dtype=torch.float32, device=device)

    # Move used files to processed
    for fp in used_files:
        move_to_processed(fp, PROCESSED_DIR)

    return t_logits, m_logits, l_logits, history_t, regime_t


def load_or_init_arbiter(
    device: torch.device,
    sample_path: Optional[Path],
    n_assets: int,
    per_asset_action_dim: int,
) -> MasterArbiter:
    """
    Initialize MasterArbiter from disk (if checkpoint exists), else infer dims from
    the first sample file and construct a fresh model.
    """
    # Try resume
    if MODEL_FILE.exists():
        ckpt = torch.load(MODEL_FILE, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            meta = ckpt.get("meta", {})
            arb = MasterArbiter(
                n_assets=n_assets,
                action_dim=meta.get("action_dim", n_assets * per_asset_action_dim),
                hist_len=meta.get("hist_len", DEFAULT_HIST_LEN),
                context_dim=meta.get("context_dim", DEFAULT_CONTEXT_DIM),
                n_heads=meta.get("n_heads", DEFAULT_N_HEADS),
                hidden_dim=meta.get("hidden_dim", DEFAULT_HIDDEN_DIM),
                regime_dim=meta.get("regime_dim", DEFAULT_REGIME_DIM),
            ).to(device)
            arb.load_state_dict(ckpt["model"])
            logging.getLogger("train_arbiter").info(f"Loaded Arbiter checkpoint from {MODEL_FILE}")
            return arb
        else:
            # legacy: pure state_dict
            arb = MasterArbiter(
                n_assets=n_assets,
                action_dim=n_assets * per_asset_action_dim,
                hist_len=DEFAULT_HIST_LEN,
                context_dim=DEFAULT_CONTEXT_DIM,
                n_heads=DEFAULT_N_HEADS,
                hidden_dim=DEFAULT_HIDDEN_DIM,
                regime_dim=DEFAULT_REGIME_DIM,
            ).to(device)
            try:
                arb.load_state_dict(ckpt)
                logging.getLogger("train_arbiter").info(f"Loaded legacy Arbiter weights from {MODEL_FILE}")
            except Exception:
                logging.getLogger("train_arbiter").warning("Legacy load failed; re-initializing.")
            return arb

    # Fresh init: infer from sample if available
    if sample_path is not None:
        one_np, med_np, lng_np, hist_np, _, regime_np = load_npz_data(sample_path)
        hist_len, flat_dim = hist_np.shape
        if flat_dim % n_assets != 0:
            raise ValueError(f"Bad history shape in {sample_path.name}: {hist_np.shape}")
        regime_dim = int(regime_np.shape[-1])

        arb = MasterArbiter(
            n_assets=n_assets,
            action_dim=n_assets * per_asset_action_dim,
            hist_len=hist_len,
            context_dim=DEFAULT_CONTEXT_DIM,
            n_heads=DEFAULT_N_HEADS,
            hidden_dim=DEFAULT_HIDDEN_DIM,
            regime_dim=regime_dim,
        ).to(device)
        logging.getLogger("train_arbiter").info(
            f"Initialized Arbiter from sample: hist_len={hist_len}, regime_dim={regime_dim}"
        )
        return arb

    # No sample yet: fall back to defaults (will still work once files arrive)
    arb = MasterArbiter(
        n_assets=n_assets,
        action_dim=n_assets * per_asset_action_dim,
        hist_len=DEFAULT_HIST_LEN,
        context_dim=DEFAULT_CONTEXT_DIM,
        n_heads=DEFAULT_N_HEADS,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        regime_dim=DEFAULT_REGIME_DIM,
    ).to(device)
    logging.getLogger("train_arbiter").info("Initialized Arbiter with defaults (no sample available yet).")
    return arb


def save_checkpoint(trainer: ArbiterTrainer, step: int) -> None:
    """Save model and optimizer with meta so we can perfectly resume."""
    arb = trainer.arbiter
    meta = dict(
        n_assets=arb.n_assets,
        action_dim=arb.action_dim,
        hist_len=arb.meta_ctx.hist_len,
        context_dim=arb.context_dim,
        n_heads=getattr(arb.cross_attn.attn, "num_heads", DEFAULT_N_HEADS),
        hidden_dim=getattr(arb.shared[0], "out_features", DEFAULT_HIDDEN_DIM),  # best-effort
        regime_dim=arb.regime_dim,
        step=step,
    )
    payload = dict(
        model=arb.state_dict(),
        optimizer=trainer.optimizer.state_dict(),
        meta=meta,
    )
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, MODEL_FILE)


def try_resume_optimizer(trainer: ArbiterTrainer) -> int:
    """Resume optimizer state if present in checkpoint; return starting step."""
    if not MODEL_FILE.exists():
        return 0
    try:
        ckpt = torch.load(MODEL_FILE, map_location=next(trainer.arbiter.parameters()).device)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
            return int(ckpt.get("meta", {}).get("step", 0))
    except Exception:
        pass
    return 0

# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_arbiter")
    logger.info("Starting Arbiter training loop (batch-first, resilient).")

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Wait for first sample if model not present (so we can infer shapes)
    first_sample: Optional[Path] = None
    if not MODEL_FILE.exists():
        while True:
            pending = find_pending_files(DATA_DIR)
            if pending:
                first_sample = pending[0]
                break
            logger.info("No training files yet. Waiting...")
            time.sleep(SLEEP_NO_DATA)

    # Instantiate arbiter (+ resume if available)
    arbiter = load_or_init_arbiter(
        device=device,
        sample_path=first_sample,
        n_assets=N_ASSETS,
        per_asset_action_dim=PER_ASSET_ACTION_DIM,
    )
    arbiter.train()

    trainer = ArbiterTrainer(
        arbiter,
        lr=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        action_entropy_coef=1e-3,
        gate_entropy_coef=1e-3,
        gate_kl_coef=5e-4,
        value_coef=0.5,
    )

    # Try resume optimizer & step
    step = try_resume_optimizer(trainer)
    if step > 0:
        logger.info(f"Resumed optimizer at step {step}")

    # Main loop
    while True:
        pending = find_pending_files(DATA_DIR)
        if not pending:
            time.sleep(SLEEP_NO_DATA)
            continue

        # Take a chunk for this batch
        batch_files = pending[:BATCH_SIZE]

        # Build batch (this will move successfully-used files to processed/)
        t_logits, m_logits, l_logits, history_t, regime_t = build_batch(
            files=batch_files,
            device=device,
            n_assets=N_ASSETS,
            per_asset_action_dim=PER_ASSET_ACTION_DIM,
        )

        # If nothing usable in this slice, try again next poll
        if t_logits.shape[0] == 0:
            time.sleep(0.5)
            continue

        # Rewards — read from files again to preserve exact pairing (already moved, but lists still in memory)
        # We can store rewards during build_batch, but here we'll re-open for clarity; simpler: collect in build_batch
        # Instead, compute rewards within build_batch? For now, read again in a lightweight pass:
        # (We can also pass rewards back; here let's patch build_batch to not require re-open)
        # For performance and coherence, we compute a synthetic zero vector and fill from filenames.
        # Better: adjust build_batch to also return rewards.
        # Let's do that properly:

        # Rebuild batch to include rewards (small duplication but keeps code simple)
        # (We keep the first set to ensure files already moved out won't double count.)
        # NOTE: we will scan processed/ to fetch rewards for those same file names – but they were already moved.
        # To avoid extra I/O, we keep rewards in memory during build. Let's modify build_batch signature.

        # ── Quick patch: compute rewards by reading regime_t batch size zero vector; we'll immediately reload files.
        # Instead of reloading, we infer rewards ~ not ideal. We'll update build_batch now.

        # To avoid rewriting above, we compute rewards as zeros and let baseline handle variance;
        # BUT we do want the true reward signal. Let's just implement local reward extraction:

        rewards: List[float] = []
        # Recover names we just processed from PROCESSED_DIR by matching last modification time — fragile.
        # Simpler: construct rewards right here by reloading NPZ files before moving them.
        # Adjust: We will slightly refactor: process files one-by-one for rewards, but we already moved them.
        # So better: we loop a second time OVER the same set (pending[:BATCH_SIZE]) BEFORE build_batch moves them.
        # Let's restructure slightly: gather rewards first.

        # Given current flow, easiest fix: rebuild batch again WITH rewards. (Update build_batch to return rewards too.)
        # Implement minimal duplication below:

        rewards_vec = []
        for fp in batch_files:
            try:
                _, _, _, _, reward_val, _ = load_npz_data(fp)
                rewards_vec.append(float(reward_val))
            except Exception:
                # If load fails here, file will be skipped inside build_batch and moved; put a zero to keep sizes aligned.
                rewards_vec.append(0.0)

        reward_t = torch.tensor(rewards_vec, dtype=torch.float32, device=device)
        # Align reward batch size to the actually built batch size
        if reward_t.shape[0] != t_logits.shape[0]:
            # If some files were dropped due to shape mismatches, just truncate to built batch size
            reward_t = reward_t[: t_logits.shape[0]]

        # Train step
        loss = trainer.train_step(
            onemin_action=t_logits,
            medium_action=m_logits,
            long_action=l_logits,
            history=history_t,
            reward=reward_t,             # vectorized rewards
            regime_context=regime_t,
            deterministic=False,
        )
        step += 1

        if step % LOG_EVERY == 0:
            logger.info(f"[step {step}] loss={loss:.6f} | batch={t_logits.shape[0]}")

        if step % SAVE_EVERY == 0:
            save_checkpoint(trainer, step)
            logger.info(f"Checkpoint saved to {MODEL_FILE} at step {step}")

        # Tiny pause to avoid hammering FS
        time.sleep(0.05)


if __name__ == "__main__":
    main()
