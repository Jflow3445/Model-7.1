# train_onemin_policy.py
from __future__ import annotations
import os
import glob
import random
import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import subprocess

# Policies / extractor (the recurrent policy may or may not exist in your repo)
from models.onemin_policy import (
    OneMinOHLCPolicy,
)

from config.settings import (
    ONEMIN_CSV_DIR,
    LIVE_FOREX_PAIRS,
    MODELS_DIR,
    SEED,
    SLIPPAGE_PER_UNIT,
    COMMISSION_PER_TRADE,
    INITIAL_BALANCE,
    ONEMIN_OBS_WINDOW,
    BASE_DIR,
)
from utils.reward_utils import RewardFunction

# Broker stops (floor)
BROKER_STOPS_JSON = Path(BASE_DIR) / "config" / "broker_stops.json"

# ──────────────────────────────────────────────────────────────────────────────
# Constants / Logging
# ──────────────────────────────────────────────────────────────────────────────
LOGS_DIR = os.path.join(MODELS_DIR, "logs")
SEVERE_ILLEGAL_ACTION_PENALTY = -2
ILLEGAL_ATTEMPT_PENALTY = -0.01
os.makedirs(LOGS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _to_float(x):
    try:
        return float(x.item())
    except AttributeError:
        return float(x)

def load_broker_meta(json_path: Path) -> Dict[str, Dict]:
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    logging.warning(f"[train_onemin] broker_stops.json not found at {json_path}, using zeros.")
    return {}

def compute_atr(df: pd.DataFrame, n: int = 14, mode: str = "rolling") -> pd.Series:
    """
    Robust ATR (Onemin):
    - mode="rolling": SMA ATR with min_periods=n (strict; may yield initial NaNs)
    - mode="wilder" : Wilder ATR via EMA with min_periods=1 (fallback; never empty)
    """
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)

    tr = pd.concat(
        [(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1
    ).max(axis=1)

    if mode == "wilder":
        atr = tr.ewm(alpha=1.0 / float(n), adjust=False, min_periods=1).mean()
        atr = atr.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    else:
        atr = tr.rolling(int(n), min_periods=int(n)).mean()
        atr = atr.replace([np.inf, -np.inf], np.nan)
    return atr

def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.sort_values("time").drop_duplicates(subset="time", keep="last").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

def _scale_sl_tp(entry: float, sl_norm: float, tp_norm: float,
                 is_long: bool, min_stop_price: float, atr_value: float) -> Tuple[float, float]:
    """
    Map normalized [-1,1] to price distances using ATR + broker floor.
    """
    FLOOR_FRAC_ATR = 0.40
    K_SL_ATR = 0.80
    K_TP_ATR = 1.60

    atr_value = float(atr_value) if np.isfinite(atr_value) else 0.0
    floor_price = max(float(min_stop_price or 0.0), FLOOR_FRAC_ATR * atr_value)

    sl_dist = max(floor_price, (0.5 + abs(sl_norm)) * K_SL_ATR * max(atr_value, 1e-12))
    tp_dist = max(floor_price, (0.5 + abs(tp_norm)) * K_TP_ATR * max(atr_value, 1e-12))

    if is_long:
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
    return sl, tp

# ──────────────────────────────────────────────────────────────────────────────
# Callbacks (mirrors onemin style)
# ──────────────────────────────────────────────────────────────────────────────
class CheckpointAndRcloneCallback(BaseCallback):
    """
    Save checkpoints at exact PPO timesteps (model.num_timesteps)
    and then push each .zip to the required rclone destination.
    """
    def __init__(self, checkpoint_freq: int, ckpt_dir: str,
                 name_prefix: str = "onemin_policy_ckpt",
                 rclone_dest: str = "", verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_freq = int(checkpoint_freq)
        self.ckpt_dir = ckpt_dir
        self.name_prefix = name_prefix
        self.rclone_dest = rclone_dest or os.getenv("RCLONE_DEST", "")
        if not self.rclone_dest:
            print("[Checkpoint] RCLONE_DEST not set; saving locally only.")

    def _on_step(self) -> bool:
        t = int(self.model.num_timesteps)
        if t % self.checkpoint_freq != 0:
            return True

        os.makedirs(self.ckpt_dir, exist_ok=True)
        base = f"{self.name_prefix}_{t}_steps"
        fpath = os.path.join(self.ckpt_dir, base)
        self.model.save(fpath)
        # also checkpoint VecNormalize stats
        try:
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(os.path.join(self.ckpt_dir, "vecnormalize.pkl"))
        except Exception as e:
            print(f"[VecNormalize] Save (ckpt) failed: {e}")

        fzip = fpath + ".zip"
        exists = os.path.exists(fzip)
        print(f"[Checkpoint] wrote {fzip}  exists={exists}")

        if not exists:
            raise RuntimeError(f"Checkpoint file missing after save: {fzip}")

        # Only push if a remote is configured
        if self.rclone_dest:
            cmd = [
                "rclone", "copy", fzip, self.rclone_dest,
                "--drive-chunk-size", "64M", "--transfers", "2", "--checkers", "4", "-q"
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"[Checkpoint] pushed {os.path.basename(fzip)} -> {self.rclone_dest}")

                # push vecnormalize.pkl too
                pkl_path = os.path.join(self.ckpt_dir, "vecnormalize.pkl")
                if os.path.exists(pkl_path):
                    try:
                        subprocess.run(
                            ["rclone", "copy", pkl_path, self.rclone_dest,
                            "--drive-chunk-size", "64M", "--transfers", "2", "--checkers", "4", "-q"],
                            check=True
                        )
                        print(f"[VecNormalize] pushed vecnormalize.pkl -> {self.rclone_dest}")
                    except subprocess.CalledProcessError:
                        print("[VecNormalize] pkl upload failed.")

            except subprocess.CalledProcessError as e:
                # If you prefer to continue training on upload failures, change this to a print().
                print("[Checkpoint] Upload failure.")
        else:
            print("[Checkpoint] No RCLONE_DEST set; skipped remote upload.")

        return True

    
class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, check_freq, rclone_dest: str = "", verbose=1):
        super().__init__(verbose)
        self.save_path = save_path                  # can be with or without ".zip"
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        # allow passing via arg or env; empty means "don’t upload"
        self.rclone_dest = rclone_dest or os.getenv("RCLONE_DEST", "")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        results = []
        for i in range(self.training_env.num_envs):
            monitor_file = os.path.join(LOGS_DIR, f"onemin_worker_{i}", "monitor.csv")
            if os.path.exists(monitor_file):
                df = pd.read_csv(monitor_file, skiprows=1)
                if "r" in df.columns:
                    results.extend(df["r"].values[-200:])

        if not results:
            return True

        mean_reward = float(np.mean(results))
        if self.verbose:
            print(f"[SaveBestModel] mean_reward={mean_reward:.3f} best={self.best_mean_reward:.3f}")

        if mean_reward > self.best_mean_reward + 1e-4:
            self.best_mean_reward = mean_reward
            self.model.save(self.save_path)
            fzip = self.save_path if self.save_path.endswith(".zip") else (self.save_path + ".zip")
            if self.verbose:
                print(f"[SaveBestModel] New best → saved to {fzip}")

            # persist VecNormalize stats
            try:
                if isinstance(self.training_env, VecNormalize):
                    self.training_env.save(os.path.join(os.path.dirname(self.save_path), "vecnormalize.pkl"))
            except Exception as e:
                print(f"[VecNormalize] Save (best) failed: {e}")

            # optional uploads
            if self.rclone_dest:
                try:
                    subprocess.run(
                        ["rclone", "copy", fzip, self.rclone_dest,
                        "--drive-chunk-size", "64M", "--transfers", "2", "--checkers", "4", "-q"],
                        check=True
                    )
                    if self.verbose:
                        print(f"[SaveBestModel] pushed {os.path.basename(fzip)} -> {self.rclone_dest}")
                except FileNotFoundError:
                    print("[SaveBestModel] rclone not found on PATH; skipped upload.")
                except subprocess.CalledProcessError as e:
                    print(f"[SaveBestModel] Upload failed: {e}. Training continues.")

                # push vecnormalize.pkl too
                pkl_path = os.path.join(os.path.dirname(fzip), "vecnormalize.pkl")
                if os.path.exists(pkl_path):
                    try:
                        subprocess.run(
                            ["rclone", "copy", pkl_path, self.rclone_dest,
                            "--drive-chunk-size", "64M", "--transfers", "2", "--checkers", "4", "-q"],
                            check=True
                        )
                        if self.verbose:
                            print(f"[VecNormalize] pushed vecnormalize.pkl -> {self.rclone_dest}")
                    except subprocess.CalledProcessError as e:
                        print(f"[VecNormalize] pkl upload failed: {e}")
        return True


class _NoOpCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

class EarlyStoppingCallback(EventCallback):
    """
    Patience-based early stopping using mean monitor reward (mirrors onemin).
    """
    def __init__(self, check_freq: int, patience: int, verbose=1):
        super().__init__(callback=_NoOpCallback(), verbose=verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.counter = 0
        self.verbose = verbose

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        results = []
        for i in range(self.training_env.num_envs):
            monitor_file = os.path.join(LOGS_DIR, f"onemin_worker_{i}", "monitor.csv")
            if os.path.exists(monitor_file):
                df = pd.read_csv(monitor_file, skiprows=1)
                if "r" in df.columns:
                    results.extend(df["r"].values[-200:])

        if not results:
            return True

        mean_reward = float(np.mean(results))
        if self.verbose:
            print(
                f"[EarlyStopping] step={self.num_timesteps} "
                f"mean_reward={mean_reward:.3f} best={self.best_mean_reward:.3f} "
                f"counter={self.counter}/{self.patience}"
            )

        if mean_reward > self.best_mean_reward + 1e-4:
            self.best_mean_reward = mean_reward
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("[EarlyStopping] Patience exceeded → stopping training.")
                return False
        return True

# ──────────────────────────────────────────────────────────────────────────────
# Environment (onemin) — fixed indexing + ATR handling
# ──────────────────────────────────────────────────────────────────────────────
class OneMinBacktestEnv(gym.Env):
    """
    Onemin OHLC backtest with:
      - Single open trade per symbol
      - ATR+broker-floor mapped SL/TP from normalized [-1,1]
      - Per-step reward (event terms fire on close; penalties can accrue each step)
      - Illegal-action masking + penalties
      - RAM-efficient on-demand CSV loading with safe local indexing
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_dir: str,
        symbols: List[str],
        window: int = 24,
        max_steps: Optional[int] = None,
        seed: int = SEED,
        initial_balance: float = INITIAL_BALANCE,
    ):
        super().__init__()
        self.window = int(window)
        self.symbols = list(symbols)
        self.n_assets = len(self.symbols)
        self.seed_val = int(seed)
        self.initial_balance = float(initial_balance)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)

        # Broker meta (min_stop_price per symbol)
        self.broker_meta = load_broker_meta(BROKER_STOPS_JSON)

        # CSV metadata only (read slices on reset)
        self.data_paths: Dict[str, str] = {}
        self.data_lengths: Dict[str, int] = {}
        for sym in self.symbols:
            path = os.path.join(csv_dir, f"{sym}_1min.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"1min CSV not found for '{sym}': {path}")
            with open(path, "r", encoding="utf-8") as f:
                n_rows = sum(1 for _ in f) - 1
            self.data_paths[sym] = path
            self.data_lengths[sym] = n_rows

        self.max_length = min(self.data_lengths.values())
        if max_steps is None:
            self.max_steps = min(1000, self.max_length - self.window - 1)
        else:
            self.max_steps = min(int(max_steps), self.max_length - self.window - 1)
        if self.max_steps <= 0:
            raise ValueError("max_steps ≤ 0 after clamping")

        # Spaces
        obs_dim = self.n_assets * (4 * self.window + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        act_dim = self.n_assets * 10  # [one_hot(8), sl_norm, tp_norm] per asset
        self.action_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)

        # State
        self.current_step = 0
        self.start_idx = 0
        self.balance = self.initial_balance

        self.last_trade_time = np.zeros(self.n_assets, dtype=np.float32)
        self.open_trades: List[Dict[str, Any]] = []   # one per symbol at most
        self.closed_trades: List[Dict[str, Any]] = [] # rolling buffer

        self.dfs: Dict[str, pd.DataFrame] = {}  # chunked data loaded on reset

        self.reward_fn = RewardFunction(
        initial_balance=self.initial_balance,
        slippage_per_unit=SLIPPAGE_PER_UNIT,
        commission_per_trade=COMMISSION_PER_TRADE,

        # time pressure ↓ and only when flat (see reward_utils change)
        inactivity_weight=0.0001,
        inactivity_grace_steps=120,
        holding_threshold_steps=180,
        holding_penalty_per_step=0.001,

        # slightly stronger positive signal for realized R,
        # slightly softer overexposure pressure
        realized_R_weight=1.5,
        risk_budget_R=4.0,
        overexposure_weight=0.015,

        unrealized_weight=0.03,
        component_clip=2.0,
        final_clip=2.5,
        integrate_costs_in_reward=False,
    )

        # Local indexing controls (fix)
        self.cursor: int = 0               # local index within the sliced DataFrame
        self.runtime_max_steps: int = 0    # per-reset cap after ATR/cleaning across symbols

        self.seed(self.seed_val)

    # Gymnasium API
    def seed(self, seed: Optional[int] = None):
        self.seed_val = int(seed) if seed is not None else self.seed_val
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        try:
            self.action_space.seed(self.seed_val)
            self.observation_space.seed(self.seed_val)
        except Exception:
            pass
        return [self.seed_val]

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        low = self.window
        high = self.max_length - self.max_steps - 1
        self.start_idx = random.randint(low, high)
        self.current_step = 0

        # RAM-efficient load of only the needed rows
        self.dfs.clear()
        min_len_after_atr = float("inf")

        for sym in self.symbols:
            path = self.data_paths[sym]
            needed_rows = self.start_idx + self.max_steps + self.window + 2
            skip = self.start_idx - self.window if self.start_idx > self.window else 0
            skiprows = list(range(1, skip + 1)) if skip > 0 else []

            df = pd.read_csv(path, parse_dates=["time"], date_format="ISO8601", skiprows=skiprows, nrows=needed_rows - skip)
            df = _safe_numeric(df)

            # Compute ATR with strict rolling; fallback to Wilder if too short
            df["atr"] = compute_atr(df, n=14, mode="rolling")
            tmp = df.dropna(subset=["atr"])
            if len(tmp) >= self.window + 2:
                df = tmp.reset_index(drop=True)
                df["atr"] = df["atr"].clip(lower=1e-12)
            else:
                df["atr"] = compute_atr(df, n=14, mode="wilder").clip(lower=1e-12)
                df = df.reset_index(drop=True)

            if len(df) < self.window + 2:
                raise ValueError(f"[{sym}] Not enough onemin bars after ATR handling (have {len(df)}, need {self.window + 2})")

            self.dfs[sym] = df
            min_len_after_atr = min(min_len_after_atr, len(df))

        # We loaded from (start_idx - window), so "start" within local slices == window
        self.cursor = self.window

        # After cleaning, clamp per-reset max steps by what's truly aligned across symbols
        self.runtime_max_steps = min(self.max_steps, int(min_len_after_atr) - self.window - 1)
        if self.runtime_max_steps <= 0:
            raise ValueError(
                f"Not enough aligned 1min bars across symbols after ATR handling "
                f"(min_len={min_len_after_atr}, window={self.window})"
            )

        self.balance = self.initial_balance
        self.last_trade_time.fill(0.0)
        self.open_trades.clear()
        self.closed_trades.clear()
        self.reward_fn.reset()

        obs = self._get_observation(self.cursor)
        return obs.astype(np.float32), {}

    def _get_observation(self, idx: int) -> np.ndarray:
        obs_parts = []
        for i, sym in enumerate(self.symbols):
            df = self.dfs[sym]
            window_start = idx - self.window
            slice_df = df.iloc[window_start:idx]
            opens = slice_df["open"].to_numpy(dtype=np.float32)
            highs = slice_df["high"].to_numpy(dtype=np.float32)
            lows = slice_df["low"].to_numpy(dtype=np.float32)
            closes = slice_df["close"].to_numpy(dtype=np.float32)
            if any(len(arr) != self.window for arr in (opens, highs, lows, closes)):
                raise RuntimeError(
                    f"[{sym}] expected window={self.window} but got "
                    f"o={len(opens)} h={len(highs)} l={len(lows)} c={len(closes)} (idx={idx})"
                )
            ot = self._get_open_trade(sym)
            if ot is None:
                extra = np.array([0.0], dtype=np.float32)
            else:
                held = float(self.current_step - ot.get("open_step", self.current_step))
                extra = np.array([held if ot["trade_type"] == "long" else -held], dtype=np.float32)
            obs_parts.append(np.concatenate([opens, highs, lows, closes, extra], axis=0))
        return np.concatenate(obs_parts, axis=0)

    # Single open trade per symbol
    def _get_open_trade(self, sym: str) -> Optional[Dict[str, Any]]:
        for t in self.open_trades:
            if t["symbol"] == sym:
                return t
        return None

    def _mask_illegal_actions(self, i: int, arr: np.ndarray) -> np.ndarray:
        valid = np.ones(8, dtype=bool)
        sym = self.symbols[i]
        ot = self._get_open_trade(sym)
        if ot is not None:
            # In a trade -> cannot buy/sell; can close (3/4/7) and adjust (5/6)
            valid[1] = False  # buy
            valid[2] = False  # sell
            if ot["trade_type"] == "long":
                valid[4] = False  # can't close short
            elif ot["trade_type"] == "short":
                valid[3] = False  # can't close long
            # 5 (adjust SL) and 6 (adjust TP) remain valid while in a trade
        else:
            # Not in trade -> cannot close/adjust/close-all
            valid[3:8] = False

        masked = arr.copy()
        masked[:8] = np.where(valid, arr[:8], -np.inf)
        return masked


    def step(self, action: np.ndarray):
        idx = self.cursor
        next_idx = idx + 1

        reward = 0.0
        info: Dict[str, Any] = {"symbols": {}, "closed_trades": []}
        illegal_penalty_total = 0.0
        for i, sym in enumerate(self.symbols):
            df = self.dfs[sym]
            act = action[i * 10: (i + 1) * 10]
            # Safety net: never allow >1 open trade per symbol.
            open_for_sym = [t for t in self.open_trades if t["symbol"] == sym]
            if len(open_for_sym) > 1:
                logging.error(f"[{sym}] Multiple open trades detected; auto-closing extras.")
                illegal_penalty_total += SEVERE_ILLEGAL_ACTION_PENALTY
                # Close all but the first at breakeven (costs were charged on open)
                for extra in open_for_sym[1:]:
                    closed = {
                        **extra,
                        "exit_price": extra["entry_price"],
                        "pnl": 0.0,
                        "slippage": _to_float(self.reward_fn.slippage_per_unit),
                        "commission": _to_float(self.reward_fn.commission_per_trade),
                        "close_step": self.current_step,
                        "stop_type": "illegal_auto",
                    }
                    self.closed_trades.append(closed)
                    self.balance += closed["pnl"]
                    info["closed_trades"].append(closed)
                # Keep only one
                keep = open_for_sym[0]
                self.open_trades = [t for t in self.open_trades if t["symbol"] != sym] + [keep]

            # (A) What the policy *wanted* to do (before masking) — for penalty accounting
            raw_head = np.where(np.isfinite(act[:8]), act[:8], -np.inf)
            orig_action = int(np.argmax(raw_head))

            # (B) Mask invalid heads *before* argmax so sampled==executed (clean credit assignment)
            masked = self._mask_illegal_actions(i, act)         # sets invalid heads to -inf
            masked_head = masked[:8]
            act_id = int(np.argmax(masked_head))

            # (C) Optional: small penalty if the policy's original choice was illegal
            #      (define the valid mask the same way as the masker does)
            valid = np.ones(8, dtype=bool)
            ot = self._get_open_trade(sym)
            if ot is not None:
                valid[1] = False  # buy
                valid[2] = False  # sell
                if ot["trade_type"] == "long":
                    valid[4] = False  # can't close short
                elif ot["trade_type"] == "short":
                    valid[3] = False  # can't close long
                # 5 (adjust SL) and 6 (adjust TP) are valid while in a trade
            else:
                valid[3:8] = False  # cannot close/adjust when flat


            attempted_illegal = not valid[orig_action]
            if attempted_illegal:
                illegal_penalty_total += ILLEGAL_ATTEMPT_PENALTY

            # Prices
            # Use ATR from the last bar inside the observation window (no look-ahead)
            # Use ATR from the last bar inside the observation window (no look-ahead)
            curr_idx = idx - 1
            if curr_idx < 0:
                curr_idx = 0
            atr_val    = float(df.at[curr_idx, "atr"])
            curr_close = float(df.at[curr_idx, "close"])  # <-- needed for SL/TP adjustments

            # Next bar (orders execute / SL-TP evaluated here)
            next_open  = float(df.at[next_idx, "open"])
            next_close = float(df.at[next_idx, "close"])
            next_high  = float(df.at[next_idx, "high"])
            next_low   = float(df.at[next_idx, "low"])

            min_stop_price = float(self.broker_meta.get(sym, {}).get("min_stop_price", 0.0))
            sl_norm = float(act[8])
            tp_norm = float(act[9])

            trade_executed = False
            ot = self._get_open_trade(sym)

            # === Actions ===
            if act_id == 1 and ot is None:  # buy
                entry = next_open
                sl_final, tp_final = _scale_sl_tp(
                    entry=entry,
                    sl_norm=sl_norm,
                    tp_norm=tp_norm,
                    is_long=True,
                    min_stop_price=min_stop_price,
                    atr_value=atr_val,
                )
                self._open_trade(sym, "long", entry, sl_final, tp_final)
                trade_executed = True

            elif act_id == 2 and ot is None:  # sell
                entry = next_open
                sl_final, tp_final = _scale_sl_tp(
                    entry=entry,
                    sl_norm=sl_norm,
                    tp_norm=tp_norm,
                    is_long=False,
                    min_stop_price=min_stop_price,
                    atr_value=atr_val,
                )
                self._open_trade(sym, "short", entry, sl_final, tp_final)
                trade_executed = True

            elif act_id in (3, 4, 7) and ot is not None:
                # Manual close at NEXT OPEN (no future leakage)
                exit_px = next_open
                if ot["trade_type"] == "long":
                    pnl = exit_px - ot["entry_price"]
                else:
                    pnl = ot["entry_price"] - exit_px
                closed = dict(
                    **ot,
                    exit_price=exit_px,
                    pnl=pnl,
                    slippage=_to_float(self.reward_fn.slippage_per_unit),
                    commission=_to_float(self.reward_fn.commission_per_trade),
                    close_step=self.current_step,
                    stop_type="manual",
                )
                self.closed_trades.append(closed)
                self.balance += closed["pnl"]
                self.open_trades = [t for t in self.open_trades if t["symbol"] != sym]
                info["closed_trades"].append(closed)
                trade_executed = True
            
            elif act_id == 5 and ot is not None:
                # Adjust Stop-Loss using the LAST OBSERVED close as reference (curr_close)
                ref_price = curr_close
                is_long = (ot["trade_type"] == "long")
                new_sl, _unused = _scale_sl_tp(
                    entry=ref_price,
                    sl_norm=sl_norm,
                    tp_norm=0.0,
                    is_long=is_long,
                    min_stop_price=min_stop_price,
                    atr_value=atr_val,
                )
                floor_price = max(float(min_stop_price or 0.0), 0.40 * atr_val)
                if is_long:
                    new_sl = min(new_sl, ot["take_profit"] - floor_price, ref_price - floor_price)
                else:
                    new_sl = max(new_sl, ot["take_profit"] + floor_price, ref_price + floor_price)
                ot["stop_loss"] = float(new_sl)
                trade_executed = False
                info["symbols"][sym] = {**info["symbols"].get(sym, {}), "adjusted": "sl"}


            elif act_id == 6 and ot is not None:
                # Adjust Take-Profit using the LAST OBSERVED close as reference (curr_close)
                ref_price = curr_close
                is_long = (ot["trade_type"] == "long")
                _unused, new_tp = _scale_sl_tp(
                    entry=ref_price,
                    sl_norm=0.0,
                    tp_norm=tp_norm,
                    is_long=is_long,
                    min_stop_price=min_stop_price,
                    atr_value=atr_val,
                )
                floor_price = max(float(min_stop_price or 0.0), 0.40 * atr_val)
                if is_long:
                    new_tp = max(new_tp, ot["stop_loss"] + floor_price, ref_price + floor_price)
                else:
                    new_tp = min(new_tp, ot["stop_loss"] - floor_price, ref_price - floor_price)
                ot["take_profit"] = float(new_tp)
                trade_executed = False
                info["symbols"][sym] = {**info["symbols"].get(sym, {}), "adjusted": "tp"}

  
           # === Auto-close by SL/TP using next bar high/low ===
            if not trade_executed:
                ot = self._get_open_trade(sym)
                if ot is not None:
                    long_side = (ot["trade_type"] == "long")
                    sl = float(ot["stop_loss"])
                    tp = float(ot["take_profit"])
                    entry = float(ot["entry_price"])

                    hit_sl = (next_low <= sl) if long_side else (next_high >= sl)
                    hit_tp = (next_high >= tp) if long_side else (next_low <= tp)

                    # If both levels are inside the next bar, decide by proximity to next_open (proxy for first touch)
                    if hit_sl and hit_tp:
                        sl_dist = abs(next_open - sl)
                        tp_dist = abs(next_open - tp)
                        prefer_sl = sl_dist <= tp_dist
                        hit_sl, hit_tp = prefer_sl, (not prefer_sl)

                    if hit_sl:
                        exit_px = sl
                        pnl = (sl - entry) if long_side else (entry - sl)
                        stop_type = "sl"
                        hit = True
                    elif hit_tp:
                        exit_px = tp
                        pnl = (tp - entry) if long_side else (entry - tp)
                        stop_type = "tp"
                        hit = True
                    else:
                        hit = False

                    if hit:
                        closed = dict(
                            **ot,
                            exit_price=exit_px,
                            pnl=pnl,
                            slippage=_to_float(self.reward_fn.slippage_per_unit),
                            commission=_to_float(self.reward_fn.commission_per_trade),
                            close_step=self.current_step,
                            stop_type=stop_type,
                        )
                        self.closed_trades.append(closed)
                        self.balance += closed["pnl"]
                        self.open_trades = [t for t in self.open_trades if t["symbol"] != sym]
                        info["closed_trades"].append(closed)
                        trade_executed = True


            self.last_trade_time[i] = 0.0 if trade_executed else (self.last_trade_time[i] + 1.0)
            info["symbols"][sym] = {
                **info["symbols"].get(sym, {}),
                "executed": trade_executed,
                "action_id": act_id,
                "attempted_illegal": bool(attempted_illegal),
            }

        # === Event-based reward: only when trades close ===
        global_since_last_trade = float(np.min(self.last_trade_time)) if len(self.last_trade_time) else 0.0

        # include holding_time for open trades so C5 works
        open_trades_for_reward = []
        for t in self.open_trades:
            td = dict(t)
            td["holding_time"] = self.current_step - td.get("open_step", self.current_step)
            open_trades_for_reward.append(td)

        reward = float(self.reward_fn(
            closed_trades=info["closed_trades"],      # may be []
            open_trades=open_trades_for_reward,       # includes holding_time
            account_balance=self.balance,
            unrealized_pnl=0.0,
            time_since_last_trade=global_since_last_trade
        ).item())

        # keep your existing illegal-action penalties
        reward += illegal_penalty_total

        # final clip AFTER penalties, same as 1-min
        final_cap = getattr(self.reward_fn, "final_clip", 5.0) or 5.0
        reward = float(np.clip(reward, -final_cap, final_cap))
        # Keep only recent closed trades
        MAX_CLOSED_TRADES = 1000
        if len(self.closed_trades) > MAX_CLOSED_TRADES:
            self.closed_trades = self.closed_trades[-MAX_CLOSED_TRADES:]

        self.current_step += 1
        terminated = self.current_step >= self.runtime_max_steps
        truncated = False

        if not (terminated or truncated):
            self.cursor += 1
            obs = self._get_observation(self.cursor).astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, float(reward), terminated, truncated, info

    def _open_trade(self, sym: str, direction: str, entry: float, sl: float, tp: float):
        self.open_trades = [t for t in self.open_trades if t["symbol"] != sym]  # ensure at most one
        trade = dict(
            symbol=sym,
            trade_type=direction,
            entry_price=float(entry),
            stop_loss=float(sl),
            take_profit=float(tp),
            volume=1.0,
            open_step=self.current_step,
        )
        self.open_trades.append(trade)
        # Pay costs upfront
        cost = _to_float(self.reward_fn.slippage_per_unit) + _to_float(self.reward_fn.commission_per_trade)
        self.balance -= cost

    def render(self, mode="human"):
        idx = self.cursor
        prices = {sym: float(self.dfs[sym].at[idx, "close"]) for sym in self.symbols}
        print(f"Step {self.current_step} | Prices={prices} | OpenTrades={len(self.open_trades)} | Balance={self.balance:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# VecEnv factory
# ──────────────────────────────────────────────────────────────────────────────
def make_onemin_env(rank: int, seed: int, window: int, symbols=None) -> Callable[[], gym.Env]:
    def _init():
        env = OneMinBacktestEnv(
            csv_dir=ONEMIN_CSV_DIR,
            symbols=symbols or LIVE_FOREX_PAIRS,
            window=window,
            max_steps=1000,
            seed=seed + rank,
        )
        env.seed(seed + rank)
        log_dir = os.path.join(LOGS_DIR, f"onemin_worker_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        return env
    return _init

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_latest_checkpoint(ckpt_dir: str, last_ckpt_path: str, main_save_path: str) -> Optional[str]:
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "onemin_policy_ckpt_*_steps.zip"))
    if ckpt_files:
        extract_steps = lambda f: int(re.findall(r"ckpt_(\d+)_steps", f)[0])
        latest_ckpt = max(ckpt_files, key=extract_steps)
        return latest_ckpt
    elif os.path.exists(last_ckpt_path):
        return last_ckpt_path
    elif os.path.exists(main_save_path):
        return main_save_path
    else:
        return None

def steps_from_ckpt_name(path: str) -> int:
    m = re.search(r"ckpt_(\d+)_steps", os.path.basename(path))
    return int(m.group(1)) if m else 0

# ──────────────────────────────────────────────────────────────────────────────
# Training (mirrors onemin structure, prefers recurrent)
# ──────────────────────────────────────────────────────────────────────────────
def train_onemin_policy(
    window: int = ONEMIN_OBS_WINDOW,
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    checkpoint_freq: int = 10_000,
    patience: int = 500,
    early_stopping_check_freq: int = 10_000,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_onemin_policy")
    logger.info("Starting onemin policy training with PPO and event-based reward.")

    # VecEnv (+ VecNormalize load/resume)
    env_fns = [make_onemin_env(i, SEED, window) for i in range(n_envs)]
    base_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv([env_fns[0]])

    pkl_path = os.path.join(MODELS_DIR, "checkpoints_onemin", "vecnormalize.pkl")
    if os.path.exists(pkl_path):
        vec_env = VecNormalize.load(pkl_path, base_env)
        vec_env.training = True
    else:
        vec_env = VecNormalize(
            base_env,
            norm_obs=False,
            norm_reward=True,
            gamma=0.995,
            clip_reward=np.inf,
        )


    n_steps = 2048
    rollout = n_steps * n_envs
    for cand in (1024, 512, 256, 128, 64):
        if rollout % cand == 0:
            batch_size = cand
            break
    print(f"[train_onemin] n_steps={n_steps} n_envs={n_envs} batch_size={batch_size} rollout={rollout}")
    assert (n_steps * n_envs) % batch_size == 0, "n_steps * n_envs must be divisible by batch_size."
    algo_cls = PPO
    policy_cls = OneMinOHLCPolicy
    algo_kwargs = dict(
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=2e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(LOGS_DIR, "tb_onemin_policy"),
        device="cuda",
    )
    policy_kwargs = dict(window=window)

    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_dir = os.path.join(MODELS_DIR, "checkpoints_onemin")
    os.makedirs(ckpt_dir, exist_ok=True)
    main_save_path = os.path.join(ckpt_dir, "onemin_policy.zip")
    last_ckpt_path = os.path.join(ckpt_dir, "onemin_policy_last.zip")

    # Resume logic (unchanged)
    resume_path = get_latest_checkpoint(ckpt_dir, last_ckpt_path, main_save_path)
    target_total_timesteps = int(total_timesteps)

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        model = algo_cls.load(resume_path, env=vec_env, device="cuda")
        already_trained = getattr(model, "num_timesteps", steps_from_ckpt_name(resume_path))
        print(f"Already trained (model counter): {already_trained} steps")
        timesteps_left = max(target_total_timesteps - already_trained, 0)
    else:
        model = algo_cls(
            policy=policy_cls,
            env=vec_env,
            verbose=1,
            seed=SEED,
            policy_kwargs=policy_kwargs,
            **algo_kwargs,
        )
        timesteps_left = target_total_timesteps

    # Callbacks (keep your existing classes)
    # Build callback list (checkpoint only if RCLONE_DEST is set)
    rclone_dest = os.getenv("RCLONE_DEST")

    best_model_callback = SaveBestModelCallback(
        save_path=os.path.join(ckpt_dir, "onemin_policy_best.zip"),
        check_freq=checkpoint_freq,
        rclone_dest=rclone_dest, 
        verbose=1,
    )
    early_stopping_callback = EarlyStoppingCallback(
        check_freq=early_stopping_check_freq,
        patience=patience,
        verbose=1,
    )

    callbacks = [best_model_callback, early_stopping_callback]
    if rclone_dest:
        checkpoint_callback = CheckpointAndRcloneCallback(
            checkpoint_freq=checkpoint_freq,
            ckpt_dir=ckpt_dir,
            name_prefix="onemin_policy_ckpt",
            rclone_dest=rclone_dest,
        )
        callbacks.insert(0, checkpoint_callback)
    else:
        print("[Checkpoint] RCLONE_DEST not set; saving locally only.")
    # ── Train & save (same pattern as onemin) ────────────────────────────────
    try:
        if timesteps_left > 0:
            model.learn(
                total_timesteps=timesteps_left,
                callback=callbacks,
                reset_num_timesteps=False,
            )
            model.save(last_ckpt_path)
            model.save(main_save_path)
            # persist VecNormalize running stats for clean resumes
            try:
                vec_env.save(os.path.join(ckpt_dir, "vecnormalize.pkl"))
                print(f"[VecNormalize] Saved running stats -> {os.path.join(ckpt_dir, 'vecnormalize.pkl')}")
            except Exception as e:
                print(f"[VecNormalize] Could not save stats: {e}")
            logger.info(f"Training complete. Model saved to {main_save_path} and {last_ckpt_path}")
        else:
            logger.info(f"Training already completed by checkpoint/model counter: >= target {target_total_timesteps}")
    finally:
        vec_env.close()
if __name__ == "__main__":
    train_onemin_policy()
