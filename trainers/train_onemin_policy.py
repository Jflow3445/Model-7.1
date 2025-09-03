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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import subprocess
from stable_baselines3.common.vec_env import VecNormalize
import queue, threading, shutil, atexit, sys
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
    LOT_MULTIPLIER,
)
from utils.reward_utils import RewardFunction

# Broker stops (floor)
BROKER_STOPS_JSON = Path(BASE_DIR) / "config" / "broker_stops.json"

# ──────────────────────────────────────────────────────────────────────────────
# Constants / Logging
# ──────────────────────────────────────────────────────────────────────────────
LOGS_DIR = os.path.join(MODELS_DIR, "logs")
SEVERE_ILLEGAL_ACTION_PENALTY = -2
ILLEGAL_ATTEMPT_PENALTY = -0.002
MIN_MANUAL_HOLD_STEPS = 2
SL_ILLEGAL_PENALTY = -0.02
SL_COOLDOWN_STEPS = 2
SL_EARLY_STEPS = 3

os.makedirs(LOGS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
class AsyncUploader:
    def __init__(self, dest: str, max_queue:int=4):
        self.dest = dest
        self.q = queue.Queue(max_queue)
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()
        atexit.register(self.wait)

    def submit(self, file_path: str):
        try:
            self.q.put_nowait(file_path)
            return True
        except queue.Full:
            print(f"[Checkpoint] upload queue full; skipping {os.path.basename(file_path)}")
            return False

    def _worker(self):
        while True:
            path = self.q.get()
            if path is None:  # shutdown
                self.q.task_done()
                break
            try:
                if not os.path.exists(path):
                    print(f"[Checkpoint] file vanished, skip: {path}")
                elif not self.dest:
                    # nothing to do if no remote set
                    pass
                else:
                    # low-priority, non-blocking process; no stdout/stderr spam
                    cmd = ["rclone", "copy", path, self.dest,
                           "--drive-chunk-size", "64M", "--transfers", "1",
                           "--checkers", "2", "-q"]
                    # Optionally "nice" the process on Linux:
                    if shutil.which("nice"):
                        cmd = ["nice", "-n", "19"] + cmd
                    # Start and do NOT wait:
                    subprocess.Popen(cmd,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     start_new_session=True)
                    print(f"[Checkpoint] enqueued → {os.path.basename(path)} -> {self.dest}")
            except Exception as e:
                print(f"[Checkpoint] async push error: {e}", file=sys.stderr)
            finally:
                self.q.task_done()

    def wait(self):
        """Drain outstanding uploads at process exit (non-blocking during training)."""
        try:
            self.q.put(None)
            self.q.join()
        except Exception:
            pass

def _to_float(x):
    try:
        return float(x.item())
    except AttributeError:
        return float(x)

def load_broker_meta(json_path: Path) -> Dict[str, Dict]:
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    logging.warning(f"[train_oneminute] broker_stops.json not found at {json_path}, using zeros.")
    return {}

def compute_atr(df: pd.DataFrame, n: int = 14, mode: str = "rolling") -> pd.Series:
    """
    Robust ATR (onemin):
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
    Map normalized [-1,1] to price distances using ATR + broker + price floors.
    Prevents microscopic stops when ATR is tiny.
    """
    FLOOR_FRAC_ATR = 0.40
    K_SL_ATR = 0.80
    K_TP_ATR = 1.60

    # NEW: price-based floor (e.g., 5 bps of price)
    PRICE_FRAC_FLOOR = 5e-4  # 0.05% of price

    entry = float(entry)
    atr_value = float(atr_value) if np.isfinite(atr_value) else 0.0

    price_floor = PRICE_FRAC_FLOOR * abs(entry)
    atr_base = max(atr_value, price_floor)  # NEW: never smaller than price floor

    floor_price = max(float(min_stop_price or 0.0), FLOOR_FRAC_ATR * atr_base)

    sl_dist = max(floor_price, (0.5 + abs(sl_norm)) * K_SL_ATR * atr_base)
    tp_dist = max(floor_price, (0.5 + abs(tp_norm)) * K_TP_ATR * atr_base)

    if is_long:
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
    return sl, tp

RCLONE_DEST = os.getenv("RCLONE_DEST", "")
ASYNC_UPLOADER = AsyncUploader(RCLONE_DEST) if RCLONE_DEST else None
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
        fzip = fpath + ".zip"
        exists = os.path.exists(fzip)
        print(f"[Checkpoint] wrote {fzip}  exists={exists}")

        if not exists:
            raise RuntimeError(f"Checkpoint file missing after save: {fzip}")

        # Only push if a remote is configured
        if self.rclone_dest and ASYNC_UPLOADER:
            ASYNC_UPLOADER.submit(fzip)  # returns immediately
        else:
            print("[Checkpoint] No RCLONE_DEST set; skipped remote upload.")
        return True

class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, check_freq, rclone_dest: str = "", verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = int(check_freq)
        self.best_mean_reward = -np.inf
        # Allow either explicit arg or env var
        self.rclone_dest = rclone_dest or os.getenv("RCLONE_DEST", "")

    def _on_step(self) -> bool:
        if int(self.model.num_timesteps) % int(self.check_freq) != 0:
            return True
        # Gather recent episodic returns
        results = []
        for i in range(self.training_env.num_envs):
            monitor_file = os.path.join(LOGS_DIR, f"onemin_worker_{i}", "monitor.csv")
            if os.path.exists(monitor_file):
                df = pd.read_csv(monitor_file, skiprows=1)
                if "r" in df.columns:
                    results.extend(df["r"].values[-200:])  # last 200 episodic returns

        if not results:
            return True

        mean_reward = float(np.mean(results))
        # Print last-200 episode means for any rw_* columns if present
        try:
            comp_means = {}
            for i in range(self.training_env.num_envs):
                monitor_file = os.path.join(LOGS_DIR, f"onemin_worker_{i}", "monitor.csv")
                if os.path.exists(monitor_file):
                    df = pd.read_csv(monitor_file, skiprows=1)
                    cols = [c for c in df.columns if c.startswith("rw_")]
                    if cols:
                        tail = df[cols].tail(200)
                        # merge (mean across envs)
                        for c in cols:
                            comp_means[c] = comp_means.get(c, []) + [tail[c].mean()]
            if comp_means and self.verbose:
                msg = " | ".join(
                    f"{k}={np.nanmean(v):.3f}" for k, v in sorted(comp_means.items())
                )
                print(f"[Diagnostics] {msg}")
        except Exception:
            pass

        if self.verbose:
            print(f"[SaveBestModel] mean_reward={mean_reward:.3f} best={self.best_mean_reward:.3f}")

        if mean_reward > self.best_mean_reward + 1e-4:
            self.best_mean_reward = mean_reward
            # Save best model
            self.model.save(self.save_path)
            # Ensure path points to the .zip file SB3 writes
            best_zip = self.save_path if self.save_path.endswith(".zip") else self.save_path + ".zip"
            if self.verbose:
                print(f"[SaveBestModel] New best → saved to {best_zip}")

            # Optionally push to remote (non-blocking)
            if self.rclone_dest and ASYNC_UPLOADER:
                ASYNC_UPLOADER.submit(best_zip)

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
        # Trigger on exact timesteps so it lines up with checkpoint_freq etc.
        if int(self.model.num_timesteps) % self.check_freq != 0:
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

class LogStdCallback(BaseCallback):
    def _on_step(self) -> bool:
        try:
            log_std = None

            # 1) Preferred: what the policy actually used this rollout
            if hasattr(self.model.policy, "get_log_std_for_logging"):
                log_std = self.model.policy.get_log_std_for_logging()

            # 2) Fallbacks for other policies / modes
            if log_std is None:
                log_std = getattr(self.model.policy, "log_std", None)           # global param path
            if log_std is None:
                # our hybrid distribution lives under _hybrid
                hybrid = getattr(self.model.policy, "_hybrid", None)
                if hybrid is not None:
                    cont_dist = getattr(hybrid, "cont_dist", None)
                    log_std = getattr(cont_dist, "log_std", None)

            if isinstance(log_std, (np.ndarray, float)):
                std_val = float(np.exp(log_std).mean())
                self.model.logger.record("policy/std", std_val)
            elif hasattr(log_std, "exp"):
                self.model.logger.record("policy/std", float(log_std.exp().mean().item()))
        except Exception:
            pass
        return True

class LogRewardComponentsCallback(BaseCallback):
    """
    Aggregate env.info[] keys across the last rollout and push them into the SB3 logger
    so they show up in the progress table.
    """
    def __init__(self, keys=None, section: str = "rollout", verbose: int = 0):
        super().__init__(verbose)
        self.section = section
        self.keys = keys or [
        "rw_C1_realizedR", "rw_C2_quality", "rw_C3_unreal", "rw_C4_inactivity",
        "rw_C5_holding", "rw_C6_overexp", "rw_C7_conflict", "rw_C8_churnSLTP",
        "rw_realized_R_mean",              # NEW
        "rw_total_before_clip",
        "n_open", "n_closed", "illegal_attempts", "since_last_trade",
        "c_sl", "c_tp", "c_manual",        # NEW
        ]

        self._sums = {}
        self._counts = {}

    def _on_rollout_start(self) -> None:
        self._sums = {k: 0.0 for k in self.keys}
        self._counts = {k: 0 for k in self.keys}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            for k in self.keys:
                v = info.get(k, None)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    self._sums[k] += fv
                    self._counts[k] += 1
        return True

    def _on_rollout_end(self) -> None:
        # Record means so they appear in the console table
        for k in self.keys:
            c = self._counts.get(k, 0)
            if c > 0:
                mean = self._sums[k] / c
                self.model.logger.record(f"{self.section}/{k}", mean)
        # PPO will call logger.dump() itself after this, so no need to dump here.

# ──────────────────────────────────────────────────────────────────────────────
# Environment (onemin) — fixed indexing + ATR handling
# ──────────────────────────────────────────────────────────────────────────────
class OneMinBacktestEnv(gym.Env):
    """
    OneMin OHLC backtest with:
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
        self.arr: Dict[str, Dict[str, np.ndarray]] = {}  # per-symbol NumPy views


        # Broker meta (min_stop_price per symbol)
        self.broker_meta = load_broker_meta(BROKER_STOPS_JSON)

        # CSV metadata only (read slices on reset)
        self.data_paths: Dict[str, str] = {}
        self.data_lengths: Dict[str, int] = {}
        for sym in self.symbols:
            path = os.path.join(csv_dir, f"{sym}_onemin.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"OneMin CSV not found for '{sym}': {path}")
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
        obs_dim = self.n_assets * (5 * self.window + 1)
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
        self.sl_cooldown = np.zeros(self.n_assets, dtype=np.int32)
        self.dfs: Dict[str, pd.DataFrame] = {}  # chunked data loaded on reset

        risk_budget = max(6.0, 0.75 * self.n_assets)

        self.reward_fn = RewardFunction(
        initial_balance=self.initial_balance,
        slippage_per_unit=SLIPPAGE_PER_UNIT,
        commission_per_trade=COMMISSION_PER_TRADE,
        integrate_costs_in_reward=True,
        price_to_ccy_scale=LOT_MULTIPLIER,

        # NEW: safer R floor; adjust if your price scale differs
        min_risk=5e-4,

        # time pressure ↓ and only when flat
        inactivity_weight=0.00005,
        inactivity_grace_steps=2,

        # holding penalty gentler
        holding_threshold_steps=5,
        holding_penalty_per_step=0.0006,

        # slightly lower so C1 less likely to clip after switching to mean
        realized_R_weight=10.0,

        risk_budget_R=risk_budget,
        overexposure_weight=0.015,
        unrealized_weight=0.03,

        component_clip=3.0,
        final_clip=5.0,
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
        self.arr.clear()
        min_len_after_atr = float("inf")
        self.sl_cooldown.fill(0)

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

            # NumPy caches for fast access in _get_observation/step
            self.arr[sym] = {
                "open":  df["open"].to_numpy(np.float32),
                "high":  df["high"].to_numpy(np.float32),
                "low":   df["low"].to_numpy(np.float32),
                "close": df["close"].to_numpy(np.float32),
                "atr":   df["atr"].to_numpy(np.float32),
            }


        # We loaded from (start_idx - window), so "start" within local slices == window
        self.cursor = self.window

        # After cleaning, clamp per-reset max steps by what's truly aligned across symbols
        self.runtime_max_steps = min(self.max_steps, int(min_len_after_atr) - self.window - 1)
        if self.runtime_max_steps <= 0:
            raise ValueError(
                f"Not enough aligned OneMin bars across symbols after ATR handling "
                f"(min_len={min_len_after_atr}, window={self.window})"
            )

        self.balance = self.initial_balance
        self.last_trade_time.fill(0.0)
        self.open_trades.clear()
        self.closed_trades.clear()
        self.reward_fn.reset()

        obs = self._get_observation(self.cursor)
        # ── NEW: guard against NaN/Inf in observations
        if not np.isfinite(obs).all():
            bad = np.where(~np.isfinite(obs))[0][:10]
            raise RuntimeError(f"NaN/Inf in reset observation at cursor={self.cursor}, first_bad_idxs={bad}")
        return obs.astype(np.float32), {}
    def _get_observation(self, idx: int) -> np.ndarray:
        obs_parts = []
        ws = self.window
        s = idx - ws
        e = idx

        for i, sym in enumerate(self.symbols):
            arr = self.arr[sym]
            opens  = arr["open"][s:e]
            highs  = arr["high"][s:e]
            lows   = arr["low"][s:e]
            closes = arr["close"][s:e]
            atr    = arr["atr"][s:e] 

            if any(len(a) != ws for a in (opens, highs, lows, closes, atr)):
                raise RuntimeError(
                    f"[{sym}] expected window={ws} but got "
                    f"o={len(opens)} h={len(highs)} l={len(lows)} c={len(closes)} (idx={idx})"
                )

            ot = self._get_open_trade(sym)
            if ot is None:
                extra = np.array([0.0], dtype=np.float32)
            else:
                held = float(self.current_step - ot.get("open_step", self.current_step))
                extra = np.array([held if ot["trade_type"] == "long" else -held], dtype=np.float32)

            obs_parts.append(
                np.concatenate([opens, highs, lows, closes, atr, extra], axis=0).astype(np.float32)
            )

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

            # NEW: enforce a minimum hold before manual close is allowed
            held = self.current_step - ot.get("open_step", self.current_step)
            if held < MIN_MANUAL_HOLD_STEPS:
                valid[3] = False  # close long
                valid[4] = False  # close short
                valid[7] = False  # close all
            # 5 (adjust SL) and 6 (adjust TP) remain valid while in a trade
        else:
            # Not in trade -> cannot close/adjust/close-all
            valid[3:8] = False
            if self.sl_cooldown[i] > 0:
                valid[1] = False  # buy
                valid[2] = False  # sell

        masked = arr.copy()
        masked[:8] = np.where(valid, arr[:8], -np.inf)
        return masked


    def step(self, action: np.ndarray):
        idx = self.cursor
        next_idx = idx + 1

        reward = 0.0
        info: Dict[str, Any] = {"symbols": {}, "closed_trades": []}
        illegal_penalty_total = 0.0
        any_illegal_attempt = False
        sl_penalty_total = 0.0
        curr_close_by_sym: Dict[str, float] = {}
        for i, sym in enumerate(self.symbols):
            arr = self.arr[sym]
            o = arr["open"]; h = arr["high"]; l = arr["low"]; c = arr["close"]; a = arr["atr"]
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
            valid = np.ones(8, dtype=bool)
            ot = self._get_open_trade(sym)
            if ot is not None:
                valid[1] = False  # buy
                valid[2] = False  # sell
                if ot["trade_type"] == "long":
                    valid[4] = False  # can't close short
                elif ot["trade_type"] == "short":
                    valid[3] = False  # can't close long

                # Enforce minimum hold before manual close (mirror _mask_illegal_actions)
                held = self.current_step - ot.get("open_step", self.current_step)
                if held < MIN_MANUAL_HOLD_STEPS:
                    valid[3] = False  # close long
                    valid[4] = False  # close short
                    valid[7] = False  # close all
            else:
                valid[3:8] = False  # cannot close/adjust when flat
                # mirror cooldown rule used in _mask_illegal_actions
                if self.sl_cooldown[i] > 0:
                    valid[1] = False  # buy
                    valid[2] = False  # sell
            attempted_illegal = not valid[orig_action]
            if attempted_illegal:
                any_illegal_attempt = True
            # Prices
            # Use ONLY information available in the current observation (idx-1):
            curr_idx   = idx - 1  # last bar included in the observation slice
            if curr_idx < 0:
                curr_idx = 0
            curr_close = float(c[curr_idx])
            atr_val    = float(a[curr_idx])
            curr_close_by_sym[sym] = curr_close
            next_open  = float(o[next_idx])
            next_close = float(c[next_idx])
            next_high  = float(h[next_idx])
            next_low   = float(l[next_idx])

            min_stop_price = float(self.broker_meta.get(sym, {}).get("min_stop_price", 0.0))

            sl_norm = float(act[8])
            tp_norm = float(act[9])

            trade_executed = False

            # --- Pre-check: SL/TP for an existing open trade on the next bar ---
            ot = self._get_open_trade(sym)
            if ot is not None:
                long_side = (ot["trade_type"] == "long")
                sl = float(ot["stop_loss"]); tp = float(ot["take_profit"]); entry = float(ot["entry_price"])

                hit_sl = (next_low <= sl) if long_side else (next_high >= sl)
                hit_tp = (next_high >= tp) if long_side else (next_low <= tp)

                if hit_sl and hit_tp:
                    # first-touch proxy by proximity to next_open
                    sl_dist = abs(next_open - sl)
                    tp_dist = abs(next_open - tp)
                    prefer_sl = sl_dist <= tp_dist
                    hit_sl, hit_tp = prefer_sl, (not prefer_sl)

                if hit_sl or hit_tp:
                    exit_px = sl if hit_sl else tp
                    pnl = (exit_px - entry) if long_side else (entry - exit_px)
                    stop_type = "sl" if hit_sl else "tp"

                    # NEW: cooldown + penalty on SL (extra penalty if stopped out very fast)
                    if stop_type == "sl":
                        self.sl_cooldown[i] = SL_COOLDOWN_STEPS
                        sl_penalty_total += SL_ILLEGAL_PENALTY
                        held = self.current_step - ot.get("open_step", self.current_step)
                        if held <= SL_EARLY_STEPS:
                            sl_penalty_total += SL_ILLEGAL_PENALTY

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

            # --- Action: only if nothing executed yet ---
            # --- Post-check: if still open now, allow SL/TP hits — but NOT for trades opened this step ---
            # --- Action: only if nothing executed yet ---
            if not trade_executed:
                ot = self._get_open_trade(sym)  # refresh

                if act_id == 1 and ot is None:  # buy
                    entry = next_open
                    sl_final, tp_final = _scale_sl_tp(
                        entry=entry, sl_norm=sl_norm, tp_norm=tp_norm,
                        is_long=True, min_stop_price=min_stop_price, atr_value=atr_val,
                    )
                    self._open_trade(sym, "long", entry, sl_final, tp_final)
                    trade_executed = True

                elif act_id == 2 and ot is None:  # sell
                    entry = next_open
                    sl_final, tp_final = _scale_sl_tp(
                        entry=entry, sl_norm=sl_norm, tp_norm=tp_norm,
                        is_long=False, min_stop_price=min_stop_price, atr_value=atr_val,
                    )
                    self._open_trade(sym, "short", entry, sl_final, tp_final)
                    trade_executed = True

                elif act_id in (3, 4, 7) and ot is not None:
                    # Manual close at NEXT OPEN (only if not already SL/TP-closed above)
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
                        entry=ref_price, sl_norm=sl_norm, tp_norm=0.0,
                        is_long=is_long, min_stop_price=min_stop_price, atr_value=atr_val,
                    )
                    floor_price = max(float(min_stop_price or 0.0), 0.40 * atr_val)
                    old_sl = float(ot["stop_loss"])
                    if is_long:
                        upper_bound = min(ot["take_profit"] - floor_price, ref_price - floor_price)
                        candidate   = min(new_sl, upper_bound)
                        new_sl      = max(old_sl, candidate)  # tighten-only
                    else:
                        lower_bound = max(ot["take_profit"] + floor_price, ref_price + floor_price)
                        candidate   = max(new_sl, lower_bound)
                        new_sl      = min(old_sl, candidate)  # tighten-only
                    ot["stop_loss"] = float(new_sl)
                    info["symbols"][sym] = {**info["symbols"].get(sym, {}), "adjusted": "sl"}

                elif act_id == 6 and ot is not None:
                    # Adjust Take-Profit using the LAST OBSERVED close as reference (curr_close)
                    ref_price = curr_close
                    is_long = (ot["trade_type"] == "long")
                    _unused, new_tp = _scale_sl_tp(
                        entry=ref_price, sl_norm=0.0, tp_norm=tp_norm,
                        is_long=is_long, min_stop_price=min_stop_price, atr_value=atr_val,
                    )
                    floor_price = max(float(min_stop_price or 0.0), 0.40 * atr_val)
                    if is_long:
                        new_tp = max(new_tp, ot["stop_loss"] + floor_price, ref_price + floor_price)
                    else:
                        new_tp = min(new_tp, ot["stop_loss"] - floor_price, ref_price - floor_price)
                    ot["take_profit"] = float(new_tp)
                    info["symbols"][sym] = {**info["symbols"].get(sym, {}), "adjusted": "tp"}

            # --- Post-check: if still open now, allow SL/TP hits — but NOT for trades opened this step ---
            if not trade_executed:
                ot = self._get_open_trade(sym)
                if ot is not None and ot.get("open_step", -1) != self.current_step:
                    long_side = (ot["trade_type"] == "long")
                    sl = float(ot["stop_loss"]); tp = float(ot["take_profit"]); entry = float(ot["entry_price"])

                    hit_sl = (next_low <= sl) if long_side else (next_high >= sl)
                    hit_tp = (next_high >= tp) if long_side else (next_low <= tp)

                    if hit_sl and hit_tp:
                        sl_dist = abs(next_open - sl)
                        tp_dist = abs(next_open - tp)
                        prefer_sl = sl_dist <= tp_dist
                        hit_sl, hit_tp = prefer_sl, (not prefer_sl)

                    if hit_sl or hit_tp:
                        exit_px = sl if hit_sl else tp
                        pnl = (exit_px - entry) if long_side else (entry - exit_px)
                        stop_type = "sl" if hit_sl else "tp"

                        # NEW: cooldown + penalty on SL (extra penalty if stopped out very fast)
                        if stop_type == "sl":
                            self.sl_cooldown[i] = SL_COOLDOWN_STEPS
                            sl_penalty_total += SL_ILLEGAL_PENALTY
                            held = self.current_step - ot.get("open_step", self.current_step)
                            if held <= SL_EARLY_STEPS:
                                sl_penalty_total += SL_ILLEGAL_PENALTY

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
        global_since_last_trade = float(np.mean(self.last_trade_time)) if len(self.last_trade_time) else 0.0

        # include holding_time for open trades so C5 works
        open_trades_for_reward = []
        
        for t in self.open_trades:
            td = dict(t)
            td["holding_time"] = self.current_step - td.get("open_step", self.current_step)
            open_trades_for_reward.append(td)

        # Compute unrealized PnL using last observed closes (curr_idx), no leakage
        unrealized_pnl = 0.0
        for t in self.open_trades:
            sym = t["symbol"]
            cc = curr_close_by_sym.get(sym)
            if cc is None:
                ci = max(idx - 1, 0)
                cc = float(self.arr[sym]["close"][ci])
            entry = float(t["entry_price"])
            if t["trade_type"] == "long":
                unrealized_pnl += (cc - entry)
            else:
                unrealized_pnl += (entry - cc)

        reward = float(self.reward_fn(
            closed_trades=info["closed_trades"],      # may be []
            open_trades=open_trades_for_reward,       # includes holding_time
            account_balance=self.balance,
            unrealized_pnl=float(unrealized_pnl),
            time_since_last_trade=global_since_last_trade
        ).item())
        # ── NEW: never let reward be NaN/Inf
        if not np.isfinite(reward):
            reward = 0.0


        info["reward_components"] = getattr(self.reward_fn, "last_components", None)
        # Stop-type counts for this step
        info["c_sl"] = sum(1 for ct in info["closed_trades"] if ct.get("stop_type") == "sl")
        info["c_tp"] = sum(1 for ct in info["closed_trades"] if ct.get("stop_type") == "tp")
        info["c_manual"] = sum(1 for ct in info["closed_trades"] if ct.get("stop_type") == "manual")

        # ── Diagnostics to log via Monitor (flatten keys) ─────────────────────────────
        rc = getattr(self.reward_fn, "last_components", {}) or {}
        for k in (
        "C1_realizedR", "C2_quality", "C3_unreal", "C4_inactivity",
        "C5_holding", "C6_overexp", "C7_conflict", "C8_churnSLTP",
        "realized_R_mean",  # NEW: per-step mean R of closes
        "total_before_clip",
        ):

            v = rc.get(k, None)
            if v is not None:
                info[f"rw_{k}"] = float(v)

        info["n_open"] = int(len(self.open_trades))
        info["n_closed"] = int(len(info["closed_trades"]))
        info["illegal_attempts"] = int(any_illegal_attempt)
        info["since_last_trade"] = float(global_since_last_trade)
        reward += sl_penalty_total
        # ──────────────────────────────────────────────────────────────────────────────
        # keep your existing illegal-action penalties
        if any_illegal_attempt:
            reward += ILLEGAL_ATTEMPT_PENALTY

        # also add the severe penalty you accumulated
        reward += float(illegal_penalty_total)

        # final clip AFTER penalties, same as 1-min
        final_cap = getattr(self.reward_fn, "final_clip", 5.0) or 5.0
        reward = float(np.clip(reward, -final_cap, final_cap))
        # Keep only recent closed trades
        MAX_CLOSED_TRADES = 1000
        if len(self.closed_trades) > MAX_CLOSED_TRADES:
            self.closed_trades = self.closed_trades[-MAX_CLOSED_TRADES:]
        
        self.sl_cooldown[self.sl_cooldown > 0] -= 1

        self.current_step += 1
        terminated = self.current_step >= self.runtime_max_steps
        truncated = False

        if not (terminated or truncated):
            self.cursor += 1
            obs = self._get_observation(self.cursor).astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # ── NEW: guard against NaN/Inf in observations
        if not np.isfinite(obs).all():
            raise RuntimeError(
                f"NaN/Inf in observation at step={self.current_step}, idx={self.cursor}, symbols={self.symbols}"
            )
        return obs, float(reward), terminated, truncated, info

    def _open_trade(self, sym: str, direction: str, entry: float, sl: float, tp: float):
        self.open_trades = [t for t in self.open_trades if t["symbol"] != sym]  # ensure at most one
        risk_at_open = abs(float(entry) - float(sl))
        trade = dict(
            symbol=sym,
            trade_type=direction,
            entry_price=float(entry),
            stop_loss=float(sl),
            take_profit=float(tp),
            volume=1.0,
            open_step=self.current_step,
            risk_at_open=float(risk_at_open),
        )
        self.open_trades.append(trade)
        # Pay costs upfront
        # Pay costs to balance only if NOT integrated into reward
        if not getattr(self.reward_fn, "integrate_costs_in_reward", False):
            cost = _to_float(self.reward_fn.slippage_per_unit) + _to_float(self.reward_fn.commission_per_trade)
            self.balance -= cost


    def render(self, mode="human"):
        idx = self.cursor
        prices = {sym: float(self.arr[sym]["close"][idx]) for sym in self.symbols}
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
        env = Monitor(
        env,
        filename=os.path.join(log_dir, "monitor.csv"),
        info_keywords=(
            "n_open", "n_closed", "illegal_attempts", "since_last_trade",
            "rw_C1_realizedR", "rw_C2_quality", "rw_C3_unreal", "rw_C4_inactivity",
            "rw_C5_holding", "rw_C6_overexp", "rw_C7_conflict", "rw_C8_churnSLTP",
            "rw_realized_R_mean",              # NEW
            "rw_total_before_clip",
            "c_sl", "c_tp", "c_manual",        # NEW
             ),
             )


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
    logger.info("Starting Onemin-term training with PPO and event-based reward.")
    # Clean stale monitor files so new rw_* columns are written in the header
    try:
        for i in range(n_envs):
            _log_dir = os.path.join(LOGS_DIR, f"onemin_worker_{i}")
            os.makedirs(_log_dir, exist_ok=True)
            mon_csv = os.path.join(_log_dir, "monitor.csv")
            if os.path.exists(mon_csv):
                os.remove(mon_csv)
                print(f"[Monitor] Removed stale {mon_csv} to refresh headers.")
    except Exception as e:
        print(f"[Monitor] Cleanup skipped: {e}")
    # VecEnv
    # VecEnv (+ VecNormalize load/resume)
    env_fns = [make_onemin_env(i, SEED, window) for i in range(n_envs)]
    base_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv([env_fns[0]])

    pkl_path = os.path.join(MODELS_DIR, "checkpoints_onemin", "vecnormalize.pkl")
    # ── NEW: force fresh stats to avoid NaNs from stale/mismatched running means/vars
    if os.path.exists(pkl_path):
        try:
            os.remove(pkl_path)
            print(f"[VecNormalize] Deleted stale stats at {pkl_path}")
        except Exception as e:
            print(f"[VecNormalize] Could not delete {pkl_path}: {e}")

    vec_env = VecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=True,
        gamma=0.995,
        clip_obs=10.0,               # ── NEW: clip obs to keep them sane
        clip_reward=float("inf"),
    )
    n_steps = 1024
    rollout = n_steps * n_envs
    for cand in (1024, 512, 256, 128, 64):
        if rollout % cand == 0:
            batch_size = cand
            break
    print(f"[train_onemin] n_steps={n_steps} n_envs={n_envs} batch_size={batch_size} rollout={rollout}")
    check_every = rollout
    
    assert (n_steps * n_envs) % batch_size == 0, "n_steps * n_envs must be divisible by batch_size."

    algo_cls = PPO
    policy_cls = OneMinOHLCPolicy
    algo_kwargs = dict(
    n_steps=n_steps,
    batch_size=batch_size,
    learning_rate=1e-4,        # ── NEW: smaller LR = fewer spikes
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=os.path.join(LOGS_DIR, "tb_onemin_policy"),
    device="cuda",
    n_epochs=10,               # ── NEW: fewer epochs, less overfitting per batch
    target_kl=0.2,             # ── NEW: early-stop updates if policy drifts too far
    )

    policy_kwargs = dict(
    window=window,
    embed_dim=128,
    tcn_hidden=128,
    n_heads= 8,
    n_layers= 4,
    )

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
        check_freq=check_every,           # align with rollout
        rclone_dest=rclone_dest,
        verbose=1,
    )
    early_stopping_callback = EarlyStoppingCallback(
    check_freq=early_stopping_check_freq or check_every,
    patience=patience,
    verbose=1,
    )
    comp_logger_callback = LogRewardComponentsCallback(section="rollout", verbose=0)
    log_std_callback = LogStdCallback()
    callbacks = [comp_logger_callback, log_std_callback, best_model_callback, early_stopping_callback]

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
        if ASYNC_UPLOADER:
            ASYNC_UPLOADER.wait()  # drain queue at the very end only
        vec_env.close()
if __name__ == "__main__":
    train_onemin_policy()
