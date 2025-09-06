# train_long_policy.py
from __future__ import annotations
import os
import glob
import random
import logging
import re
import json
import torch
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

# ── LOGGER SETUP START (add near imports) ─────────────────────────────────────
LOG_LEVEL = os.getenv("TRAIN_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("train_long_policy")
# ── LOGGER SETUP END ──────────────────────────────────────────────────────────

# Policies / extractor (the recurrent policy may or may not exist in your repo)
from models.long_policy import (
    LongTermOHLCPolicy,
)

from config.settings import (
    DAILY_CSV_DIR,
    LIVE_FOREX_PAIRS,
    MODELS_DIR,
    SEED,
    SLIPPAGE_PER_UNIT,
    COMMISSION_PER_TRADE,
    INITIAL_BALANCE,
    LONG_OBS_WINDOW,
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
ILLEGAL_ATTEMPT_PENALTY = -0.005
MIN_MANUAL_HOLD_STEPS = 2
SL_ILLEGAL_PENALTY   = -0.02
SL_COOLDOWN_STEPS    = 3
SL_EARLY_STEPS       = 3
MAX_RISK_FRAC = float(os.getenv("MAX_RISK_FRAC", "0.25"))
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
            log.debug("Checkpoint upload queue full; skipping %s", os.path.basename(file_path))
            return False

    def _worker(self):
        while True:
            path = self.q.get()
            if path is None:  # shutdown
                self.q.task_done()
                break
            try:
                if not os.path.exists(path):
                    log.debug("Checkpoint file vanished, skip: %s", path)
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
                    log.info("Checkpoint enqueued: %s -> %s", os.path.basename(path), self.dest)
            except Exception as e:
                log.exception("AsyncUploader error")
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
    logging.warning(f"[train_long] broker_stops.json not found at {json_path}, using zeros.")
    return {}

def compute_atr(df: pd.DataFrame, n: int = 14, mode: str = "rolling") -> pd.Series:
    """
    Robust ATR (daily):
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
    FLOOR_FRAC_ATR = 0.15
    K_SL_ATR = 0.8
    K_TP_ATR = 3.0

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
                 name_prefix: str = "long_policy_ckpt",
                 rclone_dest: str = "", verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_freq = int(checkpoint_freq)
        self.ckpt_dir = ckpt_dir
        self.name_prefix = name_prefix
        self.rclone_dest = rclone_dest or os.getenv("RCLONE_DEST", "")
        if not self.rclone_dest:
            log.debug("RCLONE_DEST not set; saving locally only.")
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
        log.info("Checkpoint wrote %s exists=%s", fzip, exists)
        if not exists:
            raise RuntimeError(f"Checkpoint file missing after save: {fzip}")

        # Only push if a remote is configured
        if self.rclone_dest and ASYNC_UPLOADER:
            ASYNC_UPLOADER.submit(fzip)  # returns immediately
        else:
            log.debug("No RCLONE_DEST set; skipped remote upload.")
        return True
# ── REPLACEMENT START (SaveTrainBestByEpBuffer + EarlyStopping) ───────────────
class SaveTrainBestByEpBuffer(BaseCallback):
    """
    Save 'training best' exactly by SB3's ep_info_buffer (same as ep_rew_mean).
    Falls back to monitor.csv only if the buffer is empty.
    """
    def __init__(self, save_path: str, check_freq: int, rclone_dest: str = "", verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = int(check_freq)
        self.best_mean = -np.inf
        self.rclone_dest = rclone_dest or os.getenv("RCLONE_DEST", "")

    def _mean_from_ep_buffer(self) -> Optional[float]:
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf and len(buf) > 0:
            try:
                return float(np.mean([e["r"] for e in buf]))
            except Exception:
                return None
        return None

    def _on_step(self) -> bool:
        if (int(self.model.num_timesteps) % self.check_freq) != 0:
            return True

        # ── REPLACEMENT inside _on_step (both callbacks) ──────────────────────────────
        mean_r = self._mean_from_ep_buffer()
        if mean_r is None:
            return True  # no disk fallback
        # ── END REPLACEMENT ───────────────────────────────────────────────────────────

        if self.verbose:
            log.debug("TrainBest mean=%.3f best=%.3f", mean_r, self.best_mean)

        if mean_r > self.best_mean + 1e-4:
            self.best_mean = mean_r
            self.model.save(self.save_path)
            best_zip = self.save_path if self.save_path.endswith(".zip") else self.save_path + ".zip"
            log.info("TrainBest new best: %s", best_zip)
            if self.rclone_dest and ASYNC_UPLOADER:
                ASYNC_UPLOADER.submit(best_zip)
        return True

class _NoOpCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

class EarlyStoppingCallback(EventCallback):
    """
    Patience-based early stopping using the SAME metric as SB3 ep_rew_mean,
    with fallback to monitor.csv when the buffer is empty.
    """
    def __init__(self, check_freq: int, patience: int, verbose=1):
        super().__init__(callback=_NoOpCallback(), verbose=verbose)
        self.check_freq = int(check_freq)
        self.patience = int(patience)
        self.best_mean_reward = -np.inf
        self.counter = 0
        self.verbose = verbose

    def _mean_from_ep_buffer(self) -> Optional[float]:
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf and len(buf) > 0:
            try:
                return float(np.mean([e["r"] for e in buf]))
            except Exception:
                return None
        return None

    def _on_step(self) -> bool:
        if int(self.model.num_timesteps) % self.check_freq != 0:
            return True

        mean_reward = self._mean_from_ep_buffer()
        if mean_reward is None:
            return True
        log.debug(
        "EarlyStopping step=%s mean=%.3f best=%.3f counter=%d/%d",
        getattr(self, "num_timesteps", -1), mean_reward, self.best_mean_reward,
        self.counter, self.patience
        )
        if mean_reward > self.best_mean_reward + 1e-4:
            self.best_mean_reward = mean_reward
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    log.info("EarlyStopping: patience exceeded, stopping training.")
                return False
        return True

def build_eval_env(make_env_fn, n_envs: int, vecnorm_stats_path: str) -> VecNormalize:
    """
    Build a frozen-eval VecNormalize env that shares preprocessing
    but does NOT update running stats.
    """
    envs = [make_env_fn(i, SEED + 10_000 + i, window=LONG_OBS_WINDOW) for i in range(n_envs)]
    base = DummyVecEnv(envs)
    if os.path.exists(vecnorm_stats_path):
        eval_vec = VecNormalize.load(vecnorm_stats_path, venv=base)  # or: VecNormalize.load(vecnorm_stats_path, base)
    else:
        eval_vec = VecNormalize(base, norm_obs=True, norm_reward=True, gamma=0.995, clip_obs=10.0, clip_reward=float("inf"))

    eval_vec.training = False
    eval_vec.norm_reward = False  # report raw env reward
    return eval_vec

@torch.no_grad()
def evaluate_model(model, eval_env: VecNormalize, episodes: int = 20, deterministic: bool = True):
    """
    Vectorized, deterministic evaluation:
      - Accumulates per-env episodic returns/lengths
      - Stops when total completed episodes >= `episodes`
      - Tracks illegal attempts across infos
    """
    n = getattr(eval_env, "num_envs", 1)
    _reset = eval_env.reset()
    obs = _reset[0] if isinstance(_reset, tuple) else _reset
    ep_ret = np.zeros(n, dtype=np.float64)
    ep_len = np.zeros(n, dtype=np.int32)
    ep_returns: List[float] = []
    ep_lens: List[int] = []
    illegal = 0
    steps = 0
    done_count = 0

    while done_count < episodes:
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(actions)
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        ep_ret += r
        ep_len += 1
        steps += n

        # count illegal attempts from info dicts
        if isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict):
                    illegal += int(info.get("illegal_attempts", 0))

        # handle per-env episode completion
        dones = np.asarray(dones).reshape(-1)
        for i in range(n):
            if dones[i]:
                ep_returns.append(float(ep_ret[i]))
                ep_lens.append(int(ep_len[i]))
                done_count += 1
                # auto-reset occurs inside Dummy/SubprocVecEnv; counters must reset too
                ep_ret[i] = 0.0
                ep_len[i] = 0
                if done_count >= episodes:
                    break

    mean_ret = float(np.mean(ep_returns))
    std_ret = float(np.std(ep_returns) + 1e-8)
    # downside deviation proxy
    downside = np.std([min(0.0, r - mean_ret) for r in ep_returns]) + 1e-8
    sortino = mean_ret / downside

    # crude max drawdown proxy from episodic return set (replace with equity MDD if available)
    max_dd = float(max(0.0, -min(ep_returns))) if ep_returns else 0.0
    illegal_rate = float(illegal) / max(1, steps)

    return {
        "mean_reward": mean_ret,
        "std_reward": std_ret,
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "illegal_rate": float(illegal_rate),
        "episodes": int(episodes),
    }

def eval_score(metrics: dict) -> float:
    # Tune to taste.
    return (metrics["mean_reward"]
            - 5.0 * metrics["max_drawdown"]
            - 0.5 * metrics["illegal_rate"]
            + 0.2 * metrics["sortino"])

class EvalAndSaveDeployBestCallback(BaseCallback):
    """
    Periodically run a frozen eval on held-out seeds/time and save deploy best by composite score.
    """
    def __init__(self, vecnorm_stats_path: str, make_env_fn, n_eval_envs: int,
                 episodes: int, check_freq: int, save_path: str, rclone_dest: str = "", verbose: int = 1):
        super().__init__(verbose)
        self.vecnorm_stats_path = vecnorm_stats_path
        self.make_env_fn = make_env_fn
        self.n_eval_envs = int(n_eval_envs)
        self.episodes = int(episodes)
        self.check_freq = int(check_freq)
        self.save_path = save_path
        self.best_score = -np.inf
        self.rclone_dest = rclone_dest or os.getenv("RCLONE_DEST", "")

    def _on_step(self) -> bool:
        if (int(self.model.num_timesteps) % self.check_freq) != 0:
            return True

        eval_env = build_eval_env(self.make_env_fn, self.n_eval_envs, self.vecnorm_stats_path)
        try:
            metrics = evaluate_model(self.model, eval_env, episodes=self.episodes, deterministic=True)
        finally:
            eval_env.close()

        score = eval_score(metrics)
        log.debug(
        "DeployEval score=%.3f mean=%.3f sortino=%.3f mdd=%.3f ill=%.4f",
        score, metrics["mean_reward"], metrics["sortino"],
        metrics["max_drawdown"], metrics["illegal_rate"]
        )
        self.model.logger.record("eval/score", score)
        for k, v in metrics.items():
            self.model.logger.record(f"eval/{k}", v)

        if score > self.best_score + 1e-4:
            self.best_score = score
            self.model.save(self.save_path)
            best_zip = self.save_path if self.save_path.endswith(".zip") else self.save_path + ".zip"
            log.info("DeployEval new best: %s", best_zip)
            if self.rclone_dest and ASYNC_UPLOADER:
                ASYNC_UPLOADER.submit(best_zip)
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
            "rw_realized_R_mean",
            "rw_total_before_clip",
            "n_open", "n_closed", "illegal_attempts", "since_last_trade",
            "c_sl", "c_tp", "c_manual",
            "risk_veto_count",           # NEW
            "explore_bonus",             # NEW
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
# Environment (daily) — fixed indexing + ATR handling
# ──────────────────────────────────────────────────────────────────────────────
class LongBacktestEnv(gym.Env):
    """
    Daily OHLC backtest with:
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
        self.sl_cooldown = np.zeros(self.n_assets, dtype=np.int32)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        self.arr: Dict[str, Dict[str, np.ndarray]] = {}  # per-symbol NumPy views


        # Broker meta (min_stop_price per symbol)
        self.broker_meta = load_broker_meta(BROKER_STOPS_JSON)

        # CSV metadata only (read slices on reset)
        self.data_paths: Dict[str, str] = {}
        self.data_lengths: Dict[str, int] = {}
        for sym in self.symbols:
            path = os.path.join(csv_dir, f"{sym}_daily.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Daily CSV not found for '{sym}': {path}")
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
        # ── PRELOAD START (place in __init__, after self.max_length is set) ───────────
        self.arr_preloaded: Dict[str, Dict[str, np.ndarray]] = {}
        self.valid_lengths_after_atr: Dict[str, int] = {}

        for sym in self.symbols:
            path = self.data_paths[sym]
            # One-time load for the whole CSV
            df = pd.read_csv(path, parse_dates=["time"])
            df = _safe_numeric(df)

            # One-time ATR computation (rolling with fallback)
            atr_roll = compute_atr(df, n=14, mode="rolling")

            if atr_roll.notna().sum() >= self.window + 2:
                df = df.assign(atr=atr_roll).dropna(subset=["atr"]).reset_index(drop=True)
            else:
                atr_wilder = compute_atr(df, n=14, mode="wilder")
                df = df.assign(atr=atr_wilder).reset_index(drop=True)

            df["atr"] = df["atr"].replace([np.inf, -np.inf], np.nan).clip(lower=1e-12)

            self.arr_preloaded[sym] = {
                "open":  df["open"].to_numpy(np.float32),
                "high":  df["high"].to_numpy(np.float32),
                "low":   df["low"].to_numpy(np.float32),
                "close": df["close"].to_numpy(np.float32),
                "atr":   df["atr"].to_numpy(np.float32),
            }
            self.valid_lengths_after_atr[sym] = len(df)

        # Use the preloaded arrays for obs/step
        self.arr = self.arr_preloaded
        # Align max_length to the shortest preloaded series
        self.max_length = min(self.valid_lengths_after_atr.values())
        # ── PRELOAD END ───────────────────────────────────────────────────────────────

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
        self.max_total_open = int(np.ceil(0.35 * self.n_assets))
        self.closed_trades: List[Dict[str, Any]] = [] # rolling buffer

        self.dfs: Dict[str, pd.DataFrame] = {}  # chunked data loaded on reset

        self.reward_fn = RewardFunction(
            initial_balance=self.initial_balance,
            slippage_per_unit=SLIPPAGE_PER_UNIT,
            commission_per_trade=COMMISSION_PER_TRADE,
            integrate_costs_in_reward=True,
            price_to_ccy_scale=LOT_MULTIPLIER,

            min_risk=5e-4,

            inactivity_weight=0.015,
            inactivity_grace_steps=5,

            holding_threshold_steps=8,
            holding_penalty_per_step=0.0004,

            realized_R_weight=3.0, 
            quality_weight=1.0,
            unrealized_weight=0.3,

            risk_budget_R=max(4.0, 0.25 * self.n_assets),
            overexposure_weight=0.10,

            component_clip=4.0,
            final_clip=6.0,
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

        # episode counters
        self.current_step = 0
        self.balance = self.initial_balance
        self.last_trade_time.fill(0.0)
        self.open_trades.clear()
        self.closed_trades.clear()
        self.reward_fn.reset()

        # choose a random start that leaves enough room for window + steps
        min_len_after_atr = min(self.valid_lengths_after_atr[sym] for sym in self.symbols)
        # max start so that: start + window + max_steps + 1 <= min_len
        max_start = min_len_after_atr - self.window - 1 - self.max_steps
        max_start = max(max_start, self.window)
        self.start_idx = random.randint(self.window, max_start)

        # base offset into the preloaded arrays
        self.base = int(self.start_idx)

        # cursor always starts at 'window' within the local episode slice
        self.cursor = self.window

        # clamp per-episode steps so we never index out of bounds
        self.runtime_max_steps = min(self.max_steps, min_len_after_atr - self.base - self.window - 1)
        if self.runtime_max_steps <= 0:
            raise ValueError(
                f"Not enough aligned Daily bars across symbols after ATR handling "
                f"(min_len={min_len_after_atr}, window={self.window})"
            )

        # first observation
        obs = self._get_observation(self.cursor)
        if not np.isfinite(obs).all():
            bad = np.where(~np.isfinite(obs))[0][:10]
            raise RuntimeError(f"NaN/Inf in reset observation at cursor={self.cursor}, first_bad_idxs={bad}")
        return obs.astype(np.float32), {}

    def _equity_estimate(self, idx_local: int) -> float:
        """
        Conservative equity estimate at this step using last observed closes
        (no look-ahead): balance + unrealized PnL marked to previous close.
        """
        ci = max(self.base + idx_local - 1, 0)
        unreal = 0.0
        for t in self.open_trades:
            sym = t["symbol"]
            arr = self.arr[sym]
            price_now = float(arr["close"][ci])
            entry = float(t["entry_price"])
            if t["trade_type"] == "long":
                unreal += (price_now - entry)
            else:
                unreal += (entry - price_now)
        return float(self.balance + unreal)
    def set_inactivity_weight(self, w: float):
        try:
            self.reward_fn.inactivity_weight = float(w)
        except Exception:
            pass

    def _risk_usage(self) -> float:
        """
        Current worst-case risk across open trades, valued in account currency
        via LOT_MULTIPLIER (same unit you use in RewardFunction).
        """
        tot = 0.0
        for t in self.open_trades:
            tot += abs(float(t["entry_price"]) - float(t["stop_loss"])) * float(LOT_MULTIPLIER)
        return float(tot)

    def _risk_of(self, entry: float, sl: float) -> float:
        """Worst-case risk for a candidate trade (unit volume)."""
        return float(abs(float(entry) - float(sl)) * float(LOT_MULTIPLIER))


    def _get_observation(self, idx: int) -> np.ndarray:
        obs_parts = []
        ws = self.window
        s = self.base + idx - ws
        e = self.base + idx

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
                    f"o={len(opens)} h={len(highs)} l={len(lows)} c={len(closes)} atr={len(atr)} (idx={idx})"
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
            # Global cap on concurrent positions: when cap reached, forbid new entries
            if len(self.open_trades) >= self.max_total_open:
                valid[1] = False  # buy
                valid[2] = False  # sell
                # Block new entries on this symbol while in SL cooldown
        if self.sl_cooldown[i] > 0:
            valid[1] = False  # buy
            valid[2] = False  # sell
        masked = arr.copy()
        masked[:8] = np.where(valid, arr[:8], -np.inf)
        return masked

    def step(self, action: np.ndarray):
        sl_penalty_total = 0.0
        idx = self.cursor
        reward = 0.0
        info: Dict[str, Any] = {"symbols": {}, "closed_trades": []}
        illegal_penalty_total = 0.0
        any_illegal_attempt = False
        curr_close_by_sym: Dict[str, float] = {}
        explore_bonus = 0.0 
        equity_est = self._equity_estimate(idx)
        risk_cap   = float(MAX_RISK_FRAC) * max(1.0, float(equity_est))  # avoid tiny/zero edge
        curr_risk  = self._risk_usage()
        for i, sym in enumerate(self.symbols):
            arr = self.arr[sym]
            o = arr["open"]; h = arr["high"]; l = arr["low"]; c = arr["close"]; a = arr["atr"]
            act = action[i * 10: (i + 1) * 10]

            # Safety net: never allow >1 open trade per symbol.
            open_for_sym = [t for t in self.open_trades if t["symbol"] == sym]
            if len(open_for_sym) > 1:
                log.error("[%s] Multiple open trades detected; auto-closing extras.", sym)
                illegal_penalty_total += SEVERE_ILLEGAL_ACTION_PENALTY
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
                keep = open_for_sym[0]
                self.open_trades = [t for t in self.open_trades if t["symbol"] != sym] + [keep]

            # (A) Action head the policy WANTED (before masking) — for penalty accounting
            raw_head = np.where(np.isfinite(act[:8]), act[:8], -np.inf)
            orig_action = int(np.argmax(raw_head))

            # (B) Mask invalid heads before argmax so sampled==executed
            masked = self._mask_illegal_actions(i, act)  # sets invalid heads to -inf
            masked_head = masked[:8]
            act_id = int(np.argmax(masked_head))

            # recompute local validity for "attempted_illegal" flag
            valid = np.ones(8, dtype=bool)
            ot = self._get_open_trade(sym)
            if ot is not None:
                valid[1] = False  # buy
                valid[2] = False  # sell
                if ot["trade_type"] == "long":
                    valid[4] = False  # can't close short
                else:
                    valid[3] = False  # can't close long
                held = self.current_step - ot.get("open_step", self.current_step)
                if held < MIN_MANUAL_HOLD_STEPS:
                    valid[3] = False
                    valid[4] = False
                    valid[7] = False
            else:
                valid[3:8] = False
                if len(self.open_trades) >= self.max_total_open:
                    valid[1] = False
                    valid[2] = False
            attempted_illegal = not valid[orig_action]
            if attempted_illegal:
                any_illegal_attempt = True

            # Indexing (episode-local → absolute)
            next_idx = self.base + idx
            curr_idx = max(self.base + idx - 1, 0)

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

                        # --- Simple auto-breakeven trailing to cut losers early ---
            ot = self._get_open_trade(sym)
            if ot is not None:
                long_side = (ot["trade_type"] == "long")
                entry = float(ot["entry_price"])
                old_sl = float(ot["stop_loss"])
                held_bars = self.current_step - ot.get("open_step", self.current_step)

                # Only consider after a few bars, and if move >= 0.75 * ATR in our favor
                if held_bars >= SL_EARLY_STEPS and np.isfinite(atr_val) and atr_val > 0.0:
                    moved_ok = (curr_close - entry) >= 0.75 * atr_val if long_side else (entry - curr_close) >= 0.75 * atr_val
                    if moved_ok:
                        be = entry  # breakeven
                        if long_side and be > old_sl:
                            ot["stop_loss"] = float(be)
                        elif (not long_side) and be < old_sl:
                            ot["stop_loss"] = float(be)



            # --- Pre-check: SL/TP for an existing open trade on the next bar ---
            ot = self._get_open_trade(sym)
            if ot is not None:
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

                    if stop_type == "sl":
                        self.sl_cooldown[i] = SL_COOLDOWN_STEPS
                        sl_penalty_total += SL_ILLEGAL_PENALTY
                        held_bars = self.current_step - ot.get("open_step", self.current_step)
                        if held_bars <= SL_EARLY_STEPS:
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
                    idle = self.last_trade_time[i]
                    if idle >= 5:
                        # scale bonus with idle time, capped; helps break local minima of not trading
                        explore_bonus += min(0.15, 0.01 * idle)

            # --- Action: only if nothing executed yet ---
            if not trade_executed:
                ot = self._get_open_trade(sym)  # refresh

                if act_id == 1 and ot is None:  # buy
                    entry = next_open
                    sl_final, tp_final = _scale_sl_tp(
                        entry=entry, sl_norm=sl_norm, tp_norm=tp_norm,
                        is_long=True, min_stop_price=min_stop_price, atr_value=atr_val,
                    )
                    add_risk = self._risk_of(entry, sl_final)
                    if (curr_risk + add_risk) <= risk_cap:
                        self._open_trade(sym, "long", entry, sl_final, tp_final)
                        curr_risk += add_risk
                        trade_executed = True
                        # Reward breaking long inactivity on this symbol
                        if self.last_trade_time[i] >= 10:  # ~50 daily bars idle
                            explore_bonus += 0.05

                    else:
                        info["symbols"][sym] = {**info["symbols"].get(sym, {}), "risk_veto": True}
                elif act_id == 2 and ot is None:  # sell
                    entry = next_open
                    sl_final, tp_final = _scale_sl_tp(
                        entry=entry, sl_norm=sl_norm, tp_norm=tp_norm,
                        is_long=False, min_stop_price=min_stop_price, atr_value=atr_val,
                    )
                    add_risk = self._risk_of(entry, sl_final)
                    if (curr_risk + add_risk) <= risk_cap:
                        self._open_trade(sym, "short", entry, sl_final, tp_final)
                        curr_risk += add_risk
                        trade_executed = True
                        # Reward breaking long inactivity on this symbol
                        if self.last_trade_time[i] >= 10:  # ~50 daily bars idle
                            explore_bonus += 0.05
                    else:
                        info["symbols"][sym] = {**info["symbols"].get(sym, {}), "risk_veto": True}
                elif act_id in (3, 4, 7) and ot is not None:
                    exit_px = next_open
                    pnl = (exit_px - ot["entry_price"]) if ot["trade_type"] == "long" else (ot["entry_price"] - exit_px)
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
                    # Reward breaking long inactivity on this symbol
                    if self.last_trade_time[i] >=10:  # ~50 daily bars idle
                        explore_bonus += 0.05
                elif act_id == 5 and ot is not None:
                    ref_price = curr_close
                    is_long = (ot["trade_type"] == "long")
                    new_sl, _ = _scale_sl_tp(
                        entry=ref_price, sl_norm=sl_norm, tp_norm=0.0,
                        is_long=is_long, min_stop_price=min_stop_price, atr_value=atr_val,
                    )
                    floor_price = max(float(min_stop_price or 0.0), 0.30 * atr_val)
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
                    ref_price = curr_close
                    is_long = (ot["trade_type"] == "long")
                    _, new_tp = _scale_sl_tp(
                        entry=ref_price, sl_norm=0.0, tp_norm=tp_norm,
                        is_long=is_long, min_stop_price=min_stop_price, atr_value=atr_val,
                    )
                    floor_price = max(float(min_stop_price or 0.0), 0.30 * atr_val)
                    if is_long:
                        new_tp = max(new_tp, ot["stop_loss"] + floor_price, ref_price + floor_price)
                    else:
                        new_tp = min(new_tp, ot["stop_loss"] - floor_price, ref_price - floor_price)
                    ot["take_profit"] = float(new_tp)
                    info["symbols"][sym] = {**info["symbols"].get(sym, {}), "adjusted": "tp"}

            # --- Post-check: allow SL/TP hits if still open and not opened this step ---
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

                        if stop_type == "sl":
                            self.sl_cooldown[i] = SL_COOLDOWN_STEPS
                            sl_penalty_total += SL_ILLEGAL_PENALTY
                            held_bars = self.current_step - ot.get("open_step", self.current_step)
                            if held_bars <= SL_EARLY_STEPS:
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
                        # Reward breaking long inactivity on this symbol
                        if self.last_trade_time[i] >= 10:  # ~50 daily bars idle
                            explore_bonus += 0.05
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

        # Compute unrealized PnL using last observed closes (no leakage)
        unrealized_pnl = 0.0
        for t in self.open_trades:
            entry = float(t["entry_price"])
            if int(t.get("open_step", -1)) == self.current_step:
                price_now = entry
            else:
                sym = t["symbol"]
                price_now = curr_close_by_sym.get(sym)
                if price_now is None:
                    ci = max(self.base + idx - 1, 0)
                    price_now = float(self.arr[sym]["close"][ci])
            if t["trade_type"] == "long":
                unrealized_pnl += (price_now - entry)
            else:
                unrealized_pnl += (entry - price_now)

        reward = float(self.reward_fn(
            closed_trades=info["closed_trades"],
            open_trades=open_trades_for_reward,
            account_balance=self.balance,
            unrealized_pnl=float(unrealized_pnl),
            time_since_last_trade=global_since_last_trade
        ).item())
        reward += float(explore_bonus)
        info["explore_bonus"] = float(explore_bonus)
        if not np.isfinite(reward):
            reward = 0.0
        # log reward components flat in info
        rc = getattr(self.reward_fn, "last_components", {}) or {}
        for k in ("C1_realizedR","C2_quality","C3_unreal","C4_inactivity",
                "C5_holding","C6_overexp","C7_conflict","C8_churnSLTP",
                "realized_R_mean","total_before_clip"):
            v = rc.get(k, None)
            if v is not None:
                info[f"rw_{k}"] = float(v)

        info["n_open"] = int(len(self.open_trades))
        info["n_closed"] = int(len(info["closed_trades"]))
        info["illegal_attempts"] = int(any_illegal_attempt)
        info["since_last_trade"] = float(global_since_last_trade)
        info["c_sl"] = sum(1 for ct in info["closed_trades"] if ct.get("stop_type") == "sl")
        info["c_tp"] = sum(1 for ct in info["closed_trades"] if ct.get("stop_type") == "tp")
        info["c_manual"] = sum(1 for ct in info["closed_trades"] if ct.get("stop_type") == "manual")

        if any_illegal_attempt:
            reward += ILLEGAL_ATTEMPT_PENALTY
        reward += float(illegal_penalty_total)
        reward += float(sl_penalty_total)

        final_cap = getattr(self.reward_fn, "final_clip", 5.0) or 5.0
        reward = float(np.clip(reward, -final_cap, final_cap))
        risk_vetos = sum(1 for s in info["symbols"].values() if s.get("risk_veto"))
        info["risk_veto_count"] = float(risk_vetos)
        # housekeeping
        MAX_CLOSED_TRADES = 1000
        if len(self.closed_trades) > MAX_CLOSED_TRADES:
            self.closed_trades = self.closed_trades[-MAX_CLOSED_TRADES:]
        self.sl_cooldown = np.maximum(self.sl_cooldown - 1, 0)

        self.current_step += 1
        terminated = self.current_step >= self.runtime_max_steps
        truncated = False

        if not (terminated or truncated):
            self.cursor += 1
            obs = self._get_observation(self.cursor).astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

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
def make_long_env(rank: int, seed: int, window: int, symbols=None) -> Callable[[], gym.Env]:
    def _init():
        env = LongBacktestEnv(
            csv_dir=DAILY_CSV_DIR,
            symbols=symbols or LIVE_FOREX_PAIRS,
            window=window,
            max_steps=1000,
            seed=seed + rank,
        )
        env.seed(seed + rank)
        log_dir = os.path.join(LOGS_DIR, f"long_worker_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env)  # <-- add this
        return env
    return _init

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_latest_checkpoint(ckpt_dir: str, last_ckpt_path: str, main_save_path: str) -> Optional[str]:
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "long_policy_ckpt_*_steps.zip"))
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

class FeatureUsageCallback(BaseCallback):
    def _on_step(self) -> bool:
        try:
            fe = getattr(self.model.policy, "features_extractor", None)
            if fe is not None:
                gate = getattr(fe, "extras_gate", None)
                extras_l1 = getattr(fe, "_extras_l1", None)
                core_l1 = getattr(fe, "_core_l1", None)
                if gate is not None:
                    self.model.logger.record("features/extras_gate", float(gate.detach().cpu().item()))
                if extras_l1 is not None:
                    self.model.logger.record("features/extras_l1", float(extras_l1.cpu().item()))
                if core_l1 is not None:
                    self.model.logger.record("features/core_l1", float(core_l1.cpu().item()))
        except Exception:
            pass
        return True
class AdjustInactivityWeightCallback(BaseCallback):
    def __init__(self, start_w: float, end_w: float, end_steps: int):
        super().__init__(verbose=0)
        self.start_w = float(start_w)
        self.end_w = float(end_w)
        self.end_steps = int(end_steps)

    def _on_step(self) -> bool:
        t = int(self.model.num_timesteps)
        if t >= self.end_steps:
            w = self.end_w
        else:
            # linear decay
            w = self.start_w + (self.end_w - self.start_w) * (t / max(1, self.end_steps))
        try:
            # Fan-out to all subproc envs
            self.model.get_env().env_method("set_inactivity_weight", w)
        except Exception:
            pass
        return True

# ──────────────────────────────────────────────────────────────────────────────
# Training (mirrors onemin structure, prefers recurrent)
# ──────────────────────────────────────────────────────────────────────────────
def train_long_policy(
    window: int = LONG_OBS_WINDOW,
    total_timesteps: int = 100_000_000,
    n_envs: int = 8,
    checkpoint_freq: int = 1_000_000,
    patience: int = 10_000,
    early_stopping_check_freq: int = 10_000,
):
    log.info("Starting Long-term training with PPO and event-based reward.")
    # Clean stale monitor files so new rw_* columns are written in the header
    try:
        for i in range(n_envs):
            _log_dir = os.path.join(LOGS_DIR, f"long_worker_{i}")
            os.makedirs(_log_dir, exist_ok=True)
            mon_csv = os.path.join(_log_dir, "monitor.csv")
            if os.path.exists(mon_csv):
                os.remove(mon_csv)
                log.debug("Monitor removed stale %s to refresh headers.", mon_csv)
    except Exception as e:
        log.debug("Monitor cleanup skipped: %s", e)
    # VecEnv
    # VecEnv (+ VecNormalize load/resume)
    env_fns = [make_long_env(i, SEED, window) for i in range(n_envs)]
    base_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv([env_fns[0]])

    pkl_path = os.path.join(MODELS_DIR, "checkpoints_long", "vecnormalize.pkl")
    # ── NEW: force fresh stats to avoid NaNs from stale/mismatched running means/vars
    if os.path.exists(pkl_path):
        try:
            os.remove(pkl_path)
            log.info("VecNormalize deleted stale stats at %s", pkl_path)
        except Exception as e:
            log.debug("VecNormalize could not delete %s: %s", pkl_path, e)
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
    log.info("[train_long] n_steps=%d n_envs=%d batch_size=%d rollout=%d", n_steps, n_envs, batch_size, rollout)
    check_every = rollout
    
    assert (n_steps * n_envs) % batch_size == 0, "n_steps * n_envs must be divisible by batch_size."

    algo_cls = PPO
    policy_cls = LongTermOHLCPolicy
    algo_kwargs = dict(
    n_steps=n_steps,
    batch_size=batch_size,
    learning_rate=1e-4,      
    gamma=0.995,
    gae_lambda=0.92,
    clip_range=0.3,
    ent_coef=0.01,
    vf_coef=0.25,
    max_grad_norm=1.0,
    tensorboard_log=os.path.join(LOGS_DIR, "tb_long_policy"),
    device="cuda",
    n_epochs=12,            
    target_kl=0.12,       
    )

    policy_kwargs = dict(
    window=window,
    embed_dim=128,
    tcn_hidden=128,
    n_heads= 8,
    n_layers= 4,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_dir = os.path.join(MODELS_DIR, "checkpoints_long")
    os.makedirs(ckpt_dir, exist_ok=True)
    main_save_path = os.path.join(ckpt_dir, "long_policy.zip")
    last_ckpt_path = os.path.join(ckpt_dir, "long_policy_last.zip")

    # Resume logic (unchanged)
    resume_path = get_latest_checkpoint(ckpt_dir, last_ckpt_path, main_save_path)
    target_total_timesteps = int(total_timesteps)

    if resume_path:
        log.info("Resuming from checkpoint: %s", resume_path)
        model = algo_cls.load(resume_path, env=vec_env, device="cuda")
        already_trained = getattr(model, "num_timesteps", steps_from_ckpt_name(resume_path))
        log.info("Already trained steps: %d", already_trained)
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

        # ── REPLACEMENT START (callback wiring) ────────────────────────────────────
    rclone_dest = os.getenv("RCLONE_DEST")
    vecnorm_stats_path = os.path.join(ckpt_dir, "vecnormalize.pkl")
    try:
        vec_env.save(vecnorm_stats_path)
    except Exception as e:
        log.debug("VecNormalize initial stats snapshot failed: %s", e)

    comp_logger_callback = LogRewardComponentsCallback(section="rollout", verbose=0)
    log_std_callback = LogStdCallback()
    feature_usage_callback = FeatureUsageCallback()

    # Save "training best" using the exact SB3 ep_rew_mean metric
    train_best_cb = SaveTrainBestByEpBuffer(
        save_path=os.path.join(ckpt_dir, "long_policy_trainbest.zip"),
        check_freq=check_every,  # one full rollout
        rclone_dest=rclone_dest,
        verbose=1,
    )

    rollouts_per_eval = 20  # tune to taste (5–10 typical)
    deploy_best_cb = EvalAndSaveDeployBestCallback(
        vecnorm_stats_path=vecnorm_stats_path,
        make_env_fn=lambda rank, seed, window: make_long_env(rank, seed + 12345, window, symbols=LIVE_FOREX_PAIRS),
        n_eval_envs=max(1, n_envs // 2),
        episodes = max(5, n_envs // 2),
        check_freq=rollouts_per_eval * check_every,
        save_path=os.path.join(ckpt_dir, "long_policy_deploybest.zip"),
        rclone_dest=rclone_dest,
        verbose=1,
    )

    early_stopping_callback = EarlyStoppingCallback(
        check_freq=early_stopping_check_freq or check_every,
        patience=patience,
        verbose=1,
    )

    callbacks = [
        comp_logger_callback,
        log_std_callback,
        feature_usage_callback,
        train_best_cb,
        deploy_best_cb,
        early_stopping_callback,
    ]
    # Keep a gentle pressure to trade, but not overwhelming
    adjust_inactivity_cb = AdjustInactivityWeightCallback(
        start_w=0.015, end_w=0.0015, end_steps=2_000_000
    )

    callbacks.append(adjust_inactivity_cb)


    if rclone_dest:
        checkpoint_callback = CheckpointAndRcloneCallback(
            checkpoint_freq=checkpoint_freq,
            ckpt_dir=ckpt_dir,
            name_prefix="long_policy_ckpt",
            rclone_dest=rclone_dest,
        )
        callbacks.insert(0, checkpoint_callback)
    else:
        log.debug("RCLONE_DEST not set; saving locally only.")
    # ── REPLACEMENT END ────────────────────────────────────────────────────────
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
                log.info("VecNormalize saved running stats to %s", os.path.join(ckpt_dir, "vecnormalize.pkl"))
            except Exception as e:
                log.warning("VecNormalize could not save stats: %s", e)

            log.info("Training complete. Model saved to %s and %s", main_save_path, last_ckpt_path)
        else:
            log.info("Training already completed by checkpoint/model counter: >= target %d", target_total_timesteps)
    finally:
        if ASYNC_UPLOADER:
            ASYNC_UPLOADER.wait()  # drain queue at the very end only
        vec_env.close()
if __name__ == "__main__":
    train_long_policy()
