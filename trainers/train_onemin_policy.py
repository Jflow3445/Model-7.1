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
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EventCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


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
ILLEGAL_ACTION_PENALTY = -10
SEVERE_ILLEGAL_ACTION_PENALTY = -20
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
    Map normalized [-1,1] to price distances using ATR + broker floor.
    """
    FLOOR_FRAC_ATR = 0.20  # min fraction of ATR for floors
    K_SL_ATR = 0.60
    K_TP_ATR = 1.20

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
class CheckpointAndRcloneCallback(CheckpointCallback):
    def __init__(self, *args, models_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.models_dir = models_dir

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.num_timesteps % self.save_freq == 0:
            print(f"[CheckpointAndRcloneCallback] Saved checkpoint at {self.num_timesteps} steps.")
            print(f"  Files in {self.save_path}: {os.listdir(self.save_path)}")
            # If rclone is configured on your machine, you can sync:
            # os.system(f"rclone copy {self.models_dir} gdrive:models")
        return result

class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, check_freq, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        results = []
        for i in range(self.training_env.num_envs):
            monitor_file = os.path.join(LOGS_DIR, f"onemin_worker_{i}", "monitor.csv")
            if os.path.exists(monitor_file):
                df = pd.read_csv(monitor_file, skiprows=1)
                if "r" in df.columns:
                    results.extend(df["r"].values[-200:])  # last 200 episodic returns

        if results:
            mean_reward = float(np.mean(results))
            if self.verbose:
                print(f"[SaveBestModel] mean_reward={mean_reward:.3f} best={self.best_mean_reward:.3f}")
            if mean_reward > self.best_mean_reward + 1e-4:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self.verbose:
                    print(f"[SaveBestModel] New best → saved to {self.save_path}")
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
    onemin OHLC backtest with:
      - Single open trade per symbol
      - ATR+broker-floor mapped SL/TP from normalized [-1,1]
      - Event-based reward (reward only on close)
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
                raise FileNotFoundError(f"Onemin CSV not found for '{sym}': {path}")
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

            df = pd.read_csv(path, parse_dates=["time"], skiprows=skiprows, nrows=needed_rows - skip)
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
                f"Not enough aligned onemin bars across symbols after ATR handling "
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
            elapsed = np.array([self.last_trade_time[i]], dtype=np.float32)
            obs_parts.append(np.concatenate([opens, highs, lows, closes, elapsed], axis=0))
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
            # In a trade -> cannot buy/sell; only close/adjust
            valid[1] = False
            valid[2] = False
            if ot["trade_type"] == "long":
                valid[4] = False  # can't close short
            elif ot["trade_type"] == "short":
                valid[3] = False  # can't close long
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
            raw_head = np.where(np.isfinite(act[:8]), act[:8], -np.inf)
            orig_action = int(np.argmax(raw_head))

            # Ensure <= 1 open trade per symbol
            open_for_sym = [t for t in self.open_trades if t["symbol"] == sym]
            if len(open_for_sym) > 1:
                logging.error(f"[{sym}] Multiple open trades detected, auto-closing extras.")
                illegal_penalty_total += SEVERE_ILLEGAL_ACTION_PENALTY
                for ot in open_for_sym[1:]:
                    self.closed_trades.append(dict(
                        **ot,
                        exit_price=ot["entry_price"],
                        pnl=0.0,
                        slippage=_to_float(self.reward_fn.slippage_per_unit),
                        commission=_to_float(self.reward_fn.commission_per_trade),
                        close_step=self.current_step,
                        stop_type="illegal_auto",
                    ))
                keep = open_for_sym[0]
                self.open_trades = [t for t in self.open_trades if t["symbol"] != sym] + [keep]

            masked = self._mask_illegal_actions(i, act)
            if np.all(np.isneginf(masked[:8])):
                act_id = 0  # noop
            else:
                act_id = int(np.nanargmax(masked[:8]))

            if masked[orig_action] == -np.inf:
                illegal_penalty_total += ILLEGAL_ACTION_PENALTY

            # Prices
            next_open = float(df.at[next_idx, "open"])
            next_close = float(df.at[next_idx, "close"])
            next_high = float(df.at[next_idx, "high"])
            next_low = float(df.at[next_idx, "low"])
            atr_val = float(df.at[idx, "atr"])
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
                # Manual close at next close
                if ot["trade_type"] == "long":
                    pnl = next_close - ot["entry_price"]
                else:
                    pnl = ot["entry_price"] - next_close
                closed = dict(
                    **ot,
                    exit_price=next_close,
                    pnl=pnl,
                    slippage=_to_float(self.reward_fn.slippage_per_unit),
                    commission=_to_float(self.reward_fn.commission_per_trade),
                    close_step=self.current_step,
                    stop_type="manual",
                )
                self.closed_trades.append(closed)
                self.balance += (closed["pnl"] - closed["slippage"] - closed["commission"])
                self.open_trades = [t for t in self.open_trades if t["symbol"] != sym]
                info["closed_trades"].append(closed)
                trade_executed = True

            # === Auto-close by SL/TP using next bar high/low ===
            if not trade_executed:
                ot = self._get_open_trade(sym)
                if ot is not None:
                    hit = False
                    if ot["trade_type"] == "long":
                        if next_low <= ot["stop_loss"]:
                            pnl = ot["stop_loss"] - ot["entry_price"]
                            exit_px = ot["stop_loss"]; stop_type = "sl"; hit = True
                        elif next_high >= ot["take_profit"]:
                            pnl = ot["take_profit"] - ot["entry_price"]
                            exit_px = ot["take_profit"]; stop_type = "tp"; hit = True
                    else:
                        if next_high >= ot["stop_loss"]:
                            pnl = ot["entry_price"] - ot["stop_loss"]
                            exit_px = ot["stop_loss"]; stop_type = "sl"; hit = True
                        elif next_low <= ot["take_profit"]:
                            pnl = ot["entry_price"] - ot["take_profit"]
                            exit_px = ot["take_profit"]; stop_type = "tp"; hit = True

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
                        self.balance += (closed["pnl"] - closed["slippage"] - closed["commission"])
                        self.open_trades = [t for t in self.open_trades if t["symbol"] != sym]
                        info["closed_trades"].append(closed)
                        trade_executed = True

            self.last_trade_time[i] = 0.0 if trade_executed else (self.last_trade_time[i] + 1.0)
            info["symbols"][sym] = {"executed": trade_executed, "action_id": act_id}

        # === Event-based reward: only when trades close ===
        if info["closed_trades"]:
            reward = float(self.reward_fn(
                closed_trades=info["closed_trades"],
                open_trades=self.open_trades.copy(),
                account_balance=self.balance,
                unrealized_pnl=0.0,
                time_since_last_trade=0.0,
            ).item())
        else:
            reward = 0.0

        reward += illegal_penalty_total

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
    n_envs: int = 16,
    checkpoint_freq: int = 10_000,
    patience: int = 100,
    early_stopping_check_freq: int = 10_000,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_onemin_policy")
    logger.info("Starting onemin-term training with PPO and event-based reward.")

    # VecEnv
    env_fns = [make_onemin_env(i, SEED, window) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv([env_fns[0]])

    n_steps = 2048
    batch_size = 1024
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
    checkpoint_callback = CheckpointAndRcloneCallback(
        save_freq=checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="onemin_policy_ckpt",
        models_dir=MODELS_DIR,
    )
    best_model_callback = SaveBestModelCallback(
        save_path=os.path.join(ckpt_dir, "onemin_policy_best.zip"),
        check_freq=checkpoint_freq,
        verbose=1,
    )
    early_stopping_callback = EarlyStoppingCallback(
        check_freq=early_stopping_check_freq,
        patience=patience,
        verbose=1,
    )

    try:
        if timesteps_left > 0:
            model.learn(
                total_timesteps=timesteps_left,
                callback=[checkpoint_callback, best_model_callback, early_stopping_callback],
                reset_num_timesteps=False,
            )
            model.save(last_ckpt_path)
            model.save(main_save_path)
            logger.info(f"Training complete. Model saved to {main_save_path} and {last_ckpt_path}")
        else:
            logger.info(f"Training already completed by checkpoint/model counter: >= target {target_total_timesteps}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    train_onemin_policy()
