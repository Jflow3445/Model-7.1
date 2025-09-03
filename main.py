# main.py
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import time
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import gymnasium as gym
import sys as _sys
_sys.modules.setdefault("gym", gym) 
import torch
from stable_baselines3 import PPO
from config.settings import (
    BASE_DIR,
    ONEMIN_OBS_WINDOW,
    MEDIUM_OBS_WINDOW,
    LONG_OBS_WINDOW,
    LIVE_FOREX_PAIRS,
    MODELS_DIR,
    RETRAIN_SCRIPTS,
    RETRAIN_INTERVALS,
    SAVE_INTERVAL,
    ENABLE_ONEMIN, 
    ENABLE_MEDIUM, 
    ENABLE_LONG
)
from envs.live_env import LiveTradeEnv
from models.onemin_policy import OneMinOHLCPolicy
from models.medium_policy import MediumTermOHLCPolicy
from models.long_policy import LongTermOHLCPolicy
from arbiter import MasterArbiter
try:
    from dotenv import load_dotenv 
    load_dotenv()
except Exception:
    pass
from nister_client import send_signal, send_trade
import uuid

import MetaTrader5 as mt5

USE_MT5 = False
medium_window = MEDIUM_OBS_WINDOW
long_window = LONG_OBS_WINDOW
# ── MT5 helper: get account balance ──────────────────────────────────────────

def _get_mt5_balance() -> float:
    info = mt5.account_info()
    return float(info.balance) if info else 0.0

def safe_predict(model_manager, obs, label):
    action, _ = model_manager.predict(obs, deterministic=True)
    if np.isnan(action).any():
        logging.error(f"[main] NaN in {label} output! Replacing with zeros and skipping trade for this policy.")
        return np.zeros_like(action), _
    return action, _

def zeros_logits(n_symbols: int) -> np.ndarray:
    # per asset we always use 10-dim action head
    return np.zeros((n_symbols, 10), dtype=np.float32)

def zeros_obs_vec(n_symbols: int) -> np.ndarray:
    # used only if you ever need a placeholder; not strictly required
    return np.zeros((n_symbols * 10,), dtype=np.float32)

def zeros_regime(n_assets: int) -> np.ndarray:
    # 4 logits per policy “regime”; one block per asset
    return np.zeros(4 * n_assets, dtype=np.float32)

# ── Dynamic lot‐size calculation ─────────────────────────────────────────────

def get_dynamic_lot_size(account_balance: float, symbol: str, sl_pips: int) -> float:
    risk_amount = account_balance * 0.01
    if sl_pips <= 0:
        return 0.01
    pip_value_per_lot = 10.0
    lots = risk_amount / (sl_pips * pip_value_per_lot + 1e-8)
    return max(round(lots, 2), 0.01)


# ── Build tick‐policy observation vector ───────────────────────────────────────

def build_onemin_obs(env: LiveTradeEnv, symbols: List[str], window: int) -> Optional[np.ndarray]:
    obs_list: List[np.ndarray] = []
    for s in symbols:
        o_full = env.get_onemin_observation(s)
        if o_full is None:
            logging.warning(f"[build_onemin_obs] Insufficient ticks for {s}")
            return None
        if o_full.shape[0] != 4 * window + 1:
            raise ValueError(f"[build_onemin_obs] Expected length {4*window+1} for {s}, got {o_full.shape[0]}")
        obs_list.append(o_full)
    return np.concatenate(obs_list, axis=0)
# ── Build medium‐policy observation vector ────────────────────────────────────

def build_medium_obs(
    env: LiveTradeEnv,
    symbols: List[str],
    medium_window: int
) -> Optional[np.ndarray]:
    obs_list: List[np.ndarray] = []
    expected_bars = medium_window
    for s in symbols:
        bars = list(env.medium_deques[s])
        if len(bars) < expected_bars:
            logging.warning(f"[build_medium_obs] Insufficient medium‐bars for {s}")
            return None
        # Fix: Use full OHLC per bar
                # Long policy expects 5 values/bar + 1 extra → pad a 5th channel if you only have OHLC
        arr = np.array([[o, h, l, c, 0.0] for (o, h, l, c) in bars[-expected_bars:]], dtype=np.float32).flatten()
        # elapsed is not used in long-term but keep for compatibility
        elapsed = np.array([0.0], dtype=np.float32)
        o_full = np.concatenate([arr, elapsed], axis=0)
        # Expect 5*window + 1 features per symbol
        if o_full.shape[0] != 5 * expected_bars + 1:
            raise ValueError(f"[build_long_obs] Expected {5*expected_bars+1} elements for {s}, got {o_full.shape[0]}")

        obs_list.append(o_full)
    return np.concatenate(obs_list, axis=0)


# ── Build long‐policy observation vector ──────────────────────────────────────

def build_long_obs(
    env: LiveTradeEnv,
    symbols: List[str],
    long_window: int
) -> Optional[np.ndarray]:
    obs_list: List[np.ndarray] = []
    expected_bars = long_window
    for s in symbols:
        bars = list(env.long_deques[s])
        if len(bars) < expected_bars:
            logging.warning(f"[build_long_obs] Insufficient long‐bars for {s}")
            return None
        # Use full OHLC per bar (open, high, low, close)
        arr = np.array([[o, h, l, c] for (o, h, l, c) in bars[-expected_bars:]], dtype=np.float32).flatten()
        # elapsed is not used in long-term but keep for compatibility
        elapsed = np.array([0.0], dtype=np.float32)
        o_full = np.concatenate([arr, elapsed], axis=0)
        # Now expect 4*window + 1 features per symbol
        if o_full.shape[0] != 4 * expected_bars + 1:
            raise ValueError(f"[build_long_obs] Expected {4*expected_bars+1} elements for {s}, got {o_full.shape[0]}")
        obs_list.append(o_full)
    return np.concatenate(obs_list, axis=0)

# ── Dummy Environments for ModelManager ───────────────────────────────────────

class DummyOneMinEnv(gym.Env):
    def __init__(self, n_symbols: int, window: int):
        super().__init__()
        obs_dim = n_symbols * (4* window + 1)
        low = np.full((obs_dim,), -np.inf, dtype=np.float32)
        high = np.full((obs_dim,),  np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10 * n_symbols,), dtype=np.float32)

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, {}


class DummyMediumEnv(gym.Env):
    def __init__(self, n_symbols: int, medium_window: int):
        super().__init__()
        obs_dim = n_symbols * (4 * medium_window + 1)
        low = np.full((obs_dim,), -np.inf, dtype=np.float32)
        high = np.full((obs_dim,),  np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10 * n_symbols,), dtype=np.float32)

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, {}


class DummyLongEnv(gym.Env):
    def __init__(self, n_symbols: int, long_window: int):
        super().__init__()
        obs_dim = n_symbols * (5 * long_window + 1)
        low = np.full((obs_dim,), -np.inf, dtype=np.float32)
        high = np.full((obs_dim,),  np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10 * n_symbols,), dtype=np.float32)

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, {}


# ── ModelManager ─────────────────────────────────────────────────────────────

class ModelManager:
    """
    Manages loading, saving, predicting, and periodic retraining of a PPO model.
    """
    def __init__(
        self,
        policy_class: Any,
        model_path: str,
        policy_kwargs: Dict[str, Any],
        dummy_env: gym.Env,
    ):
        self.policy_class = policy_class
        self.model_path = model_path
        self._all_kwargs = policy_kwargs.copy()
        self.dummy_env = dummy_env
        self.model: Optional[PPO] = None
        self.last_retrain: float = 0.0
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(
                    self.model_path,
                    env=self.dummy_env,
                    device="auto",
                    custom_objects={
                        # Replace old schedule callables with constants
                        "learning_rate": self._all_kwargs.get("learning_rate", 2e-4),
                        "clip_range": self._all_kwargs.get("clip_range", 0.2),
                        # Provide sane defaults if missing in old zips
                        "sde_sample_freq": 4,
                        "n_steps": self._all_kwargs.get("n_steps", 2048),
                    },
                )

                logging.info(f"[ModelManager] Loaded model from {self.model_path}")
            except Exception as e:
                logging.warning(f"[ModelManager] Failed to load {self.model_path} ({e}); initializing new model.")
                self._init_new()
        else:
            self._init_new()

    def _init_new(self):
        logging.info(f"[ModelManager] Initializing new PPO for {self.policy_class.__name__}")
        policy_network_kwargs: Dict[str, Any] = {}
        for key in ("window", "net_arch"):
            if key in self._all_kwargs:
                policy_network_kwargs[key] = self._all_kwargs.pop(key)

        # Mirror use_sde into the policy (do NOT pop; we want PPO to see it too)
        ppo_kwargs = self._all_kwargs
        policy_network_kwargs.pop("use_sde", None)  # defensive: ensure it's not present
        try:
            self.model = PPO(
                self.policy_class,
                self.dummy_env,
                verbose=0,
                policy_kwargs=policy_network_kwargs,
                **ppo_kwargs,
            )
        except Exception as e:
            logging.error(f"[ModelManager] Exception initializing PPO: {e}")
            raise

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action, _

    def maybe_retrain(
        self,
        now: float,
        retrain_interval: float,
        train_script: str,
    ):
        if retrain_interval <= 0 or not train_script:
            return
        if (now - self.last_retrain) >= retrain_interval:
            logging.info(f"[ModelManager] Triggering retrain for {self.policy_class.__name__}")
            try:
                subprocess.Popen(["python", train_script])
                self.last_retrain = now
            except Exception as e:
                logging.error(f"[ModelManager] Failed to start retraining: {e}")


# ── Main Live‐Trading Loop ────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("[main] Starting live‐trading main loop.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Launch data system (MT5 + ForexDataSystem)
    from data.forex_data_system import ForexDataSystem

    # Set this flag to False to run historical/CSV-only, or True for live MT5 integration
    USE_MT5 = True  # <-- Set as needed, or use an argument/env variable for dynamic switching

    forex_data = ForexDataSystem(
        requested_symbols=LIVE_FOREX_PAIRS,
        use_mt5=USE_MT5
    )


    # 2) Initialize LiveTradeEnv
    env = LiveTradeEnv(symbols=LIVE_FOREX_PAIRS, data_system=forex_data)
    #env.hydrate_from_logs()
    env.reset()
    symbols = LIVE_FOREX_PAIRS
    n_symbols = len(symbols)
    last_env_action = {s: None for s in symbols}
    last_sent_signal = {s: None for s in symbols}

    # 3) Instantiate sub‐policy ModelManagers
    # 3) Instantiate sub‐policy ModelManagers
    onemin_manager = None
    if ENABLE_ONEMIN:
        onemin_manager = ModelManager(
            policy_class=OneMinOHLCPolicy,
            model_path=str(MODELS_DIR / "onemin_policy.zip"),
            policy_kwargs={
                "n_steps": 2048,
                "batch_size": 256,
                "learning_rate": 2e-4,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "use_sde": True,
                "sde_sample_freq": 4,
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "window": ONEMIN_OBS_WINDOW,
                "tensorboard_log": str(BASE_DIR / "logs/tb_onemin_policy"),
            },
            dummy_env=DummyOneMinEnv(n_symbols, ONEMIN_OBS_WINDOW),
        )
    else:
        logging.info("[main] OneMin policy DISABLED")

    medium_manager = None
    if ENABLE_MEDIUM:
        medium_manager = ModelManager(
            policy_class=MediumTermOHLCPolicy,
            model_path=str(MODELS_DIR / "medium_policy.zip"),
            policy_kwargs={
                "n_steps": 2048,
                "batch_size": 256,
                "learning_rate": 2e-4,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "tensorboard_log": str(BASE_DIR / "logs/tb_medium_policy"),
                "window": MEDIUM_OBS_WINDOW,
                "use_sde": True,
                "sde_sample_freq": 4,
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            },
            dummy_env=DummyMediumEnv(n_symbols, medium_window),
        )
    else:
        logging.info("[main] Medium policy DISABLED")

    long_manager = None
    if ENABLE_LONG:
        long_manager = ModelManager(
            policy_class=LongTermOHLCPolicy,
            model_path=str(MODELS_DIR / "long_policy.zip"),
            policy_kwargs={
                "n_steps": 2048,
                "batch_size": 256,
                "learning_rate": 2e-4,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "tensorboard_log": str(BASE_DIR / "logs/tb_long_policy"),
                "window": LONG_OBS_WINDOW,
                "use_sde": True,
                "sde_sample_freq": 4,
                "net_arch": dict(pi=[192, 128], vf=[192, 128]),
            },
            dummy_env=DummyLongEnv(n_symbols, long_window),
        )
    else:
        logging.info("[main] Long policy DISABLED")


    # 4) Instantiate Arbiter (final decision maker)
    context_dim = 24
    history_len = 20
    regime_dim = 2 * n_symbols + 3 * 4 * n_symbols  # 2 per symbol + 4*3 policies per symbol
    arbiter = MasterArbiter(
        n_assets=n_symbols,
        action_dim=10 * n_symbols,
        context_dim=context_dim,
        n_heads=2,
        hidden_dim=128,
        hist_len=history_len,
        regime_dim=regime_dim,
    ).to(device)
    arbiter.eval()

    arbiter_model_path = MODELS_DIR / "arbiter_model.pt"
    if arbiter_model_path.exists():
        try:
            arbiter.load_state_dict(torch.load(arbiter_model_path, map_location=device))
            logging.info(f"[main] Loaded Arbiter weights from {arbiter_model_path}")
        except Exception as e:
            logging.warning(f"[main] Failed to load Arbiter weights ({e}); proceeding with random init.")

    # 5) Warm‐up: wait until env has enough data for initial observations
    # 5) Warm‐up: wait until env has enough data for required observations
    while True:
        ok = True
        if ENABLE_ONEMIN:
            ok &= (build_onemin_obs(env, symbols, ONEMIN_OBS_WINDOW) is not None)
        if ENABLE_MEDIUM:
            ok &= (build_medium_obs(env, symbols, medium_window) is not None)
        if ENABLE_LONG:
            ok &= (build_long_obs(env, symbols, long_window) is not None)
        if ok:
            break
        time.sleep(0.1)

    # 6) Compute initial sub‐policy logits to build first context
    if ENABLE_ONEMIN:
        onemin_vec = build_onemin_obs(env, symbols, ONEMIN_OBS_WINDOW)
        onemin_input = np.expand_dims(onemin_vec, 0)
        onemin_action_raw, _ = safe_predict(onemin_manager, onemin_input, "OneMinOHLCPolicy")
        onemin_action_raw = onemin_action_raw.reshape(n_symbols, 10)
    else:
        onemin_vec = None
        onemin_action_raw = zeros_logits(n_symbols)

    # medium
    if ENABLE_MEDIUM:
        medium_vec = build_medium_obs(env, symbols, medium_window)
        med_input = np.expand_dims(medium_vec, 0)
        medium_action_raw, _ = safe_predict(medium_manager, med_input, "MediumOHLCPolicy")
        medium_action_raw = medium_action_raw.reshape(n_symbols, 10)
    else:
        medium_vec = None
        medium_action_raw = zeros_logits(n_symbols)

    # long
    if ENABLE_LONG:
        long_vec = build_long_obs(env, symbols, long_window)
        long_input = np.expand_dims(long_vec, 0)
        long_action_raw, _ = safe_predict(long_manager, long_input, "LongOHLCPolicy")
        long_action_raw = long_action_raw.reshape(n_symbols, 10)
    else:
        long_vec = None
        long_action_raw = zeros_logits(n_symbols)

    # 7) Prepare history buffer for Arbiter
    context_per_asset_initial: List[torch.Tensor] = []
    for i in range(n_symbols):
        ti = torch.tensor(onemin_action_raw[i], dtype=torch.float32, device=device) if ENABLE_ONEMIN else torch.zeros(10, device=device)
        mi = torch.tensor(medium_action_raw[i], dtype=torch.float32, device=device) if ENABLE_MEDIUM else torch.zeros(10, device=device)
        li = torch.tensor(long_action_raw[i], dtype=torch.float32, device=device) if ENABLE_LONG   else torch.zeros(10, device=device)
        extras = torch.zeros(3, dtype=torch.float32, device=device)  # [boot_time, total_reward, done] at t=0
        context_per_asset_initial.append(torch.cat([ti, mi, li, extras], dim=0))

    context_torch_initial = torch.cat(context_per_asset_initial, dim=0)  # shape: n_symbols * (10+10+10+3)

    history_buffer = deque(maxlen=history_len)
    for _ in range(history_len):
        history_buffer.append(context_torch_initial.clone())

    # 8) Track start time to compute boot_time
    start_time = time.time()

    # 9) Ensure arbiter training‐data directory exists
    arbiter_data_dir = MODELS_DIR / "arbiter_training_data"
    arbiter_data_dir.mkdir(parents=True, exist_ok=True)

    # 10) Main live‐trading loop
    last_save = time.time()
    step_counter = 0
    last_info: Optional[Dict[str, Any]] = None

    while True:
        now = time.time()

        # 10a) Possibly retrain sub‐policies
        if ENABLE_ONEMIN and onemin_manager is not None:
            onemin_manager.maybe_retrain(
                now,
                RETRAIN_INTERVALS.get("onemin_policy", 0.0),
                RETRAIN_SCRIPTS.get("onemin_policy", "")
            )
        if ENABLE_MEDIUM and medium_manager is not None:
            medium_manager.maybe_retrain(
                now,
                RETRAIN_INTERVALS.get("medium_policy", 0.0),
                RETRAIN_SCRIPTS.get("medium_policy", "")
            )
        if ENABLE_LONG and long_manager is not None:
            long_manager.maybe_retrain(
                now,
                RETRAIN_INTERVALS.get("long_policy", 0.0),
                RETRAIN_SCRIPTS.get("long_policy", "")
            )

        # 10b) Build and run sub‐policy predictions (only for enabled policies)
        if ENABLE_ONEMIN:
            try:
                onemin_vec = build_onemin_obs(env, symbols, ONEMIN_OBS_WINDOW)
                if onemin_vec is None:
                    logging.warning("[main] onemin_vec is None, sleeping 60s.")
                    time.sleep(60); continue
                onemin_input = np.expand_dims(onemin_vec, 0)
                onemin_action_raw, _ = safe_predict(onemin_manager, onemin_input, "OneMinOHLCPolicy")
                onemin_action_raw = onemin_action_raw.reshape(n_symbols, 10)
                onemin_obs_torch = torch.tensor(onemin_input, dtype=torch.float32, device=device)
                onemin_regime = onemin_manager.model.policy.get_regime_logits(onemin_obs_torch)
            except Exception as e:
                logging.error(f"[main] onemin block error: {e}")
                time.sleep(60); continue
        else:
            onemin_action_raw = zeros_logits(n_symbols)
            onemin_regime = None

        # medium
        if ENABLE_MEDIUM:
            try:
                medium_vec = build_medium_obs(env, symbols, medium_window)
                if medium_vec is None:
                    logging.warning("[main] medium_vec is None, sleeping 60s.")
                    time.sleep(60); continue
                med_input = np.expand_dims(medium_vec, 0)
                medium_action_raw, _ = safe_predict(medium_manager, med_input, "MediumOHLCPolicy")
                medium_action_raw = medium_action_raw.reshape(n_symbols, 10)
                medium_obs_torch = torch.tensor(med_input, dtype=torch.float32, device=device)
                medium_regime = medium_manager.model.policy.get_regime_logits(medium_obs_torch)
            except Exception as e:
                logging.error(f"[main] medium block error: {e}")
                time.sleep(60); continue
        else:
            medium_action_raw = zeros_logits(n_symbols)
            medium_regime = None

        # long
        if ENABLE_LONG:
            try:
                long_vec = build_long_obs(env, symbols, long_window)
                if long_vec is None:
                    logging.warning("[main] long_vec is None, sleeping 60s.")
                    time.sleep(60); continue
                long_input = np.expand_dims(long_vec, 0)
                long_action_raw, _ = safe_predict(long_manager, long_input, "LongOHLCPolicy")
                long_action_raw = long_action_raw.reshape(n_symbols, 10)
                long_obs_torch = torch.tensor(long_input, dtype=torch.float32, device=device)
                long_regime = long_manager.model.policy.get_regime_logits(long_obs_torch)
            except Exception as e:
                logging.error(f"[main] long block error: {e}")
                time.sleep(60); continue
        else:
            long_action_raw = zeros_logits(n_symbols)
            long_regime = None

        # 10c) Prepare logits tensors
        t_logits_t = torch.tensor(onemin_action_raw, dtype=torch.float32, device=device).unsqueeze(0)
        m_logits_t = torch.tensor(medium_action_raw, dtype=torch.float32, device=device).unsqueeze(0)
        l_logits_t = torch.tensor(long_action_raw, dtype=torch.float32, device=device).unsqueeze(0)

        # 10d) Build regime_context (zeros for disabled or early steps)
        status = env.get_status()
        need_zero = False
        if ENABLE_ONEMIN:
            need_zero |= any(status[s].get("onemin_buffer", 0) < ONEMIN_OBS_WINDOW for s in symbols)
        if ENABLE_MEDIUM:
            need_zero |= any(status[s].get("medium_buffer", 0) < medium_window for s in symbols)
        if ENABLE_LONG:
            need_zero |= any(status[s].get("long_buffer", 0) < long_window for s in symbols)

        if need_zero or last_info is None:
            regime_context_t = torch.zeros((1, regime_dim), device=device)
        else:
            vol_list = [last_info[s].get("volatility", 0.0) for s in symbols]
            spread_list = [last_info[s].get("spread", 0.0) for s in symbols]
            vol_arr = np.array(vol_list, dtype=np.float32)
            spread_arr = np.array(spread_list, dtype=np.float32)

            def get_regime_flat(regime, n):
                if regime is None:
                    return zeros_regime(n)
                arr = regime.detach().cpu().numpy().flatten() if isinstance(regime, torch.Tensor) else np.array(regime).flatten()
                if arr.size == 4:         # one 4-d block → repeat per asset
                    return np.tile(arr, n)
                if arr.size == 4 * n:     # already per-asset
                    return arr
                # unexpected shape → safe fallback
                return zeros_regime(n)
            onemin_regime_flat = get_regime_flat(onemin_regime, n_symbols)
            medium_regime_flat = get_regime_flat(medium_regime, n_symbols)
            long_regime_flat = get_regime_flat(long_regime, n_symbols)

            regime_np = np.concatenate([
                vol_arr,
                spread_arr,
                onemin_regime_flat,
                medium_regime_flat,
                long_regime_flat,
            ], axis=0)
            regime_context_t = torch.tensor(regime_np[None, :], dtype=torch.float32, device=device)


        # 10e) Build history tensor
        history_stack = torch.stack(list(history_buffer), dim=0)                    # [hist_len, n_assets*input_dim]
        history_stack = history_stack.view(history_len, n_symbols, -1)             # [hist_len, n_assets, input_dim]
        history_t = history_stack.permute(1, 0, 2).unsqueeze(0)                     # [1, n_assets, hist_len, input_dim]

        # 10f) Fuse policies via Arbiter—flatten logits before calling
        boot_time_t = torch.tensor([[now - start_time]], dtype=torch.float32, device=device)
        batch_size = t_logits_t.shape[0]
        onemin_flat = t_logits_t.view(batch_size, -1)    # [1, n_assets*10]
        med_flat  = m_logits_t.view(batch_size, -1)
        long_flat = l_logits_t.view(batch_size, -1)
        with torch.no_grad():
            fused_actions_t, *_ = arbiter(
                onemin_action=onemin_flat,
                medium_action=med_flat,
                long_action=long_flat,
                history=history_t,
                # boot_time=boot_time_t,  # optional in your build; commented in your code
                regime_context=regime_context_t,
                deterministic=True,  # matches your usage pattern for live execution
            )
        if torch.isnan(fused_actions_t).any():
            logging.error("[main] NaN detected in arbiter fused actions! Skipping this trading step.")
            time.sleep(1)
            continue
        # 10g) Build final action dict per symbol
        fused_flat = fused_actions_t.squeeze(0).cpu().numpy()           
        fused_np = fused_flat.reshape(n_symbols, 10)                   
        final_action_dict: Dict[str, Any] = {}
        for i, s in enumerate(symbols):
            fused_i = fused_np[i]
            act_id = int(np.argmax(fused_i[:8]))
            sl_norm = float(fused_i[8])
            tp_norm = float(fused_i[9])
            sl_pips = int(30 + ((sl_norm + 1.0) / 2.0) * (200 - 30))
            tp_pips = int(40 + ((tp_norm + 1.0) / 2.0) * (500 - 40))
            balance = _get_mt5_balance()
            if balance == 0.0:
                logging.error("[main] MT5 balance fetch failed (account_info None or zero). Skipping trading for this round.")
                time.sleep(10)
                continue
            lot_size = get_dynamic_lot_size(balance, s, sl_pips)

            logits_arr = np.zeros(10, dtype=np.float32)
            logits_arr[act_id] = 1.0

            current_action = {
                "action": logits_arr.tolist(),  # Ensure lists (not numpy arrays) for comparison
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "lot_size": lot_size,
            }

            # Only include if action differs from last_env_action
            if current_action != last_env_action.get(s):
                final_action_dict[s] = current_action
                last_env_action[s] = current_action
        # 10h) Execute in LiveTradeEnv
        step_results = env.step(final_action_dict)
        if step_results is None or any(s not in step_results for s in symbols):
            logging.error("[main] env.step returned None or missing symbol keys. Skipping.")
            time.sleep(1)
            continue
        if any(np.isnan(info.get("reward", 0.0)) for info in step_results.values()):
            logging.error("[main] env.step reward NaN detected! Skipping.")
            time.sleep(1)
            continue
        last_info = step_results
                        # -- Send executed trades to server (secure client) --
        for sym, sym_info in step_results.items():
            # Only log/send if a trade was actually closed
            if sym_info.get("trade_closed", False):
                trade_payload = {
                    "symbol": sym,
                    "side": sym_info.get("side"),  # "buy"/"sell"
                    "entry_price": sym_info.get("entry_price"),
                    "exit_price": sym_info.get("exit_price"),
                    "volume": sym_info.get("lot_size"),
                    "pnl": sym_info.get("pnl"),  # Profit/loss
                    "duration": sym_info.get("duration", ""),  # Optional
                    "open_time": sym_info.get("open_time"),
                    "close_time": sym_info.get("close_time"),
                    "details": {
                        "sl_pips": sym_info.get("sl_pips"),
                        "tp_pips": sym_info.get("tp_pips"),
                    }
                }
                ok = send_trade(trade_payload, idem=f"trade-{uuid.uuid4()}")
                if ok:
                    logging.info(f"[main] Trade for {sym} sent to /trades")
                else:
                    logging.warning(f"[main] Failed to send trade for {sym}")


        total_reward = sum(info["reward"] for info in step_results.values())
        done = False
        # --- Send executed trades to FastAPI server ---
                # --- Send executed signals to trade server (secure client) ---
        for sym, sym_info in last_info.items():
            if sym_info.get("executed", False):
                payload = {
                    "symbol": sym,
                    "action": sym_info.get("action_taken", "unknown"),
                    "sl_pips": sym_info.get("sl_pips", 0),
                    "tp_pips": sym_info.get("tp_pips", 0),
                    "lot_size": sym_info.get("lot_size", 0.0),
                    "details": {
                        "reward": sym_info.get("reward", 0.0),
                        "entry_price": sym_info.get("entry_price", 0.0),
                        "exit_price": sym_info.get("exit_price", 0.0),
                    }
                }
                # Only send if different from last sent for this symbol
                if payload != last_sent_signal.get(sym):
                    ok = send_signal(payload, idem=f"sig-{uuid.uuid4()}")
                    if ok:
                        logging.info(f"[main] Signal for {sym} sent to server")
                        last_sent_signal[sym] = payload
                    else:
                        logging.warning(f"[main] Failed to send signal for {sym}")

        # 10i) Save Arbiter training data
        try:
            filepath = arbiter_data_dir / f"arbiter_data_step_{step_counter}.npz"
            np.savez(
                filepath,
                onemin_logits=onemin_action_raw,
                medium_logits=medium_action_raw,
                long_logits=long_action_raw,
                history=np.stack([h.cpu().numpy() for h in history_buffer], axis=0),
                boot_time=(now - start_time),
                reward=total_reward,
                regime_context=regime_context_t.squeeze(0).cpu().numpy(),
            )
            step_counter += 1
        except Exception as e:
            logging.error(f"[main] Failed to save Arbiter training data: {e}", exc_info=True)
            # Optionally: set a failure counter or pause trading if repeated
            # For production, you might want to break/continue or alert here
            continue
        # 10j) Update history buffer
        context_per_asset: List[torch.Tensor] = []
        for i in range(n_symbols):
            ti = t_logits_t[0, i] if ENABLE_ONEMIN else torch.zeros(10, device=device)
            mi = m_logits_t[0, i] if ENABLE_MEDIUM else torch.zeros(10, device=device)
            li = l_logits_t[0, i] if ENABLE_LONG   else torch.zeros(10, device=device)
            extras = torch.tensor([now - start_time, total_reward, float(done)], dtype=torch.float32, device=device)
            ctx = torch.cat([ti, mi, li, extras], dim=0)  # 10 + 10 + 10 + 3 = 33 per asset
            context_per_asset.append(ctx)
        context_flat = torch.cat(context_per_asset, dim=0)
        history_buffer.append(context_flat)

        # 10k) Log transitions & errors (use the results we already have)
        obs_for_log = {}
        action_for_log = {}
        for sym in symbols:
            if ENABLE_ONEMIN:
                obs_for_log[sym] = build_onemin_obs(env, [sym], ONEMIN_OBS_WINDOW)
            elif ENABLE_MEDIUM:
                obs_for_log[sym] = build_medium_obs(env, [sym], medium_window)
            elif ENABLE_LONG:
                obs_for_log[sym] = build_long_obs(env, [sym], long_window)
            else:
                obs_for_log[sym] = None  # all policies disabled

            action_for_log[sym] = final_action_dict.get(sym, {}).get("action", [0.0] * 10)

        # 10l) Periodically save models (only the enabled ones)
        if (now - last_save) >= SAVE_INTERVAL:
            try:
                if ENABLE_ONEMIN and onemin_manager is not None:
                    onemin_manager.model.save(str(MODELS_DIR / "onemin_policy.zip"))
                if ENABLE_MEDIUM and medium_manager is not None:
                    medium_manager.model.save(str(MODELS_DIR / "medium_policy.zip"))
                if ENABLE_LONG and long_manager is not None:
                    long_manager.model.save(str(MODELS_DIR / "long_policy.zip"))
                torch.save(arbiter.state_dict(), arbiter_model_path)
                logging.info("[main] Saved enabled models and Arbiter weights to disk.")
                last_save = now
            except Exception as e:
                logging.error(f"[main] Failed to save one or more model files: {e}", exc_info=True)
                continue
        # 10m) Sleep briefly
        time.sleep(0.01)


if __name__ == "__main__":
    main()
