# live_env.py
from __future__ import annotations
import atexit
import logging
import logging.handlers
import threading
import time
import uuid
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import MetaTrader5 as mt5
from gymnasium.utils import seeding
import pandas as pd
from data.forex_data_system import ForexDataSystem
from config.settings import (
    BASE_DIR,
    LIVE_OBS_WINDOW,
    INITIAL_BALANCE,
    LOG_LEVEL,
    MAX_OPEN_TRADES,
    MEDIUM_OBS_WINDOW,
    LONG_OBS_WINDOW,
    ONEMIN_OBS_WINDOW,
    DEBUG_MODE
)
from utils.logging_utils import log_event
from utils.reward_utils import RewardFunction
from utils.file_utils import log_transition
EPS = 1e-8
TICK_FETCH_INTERVAL = 0.1
MT5_RETRY_ATTEMPTS = 3
MT5_RETRY_DELAY = 0.5
MIN_TRADE_VOLUME = 0.01

# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("LiveTradeEnv")
logger.setLevel(LOG_LEVEL)
log_dir = Path(BASE_DIR) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
rotator = logging.handlers.TimedRotatingFileHandler(
    filename=log_dir / "live_env.log",
    when="midnight",
    backupCount=7,
)
rotator.setLevel(LOG_LEVEL)
rotator.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(rotator)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)
# ────────────────────────────────────────────────────────────────────────────────

def _mt5_call(func: Callable, *args, **kwargs):
    """
    Wrap an MT5 call (except order_send) with retry logic.
    Raises RuntimeError if the call repeatedly returns None or raises.
    """
    for attempt in range(1, MT5_RETRY_ATTEMPTS + 1):
        try:
            res = func(*args, **kwargs)
            if res is None:
                raise RuntimeError(f"{func.__name__} returned None")
            return res
        except Exception as e:
            logger.warning(f"MT5 call {func.__name__} failed (attempt {attempt}): {e}")
            time.sleep(MT5_RETRY_DELAY)
    raise RuntimeError(f"MT5 call {func.__name__} failed after {MT5_RETRY_ATTEMPTS} attempts")

def _build_symbol_map() -> Dict[str, str]:
    """
    Build a map from base symbol (e.g. "EURUSD") to the actual
    tradable symbol name returned by mt5.symbols_get().
    Prefers symbols ending in ".s" if available, otherwise first tradable.
    """
    available = _mt5_call(mt5.symbols_get)
    groups: Dict[str, List[str]] = defaultdict(list)
    for sym in available:
        base = sym.name.split('.', 1)[0].upper()
        groups[base].append(sym.name)
    cleaned: Dict[str, str] = {}
    for base, names in groups.items():
        tradable = [n for n in names if not n.lower().endswith('.view')]
        if not tradable:
            continue
        s_pref = [n for n in tradable if n.lower().endswith('.s')]
        chosen = s_pref[0] if s_pref else tradable[0]
        cleaned[base] = chosen
        logger.debug(f"Mapped base '{base}' -> '{chosen}'")
    return cleaned

def _get_mt5_balance() -> float:
    """Fetch the current account balance via mt5.account_info()."""
    info = _mt5_call(mt5.account_info)
    return float(info.balance) if info else INITIAL_BALANCE

ORDER_FILLING_MODE = mt5.ORDER_FILLING_IOC
class LiveTradeEnv(gym.Env):
    """
    Live trading environment backed by MetaTrader5 and ForexDataSystem.
    """
    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        symbols: Union[str, List[str]],
        data_system: ForexDataSystem,
        
    ):
        super().__init__()
        self.medium_window = MEDIUM_OBS_WINDOW
        self.long_window = LONG_OBS_WINDOW
        self.last_modify_time = defaultdict(lambda: 0.0)    # ticket -> timestamp
        self.last_global_modify_time = 0.0                  # global last modify time
        self.modify_cooldown = 5.0                          # seconds per ticket
        self.global_modify_cooldown = 1.0                   # seconds between ANY modifies

        # Validate that MT5 trading is allowed on current account
        acct = _mt5_call(mt5.account_info)
        if not acct.trade_allowed:
            raise RuntimeError(f"MT5 trading not allowed: {acct}")
        logger.info(f"MT5 account {acct.login} on {acct.server}")

        # Resolve user‐passed symbols into actual MT5 names
        raw = [symbols] if isinstance(symbols, str) else list(symbols)
        self.symbol_map = _build_symbol_map()
        self.symbols: List[str] = []
        self.symbol_status: Dict[str, str] = {}
        for s in raw:
            base = s.replace('/', '').upper().split('.')[0]
            actual = self.symbol_map.get(base)
            if actual and _mt5_call(mt5.symbol_select, actual, True):
                self.symbols.append(actual)
                self.symbol_status[actual] = "ready"
                logger.info(f"Using symbol: {actual}")
            else:
                self.symbol_status[s] = "not tradable"
                logger.error(f"Symbol {s}->{actual} not tradable")

        # Initialize per_symbol_info to avoid AttributeError in main.py
        self.per_symbol_info: Dict[str, dict] = { sym: {} for sym in self.symbols }

        # Force ORDER_FILLING_IOC for all symbols (broker supports only IOC)
        self.filling_mode: Dict[str, int] = {}
        for sym in self.symbols:
            fm = ORDER_FILLING_MODE
            self.filling_mode[sym] = fm
            logger.info(f"{sym} filling mode forced to {'FOK' if fm == mt5.ORDER_FILLING_FOK else 'IOC'}")

        # Environment parameters
        self.window = LIVE_OBS_WINDOW
        self.max_trades_per_symbol = MAX_OPEN_TRADES

        # In‐memory tick queues for each symbol, storing (bid, ask, volume)
        # We’ll keep volume if MT5 tick data provides it, otherwise you can stash 0.
        self.tick_queues: Dict[str, deque] = {
            sym: deque(maxlen=self.window) for sym in self.symbols
        }
        # Also track a separate “mid‐price” queue so we can compute volatility
        self.mid_queues: Dict[str, deque] = {
            sym: deque(maxlen=self.window) for sym in self.symbols
        }

        # Trade bookkeeping
        self.trade_id_map: Dict[int, str] = {}
        self.open_positions: Dict[str, List[dict]] = defaultdict(list)
        self.closed_trades: Dict[str, List[dict]] = defaultdict(list)
        self.last_trade_time: Dict[str, float] = defaultdict(lambda: time.time())
        self.last_error: Dict[str, str] = defaultdict(str)
        self.last_tick_time: Dict[str, Optional[float]] = defaultdict(lambda: None)

        # Hook into ForexDataSystem for OHLC bars
        self.data_system = data_system
        if hasattr(data_system, "medium_window"):
            self.medium_window = data_system.medium_window
            # else keep the pre-set MEDIUM_OBS_WINDOW
        if hasattr(data_system, "long_window"):
            self.long_window = data_system.long_window
            # else keep LONG_OBS_WINDOW
        self.medium_deques = {sym: deque(maxlen=self.medium_window) for sym in self.symbols}
        self.long_deques   = {sym: deque(maxlen=self.long_window)   for sym in self.symbols}
        self.onemin_deques = {sym: deque(maxlen=ONEMIN_OBS_WINDOW) for sym in self.symbols}

        def _on_1min(sym: str, when, bar):
            # bar is a 4-tuple: (open, high, low, close)
            base = sym.split('.')[0].upper()
            actual_sym = next((e for e in self.symbols if e.split('.')[0].upper() == base), sym)
            self.onemin_deques[actual_sym].append(tuple(map(float, bar)))

        data_system.register_1min_handler(_on_1min)

        def _on_hour(sym: str, when, bar):
            base = sym.split('.')[0].upper()
            actual_sym = next((e for e in self.symbols if e.split('.')[0].upper() == base), sym)
            self.medium_deques[actual_sym].append(tuple(map(float, bar)))

        data_system.register_hourly_handler(_on_hour)

        def _on_day(sym: str, when, bar):
            base = sym.split('.')[0].upper()
            actual_sym = next((e for e in self.symbols if e.split('.')[0].upper() == base), sym)
            self.long_deques[actual_sym].append(tuple(map(float, bar)))

        data_system.register_daily_handler(_on_day)

                # --- Hydrate env OHLC deques from data_system history (immediate warm start) ---
        try:
            for env_sym in self.symbols:
                # Use the env's actual symbol string; ForexDataSystem resolved to actual names too
                m1_arr = self.data_system.get_minute_ohlc(env_sym, ONEMIN_OBS_WINDOW) or None
                if m1_arr is not None:
                    self.onemin_deques[env_sym].clear()
                    for (o, h, l, c) in m1_arr[-ONEMIN_OBS_WINDOW:]:
                        self.onemin_deques[env_sym].append((o, h, l, c))

                h1_arr = self.data_system.get_hourly_ohlc(env_sym, self.medium_window) or None
                if h1_arr is not None:
                    self.medium_deques[env_sym].clear()
                    for (o, h, l, c) in h1_arr[-self.medium_window:]:
                        self.medium_deques[env_sym].append((o, h, l, c))

                d1_arr = self.data_system.get_daily_ohlc(env_sym, self.long_window) or None
                if d1_arr is not None:
                    self.long_deques[env_sym].clear()
                    for (o, h, l, c) in d1_arr[-self.long_window:]:
                        self.long_deques[env_sym].append((o, h, l, c))
        except Exception as _e:
            logger.warning(f"Initial OHLC hydration from data_system failed: {_e}")
        # Reward functions per symbol
        self.reward_step_seconds = 60.0  # 1 “step” = 60s for reward timing
        self.reward_fns: Dict[str, RewardFunction] = {
            sym: RewardFunction(
                initial_balance=INITIAL_BALANCE,
                realized_R_weight=1.0,
                unrealized_weight=0.1,
                stats_alpha=0.1,
                quality_weight=0.6,
                inactivity_weight=0.02,
                inactivity_grace_steps=0,
                holding_threshold_steps=0,
                holding_penalty_per_step=0.01,
                risk_budget_R=2.0,
                overexposure_weight=0.1,
                conflict_weight=0.2,
                churn_count_threshold=3,
                churn_absR_threshold=0.15,
                churn_weight=0.1,
                component_clip=3.0,
                final_clip=5.0,
                smoothing_alpha=0.0,
                require_sl_tp=True,
            )
            for sym in self.symbols
        }

        # Thread locks and tick‐fetching threads
        self.trade_locks = {sym: threading.Lock() for sym in self.symbols}
        self._stop_event = threading.Event()
        self._tick_threads = [
            threading.Thread(target=self._tick_fetcher, args=(sym,), daemon=True)
            for sym in self.symbols
        ]
        for t in self._tick_threads:
            t.start()
        atexit.register(self.close)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _tick_fetcher(self, symbol: str):
        """Continuously fetch live tick data for one symbol."""
        logger.debug(f"Starting tick fetcher for {symbol}")
        while not self._stop_event.is_set():
            try:
                tick = _mt5_call(mt5.symbol_info_tick, symbol)
                if tick.bid is None or tick.ask is None:
                    self.last_error[symbol] = "invalid_tick"
                    logger.warning(f"{symbol} invalid tick data")
                else:
                    # Append (bid, ask). MT5 tick object may have volume in .volume
                    bid, ask = float(tick.bid), float(tick.ask)
                    vol = float(tick.volume) if hasattr(tick, "volume") else 0.0
                    self.tick_queues[symbol].append((bid, ask, vol))

                    # Maintain a mid‐price queue for volatility computation
                    mid = 0.5 * (bid + ask)
                    self.mid_queues[symbol].append(mid)

                    self.last_tick_time[symbol] = time.time()
            except Exception as e:
                self.last_error[symbol] = str(e)
                logger.warning(f"Tick fetch error for {symbol}: {e}")
            time.sleep(TICK_FETCH_INTERVAL)

    def close(self):
        """Stop all background threads cleanly."""
        self._stop_event.set()
        logger.info("LiveTradeEnv shutting down.")

    def _mask_illegal_actions(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a dictionary of proposed actions (per symbol), return a new dict
        where illegal action indices are masked to -inf so they cannot be chosen.
        """
        masked = {}
        for sym, act in action_dict.items():
            arr = np.array(act.get("action", []), dtype=np.float32)
            if sym not in self.symbols or len(self.tick_queues[sym]) < self.window:
                # If not tradable or not enough ticks, mask everything
                masked_arr = np.full_like(arr, -np.inf)
                masked[sym] = {**act, "action": masked_arr}
                continue
            longs = [p for p in self.open_positions[sym] if p["volume"] > 0]
            shorts = [p for p in self.open_positions[sym] if p["volume"] < 0]
            n_open = len(longs) + len(shorts)

            # Build mask for 8 actions:
            # 0: no-op, 1: buy, 2: sell, 3: close-long, 4: close-short,
            # 5: adjust-sl, 6: adjust-tp, 7: close-all
            valid = np.ones(8, dtype=bool)
            # If already have a long OR reached max trades, cannot buy
            if longs or n_open >= self.max_trades_per_symbol:
                valid[1] = False
            # If already have a short OR reached max trades, cannot sell
            if shorts or n_open >= self.max_trades_per_symbol:
                valid[2] = False
            # If no longs, cannot close a long
            if not longs:
                valid[3] = False
            # If no shorts, cannot close a short
            if not shorts:
                valid[4] = False
            # If no open trades, cannot adjust SL/TP or close‐all
            if n_open == 0:
                valid[5:8] = False

            # Apply mask: keep value if valid, else -inf
            masked_arr = np.where(valid, arr[:8], -np.inf).astype(np.float32)
            masked[sym] = {**act, "action": masked_arr}
        return masked
    def _modify_position(self, symbol, pos, new_sl=None, new_tp=None):
        ticket = pos["ticket"]
        now = time.time()
        # --- COOLDOWN PROTECTION ---
        if now - self.last_modify_time[ticket] < self.modify_cooldown:
            logger.warning(f"[{symbol}] Modify for ticket={ticket} skipped due to per-ticket cooldown.")
            return False
        if now - self.last_global_modify_time < self.global_modify_cooldown:
            logger.warning(f"[{symbol}] Global modify cooldown active. Skipping modify for ticket={ticket}.")
            return False

        info = _mt5_call(mt5.symbol_info, symbol)
        if not info:
            self.last_error[symbol] = "symbol_info_none"
            logger.warning(f"[{symbol}] No symbol info for SL/TP modify on ticket {ticket}")
            return False

        digits = info.digits
        tick_size = info.point
        stop_level = info.trade_stops_level * tick_size
        is_buy = pos["volume"] > 0

        max_retries = 3
        for attempt in range(max_retries):
            tick = _mt5_call(mt5.symbol_info_tick, symbol)
            if not tick:
                self.last_error[symbol] = "tick_unavailable"
                logger.warning(f"[{symbol}] No tick for SL/TP modify on ticket {ticket}")
                return False

            market_price = tick.bid if is_buy else tick.ask

            # --- SL calculation ---
            if new_sl is not None:
                if is_buy:
                    # BUY: SL must be below (market - stop_level)
                    min_sl = round(market_price - stop_level, digits)
                    sl = min(new_sl, min_sl)
                    # Ensure it's strictly below min_sl by at least 1 tick
                    if sl >= min_sl:
                        sl = round(min_sl - tick_size, digits)
                else:
                    # SELL: SL must be above (market + stop_level)
                    min_sl = round(market_price + stop_level, digits)
                    sl = max(new_sl, min_sl)
                    # Ensure it's strictly above min_sl by at least 1 tick
                    if sl <= min_sl:
                        sl = round(min_sl + tick_size, digits)
                sl = round(sl, digits)
            else:
                sl = pos.get("stop_loss")

            # --- TP calculation ---
            if new_tp is not None:
                if is_buy:
                    # BUY: TP must be above (market + stop_level)
                    min_tp = round(market_price + stop_level, digits)
                    tp = max(new_tp, min_tp)
                    if tp <= min_tp:
                        tp = round(min_tp + tick_size, digits)
                else:
                    # SELL: TP must be below (market - stop_level)
                    min_tp = round(market_price - stop_level, digits)
                    tp = min(new_tp, min_tp)
                    if tp >= min_tp:
                        tp = round(min_tp - tick_size, digits)
                tp = round(tp, digits)
            else:
                tp = pos.get("take_profit")

            # Prevent unnecessary modify if no change or SL/TP None
            if ((sl == pos.get("stop_loss") or sl is None) and
                (tp == pos.get("take_profit") or tp is None)):
                logger.info(f"[{symbol}] SL/TP unchanged for ticket={ticket}, skipping modify.")
                return True

            req = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   symbol,
                "position": ticket,
                "sl":       sl,
                "tp":       tp,
            }
            res = mt5.order_send(req)
            retcode = getattr(res, "retcode", None)
            comment = getattr(res, "comment", "")
            if res is not None and retcode == mt5.TRADE_RETCODE_DONE:
                if new_sl is not None:
                    pos["stop_loss"] = sl
                if new_tp is not None:
                    pos["take_profit"] = tp
                logger.info(f"[{symbol}] Modified SL/TP for ticket={ticket} to SL={sl}, TP={tp} on attempt {attempt+1}")
                self.last_error[symbol] = ""
                self.last_modify_time[ticket] = time.time()      # update per-ticket cooldown
                self.last_global_modify_time = time.time()       # update global cooldown
                return True
            else:
                self.last_error[symbol] = f"modify_reject [{retcode}]"
                logger.warning(f"[{symbol}] Failed to modify SL/TP for ticket={ticket} on attempt {attempt+1}: retcode={retcode}, SL={sl}, TP={tp}, comment={comment}")
                time.sleep(0.2)  # pace between attempts

        logger.error(f"[{symbol}] Could not modify SL/TP for ticket={ticket} after {max_retries} attempts")
        return False

    def reset(self) -> Dict[str, np.ndarray]:
    
        TICK_CSV_DIR = Path("logs/ticks/csv")
        OHLC_CSV_DIR = Path("logs/ohlc")

        for sym in self.symbols:
            self.closed_trades[sym].clear()
            self.last_trade_time[sym] = time.time()
            self.last_error[sym] = ""
            if sym in self.reward_fns:
                self.reward_fns[sym].reset()

            # TICKS
            tick_csv = TICK_CSV_DIR / f"{sym.split('.')[0]}_ticks.csv"
            self.tick_queues[sym].clear()
            self.mid_queues[sym].clear()
            if tick_csv.exists():
                try:
                    df_tick = pd.read_csv(tick_csv).tail(self.window)
                    for row in df_tick.itertuples(index=False):
                        self.tick_queues[sym].append((row.bid, row.ask, getattr(row, "vol", 0.0)))
                        mid = 0.5 * (row.bid + row.ask)
                        self.mid_queues[sym].append(mid)
                except Exception as e:
                    logger.warning(f"Could not hydrate tick data for {sym}: {e}")

            # HOURLY (medium)
            hourly_csv = OHLC_CSV_DIR / f"{sym.split('.')[0]}_hourly.csv"
            self.medium_deques[sym].clear()
            if hourly_csv.exists():
                try:
                    df_hour = pd.read_csv(hourly_csv).tail(self.medium_window)
                    for row in df_hour.itertuples(index=False):
                        self.medium_deques[sym].append((row.open, row.high, row.low, row.close))
                except Exception as e:
                    logger.warning(f"Could not hydrate hourly OHLC for {sym}: {e}")

            # DAILY (long)
            daily_csv = OHLC_CSV_DIR / f"{sym.split('.')[0]}_daily.csv"
            self.long_deques[sym].clear()
            if daily_csv.exists():
                try:
                    df_day = pd.read_csv(daily_csv).tail(self.long_window)
                    for row in df_day.itertuples(index=False):
                        self.long_deques[sym].append((row.open, row.high, row.low, row.close))
                except Exception as e:
                    logger.warning(f"Could not hydrate daily OHLC for {sym}: {e}")
            # 1-MINUTE (onemin)  <-- ADD THIS BLOCK
            onemin_csv = OHLC_CSV_DIR / f"{sym.split('.')[0]}_1min.csv"
            self.onemin_deques[sym].clear()
            if onemin_csv.exists():
                try:
                    df_onemin = pd.read_csv(onemin_csv).tail(ONEMIN_OBS_WINDOW)
                    for row in df_onemin.itertuples(index=False):
                        self.onemin_deques[sym].append((row.open, row.high, row.low, row.close))
                except Exception as e:
                    logger.warning(f"Could not hydrate 1min OHLC for {sym}: {e}")
        return {
            sym: self.get_observation(sym)
            for sym in self.symbols
            if self.get_observation(sym) is not None
        }

    def step(self, action_dict):
        masked_actions = self._mask_illegal_actions(action_dict)
        results = {}
        balance = _get_mt5_balance()
        now = time.time()

        # Synchronize open_positions with actual MT5 positions
        try:
            live_positions = _mt5_call(mt5.positions_get)
        except Exception as e:
            live_positions = []
            logger.warning(f"Failed to fetch positions: {e}")

        self.open_positions = defaultdict(list)
        for p in live_positions:
            if p.symbol in self.symbols:
                ticket = p.ticket
                trade_id = self.trade_id_map.get(ticket) or str(uuid.uuid4())
                self.trade_id_map[ticket] = trade_id

                # MT5: type 0=BUY, 1=SELL
                is_buy = (p.type == mt5.POSITION_TYPE_BUY)
                vol_signed = p.volume if is_buy else -p.volume
                trade_type = "long" if is_buy else "short"

                self.open_positions[p.symbol].append({
                    "trade_id":    trade_id,
                    "ticket":      ticket,
                    "symbol":      p.symbol,
                    "volume":      vol_signed,          # SIGNED volume (long=+, short=-)
                    "entry_price": float(p.price_open),
                    "stop_loss":   float(p.sl),
                    "take_profit": float(p.tp),
                    "open_time":   float(p.time),       # epoch seconds
                    "type":        int(p.type),         # 0/1 for convenience
                    "trade_type":  trade_type,          # "long"/"short" for reward
                })
        for sym, act in masked_actions.items():
            if sym not in self.symbols:
                continue
            executed = False
            action_taken = "none"
            trade_event_dict = {}
            n_closed_before = len(self.closed_trades[sym])


            arr = act.get("action", [])
            if len(arr) < 8 or np.all(np.isneginf(arr)):
                self.last_error[sym] = "no_valid_action"
                results[sym] = {
                    "reward":      0.0,
                    "balance":     balance,
                    "open_trades": len(self.open_positions[sym]),
                    "last_error":  self.last_error[sym],
                    "spread":      0.0,
                    "volume":      0.0,
                    "volatility":  0.0,
                }
                continue
            idx = int(np.nanargmax(arr))
            sl_pips = act.get("sl_pips", 0)
            tp_pips = act.get("tp_pips", 0)
            lot_size = max(float(act.get("lot_size", 0)), MIN_TRADE_VOLUME)
            bid, ask, _ = self.tick_queues[sym][-1]
            longs = [p for p in self.open_positions[sym] if p["volume"] > 0]
            shorts = [p for p in self.open_positions[sym] if p["volume"] < 0]
            n_open = len(longs) + len(shorts)

            # idx == 1: buy
            if idx == 1 and n_open < self.max_trades_per_symbol and not longs:
                self._place_order(sym, lot_size, bid, ask, sl_pips, tp_pips, "buy")
                executed = True
                action_taken = "buy"
                if self.open_positions[sym]:
                    last_open = self.open_positions[sym][-1]
                    trade_event_dict.update({
                        "trade_opened": True,
                        "entry_price": last_open.get("entry_price"),
                        "open_time": last_open.get("open_time"),
                        "volume": last_open.get("volume"),
                    })

            # idx == 2: sell
            elif idx == 2 and n_open < self.max_trades_per_symbol and not shorts:
                self._place_order(sym, lot_size, bid, ask, sl_pips, tp_pips, "sell")
                executed = True
                action_taken = "sell"
                if self.open_positions[sym]:
                    last_open = self.open_positions[sym][-1]
                    trade_event_dict.update({
                        "trade_opened": True,
                        "entry_price": last_open.get("entry_price"),
                        "open_time": last_open.get("open_time"),
                        "volume": last_open.get("volume"),
                    })

            # idx == 3: close_long
            elif idx == 3 and longs:
                self._close_position(sym, longs[-1], "close_long")
                executed = True
                action_taken = "close_long"
                if self.closed_trades[sym]:
                    last_trade = self.closed_trades[sym][-1]
                    trade_event_dict.update({
                        "trade_closed": True,
                        "side": "buy",
                        "entry_price": last_trade.get("entry_price"),
                        "exit_price": last_trade.get("exit_price"),
                        "open_time": last_trade.get("open_time"),
                        "close_time": last_trade.get("close_time"),
                        "pnl": last_trade.get("pnl"),
                    })

            # idx == 4: close_short
            elif idx == 4 and shorts:
                self._close_position(sym, shorts[-1], "close_short")
                executed = True
                action_taken = "close_short"
                if self.closed_trades[sym]:
                    last_trade = self.closed_trades[sym][-1]
                    trade_event_dict.update({
                        "trade_closed": True,
                        "side": "sell",
                        "entry_price": last_trade.get("entry_price"),
                        "exit_price": last_trade.get("exit_price"),
                        "open_time": last_trade.get("open_time"),
                        "close_time": last_trade.get("close_time"),
                        "pnl": last_trade.get("pnl"),
                    })

            # idx == 5: adjust SL
            elif idx == 5 and n_open > 0:
                p = self.open_positions[sym][-1]
                adj = self._pips_to_price(sym, sl_pips)
                new_sl = (p["entry_price"] - adj) if p["volume"] > 0 else (p["entry_price"] + adj)
                self._modify_position(sym, p, new_sl=new_sl)
                executed = True
                action_taken = "adjust_sl"
                trade_event_dict["sl_modified"] = True

            # idx == 6: adjust TP
            elif idx == 6 and n_open > 0:
                p = self.open_positions[sym][-1]
                adj = self._pips_to_price(sym, tp_pips)
                new_tp = (p["entry_price"] + adj) if p["volume"] > 0 else (p["entry_price"] - adj)
                self._modify_position(sym, p, new_tp=new_tp)
                executed = True
                action_taken = "adjust_tp"
                trade_event_dict["tp_modified"] = True

            # idx == 7: close_all
            elif idx == 7 and n_open > 0:
                self._close_all_positions(sym)
                executed = True
                action_taken = "close_all"
                # If positions were closed, try to summarize most recent close
                if self.closed_trades[sym]:
                    last_trade = self.closed_trades[sym][-1]
                    trade_event_dict.update({
                        "trade_closed": True,
                        "side": "buy" if last_trade.get("trade_type") == "long" else "sell",
                        "entry_price": last_trade.get("entry_price"),
                        "exit_price": last_trade.get("exit_price"),
                        "open_time": last_trade.get("open_time"),
                        "close_time": last_trade.get("close_time"),
                        "pnl": last_trade.get("pnl"),
                    })
            # idx == 0: no-op
            else:
                if DEBUG_MODE:
                    logging.info(f"[{sym}] No-op or waiting/observing idx={idx}")

            # Compute reward for this symbol
            mid = 0.5 * (bid + ask)
            unreal = sum(p["volume"] * (mid - p["entry_price"]) for p in self.open_positions[sym])
            equity = balance + unreal
            elapsed_steps = int(max(0.0, now - self.last_trade_time[sym]) // self.reward_step_seconds)

            # Ensure holding_time (in steps) exists for each open position
            for p in self.open_positions[sym]:
                ot = float(p.get("open_time", now))
                p["holding_time"] = int(max(0.0, now - ot) // self.reward_step_seconds)

            closed_now = self.closed_trades[sym][n_closed_before:]  # only those closed during this step
            rw = self.reward_fns[sym](
                closed_trades=closed_now,
                open_trades=self.open_positions[sym],
                account_balance=equity,
                unrealized_pnl=unreal,
                time_since_last_trade=elapsed_steps,
            ).item()
            spread = float(ask - bid)
            mids = list(self.mid_queues[sym])
            if len(mids) >= 2:
                diffs = np.diff(np.array(mids, dtype=np.float64))
                volatility = float(np.std(diffs, ddof=0))
            else:
                volatility = 0.0
            latest_volume = float(self.tick_queues[sym][-1][2]) if self.tick_queues[sym] else 0.0

            # --- Generic results dict
            results[sym] = {
                "reward":      float(rw),
                "balance":     equity,
                "open_trades": len(self.open_positions[sym]),
                "last_error":  self.last_error[sym],
                "spread":      spread,
                "volume":      latest_volume,
                "volatility":  volatility,
                "executed":    executed,
                "action_taken": action_taken,
                "sl_pips":     sl_pips,
                "tp_pips":     tp_pips,
                "lot_size":    lot_size,
                "entry_price": ask if action_taken == "buy" else bid if action_taken == "sell" else None,

            }
            # --- Overlay trade event info if present
            results[sym].update(trade_event_dict)

        for sym in self.symbols:
            if sym not in results:
                results[sym] = {
                    "reward": 0.0,
                    "balance": _get_mt5_balance(),
                    "open_trades": len(self.open_positions[sym]),
                    "last_error": "step_not_processed",
                    "spread": 0.0,
                    "volume": 0.0,
                    "volatility": 0.0,
                    "executed": False,
                    "action_taken": "none",
                    "sl_pips": 0,
                    "tp_pips": 0,
                    "lot_size": 0,
                    "entry_price": None
                }

        return results

    # -- Helper method implementations --
    def _close_all_positions(self, symbol):
        """
        Close all open positions for a symbol by sending counter-orders via MT5.
        Each close operation checks for broker confirmation and logs accordingly.
        """
        if symbol not in self.open_positions or not self.open_positions[symbol]:
            logger.info(f"[{symbol}] No open positions to close.")
            return

        # Work on a copy since closing modifies the list
        positions = list(self.open_positions[symbol])

        for pos in positions:
            # Use latest bid for closing longs, ask for closing shorts
            if not self.tick_queues[symbol]:
                logger.warning(f"[{symbol}] No tick data available to close position.")
                continue

            bid, ask, _ = self.tick_queues[symbol][-1]
            price = bid if pos["volume"] > 0 else ask

            before_count = len(self.open_positions[symbol])
            self._close_position(symbol, pos, "close_all")
            after_count = len(self.open_positions[symbol])

            # Confirm the close (for robust automation)
            if after_count < before_count:
                logger.info(f"[{symbol}] Closed position {pos.get('ticket', '')} successfully in 'close_all'.")
            else:
                logger.warning(f"[{symbol}] Failed to close position {pos.get('ticket', '')} in 'close_all'.")

    def _place_order(self, symbol, lot_size, bid, ask, sl_pips, tp_pips, direction):
        info = _mt5_call(mt5.symbol_info, symbol)
        if not info:
            self.last_error[symbol] = "symbol_info_none"
            return
        if not _mt5_call(mt5.symbol_select, symbol, True):
            self.last_error[symbol] = "symbol_not_selected"
            return
        digits = info.digits
        tick_size = info.point
        stop_level_ticks = info.trade_stops_level
        price = ask if direction == "buy" else bid
        entry_offset = tick_size if direction == "buy" else -tick_size
        entry = round(price + entry_offset, digits)
        ticks_per_pip = 10 if digits in (5, 3) else 1
        sl_distance = sl_pips * ticks_per_pip * tick_size
        tp_distance = tp_pips * ticks_per_pip * tick_size

        if direction == "buy":
            raw_sl = entry - sl_distance
            raw_tp = entry + tp_distance
        else:
            raw_sl = entry + sl_distance
            raw_tp = entry - tp_distance

        sl_price = round(raw_sl, digits)
        tp_price = round(raw_tp, digits)
        min_dist_price = stop_level_ticks * tick_size

        if direction == "buy":
            if (entry - sl_price) < min_dist_price:
                sl_price = round(entry - min_dist_price, digits)
            if (tp_price - entry) < min_dist_price:
                tp_price = round(entry + min_dist_price, digits)
        else:
            if (sl_price - entry) < min_dist_price:
                sl_price = round(entry + min_dist_price, digits)
            if (entry - tp_price) < min_dist_price:
                tp_price = round(entry - min_dist_price, digits)

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot_size,
            "type":         (mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL),
            "price":        entry,
            "sl":           sl_price,
            "tp":           tp_price,
            "deviation":    100,
            "magic":        42,
            "comment":      "",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": ORDER_FILLING_MODE,

        }

        res = mt5.order_send(req)
        if res is None or not hasattr(res, "retcode") or res.retcode != mt5.TRADE_RETCODE_DONE:
            self.last_error[symbol] = f"order_reject [{getattr(res,'retcode','None')}]"
            return
        self.last_trade_time[symbol] = time.time()
        ticket = getattr(res, "order", None) or getattr(res, "ticket", None)
        trade_id = str(uuid.uuid4())
        if ticket is not None:
            self.trade_id_map[ticket] = trade_id
        self.open_positions[symbol].append({
            "trade_id":    trade_id,
            "ticket":      ticket,
            "volume":      lot_size if direction == "buy" else -lot_size,
            "entry_price": entry,
            "stop_loss":   sl_price,
            "take_profit": tp_price,
            "open_time":   time.time(),
            "trade_type":  "long" if direction == "buy" else "short",
            "type":        0 if direction == "buy" else 1,

        })
        self.last_error[symbol] = ""
        logger.info(f"[{symbol}] Placed order ticket={ticket}, vol={lot_size}, entry={entry}, SL={sl_price}, TP={tp_price}")

    def _close_position(self, symbol, pos, reason):
        ticket = pos.get("ticket")
        volume = abs(pos.get("volume", 0))
        pos_type = pos.get("type", None)  # should be 0 or 1 (BUY or SELL)
        if pos_type is None:
            logger.error(f"[{symbol}] ERROR: Missing 'type' in position dict: {pos}")
            self.last_error[symbol] = "missing_type"
            return

        info = mt5.symbol_info(symbol)
        if not info:
            logger.error(f"[{symbol}] Symbol info not found!")
            self.last_error[symbol] = "symbol_info_missing"
            return

        volume_min = info.volume_min
        volume_step = info.volume_step
        precision = str(volume_step)[::-1].find('.')
        volume = round(max(volume, volume_min), precision)

        if not mt5.symbol_select(symbol, True):
            logger.error(f"[{symbol}] Symbol not selected!")
            self.last_error[symbol] = "symbol_not_selected"
            return

        def _wait_for_context(timeout=5):
            start = time.time()
            while time.time() - start < timeout:
                last_error = mt5.last_error()
                if last_error[0] == 0:
                    return True
                time.sleep(0.25)
            return False

        _wait_for_context(timeout=5)
        time.sleep(0.2)

        # Correct close type logic (0 = BUY, 1 = SELL)
        close_type = mt5.ORDER_TYPE_BUY if pos_type == 1 else mt5.ORDER_TYPE_SELL

        digits = info.digits
        allowed_filling = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
        allowed_deviation = [100, 300, 500, 1000, 2000]

        # Price functions: first try price=0 (market), then best current bid/ask
        price_funcs = [
            lambda: 0,
            lambda: round(mt5.symbol_info_tick(symbol).bid, digits) if close_type == mt5.ORDER_TYPE_SELL else round(mt5.symbol_info_tick(symbol).ask, digits)
        ]

        logger.info(f"[{symbol}] Attempting close: ticket={ticket}, volume={volume}, type={close_type}, reason={reason}")
        logger.debug(f"Position dict: {pos}")

        # Main attempt block, with at least 20s pacing for this position
        start_time = time.time()
        closed = False
        for price_func in price_funcs:
            try:
                price = price_func()
            except Exception as ex:
                logger.warning(f"[{symbol}] Error fetching price: {ex}")
                price = 0

            for filling in allowed_filling:
                for deviation in allowed_deviation:
                    # Confirm the position still exists
                    positions = mt5.positions_get(ticket=ticket)
                    if not positions or not any(p.ticket == ticket for p in positions):
                        logger.info(f"[{symbol}] Position {ticket} no longer exists.")
                        self.last_error[symbol] = "position_not_found"
                        closed = True
                        break

                    req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": volume,
                        "type": close_type,
                        "position": ticket,
                        "price": price,
                        "deviation": deviation,
                        "magic": 42,
                        "comment": f"close_{reason}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": filling,
                    }
                    res = mt5.order_send(req)
                    retcode = getattr(res, "retcode", None)
                    comment = getattr(res, "comment", None)
                    logger.info(f"[{symbol}] Close attempt ticket={ticket} price={price} filling={filling} deviation={deviation} retcode={retcode} comment={comment}")

                    # MT5 closes return 10009 for partial, 10008 for requote, 10013 for invalid request
                    if retcode == mt5.TRADE_RETCODE_DONE or retcode == 10009:
                        logger.info(f"[{symbol}] Closed position {ticket} successfully.")
                        self.last_error[symbol] = ""
                        closed = True

                        # --- append closed trade + remove from open_positions ---
                        try:
                            tick_now = _mt5_call(mt5.symbol_info_tick, symbol)
                            # long if signed volume > 0 (we set signed volumes elsewhere)
                            is_long = float(pos.get("volume", 0.0)) > 0.0
                            exit_price = float(tick_now.bid if is_long else tick_now.ask)
                            entry_price = float(pos.get("entry_price", exit_price))
                            vol_abs = abs(float(pos.get("volume", 0.0)))

                            # PnL in price-units * lots (consistent with reward’s normalization)
                            pnl = (exit_price - entry_price) * vol_abs if is_long else (entry_price - exit_price) * vol_abs

                            # Ensure trade_type is "long"/"short" (reward expects these strings)
                            trade_type = "long" if is_long else "short"

                            self.closed_trades[symbol].append({
                                "symbol":      symbol,
                                "trade_type":  trade_type,
                                "volume":      vol_abs,
                                "entry_price": entry_price,
                                "exit_price":  exit_price,
                                "open_time":   float(pos.get("open_time", time.time())),
                                "close_time":  time.time(),
                                "pnl":         float(pnl),
                                "stop_loss":   float(pos.get("stop_loss", 0.0)),
                                "take_profit": float(pos.get("take_profit", 0.0)),
                            })

                            # Remove the position from our open list by ticket
                            self.open_positions[symbol] = [
                                q for q in self.open_positions[symbol] if q.get("ticket") != ticket
                            ]
                            # Reset inactivity timer for reward
                            self.last_trade_time[symbol] = time.time()
                        except Exception as _e:
                            logger.warning(f"[{symbol}] Closed position bookkeeping failed: {_e}")

                        break

                    time.sleep(0.5)  # pace between sub-attempts
                if closed:
                    break
            if closed:
                break

        # Ensure 20s pacing between close attempts for this position
        elapsed = time.time() - start_time
        if elapsed < 20:
            time.sleep(20 - elapsed)

        if not closed:
            logger.error(f"[{symbol}] FAILED to close position {ticket}.")
            self.last_error[symbol] = f"close_reject"

    def _adjust_sl(self, symbol, pos, new_sl):
        ticket = pos["ticket"]
        tp = pos["take_profit"]
        req = {
            "action":       mt5.TRADE_ACTION_SLTP,
            "symbol":       symbol,
            "position":     ticket,
            "sl":           new_sl,
            "tp":           tp,
        }
        res = mt5.order_send(req)
        if res is None or not hasattr(res, "retcode") or res.retcode != mt5.TRADE_RETCODE_DONE:
            self.last_error[symbol] = f"adjust_sl_reject [{getattr(res,'retcode','None')}]"
            return
        pos["stop_loss"] = new_sl
        logger.info(f"[{symbol}] Adjusted SL for ticket={ticket} to {new_sl}")

    def _adjust_tp(self, symbol, pos, new_tp):
        ticket = pos["ticket"]
        sl = pos["stop_loss"]
        req = {
            "action":       mt5.TRADE_ACTION_SLTP,
            "symbol":       symbol,
            "position":     ticket,
            "sl":           sl,
            "tp":           new_tp,
        }
        res = mt5.order_send(req)
        if res is None or not hasattr(res, "retcode") or res.retcode != mt5.TRADE_RETCODE_DONE:
            self.last_error[symbol] = f"adjust_tp_reject [{getattr(res,'retcode','None')}]"
            return
        pos["take_profit"] = new_tp
        logger.info(f"[{symbol}] Adjusted TP for ticket={ticket} to {new_tp}")

    def get_observation(self, symbol: str) -> Optional[np.ndarray]:
        """
        Return observation for a symbol: flattened tick window plus net open position plus elapsed seconds.
        """
        if symbol not in self.symbols or len(self.tick_queues[symbol]) < self.window:
            return None

        # 1) Flatten last self.window ticks: [(bid, ask, vol), ...]
        arr = np.array(self.tick_queues[symbol], dtype=np.float32)  # shape = (window, 3)
        flat = arr.flatten()  # shape = (3*window,)

        # 2) Net position (sum of volumes)
        pos_sum = float(sum(p["volume"] for p in self.open_positions[symbol]))  # scalar

        # 3) Elapsed seconds since last trade on this symbol
        elapsed = float(time.time() - self.last_trade_time[symbol])  # scalar

        # Return concatenated array: [3*window values] + [pos_sum] + [elapsed]
        return np.concatenate([flat, [pos_sum], [elapsed]]).astype(np.float32)
    
    def get_onemin_observation(self, symbol: str) -> Optional[np.ndarray]:
        dq = self.onemin_deques[symbol]
        if len(dq) < ONEMIN_OBS_WINDOW:
            return None
        arr = np.array(dq, dtype=np.float32)
        flat = arr.flatten()
        elapsed = float(time.time() - self.last_trade_time[symbol])
        return np.concatenate([flat, [elapsed]]).astype(np.float32)

    def get_status(self) -> Dict[str, Any]:
        """
        Return diagnostic info per symbol.
        """
        return {
            s: {
                "tick_buffer":   len(self.tick_queues[s]),
                "onemin_buffer": len(self.onemin_deques[s]) if s in self.onemin_deques else 0,
                "medium_buffer": len(self.medium_deques[s]) if s in self.medium_deques else 0,
                "long_buffer":   len(self.long_deques[s]) if s in self.long_deques else 0,
                "open_trades":   len(self.open_positions[s]),
                "closed_trades": len(self.closed_trades[s]),
                "last_error":    self.last_error[s],
            }
            for s in self.symbols
        }

    def _pips_to_price(self, symbol: str, pips: int) -> float:
        """
        Convert a number of pips into price units, based on symbol digits.
        """
        info = _mt5_call(mt5.symbol_info, symbol)
        if not info:
            return pips * 0.0001  # fallback

        d = info.digits
        # Standard FX:
        #  - 5 or 4 digits → pip = 0.0001
        #  - 3 or 2 digits → pip = 0.01 (JPY-style)
        if d in (5, 4):
            return pips * 0.0001
        elif d in (3, 2):
            return pips * 0.01
        else:
            # Exotic fallback: assume 10 points per pip
            return pips * (info.point * 10.0)
        
    def hydrate_from_logs(self):
            TICK_CSV_DIR = Path("logs/ticks/csv")
            OHLC_CSV_DIR = Path("logs/ohlc")

            for sym in self.symbols:
                # TICKS
                tick_csv = TICK_CSV_DIR / f"{sym.split('.')[0]}_ticks.csv"
                if tick_csv.exists():
                    try:
                        df_tick = pd.read_csv(tick_csv).tail(self.window)
                        self.tick_queues[sym].clear()
                        for row in df_tick.itertuples(index=False):
                            self.tick_queues[sym].append((row.bid, row.ask, getattr(row, "vol", 0.0)))
                            mid = 0.5 * (row.bid + row.ask)
                            self.mid_queues[sym].append(mid)
                    except Exception as e:
                        logger.warning(f"Could not hydrate tick data for {sym}: {e}")

                # HOURLY
                hourly_csv = OHLC_CSV_DIR / f"{sym.split('.')[0]}_hourly.csv"
                if hourly_csv.exists():
                    try:
                        df_hour = pd.read_csv(hourly_csv).tail(self.medium_window)
                        print(f"[{sym}] hourly rows in CSV: {len(df_hour)} columns: {list(df_hour.columns)}")
                        self.medium_deques[sym].clear()
                        for row in df_hour.itertuples(index=False):
                            self.medium_deques[sym].append((row.open, row.high, row.low, row.close))
                        print(f"[{sym}] medium_deques filled: {len(self.medium_deques[sym])}")
                    except Exception as e:
                        logger.warning(f"Could not hydrate hourly OHLC for {sym}: {e}")
                else:
                    print(f"[{sym}] HOURLY CSV NOT FOUND: {hourly_csv}")

                # DAILY
                daily_csv = OHLC_CSV_DIR / f"{sym.split('.')[0]}_daily.csv"
                if daily_csv.exists():
                    try:
                        df_day = pd.read_csv(daily_csv).tail(self.long_window)
                        self.long_deques[sym].clear()
                        for row in df_day.itertuples(index=False):
                            self.long_deques[sym].append((row.open, row.high, row.low, row.close))
                    except Exception as e:
                        logger.warning(f"Could not hydrate daily OHLC for {sym}: {e}")