# data/forex_data_system.py
from __future__ import annotations
import os
import time
import json
import threading
import pickle
import collections
import datetime as dt
from typing import Dict, Deque, Tuple, Optional, List, Callable

import numpy as np
import pandas as pd

from config.settings import BASE_DIR, LIVE_FOREX_PAIRS, SPREAD_COST

# Optional: API key pulled from env to avoid leaking in code/logs
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
DATA_CACHE_FILE = str((BASE_DIR / "data" / "ohlc_cache.pkl"))

# Tunables
INITIAL_MINUTE_DAYS: int = int(os.environ.get("INITIAL_MINUTE_DAYS", "2"))
INITIAL_HOURLY_YEARS: int = int(os.environ.get("INITIAL_HOURLY_YEARS", "3"))
INITIAL_DAILY_YEARS: int  = int(os.environ.get("INITIAL_DAILY_YEARS", "5"))
TD_CHUNK_DAYS: int = 208
TD_REQ_SLEEP_SEC: float = 7.5

# Cache dirs
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "symbol_cache"
M1_DIR = DATA_DIR / "minute"
H1_DIR = DATA_DIR / "hourly"
D1_DIR = DATA_DIR / "daily"
for _p in (DATA_DIR, CACHE_DIR, M1_DIR, H1_DIR, D1_DIR):
    _p.mkdir(parents=True, exist_ok=True)

OHLC = Tuple[float, float, float, float]  # (open, high, low, close)

def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("=X", "")

def _atomic_write_bytes(path: os.PathLike, data: bytes) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def _utcnow() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

class _BarAgg:
    """Incremental OHLC aggregator for a timeframe boundary."""
    def __init__(self, tf: str):
        self.tf = tf
        self.reset()

    @staticmethod
    def _bucket(timestamp: dt.datetime, tf: str) -> dt.datetime:
        if tf == "M1":
            return timestamp.replace(second=0, microsecond=0)
        if tf == "H1":
            return timestamp.replace(minute=0, second=0, microsecond=0)
        if tf == "D1":
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        raise ValueError(f"Unknown tf: {tf}")

    def reset(self):
        self.cur_bucket: Optional[dt.datetime] = None
        self.o = self.h = self.l = self.c = None

    def update(self, price: float, ts: dt.datetime) -> Tuple[Optional[dt.datetime], Optional[OHLC]]:
        bucket = self._bucket(ts, self.tf)
        closed: Optional[OHLC] = None
        closed_bucket: Optional[dt.datetime] = None

        if self.cur_bucket is None:
            self.cur_bucket = bucket
            self.o = self.h = self.l = self.c = price
            return None, None

        if bucket != self.cur_bucket:
            closed = (float(self.o), float(self.h), float(self.l), float(self.c))
            closed_bucket = self.cur_bucket
            self.cur_bucket = bucket
            self.o = self.h = self.l = self.c = price
        else:
            self.h = price if self.h is None else max(self.h, price)
            self.l = price if self.l is None else min(self.l, price)
            self.c = price

        return closed_bucket, closed

class ForexDataSystem:
    """
    Live/offline data engine with bar-close callbacks required by LiveTradeEnv.

    Public read API:
      - get_minute_ohlc(symbol, n) -> np.ndarray [n,4] or None
      - get_hourly_ohlc(symbol, n) -> np.ndarray [n,4] or None
      - get_daily_ohlc(symbol, n)  -> np.ndarray [n,4] or None
      - get_meta(symbol) -> {"spread": float, "volatility": float}
      - buffer_counts() -> warm-up counts

    Callback registration (what your env is calling):
      - register_1min_handler(fn)
      - register_hourly_handler(fn)    # alias: register_h1_handler
      - register_daily_handler(fn)     # alias: register_d1_handler

    Each callback is called as: fn(symbol: str, when: datetime, bar: OHLC)
    """
    def __init__(self, requested_symbols: List[str], use_mt5: bool = True):
        self.symbols = list(requested_symbols)
        self.use_mt5 = bool(use_mt5)

        # Rolling buffers
        self._m1: Dict[str, Deque[OHLC]] = {s: collections.deque(maxlen=50_000) for s in self.symbols}
        self._h1: Dict[str, Deque[OHLC]] = {s: collections.deque(maxlen=50_000) for s in self.symbols}
        self._d1: Dict[str, Deque[OHLC]] = {s: collections.deque(maxlen=50_000) for s in self.symbols}

        # Live meta
        self._spread: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self._volatility: Dict[str, float] = {s: 0.0 for s in self.symbols}

        # Aggregators
        self._agg_m1: Dict[str, _BarAgg] = {s: _BarAgg("M1") for s in self.symbols}
        self._agg_h1: Dict[str, _BarAgg] = {s: _BarAgg("H1") for s in self.symbols}
        self._agg_d1: Dict[str, _BarAgg] = {s: _BarAgg("D1") for s in self.symbols}

        # Callback registries
        self._m1_handlers: List[Callable[[str, dt.datetime, OHLC], None]] = []
        self._h1_handlers: List[Callable[[str, dt.datetime, OHLC], None]] = []
        self._d1_handlers: List[Callable[[str, dt.datetime, OHLC], None]] = []

        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="forex-data", daemon=True)

        self._session = None  # requests.Session for TD; created on demand

        self._init_backends()
        self._hydrate_from_cache()
        self._thread.start()

    # ---------------- Public read API ----------------

    def get_minute_ohlc(self, symbol: str, n: int) -> Optional[np.ndarray]:
        with self._lock:
            dq = self._m1[symbol]
            if len(dq) < n:
                return None
            return np.asarray(list(dq)[-n:], dtype=np.float32)

    def get_hourly_ohlc(self, symbol: str, n: int) -> Optional[np.ndarray]:
        with self._lock:
            dq = self._h1[symbol]
            if len(dq) < n:
                return None
            return np.asarray(list(dq)[-n:], dtype=np.float32)

    def get_daily_ohlc(self, symbol: str, n: int) -> Optional[np.ndarray]:
        with self._lock:
            dq = self._d1[symbol]
            if len(dq) < n:
                return None
            return np.asarray(list(dq)[-n:], dtype=np.float32)

    def get_meta(self, symbol: str) -> Dict[str, float]:
        with self._lock:
            return {"spread": float(self._spread[symbol]), "volatility": float(self._volatility[symbol])}

    def buffer_counts(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {
                s: {
                    "onemin_buffer": len(self._m1[s]),
                    "medium_buffer": len(self._h1[s]),
                    "long_buffer":   len(self._d1[s]),
                } for s in self.symbols
            }

    # ---------------- Callback registration ----------------

    def register_1min_handler(self, fn: Callable[[str, dt.datetime, OHLC], None]) -> None:
        with self._lock:
            if fn not in self._m1_handlers:
                self._m1_handlers.append(fn)

    def register_hourly_handler(self, fn: Callable[[str, dt.datetime, OHLC], None]) -> None:
        with self._lock:
            if fn not in self._h1_handlers:
                self._h1_handlers.append(fn)

    def register_daily_handler(self, fn: Callable[[str, dt.datetime, OHLC], None]) -> None:
        with self._lock:
            if fn not in self._d1_handlers:
                self._d1_handlers.append(fn)

    # Aliases (in case env uses different names)
    def register_h1_handler(self, fn: Callable[[str, dt.datetime, OHLC], None]) -> None:
        self.register_hourly_handler(fn)

    def register_d1_handler(self, fn: Callable[[str, dt.datetime, OHLC], None]) -> None:
        self.register_daily_handler(fn)

    # ---------------- Lifecycle ----------------

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=2.0)

    # ---------------- Backend init / warmup / cache ----------------

    def _init_backends(self):
        if self.use_mt5:
            try:
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    raise RuntimeError("MetaTrader5.initialize() failed")

                self._mt5 = mt5
                # Resolve broker-specific symbol names (e.g. EURUSD -> EURUSD.s)
                self._resolve_mt5_symbols()
                for s in self.symbols:
                    mt5.symbol_select(s, True)

                # >>> Warm up internal buffers from MT5 history <<<
                self._mt5_backfill()

            except Exception as e:
                print(f"[ForexDataSystem] MT5 init failed ({e}); falling back to offline Twelve Data.")
                self.use_mt5 = False
                self._mt5 = None

        else:
            self._mt5 = None

        if not self.use_mt5:
            self._warmup_offline()

    def _hydrate_from_cache(self):
        try:
            if os.path.exists(DATA_CACHE_FILE):
                with open(DATA_CACHE_FILE, "rb") as f:
                    blob = pickle.load(f)
                with self._lock:
                    for s in self.symbols:
                        for tf, target in (("M1", self._m1), ("H1", self._h1), ("D1", self._d1)):
                            arr = blob.get(s, {}).get(tf)
                            if isinstance(arr, list):
                                target[s].extend(arr[-5000:])
        except Exception:
            pass  # non-fatal

    def _persist_bar(self, symbol: str, tf: str, when: dt.datetime, bar: OHLC):
        fname = f"{_safe_symbol(symbol)}.csv"
        path = {"M1": M1_DIR, "H1": H1_DIR, "D1": D1_DIR}[tf] / fname
        line = f"{when.isoformat()},{bar[0]},{bar[1]},{bar[2]},{bar[3]}\n"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def _snapshot_pickle(self):
        try:
            snap: Dict[str, Dict[str, List[OHLC]]] = {}
            with self._lock:
                for s in self.symbols:
                    snap[s] = {
                        "M1": list(self._m1[s])[-5000:],
                        "H1": list(self._h1[s])[-5000:],
                        "D1": list(self._d1[s])[-5000:],
                    }
            _atomic_write_bytes(DATA_CACHE_FILE, pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            pass

    # ---------------- Twelve Data helpers (offline mode) ----------------

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "Nister-ForexDataSystem/1.0"})
        return self._session

    def _td_download(self, symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if not TWELVE_DATA_API_KEY:
            raise RuntimeError("TWELVE_DATA_API_KEY is not set for offline mode.")
        url = (
            "https://api.twelvedata.com/time_series"
            f"?symbol={_safe_symbol(symbol)}&interval={interval}"
            f"&start_date={start.strftime('%Y-%m-%d %H:%M:%S')}"
            f"&end_date={end.strftime('%Y-%m-%d %H:%M:%S')}"
            f"&apikey={TWELVE_DATA_API_KEY}&format=JSON"
        )
        s = self._get_session()
        r = s.get(url, timeout=30)
        data = r.json()
        if "values" not in data:
            raise RuntimeError(f"Twelve Data error for {symbol}@{interval}: {data}")
        df = pd.DataFrame(data["values"])
        df.columns = [c.lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        cols = ["open", "high", "low", "close"]
        return df[cols].astype(float)

    def _td_backfill(self, symbol: str, total_days: int, interval: str) -> pd.DataFrame:
        end = _utcnow()
        out = []
        remaining = int(total_days)
        while remaining > 0:
            span = min(TD_CHUNK_DAYS, remaining)
            start = end - dt.timedelta(days=span)
            try:
                df = self._td_download(symbol, interval, start, end)
                out.append(df)
            except Exception as e:
                print(f"[ForexDataSystem] TD backfill failed ({symbol} {interval}): {e}")
                break
            end = start
            remaining -= span
            time.sleep(TD_REQ_SLEEP_SEC)
        if not out:
            return pd.DataFrame(columns=["open","high","low","close"])
        return pd.concat(out).sort_index()

    def _warmup_offline(self):
        for s in self.symbols:
            try:
                h1_df = self._td_backfill(s, total_days=int(INITIAL_HOURLY_YEARS * 365), interval="1h")
                with self._lock:
                    for _, row in h1_df.iterrows():
                        self._h1[s].append((float(row.open), float(row.high), float(row.low), float(row.close)))
            except Exception as e:
                print(f"[ForexDataSystem] Offline H1 warmup error for {s}: {e}")

            try:
                d1_df = self._td_backfill(s, total_days=int(INITIAL_DAILY_YEARS * 365), interval="1day")
                with self._lock:
                    for _, row in d1_df.iterrows():
                        self._d1[s].append((float(row.open), float(row.high), float(row.low), float(row.close)))
            except Exception as e:
                print(f"[ForexDataSystem] Offline D1 warmup error for {s}: {e}")

            if INITIAL_MINUTE_DAYS > 0:
                try:
                    m1_df = self._td_backfill(s, total_days=int(INITIAL_MINUTE_DAYS), interval="1min")
                    with self._lock:
                        for _, row in m1_df.iterrows():
                            self._m1[s].append((float(row.open), float(row.high), float(row.low), float(row.close)))
                except Exception as e:
                    print(f"[ForexDataSystem] Offline M1 warmup error for {s}: {e}")

            with self._lock:
                closes = [b[3] for b in list(self._h1[s])[-24:]]
                if len(closes) >= 5:
                    rets = np.diff(np.log(np.asarray(closes)))
                    self._volatility[s] = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0

    # ---------------- MT5 backfill (history load) ----------------

    def _resolve_mt5_symbols(self):
        """Map requested base symbols to actual broker symbols (e.g., EURUSD -> EURUSD.s)."""
        try:
            mt5 = self._mt5
            available = mt5.symbols_get()
            by_base = {}
            for sym in available:
                base = sym.name.split('.', 1)[0].upper()
                by_base.setdefault(base, []).append(sym.name)
            resolved = []
            for s in list(self.symbols):
                base = s.replace('/', '').split('.', 1)[0].upper()
                names = by_base.get(base, [])
                if not names:
                    continue
                # Prefer suffix ".s", otherwise first non ".view"
                tradable = [n for n in names if not n.lower().endswith('.view')]
                preferred = [n for n in tradable if n.lower().endswith('.s')]
                chosen = preferred[0] if preferred else (tradable[0] if tradable else names[0])
                resolved.append(chosen)
            if resolved:
                self.symbols = resolved
        except Exception:
            # If anything goes wrong, keep the originally requested names
            pass

    def _mt5_backfill(self):
        """
        Load historical bars from MT5 to warm up hourly/daily (and minute) buffers.
        Uses INITIAL_* constants for depth.
        """
        mt5 = self._mt5
        if mt5 is None:
            return

        # how many bars to request
        n_h1 = max(1, int(INITIAL_HOURLY_YEARS * 365 * 24))
        n_d1 = max(1, int(INITIAL_DAILY_YEARS  * 365))
        n_m1 = max(1, int(INITIAL_MINUTE_DAYS  * 1440))

        def _copy(symbol, timeframe, count):
            try:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
                if rates is None:
                    return []
                return list(rates)
            except Exception:
                return []

        for s in self.symbols:
            # Select symbol
            try:
                mt5.symbol_select(s, True)
            except Exception:
                continue

            # H1
            if n_h1 > 0:
                rates_h1 = _copy(s, mt5.TIMEFRAME_H1, n_h1)
                with self._lock:
                    self._h1[s].clear()
                    for r in rates_h1:
                        self._h1[s].append((float(r['open']), float(r['high']), float(r['low']), float(r['close'])))
                self._update_vol(s)

            # D1
            if n_d1 > 0:
                rates_d1 = _copy(s, mt5.TIMEFRAME_D1, n_d1)
                with self._lock:
                    self._d1[s].clear()
                    for r in rates_d1:
                        self._d1[s].append((float(r['open']), float(r['high']), float(r['low']), float(r['close'])))

            # M1 (optional, if you want onemin history too)
            if n_m1 > 0:
                rates_m1 = _copy(s, mt5.TIMEFRAME_M1, n_m1)
                with self._lock:
                    self._m1[s].clear()
                    for r in rates_m1:
                        self._m1[s].append((float(r['open']), float(r['high']), float(r['low']), float(r['close'])))


    # ---------------- Main data loop ----------------

    def _run(self):
        last_snap = time.time()
        while not self._stop_evt.is_set():
            try:
                if self.use_mt5 and getattr(self, "_mt5", None) is not None:
                    self._loop_mt5()
                else:
                    self._loop_offline_sim()
            except Exception:
                time.sleep(0.5)

            if time.time() - last_snap > 60.0:
                self._snapshot_pickle()
                last_snap = time.time()

            time.sleep(0.05)

    # ---------------- MT5 live loop ----------------

    def _loop_mt5(self):
        mt5 = self._mt5
        for s in self.symbols:
            tick = mt5.symbol_info_tick(s)
            if tick is None:
                continue
            ts = dt.datetime.fromtimestamp(tick.time, tz=dt.timezone.utc)
            bid = float(tick.bid)
            ask = float(tick.ask if tick.ask else bid + SPREAD_COST)
            mid = (bid + ask) * 0.5

            with self._lock:
                self._spread[s] = max(ask - bid, 0.0)

                bkt_m, bar_m = self._agg_m1[s].update(mid, ts)
                if bar_m and bkt_m:
                    self._m1[s].append(bar_m)
                    self._persist_bar(s, "M1", bkt_m, bar_m)
                    self._emit_callbacks("M1", s, bkt_m, bar_m)

                    bkt_h, bar_h = self._agg_h1[s].update(bar_m[3], bkt_m)
                    if bar_h and bkt_h:
                        self._h1[s].append(bar_h)
                        self._persist_bar(s, "H1", bkt_h, bar_h)
                        self._update_vol(s)
                        self._emit_callbacks("H1", s, bkt_h, bar_h)

                    bkt_d, bar_d = self._agg_d1[s].update(bar_m[3], bkt_m)
                    if bar_d and bkt_d:
                        self._d1[s].append(bar_d)
                        self._persist_bar(s, "D1", bkt_d, bar_d)
                        self._emit_callbacks("D1", s, bkt_d, bar_d)

    def _update_vol(self, symbol: str):
        closes = [b[3] for b in list(self._h1[symbol])[-24:]]
        if len(closes) >= 5:
            rets = np.diff(np.log(np.asarray(closes)))
            self._volatility[symbol] = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0

    # ---------------- Offline "sim" loop ----------------

    def _loop_offline_sim(self):
        now = _utcnow()
        for s in self.symbols:
            with self._lock:
                last_mid = (self._h1[s][-1][3] if len(self._h1[s]) else 1.0)
                bid = max(last_mid - SPREAD_COST * 0.5, 1e-8)
                ask = bid + SPREAD_COST
                mid = (bid + ask) * 0.5

                bkt_m, bar_m = self._agg_m1[s].update(mid, now)
                if bar_m and bkt_m:
                    self._m1[s].append(bar_m)
                    self._persist_bar(s, "M1", bkt_m, bar_m)
                    self._emit_callbacks("M1", s, bkt_m, bar_m)

                    bkt_h, bar_h = self._agg_h1[s].update(bar_m[3], bkt_m)
                    if bar_h and bkt_h:
                        self._h1[s].append(bar_h)
                        self._persist_bar(s, "H1", bkt_h, bar_h)
                        self._update_vol(s)
                        self._emit_callbacks("H1", s, bkt_h, bar_h)

                    bkt_d, bar_d = self._agg_d1[s].update(bar_m[3], bkt_m)
                    if bar_d and bkt_d:
                        self._d1[s].append(bar_d)
                        self._persist_bar(s, "D1", bkt_d, bar_d)
                        self._emit_callbacks("D1", s, bkt_d, bar_d)

        time.sleep(1.0)

    # ---------------- Callback emission (thread-safe) ----------------

    def _emit_callbacks(self, tf: str, symbol: str, when: dt.datetime, bar: OHLC) -> None:
        # Copy handler lists under lock, then call without holding lock
        with self._lock:
            if tf == "M1":
                handlers = list(self._m1_handlers)
            elif tf == "H1":
                handlers = list(self._h1_handlers)
            elif tf == "D1":
                handlers = list(self._d1_handlers)
            else:
                return
        for fn in handlers:
            try:
                fn(symbol, when, bar)
            except Exception:
                # Never let a bad handler kill the data thread
                pass
