# data/dukascopy_backfill.py
# Version: 2 (daily rebuilt directly from minute; safe tz handling)
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

__version__ = "2.0.0"

# ──────────────────────────────────────────────────────────────────────────────
# Project settings (symbols/paths/API key)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from config.settings import (
        LIVE_FOREX_PAIRS,
        OHLC_CSV_DIR,     # directory where CSVs live
        POLYGON_KEY,      # paid key stored here
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import required items from config.settings. "
        "Ensure PYTHONPATH includes project root and settings defines "
        "LIVE_FOREX_PAIRS, OHLC_CSV_DIR, POLYGON_KEY."
    ) from e

POLYGON_KEY = os.getenv("POLYGON_KEY", POLYGON_KEY).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Tunables (env-overridable; designed for high throughput)
# ──────────────────────────────────────────────────────────────────────────────
MAX_WORKERS          = int(os.getenv("POLY_MAX_WORKERS", "32"))   # per-symbol chunk workers
PAIRS_MAX_WORKERS    = int(os.getenv("POLY_PAIRS_MAX_WORKERS", "8"))
HTTP_TIMEOUT         = float(os.getenv("POLY_HTTP_TIMEOUT", "30.0"))
RETRY_MAX            = int(os.getenv("POLY_RETRY_MAX", "5"))
RETRY_BASE_SLEEP     = float(os.getenv("POLY_RETRY_BASE_SLEEP", "0.5"))
CHUNK_DAYS           = int(os.getenv("POLY_CHUNK_DAYS", "30"))    # keep under Polygon 50k/min cap
MERGE_EVERY_CHUNKS   = int(os.getenv("POLY_MERGE_EVERY_CHUNKS", "6"))
WRITE_LAG_MINUTES    = int(os.getenv("POLY_WRITE_LAG_MINUTES", "3"))  # avoid partial last bar
USE_HTTP2            = bool(int(os.getenv("POLY_HTTP2", "0")))

# ──────────────────────────────────────────────────────────────────────────────
# File locking for atomic merges
# ──────────────────────────────────────────────────────────────────────────────
try:
    from filelock import FileLock
except Exception as e:
    raise RuntimeError(
        "Missing dependency 'filelock'. Please install: pip install filelock"
    ) from e

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
API_ROOT = "https://api.polygon.io"

def _polygon_ticker(symbol: str) -> str:
    # Polygon uses C:<PAIR> for forex/metals (e.g., C:EURUSD, C:XAUUSD)
    return f"C:{symbol.upper()}"

def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize to columns [open, high, low, close, volume], tz-aware UTC index,
    sorted, de-duplicated. Safe to call multiple times.
    """
    cols = ["open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        out = pd.DataFrame(columns=cols)
        out.index.name = "time"
        return out

    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

    # Ensure tz-aware UTC, dedup, sort
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    out.index.name = "time"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Types
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].astype(float)
    out["volume"] = out["volume"].astype(float)
    return out[cols]

def _merge_save_atomic(target_csv: Path, new_df: pd.DataFrame) -> int:
    if new_df is None or new_df.empty:
        return 0
    target_csv = Path(target_csv)
    new_df = _standardize_ohlc(new_df)
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_csv.with_suffix(target_csv.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    lock = FileLock(str(target_csv) + ".lock")
    with lock:
        try:
            if target_csv.exists():
                try:
                    existing = pd.read_csv(target_csv, parse_dates=["time"], index_col="time")
                except Exception:
                    existing = pd.read_csv(target_csv, index_col=0, parse_dates=True)
                    existing.index.name = "time"
                existing = _standardize_ohlc(existing)
                before = len(existing)
                merged = pd.concat([existing, new_df[~new_df.index.isin(existing.index)]], axis=0)
                added = len(merged) - before
            else:
                merged = new_df
                added = len(new_df)
            merged.to_csv(tmp, index=True)
            os.replace(tmp, target_csv)
            return added
        except FileNotFoundError as e:
            missing = getattr(e, "filename", None) or getattr(e, "filename2", None) or str(tmp)
            print(f"[PolygonBackfill] FILE NOT FOUND while writing → {missing}")
            raise


def _latest_timestamp(csv_path: Path) -> Optional[pd.Timestamp]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time", usecols=["time", "close"])
    except Exception:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = "time"
    if df.empty:
        return None
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    idx = idx[~pd.isna(idx)]
    return idx.max() if len(idx) else None

def _resume_start_from_existing(
    start: dt.datetime,
    minute_csv: Path,
    hourly_csv: Path
) -> dt.datetime:
    """
    If local CSVs already have history, nudge start forward to (max(last_minute, last_hourly) + 1m).
    """
    last_m = _latest_timestamp(minute_csv)
    last_h = _latest_timestamp(hourly_csv)
    last = None
    if last_m is not None and (last is None or last_m > last):
        last = last_m
    if last_h is not None and (last is None or last_h > last):
        last = last_h
    if last is None:
        return start
    last_plus = (
        last.to_pydatetime()
            .astimezone(dt.timezone.utc)
            .replace(tzinfo=dt.timezone.utc)
        + dt.timedelta(minutes=1)
    )
    return max(start, last_plus)

# ──────────────────────────────────────────────────────────────────────────────
# Polygon fetchers
# ──────────────────────────────────────────────────────────────────────────────
def _agg_url_1m(ticker: str, start: dt.datetime, end: dt.datetime) -> str:
    # v2 aggregates: /v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}
    f = start.strftime("%Y-%m-%d")
    t = end.strftime("%Y-%m-%d")
    return (
        f"{API_ROOT}/v2/aggs/ticker/{ticker}/range/1/minute/{f}/{t}"
        "?adjusted=true&sort=asc&limit=50000"
    )

def _http_get(client: httpx.Client, url: str, attempt: int) -> Optional[httpx.Response]:
    try:
        r = client.get(url, timeout=HTTP_TIMEOUT)
        if r.status_code == 429:
            # backoff even if plan says "unlimited" (protect against soft limits)
            time.sleep(RETRY_BASE_SLEEP * (2 ** attempt) + random.uniform(0.05, 0.2))
            return None
        if 200 <= r.status_code < 300:
            return r
        if r.status_code == 404:
            return r  # treat as empty
        return None
    except Exception:
        return None

def _fetch_chunk_minutes(
    client: httpx.Client,
    ticker: str,
    start: dt.datetime,
    end: dt.datetime
) -> pd.DataFrame:
    """
    Fetch 1-minute aggregates for [start, end] inclusive (chunked day span).
    Returns standardized OHLCV DF indexed by UTC.
    """
    url = _agg_url_1m(ticker, start, end)
    resp: Optional[httpx.Response] = None
    for attempt in range(RETRY_MAX):
        resp = _http_get(client, url, attempt)
        if resp is not None:
            break
        time.sleep(RETRY_BASE_SLEEP * (2 ** attempt) + random.uniform(0.05, 0.3))
    if resp is None:
        raise RuntimeError(f"Polygon request failed after retries: {url}")
    if resp.status_code == 404:
        return _standardize_ohlc(pd.DataFrame(columns=["open","high","low","close","volume"], index=pd.to_datetime([], utc=True)))

    data = resp.json()
    results = data.get("results") or []
    if not results:
        return _standardize_ohlc(pd.DataFrame(columns=["open","high","low","close","volume"], index=pd.to_datetime([], utc=True)))

    # Polygon returns 't' (ms epoch), 'o','h','l','c','v'
    ts = pd.to_datetime([r["t"] for r in results], unit="ms", utc=True)
    df = pd.DataFrame({
        "open":   [float(r.get("o", float("nan"))) for r in results],
        "high":   [float(r.get("h", float("nan"))) for r in results],
        "low":    [float(r.get("l", float("nan"))) for r in results],
        "close":  [float(r.get("c", float("nan"))) for r in results],
        "volume": [float(r.get("v", 0.0))          for r in results],
    }, index=ts)
    return _standardize_ohlc(df)

def _chunk_windows(start: dt.datetime, end: dt.datetime, days: int) -> List[Tuple[dt.datetime, dt.datetime]]:
    out: List[Tuple[dt.datetime, dt.datetime]] = []
    cur = start
    while cur <= end:
        nxt = cur + dt.timedelta(days=days - 1)
        if nxt > end:
            nxt = end
        out.append((cur, nxt))
        cur = nxt + dt.timedelta(days=1)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Resampling (minute → hourly / daily) — hardened tz logic
# ──────────────────────────────────────────────────────────────────────────────
# in data/dukascopy_backfill.py
def _resample_minute_to_hourly(m1: pd.DataFrame) -> pd.DataFrame:
    if m1 is None or m1.empty:
        return _standardize_ohlc(pd.DataFrame(columns=["open","high","low","close","volume"]))
    m1 = _standardize_ohlc(m1)
    o = m1["open"].resample("1h").first()
    h = m1["high"].resample("1h").max()
    l = m1["low"].resample("1h").min()
    c = m1["close"].resample("1h").last()
    v = m1["volume"].resample("1h").sum(min_count=1)
    out = pd.concat([o, h, l, c, v], axis=1)
    out = out.dropna(how="all")          # ← drop empty hours
    return _standardize_ohlc(out)

def _resample_minute_to_daily(
    m1: pd.DataFrame,
    close_tz: str = "America/New_York",
    close_hour_local: int = 17
) -> pd.DataFrame:
    """
    Minute → Daily OHLCV with a 5pm New York session close (DST-aware).
    Safe timezone handling: never tz_localize an already tz-aware index.
    Steps:
      1) ensure m1 is UTC tz-aware
      2) convert to close_tz
      3) shift index back by close_hour_local so "session day" starts 00:00 local
      4) resample by 1D on shifted index
      5) shift forward by close_hour_local and tz-convert back to UTC
    """
    if m1 is None or m1.empty:
        return _standardize_ohlc(pd.DataFrame(columns=["open","high","low","close","volume"]))
    m1 = _standardize_ohlc(m1)  # UTC tz-aware

    # Convert to session timezone and shift index to align the session day
    local = m1.tz_convert(close_tz)
    shifted_idx = local.index - pd.Timedelta(hours=close_hour_local)
    df = local.copy()
    df.index = shifted_idx  # still tz-aware in close_tz

    # Aggregate over the shifted calendar day
    o = df["open"].resample("1D").first()
    h = df["high"].resample("1D").max()
    l = df["low"].resample("1D").min()
    c = df["close"].resample("1D").last()
    v = df["volume"].resample("1D").sum(min_count=1)
    out = pd.concat([o, h, l, c, v], axis=1)

    # Shift back to session close time; the index remains tz-aware (close_tz)
    out.index = out.index + pd.Timedelta(hours=close_hour_local)

    # Convert to UTC (DO NOT tz_localize here; it's already tz-aware)
    out = out.tz_convert("UTC")

    return _standardize_ohlc(out.dropna(how="all"))

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def backfill_symbol(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    one_min_csv_dir: Path = OHLC_CSV_DIR,
    hourly_csv_dir: Path = OHLC_CSV_DIR,
    resume: bool = True,
    max_workers: int = MAX_WORKERS,
) -> Dict[str, int]:
    """
    Download Polygon 1-minute OHLCV for `symbol` in [start, end], merge into:
      {symbol}_1min.csv, then rebuild {symbol}_hourly.csv and {symbol}_daily.csv
      DIRECTLY from minute data (daily uses 5pm NY session close).
    """
    if not POLYGON_KEY:
        raise RuntimeError("POLYGON_KEY is empty; set it in config.settings or environment.")

    # Normalize to UTC
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    else:
        start = start.astimezone(dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    else:
        end = end.astimezone(dt.timezone.utc)

    # Safety horizon to avoid partial newest bars
    horizon = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=WRITE_LAG_MINUTES)
    end = min(end, horizon)

    minute_csv = Path(one_min_csv_dir) / f"{symbol}_1min.csv"
    hourly_csv = Path(hourly_csv_dir)  / f"{symbol}_hourly.csv"
    daily_csv  = Path(one_min_csv_dir) / f"{symbol}_daily.csv"

    # NEW: make sure all parent dirs exist even if we end up not writing anything
    for p in (minute_csv, hourly_csv, daily_csv):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If even this fails, better to surface the path in the outer handler
            pass

    if resume:
        start = _resume_start_from_existing(start, minute_csv, hourly_csv)


    if end <= start:
        return {"1min": 0, "hourly": 0, "daily": 0}

    ticker = _polygon_ticker(symbol)
    windows = _chunk_windows(start, end, CHUNK_DAYS)

    limits = httpx.Limits(max_keepalive_connections=max_workers, max_connections=max_workers)
    headers = {
        "User-Agent": f"nister-polygon-backfill/{__version__}",
        "Authorization": f"Bearer {POLYGON_KEY}",
        "Accept": "application/json",
    }

    added_min_total = 0
    added_h1_total  = 0
    added_d1_total  = 0

    # Accumulator for batched merges
    batch_minute: List[pd.DataFrame] = []

    def flush_minute_batch() -> int:
        nonlocal batch_minute
        added = 0
        if batch_minute:
            df_min = _standardize_ohlc(pd.concat(batch_minute, axis=0))
            added = _merge_save_atomic(minute_csv, df_min)
            batch_minute = []
        return added

    # Download minute chunks in parallel
    with httpx.Client(http2=USE_HTTP2, limits=limits, headers=headers, follow_redirects=True) as client:
        def _fetch(w):
            s, e = w
            return _fetch_chunk_minutes(client, ticker, s, e)

        chunk_count = 0
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for mdf in pool.map(_fetch, windows):
                if not mdf.empty:
                    batch_minute.append(mdf)
                chunk_count += 1
                if MERGE_EVERY_CHUNKS > 0 and (chunk_count % MERGE_EVERY_CHUNKS == 0):
                    added_min_total += flush_minute_batch()

        # Final flush for minute CSV
        added_min_total += flush_minute_batch()

    # Load the updated minute file (restrict to the request window to keep it light)
    # Load the updated minute file (restrict to the request window to keep it light)
    if minute_csv.exists():
        try:
            m1 = pd.read_csv(minute_csv, parse_dates=["time"], index_col="time")
        except Exception:
            m1 = pd.read_csv(minute_csv, index_col=0, parse_dates=True)
            m1.index.name = "time"
        m1 = _standardize_ohlc(m1)
        m1 = m1[(m1.index >= pd.to_datetime(start, utc=True)) & (m1.index <= pd.to_datetime(end, utc=True))]
    else:
        # No minute data was written (e.g., empty API response) → skip resampling
        m1 = _standardize_ohlc(pd.DataFrame(columns=["open","high","low","close","volume"]))
    # Rebuild H1 directly from M1
    h1 = _resample_minute_to_hourly(m1)
    if not h1.empty:
        added_h1_total = _merge_save_atomic(hourly_csv, h1)

    # Rebuild D1 directly from M1 (5pm NY close), hardened tz logic
    d1 = _resample_minute_to_daily(m1)
    if not d1.empty:
        added_d1_total = _merge_save_atomic(daily_csv, d1)

    return {"1min": added_min_total, "hourly": added_h1_total, "daily": added_d1_total}

def rebuild_from_minute_only(
    pairs: Iterable[str],
    minute_csv_dir: Path = OHLC_CSV_DIR,
    hourly_csv_dir: Path = OHLC_CSV_DIR,
) -> Dict[str, Dict[str, int]]:
    """
    Rebuild hourly & daily purely from existing {symbol}_1min.csv for each symbol.
    No downloads. Useful after any prior minute updates or if hourly/daily got out of sync.
    """
    results: Dict[str, Dict[str, int]] = {}
    for sym in pairs:
        minute_csv = Path(minute_csv_dir) / f"{sym}_1min.csv"
        hourly_csv = Path(hourly_csv_dir)  / f"{sym}_hourly.csv"
        daily_csv  = Path(minute_csv_dir) / f"{sym}_daily.csv"

        if not minute_csv.exists():
            results[sym] = {"hourly": 0, "daily": 0}
            continue

        try:
            m1 = pd.read_csv(minute_csv, parse_dates=["time"], index_col="time")
        except Exception:
            m1 = pd.read_csv(minute_csv, index_col=0, parse_dates=True)
            m1.index.name = "time"
        m1 = _standardize_ohlc(m1)

        h1 = _resample_minute_to_hourly(m1)
        d1 = _resample_minute_to_daily(m1)

        added_h1 = _merge_save_atomic(hourly_csv, h1) if not h1.empty else 0
        added_d1 = _merge_save_atomic(daily_csv,  d1) if not d1.empty else 0
        results[sym] = {"hourly": added_h1, "daily": added_d1}
    return results

def backfill_all_pairs(
    years: int = 5,
    pairs: Optional[Iterable[str]] = None,
    resume: bool = True,
    symbol_workers: int = MAX_WORKERS,
) -> Dict[str, Dict[str, int]]:
    """
    Backfill all pairs in LIVE_FOREX_PAIRS (or provided `pairs`) for last `years`.
    Parallelizes across symbols (PAIRS_MAX_WORKERS).
    """
    if pairs is None:
        pairs = LIVE_FOREX_PAIRS

    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=years * 365)
    end = now

    results: Dict[str, Dict[str, int]] = {}

    def _do(sym: str) -> Tuple[str, Dict[str, int]]:
        try:
            return sym, backfill_symbol(sym, start, end, resume=resume, max_workers=symbol_workers)
        except FileNotFoundError as e:
            import traceback
            missing = getattr(e, "filename", None) or getattr(e, "filename2", None) or "<unknown>"
            print(f"[PolygonBackfill] {sym}: FILE NOT FOUND → {missing}")
            print(traceback.format_exc())
            return sym, {"1min": 0, "hourly": 0, "daily": 0}
        except Exception:
            import traceback
            print(f"[PolygonBackfill] {sym}: ERROR\n{traceback.format_exc()}")
            return sym, {"1min": 0, "hourly": 0, "daily": 0}
        
    with cf.ThreadPoolExecutor(max_workers=PAIRS_MAX_WORKERS) as pool:
        for sym, res in pool.map(_do, list(pairs)):
            results[sym] = res
    return results

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Polygon backfill → 1-minute, rebuild hourly & daily from minute (daily uses 5pm NY close). Safe tz conversions."
    )
    parser.add_argument("--years", type=int, default=5, help="Years of history to fetch (default 5).")
    parser.add_argument("--pairs", type=str, default="", help="Comma-separated symbols; default uses LIVE_FOREX_PAIRS.")
    parser.add_argument("--resume", action="store_true", help="Resume from last local bar if CSVs exist.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing CSVs; start from computed window.")
    parser.add_argument("--symbol-workers", type=int, default=MAX_WORKERS, help="Per-symbol chunk workers.")
    parser.add_argument("--rebuild-from-minute-only", action="store_true",
                        help="Skip downloads; rebuild hourly & daily for selected pairs from existing minute CSVs.")
    args = parser.parse_args()

    selected_pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()] if args.pairs else LIVE_FOREX_PAIRS

    if args.rebuild_from_minute_only:
        results = rebuild_from_minute_only(selected_pairs, OHLC_CSV_DIR, OHLC_CSV_DIR)
        total_h1 = sum(v.get("hourly", 0) for v in results.values())
        total_d1 = sum(v.get("daily", 0) for v in results.values())
        print(f"[PolygonBackfill] Rebuilt from minute only; hourly={total_h1:,}; daily={total_d1:,} across {len(results)} symbols.")
    else:
        resume_flag = args.resume and not args.no_resume
        out = backfill_all_pairs(
            years=args.years,
            pairs=selected_pairs,
            resume=resume_flag,
            symbol_workers=args.symbol_workers,
        )
        total_min = sum(v.get("1min", 0) for v in out.values())
        total_h1  = sum(v.get("hourly", 0) for v in out.values())
        total_d1  = sum(v.get("daily", 0) for v in out.values())
        print(f"[PolygonBackfill] Done. Added 1min={total_min:,}; hourly={total_h1:,}; daily={total_d1:,} across {len(out)} symbols.")
