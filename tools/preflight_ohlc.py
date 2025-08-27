# tools/preflight_ohlc.py
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import re
import numpy as np
import pandas as pd

# Try to use your project config if available; otherwise fall back to CLI args.
TRY_CONFIG = True
HOURLY_CSV_DIR = DAILY_CSV_DIR = None
LIVE_FOREX_PAIRS = None
MEDIUM_OBS_WINDOW = 48
LONG_OBS_WINDOW = 120

if TRY_CONFIG:
    try:
        from config.settings import (
            HOURLY_CSV_DIR as _H, DAILY_CSV_DIR as _D,
            LIVE_FOREX_PAIRS as _PAIRS,
            MEDIUM_OBS_WINDOW as _W_H, LONG_OBS_WINDOW as _W_D,
        )
        HOURLY_CSV_DIR = _H; DAILY_CSV_DIR = _D
        LIVE_FOREX_PAIRS = list(_PAIRS)
        MEDIUM_OBS_WINDOW = int(_W_H); LONG_OBS_WINDOW = int(_W_D)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (same semantics as your envs)
# ──────────────────────────────────────────────────────────────────────────────
def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = (
            df.sort_values("time")
              .drop_duplicates(subset="time", keep="last")
              .reset_index(drop=True)
        )
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

def compute_atr(df: pd.DataFrame, n: int = 14, mode: str = "rolling") -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    if mode == "wilder":
        atr = tr.ewm(alpha=1.0 / float(n), adjust=False, min_periods=1).mean()
        atr = atr.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    else:
        atr = tr.rolling(int(n), min_periods=int(n)).mean()
        atr = atr.replace([np.inf, -np.inf], np.nan)
    return atr

def infer_symbols_from_dir(csv_dir: Path, suffix: str) -> list[str]:
    syms = []
    for p in csv_dir.glob(f"*_{suffix}.csv"):
        m = re.match(r"(.+?)_" + re.escape(suffix) + r"\.csv$", p.name)
        if m:
            syms.append(m.group(1))
    return sorted(syms)

# ──────────────────────────────────────────────────────────────────────────────
# Core checker for any timeframe
# ──────────────────────────────────────────────────────────────────────────────
def preflight_timeframe(*, name: str, csv_dir: Path, file_suffix: str, symbols: list[str], window: int):
    csv_dir = Path(csv_dir)
    bad, ok = [], []
    details = {}

    print(f"\n[Preflight/{name}] dir={csv_dir}  window={window}")
    for sym in symbols:
        path = csv_dir / f"{sym}_{file_suffix}.csv"
        if not path.is_file():
            print(f"[MISSING] {sym}: {path}")
            bad.append(sym); details[sym] = {"raw_len": 0, "eff_len": 0, "path": str(path)}
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[READ-ERR] {sym}: {e}")
            bad.append(sym); details[sym] = {"raw_len": 0, "eff_len": 0, "path": str(path)}
            continue

        df = _safe_numeric(df)
        raw_len = len(df)
        if raw_len < window + 2:
            print(f"[SHORT-RAW] {sym}: raw_len={raw_len} (< {window+2})  path={path}")
            bad.append(sym); details[sym] = {"raw_len": raw_len, "eff_len": raw_len, "path": str(path)}
            continue

        # strict rolling first, fallback to Wilder
        df["atr"] = compute_atr(df, n=14, mode="rolling")
        tmp = df.dropna(subset=["atr"])
        if len(tmp) >= window + 2:
            eff_len = len(tmp)
        else:
            df["atr"] = compute_atr(df, n=14, mode="wilder")
            eff_len = len(df.dropna(subset=["atr"]))

        details[sym] = {"raw_len": raw_len, "eff_len": eff_len, "path": str(path)}
        if eff_len < window + 2:
            print(f"[SHORT-EFF] {sym}: eff_len={eff_len} (< {window+2}) raw_len={raw_len}  path={path}")
            bad.append(sym)
        else:
            print(f"[OK] {sym}: raw={raw_len} eff={eff_len}")
            ok.append(sym)

    if ok:
        aligned_min = min(details[s]["eff_len"] for s in ok)
        print(f"[Preflight/{name}] aligned_min_eff_len_across_OK={aligned_min}")
    else:
        aligned_min = 0

    if bad:
        print(f"[Preflight/{name}] ❌ Problem symbols: {bad}")
    else:
        print(f"[Preflight/{name}] ✅ All symbols look usable.")

    return {"bad": bad, "ok": ok, "aligned_min_eff_len": aligned_min, "details": details}

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preflight OHLC CSVs (hourly/daily) with ATR checks.")
    parser.add_argument("--timeframe", choices=["hourly", "daily", "both"], default="both")
    parser.add_argument("--hourly-dir", type=str, default=HOURLY_CSV_DIR or "")
    parser.add_argument("--daily-dir", type=str, default=DAILY_CSV_DIR or "")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated list. If omitted, uses config or infers from dir.")
    parser.add_argument("--hourly-window", type=int, default=MEDIUM_OBS_WINDOW)
    parser.add_argument("--daily-window", type=int, default=LONG_OBS_WINDOW)
    parser.add_argument("--bad-only", action="store_true", help="Print only the final bad lists at the end.")
    args = parser.parse_args()

    # Resolve dirs
    if args.timeframe in ("hourly", "both") and not args.hourly_dir:
        parser.error("--hourly-dir is required (or set in config.settings)")
    if args.timeframe in ("daily", "both") and not args.daily_dir:
        parser.error("--daily-dir is required (or set in config.settings)")

    hourly_dir = Path(args.hourly_dir) if args.hourly_dir else None
    daily_dir = Path(args.daily_dir) if args.daily_dir else None

    # Resolve symbols
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif LIVE_FOREX_PAIRS:
        symbols = list(LIVE_FOREX_PAIRS)
    else:
        # fallback: infer from whichever dir(s) we can
        symbols = []
        if hourly_dir and hourly_dir.exists():
            symbols = infer_symbols_from_dir(hourly_dir, "hourly")
        if not symbols and daily_dir and daily_dir.exists():
            symbols = infer_symbols_from_dir(daily_dir, "daily")
        if not symbols:
            parser.error("Could not infer symbols; provide --symbols or ensure files exist.")
    if not symbols:
        parser.error("No symbols to check.")

    res_hourly = res_daily = None
    if args.timeframe in ("hourly", "both"):
        res_hourly = preflight_timeframe(
            name="hourly",
            csv_dir=hourly_dir,
            file_suffix="hourly",
            symbols=symbols,
            window=args.hourly_window,
        )
    if args.timeframe in ("daily", "both"):
        res_daily = preflight_timeframe(
            name="daily",
            csv_dir=daily_dir,
            file_suffix="daily",
            symbols=symbols,
            window=args.daily_window,
        )

    # Summary
    if args.timeframe == "both":
        bad_any = sorted(set((res_hourly or {"bad": []})["bad"]).union((res_daily or {"bad": []})["bad"]))
        ok_both = sorted(set((res_hourly or {"ok": []})["ok"]).intersection((res_daily or {"ok": []})["ok"]))
        print("\n[Summary]")
        print(f"  OK in both: {ok_both}")
        print(f"  Bad in any: {bad_any}")
        if args.bad_only:
            print("\nBAD_ANY:", ",".join(bad_any))
    elif args.timeframe == "hourly":
        if args.bad_only:
            print("\nBAD_HOURLY:", ",".join(res_hourly["bad"]))
    else:
        if args.bad_only:
            print("\nBAD_DAILY:", ",".join(res_daily["bad"]))

    # Exit non-zero if any bad symbols (useful for CI)
    bad_count = 0
    if res_hourly: bad_count += len(res_hourly["bad"])
    if res_daily:  bad_count += len(res_daily["bad"])
    sys.exit(1 if bad_count > 0 else 0)

if __name__ == "__main__":
    main()
