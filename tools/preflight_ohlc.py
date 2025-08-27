# preflight_ohlc.py
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import (
    HOURLY_CSV_DIR,
    DAILY_CSV_DIR,
    LIVE_FOREX_PAIRS,
    MEDIUM_OBS_WINDOW,
    LONG_OBS_WINDOW,
)

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

# ──────────────────────────────────────────────────────────────────────────────
# Core checker for any timeframe
# ──────────────────────────────────────────────────────────────────────────────
def preflight_timeframe(*, name: str, csv_dir: str, file_suffix: str, symbols, window: int):
    """
    Check OHLC CSVs for a timeframe, applying the same ATR rules as env.reset():
    - strict rolling ATR (min_periods=n), fallback to Wilder EMA if needed
    - require eff_len >= window + 2
    """
    csv_dir = Path(csv_dir)
    bad, ok = [], []
    details = {}  # sym -> dict(raw_len, eff_len, path)

    print(f"\n[Preflight/{name}] csv_dir={csv_dir}  window={window}")
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

        # strict rolling first (like env.reset), fallback to Wilder if needed
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

    # Alignment info (how many aligned bars we’ll really have across usable symbols)
    if ok:
        aligned_min = min(details[s]["eff_len"] for s in ok)
        print(f"[Preflight/{name}] aligned_min_eff_len_across_ok={aligned_min}")
    else:
        aligned_min = 0

    if bad:
        print(f"[Preflight/{name}] ❌ Problem symbols: {bad}")
    else:
        print(f"[Preflight/{name}] ✅ All symbols look usable.")

    return {
        "bad": bad,
        "ok": ok,
        "aligned_min_eff_len": aligned_min,
        "details": details,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper that checks both hourly & daily
# ──────────────────────────────────────────────────────────────────────────────
def preflight_all(
    symbols=LIVE_FOREX_PAIRS,
    hourly_dir=HOURLY_CSV_DIR,
    daily_dir=DAILY_CSV_DIR,
    hourly_window=MEDIUM_OBS_WINDOW,
    daily_window=LONG_OBS_WINDOW,
):
    res_hourly = preflight_timeframe(
        name="hourly",
        csv_dir=hourly_dir,
        file_suffix="hourly",
        symbols=symbols,
        window=hourly_window,
    )
    res_daily = preflight_timeframe(
        name="daily",
        csv_dir=daily_dir,
        file_suffix="daily",
        symbols=symbols,
        window=daily_window,
    )

    # Symbols that are bad in either timeframe
    bad_any = sorted(set(res_hourly["bad"]).union(res_daily["bad"]))
    ok_both = sorted(set(res_hourly["ok"]).intersection(res_daily["ok"]))

    print("\n[Preflight/summary]")
    print(f"  OK in both: {ok_both}")
    print(f"  Bad in any: {bad_any}")

    return {
        "hourly": res_hourly,
        "daily": res_daily,
        "ok_both": ok_both,
        "bad_any": bad_any,
    }

if __name__ == "__main__":
    preflight_all()
