# data/audit_and_fix_ohlc.py
# Audits 1-minute OHLC CSVs and auto-fixes (re-downloads) using data.dukascopy_backfill v2.
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# --- Project settings and backfiller (v2) ------------------------------------
from config.settings import LIVE_FOREX_PAIRS, OHLC_CSV_DIR
from data.dukascopy_backfill import (
    backfill_symbol,
    rebuild_from_minute_only,
    WRITE_LAG_MINUTES,   # reuse same horizon lag
)

# ---------- Tunables (override via CLI flags if you want) --------------------
DEFAULT_SCAN_DAYS        = 180    # scan recent N days for gaps (keeps it fast/low-RAM)
MAX_GAPS_TO_FIX_PER_PAIR = 50     # safety cap
GAP_MINUTES_THRESHOLD    = 2      # treat >=2 min jumps as a gap (minute data)
HISTORY_SLACK_DAYS       = 3      # allow this many days before deciding "missing early history"

# ---------- Helpers ----------------------------------------------------------
def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _expected_window(years: int) -> Tuple[dt.datetime, dt.datetime]:
    end = _now_utc() - dt.timedelta(minutes=WRITE_LAG_MINUTES)
    start = end - dt.timedelta(days=years * 365)
    return start, end

def _minute_csv(sym: str, base: Path) -> Path:
    return Path(base) / f"{sym}_1min.csv"

def _read_minute_index(csv_path: Path) -> pd.DatetimeIndex:
    # read minimal columns for speed/memory
    df = pd.read_csv(csv_path, usecols=["time", "close"], parse_dates=["time"], index_col="time")
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    return idx[~idx.isna()]

def _find_recent_gaps(idx: pd.DatetimeIndex, scan_days: int) -> List[Tuple[dt.Timestamp, dt.Timestamp]]:
    if len(idx) == 0:
        return []
    horizon_from = _now_utc() - dt.timedelta(days=scan_days)
    sub = idx[idx >= horizon_from]
    if len(sub) < 2:
        return []
    # Compute differences (in minutes)
    diffs = (sub[1:] - sub[:-1]).astype("timedelta64[m]").astype(int)
    jumps = diffs[diffs >= GAP_MINUTES_THRESHOLD]
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for i_pos in jumps.index:
        # i_pos corresponds to right side of the jump; left is previous
        right = sub[sub.get_loc(i_pos)]
        left  = sub[sub.get_loc(i_pos) - 1]
        # Missing window is (left+1min .. right-1min)
        start = (left + pd.Timedelta(minutes=1)).tz_convert("UTC")
        end   = (right - pd.Timedelta(minutes=1)).tz_convert("UTC")
        if start <= end:
            gaps.append((start, end))
        if len(gaps) >= MAX_GAPS_TO_FIX_PER_PAIR:
            break
    return gaps

# ---------- Audit dataclass ---------------------------------------------------
@dataclass
class AuditResult:
    symbol: str
    exists: bool
    empty: bool
    earliest: Optional[str]
    latest: Optional[str]
    expected_start: str
    expected_end: str
    missing_early_history_days: int
    stale_minutes: int
    recent_gap_count: int
    gaps_preview: List[Tuple[str, str]]
    action: str  # "ok" | "missing_file" | "empty_file" | "stale" | "gappy" | "missing_history" | "mixed"

# ---------- Core audit & fix -------------------------------------------------
def audit_symbol(
    symbol: str,
    years: int,
    minute_csv_dir: Path = OHLC_CSV_DIR,
    scan_days: int = DEFAULT_SCAN_DAYS,
) -> AuditResult:
    expected_start_dt, expected_end_dt = _expected_window(years)
    minute_path = _minute_csv(symbol, minute_csv_dir)
    exists = minute_path.exists()
    if not exists:
        return AuditResult(
            symbol=symbol, exists=False, empty=True,
            earliest=None, latest=None,
            expected_start=expected_start_dt.isoformat(),
            expected_end=expected_end_dt.isoformat(),
            missing_early_history_days=(years * 365),
            stale_minutes=math.inf,
            recent_gap_count=0,
            gaps_preview=[],
            action="missing_file",
        )

    try:
        idx = _read_minute_index(minute_path)
    except Exception:
        # unreadable/corrupt -> treat as empty
        idx = pd.DatetimeIndex([])

    empty = (len(idx) == 0)
    if empty:
        return AuditResult(
            symbol=symbol, exists=True, empty=True,
            earliest=None, latest=None,
            expected_start=expected_start_dt.isoformat(),
            expected_end=expected_end_dt.isoformat(),
            missing_early_history_days=(years * 365),
            stale_minutes=math.inf,
            recent_gap_count=0,
            gaps_preview=[],
            action="empty_file",
        )

    # Normalize
    idx = idx.sort_values()
    earliest = idx[0].to_pydatetime().astimezone(dt.timezone.utc)
    latest   = idx[-1].to_pydatetime().astimezone(dt.timezone.utc)

    # How far back do we go vs expected?
    missing_early_days = max(0, math.ceil((earliest - expected_start_dt).total_seconds() / 86400.0))
    # How stale vs horizon?
    stale_mins = max(0, math.floor((expected_end_dt - latest).total_seconds() / 60.0))

    gaps = _find_recent_gaps(idx, scan_days=scan_days)
    action_flags = []
    if missing_early_days > HISTORY_SLACK_DAYS:
        action_flags.append("missing_history")
    if stale_mins > 0:
        action_flags.append("stale")
    if len(gaps) > 0:
        action_flags.append("gappy")

    action = "ok" if not action_flags else ("mixed" if len(action_flags) > 1 else action_flags[0])

    return AuditResult(
        symbol=symbol,
        exists=True,
        empty=False,
        earliest=earliest.isoformat(),
        latest=latest.isoformat(),
        expected_start=expected_start_dt.isoformat(),
        expected_end=expected_end_dt.isoformat(),
        missing_early_history_days=missing_early_days,
        stale_minutes=stale_mins,
        recent_gap_count=len(gaps),
        gaps_preview=[(s.isoformat(), e.isoformat()) for s, e in gaps[:5]],
        action=action,
    )

def fix_symbol(
    symbol: str,
    audit: AuditResult,
    minute_csv_dir: Path = OHLC_CSV_DIR,
) -> Dict[str, int]:
    """
    Uses backfill_symbol from v2 to fetch only missing windows. Returns a dict of counts added.
    """
    # Parse times
    exp_start = pd.Timestamp(audit.expected_start, tz="UTC").to_pydatetime()
    exp_end   = pd.Timestamp(audit.expected_end, tz="UTC").to_pydatetime()
    # Default no-ops
    added = {"1min": 0, "hourly": 0, "daily": 0}

    if audit.action in ("missing_file", "empty_file"):
        # Full window
        return backfill_symbol(symbol, exp_start, exp_end, one_min_csv_dir=minute_csv_dir)

    # For mixed/stale/gappy/missing_history we may do multiple targeted calls:
    # 1) Missing early history (fetch older window)
    if audit.missing_early_history_days > HISTORY_SLACK_DAYS and audit.earliest:
        earliest = pd.Timestamp(audit.earliest, tz="UTC").to_pydatetime()
        # fetch [exp_start .. earliest-1m]
        start = exp_start
        end   = (earliest - dt.timedelta(minutes=1))
        if start < end:
            r = backfill_symbol(symbol, start, end, one_min_csv_dir=minute_csv_dir)
            for k in added: added[k] += r.get(k, 0)

    # 2) Stale tail (fetch [latest+1m .. exp_end])
    if audit.stale_minutes > 0 and audit.latest:
        latest = pd.Timestamp(audit.latest, tz="UTC").to_pydatetime()
        start = latest + dt.timedelta(minutes=1)
        end   = exp_end
        if start < end:
            r = backfill_symbol(symbol, start, end, one_min_csv_dir=minute_csv_dir)
            for k in added: added[k] += r.get(k, 0)

    # 3) Recent gaps (fetch each gap)
    #    (We recompute gaps from file in case earlier steps filled something.)
    try:
        idx = _read_minute_index(_minute_csv(symbol, minute_csv_dir))
        gaps = _find_recent_gaps(idx, scan_days=DEFAULT_SCAN_DAYS)
    except Exception:
        gaps = []

    for (gs, ge) in gaps[:MAX_GAPS_TO_FIX_PER_PAIR]:
        if gs < ge:
            r = backfill_symbol(symbol, gs.to_pydatetime(), ge.to_pydatetime(), one_min_csv_dir=minute_csv_dir)
            for k in added: added[k] += r.get(k, 0)

    return added

# ---------- Batch API ---------------------------------------------------------
def audit_pairs(
    pairs: Optional[Iterable[str]] = None,
    years: int = 10,
    minute_csv_dir: Path = OHLC_CSV_DIR,
    scan_days: int = DEFAULT_SCAN_DAYS,
) -> Dict[str, AuditResult]:
    if pairs is None:
        pairs = LIVE_FOREX_PAIRS
    results: Dict[str, AuditResult] = {}
    for s in [p.upper() for p in pairs]:
        results[s] = audit_symbol(s, years=years, minute_csv_dir=minute_csv_dir, scan_days=scan_days)
    return results

def fix_pairs(
    audits: Dict[str, AuditResult],
    minute_csv_dir: Path = OHLC_CSV_DIR,
    rebuild_after: bool = True,
) -> Dict[str, Dict[str, int]]:
    added_totals: Dict[str, Dict[str, int]] = {}
    touched: List[str] = []
    for s, a in audits.items():
        if a.action == "ok":
            continue
        r = fix_symbol(s, a, minute_csv_dir=minute_csv_dir)
        added_totals[s] = r
        if sum(r.values()) > 0:
            touched.append(s)

    if rebuild_after and touched:
        # Rebuild H1/D1 from minute for all touched pairs (idempotent & fast)
        rebuild_from_minute_only(touched, minute_csv_dir=minute_csv_dir, hourly_csv_dir=minute_csv_dir)
    return added_totals

# ---------- CLI ---------------------------------------------------------------
def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Audit minute OHLC files and auto-fix by redownloading missing windows using dukascopy_backfill v2."
    )
    p.add_argument("--pairs", type=str, default="", help="Comma-separated symbols (default: LIVE_FOREX_PAIRS).")
    p.add_argument("--years", type=int, default=10, help="History window to expect (default: 10).")
    p.add_argument("--scan-days", type=int, default=DEFAULT_SCAN_DAYS, help="Scan recent N days for gaps (default: 180).")
    p.add_argument("--dry-run", action="store_true", help="Only audit; do not download.")
    p.add_argument("--fix", action="store_true", help="Download/fix files based on audit results.")
    p.add_argument("--rebuild-after", action="store_true", help="After fixes, rebuild hourly & daily from minute for touched pairs.")
    return p.parse_args()

def main():
    args = _parse_cli()
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()] if args.pairs else LIVE_FOREX_PAIRS

    print(f"[Audit] Base CSV dir: {OHLC_CSV_DIR}")
    audits = audit_pairs(pairs=pairs, years=args.years, minute_csv_dir=OHLC_CSV_DIR, scan_days=args.scan_days)

    # Print compact JSON summary
    summary = {s: asdict(audits[s]) for s in pairs}
    print(json.dumps(summary, indent=2))

    if not args.fix and not args.dry_run:
        print("[Audit] No --fix specified; exiting after audit.")
        return

    if args.fix:
        added = fix_pairs(audits, minute_csv_dir=OHLC_CSV_DIR, rebuild_after=args.rebuild_after)
        print(json.dumps({"downloads_added": added}, indent=2))

if __name__ == "__main__":
    main()
