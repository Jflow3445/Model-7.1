# data/audit_and_fix_ohlc.py
# Ultra-fast audit/fix of minute OHLC CSVs with safe parallelism.
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Project settings and backfiller (v2) ------------------------------------
from config.settings import LIVE_FOREX_PAIRS, OHLC_CSV_DIR
from data.dukascopy_backfill import (
    backfill_symbol,
    rebuild_from_minute_only,
    WRITE_LAG_MINUTES,   # reuse same horizon lag as downloader
)

# ---------- Tunables (env-overridable; sane defaults) ------------------------
DEFAULT_YEARS                  = int(os.getenv("AUDIT_DEFAULT_YEARS", "5"))
DEFAULT_SCAN_DAYS              = int(os.getenv("AUDIT_DEFAULT_SCAN_DAYS", "180"))  # set 0 to scan all
MAX_GAPS_TO_FIX_PER_PAIR       = int(os.getenv("AUDIT_MAX_GAPS_PER_PAIR", "50"))
GAP_MINUTES_THRESHOLD          = int(os.getenv("AUDIT_GAP_MINUTES_THRESHOLD", "2"))
HISTORY_SLACK_DAYS             = int(os.getenv("AUDIT_HISTORY_SLACK_DAYS", "3"))
PAIRS_MAX_WORKERS              = int(os.getenv("AUDIT_PAIRS_MAX_WORKERS", str(max(2, (os.cpu_count() or 4) * 2))))
FIX_PAIRS_MAX_WORKERS          = int(os.getenv("AUDIT_FIX_PAIRS_MAX_WORKERS", "4"))  # keep modest; backfill has its own parallelism
CSV_ENGINE                     = os.getenv("AUDIT_CSV_ENGINE", "auto")  # "auto" | "pyarrow" | "c"
SYMBOL_REGEX                   = re.compile(r"^[A-Z0-9:_\-]{3,15}$")  # pragmatic, prevents path traversal junk

# ---------- Helpers ----------------------------------------------------------
def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _expected_window(years: int) -> Tuple[dt.datetime, dt.datetime]:
    end = _now_utc() - dt.timedelta(minutes=WRITE_LAG_MINUTES)
    start = end - dt.timedelta(days=years * 365)
    return start, end

def _minute_csv(sym: str, base: Path) -> Path:
    # defensive: keep symbols clean
    if not SYMBOL_REGEX.match(sym.upper()):
        raise ValueError(f"Invalid symbol '{sym}'")
    return Path(base) / f"{sym}_1min.csv"

def _read_minute_index(csv_path: Path) -> pd.DatetimeIndex:
    """
    Fast, low-RAM index reader. Memory-mapped for SSD speed.
    Prefers pyarrow engine if available and requested.
    """
    engine = None
    if CSV_ENGINE == "pyarrow" or (CSV_ENGINE == "auto"):
        try:
            import pyarrow  # noqa: F401
            engine = "pyarrow"
        except Exception:
            engine = "c" if CSV_ENGINE == "auto" else None

    read_kwargs = dict(
        usecols=["time", "close"],  # 'close' to keep parse stable across pandas versions
        parse_dates=["time"],
        index_col="time",
        memory_map=True if engine == "c" else False,  # pyarrow ignores memory_map
        low_memory=False if engine == "c" else None,
        engine=engine,
    )
    # strip Nones
    read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}

    try:
        df = pd.read_csv(csv_path, **read_kwargs)
    except Exception:
        # fallback to generic parse (older files / rare corruptions)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = "time"

    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    return idx[~idx.isna()].sort_values()

def _subset_index(idx: pd.DatetimeIndex, scan_days: int) -> pd.DatetimeIndex:
    if len(idx) < 2:
        return idx
    if scan_days and scan_days > 0:
        horizon_from = _now_utc() - dt.timedelta(days=scan_days)
        return idx[idx >= horizon_from]
    return idx  # scan full range

def _find_recent_gaps(
    idx: pd.DatetimeIndex,
    scan_days: int,
    threshold_min: int = GAP_MINUTES_THRESHOLD,
    max_gaps: int = MAX_GAPS_TO_FIX_PER_PAIR,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Blazing-fast gap finder:
      - Works entirely in UTC
      - Vectorized via numpy on int64 nanoseconds
      - Returns up to `max_gaps` missing windows as (start, end)
    """
    sub = _subset_index(idx, scan_days)
    n = len(sub)
    if n < 2:
        return []

    # Convert to integer minutes safely (ns -> min)
    # Use .asi8 for stability across pandas versions
    t_ns = sub.asi8  # int64 nanoseconds
    t_min = t_ns // 60_000_000_000

    # Gaps are diffs >= threshold
    diffs = np.diff(t_min)
    jump_pos = np.nonzero(diffs >= threshold_min)[0]
    if jump_pos.size == 0:
        return []

    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    one_min = pd.Timedelta(minutes=1)
    # For each gap at position i, left=sub[i], right=sub[i+1]
    for i in jump_pos[:max_gaps]:
        left = sub[i]
        right = sub[i + 1]
        start = left + one_min
        end = right - one_min
        if start <= end:
            out.append((start, end))
    return out

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
    gap_threshold_min: int = GAP_MINUTES_THRESHOLD,
    max_gaps: int = MAX_GAPS_TO_FIX_PER_PAIR,
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

    earliest = idx[0].to_pydatetime().astimezone(dt.timezone.utc)
    latest   = idx[-1].to_pydatetime().astimezone(dt.timezone.utc)

    # Coverage vs expected window
    missing_early_days = max(0, math.ceil((earliest - expected_start_dt).total_seconds() / 86400.0))
    stale_mins = max(0, math.floor((expected_end_dt - latest).total_seconds() / 60.0))

    gaps = _find_recent_gaps(idx, scan_days=scan_days, threshold_min=gap_threshold_min, max_gaps=max_gaps)

    action_flags: List[str] = []
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
    Uses backfill_symbol (which has its own concurrency) to fetch only missing windows.
    Aggregates counts added across minute/hourly/daily.
    """
    exp_start = pd.Timestamp(audit.expected_start, tz="UTC").to_pydatetime()
    exp_end   = pd.Timestamp(audit.expected_end, tz="UTC").to_pydatetime()
    added = {"1min": 0, "hourly": 0, "daily": 0}

    def _acc(r: Dict[str, int]) -> None:
        for k in ("1min", "hourly", "daily"):
            added[k] += int(r.get(k, 0))

    # Full rebuild if missing/empty
    if audit.action in ("missing_file", "empty_file"):
        _acc(backfill_symbol(symbol, exp_start, exp_end, one_min_csv_dir=minute_csv_dir))
        return added

    # Missing early history
    if audit.missing_early_history_days > HISTORY_SLACK_DAYS and audit.earliest:
        earliest = pd.Timestamp(audit.earliest, tz="UTC").to_pydatetime()
        start = exp_start
        end   = earliest - dt.timedelta(minutes=1)
        if start < end:
            _acc(backfill_symbol(symbol, start, end, one_min_csv_dir=minute_csv_dir))

    # Stale tail
    if audit.stale_minutes > 0 and audit.latest:
        latest = pd.Timestamp(audit.latest, tz="UTC").to_pydatetime()
        start = latest + dt.timedelta(minutes=1)
        end   = exp_end
        if start < end:
            _acc(backfill_symbol(symbol, start, end, one_min_csv_dir=minute_csv_dir))

    # Recent gaps (recompute from file to incorporate any newly fetched data)
    try:
        idx2 = _read_minute_index(_minute_csv(symbol, minute_csv_dir))
        gaps = _find_recent_gaps(idx2, scan_days=DEFAULT_SCAN_DAYS)
    except Exception:
        gaps = []

    for (gs, ge) in gaps[:MAX_GAPS_TO_FIX_PER_PAIR]:
        if gs < ge:
            _acc(backfill_symbol(symbol, gs.to_pydatetime(), ge.to_pydatetime(), one_min_csv_dir=minute_csv_dir))

    return added

# ---------- Batch API (parallel across pairs) --------------------------------
def audit_pairs(
    pairs: Optional[Iterable[str]] = None,
    years: int = DEFAULT_YEARS,
    minute_csv_dir: Path = OHLC_CSV_DIR,
    scan_days: int = DEFAULT_SCAN_DAYS,
    gap_threshold_min: int = GAP_MINUTES_THRESHOLD,
    max_gaps: int = MAX_GAPS_TO_FIX_PER_PAIR,
    pairs_max_workers: int = PAIRS_MAX_WORKERS,
) -> Dict[str, AuditResult]:
    symbols = [p.upper() for p in (pairs if pairs is not None else LIVE_FOREX_PAIRS)]
    results: Dict[str, AuditResult] = {}

    def _work(sym: str) -> Tuple[str, AuditResult]:
        try:
            return sym, audit_symbol(
                sym, years=years, minute_csv_dir=minute_csv_dir,
                scan_days=scan_days, gap_threshold_min=gap_threshold_min, max_gaps=max_gaps
            )
        except Exception:
            # Fail closed but continue (don’t let one symbol kill the batch)
            return sym, AuditResult(
                symbol=sym, exists=False, empty=True,
                earliest=None, latest=None,
                expected_start=_expected_window(years)[0].isoformat(),
                expected_end=_expected_window(years)[1].isoformat(),
                missing_early_history_days=(years * 365),
                stale_minutes=math.inf,
                recent_gap_count=0,
                gaps_preview=[],
                action="missing_file",
            )

    with ThreadPoolExecutor(max_workers=pairs_max_workers) as ex:
        futs = {ex.submit(_work, s): s for s in symbols}
        for f in as_completed(futs):
            s, res = f.result()
            results[s] = res
    return results

def fix_pairs(
    audits: Dict[str, AuditResult],
    minute_csv_dir: Path = OHLC_CSV_DIR,
    rebuild_after: bool = True,
    fix_pairs_max_workers: int = FIX_PAIRS_MAX_WORKERS,
) -> Dict[str, Dict[str, int]]:
    # Only work on symbols that need attention
    targets = [s for s, a in audits.items() if a.action != "ok"]
    added_totals: Dict[str, Dict[str, int]] = {}
    touched: List[str] = []

    if not targets:
        return added_totals

    def _fix(sym: str) -> Tuple[str, Dict[str, int]]:
        r = fix_symbol(sym, audits[sym], minute_csv_dir=minute_csv_dir)
        return sym, r

    # Keep modest concurrency here; backfill_symbol uses its own thread pool per symbol.
    with ThreadPoolExecutor(max_workers=fix_pairs_max_workers) as ex:
        futs = {ex.submit(_fix, s): s for s in targets}
        for f in as_completed(futs):
            s, r = f.result()
            added_totals[s] = r
            if sum(r.values()) > 0:
                touched.append(s)

    if rebuild_after and touched:
        rebuild_from_minute_only(touched, minute_csv_dir=minute_csv_dir, hourly_csv_dir=minute_csv_dir)
    return added_totals

# ---------- CLI ---------------------------------------------------------------
def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Audit minute OHLC CSVs and auto-fix by redownloading missing windows using dukascopy_backfill v2. Fast and parallel."
    )
    p.add_argument("--pairs", type=str, default="", help="Comma-separated symbols (default: LIVE_FOREX_PAIRS).")
    p.add_argument("--years", type=int, default=DEFAULT_YEARS, help=f"History window to expect (default: {DEFAULT_YEARS}).")
    p.add_argument("--scan-days", type=int, default=DEFAULT_SCAN_DAYS, help=f"Scan recent N days for gaps (0 = scan ALL; default: {DEFAULT_SCAN_DAYS}).")
    p.add_argument("--scan-all", action="store_true", help="Scan the entire available index for gaps (equivalent to --scan-days 0).")
    p.add_argument("--gap-threshold-minutes", type=int, default=GAP_MINUTES_THRESHOLD, help=f"Gap threshold in minutes (default: {GAP_MINUTES_THRESHOLD}).")
    p.add_argument("--max-gaps-per-pair", type=int, default=MAX_GAPS_TO_FIX_PER_PAIR, help=f"Safety cap on gaps per pair (default: {MAX_GAPS_TO_FIX_PER_PAIR}).")

    p.add_argument("--pairs-max-workers", type=int, default=PAIRS_MAX_WORKERS, help=f"Thread workers for auditing pairs (default: {PAIRS_MAX_WORKERS}).")
    p.add_argument("--fix-pairs-max-workers", type=int, default=FIX_PAIRS_MAX_WORKERS, help=f"Thread workers for fixing pairs (default: {FIX_PAIRS_MAX_WORKERS}).")

    p.add_argument("--dry-run", action="store_true", help="Only audit; do not download.")
    p.add_argument("--fix", action="store_true", help="Download/fix files based on audit results.")
    p.add_argument("--rebuild-after", action="store_true", help="After fixes, rebuild hourly & daily from minute for touched pairs.")
    return p.parse_args()

def main():
    args = _parse_cli()
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()] if args.pairs else LIVE_FOREX_PAIRS
    scan_days = 0 if args.scan_all else args.scan_days

    print(f"[Audit] Base CSV dir: {OHLC_CSV_DIR}")
    audits = audit_pairs(
        pairs=pairs,
        years=args.years,
        minute_csv_dir=OHLC_CSV_DIR,
        scan_days=scan_days,
        gap_threshold_min=args.gap_threshold_minutes,
        max_gaps=args.max_gaps_per_pair,
        pairs_max_workers=args.pairs_max_workers,
    )

    # Compact JSON summary (fast to parse later)
    summary = {s: asdict(audits[s]) for s in pairs}
    print(json.dumps(summary, indent=2))

    if not args.fix and not args.dry_run:
        print("[Audit] No --fix specified; exiting after audit.")
        return

    if args.fix:
        added = fix_pairs(
            audits,
            minute_csv_dir=OHLC_CSV_DIR,
            rebuild_after=args.rebuild_after,
            fix_pairs_max_workers=args.fix_pairs_max_workers,
        )
        print(json.dumps({"downloads_added": added}, indent=2))

if __name__ == "__main__":
    main()
