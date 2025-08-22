# tools/mt5_export_min_stops.py
import json
from pathlib import Path
import MetaTrader5 as mt5
# --- add repo root to sys.path so 'config.settings' resolves when run as a script ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import MT5_TERMINAL_PATH, LIVE_FOREX_PAIRS, BASE_DIR

OUT_PATH = BASE_DIR / "config" / "broker_stops.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Try common broker suffixes; you can extend via env if needed
CANDIDATE_SUFFIXES = ["", ".s", ".i", ".m", ".micro", ".pro", ".r"]

def _unique(seq):
    # order-preserving dedupe
    return list(dict.fromkeys(seq))

def _default_pip_size(sym: str, point: float) -> float:
    s = sym.upper()
    if s.startswith("XAU"):
        return 0.1
    if s.startswith("XAG"):
        return 0.01
    if s.endswith("JPY"):
        return 0.01
    # standard FX: ~1 pip = 10 * point on 5-digit quotes
    return max(10 * point, 0.0001)

def _best_symbol_match(req: str) -> str | None:
    """Find the best available MT5 symbol name for the requested symbol."""
    # 1) direct tries with known suffixes
    for suf in CANDIDATE_SUFFIXES:
        name = f"{req}{suf}"
        info = mt5.symbol_info(name)
        if info is not None:
            return name

    # 2) fallback: scan all symbols and pick shortest that startswith(req)
    all_syms = mt5.symbols_get()
    if all_syms is None:
        return None
    req_up = req.upper()
    starts = [si.name for si in all_syms if si.name.upper().startswith(req_up)]
    if starts:
        # pick the shortest name (usually the base or simplest suffix)
        return min(starts, key=len)

    # 3) loose fallback: contains(req) if nothing startswith
    contains = [si.name for si in all_syms if req_up in si.name.upper()]
    if contains:
        return min(contains, key=len)

    return None

def main():
    if not mt5.initialize(path=MT5_TERMINAL_PATH):
        raise RuntimeError(f"MT5 init failed with terminal: {MT5_TERMINAL_PATH}")

    requested = _unique(LIVE_FOREX_PAIRS)
    total = len(requested)
    found_count = 0

    out = {}
    try:
        for req in requested:
            actual = _best_symbol_match(req)
            if actual is None:
                # Write a placeholder entry so downstream code still has a key
                out[req] = {
                    "found": False,
                    "broker_symbol": None,
                    "digits": None,
                    "point": None,
                    "pip_size": 0.0001,          # safe default
                    "min_stop_points": 0,
                    "min_stop_pips": 5.0,        # safe default floor
                    "min_stop_price": 0.0005,    # 5 pips * 0.0001
                }
                print(f"[export] {req:>7} -> NOT FOUND (no matching broker symbol)")
                continue

            mt5.symbol_select(actual, True)
            si = mt5.symbol_info(actual)
            if si is None:
                out[req] = {
                    "found": False,
                    "broker_symbol": actual,
                    "digits": None,
                    "point": None,
                    "pip_size": 0.0001,
                    "min_stop_points": 0,
                    "min_stop_pips": 5.0,
                    "min_stop_price": 0.0005,
                }
                print(f"[export] {req:>7} -> {actual:>12} (symbol_info None)")
                continue

            point  = float(getattr(si, "point", 0.0) or 0.0)
            digits = int(getattr(si, "digits", 0) or 0)

            # MT5 stop level is in points (multiples of 'point')
            stops_level_points = getattr(si, "trade_stops_level", None)
            if stops_level_points is None:
                stops_level_points = getattr(si, "stops_level", 0)
            stops_level_points = int(stops_level_points or 0)

            pip_size = _default_pip_size(actual, point)
            pip_in_points = (pip_size / point) if point > 0 else 10.0
            min_stop_pips = (stops_level_points / pip_in_points) if pip_in_points > 0 else 0.0
            min_stop_price = float(stops_level_points) * point

            out[req] = {
                "found": True,
                "broker_symbol": actual,
                "digits": digits,
                "point": point,
                "pip_size": pip_size,
                "min_stop_points": stops_level_points,
                "min_stop_pips": round(min_stop_pips, 6),
                "min_stop_price": round(min_stop_price, 10),
            }
            found_count += 1

            print(
                f"[export] {req:>7} -> {actual:>12}  points={stops_level_points:<5} "
                f"pip_size={pip_size:<8g}  min_stop_pips={min_stop_pips:>7.3f}  "
                f"min_stop_price={min_stop_price:.6f}"
            )

    finally:
        mt5.shutdown()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[export] Wrote {OUT_PATH}  (found {found_count}/{total})")

if __name__ == "__main__":
    main()
