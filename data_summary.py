# data_summary.py

import os
import pandas as pd
from pathlib import Path

# Match locations in your forex_data_system.py:
TICK_CSV_DIR = Path("logs/ticks/csv")
OHLC_CSV_DIR = Path("logs/ohlc")

def count_rows_csv(path):
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return 0

def main():
    # Find all symbols
    tick_files = list(TICK_CSV_DIR.glob("*_ticks.csv"))
    hourly_files = list(OHLC_CSV_DIR.glob("*_hourly.csv"))
    daily_files = list(OHLC_CSV_DIR.glob("*_daily.csv"))

    all_symbols = set()
    for f in tick_files + hourly_files + daily_files:
        sym = f.name.split("_")[0]
        all_symbols.add(sym)

    if not all_symbols:
        print("No symbols found in logs. Is your data being saved?")
        return

    print(f"\n{'SYMBOL':<10} | {'TICKS':>8} | {'HOURLY':>8} | {'DAILY':>8} |")
    print("-" * 44)

    for sym in sorted(all_symbols):
        tick_path = TICK_CSV_DIR / f"{sym}_ticks.csv"
        hourly_path = OHLC_CSV_DIR / f"{sym}_hourly.csv"
        daily_path = OHLC_CSV_DIR / f"{sym}_daily.csv"

        n_ticks = count_rows_csv(tick_path)
        n_hourly = count_rows_csv(hourly_path)
        n_daily = count_rows_csv(daily_path)

        print(f"{sym:<10} | {n_ticks:8} | {n_hourly:8} | {n_daily:8} |")

    print("\nRecommended Max Observation Windows (per type):")
    print("  (You should pick a window smaller than the min across all symbols for stable training)")

    min_tick = min([count_rows_csv(TICK_CSV_DIR / f"{s}_ticks.csv") for s in all_symbols])
    min_hourly = min([count_rows_csv(OHLC_CSV_DIR / f"{s}_hourly.csv") for s in all_symbols])
    min_daily = min([count_rows_csv(OHLC_CSV_DIR / f"{s}_daily.csv") for s in all_symbols])

    print(f"  TICK  : <= {min_tick - 2}  (use a window < {min_tick - 2})")
    print(f"  HOURLY: <= {min_hourly - 2}  (use a window < {min_hourly - 2})")
    print(f"  DAILY : <= {min_daily - 2}  (use a window < {min_daily - 2})")

    print("\nIf you get warnings about 'Insufficient bars', reduce your window size to fit these counts.")

if __name__ == "__main__":
    main()
