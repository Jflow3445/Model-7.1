import pandas as pd, pathlib as p
csv = p.Path(r"D:\Model 7.1\logs\ohlc\EURUSD_hourly.csv")  # adjust if your OHLC_CSV_DIR differs
df = pd.read_csv(csv, parse_dates=["time"], index_col="time").sort_index()
# consider only Mon–Fri
df_wd = df[df.index.dayofweek < 5]
rng = pd.date_range(df_wd.index.min(), df_wd.index.max(), freq="H", tz="UTC")
missing = rng.difference(df_wd.index)
print("Missing hourly bars (weekdays only):", len(missing))
print(missing[:50000])  # peek at first 50
