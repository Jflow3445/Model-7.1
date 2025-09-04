# config/settings.py
from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Final, Dict, List

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _env(var: str, default, cast_fn=lambda x: x):
    val = os.getenv(var, default)
    try:
        return cast_fn(val)
    except Exception:
        return default

def _auto_path(default_path: str) -> Path:
    """
    Choose between cloud/local path using env var.
    If the path starts with 'gdrive:', replace with Google Drive mount.
    """
    path = os.getenv('CLOUD_PATH_OVERRIDE', default_path)
    if path.startswith('gdrive:'):
        # For RunPod/Colab/etc. you must mount Google Drive to '/workspace/drive' or '/content/drive'
        base_drive = os.getenv('GDRIVE_MOUNT', '/workspace/drive/MyDrive')
        sub_path = path.replace('gdrive:', '', 1).lstrip('/')
        return Path(base_drive) / sub_path
    return Path(path)

# --------------------------------------------------------------------------- #
# Modes                                                                       #
# --------------------------------------------------------------------------- #
class RunMode(str, Enum):
    LIVE     = "live"      # live market training
    RETRAIN  = "retrain"   # scheduled fine‐tune on tick logs
    TRAINING = "training"  # historical back‐test

RUN_MODE:     Final[RunMode] = RunMode(_env("RUN_MODE", "live"))
LIVE_MODE:    Final[bool]     = RUN_MODE == RunMode.LIVE
RETRAIN_MODE: Final[bool]     = RUN_MODE == RunMode.RETRAIN
TRAIN_MODE:   Final[bool]     = RUN_MODE == RunMode.TRAINING

# --- Policy toggles (1 = enabled, 0 = disabled) ---
ENABLE_ONEMIN: bool = bool(int(_env("ENABLE_ONEMIN", 0)))
ENABLE_MEDIUM: bool = bool(int(_env("ENABLE_MEDIUM", 0)))
ENABLE_LONG:   bool = bool(int(_env("ENABLE_LONG",   1)))


# --------------------------------------------------------------------------- #
# Live Market Training                                                        #
# --------------------------------------------------------------------------- #
LIVE_TICK_BUFFER:     Final[int]   = int(_env("LIVE_TICK_BUFFER", 10000))
LIVE_OBS_WINDOW:      Final[int]   = int(_env("LIVE_OBS_WINDOW", 300))
MEDIUM_OBS_WINDOW:    Final[int]   = int(_env("MEDIUM_OBS_WINDOW", 60))
LONG_OBS_WINDOW:      Final[int]   = int(_env("LONG_OBS_WINDOW", 60))
ONEMIN_OBS_WINDOW:    Final[int]   = int(_env("ONEMIN_OBS_WINDOW", 60))
SIMULATED_TICK_DATA_FREQUENCY: Final[int] = int(_env("SIMULATED_TICK_DATA_FREQUENCY", 60))

# --------------------------------------------------------------------------- #
# Account & Risk                                                              #
# --------------------------------------------------------------------------- #
INITIAL_BALANCE:    Final[float] = float(_env("INITIAL_BALANCE",    10000))
MAX_RISK_PER_TRADE: Final[float] = float(_env("MAX_RISK_PER_TRADE",      0.01))
SPREAD_COST:        Final[float] = float(_env("SPREAD_COST",           0.0001))
COMMISSION_FEE:     Final[float] = float(_env("COMMISSION_FEE",             2.0))
SLIPPAGE_FACTOR:    Final[float] = float(_env("SLIPPAGE_FACTOR",        0.0002))
LOT_MULTIPLIER:     Final[int]   = int(_env("LOT_MULTIPLIER",        100_000))
EPS:                Final[float] = 1e-8
        
# --------------------------------------------------------------------------- #
# Global Seed for reproducibility                                             #
# --------------------------------------------------------------------------- #
SEED: Final[int] = int(_env("SEED", 42))

# --------------------------------------------------------------------------- #
# Trainer‐script compatibility                                                #
# --------------------------------------------------------------------------- #
SLIPPAGE_PER_UNIT:    Final[float] = float(_env("SLIPPAGE_PER_UNIT",    SLIPPAGE_FACTOR))
COMMISSION_PER_TRADE: Final[float] = float(_env("COMMISSION_PER_TRADE", COMMISSION_FEE))

# --------------------------------------------------------------------------- #
# Action Space                                                                #
# --------------------------------------------------------------------------- #
EXPLORATION_BONUS:       Final[float] = float(_env("EXPLORATION_BONUS",        5.0))
INACTIVITY_PENALTY_RATE: Final[float] = float(_env("INACTIVITY_PENALTY_RATE", -0.1))
TRADE_COOLDOWN_MINUTES:  Final[int]   = int(_env("TRADE_COOLDOWN_MINUTES",     1))
MIN_SL_OFFSET:           Final[float] = float(_env("MIN_SL_OFFSET",           1e-3))
MAX_OPEN_TRADES:         Final[int]   = int(_env("MAX_OPEN_TRADES",              1))
MAX_CLOSED_TRADES:       Final[int]   = int(_env("MAX_CLOSED_TRADES",           50))
TRADE_SIZE_BASE:         Final[int]   = int(_env("TRADE_SIZE_BASE",        100_000))

# --------------------------------------------------------------------------- #
# Symbols & API Keys                                                          #
# --------------------------------------------------------------------------- #
LIVE_FOREX_PAIRS: Final[List[str]] = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCAD', 'USDCHF', 
                                      'NZDUSD', 'XAUUSD', 'AUDCAD', 'AUDUSD', 'CADCHF', 
                                      'CADJPY', 'CADSGD', 'CHFJPY', 'EURAUD', 'EURCAD', 
                                      'EURCHF', 'EURJPY', 'EURNZD', 'GBPAUD', 'GBPCAD', 
                                      'GBPCHF', 'GBPJPY', 'GBPNZD', 'NZDJPY', 'XAGEUR', 
                                      'XAGUSD', 'XAUAUD', 'XAUEUR', 'USDSGD',  
                                      'USDHUF', 'USDMXN',  'SGDJPY', 'AUDNZD', 
                                        'USDCNH', 'NZDCAD', 
                                      'USDZAR', 'ZARJPY', 'NZDCHF', 'AUDCHF', 'EURGBP', 
                                      'AUDJPY', 'CHFSGD', 'NZDSGD', 'NZDHUF', #EURCNH', 
                                     #'GBPDKK','USDPLN', 'USDSEK'
                                     #'USDNOK','USDCZK', 'USDTRY', 
                                      ]
NUM_PAIRS: Final[int] = len(LIVE_FOREX_PAIRS)

# --------------------------------------------------------------------------- #
# Observation Space & Features                                                #
# --------------------------------------------------------------------------- #
NUM_FEATURES_PER_WINDOW: Final[int] = int(_env("NUM_FEATURES_PER_WINDOW", 12))
TIME_WINDOWS: Final[Dict[str,int]] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400
}
DATA_FREQUENCY = _env("DATA_FREQUENCY", "1m")
USE_TIME_WINDOWS: Final[List[int]] = [TIME_WINDOWS[DATA_FREQUENCY]] if DATA_FREQUENCY in TIME_WINDOWS else [60]
LIVE_OBS_DIM: Final[int] = len(USE_TIME_WINDOWS) * NUM_FEATURES_PER_WINDOW

# --------------------------------------------------------------------------- #
# Paths & Persistence (Google Drive/Cloud/Local Switch)                       #
# --------------------------------------------------------------------------- #

# Local or cloud base directory. CLOUD_BASE_DIR overrides for RunPod/GDrive.
BASE_DIR:        Final[Path] = Path(_env('BASE_DIR', Path(__file__).resolve().parent.parent))
CLOUD_MODE:      Final[bool] = bool(int(_env('CLOUD_MODE', 0)))  # 1 for true, 0 for false

# Use GDRIVE/OHLC and GDRIVE/MODELS for cloud, else defaults
GDRIVE_DATA:     Final[str] = _env('GDRIVE_DATA', 'ohlc')
GDRIVE_MODELS:   Final[str] = _env('GDRIVE_MODELS', 'models')

if CLOUD_MODE:
    LOGS_DIR:        Final[Path] = _auto_path(f'gdrive:{GDRIVE_DATA}/logs')
    OHLC_CSV_DIR:    Final[Path] = _auto_path(f'gdrive:{GDRIVE_DATA}')
    HOURLY_CSV_DIR:  Final[Path] = OHLC_CSV_DIR
    DAILY_CSV_DIR:   Final[Path] = OHLC_CSV_DIR
    TENSEC_CSV_DIR:  Final[Path] = OHLC_CSV_DIR
    ONEMIN_CSV_DIR:  Final[Path] = OHLC_CSV_DIR
    MODELS_DIR:      Final[Path] = _auto_path(f'gdrive:{GDRIVE_MODELS}')
else:
    LOGS_DIR:        Final[Path] = BASE_DIR
    OHLC_CSV_DIR:    Final[Path] = BASE_DIR / "logs" / "ohlc"
    HOURLY_CSV_DIR:  Final[Path] = OHLC_CSV_DIR
    DAILY_CSV_DIR:   Final[Path] = OHLC_CSV_DIR
    TENSEC_CSV_DIR:  Final[Path] = OHLC_CSV_DIR
    ONEMIN_CSV_DIR:  Final[Path] = OHLC_CSV_DIR
    MODELS_DIR:      Final[Path] = BASE_DIR / "models"

# ensure directories exist
for d in (MODELS_DIR, LOGS_DIR, OHLC_CSV_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH:        Final[Path] = MODELS_DIR / "live_forex_model.zip"
TENSEC_MODEL_SAVE_PATH: Final[Path] = MODELS_DIR / "tensec_policy.zip"
ONEMIN_MODEL_SAVE_PATH: Final[Path] = MODELS_DIR / "onemin_policy.zip"
LIVE_TICK_LOG_FILE:     Final[Path] = LOGS_DIR / "live_tick_data.jsonl"
TRADE_LOG_FILE:         Final[Path] = LOGS_DIR / "live_trades.jsonl"

# --------------------------------------------------------------------------- #
# Login to MT5                                                                #
# --------------------------------------------------------------------------- #
MT5_ACCOUNT:      Final[int]   = int(_env("MT5_ACCOUNT", 61388744))
MT5_PASSWORD:     Final[str]   = _env("MT5_PASSWORD", "5s_ofAgnza")
MT5_SERVER:       Final[str]   = _env("MT5_SERVER",   "Pepperstone Demo")
MT5_TERMINAL_PATH: Final[str]  = _env(
    "MT5_TERMINAL_PATH",
    r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe"
)

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
LOG_LEVEL:  Final[str] = _env("LOG_LEVEL",  "INFO")
TIMEZONE:   Final[str] = _env("TIMEZONE",   "UTC")

# --------------------------------------------------------------------------- #
# Scheduled Retraining                                                        #
# --------------------------------------------------------------------------- #
RETRAIN_INTERVALS: Final[Dict[str,int]] = {
    "tick_policy":    9999999999,
    "medium_policy":  9999999999,
    "long_policy":    9999999999,
    "arbiter":        43200,
    "tensec_policy":   9999999999,    
    "onemin_policy":   9999999999,    
}
RETRAIN_SCRIPTS:   Final[Dict[str,str]] = {
    "tick_policy":    "trainers/train_tick_policy.py",
    "medium_policy":  "trainers/train_medium_policy.py",
    "long_policy":    "trainers/train_long_policy.py",
    "arbiter":        "trainers/train_arbiter.py",
    "tensec_policy":   "trainers/train_tensec_policy.py",
    "onemin_policy":   "trainers/train_onemin_policy.py",
}
SAVE_INTERVAL: Final[int] = int(_env("SAVE_INTERVAL", 100))
TICK_CSV_DIR: Final[Path] = LOGS_DIR / "ticks" / "csv"
DEBUG_MODE = True
Data_Key = 'fb84cce5a48043fb95b770e6e2799f5a'

# --------------------------------------------------------------------------- #
# Historical data source                                                      #
#   duka   -> use Dukascopy backfill (ticks -> 1m/1h; daily built from 1h)
#   twelve -> use Twelve Data (legacy path)
#   none   -> skip bootstrap; rely only on live MT5 stream + your own logs
# --------------------------------------------------------------------------- #
HIST_DATA_SOURCE = _env("HIST_DATA_SOURCE", "duka").lower()  # 'duka' | 'twelve' | 'none'
HIST_YEARS        = int(_env("HIST_YEARS", 5))

# Optional Dukascopy backfill at startup (polite, stays behind near-now)
POLYGON_KEY = "rYhOzITu2rfX_JCBa03uBHPiq0_iPuN7"
POLYGON_S3_ACCESS_KEY = "bba9402f-5265-4167-886d-ac5d0e57445a"
POLYGON_S3_SECRET_KEY= "rYhOzITu2rfX_JCBa03uBHPiq0_iPuN7"
# Expose all UPPERCASE names for easy import                                  #
# --------------------------------------------------------------------------- #
__all__ = [name for name in globals() if name.isupper()]
