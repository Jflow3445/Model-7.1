from __future__ import annotations
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import List, Optional, Union, Any

from config.settings import BASE_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Trade Vector Encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_trade_details(
    trades: List[dict],
    max_trades: int = 50,
    padding_value: float = 0.0
) -> np.ndarray:
    """
    Convert a list of trade dicts into a flat numpy array of shape (max_trades * 5,).
    Each trade vector: [direction, entry_price, volume, stop_loss, take_profit].
    Direction: long=1.0, short=-1.0, else 0.0.
    Pads with `padding_value` if fewer than max_trades.
    """
    vectors: List[np.ndarray] = []
    for trade in trades[-max_trades:]:
        # Extract and sanitize fields
        dir_map = {"long": 1.0, "short": -1.0}
        direction = dir_map.get(trade.get("trade_type", "").lower(), 0.0)
        entry = float(trade.get("entry_price", padding_value))
        vol = float(trade.get("volume", padding_value))
        sl = float(trade.get("stop_loss", padding_value))
        tp = float(trade.get("take_profit", padding_value))
        vec = np.array([direction, entry, vol, sl, tp], dtype=np.float32)
        vectors.append(vec)
    # Pad
    while len(vectors) < max_trades:
        vectors.append(np.full(5, padding_value, dtype=np.float32))
    return np.concatenate(vectors)


# ─────────────────────────────────────────────────────────────────────────────
# Transition Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_transition(
    symbol: str,
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    trade_history: Optional[List[dict]] = None,
    log_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Append a transition record (obs, action, reward, trades) to a per-symbol JSONL log.
    Creates directory BASE_DIR/logs by default.
    """
    base: Path = Path(log_dir) if log_dir else Path(BASE_DIR) / "logs"
    base.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "_").replace("=", "_")
    log_file = base / f"{safe_symbol}.jsonl"

    record: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "obs": obs.tolist(),
        "action": (
    action if isinstance(action, (int, float, list))
    else action.tolist()
),
        "reward": float(reward),
        "trade_history": trade_history or []
    }
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to write transition to {log_file}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Transition Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_transitions_from_log(
    log_file: Union[str, Path]
) -> List[dict]:
    """
    Load all JSON objects from a .jsonl transition log file.
    """
    log_path = Path(log_file)
    if not log_path.exists():
        return []
    transitions: List[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            transitions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# Storage Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_default_storage_dir(subdir: str = "storage") -> Path:
    """
    Get or create a storage directory under BASE_DIR.
    """
    path = Path(BASE_DIR) / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_model_filename(
    prefix: str = "forex_model",
    ext: str = ".zip"
) -> str:
    """
    Generate a timestamped filename: {prefix}_YYYYMMDD_HHMMSS{ext}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{ext}"


__all__ = [
    "encode_trade_details",
    "log_transition",
    "load_transitions_from_log",
    "get_default_storage_dir",
    "generate_model_filename",
]
