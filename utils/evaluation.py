from __future__ import annotations
import time
from datetime import datetime
from typing import Any, Dict, List, Sequence, Optional

import numpy as np
import gym
from tqdm import tqdm

import logging
from utils.logging_utils import log_event  # if you need to emit events

logger = logging.getLogger(__name__)



def get_env_attribute(env: Any, attr_name: str) -> Any:
    """
    Retrieve an attribute from a vectorized or non-vectorized env.
    If env has `get_attr`, returns the first element of the list.
    """
    if hasattr(env, attr_name):
        return getattr(env, attr_name)
    if hasattr(env, 'get_attr'):
        vals = env.get_attr(attr_name)
        if isinstance(vals, Sequence) and vals:
            return vals[0]
    return None


def compute_sharpe(rewards: Sequence[float]) -> float:
    """Sharpe ratio: mean / std (zero padded)."""
    arr = np.asarray(rewards, dtype=np.float64)
    std = arr.std()
    return float(arr.mean() / (std + 1e-8)) if std > 0 else 0.0


def compute_drawdown(equity_curve: Sequence[float]) -> float:
    """Maximum drawdown from equity curve."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    return float(np.max(np.maximum.accumulate(eq) - eq))


def compute_profit_factor(rewards: Sequence[float]) -> float:
    """Profit factor: sum(wins) / sum(losses)."""
    arr = np.asarray(rewards, dtype=np.float64)
    wins = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    return float(wins / (losses + 1e-8)) if losses > 0 else float('inf')


def compute_expectancy(rewards: Sequence[float]) -> float:
    """Expectancy = win_rate*avg_win - loss_rate*avg_loss."""
    arr = np.asarray(rewards, dtype=np.float64)
    wins = arr[arr > 0]
    losses = -arr[arr < 0]
    win_rate = len(wins) / len(arr) if arr.size else 0.0
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    return float(win_rate * avg_win - (1 - win_rate) * avg_loss)


def compute_drawdown_durations(equity_curve: Sequence[float]) -> List[int]:
    """Durations (in steps) of each drawdown period."""
    durations: List[int] = []
    peak = equity_curve[0]
    in_dd = False
    start = 0
    for i, eq in enumerate(equity_curve):
        if not in_dd:
            if eq < peak:
                in_dd = True
                start = i
        else:
            if eq >= peak:
                durations.append(i - start)
                in_dd = False
            peak = max(peak, eq)
    if in_dd:
        durations.append(len(equity_curve) - start)
    return durations


def evaluate_model(
    model: Any,
    env: gym.Env,
    n_eval_episodes: int = 10,
    max_steps: Optional[int] = None,
    show_progress: bool = False,
    render: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a model in `env`, returning performance metrics.
    """
    rewards: List[float] = []
    trade_counts: List[int] = []
    trade_durations: List[float] = []
    drawdown_durs: List[int] = []
    trade_sizes: List[float] = []
    latencies: List[float] = []
    equity_curves: List[List[float]] = []

    loop = range(n_eval_episodes)
    if show_progress:
        loop = tqdm(loop, desc="Evaluating")

    for ep in loop:
        start_time = time.perf_counter()
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        balance = get_env_attribute(env, "balance") or 0.0
        equity = [balance]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            steps += 1
            total_reward += float(np.mean(reward))
            balance = get_env_attribute(env, "balance") or balance
            equity.append(balance)
            if max_steps and steps >= max_steps:
                break

        # Latency per step
        elapsed = time.perf_counter() - start_time
        if steps:
            latencies.append(elapsed / steps)

        # Trades this episode
        th = get_env_attribute(env, "trade_history") or []
        trade_counts.append(len(th))

        # Compute trade durations (hours) and sizes
        for t in th:
            entry = t.get("entry_timestamp")
            exit_ = t.get("exit_time") or t.get("exit_timestamp")
            if isinstance(entry, datetime) and isinstance(exit_, datetime):
                trade_durations.append((exit_ - entry).total_seconds() / 3600.0)
            vol = t.get("volume") or 0.0
            trade_sizes.append(vol)

        # Drawdown durations
        drawdown_durs.extend(compute_drawdown_durations(equity))
        rewards.append(total_reward)
        equity_curves.append(equity)

        if render:
            logger.info(
                f"Episode {ep}: reward={total_reward:.4f}, steps={steps}, trades={len(th)}"
            )

    # Aggregate equity curves to common length
    max_len = max(len(ec) for ec in equity_curves)
    padded = np.array([
        np.pad(ec, (0, max_len - len(ec)), 'edge') for ec in equity_curves
    ], dtype=np.float64)
    mean_equity = padded.mean(axis=0)

    metrics: Dict[str, float] = {
        "average_reward": float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "min_reward":     float(np.min(rewards)),
        "max_reward":     float(np.max(rewards)),
        "average_trades": float(np.mean(trade_counts)),
        "sharpe_ratio":   compute_sharpe(rewards),
        "max_drawdown":   compute_drawdown(mean_equity),
        "profit_factor":  compute_profit_factor(rewards),
        "expectancy":     compute_expectancy(rewards),
        "win_rate":       100.0 * np.mean([1 if r > 0 else 0 for r in rewards]),
        # custom
        "avg_trade_duration_h": float(np.mean(trade_durations) if trade_durations else 0.0),
        "avg_drawdown_duration": float(np.mean(drawdown_durs) if drawdown_durs else 0.0),
        "avg_trade_size":       float(np.mean(trade_sizes) if trade_sizes else 0.0),
        "avg_step_latency_s":   float(np.mean(latencies) if latencies else 0.0),
    }

    return metrics
