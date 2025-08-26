# utils/reward_utils.py

from __future__ import annotations
import logging
from typing import Any, List, Dict, Optional, Union, Set, Tuple

import torch
import torch.nn as nn

from config.settings import INITIAL_BALANCE, EPS

logger = logging.getLogger("RewardFunction")
logger.setLevel(logging.INFO)

MIN_INITIAL_BALANCE = 1e-3


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device) if x.device != device else x


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _trade_side(trade: Dict) -> int:
    # +1 long, -1 short, 0 unknown
    t = (trade.get("trade_type") or "").lower()
    if t == "long":
        return 1
    if t == "short":
        return -1
    return 0


def _risk_distance(entry: float, stop_loss: float, side: int) -> float:
    # Positive distance from entry to stop for either side
    if side > 0:
        return max(entry - stop_loss, 0.0)
    elif side < 0:
        return max(stop_loss - entry, 0.0)
    return 0.0


def _r_multiple_from_trade(trade: Dict, min_risk: float, r_clip: float) -> float:
    """
    Compute per-trade R multiple: R = pnl / risk, where risk is |entry - stop_loss| * volume.
    - Floors risk by `min_risk` to avoid exploding ratios (fixes #4).
    - Clips to ±r_clip for stability.
    NOTE: We do not subtract costs here (point #1 is handled in envs as agreed).
    """
    pnl = _safe_float(trade.get("pnl", 0.0))
    entry = _safe_float(trade.get("entry_price", 0.0))
    sl = _safe_float(trade.get("stop_loss", 0.0))
    vol = _safe_float(trade.get("volume", 1.0)) or 1.0
    side = _trade_side(trade)

    risk = _risk_distance(entry, sl, side) * abs(vol)
    risk = max(risk, min_risk)
    r = pnl / risk
    # robust bound
    if r > r_clip:
        r = r_clip
    elif r < -r_clip:
        r = -r_clip
    return r


class RewardFunction(nn.Module):
    """
    A robust, PPO-friendly reward that:
      • Uses per-trade R (pnl/risk) with safe floors + clipping (fixes #4, #9)
      • Provides a bounded profit-quality signal via EMA profit factor proxy (fixes #5)
      • Uses step-expressed thresholds (holding/inactivity) (fixes #2, #3)
      • Removes brittle Sortino/maxDD terms unless supplied explicitly (fixes #4, #10)
      • Avoids cross-env state leakage; one instance per env (fixes #7)

    Inputs (per step):
      - closed_trades: list of trades closed at this step
          keys used: pnl, entry_price, stop_loss, take_profit (optional),
                     volume, trade_type, symbol (optional)
      - open_trades: list of currently-open trades
          keys used: entry_price, stop_loss, take_profit (optional), volume,
                     trade_type, symbol (optional), holding_time (in steps)
      - account_balance: optional
      - unrealized_pnl: optional (kept for compatibility; small weight)
      - time_since_last_trade: expressed in **steps** (not seconds)

    Component summary (all bounded):
      C1  realized_R_sum        (sum of clipped R for trades closed this step)
      C2  quality_ema           ((EMA GP - EMA GL)/(EMA GP + EMA GL + eps)) in [-1,1]
      C3  unrealized_norm       (unrealized_pnl / initial_balance) * unrealized_weight
      C4  inactivity_penalty    (-inactivity_weight * max(0, steps_since_last - grace))
      C5  holding_penalty       (-holding_penalty_per_step for each open trade beyond threshold steps)
      C6  overexposure_penalty  (-overexposure_weight * max(0, (exposure_R - budget_R)))
      C7  conflict_penalty      (-conflict_weight per symbol with both long and short open)
      C8  churn_penalty         (-churn_weight if many tiny-R trades closed this step)
    """

    def __init__(
        self,
        # Normalization / scaling
        initial_balance: Optional[float] = None,

        # Per-trade R computation
        min_risk: float = 1e-5,           # floors |entry-sl|*vol to avoid exploding R
        r_clip: float = 5.0,              # clip per-trade R to [-r_clip, r_clip]

        # EMA profit quality
        stats_alpha: float = 0.1,         # EMA smoothing for gross profit/loss
        quality_weight: float = 0.6,      # weight for C2

        # Realized & unrealized mix (dimensionless)
        realized_R_weight: float = 1.0,   # weight for C1
        unrealized_weight: float = 0.1,   # small; optional

        # Inactivity (steps-based)
        inactivity_weight: float = 0.02,  # penalty per extra step beyond grace
        inactivity_grace_steps: int = 0,  # steps allowed before inactivity penalty

        # Holding (steps-based)
        holding_threshold_steps: int = 0, # after this many steps, accrue holding penalty
        holding_penalty_per_step: float = 0.01,

        # Overexposure (risk budget in "R units"; each open trade ~= 1R risk)
        risk_budget_R: float = 2.0,
        overexposure_weight: float = 0.1,

        # Position conflict (same symbol long+short simultaneously)
        conflict_weight: float = 0.2,

        # Anti-churn: penalize bursts of tiny-R trades
        churn_count_threshold: int = 3,
        churn_absR_threshold: float = 0.15,
        churn_weight: float = 0.1,

        # Component bounding & smoothing
        component_clip: float = 3.0,      # clip each component to ±component_clip
        final_clip: float = 5.0,          # optional final clamp
        smoothing_alpha: float = 0.0,     # 0 disables; otherwise EMA on total reward

        # Legacy knobs kept for compatibility (ignored or repurposed):
        max_open_trades: int = 2,         # no hard use; exposure handled by risk_budget_R
        require_sl_tp: bool = True,       # still enforced to shape behavior
        device: Optional[torch.device] = None,

        # ---------------- OPTIONAL COSTS (accepted for compatibility) ----------------
        # If your ENV already subtracts costs (your current setup), leave the defaults
        # and DO NOT enable integrate_costs_in_reward to avoid double-charging.
        slippage_per_unit: float = 0.0,
        commission_per_trade: float = 0.0,
        integrate_costs_in_reward: bool = False,

        # absorb unknown future kwargs without crashing
        **extra_kwargs: Any,
    ):
        super().__init__()

        self.initial_balance = float(initial_balance or INITIAL_BALANCE)
        if self.initial_balance < MIN_INITIAL_BALANCE:
            raise ValueError(f"Initial balance must be >= {MIN_INITIAL_BALANCE}")

        # Core config
        self.min_risk = float(min_risk)
        self.r_clip = float(r_clip)

        self.stats_alpha = float(stats_alpha)
        self.quality_weight = float(quality_weight)

        self.realized_R_weight = float(realized_R_weight)
        self.unrealized_weight = float(unrealized_weight)

        self.inactivity_weight = float(inactivity_weight)
        self.inactivity_grace_steps = int(inactivity_grace_steps)

        self.holding_threshold_steps = int(holding_threshold_steps)
        self.holding_penalty_per_step = float(holding_penalty_per_step)

        self.risk_budget_R = float(risk_budget_R)
        self.overexposure_weight = float(overexposure_weight)

        self.conflict_weight = float(conflict_weight)

        self.churn_count_threshold = int(churn_count_threshold)
        self.churn_absR_threshold = float(churn_absR_threshold)
        self.churn_weight = float(churn_weight)

        self.component_clip = float(component_clip)
        self.final_clip = float(final_clip)
        self.smoothing_alpha = float(smoothing_alpha)

        self.max_open_trades = int(max_open_trades)
        self.require_sl_tp = bool(require_sl_tp)

        # Per-instance state (isolated per env)
        self.register_buffer("ema_gross_profit", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("ema_gross_loss", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("previous_reward", torch.tensor(0.0, dtype=torch.float32))

        # Device
        self.device = device or torch.device("cpu")
        self.to(self.device)

        # ---- NEW: store cost config (default keeps existing behavior) ----
        self.slippage_per_unit = float(slippage_per_unit)
        self.commission_per_trade = float(commission_per_trade)
        self.integrate_costs_in_reward = bool(integrate_costs_in_reward)

    # --------------------------
    # Lifecycle
    # --------------------------
    def reset(self) -> None:
        """Clear smoothing and EMA state at episode start."""
        self.ema_gross_profit.zero_()
        self.ema_gross_loss.zero_()
        self.previous_reward.zero_()
        # --------------------------
    # Main
    # --------------------------
    def forward(
        self,
        closed_trades: List[Dict[str, float]],
        open_trades: Optional[List[Dict[str, float]]] = None,
        account_balance: float = 0.0,
        unrealized_pnl: float = 0.0,
        time_since_last_trade: float = 0.0,  # in **steps**
    ) -> torch.Tensor:
        """
        Compute step reward (scalar tensor).
        """
        device = self.device
        eps = float(EPS) if "EPS" in globals() else 1e-8

        # --- C1: realized R (sum of clipped R across trades closed this step) ---
        realized_r_list: List[float] = []
        gross_profit_step = 0.0
        gross_loss_step = 0.0

        if closed_trades:
            for t in closed_trades:
                # Use net PnL for reward if enabled
                pnl_raw = _safe_float(t.get("pnl", 0.0))
                fees = _safe_float(t.get("slippage", 0.0)) + _safe_float(t.get("commission", 0.0))
                pnl_eff = pnl_raw - fees if self.integrate_costs_in_reward else pnl_raw

                # Feed net PnL into the R-multiple computation
                t_for_r = dict(t)
                t_for_r["pnl"] = pnl_eff
                r = _r_multiple_from_trade(t_for_r, self.min_risk, self.r_clip)
                realized_r_list.append(r)

                # EMA quality also based on effective (net) PnL
                if pnl_eff > 0:
                    gross_profit_step += pnl_eff
                elif pnl_eff < 0:
                    gross_loss_step += -pnl_eff  # absolute
        realized_R_sum = float(sum(realized_r_list)) if realized_r_list else 0.0

        # --- EMA quality in [-1,1] (bounded profit factor proxy) ---
        # EMA update
        with torch.no_grad():
            gp_prev = float(self.ema_gross_profit.item())
            gl_prev = float(self.ema_gross_loss.item())
            gp_new = (1.0 - self.stats_alpha) * gp_prev + self.stats_alpha * gross_profit_step
            gl_new = (1.0 - self.stats_alpha) * gl_prev + self.stats_alpha * gross_loss_step
            self.ema_gross_profit.copy_(torch.tensor(gp_new, dtype=torch.float32, device=device))
            self.ema_gross_loss.copy_(torch.tensor(gl_new, dtype=torch.float32, device=device))

        # bounded quality score ∈ [-1, 1]
        gp = float(self.ema_gross_profit.item())
        gl = float(self.ema_gross_loss.item())
        quality = (gp - gl) / (gp + gl + eps)

        # --- C3: unrealized normalized by balance (optional, small weight) ---
        unreal_norm = float(unrealized_pnl) / max(self.initial_balance, eps)

        # --- C4: inactivity (steps-based) ---
        steps_since_last = max(float(time_since_last_trade), 0.0)
        extra_steps = max(0.0, steps_since_last - float(self.inactivity_grace_steps))
        inactivity_pen = -self.inactivity_weight * extra_steps

        # --- C5: holding penalty (per open trade beyond threshold steps) ---
        holding_pen_total = 0.0
        if open_trades:
            for t in open_trades:
                ht = _safe_float(t.get("holding_time", 0.0))
                if ht > self.holding_threshold_steps:
                    holding_pen_total -= self.holding_penalty_per_step * (ht - self.holding_threshold_steps)

        # --- C6: overexposure penalty (risk budget in R units) ---
        # Approximate each open trade as 1R of risk unless missing SL; if missing SL, treat as >1R to force discipline
        exposure_R = 0.0
        if open_trades:
            for t in open_trades:
                entry = _safe_float(t.get("entry_price", 0.0))
                sl = _safe_float(t.get("stop_loss", 0.0))
                vol = _safe_float(t.get("volume", 1.0)) or 1.0
                side = _trade_side(t)
                # if no SL, count as 2R risk; else ≈1R
                if sl <= 0.0 or entry <= 0.0 or side == 0:
                    exposure_R += 2.0 * abs(vol)
                else:
                    rd = _risk_distance(entry, sl, side)
                    exposure_R += (1.0 if rd > 0.0 else 2.0) * abs(vol)
        overexcess = max(0.0, exposure_R - self.risk_budget_R)
        overexposure_pen = -self.overexposure_weight * overexcess

        # --- C7: conflict penalty (same symbol hedged both ways) ---
        conflict_pen = 0.0
        if open_trades:
            sym2dirs: Dict[str, Set[int]] = {}
            for t in open_trades:
                sym = str(t.get("symbol", "_")).upper()
                d = _trade_side(t)
                if d == 0:
                    continue
                sym2dirs.setdefault(sym, set()).add(d)
            for sym, dirs in sym2dirs.items():
                if 1 in dirs and -1 in dirs:  # both directions open on same symbol
                    conflict_pen -= self.conflict_weight

        # --- C8: churn penalty (many tiny-R closes) ---
        churn_pen = 0.0
        if realized_r_list:
            small_r = [abs(r) for r in realized_r_list if abs(r) < self.churn_absR_threshold]
            if len(small_r) >= self.churn_count_threshold:
                churn_pen -= self.churn_weight

        # --- SL/TP discipline (optional shaping) ---
        sltp_pen = 0.0
        if self.require_sl_tp and open_trades:
            for t in open_trades:
                if _safe_float(t.get("stop_loss", 0.0)) <= 0.0 or _safe_float(t.get("take_profit", 0.0)) <= 0.0:
                    # small shaping push to attach SL/TP
                    sltp_pen -= 0.05

        # --------------------------
        # Compose components (each bounded), then sum
        # --------------------------
        def _clip(x: float, c: float = self.component_clip) -> float:
            if x > c:
                return c
            if x < -c:
                return -c
            return x

        C1 = _clip(self.realized_R_weight * realized_R_sum)
        C2 = _clip(self.quality_weight * quality)
        C3 = _clip(self.unrealized_weight * unreal_norm)
        C4 = _clip(inactivity_pen)
        C5 = _clip(holding_pen_total)
        C6 = _clip(overexposure_pen)
        C7 = _clip(conflict_pen)
        C8 = _clip(churn_pen + sltp_pen)

        total = C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8

        # Optional final clamp (wider than ±1 to keep gradients alive)
        if self.final_clip is not None and self.final_clip > 0:
            total = max(-self.final_clip, min(self.final_clip, total))

        # Optional smoothing (per-instance)
        if self.smoothing_alpha > 0.0:
            prev = float(self.previous_reward.item())
            smoothed = self.smoothing_alpha * total + (1.0 - self.smoothing_alpha) * prev
            self.previous_reward.copy_(_to_device(torch.tensor(smoothed, dtype=torch.float32), device))
            return torch.tensor(smoothed, dtype=torch.float32, device=device)

        return torch.tensor(total, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------
# Convenience wrappers (kept, but default to NO global reuse to avoid
# cross-env leakage; use RewardFunction per-env in training code).
# ---------------------------------------------------------------------

_default_rf_instance: Optional[RewardFunction] = None

def compute_event_reward(
    closed_trades: List[Dict[str, float]],
    open_trades: Optional[List[Dict[str, float]]],
    account_balance: float,
    unrealized_pnl: float,
    time_since_last_trade: float,
    reuse: bool = False,  # default changed to False to avoid global shared state (fixes #7)
    **kwargs,
) -> torch.Tensor:
    """
    Convenience wrapper. Prefer constructing a RewardFunction per env and
    calling it directly during training. If reuse=True, a module-level instance
    is used (not recommended for vectorized/multiprocess training).
    """
    global _default_rf_instance
    if reuse:
        if _default_rf_instance is None:
            _default_rf_instance = RewardFunction(**kwargs)
        rf = _default_rf_instance
    else:
        rf = RewardFunction(**kwargs)
    return rf(closed_trades, open_trades, account_balance, unrealized_pnl, time_since_last_trade)


def compute_event_reward_gpu(
    open_trades: List[Dict[str, float]],
    bid: float,
    ask: float,
    initial_balance: Optional[float] = None,
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    GPU-friendly raw open-trade PnL with optional normalization.
    (Kept for tooling convenience; does not affect training reward.)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    n = len(open_trades)
    if n == 0:
        return torch.tensor(0.0, dtype=dtype, device=device)

    sides = torch.tensor(
        [1.0 if _trade_side(t) > 0 else -1.0 for t in open_trades],
        dtype=dtype, device=device
    )
    entry = torch.tensor([_safe_float(t.get("entry_price", 0.0)) for t in open_trades], dtype=dtype, device=device)
    vol = torch.tensor([_safe_float(t.get("volume", 0.0)) for t in open_trades], dtype=dtype, device=device)
    price_now = torch.tensor([bid if s > 0 else ask for s in sides], dtype=dtype, device=device)
    pnl = vol * (price_now - entry) * sides
    if normalize:
        initial = float(initial_balance or INITIAL_BALANCE)
        pnl = pnl / (initial + (float(EPS) if "EPS" in globals() else 1e-8))
    return pnl.sum()
