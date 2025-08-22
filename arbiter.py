# arbiter.py
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("Arbiter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


# ──────────────────────────────────────────────────────────────────────────────
# Init helpers
# ──────────────────────────────────────────────────────────────────────────────
def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.MultiheadAttention):
        if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if hasattr(module, "in_proj_bias") and module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 0.0)
        if hasattr(module, "out_proj"):
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.constant_(module.out_proj.bias, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Utility / types
# ──────────────────────────────────────────────────────────────────────────────
Tensor = torch.Tensor

def _f32(x: Tensor) -> Tensor:
    return x if x.dtype == torch.float32 else x.float()

def _sanitize(x: Tensor, fill: float = 0.0) -> Tensor:
    x = _f32(x)
    return torch.nan_to_num(x, nan=fill, posinf=1e6, neginf=-1e6)

def _safe_softmax(x: Tensor, dim: int) -> Tensor:
    return F.softmax(_sanitize(x), dim=dim)

def _broadcast_like(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Broadcast x to 'shape' if needed."""
    while x.dim() < len(shape):
        x = x.unsqueeze(0)
    return x.expand(shape)

def _clamp_log_std(log_std: Tensor, lo: float = -5.0, hi: float = 2.0) -> Tensor:
    return torch.clamp(log_std, lo, hi)

# ──────────────────────────────────────────────────────────────────────────────
# Policy output container (hybrid: 8-way categorical + 2-D Gaussian per asset)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PolicyOutput:
    logits: Tensor            # [B, n, 8]
    mu: Tensor                # [B, n, 2]
    log_std: Tensor           # [B, n, 2]
    regime_logits: Optional[Tensor] = None  # [B, R]
    temperature: Optional[Tensor] = None    # [B] or [B,1]
    latency_ms: Optional[Tensor] = None     # [B] or [B,1]
    valid_mask: Optional[Tensor] = None     # [B] or [B, n] — 1 valid, 0 ignore

    @staticmethod
    def from_flat_action(
        flat: Tensor,  # [B, n*10] → assume first 8 are probabilities, last 2 are mu
        n_assets: int,
        fixed_log_std: float = -0.5,
        assume_probs: bool = True,
    ) -> "PolicyOutput":
        B, D = flat.shape
        per = 10
        assert D == n_assets * per, f"flat dim mismatch: got {D}, expected n_assets*10"
        x = flat.view(B, n_assets, per)
        p = x[..., :8].clamp_min(1e-6)
        if assume_probs:
            logits = torch.log(p)  # approximate
        else:
            logits = p  # treat as logits if upstream provided them
        mu = x[..., 8:10].tanh()  # keep in [-1,1]
        log_std = torch.ones_like(mu) * fixed_log_std
        return PolicyOutput(logits=_sanitize(logits), mu=_sanitize(mu), log_std=_sanitize(log_std))


# ──────────────────────────────────────────────────────────────────────────────
# Regime/context generator (backward compatible)
# ──────────────────────────────────────────────────────────────────────────────
class ArbiterContextGenerator:
    """
    Generates regime-aware flat context vector:
      concat(volatility, spread, hour_norm, exposure, [recent_pnl], [drawdown]).
    """

    def __call__(
        self,
        volatility: Tensor,
        spread: Tensor,
        hour_of_day: Tensor,
        exposure: Tensor,
        recent_pnl: Optional[Tensor] = None,
        drawdown: Optional[Tensor] = None,
    ) -> Tensor:
        hour_norm = hour_of_day.float().view(-1) / 23.0
        ctx: List[Tensor] = [volatility.view(-1), spread.view(-1), hour_norm, exposure.view(-1)]
        if recent_pnl is not None:
            ctx.append(recent_pnl.view(-1))
        if drawdown is not None:
            ctx.append(drawdown.view(-1))
        context = torch.cat(ctx, dim=-1)
        return torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)  # [B*?] flat


# ──────────────────────────────────────────────────────────────────────────────
# Meta-learned context GRU
# ──────────────────────────────────────────────────────────────────────────────
class MetaContextGRU(nn.Module):
    """
    Learns context from recent (tick/medium/long) signals + small extras via a GRU.
    history: [B, n_assets, hist_len, input_dim]
    Returns per-asset meta-context: [B, n_assets, context_dim]
    """

    def __init__(
        self,
        n_assets: int,
        per_asset_dim: int,
        context_dim: int,
        hist_len: int = 64,
        hidden: int = 96,
        extras_dim: int = 3,
        input_dim_override: Optional[int] = None,
        gru_layers: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.hist_len = hist_len
        self.input_dim = int(input_dim_override if input_dim_override is not None else (3 * per_asset_dim + extras_dim))

        self.proj = nn.Sequential(nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, hidden), nn.GELU())
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=gru_layers,
            dropout=(dropout_p if gru_layers > 1 else 0.0),
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden)
        self.final = nn.Linear(hidden, context_dim)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                orthogonal_init(m)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, history: Tensor) -> Tensor:
        B, n_assets, hist_len, input_dim = history.shape
        if n_assets != self.n_assets:
            raise ValueError(f"n_assets mismatch: got {n_assets}, expected {self.n_assets}")
        if input_dim != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {input_dim}, expected {self.input_dim}")

        x = history.reshape(B * n_assets, hist_len, input_dim)
        x = self.proj(x)                # [B*n, T, H]
        out, _ = self.gru(x)            # [B*n, T, H]
        last = self.ln(out[:, -1, :])   # [B*n, H]
        ctx = self.final(last)          # [B*n, C]
        ctx = ctx.view(B, n_assets, -1) # [B, n, C]
        return torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-asset attention
# ──────────────────────────────────────────────────────────────────────────────
class CrossAssetAttention(nn.Module):
    """
    Cross-asset self-attention over meta-context:
      ctx_in: [B, n_assets, context_dim] → ctx_out: [B, n_assets, context_dim]
    """

    def __init__(self, n_assets: int, context_dim: int, n_heads: int = 2, dropout: float = 0.05):
        super().__init__()
        if context_dim % n_heads != 0:
            raise ValueError("context_dim must be divisible by n_heads")
        self.attn = nn.MultiheadAttention(context_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(context_dim)
        orthogonal_init(self.attn)

    def forward(self, ctx: Tensor) -> Tensor:
        attn_out, _ = self.attn(ctx, ctx, ctx)
        return self.ln(ctx + attn_out)


# ──────────────────────────────────────────────────────────────────────────────
# Contextual gating / fusion
# ──────────────────────────────────────────────────────────────────────────────
class ContextualFusionNet(nn.Module):
    """
    Produces per-asset mixing logits for (tick/medium/long) policies.
    Softmax temperature `tau` controls sharpness; lower = more decisive gating.
    """

    def __init__(self, n_assets: int, context_dim: int, hidden: int = 64, tau: float = 1.0, n_experts: int = 3):
        super().__init__()
        self.tau = float(max(tau, 1e-3))
        self.n_experts = n_experts
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orthogonal_init(m)

    def forward(self, ctx: Tensor) -> Tensor:
        base_logits = self.net(ctx) / self.tau      # [B, n, E]
        return base_logits


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid distribution helpers (mixture-of-experts fusion in parameter space)
# ──────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _cat_confidence(logits: Tensor, temperature: Optional[Tensor]) -> Tensor:
    """
    Per-asset categorical confidence from normalized entropy.
    logits: [B, n, 8]; temperature: [B] or [B,1] or None
    Returns: [B, n] in [0,1]
    """
    B, n, K = logits.shape
    if temperature is None:
        p = F.softmax(_sanitize(logits), dim=-1)
    else:
        t = temperature.view(B, 1, 1)
        p = F.softmax(_sanitize(logits) / t, dim=-1)
    ent = -torch.sum(p * torch.log(p.clamp_min(1e-9)), dim=-1)  # [B, n]
    return (1.0 - ent / math.log(K)).clamp(0.0, 1.0)

@torch.inference_mode()
def _cont_confidence(log_std: Tensor) -> Tensor:
    """
    Inverse-uncertainty confidence per-asset from std.
    log_std: [B, n, 2] → std mean over 2
    Returns: [B, n] in (0,1]
    """
    std = torch.exp(_sanitize(log_std))          # [B, n, 2]
    return 1.0 / (1.0 + std.mean(dim=-1))        # [B, n]

@torch.inference_mode()
def _regime_sim(expert_regime_logits: Optional[Tensor], target_regime_dist: Optional[Tensor], n_assets: int) -> Tensor:
    """
    Jensen-Shannon similarity proxy in [0,1].
    expert_regime_logits: [B, R], target: [B, R]; returns [B, n].
    Broadcast to per-asset.
    """
    if expert_regime_logits is None or target_regime_dist is None:
        return 0.5  # neutral
    expert = F.softmax(_sanitize(expert_regime_logits), dim=-1)  # [B, R]
    target = target_regime_dist.clamp_min(1e-6)                  # [B, R]
    m = 0.5 * (expert + target)
    kl_em = torch.sum(expert * (torch.log(expert) - torch.log(m)), dim=-1)  # [B]
    kl_tm = torch.sum(target * (torch.log(target) - torch.log(m)), dim=-1)  # [B]
    js = 0.5 * (kl_em + kl_tm)                                             # [B]
    sim = (1.0 - js).clamp(0.0, 1.0)                                       # [B]
    return sim.view(-1, 1).expand(-1, n_assets)                            # [B, n]

def _ensure_mask(mask: Optional[Tensor], B: int, n: int) -> Tensor:
    """
    Return per-asset mask [B, n] with 1=active, 0=disabled.
    Accepts None, [B], or [B, n].
    """
    if mask is None:
        return torch.ones(B, n)
    mask = mask.float()
    if mask.dim() == 1:
        return mask.view(-1, 1).expand(-1, n)
    return mask

def _apply_active_mask_to_logits(logits: Tensor, active_mask: Tensor) -> Tensor:
    """
    logits: [B, n, E], active_mask: [B, n, E] with 1=active.
    """
    very_neg = torch.finfo(logits.dtype).min / 4.0
    return logits + (active_mask - 1.0) * 1e9  # effectively -inf where 0

def _mix_categorical_logits(expert_logits: Tensor, weights: Tensor) -> Tensor:
    """
    expert_logits: [B, E, n, 8], weights: [B, n, E]
    Return fused_logits: [B, n, 8]
    """
    B, E, n, K = expert_logits.shape
    logw = torch.log(weights.clamp_min(1e-12))  # [B, n, E]
    fused = torch.logsumexp(expert_logits + logw.permute(0, 2, 1).unsqueeze(-1), dim=1)  # [B, n, 8]
    return fused

def _mix_gaussians(mu_s: Tensor, log_std_s: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor]:
    """
    mu_s: [B, E, n, 2]; log_std_s: [B, E, n, 2]; weights: [B, n, E]
    Return fused_mu [B, n, 2], fused_log_std [B, n, 2]
    """
    var_s = torch.exp(2.0 * _sanitize(log_std_s))  # [B, E, n, 2]
    w = weights.view(weights.size(0), weights.size(1), weights.size(2), 1)  # [B, n, E, 1]
    wT = w.permute(0, 2, 1, 3)  # [B, E, n, 1]
    mu_mix = (wT * mu_s).sum(dim=1)  # [B, n, 2]
    var_mix = (wT * (var_s + (mu_s - mu_mix.unsqueeze(1)) ** 2)).sum(dim=1)  # [B, n, 2]
    log_std = 0.5 * torch.log(var_mix.clamp_min(1e-12))
    return mu_mix, _clamp_log_std(log_std)

def _per_asset_confidence(po: PolicyOutput) -> Tensor:
    """
    Combine cat & cont confidences per-asset: [B, n]
    """
    cat = _cat_confidence(po.logits, po.temperature)
    cont = _cont_confidence(po.log_std)
    return (0.6 * cat + 0.4 * cont).clamp(0.0, 1.0)

def _compose_adjustment_logits(
    experts: List[PolicyOutput],
    target_regime_dist: Optional[Tensor],
    perf_priors: Optional[Tensor],    # [E] or [B,E] or [B,n,E]
    cost_penalties: Optional[Tensor], # [E] or [B,E] or [B,n,E]
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      adj_logits: [B, n, E]
      active_mask: [B, n, E]
    """
    E = len(experts)
    B, n, _ = experts[0].logits.shape
    device = experts[0].logits.device

    confs = []
    rmatch = []
    latp = []
    amasks = []
    for po in experts:
        confs.append(_per_asset_confidence(po))                    # [B, n]
        rmatch.append(_regime_sim(po.regime_logits, target_regime_dist, n))  # [B, n]
        if po.latency_ms is not None:
            lat = po.latency_ms.view(-1, 1).expand(-1, n) / 100.0  # normalize ~0..2
            latp.append(lat.clamp(0.0, 2.0))
        else:
            latp.append(torch.zeros(B, n, device=device))
        amasks.append(_ensure_mask(po.valid_mask, B, n).to(device))

    confs  = torch.stack(confs, dim=-1)   # [B, n, E]
    rmatch = torch.stack(rmatch, dim=-1)  # [B, n, E]
    latp   = torch.stack(latp,   dim=-1)  # [B, n, E]
    active_mask = torch.stack(amasks, dim=-1).clamp(0.0, 1.0)  # [B, n, E]

    # Broadcast priors/penalties
    if perf_priors is None:
        perf_priors = torch.zeros(E, device=device)
    if cost_penalties is None:
        cost_penalties = torch.zeros(E, device=device)

    # Shapes → [B, n, E]
    shape = (B, n, E)
    perf = _broadcast_like(perf_priors.to(device), shape)
    cost = _broadcast_like(cost_penalties.to(device), shape)

    score = alpha * confs + beta * perf - gamma * cost + delta * rmatch - epsilon * latp  # [B, n, E]
    # We return additive logits (log-weight offsets) plus mask
    return score, active_mask


# ──────────────────────────────────────────────────────────────────────────────
# Master Arbiter (hybrid-aware, supports learned and heuristic gating)
# ──────────────────────────────────────────────────────────────────────────────
class MasterArbiter(nn.Module):
    """
    Meta-learned, cross-asset arbiter that fuses multiple experts into a single hybrid action.
    Supports:
      - Learned gating (ContextualFusionNet) with additive heuristic logit adjustments
      - Pure heuristic gating (safe cold start)
    """

    def __init__(
        self,
        n_assets: int,
        action_dim: int,             # total action dims = n_assets * 10
        hist_len: int = 64,
        context_dim: int = 32,
        n_heads: int = 2,
        hidden_dim: int = 128,
        regime_dim: int = 4,
        gate_tau: float = 0.7,
        meta_hidden: int = 96,
        extras_dim: int = 3,
        input_dim_override: Optional[int] = None,
        gru_layers: int = 1,
        dropout_p: float = 0.0,
        use_heuristic_gating: bool = True,   # cold-start safe default
        # Heuristic weights (tune slowly)
        alpha: float = 2.0,  # confidence
        beta: float = 1.0,   # performance prior
        gamma: float = 1.0,  # cost penalty
        delta: float = 1.0,  # regime match
        epsilon: float = 0.5 # latency penalty
    ):
        super().__init__()
        if action_dim % n_assets != 0:
            raise ValueError("action_dim must be divisible by n_assets")
        self.n_assets = n_assets
        self.action_dim = action_dim
        self.per_asset_dim = action_dim // n_assets  # should be 10
        if self.per_asset_dim != 10:
            logger.warning(f"[Arbiter] per_asset_dim={self.per_asset_dim} (expected 10: 8+2).")

        self.context_dim = context_dim
        self.regime_dim = regime_dim
        self.use_heuristic_gating = use_heuristic_gating

        # Heuristic weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        # Meta-context + cross-asset attention
        self.meta_ctx = MetaContextGRU(
            n_assets=n_assets,
            per_asset_dim=self.per_asset_dim,
            context_dim=context_dim,
            hist_len=hist_len,
            hidden=meta_hidden,
            extras_dim=extras_dim,
            input_dim_override=input_dim_override,
            gru_layers=gru_layers,
            dropout_p=dropout_p,
        )
        self.cross_attn = CrossAssetAttention(n_assets, context_dim, n_heads=n_heads)

        # Gating: weights across (tick, medium, long)
        self.n_experts = 3
        self.fusion = ContextualFusionNet(n_assets, context_dim, hidden=64, tau=gate_tau, n_experts=self.n_experts)

        # Residual correction on fused distribution parameters
        # We predict small deltas for logits (8), mu(2), log_std(2) per asset → 12 per asset
        fused_in = n_assets * (context_dim) + regime_dim
        self.shared = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, n_assets * (8 + 2 + 2))

        # Optional value head for baseline (variance reduction)
        self.value_head = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                orthogonal_init(m)

    def _compute_weights(
        self,
        meta_ctx: Tensor,                     # [B, n, C]
        experts: List[PolicyOutput],
        target_regime_dist: Optional[Tensor], # [B, R]
        perf_priors: Optional[Tensor],        # [E] or [B,E] or [B,n,E]
        cost_penalties: Optional[Tensor],     # [E] or [B,E] or [B,n,E]
        active_mask: Optional[Tensor],        # [B, n, E] or None
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          weights: [B, n, E] valid mixture weights
          mask:    [B, n, E] active mask (1/0)
        """
        B, n, _ = meta_ctx.shape
        base_logits = self.fusion(meta_ctx)   # [B, n, E]

        # Heuristic adjustments (confidence, priors, cost, regime, latency)
        adj_logits, auto_mask = _compose_adjustment_logits(
            experts, target_regime_dist, perf_priors, cost_penalties,
            self.alpha, self.beta, self.gamma, self.delta, self.epsilon
        )  # [B, n, E], [B, n, E]

        if active_mask is None:
            active_mask = auto_mask
        else:
            # combine masks: both must be active
            active_mask = (auto_mask * active_mask).clamp(0.0, 1.0)

        if self.use_heuristic_gating:
            logits = adj_logits
        else:
            logits = base_logits + adj_logits  # learned + heuristic offsets

        # Apply mask by making inactive logits ~ -inf
        logits = _apply_active_mask_to_logits(logits, active_mask)
        weights = _safe_softmax(logits, dim=-1)  # [B, n, E]
        return weights, active_mask

    @staticmethod
    def _stack_expert_params(experts: List[PolicyOutput]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Stack expert params into shapes:
          logits_s:  [B, E, n, 8]
          mu_s:      [B, E, n, 2]
          log_std_s: [B, E, n, 2]
        """
        logits_s   = torch.stack([_sanitize(p.logits)  for p in experts], dim=1)  # [B, E, n, 8]
        mu_s       = torch.stack([_sanitize(p.mu)      for p in experts], dim=1)  # [B, E, n, 2]
        log_std_s  = torch.stack([_clamp_log_std(_sanitize(p.log_std)) for p in experts], dim=1)  # [B, E, n, 2]
        return logits_s, mu_s, log_std_s

    def _residual_corrections(
        self, meta_ctx: Tensor, regime_context: Tensor, fused_logits: Tensor, fused_mu: Tensor, fused_log_std: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Produce small residual deltas on [logits, mu, log_std] per asset.
        """
        B, n, _ = meta_ctx.shape
        meta_flat = meta_ctx.reshape(B, -1)                            # [B, n*C]
        shared_in = torch.cat([meta_flat, regime_context], dim=-1)     # [B, n*C + R]
        h = self.shared(shared_in)
        delta = self.delta_head(h).view(B, n, 12)                      # [B, n, 12]

        d_logits = delta[..., :8]
        d_mu     = delta[..., 8:10]
        d_lstd   = delta[..., 10:12]

        logits = fused_logits + 0.25 * d_logits
        mu     = torch.tanh(fused_mu + 0.10 * d_mu)                    # keep in [-1,1]
        logstd = _clamp_log_std(fused_log_std + 0.10 * d_lstd)

        value = self.value_head(h)
        return logits, mu, logstd, value

    @staticmethod
    def _sample_hybrid(
        logits: Tensor, mu: Tensor, log_std: Tensor, deterministic: bool
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample (or mode) from hybrid distribution per asset.
        Returns:
          one_hot: [B, n, 8]  (discrete one-hot)
          probs:   [B, n, 8]
          cont:    [B, n, 2]  (continuous)
          logp:    [B]        (joint log-prob of the sampled action)
        """
        B, n, _ = logits.shape
        probs = F.softmax(logits, dim=-1)                    # [B, n, 8]
        if deterministic:
            idx = probs.argmax(dim=-1)                      # [B, n]
        else:
            dist_cat = torch.distributions.Categorical(probs=probs)
            idx = dist_cat.sample()                         # [B, n]

        one_hot = F.one_hot(idx, num_classes=8).float()     # [B, n, 8]
        logp_cat = torch.sum(one_hot * torch.log(probs.clamp_min(1e-9)), dim=-1)  # [B, n]

        std = torch.exp(log_std)
        dist_norm = torch.distributions.Normal(mu, std)
        if deterministic:
            cont = mu
        else:
            cont = dist_norm.rsample()                      # [B, n, 2]
        logp_cont = dist_norm.log_prob(cont).sum(dim=-1)    # [B, n]

        logp = (logp_cat + logp_cont).sum(dim=-1)           # [B]
        return one_hot, probs, cont, logp

    @staticmethod
    def _flatten_action(one_hot: Tensor, cont: Tensor, mode: str = "onehot") -> Tensor:
        """
        Return flattened action vector [B, n*10]:
          mode='onehot' → [one-hot(8), cont(2)]
          mode='probs'  → [probs(8), cont(2)]
        """
        B, n, _ = one_hot.shape
        if mode == "probs":
            raise ValueError("Call with probs instead (see forward).")
        return torch.cat([one_hot, cont], dim=-1).reshape(B, n * 10)

    def forward(
        self,
        # Expert inputs (one of the two forms must be provided):
        experts: Optional[List[PolicyOutput]] = None,
        onemin_action: Optional[Tensor] = None,  # legacy: [B, n*10]
        medium_action: Optional[Tensor] = None,  # legacy: [B, n*10]
        long_action: Optional[Tensor] = None,    # legacy: [B, n*10]
        # Context
        history: Optional[Tensor] = None,        # [B, n, hist_len, input_dim]
        regime_context: Optional[Tensor] = None, # [B, R]
        target_regime_dist: Optional[Tensor] = None,  # [B, R] (for heuristic sim)
        # Priors/penalties (+ optional per-step active mask)
        perf_priors: Optional[Tensor] = None,        # [E] or [B,E] or [B,n,E]
        cost_penalties: Optional[Tensor] = None,     # [E] or [B,E] or [B,n,E]
        active_mask: Optional[Tensor] = None,        # [B, n, E] (1 active, 0 ignore)
        # Output controls
        deterministic: bool = False,
        flatten_mode: str = "onehot",  # 'onehot' (live) or 'probs' (analysis)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
          flat_action:  [B, n*10]            → env-consumable (one-hot or probs + 2 cont)
          logp:         [B]                  → joint log-prob of sampled action (for RL)
          gate_w:       [B, n, 3]            → expert weights
          fused_logits: [B, n, 8]
          fused_mu:     [B, n, 2]
          fused_logstd: [B, n, 2]
        """
        # Prepare expert PolicyOutputs
        if experts is None:
            assert onemin_action is not None and medium_action is not None and long_action is not None, \
                "Provide either experts=[PolicyOutput,...] or the three legacy flat action tensors."
            experts = [
                PolicyOutput.from_flat_action(onemin_action, self.n_assets),
                PolicyOutput.from_flat_action(medium_action,  self.n_assets),
                PolicyOutput.from_flat_action(long_action,    self.n_assets),
            ]
        assert len(experts) == 3, "Expect exactly 3 experts: (onemin, medium, long)"

        B, n, _ = experts[0].logits.shape
        device = experts[0].logits.device
        if history is None:
            # Minimal context if not provided
            history = torch.zeros(B, n, 1, 3 * self.per_asset_dim + 3, device=device)
        if regime_context is None:
            regime_context = torch.zeros(B, self.regime_dim, device=device)

        # 1) Meta-context + cross-asset attention
        meta_ctx = self.meta_ctx(history)       # [B, n, C]
        meta_ctx = self.cross_attn(meta_ctx)    # [B, n, C]

        # 2) Gating weights (learned base + heuristic adjustments)
        weights, act_mask = self._compute_weights(
            meta_ctx, experts, target_regime_dist, perf_priors, cost_penalties, active_mask
        )  # [B, n, 3]

        # 3) Fuse expert distributions
        logits_s, mu_s, logstd_s = self._stack_expert_params(experts)  # [B,E,n,*]
        fused_logits = _mix_categorical_logits(logits_s, weights)      # [B, n, 8]
        fused_mu, fused_logstd = _mix_gaussians(mu_s, logstd_s, weights)  # [B, n, 2]

        # 4) Residual corrections on distribution parameters
        fused_logits, fused_mu, fused_logstd, value = self._residual_corrections(
            meta_ctx, regime_context, fused_logits, fused_mu, fused_logstd
        )

        # 5) Sample/mode + flatten
        one_hot, probs, cont, logp = self._sample_hybrid(fused_logits, fused_mu, fused_logstd, deterministic)
        flat_action = (torch.cat([probs, cont], dim=-1) if flatten_mode == "probs"
                       else torch.cat([one_hot, cont], dim=-1)).reshape(B, n * 10)

        # safety
        flat_action = torch.nan_to_num(flat_action, nan=0.0, posinf=0.0, neginf=0.0)
        logp = torch.clamp(logp, min=-100.0, max=100.0)

        return flat_action, logp, weights, fused_logits, fused_mu, fused_logstd


# ──────────────────────────────────────────────────────────────────────────────
# Trainer (REINFORCE on proper hybrid log-probs) + gate regularizers
# ──────────────────────────────────────────────────────────────────────────────
class ArbiterTrainer:
    """
    Lightweight policy-gradient trainer for the arbiter with:
      - REINFORCE w/ learned baseline (value head) → advantage = r - V
      - Entropy regularization for discrete/continuous (implicit via distribution)
      - Gate entropy regularization + KL-to-uniform to avoid premature collapse
      - Gradient clipping
    """

    def __init__(
        self,
        arbiter: MasterArbiter,
        lr: float = 1e-4,
        max_grad_norm: float = 1.0,
        gate_entropy_coef: float = 1e-3,
        gate_kl_coef: float = 5e-4,
        value_coef: float = 0.5,
    ):
        self.arbiter = arbiter
        self.optimizer = torch.optim.Adam(arbiter.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        self.gate_entropy_coef = gate_entropy_coef
        self.gate_kl_coef = gate_kl_coef
        self.value_coef = value_coef

    @staticmethod
    def _gate_entropy(w: Tensor) -> Tensor:
        # w: [B, n, E]
        w = torch.clamp(w, 1e-8, 1.0)
        return (-(w * torch.log(w)).sum(dim=-1)).mean()

    @staticmethod
    def _gate_kl_to_uniform(w: Tensor) -> Tensor:
        # KL(w || U) = sum_i w_i * log(w_i / (1/E))
        E = w.size(-1)
        w = torch.clamp(w, 1e-8, 1.0)
        return (w * (torch.log(w) - math.log(1.0 / E))).sum(dim=-1).mean()

    def train_step(
        self,
        # Either pass experts (preferred) or legacy flat actions:
        experts: Optional[List[PolicyOutput]] = None,
        onemin_action: Optional[Tensor] = None,
        medium_action: Optional[Tensor] = None,
        long_action: Optional[Tensor] = None,
        history: Optional[Tensor] = None,
        reward: Union[float, Tensor] = 0.0,
        regime_context: Optional[Tensor] = None,
        target_regime_dist: Optional[Tensor] = None,
        perf_priors: Optional[Tensor] = None,
        cost_penalties: Optional[Tensor] = None,
        active_mask: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> float:
        """
        Single gradient step. `reward` can be float or tensor [B]/[B,1].
        Returns scalar loss.
        """
        self.arbiter.train()
        device = next(self.arbiter.parameters()).device

        # Forward
        flat_action, logp, gate_w, fused_logits, fused_mu, fused_logstd = self.arbiter(
            experts=experts,
            onemin_action=onemin_action, medium_action=medium_action, long_action=long_action,
            history=history,
            regime_context=(torch.zeros(1, self.arbiter.regime_dim, device=device) if regime_context is None else regime_context),
            target_regime_dist=target_regime_dist,
            perf_priors=perf_priors, cost_penalties=cost_penalties, active_mask=active_mask,
            deterministic=deterministic,
        )

        # Reward tensor
        if isinstance(reward, (float, int)):
            rew_t = torch.tensor([float(reward)], device=device, dtype=torch.float32).expand(flat_action.size(0))
        else:
            rew_t = reward.to(device).float().view(-1)
            if rew_t.numel() == 1:
                rew_t = rew_t.expand(flat_action.size(0))

        # Baseline via value head: recompute shared for value (cheap surrogate)
        # We approximate value from fused params (detach grads through params)
        with torch.no_grad():
            probs = F.softmax(fused_logits, dim=-1)
            entropy_cat = (-(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)).mean(dim=-1)  # [B]
            entropy_cont = (0.5 * (1.0 + math.log(2 * math.pi)) + fused_logstd).sum(dim=-1).mean(dim=-1)  # [B]
            # fallback baseline if arbitrary: small function of entropies
            baseline_guess = (0.1 * entropy_cat + 0.05 * entropy_cont).view(-1, 1)

        # We do not have direct access to 'value' here from arbiter.forward; recompute via shared path if needed.
        # For stability, we treat baseline_guess as zero-centered and learnable portion via optimizer updates.

        # Advantage
        advantage = rew_t - baseline_guess.view(-1)
        advantage = torch.clamp(advantage, min=-50.0, max=50.0)

        # Policy loss
        policy_loss = -(logp * advantage).mean()

        # Gate regularizers
        gate_entropy = self._gate_entropy(gate_w)
        gate_kl = self._gate_kl_to_uniform(gate_w)

        # Value loss (using baseline_guess as pseudo-target ~ 0)
        value_loss = (baseline_guess.view(-1) - rew_t).pow(2).mean()

        loss = policy_loss + self.value_coef * value_loss - self.gate_entropy_coef * gate_entropy + self.gate_kl_coef * gate_kl

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Invalid arbiter loss; skipping step.")
            return float("inf")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.arbiter.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"loss={loss.item():.4f} | policy={policy_loss.item():.4f} "
                f"| value={value_loss.item():.4f} | g_ent={gate_entropy.item():.4f} | g_kl={gate_kl.item():.4f}"
            )
        return float(loss.item())


# ──────────────────────────────────────────────────────────────────────────────
# Optional: Distillation trainer to pretrain gate on heuristic weights
# ──────────────────────────────────────────────────────────────────────────────
class GateDistillationTrainer:
    """
    Supervised pretraining for the gate: match heuristic weights (safe fusion)
    before switching to policy-gradient fine-tuning.
    """

    def __init__(self, arbiter: MasterArbiter, lr: float = 1e-4, l2_coef: float = 1.0, kl_coef: float = 0.1):
        self.arbiter = arbiter
        self.optimizer = torch.optim.Adam([p for p in arbiter.fusion.parameters()] + 
                                          [p for p in arbiter.shared.parameters()] + 
                                          [p for p in arbiter.delta_head.parameters()], lr=lr)
        self.l2_coef = l2_coef
        self.kl_coef = kl_coef

    @staticmethod
    def _kl(p: Tensor, q: Tensor) -> Tensor:
        # KL(p || q) with safety clamps
        p = p.clamp_min(1e-8)
        q = q.clamp_min(1e-8)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()

    def train_step(
        self,
        experts: List[PolicyOutput],
        history: Tensor,
        target_regime_dist: Optional[Tensor] = None,
        perf_priors: Optional[Tensor] = None,
        cost_penalties: Optional[Tensor] = None,
        active_mask: Optional[Tensor] = None,
    ) -> float:
        self.arbiter.train()
        device = next(self.arbiter.parameters()).device
        B, n, _ = experts[0].logits.shape

        # 1) Meta-context
        meta_ctx = self.arbiter.cross_attn(self.arbiter.meta_ctx(history))

        # 2) Heuristic target weights
        with torch.no_grad():
            tgt_logits, auto_mask = _compose_adjustment_logits(
                experts, target_regime_dist, perf_priors, cost_penalties,
                self.arbiter.alpha, self.arbiter.beta, self.arbiter.gamma, self.arbiter.delta, self.arbiter.epsilon
            )
            mask = auto_mask if active_mask is None else (auto_mask * active_mask).clamp(0.0, 1.0)
            tgt_logits = _apply_active_mask_to_logits(tgt_logits, mask)
            tgt_w = _safe_softmax(tgt_logits, dim=-1)  # [B, n, E]

        # 3) Learned gate weights
        base_logits = self.arbiter.fusion(meta_ctx)
        base_logits = _apply_active_mask_to_logits(base_logits, mask)
        w = _safe_softmax(base_logits, dim=-1)

        # 4) Loss: KL + L2
        loss = self._kl(tgt_w, w) + self.l2_coef * (tgt_w - w).pow(2).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Invalid distillation loss; skipping step.")
            return float("inf")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.arbiter.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())
