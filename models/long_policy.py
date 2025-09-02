# models/long_policy.py
from __future__ import annotations
import math
import logging
from typing import Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
torch.autograd.set_detect_anomaly(False)
from torch.utils.checkpoint import checkpoint
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution, SquashedDiagGaussianDistribution
try:
    from torch.nn.attention import sdpa_kernel
    with sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
        pass
except Exception:
    pass
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
logger = logging.getLogger("LongPolicy")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ──────────────────────────────────────────────────────────────────────────────
# Shared utils/blocks (mirrors long policy upgrades)
# ──────────────────────────────────────────────────────────────────────────────
def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(module.out_proj.weight)
        if module.out_proj.bias is not None:
            nn.init.constant_(module.out_proj.bias, 0.0)

def same_pad(kernel_size: int, dilation: int = 1) -> int:
    return ((kernel_size - 1) // 2) * dilation

class DSConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1, p: float = 0.0):
        super().__init__()
        pad = same_pad(k, dilation)
        self.dw = nn.Conv1d(in_ch, in_ch, k, padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(p)
        orthogonal_init(self.dw, gain=1.0)
        orthogonal_init(self.pw, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*, C_in, W]
        B, Cin, W = x.shape
        # Make sure both input AND output per chunk stay within 32-bit index math.
        # Keep a safety margin under 2^31-1.
        MAX_ELEMS = 1_500_000_000  # ~1.5e9 as a safe ceiling

        # Worst-case channels seen by kernels in this block
        Cout = self.pw.out_channels if hasattr(self.pw, "out_channels") else Cin
        per_row_in  = Cin * W
        per_row_out = max(Cin, Cout) * W
        per_row = max(per_row_in, per_row_out)

        # How many rows (batch items) can we process per chunk?
        max_rows = max(1, MAX_ELEMS // max(1, per_row))
        if B <= max_rows:
            y = self.dw(x); y = self.pw(y); y = self.bn(y)
            return self.drop(F.gelu(y))

        # Chunk along batch dim so each conv sees a small enough tensor
        chunks = int(math.ceil(B / max_rows))
        parts = []
        for xi in x.chunk(chunks, dim=0):
            yi = self.dw(xi); yi = self.pw(yi); yi = self.bn(yi)
            yi = self.drop(F.gelu(yi))
            parts.append(yi)
        return torch.cat(parts, dim=0)

class SE1d(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(1, ch // r))
        self.fc2 = nn.Linear(max(1, ch // r), ch)
        orthogonal_init(self.fc1); orthogonal_init(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2)          # [B, C]
        a = F.gelu(self.fc1(s))
        a = torch.sigmoid(self.fc2(a))
        return x * a.unsqueeze(-1)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int, dropout_p: float = 0.1):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout_p, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout_p)
        orthogonal_init(self.attn); orthogonal_init(self.ff[0]); orthogonal_init(self.ff[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype  # usually bfloat16 under AMP

        # ── MHA under bf16 (saves memory); don't return weights
        with torch.amp.autocast("cuda",enabled=True, dtype=torch.bfloat16):
            a, _ = self.attn(x, x, x, need_weights=False)
            x = x + self.drop(a)

        # ── LN1 in fp32 for stability
        with torch.amp.autocast("cuda",enabled=False):
            x = self.ln1(x.float())

        # ── FFN in bf16 to slash activation memory
        with torch.amp.autocast("cuda",enabled=True, dtype=torch.bfloat16):
            f = self.ff(x.to(torch.bfloat16))
            x = x + self.drop(f)

        # ── LN2 in fp32, then back to original dtype
        with torch.amp.autocast("cuda",enabled=False):
            x = self.ln2(x.float()).to(orig_dtype)

        return x

class CrossAssetAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.08):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        orthogonal_init(self.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype
        B = x.size(0)
        max_rows = int(os.environ.get("LONG_XATTN_MAX_ROWS", "4096"))
        outs = []
        for xi in x.split(max_rows, dim=0):
            with torch.amp.autocast("cuda", enabled=True):  # fp32 numerics
                y, _ = self.attn(xi.float(), xi.float(), xi.float(), need_weights=False)
                o = self.ln((xi.float() + y))
            outs.append(o.to(dtype_in))
        return torch.cat(outs, dim=0)


class AttnPool1D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(embed_dim))
        nn.init.normal_(self.q, std=0.02)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q / (x.size(-1) ** 0.5)
        w = torch.softmax((x * q).sum(dim=-1, keepdim=True), dim=1)
        return (w * x).sum(dim=1)

# ──────────────────────────────────────────────────────────────────────────────
#Long-term feature extractor (hourly)
# ──────────────────────────────────────────────────────────────────────────────
class LongTermFeatureExtractor(BaseFeaturesExtractor):
    """
    Drop-in replacement (keeps features_dim = n_assets * (embed_dim + 5)):
      • DS-TCN + Token Transformer + AttnPool + Cross-asset + SE (unchanged)
      • Extras redesigned for trend-change sensitivity (still 5 per-asset):
          1) mean_logret (trend)
          2) atr_pct     (scale-free range)
          3) macd_delta  (acceleration, normalized, bounded)
          4) ER_now      (efficiency ratio: trend vs noise)
          5) rolling_corr (vs base asset; confirmations/divergences)
    Obs per-asset: [open, high, low, close] * window + [elapsed]  (unchanged)
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        n_assets: int,
        window: int = 48,    
        embed_dim: int = 32,
        tcn_hidden: int = 32,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.10,
        **_: Any,
    ):
        obs_dim = int(observation_space.shape[0])
        assert obs_dim % n_assets == 0, f"Obs dim {obs_dim} not divisible by n_assets {n_assets}"
        per_asset = obs_dim // n_assets
        assert per_asset == 4 * window + 1, f"per_asset={per_asset} != 4*window+1={4*window+1}"

        # Keep features_dim identical to the old extractor: embed_dim + 5 per asset
        super().__init__(observation_space, features_dim=n_assets * (embed_dim + 5))
        self.n_assets = n_assets
        self.window = window
        self.per_asset = per_asset
        self.embed_dim = embed_dim
        self._features_dim = n_assets * (embed_dim + 5)

        self.in_ln = nn.LayerNorm(4)
        self.token = nn.Linear(4, embed_dim)

        self.tcn1 = DSConv1d(4, tcn_hidden, k=5, dilation=1, p=dropout * 0.5)
        self.tcn2 = DSConv1d(tcn_hidden, tcn_hidden, k=5, dilation=2, p=dropout * 0.5)
        self.tcn3 = DSConv1d(tcn_hidden, embed_dim, k=5, dilation=4, p=dropout * 0.5)
        self.se = SE1d(embed_dim, r=8)

        self.asset_pos = nn.Embedding(n_assets, embed_dim)
        self.time_pos = nn.Embedding(window, embed_dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, embed_dim * 4, dropout_p=dropout)
            for _ in range(n_layers)
        ])
        self.use_checkpoint = True  # saves VRAM during training
        self.max_tokens_per_chunk = int(os.environ.get("LONG_MAX_TOKENS", "120000"))

        self.time_pool = AttnPool1D(embed_dim)
        self.cross = CrossAssetAttention(embed_dim, n_heads, dropout)
        self.final_ln = nn.LayerNorm(embed_dim)
        self.final_drop = nn.Dropout(dropout)

        # Project 24 window-level stats → 5 extras per asset (keeps +5 contract)
        self.extras_proj = nn.Linear(24, 5)
        orthogonal_init(self.extras_proj, gain=1.0)

        # Keep output projection shape identical for head compatibility
        self.out_proj = nn.Linear(self._features_dim, self._features_dim)
        orthogonal_init(self.out_proj, gain=0.10)

        self.register_buffer("asset_idx", torch.arange(n_assets))
        self.register_buffer("time_idx", torch.arange(window))

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding, nn.MultiheadAttention)):
                orthogonal_init(m)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.size(0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # [B, N, per_asset], then slice bars -> [B, N, W, 4]
        obs = obs.reshape(B, self.n_assets, self.per_asset)
        bars = obs[:, :, : 4 * self.window].reshape(B, self.n_assets, self.window, 4)

        use_amp = torch.cuda.is_available()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            # Token path
            tok = self.token(self.in_ln(bars))  # [B,N,W,E]

            # DS-TCN path on channels-first then back
            x_cnn = bars.permute(0, 1, 3, 2).contiguous()         # [B,N,4,W]
            x_cnn = x_cnn.reshape(B * self.n_assets, 4, self.window)
            x_cnn = self.tcn3(self.tcn2(self.tcn1(x_cnn)))        # [B*N,E,W]
            x_cnn = self.se(x_cnn)
            x_cnn = x_cnn.permute(0, 2, 1).reshape(B, self.n_assets, self.window, self.embed_dim)

            # Fuse + positions
            # Fuse + positions (avoid big temporaries)
        fused = tok + x_cnn
        a = self.asset_pos(self.asset_idx).view(1, self.n_assets, 1, self.embed_dim)
        t = self.time_pos(self.time_idx).view(1, 1, self.window, self.embed_dim)
        # match AMP device & dtype so in-place works without extra casts
        a = a.to(device=fused.device, dtype=fused.dtype)
        t = t.to(device=fused.device, dtype=fused.dtype)
        fused.add_(a)
        fused.add_(t)
        # Transformer over TIME per asset (memory-safe): L = window, not assets*window
        seq = fused.reshape(B * self.n_assets, self.window, self.embed_dim)

        # ── Adaptive row chunking across the whole stack (robust against OOM)
        BN, L, E = seq.shape
        bytes_per = 2 if seq.dtype in (torch.float16, torch.bfloat16) else 4
        max_rows_cap = int(os.environ.get("LONG_MAX_ROWS", "8192"))  # default higher to reduce overhead

        def _adaptive_rows():
            # try to size chunks from *free* VRAM so we skip chunking when possible
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
            except Exception:
                return max_rows_cap
            # generous safety: account for FFN/attn workspaces
            denom = max(1, L * E * bytes_per * 8)
            rows = int((free_bytes * 0.85) // denom)
            return max(64, min(max_rows_cap, rows))

        rows = _adaptive_rows()
        if rows >= BN:
            # fast path: whole batch at once
            seq = self.blocks(seq)
        else:
            parts = []
            start = 0
            while start < BN:
                end = min(BN, start + rows)
                xi = seq[start:end]
                yi = self.blocks(xi)
                parts.append(yi)
                start = end
            seq = torch.cat(parts, dim=0)


        dtype_in = seq.dtype
        with torch.amp.autocast("cuda", enabled=False):
            seq = self.final_ln(seq.float()).to(dtype_in)

        y = seq.reshape(B, self.n_assets, self.window, self.embed_dim)

        # Attention pool over TIME per asset
        y = y.reshape(B * self.n_assets, self.window, self.embed_dim)
        y = self.time_pool(y)                         # [B*N,E]
        y = y.reshape(B, self.n_assets, self.embed_dim)

        # Cross-asset attention (short sequence length = N assets)
        y = self.cross(y)
        y = self.final_drop(y)                        # [B,N,E]
        fdtype = self.extras_proj.weight.dtype  # typically torch.float32
        y = y.to(fdtype)

        # ── Extras: window-level OHLC features (24 -> project to 5 per asset) ────────
        # Compute stats without tracking grads to save a LOT of memory
        with torch.no_grad():
            eps = 1e-6
            open_  = bars[..., 0]     # [B,N,W]
            high_  = bars[..., 1]
            low_   = bars[..., 2]
            close_ = bars[..., 3]

            # Current and previous bars
            o_t, h_t, l_t, c_t = open_[..., -1], high_[..., -1], low_[..., -1], close_[..., -1]
            o_p, h_p, l_p, c_p = open_[..., -2], high_[..., -2], low_[..., -2], close_[..., -2]

            # Sequences (safe for log even if inputs were ever normalized)
            close_next = close_[..., 1:].clamp_min(eps)
            close_prev = close_[..., :-1].clamp_min(eps)
            r_seq = torch.log(close_next / close_prev)                                  # [B,N,W-1]

            R_seq = (high_ - low_).abs()                                             # [B,N,W]
            R_t   = (h_t - l_t).abs()
            R_p   = (h_p - l_p).abs().clamp_min(eps)
            denp  = c_p.abs().clamp_min(eps)

            # Last-bar / bar-on-bar (6)
            logret1     = torch.log((c_t + eps) / (c_p + eps))
            gap_pct     = (o_t - c_p) / denp
            range_rel   = R_t / denp
            body_frac   = ((c_t - o_t) / (R_t + eps)).clamp(-1.0, 1.0)
            up_wick     = (h_t - torch.maximum(o_t, c_t)) / (R_t + eps)
            dn_wick     = (torch.minimum(o_t, c_t) - l_t) / (R_t + eps)
            wick_imb    = (up_wick - dn_wick).clamp(-1.0, 1.0)
            up_break    = F.relu(h_t - h_p) / R_p
            dn_break    = F.relu(l_p - l_t) / R_p
            break_score = torch.tanh(up_break - dn_break)

            # Window direction/vol shape (8)
            up_ratio    = (r_seq > 0).float().mean(dim=-1) * 2.0 - 1.0
            s           = torch.sign(r_seq)
            persistence = (s[..., 1:] * s[..., :-1]).mean(dim=-1)
            ret_std     = torch.tanh(r_seq.std(dim=-1))
            r_mean      = r_seq.mean(dim=-1, keepdim=True)
            r_std       = r_seq.std(dim=-1, keepdim=True).clamp_min(eps)
            z           = (r_seq - r_mean) / r_std
            ret_skew    = torch.tanh((z**3).mean(dim=-1))
            ret_kurt    = torch.tanh((z**4).mean(dim=-1) - 3.0)
            r_med       = r_seq.median(dim=-1, keepdim=True).values
            mad         = (r_seq - r_med).abs().median(dim=-1).values.clamp_min(eps)
            last_ret_z  = torch.tanh((r_seq[..., -1] - r_med.squeeze(-1)) / mad)
            R_med       = R_seq.median(dim=-1).values.clamp_min(eps)
            range_ratio_win = torch.tanh(R_t / R_med)
            W = close_.size(-1)
            half = W // 2
            if half < 1:
                range_trend = torch.zeros_like(range_ratio_win)
            else:
                range_trend = torch.tanh((R_seq[..., half:].mean(dim=-1) - R_seq[..., :half].mean(dim=-1)) / (R_med))

            # Geometry across window (6)
            body_seq        = ((close_ - open_) / (R_seq + eps)).clamp(-1.0, 1.0)
            body_frac_delta = (body_frac - body_seq.median(dim=-1).values).clamp(-1.0, 1.0)
            up_w_seq        = (high_ - torch.maximum(open_, close_)) / (R_seq + eps)
            dn_w_seq        = (torch.minimum(open_, close_) - low_) / (R_seq + eps)
            wi_seq          = (up_w_seq - dn_w_seq).clamp(-1.0, 1.0)
            wick_imb_delta  = (wick_imb - wi_seq.median(dim=-1).values).clamp(-1.0, 1.0)
            inside_rate     = ((high_[..., 1:] <= high_[..., :-1]) & (low_[..., 1:] >= low_[..., :-1])).float().mean(dim=-1)
            outside_rate    = ((high_[..., 1:] >  high_[..., :-1]) & (low_[..., 1:] <  low_[..., :-1])).float().mean(dim=-1)
            c_min = close_.amin(dim=-1); c_max = close_.amax(dim=-1)
            close_pos_win   = ((c_t - c_min) / ((c_max - c_min).clamp_min(eps)) * 2.0 - 1.0).clamp(-1.0, 1.0)

            if W < 4:
                ret_drift_shift = torch.zeros_like(close_pos_win)
            else:
                ret_drift_shift = torch.tanh(r_seq[..., half-1:].mean(dim=-1) - r_seq[..., :half-1].mean(dim=-1))

            # Recency vs extremes (4)
            idx_high = torch.argmax(high_, dim=-1).to(dtype=close_.dtype)
            idx_low  = torch.argmin(low_ , dim=-1).to(dtype=close_.dtype)
            den_idx  = float(max(W - 1, 1))
            bars_since_high = 2.0 * ((W - 1.0 - idx_high) / den_idx) - 1.0
            bars_since_low  = 2.0 * ((W - 1.0 - idx_low ) / den_idx) - 1.0
            close_pos_prev  = ((c_t - l_p) / (R_p + eps) * 2.0 - 1.0).clamp(-1.0, 1.0)
            open_pos_prev   = ((o_t - l_p) / (R_p + eps) * 2.0 - 1.0).clamp(-1.0, 1.0)

            extras_24 = torch.stack([
                logret1, gap_pct, range_rel, body_frac, wick_imb, break_score,
                up_ratio, persistence, ret_std, ret_skew, ret_kurt, last_ret_z, range_ratio_win, range_trend,
                body_frac_delta, wick_imb_delta, inside_rate, outside_rate, close_pos_win, ret_drift_shift,
                bars_since_high, bars_since_low, close_pos_prev, open_pos_prev
            ], dim=-1)   # [B,N,24]

        # Project to 5 per asset; grads only flow into this linear
        extras_24 = torch.nan_to_num(extras_24, nan=0.0, posinf=0.0, neginf=0.0)
        extras_24 = extras_24.clamp_(-50.0, 50.0).to(fdtype)
        extras_5 = self.extras_proj(extras_24)      # [B,N,5]
        extras   = extras_5.reshape(B, -1)          # [B, N*5]

        # Concatenate pooled transformer features with extras
        y_flat = y.reshape(B, -1)                       # [B, N*E]
        feats = torch.cat([y_flat, extras], dim=1)      # [B, N*(E + 5)]

        # ── NEW: sanitize & bound features to keep matmuls stable
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = feats.clamp_(-50.0, 50.0)

        out = self.out_proj(feats)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def features_dim(self) -> int:
        return self._features_dim
# ──────────────────────────────────────────────────────────────────────────────
# Hybrid distribution (identical to long)
# ──────────────────────────────────────────────────────────────────────────────
class HybridActionDistribution(Distribution):
    def __init__(self, n_assets: int, n_discrete: int = 8, n_cont: int = 2, squash_gaussian: bool = True):
        super().__init__()
        self.n_assets = n_assets
        self.n_discrete = n_discrete
        self.n_cont = n_cont
        self.cont_dim = n_assets * n_cont
        if squash_gaussian:
            self.cont_dist = SquashedDiagGaussianDistribution(self.cont_dim)
        else:
            from stable_baselines3.common.distributions import DiagGaussianDistribution
            self.cont_dist = DiagGaussianDistribution(self.cont_dim)
        self._disc_logits: Optional[torch.Tensor] = None

    def proba_distribution(self, discrete_logits: torch.Tensor, cont_mean: torch.Tensor, cont_log_std: torch.Tensor):
        # ── Clamp categorical logits first for numerical stability
        discrete_logits = torch.clamp(discrete_logits, -60.0, 60.0)

        if discrete_logits.dim() == 2:
            B = discrete_logits.shape[0]
            self._disc_logits = discrete_logits.view(B, self.n_assets, self.n_discrete)
        elif discrete_logits.dim() == 3:
            self._disc_logits = discrete_logits
        else:
            raise ValueError("discrete_logits must be [B, N*8] or [B, N, 8]")

        # ── Final safety before handing to SB3 Normal
        cont_mean    = torch.nan_to_num(cont_mean, nan=0.0, posinf=1e6, neginf=-1e6)
        cont_log_std = torch.clamp(cont_log_std, min=-5.0, max=2.0)
        self.cont_dist = self.cont_dist.proba_distribution(cont_mean, cont_log_std)
        return self
    
    def proba_distribution_net(self, *args, **kwargs):
        """
        SB3 requires distributions to expose this builder even if unused.
        Our policy builds the heads itself, so return a no-op.
        """
        return nn.Identity(), None


    def actions_from_params(self, discrete_logits: torch.Tensor, cont_mean: torch.Tensor, cont_log_std: torch.Tensor, deterministic: bool = False):
        self.proba_distribution(discrete_logits, cont_mean, cont_log_std)
        return self.get_actions(deterministic)

    def log_prob_from_params(self, discrete_logits: torch.Tensor, cont_mean: torch.Tensor, cont_log_std: torch.Tensor):
        self.proba_distribution(discrete_logits, cont_mean, cont_log_std)
        actions = self.get_actions(deterministic=False)
        return actions, self.log_prob(actions)

    def _cat(self) -> torch.distributions.Categorical:
        if self._disc_logits is None:
            raise RuntimeError("Call proba_distribution first")
        return torch.distributions.Categorical(logits=self._disc_logits)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        disc = actions[:, : self.n_assets * self.n_discrete].view(B, self.n_assets, self.n_discrete)
        cont = actions[:, self.n_assets * self.n_discrete :]
        disc = torch.nan_to_num(disc, nan=0.0).clamp(min=0.0)
        disc = disc / (disc.sum(dim=2, keepdim=True) + 1e-8)
        idx = disc.argmax(dim=2)
        cat_lp = self._cat().log_prob(idx).sum(dim=1)
        cont_lp = self.cont_dist.log_prob(cont)
        lp = cat_lp + cont_lp
        return torch.nan_to_num(lp, nan=-1e6, posinf=-1e6, neginf=-1e6)


    def entropy(self) -> torch.Tensor:
        disc_H = self._cat().entropy().sum(dim=1)

        cont_H = self.cont_dist.entropy()
        if isinstance(cont_H, torch.Tensor):
            # Unsquashed Gaussian path (DiagGaussian) → already a tensor
            return torch.nan_to_num(disc_H + cont_H, nan=0.0, posinf=0.0, neginf=0.0)


        # Squashed Gaussian path: SB3 returns None. Try to approximate using log_std.
        log_std = getattr(self.cont_dist, "log_std", None)
        if isinstance(log_std, torch.Tensor):
            # Differential entropy of the underlying Normal (pre-squash), per batch
            # h = 0.5 * n * (1 + ln(2π)) + sum(log_std)
            import math
            n = log_std.size(-1)
            base_H = 0.5 * n * (1.0 + math.log(2 * math.pi)) + log_std.sum(dim=1)
            return disc_H + base_H

        # Last-resort: ignore continuous entropy
        return disc_H


    def sample(self) -> torch.Tensor:
        cat = self._cat()
        idx = cat.sample()
        B = idx.size(0)
        disc_one_hot = F.one_hot(idx, num_classes=self.n_discrete).float().view(B, -1)
        cont = self.cont_dist.sample()
        return torch.cat([disc_one_hot, cont], dim=1)

    def mode(self) -> torch.Tensor:
        logits = self._cat().logits
        idx = logits.argmax(dim=-1)
        B = idx.size(0)
        disc_one_hot = F.one_hot(idx, num_classes=self.n_discrete).float().view(B, -1)
        cont_mode = self.cont_dist.mode()
        return torch.cat([disc_one_hot, cont_mode], dim=1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.mode() if deterministic else self.sample()


class LongTermOHLCPolicy(ActorCriticPolicy):
    """
    Advanced hourly policy:
      • Extractor: DS-TCN + Transformer + AttnPool + Cross-asset + SE + extras
      • Hybrid head: Cat(8) + Squashed Gaussian(2) per asset
      • State-dependent std + per-asset learned temperatures
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        window: int = 48,
        embed_dim: int = 128,
        tcn_hidden: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        base_disc_temperature: float = 1.0,
        state_dependent_std: bool = True,
        **kwargs: Any,
    ):
        act_dim = action_space.shape[0]
        assert act_dim % 10 == 0, f"Action dim must be 10*n_assets (got {act_dim})"
        self.n_assets = act_dim // 10
        self.n_disc = 8
        self.n_cont = 2
        self.base_disc_temperature = float(max(0.4, min(2.0, base_disc_temperature)))
        self.state_dependent_std = bool(state_dependent_std)
        self._last_log_std: Optional[torch.Tensor] = None

        kwargs.pop("net_arch", None)

        default_policy_kwargs: dict[str, Any] = dict(
            # SB3 ≥ 1.8 prefers dict(pi=..., vf=...) instead of [dict(...)]
            net_arch=dict(pi=[192, 128], vf=[192, 128]),
            features_extractor_class=LongTermFeatureExtractor,
            features_extractor_kwargs=dict(
                n_assets=self.n_assets,
                window=window,
                embed_dim=embed_dim,
                tcn_hidden=tcn_hidden,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=0.08,
            ),
            # IMPORTANT: do not squash unless using gSDE
            squash_output=False,
        )
        super().__init__(observation_space, action_space, lr_schedule, **{**default_policy_kwargs, **kwargs})

        self._build(lr_schedule)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.latent_ln = nn.LayerNorm(latent_dim_pi)
        self.pi_disc = nn.Linear(latent_dim_pi, self.n_assets * self.n_disc)
        self.pi_cont_mean = nn.Linear(latent_dim_pi, self.n_assets * self.n_cont)

        if self.state_dependent_std:
            self.pi_cont_log_std = nn.Linear(latent_dim_pi, self.n_assets * self.n_cont)
        else:
            self.log_std = nn.Parameter(-0.5 * torch.ones(self.n_assets * self.n_cont))

        self.temp_head = nn.Linear(self.features_extractor.features_dim, self.n_assets)
        self.regime_classifier = nn.Linear(self.features_extractor.features_dim, 4)

        for m in [self.pi_disc, self.pi_cont_mean, self.temp_head, self.regime_classifier]:
            orthogonal_init(m, gain=0.01)
        if self.state_dependent_std:
            orthogonal_init(self.pi_cont_log_std, gain=0.01)

        self._hybrid = HybridActionDistribution(self.n_assets, self.n_disc, self.n_cont, squash_gaussian=True)

    def _get_action_dist_from_latent(
        self,
        latent_pi: torch.Tensor,
        features: Optional[torch.Tensor] = None,  # ← make optional
    ) -> Distribution:
        z = self.latent_ln(latent_pi)
        disc_logits = self.pi_disc(z)
        cont_mean   = self.pi_cont_mean(z)

        if hasattr(self, "pi_cont_log_std"):
            cont_log_std = self.pi_cont_log_std(z).clamp(-6.0, 1.5)
        else:
            cont_log_std = self.log_std.expand_as(cont_mean).clamp(-6.0, 1.5)

        # If we have features (forward/evaluate), use per-asset temps.
        # If not (some SB3 internals), fall back to base temperature only.
        if features is not None:
            temps = 0.2 + F.softplus(self.temp_head(features))   # [B, N]
            temps = temps.unsqueeze(-1).expand(-1, -1, self.n_disc).reshape(disc_logits.size(0), -1)
            disc_logits = disc_logits / (self.base_disc_temperature * temps.clamp(min=0.2))
        else:
            disc_logits = disc_logits / self.base_disc_temperature

        # ── NEW: keep categorical logits numerically safe
        disc_logits = torch.clamp(disc_logits, -60.0, 60.0)

        # ── NEW: sanitize continuous head before creating the distribution
        cont_mean    = torch.nan_to_num(cont_mean, nan=0.0, posinf=1e6, neginf=-1e6)
        cont_log_std = torch.clamp(cont_log_std, min=-5.0, max=2.0)
        self._last_log_std = cont_log_std.detach()
        return self._hybrid.proba_distribution(disc_logits, cont_mean, cont_log_std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._get_action_dist_from_latent(latent_pi, features)
        value = self.value_net(latent_vf)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)

        # ── NEW: sanitize what PPO consumes
        value    = torch.nan_to_num(value,    nan=0.0, posinf=0.0, neginf=0.0)
        log_prob = torch.nan_to_num(log_prob, nan=-1e6, posinf=-1e6, neginf=-1e6)

        self._last_regime_logits = torch.nan_to_num(self.regime_classifier(features), nan=0.0, posinf=0.0, neginf=0.0)
        return actions, value, log_prob

    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        # SB3 calls this during PPO updates; we pass features so temp_head is used
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        dist = self._get_action_dist_from_latent(latent_pi, features=features)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self.value_net(latent_vf)

        # ── NEW: sanitize before loss is computed
        values   = torch.nan_to_num(values,   nan=0.0,  posinf=0.0,  neginf=0.0)
        log_prob = torch.nan_to_num(log_prob, nan=-1e6, posinf=-1e6, neginf=-1e6)
        entropy  = torch.nan_to_num(entropy,  nan=0.0,  posinf=0.0,  neginf=0.0)

        return values, log_prob, entropy

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        a, _, _ = self.forward(observation, deterministic=deterministic)
        disc = a[..., : self.n_assets * self.n_disc]
        cont = a[..., self.n_assets * self.n_disc :].clamp(-1.0, 1.0)
        out = torch.cat([disc, cont], dim=-1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    def get_log_std_for_logging(self) -> Optional[torch.Tensor]:
        """
        Prefer the last log_std we actually used to build the distribution.
        Fallback to the underlying hybrid's continuous distribution if needed.
        """
        if isinstance(self._last_log_std, torch.Tensor):
            return self._last_log_std
        try:
            cd = getattr(self._hybrid, "cont_dist", None)
            return getattr(cd, "log_std", None)
        except Exception:
            return None
    def get_regime_logits(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_regime_logits", None)
# Export list
__all__ = ["LongTermFeatureExtractor", "LongTermOHLCPolicy"]
