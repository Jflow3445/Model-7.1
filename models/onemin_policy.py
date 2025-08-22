# models/onemin_policy.py
from __future__ import annotations
import math
import logging
from typing import Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution, SquashedDiagGaussianDistribution

logger = logging.getLogger("OneMinPolicy")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)

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
        x = self.dw(x); x = self.pw(x); x = self.bn(x)
        return self.drop(F.gelu(x))

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
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + self.drop(a))
        f = self.ff(x)
        return self.ln2(x + self.drop(f))

class CrossAssetAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.08):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        orthogonal_init(self.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x)
        return self.ln(x + y)

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
# OneMin-term feature extractor (hourly)
# ──────────────────────────────────────────────────────────────────────────────
class OneMinFeatureExtractor(BaseFeaturesExtractor):
    """
    Advanced hourly OHLC extractor:
      • Multi-scale DS-TCN + Token Transformer + AttnPool + Cross-asset + SE
      • Extras identical to long extractor for head compatibility
    Obs per-asset: [open, high, low, close] * window + [elapsed]
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        n_assets: int,
        window: int = 48,        # e.g., 2 days of hourly bars
        embed_dim: int = 48,
        tcn_hidden: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
        **_: Any,
    ):
        obs_dim = int(observation_space.shape[0])
        assert obs_dim % n_assets == 0, f"Obs dim {obs_dim} not divisible by n_assets {n_assets}"
        per_asset = obs_dim // n_assets
        assert per_asset == 4 * window + 1, f"per_asset={per_asset} != 4*window+1={4*window+1}"

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
        self.time_pool = AttnPool1D(embed_dim)
        self.cross = CrossAssetAttention(embed_dim, n_heads, dropout)
        self.final_ln = nn.LayerNorm(embed_dim)
        self.final_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self._features_dim, self._features_dim)

        self.register_buffer("asset_idx", torch.arange(n_assets))
        self.register_buffer("time_idx", torch.arange(window))

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding, nn.MultiheadAttention)):
                orthogonal_init(m)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.size(0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        obs = obs.reshape(B, self.n_assets, self.per_asset)
        bars = obs[:, :, : 4 * self.window].reshape(B, self.n_assets, self.window, 4)
        tok = self.token(self.in_ln(bars))

        x_cnn = bars.permute(0, 1, 3, 2).contiguous()  # [B,N,4,W]
        x_cnn = x_cnn.reshape(B * self.n_assets, 4, self.window)
        x_cnn = self.tcn3(self.tcn2(self.tcn1(x_cnn))) # [B*N,E,W]
        x_cnn = self.se(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 1).reshape(B, self.n_assets, self.window, self.embed_dim)

        fused = tok + x_cnn
        a = self.asset_pos(self.asset_idx).view(1, self.n_assets, 1, self.embed_dim)
        t = self.time_pos(self.time_idx).view(1, 1, self.window, self.embed_dim)
        fused = fused + a + t

        seq = fused.reshape(B, self.n_assets * self.window, self.embed_dim)
        y = self.blocks(seq)
        y = self.final_ln(y)
        y = y.reshape(B, self.n_assets, self.window, self.embed_dim)

        # pool over TIME per asset: (B, N, W, E) -> (B*N, W, E) -> (B*N, E) -> (B, N, E)
        y = y.reshape(B * self.n_assets, self.window, self.embed_dim)
        y = self.time_pool(y)
        y = y.reshape(B, self.n_assets, self.embed_dim)

        y = self.cross(y)
        y = self.final_drop(y)

        open_ = bars[..., 0]; high_ = bars[..., 1]; low_ = bars[..., 2]; close_ = bars[..., 3]
        momentum   = (close_ - open_).mean(dim=2)
        volatility = (close_ - open_).std(dim=2, unbiased=False)
        spread     = (high_ - low_).mean(dim=2)

        base = close_[:, 0, :] - open_[:, 0, :]
        base_c = base - base.mean(dim=1, keepdim=True)
        base_std = base_c.std(dim=1, unbiased=False) + 1e-6
        corr_list = []
        for i in range(self.n_assets):
            if i == 0:
                corr_list.append(torch.ones(B, device=obs.device))
            else:
                r = close_[:, i, :] - open_[:, i, :]
                r_c = r - r.mean(dim=1, keepdim=True)
                r_std = r_c.std(dim=1, unbiased=False) + 1e-6
                corr_list.append(torch.tanh((base_c * r_c).mean(dim=1) / (base_std * r_std)))
        rolling_corr = torch.stack(corr_list, dim=1)

        mean_close = close_.mean(dim=2)
        mean_open  = open_.mean(dim=2)
        regime_hint = torch.tanh((mean_close - mean_open) / (mean_open.abs() + 1e-6))

        y_flat = y.reshape(B, -1)
        extras = torch.cat([momentum, volatility, spread, rolling_corr, regime_hint], dim=1)  # [B,N*5]
        feats = torch.cat([y_flat, extras], dim=1)
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
        if discrete_logits.dim() == 2:
            B = discrete_logits.shape[0]
            self._disc_logits = discrete_logits.view(B, self.n_assets, self.n_discrete)
        elif discrete_logits.dim() == 3:
            self._disc_logits = discrete_logits
        else:
            raise ValueError("discrete_logits must be [B, N*8] or [B, N, 8]")
        self.cont_dist = self.cont_dist.proba_distribution(cont_mean, cont_log_std)
        return self
    def proba_distribution_net(self, *args, **kwargs):
        """
        SB3 API compatibility stub:
        ActorCriticPolicy expects distributions to expose `proba_distribution_net`
        when the framework builds default action heads. We build the heads inside
        the policy itself, so this net is unused. Returning (Identity, None)
        satisfies the abstract interface without side-effects.
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
        return cat_lp + cont_lp

    def entropy(self) -> torch.Tensor:
        return self._cat().entropy().sum(dim=1) + self.cont_dist.entropy()

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


class OneMinOHLCPolicy(ActorCriticPolicy):
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
        embed_dim: int = 48,
        tcn_hidden: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
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
        kwargs.pop("net_arch", None)

        default_policy_kwargs: dict[str, Any] = dict(
            # SB3 ≥ 1.8 prefers dict(pi=..., vf=...) instead of [dict(...)]
            net_arch=dict(pi=[288, 192], vf=[288, 192]),
            features_extractor_class=OneMinFeatureExtractor,
            features_extractor_kwargs=dict(
                n_assets=self.n_assets,
                window=window,
                embed_dim=embed_dim,
                tcn_hidden=tcn_hidden,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=0.10,
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

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, features: torch.Tensor) -> Distribution:
        z = self.latent_ln(latent_pi)
        disc_logits = self.pi_disc(z)
        cont_mean   = self.pi_cont_mean(z)

        temps = 0.2 + F.softplus(self.temp_head(features))  # [B,N] >= 0.2
        temps = temps.unsqueeze(-1).expand(-1, -1, self.n_disc).reshape(disc_logits.size(0), -1)
        disc_logits = disc_logits / (self.base_disc_temperature * temps.clamp(min=0.2))

        if hasattr(self, "pi_cont_log_std"):
            cont_log_std = self.pi_cont_log_std(z).clamp(-6.0, 1.5)
        else:
            cont_log_std = self.log_std.expand_as(cont_mean).clamp(-6.0, 1.5)
        return self._hybrid.proba_distribution(disc_logits, cont_mean, cont_log_std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._get_action_dist_from_latent(latent_pi, features)
        value = self.value_net(latent_vf)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        self._last_regime_logits = torch.nan_to_num(self.regime_classifier(features), nan=0.0, posinf=0.0, neginf=0.0)
        return actions, value, log_prob

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        a, _, _ = self.forward(observation, deterministic=deterministic)
        disc = a[..., : self.n_assets * self.n_disc]
        cont = a[..., self.n_assets * self.n_disc :].clamp(-1.0, 1.0)
        out = torch.cat([disc, cont], dim=-1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def get_regime_logits(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_regime_logits", None)
# Export list
__all__ = ["OneMinFeatureExtractor", "OneMinOHLCPolicy"]
