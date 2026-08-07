"""MAST encoder: tokenizer embedding + pre-norm transformer +
heteroscedastic reconstruction head (plan §3.1).

Input batch (all tensors [B, L] or [B, L, k]):
  values      float [B, L, 2]  — (reflectance-like value, sigma)
  lam         float [B, L]     — lambda_eff in nm (0 where invalid)
  dlam        float [B, L]     — delta_lambda in nm
  token_type  long  [B, L]     — 0 spectral, 1 albedo, 2 H, 3 pivot-offset,
                                 4 slope (reserved), 5.. [MISSING-m]
  instrument  long  [B]        — source catalog id
  flags       long  [B, L]     — per-token quality flag (0 clean)
  valid       bool  [B, L]     — real token (False = padding)

Scalar tokens carry their value in `values` with lam/dlam = 0 (their
Fourier features are zeroed; identity comes from token_type).

Config flags implement the pre-registered §3.3 contingencies:
beta_nll, mse_warmup_epochs, per-modality loss normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TOKEN_TYPES = 12
N_INSTRUMENTS = 8   # gaia, sdss, skymapper, movis, spex, ground-vis, spare
N_FLAG_VALUES = 8


@dataclass
class MASTConfig:
    d_model: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    drop_path: float = 0.1
    layerscale_init: float = 1e-4
    fourier_feats: int = 24          # frequencies per coordinate (sin+cos)
    # loss
    beta_nll: float = 0.5            # 0 = plain NLL; 0.5 = beta-NLL
    mse_warmup_epochs: int = 0
    per_modality_norm: bool = False
    # aux
    contrastive_weight: float = 0.1
    contrastive_temp: float = 0.1

    def n_params_estimate(self) -> int:
        d = self.d_model
        per_layer = 4 * d * d + 2 * d * int(self.mlp_ratio * d)
        return self.depth * per_layer


class FourierFeatures(nn.Module):
    """Deterministic log-spaced frequencies over (log lam, log dlam)."""

    def __init__(self, n_feats: int):
        super().__init__()
        freqs = torch.logspace(0.0, 2.0, n_feats) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, lam: torch.Tensor, dlam: torch.Tensor) -> torch.Tensor:
        # normalize to ~[0, 1]: lambda in [300, 2500] nm; dlam in [5, 400]
        x1 = (torch.log(lam.clamp(min=1.0)) - math.log(300.0)) / math.log(2500.0 / 300.0)
        x2 = (torch.log(dlam.clamp(min=1.0)) - math.log(5.0)) / math.log(400.0 / 5.0)
        out = []
        for x in (x1, x2):
            ang = x.unsqueeze(-1) * self.freqs
            out.extend([torch.sin(ang), torch.cos(ang)])
        feats = torch.cat(out, dim=-1)
        # zero positional features where lam is 0 (scalar tokens)
        return feats * (lam > 0).unsqueeze(-1)


class DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, cfg: MASTConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.heads = cfg.heads
        self.norm2 = nn.LayerNorm(d)
        hidden = int(cfg.mlp_ratio * d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.ls1 = nn.Parameter(cfg.layerscale_init * torch.ones(d))
        self.ls2 = nn.Parameter(cfg.layerscale_init * torch.ones(d))
        self.drop_path = DropPath(cfg.drop_path)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_mask = key_padding_mask[:, None, None, :]  # True = keep
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L, d)
        x = x + self.drop_path(self.ls1 * self.proj(out))
        x = x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))
        return x


class MAST(nn.Module):
    def __init__(self, cfg: MASTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.fourier = FourierFeatures(cfg.fourier_feats)
        fdim = 4 * cfg.fourier_feats
        self.value_mlp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.pos_proj = nn.Linear(fdim, d)
        self.type_emb = nn.Embedding(N_TOKEN_TYPES, d)
        self.instrument_emb = nn.Embedding(N_INSTRUMENTS, d)
        self.flag_emb = nn.Embedding(N_FLAG_VALUES, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def embed(self, batch: dict, mask: torch.Tensor | None) -> torch.Tensor:
        """mask: bool [B, L], True = value hidden (replaced by mask token)."""
        values = batch["values"]
        sigma_feat = torch.log1p(values[..., 1].clamp(min=0.0))
        val = self.value_mlp(torch.stack([values[..., 0], sigma_feat], dim=-1))
        if mask is not None:
            val = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(val), val)
        x = (val
             + self.pos_proj(self.fourier(batch["lam"], batch["dlam"]))
             + self.type_emb(batch["token_type"])
             + self.flag_emb(batch["flags"].clamp(max=N_FLAG_VALUES - 1))
             + self.instrument_emb(batch["instrument"]).unsqueeze(1))
        return x

    def forward(self, batch: dict, mask: torch.Tensor | None = None):
        x = self.embed(batch, mask)
        valid = batch["valid"]
        for blk in self.blocks:
            x = blk(x, valid)
        x = self.norm(x)
        mu, log_var = self.head(x).unbind(-1)
        pooled = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        return {"mu": mu, "log_var": log_var, "pooled": pooled, "tokens": x}

    @torch.no_grad()
    def encode(self, batch: dict) -> torch.Tensor:
        """Frozen-encoder embedding for probes (no masking)."""
        return self.forward(batch, mask=None)["pooled"]


def reconstruction_loss(
    out: dict, batch: dict, mask: torch.Tensor, cfg: MASTConfig,
    mse_only: bool = False,
) -> dict:
    """Heteroscedastic Gaussian NLL on masked tokens.

    Total variance = predicted variance + measurement variance.
    beta-NLL (Seitzer et al. 2022): gradient weighted by sigma^(2*beta),
    implemented via the detached-variance weighting factor.
    """
    target = batch["values"][..., 0]
    meas_var = batch["values"][..., 1].clamp(min=0.0) ** 2
    sel = mask & batch["valid"]
    if sel.sum() == 0:
        zero = out["mu"].sum() * 0.0
        return {"loss": zero, "nll": zero.detach(), "mse": zero.detach()}
    mu, log_var = out["mu"][sel], out["log_var"][sel].clamp(-10.0, 10.0)
    err2 = (mu - target[sel]) ** 2
    mse = err2.mean()
    var = log_var.exp() + meas_var[sel]
    nll = 0.5 * (torch.log(var) + err2 / var)
    if cfg.beta_nll > 0:
        nll = nll * var.detach() ** cfg.beta_nll
    if cfg.per_modality_norm:
        # normalize contribution per token_type so trivially-predictable
        # modalities cannot dominate (contingency §3.3-3)
        types = batch["token_type"][sel]
        loss = 0.0
        n_groups = 0
        for t in types.unique():
            m = types == t
            loss = loss + nll[m].mean()
            n_groups += 1
        nll_mean = loss / max(n_groups, 1)
    else:
        nll_mean = nll.mean()
    loss = mse if mse_only else nll_mean
    return {"loss": loss, "nll": nll_mean.detach(), "mse": mse.detach()}


def contrastive_loss(pooled_a, pooled_b, temp: float) -> torch.Tensor:
    """In-batch InfoNCE between paired views (rows aligned)."""
    a = F.normalize(pooled_a, dim=-1)
    b = F.normalize(pooled_b, dim=-1)
    logits = a @ b.T / temp
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
