"""Masking strategies for masked-reconstruction pretraining (plan §3.2).

Three granularities, applied on-device per step with a seeded generator:

  - "token": Bernoulli(rho) over valid tokens.
  - "block": whole modality blocks (spectral / albedo / H) masked until
    the masked fraction reaches rho. Records with no scalar tokens
    degenerate — fallback is a contiguous lambda-span covering ~rho of
    the spectral tokens (documented design choice; the spec leaves the
    scalar-free case open).
  - "mixed": per-record 50/50 choice between the two.

Every record is guaranteed >=1 masked and >=1 visible valid token.
"""

from __future__ import annotations

import torch

SPECTRAL_TYPE = 0
SCALAR_TYPES = (1, 2)  # albedo, H (pivot-offset token type 3 is never masked)


def _ensure_bounds(mask: torch.Tensor, valid: torch.Tensor, gen: torch.Generator):
    """Guarantee >=1 masked and >=1 visible valid token per record."""
    n_valid = valid.sum(1)
    n_masked = (mask & valid).sum(1)
    # none masked -> mask one random valid token
    need_mask = (n_masked == 0) & (n_valid > 1)
    if need_mask.any():
        scores = torch.rand(valid.shape, device=valid.device, generator=gen)
        scores = scores.masked_fill(~valid, -1.0)
        idx = scores.argmax(1)
        rows = torch.nonzero(need_mask).squeeze(1)
        mask[rows, idx[rows]] = True
    # all masked -> unmask one random valid token
    n_masked = (mask & valid).sum(1)
    all_masked = (n_masked == n_valid) & (n_valid > 1)
    if all_masked.any():
        scores = torch.rand(valid.shape, device=valid.device, generator=gen)
        scores = scores.masked_fill(~(mask & valid), -1.0)
        idx = scores.argmax(1)
        rows = torch.nonzero(all_masked).squeeze(1)
        mask[rows, idx[rows]] = False
    return mask


def token_mask(valid, token_type, rho: float, gen: torch.Generator):
    r = torch.rand(valid.shape, device=valid.device, generator=gen)
    mask = (r < rho) & valid & (token_type != 3)
    return _ensure_bounds(mask, valid, gen)


def _span_mask(valid, token_type, rho: float, gen: torch.Generator):
    """Contiguous span over spectral tokens covering ~rho of them."""
    B, L = valid.shape
    spectral = valid & (token_type == SPECTRAL_TYPE)
    n_spec = spectral.sum(1).clamp(min=1)
    span = (n_spec.float() * rho).round().long().clamp(min=1)
    start_frac = torch.rand(B, device=valid.device, generator=gen)
    mask = torch.zeros_like(valid)
    # positions among spectral tokens
    order = torch.cumsum(spectral.long(), dim=1) - 1  # index within spectral
    start = (start_frac * (n_spec - span + 1).clamp(min=1).float()).long()
    sel = (order >= start.unsqueeze(1)) & (order < (start + span).unsqueeze(1))
    mask = sel & spectral
    return mask


def block_mask(valid, token_type, rho: float, gen: torch.Generator):
    B, L = valid.shape
    has_scalar = ((token_type == SCALAR_TYPES[0]) | (token_type == SCALAR_TYPES[1])) & valid
    has_scalar = has_scalar.any(1)
    mask = torch.zeros_like(valid)

    # records with scalars: randomly choose blocks until fraction >= rho
    if has_scalar.any():
        blocks = [token_type == SPECTRAL_TYPE] + [token_type == t for t in SCALAR_TYPES]
        order = torch.rand(B, len(blocks), device=valid.device, generator=gen).argsort(1)
        frac = torch.zeros(B, device=valid.device)
        n_valid = valid.sum(1).float().clamp(min=1)
        for k in range(len(blocks)):
            todo = has_scalar & (frac < rho)
            if not todo.any():
                break
            for b_idx in range(len(blocks)):
                chosen = todo & (order[:, k] == b_idx)
                if chosen.any():
                    blk = blocks[b_idx] & valid
                    mask = mask | (blk & chosen.unsqueeze(1))
            frac = (mask & valid).sum(1) / n_valid
    # scalar-free records: contiguous span fallback
    span = _span_mask(valid, token_type, rho, gen)
    mask = torch.where(has_scalar.unsqueeze(1), mask, span)
    return _ensure_bounds(mask & (token_type != 3), valid, gen)


def make_mask(granularity: str, valid, token_type, rho: float, gen: torch.Generator):
    if granularity == "token":
        return token_mask(valid, token_type, rho, gen)
    if granularity == "block":
        return block_mask(valid, token_type, rho, gen)
    if granularity == "mixed":
        pick = torch.rand(valid.shape[0], device=valid.device, generator=gen) < 0.5
        t = token_mask(valid.clone(), token_type, rho, gen)
        b = block_mask(valid.clone(), token_type, rho, gen)
        return torch.where(pick.unsqueeze(1), t, b)
    raise ValueError(f"unknown granularity {granularity!r}")
