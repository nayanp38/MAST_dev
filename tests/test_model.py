"""MAST model unit tests (P2 W2)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mast.model import MAST, MASTConfig, reconstruction_loss


def _batch(B=8, L=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "values": torch.rand(B, L, 2, generator=g),
        "lam": torch.rand(B, L, generator=g) * 800 + 374,
        "dlam": torch.rand(B, L, generator=g) * 50 + 8,
        "token_type": torch.zeros(B, L, dtype=torch.long),
        "instrument": torch.zeros(B, dtype=torch.long),
        "flags": torch.zeros(B, L, dtype=torch.long),
        "valid": torch.ones(B, L, dtype=torch.bool),
    }


def test_param_count_near_2p7m():
    n = sum(p.numel() for p in MAST(MASTConfig()).parameters())
    assert 2.4e6 < n < 3.1e6


def test_forward_shapes():
    model = MAST(MASTConfig())
    batch = _batch()
    out = model(batch)
    assert out["mu"].shape == (8, 16)
    assert out["log_var"].shape == (8, 16)
    assert out["pooled"].shape == (8, 192)


def test_padding_invariance():
    """Invalid (absent) tokens must not influence outputs — the
    no-imputation guarantee at the architecture level."""
    model = MAST(MASTConfig(drop_path=0.0)).eval()
    batch = _batch()
    batch["valid"][:, 10:] = False
    with torch.no_grad():
        ref = model(batch)["pooled"]
        # garbage in the padded region must change nothing
        batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch2["values"][:, 10:] = 99.0
        batch2["lam"][:, 10:] = 2400.0
        out = model(batch2)["pooled"]
    assert torch.allclose(ref, out, atol=1e-5)


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    model = MAST(MASTConfig(drop_path=0.0))
    batch = _batch(B=4)
    mask = torch.zeros(4, 16, dtype=torch.bool)
    mask[:, ::3] = True
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(150):
        out = model(batch, mask)
        losses = reconstruction_loss(out, batch, mask, model.cfg)
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        mse = float(losses["mse"])
        first = mse if first is None else first
        last = mse
    assert last < first * 0.15, (first, last)


def test_mask_token_changes_output():
    model = MAST(MASTConfig(drop_path=0.0)).eval()
    batch = _batch()
    mask = torch.zeros(8, 16, dtype=torch.bool)
    mask[:, :4] = True
    with torch.no_grad():
        a = model(batch, mask=None)["mu"]
        b = model(batch, mask=mask)["mu"]
    assert not torch.allclose(a, b)
