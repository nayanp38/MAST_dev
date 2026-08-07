"""Masking invariants (P2 W2)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mast import masking


def _batch(B=64, L=20, n_scalar=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    valid = torch.rand(B, L, generator=g) > 0.2
    valid[:, 0] = True
    token_type = torch.zeros(B, L, dtype=torch.long)
    if n_scalar:
        token_type[:, -n_scalar] = 1   # albedo
        token_type[:, -n_scalar + 1] = 2  # H
    return valid, token_type


@pytest.mark.parametrize("granularity", ["token", "block", "mixed"])
@pytest.mark.parametrize("rho", [0.3, 0.6, 0.75])
def test_bounds_invariant(granularity, rho):
    valid, token_type = _batch()
    gen = torch.Generator().manual_seed(1)
    mask = masking.make_mask(granularity, valid, token_type, rho, gen)
    masked = (mask & valid).sum(1)
    visible = (valid & ~mask).sum(1)
    multi = valid.sum(1) > 1
    assert (masked[multi] >= 1).all()
    assert (visible[multi] >= 1).all()
    assert not (mask & ~valid).any()  # never mask padding


def test_token_ratio_approx():
    valid, token_type = _batch(B=512)
    gen = torch.Generator().manual_seed(2)
    mask = masking.make_mask("token", valid, token_type, 0.5, gen)
    frac = (mask & valid).float().sum() / valid.float().sum()
    assert 0.4 < frac < 0.6


def test_block_fallback_is_contiguous_for_scalar_free():
    valid, token_type = _batch(B=32, n_scalar=0)
    gen = torch.Generator().manual_seed(3)
    mask = masking.make_mask("block", valid, token_type, 0.5, gen)
    for i in range(32):
        sel = mask[i] & valid[i]
        if sel.sum() < 2:
            continue
        # masked spectral positions (in valid-token order) are contiguous
        order = torch.nonzero(valid[i]).squeeze(1)
        flags = mask[i][order]
        idx = torch.nonzero(flags).squeeze(1)
        assert (idx[-1] - idx[0] + 1) == len(idx)


def test_seeded_determinism():
    valid, token_type = _batch()
    m1 = masking.make_mask("mixed", valid, token_type, 0.45,
                           torch.Generator().manual_seed(7))
    m2 = masking.make_mask("mixed", valid, token_type, 0.45,
                           torch.Generator().manual_seed(7))
    assert torch.equal(m1, m2)


def test_pivot_token_never_masked():
    valid, token_type = _batch()
    token_type[:, 1] = 3  # pivot-offset token
    gen = torch.Generator().manual_seed(4)
    for granularity in ["token", "block", "mixed"]:
        mask = masking.make_mask(granularity, valid, token_type, 0.75, gen)
        assert not (mask & (token_type == 3)).any()
