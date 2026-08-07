"""Token-cache integrity tests (P2 W1/W2)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mast.tokenize import GAIA_KEEP, TOKENS_DIR, TYPE_PIVOT

pytestmark = pytest.mark.skipif(
    not (TOKENS_DIR / "meta.json").exists(),
    reason="token cache not built",
)


@pytest.fixture(scope="module")
def meta():
    return json.loads((TOKENS_DIR / "meta.json").read_text())


def test_counts_match_spec_scale(meta):
    n = meta["sources"]
    assert n["gaia"]["n"] == 60518
    assert abs(n["sdss"]["n"] - 1_036_000) < 40_000
    assert n["skymapper"]["n"] > 200_000
    assert n["movis"]["n"] > 40_000
    assert meta["n_records_total"] > 1_300_000


def test_gaia_interior_bands(meta):
    assert meta["gaia_bands_kept"] == GAIA_KEEP
    assert len(GAIA_KEEP) == 12
    assert GAIA_KEEP[0] == 462 and GAIA_KEEP[-1] == 946


def test_no_nan_in_valid_tokens():
    for source in ["gaia", "sdss", "movis"]:
        npz = np.load(TOKENS_DIR / f"{source}.npz")
        valid = npz["valid"]
        values = npz["values"]
        assert np.isfinite(values[valid]).all(), source
        # absent tokens are zeroed, never imputed with plausible values
        assert (values[~valid] == 0).all(), source


def test_photometry_has_pivot_token():
    npz = np.load(TOKENS_DIR / "sdss.npz")
    has_pivot = ((npz["token_type"] == TYPE_PIVOT) & npz["valid"]).any(axis=1)
    assert has_pivot.all()


def test_gaia_reflectance_scale():
    npz = np.load(TOKENS_DIR / "gaia.npz")
    vals = npz["values"][..., 0][npz["valid"] & (npz["token_type"] == 0)]
    assert 0.5 < np.median(vals) < 1.5


def test_record_index_alignment(meta):
    index = pd.read_parquet(TOKENS_DIR / "record_index.parquet")
    for source, info in meta["sources"].items():
        assert (index.source == source).sum() == info["n"]
