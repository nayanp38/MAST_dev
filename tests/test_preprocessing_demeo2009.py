"""R1 as a permanent unit test: DeMeo 2009 preprocessing certification.

Runs against the original DeMeo 2009 input corpus
(data/raw/busdemeo2009/DeMeo2009data) when present, else the public
reconstruction (data/external/smass_demeo2009, fetched by
scripts/fetch_demeo2009_spectra.py).

Run: PYTHONPATH=src .venv/bin/pytest tests/test_preprocessing_demeo2009.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from baselines import demeo2009_r1
from mast import preprocessing as pp

pytestmark = pytest.mark.skipif(
    not demeo2009_r1.OFFICIAL_DIR.is_dir() and not demeo2009_r1.FALLBACK_DIR.is_dir(),
    reason="no R1 spectra corpus on disk",
)


@pytest.fixture(scope="module")
def r1_results():
    return demeo2009_r1.run()


def test_r1_pass(r1_results):
    """|r| >= 0.99 for PC1-PC3 on the screened evaluation set."""
    primary = r1_results["screened"]
    for i in (1, 2, 3):
        assert abs(primary["projection_r"][f"PC{i}"]) >= 0.99
        assert abs(primary["own_pca_r"][f"PC{i}"]) >= 0.99
    assert r1_results["pass"]


def test_r1_full_set_floors(r1_results):
    """Regression guard on the unscreened set (known outliers included)."""
    full = r1_results["full"]
    assert abs(full["projection_r"]["PC1"]) >= 0.985
    assert abs(full["projection_r"]["PC2"]) >= 0.99
    assert abs(full["projection_r"]["PC3"]) >= 0.99


def test_r1_slope(r1_results):
    """Slope feature tracks the published slopes."""
    assert abs(r1_results["screened"]["projection_r"]["slope"]) >= 0.99


def test_r1_outlier_budget(r1_results):
    """The screened-out tail must stay small (<= 5% of evaluated objects)."""
    assert len(r1_results["outliers"]) <= 0.05 * r1_results["n_evaluated"]


def test_barbara_known_input():
    """(234) Barbara: scores from its archived input spectrum must land
    near the published entry (0.6665, -0.3380, 0.0566). Loose per-object
    tolerance — archival re-reductions shift single objects by ~0.1-0.2;
    the ensemble correlation (test_r1_pass) is the certification
    criterion."""
    data_mean, eig = demeo2009_r1.load_published_basis()
    for d in (demeo2009_r1.OFFICIAL_DIR, demeo2009_r1.FALLBACK_DIR):
        candidates = sorted(d.glob("a000234*.txt")) if d.is_dir() else []
        if candidates:
            break
    assert candidates, "no archived spectrum for (234) Barbara on disk"
    wave, refl = pp.read_spectrum_file(candidates[0])
    gamma, s41 = pp.preprocess_demeo(wave, refl)
    scores = eig @ (s41[pp.PCA_CHANNELS] - data_mean)
    published = np.array([0.6665, -0.3380, 0.0566])
    assert np.all(np.abs(scores[:3] - published) < 0.2)


def test_grid_definition():
    assert len(pp.DEMEO_GRID) == 41
    assert pp.DEMEO_GRID[0] == 0.45 and pp.DEMEO_GRID[-1] == 2.45
    assert len(pp.PCA_CHANNELS) == 40


def test_rejects_partial_coverage():
    """No imputation: spectra missing more than the canonical 4.7% edge
    tolerance are rejected."""
    wave = np.linspace(0.6, 2.45, 100)  # missing 0.15 um at the blue end
    refl = np.ones_like(wave)
    with pytest.raises(ValueError):
        pp.preprocess_demeo(wave, refl)
