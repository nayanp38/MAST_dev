"""Pin the band registry against literature values (P2 W1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mast.filters import SVO_METADATA, registry

pytestmark = pytest.mark.skipif(
    not SVO_METADATA.exists(),
    reason="SVO metadata not fetched (run scripts/fetch_svo_filters.py)",
)


def test_sdss_solar_colors_match_literature():
    """Derived SDSS AB solar colors vs Willmer 2018 / Holmberg 2006."""
    r = registry()
    assert abs(r.solar_color("sdss", "g", "r") - 0.46) < 0.05
    assert abs(r.solar_color("sdss", "r", "i") - 0.11) < 0.03
    assert abs(r.solar_color("sdss", "i", "z") - 0.03) < 0.03
    assert abs(r.solar_color("sdss", "u", "g") - 1.35) < 0.15


def test_lambda_eff_sane():
    r = registry()
    assert 600 < r.lambda_eff("sdss", "r") < 630
    assert 1000 < r.lambda_eff("movis", "Y") < 1040
    assert 2100 < r.lambda_eff("movis", "Ks") < 2200
    # Gaia pseudo-bands on the 44 nm grid
    assert r.lambda_eff("gaia", "550") == 550.0
    assert len(r.bands("gaia")) == 16


def test_delta_lambda_positive_and_reasonable():
    r = registry()
    frame = r.as_frame()
    assert (frame.delta_lambda_nm > 5).all()
    assert (frame.delta_lambda_nm < 400).all()
    # Gaia widths grow roughly with wavelength (resolution curve)
    assert r.delta_lambda("gaia", "374") > r.delta_lambda("gaia", "1034") * 0.5


def test_solar_color_antisymmetry():
    r = registry()
    assert np.isclose(
        r.solar_color("sdss", "g", "r"), -r.solar_color("sdss", "r", "g")
    )


def test_movis_is_vega_sdss_is_ab():
    r = registry()
    frame = r.as_frame().set_index(["instrument", "band"])
    assert frame.loc[("movis", "J"), "mag_sys"] == "Vega"
    assert frame.loc[("sdss", "r"), "mag_sys"] == "AB"
