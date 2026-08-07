"""Band registry for the pretraining corpus (plan §2.3/§3.1).

Every band used by the tokenizer gets (lambda_eff, delta_lambda, solar
magnitude). Sources:

- SDSS / SkyMapper / VISTA: SVO Filter Profile Service metadata cached
  at data/external/filters_svo/filter_metadata.csv by
  scripts/fetch_svo_filters.py (lambda_eff, FWHM, ZeroPoint, Fsun).
  Solar magnitudes are DERIVED from SVO's Fsun (solar flux through the
  filter, erg/cm2/s/A) via f_nu = Fsun * lambda_pivot^2 / c and
  m_sun = -2.5 log10(f_nu / ZeroPoint) — one consistent source for all
  photometric systems (AB for SDSS/SkyMapper, Vega for VISTA). Pinned
  against literature solar colors in tests/test_filters.py (e.g.
  (g-r)_sun = 0.46 +/- 0.03, Holmberg et al. 2006).

- Gaia DR3 SSO reflectance bands: 16 pseudo-bands at 374 + 44k nm.
  These are already reflectance (no solar color needed). Effective
  widths Delta_lambda = lambda / R(lambda) with the BP/RP resolving
  power R(lambda) read approximately from Montegriffo et al. 2023
  (A&A 674, A3, their resolution curves); recorded here as the
  documented provenance choice per plan §2.3 — the A7 ablation replaces
  this with full transmission-curve encoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SVO_METADATA = ROOT / "data/external/filters_svo/filter_metadata.csv"

C_ANGSTROM_PER_S = 2.99792458e18
AB_ZEROPOINT_JY = 3631.0

# Magnitude system of each survey's CATALOG values (not SVO's default
# calibration — SVO ZeroPoint is Vega-based for these filters, which is
# only correct for VISTA/MOVIS; SDSS and SkyMapper catalogs are AB).
CATALOG_MAG_SYS = {"sdss": "AB", "skymapper": "AB", "movis": "Vega"}

# survey band name -> SVO filter id
SVO_IDS = {
    ("sdss", "u"): "SLOAN/SDSS.u",
    ("sdss", "g"): "SLOAN/SDSS.g",
    ("sdss", "r"): "SLOAN/SDSS.r",
    ("sdss", "i"): "SLOAN/SDSS.i",
    ("sdss", "z"): "SLOAN/SDSS.z",
    ("skymapper", "u"): "SkyMapper/SkyMapper.u",
    ("skymapper", "v"): "SkyMapper/SkyMapper.v",
    ("skymapper", "g"): "SkyMapper/SkyMapper.g",
    ("skymapper", "r"): "SkyMapper/SkyMapper.r",
    ("skymapper", "i"): "SkyMapper/SkyMapper.i",
    ("skymapper", "z"): "SkyMapper/SkyMapper.z",
    ("movis", "Y"): "Paranal/VISTA.Y",
    ("movis", "J"): "Paranal/VISTA.J",
    ("movis", "H"): "Paranal/VISTA.H",
    ("movis", "Ks"): "Paranal/VISTA.Ks",
}

# Gaia BP/RP approximate resolving power per SSO band (Montegriffo+ 2023).
GAIA_WAVELENGTHS_NM = np.arange(374.0, 1035.0, 44.0)
_GAIA_RESOLVING_POWER = {
    374.0: 25.0, 418.0: 30.0, 462.0: 40.0, 506.0: 50.0, 550.0: 60.0,
    594.0: 70.0, 638.0: 75.0, 682.0: 72.0, 726.0: 75.0, 770.0: 80.0,
    814.0: 85.0, 858.0: 90.0, 902.0: 95.0, 946.0: 98.0, 990.0: 100.0,
    1034.0: 100.0,
}


class BandRegistry:
    """(instrument, band) -> lambda_eff_nm, delta_lambda_nm, m_sun."""

    def __init__(self, metadata_path: Path = SVO_METADATA):
        meta = pd.read_csv(metadata_path).set_index("filter_id")
        self._table: dict[tuple[str, str], dict] = {}
        for (instrument, band), fid in SVO_IDS.items():
            rec = meta.loc[fid]
            f_nu_jy = (
                rec.fsun * (rec.lambda_pivot_nm * 10.0) ** 2 / C_ANGSTROM_PER_S * 1e23
            )
            mag_sys = CATALOG_MAG_SYS[instrument]
            zeropoint = AB_ZEROPOINT_JY if mag_sys == "AB" else float(rec.zeropoint_jy)
            self._table[(instrument, band)] = {
                "lambda_eff_nm": float(rec.lambda_eff_nm),
                "delta_lambda_nm": float(rec.fwhm_nm),
                "m_sun": float(-2.5 * np.log10(f_nu_jy / zeropoint)),
                "mag_sys": mag_sys,
            }
        for wl in GAIA_WAVELENGTHS_NM:
            self._table[("gaia", f"{int(wl)}")] = {
                "lambda_eff_nm": float(wl),
                "delta_lambda_nm": float(wl / _GAIA_RESOLVING_POWER[wl]),
                "m_sun": np.nan,  # Gaia SSO data are already reflectance
                "mag_sys": "reflectance",
            }

    def lambda_eff(self, instrument: str, band: str) -> float:
        return self._table[(instrument, band)]["lambda_eff_nm"]

    def delta_lambda(self, instrument: str, band: str) -> float:
        return self._table[(instrument, band)]["delta_lambda_nm"]

    def m_sun(self, instrument: str, band: str) -> float:
        return self._table[(instrument, band)]["m_sun"]

    def solar_color(self, instrument: str, band_a: str, band_b: str) -> float:
        """(m_a - m_b) of the Sun in the instrument's magnitude system."""
        return self.m_sun(instrument, band_a) - self.m_sun(instrument, band_b)

    def bands(self, instrument: str) -> list[str]:
        return [b for (ins, b) in self._table if ins == instrument]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"instrument": ins, "band": b, **rec}
             for (ins, b), rec in self._table.items()]
        )


_default: BandRegistry | None = None


def registry() -> BandRegistry:
    global _default
    if _default is None:
        _default = BandRegistry()
    return _default
