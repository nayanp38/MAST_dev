"""Gaia DR3 SSO reflectance-spectrum ingestion (R4-certified).

Parses the 20 bulk chunks in data/raw/gaia_dr3_sso into one wide table:
one row per object, 16 reflectance bands (374–1034 nm), errors, flags,
and a per-object S/N figure.

S/N definition (calibrated against the Delbo et al. 2026 supplement in
R4, corr(log)=1.0000, median ratio 1.000): mean of reflectance/error
over the 12 interior bands — the first two (374, 418 nm) and last two
(946, 990/1034 nm) bands are excluded, matching Delbo's usage.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data/raw/gaia_dr3_sso"

# The 16 Gaia DR3 SSO wavelengths (nm), fixed for every object.
GAIA_WAVELENGTHS = np.arange(374.0, 1035.0, 44.0)


def load_long(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatenate all chunks in long form (16 rows per object)."""
    files = sorted(glob.glob(str(raw_dir / "SsoReflectanceSpectrum_*.csv.gz")))
    if len(files) != 20:
        raise FileNotFoundError(f"expected 20 Gaia chunks in {raw_dir}, found {len(files)}")
    frames = [pd.read_csv(f, comment="#") for f in files]
    return pd.concat(frames, ignore_index=True)


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per object with 16-band arrays as columns.

    Object identity is `denomination` (unique per SSO; `number_mp` is
    empty for unnumbered objects).
    """
    long = long.sort_values(["denomination", "wavelength"])
    n_bands = len(GAIA_WAVELENGTHS)

    grouped = long.groupby("denomination", sort=True)
    meta = grouped.agg(
        source_id=("source_id", "first"),
        number_mp=("number_mp", "first"),
        nb_samples=("nb_samples", "first"),
        num_of_spectra=("num_of_spectra", "first"),
        n_rows=("wavelength", "size"),
    )
    if not (meta["n_rows"] == n_bands).all():
        bad = meta[meta["n_rows"] != n_bands]
        raise ValueError(f"{len(bad)} objects without exactly {n_bands} bands, e.g.\n{bad.head()}")

    refl = long.pivot(index="denomination", columns="wavelength", values="reflectance_spectrum")
    err = long.pivot(index="denomination", columns="wavelength", values="reflectance_spectrum_err")
    flag = long.pivot(index="denomination", columns="wavelength", values="reflectance_spectrum_flag")

    wide = meta.drop(columns="n_rows")
    for i, wl in enumerate(GAIA_WAVELENGTHS):
        wide[f"refl_{int(wl)}"] = refl[wl]
        wide[f"err_{int(wl)}"] = err[wl]
        wide[f"flag_{int(wl)}"] = flag[wl].astype("Int8")
    wide["snr"] = compute_snr(wide)
    return wide.reset_index()


def compute_snr(wide: pd.DataFrame) -> np.ndarray:
    """Mean R/sigma over the 12 interior bands (see module docstring)."""
    r = wide[[f"refl_{int(w)}" for w in GAIA_WAVELENGTHS]].to_numpy(float)
    e = wide[[f"err_{int(w)}" for w in GAIA_WAVELENGTHS]].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(e > 0, r / e, np.nan)
    return np.nanmean(ratio[:, 2:14], axis=1)


def load_wide(raw_dir: Path = RAW_DIR, cache: Path | None = None) -> pd.DataFrame:
    """Load (or build and cache) the wide per-object table."""
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    wide = to_wide(load_long(raw_dir))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(cache, index=False)
    return wide
