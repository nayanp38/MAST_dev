"""Pretraining-corpus builders: SDSS, SkyMapper, MOVIS → per-source
observation-record parquet (plan §2.1/§2.3).

One output row = one observation record = the set of bands measured in
one near-simultaneous visit of one object:

  - SDSS: one `sso.dat` row (all five ugriz measured together); kept if
    the object is a known SSO (`bknown=1`, needed for object-level
    splits) and the pivot band is valid.
  - SkyMapper: `skm-obs.dat` per-filter rows grouped by (object, night);
    within a night, repeated same-band mags are inverse-variance
    averaged.
  - MOVIS: `movis-m.dat` per-night magnitude sets (Y/J/H/Ks aperMag3).

Magnitudes become relative reflectances against the record's pivot band
(nearest 0.55 um present in the record; §2.3):

    R_b = 10^(-0.4 [ (m_b - m_pivot) - (m_b,sun - m_pivot,sun) ])

with sigma_R = 0.4 ln10 R sqrt(sigma_b^2 + sigma_pivot^2) (pivot error
propagated in quadrature — documented simplification; band-pivot
covariance ignored). Solar magnitudes come from mast.filters (SVO
Fsun-derived; see that module's provenance note). The pivot band's own
reflectance is exactly 1 with sigma from its magnitude error alone.

Schema (per source, data/processed/corpus/<source>_records.parquet):
  object_id (str), number (Int64, -1 if unnumbered), epoch_jd (float),
  phase_deg (float, NaN unknown), pivot_band (str),
  R_<band>, e_<band> (float, NaN = band absent), flag_<band> (uint8,
  source-specific quality bits, 0 = clean), n_bands (int)
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from mast.filters import registry

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/corpus"
SDSS_DAT = ROOT / "data/raw/sdss_sergeyev2021/sso.dat.gz"
SKM_OBS = ROOT / "data/raw/skymapper_sergeyev2022/skm-obs.dat.gz"
MOVIS_M = ROOT / "data/raw/movis_photometry/movis-m.dat.gz"

LN10_04 = 0.4 * np.log(10.0)


def _object_id(number, name) -> str:
    """number if present else normalized designation string."""
    if pd.notna(number) and str(number).strip() not in ("", "-"):
        return str(int(number))
    name = str(name).strip()
    return name if name and name != "-" else ""


def _to_reflectance(mags: dict, errs: dict, instrument: str):
    """mags/errs: band -> value (NaN = absent). Returns (pivot, R, eR)."""
    reg = registry()
    present = [b for b, m in mags.items() if np.isfinite(m) and np.isfinite(errs[b])]
    if len(present) < 2:
        return None
    pivot = min(present, key=lambda b: abs(reg.lambda_eff(instrument, b) - 550.0))
    m_p, e_p = mags[pivot], errs[pivot]
    R, eR = {}, {}
    for b in present:
        color = (mags[b] - m_p) - reg.solar_color(instrument, b, pivot)
        R[b] = 10.0 ** (-0.4 * color)
        if b == pivot:
            eR[b] = LN10_04 * R[b] * errs[b]
        else:
            eR[b] = LN10_04 * R[b] * float(np.hypot(errs[b], e_p))
    return pivot, R, eR


def _records_frame(rows: list[dict], bands: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for b in bands:
        for prefix in ("R_", "e_"):
            col = f"{prefix}{b}"
            if col not in df:
                df[col] = np.nan
        fcol = f"flag_{b}"
        if fcol not in df:
            df[fcol] = 0
        df[fcol] = df[fcol].fillna(0).astype("uint8")
    df["number"] = df.number.fillna(-1).astype("int64")
    return df


# ---------------------------------------------------------------------------
# SDSS

_SDSS_BANDS = ["u", "g", "r", "i", "z"]
# 0-based colspecs from the CDS ReadMe (1-based bytes - 1)
_SDSS_COLSPECS = {
    "JD": (73, 87),
    "umag": (149, 156), "gmag": (157, 164), "rmag": (165, 172),
    "imag": (173, 180), "zmag": (181, 188),
    "e_umag": (189, 196), "e_gmag": (197, 204), "e_rmag": (205, 211),
    "e_imag": (212, 219), "e_zmag": (220, 227),
    "Number": (338, 344), "Name": (345, 362),
    "alpha": (465, 470),
    "bphotu": (480, 481), "bphotg": (482, 483), "bphotr": (484, 485),
    "bphoti": (486, 487), "bphotz": (488, 489),
    "bknown": (504, 505),
}


def build_sdss(limit: int | None = None) -> pd.DataFrame:
    names = list(_SDSS_COLSPECS)
    colspecs = list(_SDSS_COLSPECS.values())
    rows = []
    n_raw = n_known = 0
    with gzip.open(SDSS_DAT, "rt") as f:
        reader = pd.read_fwf(f, colspecs=colspecs, names=names,
                             chunksize=200_000, nrows=limit)
        for chunk in reader:
            n_raw += len(chunk)
            chunk = chunk[pd.to_numeric(chunk.bknown, errors="coerce") == 1]
            n_known += len(chunk)
            for col in names:
                if col not in ("Name",):
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            for rec in chunk.itertuples(index=False):
                oid = _object_id(rec.Number, rec.Name)
                if not oid:
                    continue
                mags = {b: getattr(rec, f"{b}mag") for b in _SDSS_BANDS}
                errs = {b: getattr(rec, f"e_{b}mag") for b in _SDSS_BANDS}
                conv = _to_reflectance(mags, errs, "sdss")
                if conv is None:
                    continue
                pivot, R, eR = conv
                rows.append(
                    {"object_id": oid, "number": rec.Number,
                     "epoch_jd": rec.JD, "phase_deg": rec.alpha,
                     "pivot_band": pivot, "n_bands": len(R),
                     **{f"R_{b}": v for b, v in R.items()},
                     **{f"e_{b}": v for b, v in eR.items()},
                     **{f"flag_{b}": int(getattr(rec, f"bphot{b}") or 0)
                        for b in _SDSS_BANDS}}
                )
    df = _records_frame(rows, _SDSS_BANDS)
    df.attrs["n_raw"] = n_raw
    df.attrs["n_known"] = n_known
    return df


# ---------------------------------------------------------------------------
# SkyMapper

_SKM_BANDS = ["u", "v", "g", "r", "i", "z"]
_SKM_COLSPECS = {
    "JD": (26, 40), "Filter": (46, 47),
    "psfmag": (79, 86), "e_psfmag": (91, 97),
    "Number": (122, 128), "Name": (129, 145),
}


def build_skymapper(limit: int | None = None) -> pd.DataFrame:
    names = list(_SKM_COLSPECS)
    with gzip.open(SKM_OBS, "rt") as f:
        obs = pd.read_fwf(f, colspecs=list(_SKM_COLSPECS.values()), names=names,
                          nrows=limit)
    n_raw = len(obs)
    obs["psfmag"] = pd.to_numeric(obs.psfmag, errors="coerce")
    obs["e_psfmag"] = pd.to_numeric(obs.e_psfmag, errors="coerce")
    obs["JD"] = pd.to_numeric(obs.JD, errors="coerce")
    obs = obs.dropna(subset=["psfmag", "e_psfmag", "JD"])
    obs = obs[obs.e_psfmag > 0]
    obs["object_id"] = [
        _object_id(n, nm) for n, nm in zip(obs.Number, obs.Name)
    ]
    obs = obs[obs.object_id != ""]
    obs["night"] = np.floor(obs.JD - 0.5).astype(int)
    obs["Filter"] = obs.Filter.astype(str).str.strip()
    obs = obs[obs.Filter.isin(_SKM_BANDS)]

    rows = []
    for (oid, night), g in obs.groupby(["object_id", "night"], sort=False):
        mags, errs = {}, {}
        for b, gb in g.groupby("Filter"):
            w = 1.0 / gb.e_psfmag**2
            mags[b] = float(np.average(gb.psfmag, weights=w))
            errs[b] = float(np.sqrt(1.0 / w.sum()))
        for b in _SKM_BANDS:
            mags.setdefault(b, np.nan)
            errs.setdefault(b, np.nan)
        conv = _to_reflectance(mags, errs, "skymapper")
        if conv is None:
            continue
        pivot, R, eR = conv
        number = pd.to_numeric(g.Number.iloc[0], errors="coerce")
        rows.append(
            {"object_id": oid, "number": number,
             "epoch_jd": float(g.JD.mean()), "phase_deg": np.nan,
             "pivot_band": pivot, "n_bands": len(R),
             **{f"R_{b}": v for b, v in R.items()},
             **{f"e_{b}": v for b, v in eR.items()}}
        )
    df = _records_frame(rows, _SKM_BANDS)
    df.attrs["n_raw"] = n_raw
    return df


# ---------------------------------------------------------------------------
# MOVIS

_MOVIS_BANDS = ["Y", "J", "H", "Ks"]
_MOVIS_COLSPECS = {
    "Number": (0, 16), "Name": (17, 33),
    "Phase": (54, 59),
    "MeanMJD": (69, 86),
    "Ymag": (87, 94), "e_Ymag": (95, 100),
    "Jmag": (113, 120), "e_Jmag": (121, 128),
    "Hmag": (141, 148), "e_Hmag": (149, 156),
    "Ksmag": (170, 177), "e_Ksmag": (178, 185),
}


def build_movis() -> pd.DataFrame:
    with gzip.open(MOVIS_M, "rt") as f:
        m = pd.read_fwf(f, colspecs=list(_MOVIS_COLSPECS.values()),
                        names=list(_MOVIS_COLSPECS))
    n_raw = len(m)
    for col in m.columns:
        if col not in ("Number", "Name"):
            m[col] = pd.to_numeric(m[col], errors="coerce")
    rows = []
    for rec in m.itertuples(index=False):
        num = pd.to_numeric(rec.Number, errors="coerce")
        oid = _object_id(num, rec.Name)
        if not oid:
            continue
        mags = {b: getattr(rec, f"{'Ks' if b == 'Ks' else b}mag") for b in _MOVIS_BANDS}
        errs = {b: getattr(rec, f"e_{'Ks' if b == 'Ks' else b}mag") for b in _MOVIS_BANDS}
        conv = _to_reflectance(mags, errs, "movis")
        if conv is None:
            continue
        pivot, R, eR = conv
        rows.append(
            {"object_id": oid, "number": num,
             "epoch_jd": (rec.MeanMJD + 2400000.5) if np.isfinite(rec.MeanMJD) else np.nan,
             "phase_deg": rec.Phase, "pivot_band": pivot, "n_bands": len(R),
             **{f"R_{b}": v for b, v in R.items()},
             **{f"e_{b}": v for b, v in eR.items()}}
        )
    df = _records_frame(rows, _MOVIS_BANDS)
    df.attrs["n_raw"] = n_raw
    return df


# ---------------------------------------------------------------------------

def build_all(out_dir: Path = OUT_DIR, limit: int | None = None) -> dict:
    """Build every per-source record table; returns summary counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, builder in [("sdss", build_sdss), ("skymapper", build_skymapper),
                          ("movis", build_movis)]:
        path = out_dir / f"{name}_records.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = builder(limit=limit) if name != "movis" else builder()
            df.to_parquet(path, index=False)
        summary[name] = {"records": len(df),
                         "objects": df.object_id.nunique(),
                         "numbered": int((df.number >= 0).sum())}
    return summary
