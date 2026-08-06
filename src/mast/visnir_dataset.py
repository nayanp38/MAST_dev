"""Shared VIS+NIR spectrum dataset for R2/R3 (DeMeo 2009 + MITHNEOS).

One spectrum per object, deduplicated per the recreation plan:
  - DeMeo 2009 objects: the original input spectrum from
    data/raw/busdemeo2009/DeMeo2009data/, label = harmonized bdm_class
    (tier-1).
  - Additional MITHNEOS objects (tier-2 labels): the latest classy-
    classified VIS+NIR spectrum whose individual class matches the
    object's aggregated label (falls back to the latest classifiable
    file).

Two feature representations are produced from the same raw spectra:
  - R2 (Penttilä 2021): 200 channels, 0.45–2.45 um, normalized at
    0.55 um, slope retained.
  - R3 (Klimczak 2021): the R1-certified 41-channel slope-removed
    representation + the slope gamma (top-5 PC scores are computed in
    the R3 baseline itself).
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mast import labels as L
from mast import preprocessing as pp

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "data/raw/busdemeo2009/DeMeo2009data"
CACHE = ROOT / "data/processed/visnir_dataset.parquet"

GRID200 = np.linspace(0.45, 2.45, 200)


def _demeo_file_index() -> dict[str, str]:
    """object_id -> official DeMeo 2009 spectrum file."""
    out = {}
    for f in sorted(glob.glob(str(OFFICIAL_DIR / "*.txt"))):
        base = os.path.basename(f)
        m = re.match(r"a(\d+)\.", base) or re.match(r"a(\d+)\.txt", base)
        if m:
            out[str(int(m.group(1)))] = f
            continue
        m = re.match(r"au(\d{4})([A-Z]+\d*)\.", base)
        if m:
            out[f"{m.group(1)} {m.group(2)}"] = f
    return out


def _mithneos_file(oid: str, cls: str, per_spectrum: pd.DataFrame) -> str | None:
    g = per_spectrum[(per_spectrum.object_id == oid) & (per_spectrum.bdm == cls)]
    if not len(g):
        g = per_spectrum[
            (per_spectrum.object_id == oid) & (per_spectrum.bdm != "indeterminate")
        ]
    if not len(g):
        return None
    base = g.sort_values("year").iloc[-1].file
    hits = glob.glob(str(Path.home() / "Library/Caches/classy/mithneos/*" / base))
    return hits[0] if hits else None


def build(cache: Path | None = CACHE) -> pd.DataFrame:
    """Assemble the dataset; cached as parquet."""
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)

    table = pd.read_parquet(ROOT / "data/processed/label_table_v1.parquet")
    pool = table[table.bdm_class.notna()].copy()
    per_spectrum = pd.read_csv(
        ROOT / "data/processed/mithneos_demeo_per_spectrum.csv",
        dtype={"object_id": str},
    )
    demeo_files = _demeo_file_index()

    rows = []
    for _, obj in pool.iterrows():
        oid = str(obj.object_id)
        tier1 = pd.notna(obj.bdm_demeo2009)
        f = demeo_files.get(oid) if tier1 else _mithneos_file(
            oid, obj.bdm_mithneos, per_spectrum
        )
        if f is None:
            continue
        wave, refl = pp.read_spectrum_file(f)
        wave, refl = pp._clean(wave, refl)
        try:
            # R2 representation: 200 channels, normalized, slope kept
            refl_n = pp.normalize_at(wave, refl)
            r200 = pp.resample_to_demeo_grid(wave, refl_n, grid=GRID200)
            # R3 representation: certified 41-channel slope-removed
            gamma, s41 = pp.preprocess_demeo(wave, refl)
        except ValueError:
            continue
        rows.append(
            {"object_id": oid, "bdm_class": obj.bdm_class,
             "bdm_complex": obj.bdm_complex, "disputed": bool(obj.disputed),
             "source": "demeo2009" if tier1 else "mithneos",
             "file": os.path.basename(f), "gamma": gamma,
             **{f"r200_{i}": v for i, v in enumerate(r200)},
             **{f"s41_{i}": v for i, v in enumerate(s41)}}
        )
    df = pd.DataFrame(rows)
    if cache is not None:
        df.to_parquet(cache, index=False)
    return df


def matrices(df: pd.DataFrame):
    """(X200, X41, gamma) numpy views of the feature columns."""
    x200 = df[[f"r200_{i}" for i in range(200)]].to_numpy(float)
    x41 = df[[f"s41_{i}" for i in range(41)]].to_numpy(float)
    return x200, x41, df.gamma.to_numpy(float)
