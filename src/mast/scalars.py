"""Per-object auxiliary scalars for the pretraining corpus (plan §2.1).

Builds one row per asteroid number with visible geometric albedo pV
(+ sigma + source) and absolute magnitude H:

  - NEOWISE diameters/albedos V2.0 (data/raw/neowise): headerless CSVs,
    columns positional per the PDS4 labels (pV = col 14, err = col 15);
    multiple epoch fits per object -> inverse-variance weighted mean.
  - Ali-Lagoa & Delbo 2018 AKARI re-fit (data/raw/akari_alilagoa2018):
    MPC-packed designations; error floors 20% (eta fitted) / 40% (not)
    already reflected in e_pV per the ReadMe.
  - AcuA V1 (data/raw/akari): one row per object.

Priority for the "best" value: NEOWISE > Ali-Lagoa 2018 > AcuA
(most objects, most homogeneous first). All sources are kept as
columns; the chosen one is flagged. H comes from the same source's fit
input (NEOWISE Absolute_mag / AKARI HMAG / AL18 H).

Only numbered asteroids are covered (all three catalogs are effectively
number-keyed); unnumbered corpus objects simply get no scalar tokens —
absent, never imputed.
"""

from __future__ import annotations

import glob
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NEOWISE_DIR = ROOT / "data/raw/neowise/neowise_diameters_albedos_V2_0/data"
ACUA = ROOT / "data/raw/akari/AcuA_V1.txt.gz"
AL18 = ROOT / "data/raw/akari_alilagoa2018/table1.dat.gz"
CACHE = ROOT / "data/processed/corpus/scalars.parquet"

_NEOWISE_FILES = [
    "neowise_mainbelt.csv", "neowise_neos.csv", "neowise_jupiter_trojans.csv",
    "neowise_hildas.csv", "neowise_centaurs.csv",
]


def _unpack_mpc(packed: str) -> int | None:
    """MPC-packed permanent designation -> asteroid number (else None)."""
    packed = packed.strip()
    if not packed:
        return None
    if packed.isdigit():
        return int(packed)
    head = packed[0]
    if head.isalpha() and packed[1:].isdigit() and len(packed) == 5:
        base = ord(head) - ord("A") - (6 if head >= "a" else 0)
        # A=10 ... Z=35, a=36 ...
        value = ord(head) - 55 if head.isupper() else ord(head) - 61
        return value * 10000 + int(packed[1:])
    return None


def load_neowise() -> pd.DataFrame:
    frames = []
    for name in _NEOWISE_FILES:
        df = pd.read_csv(
            NEOWISE_DIR / name, header=None,
            usecols=[0, 3, 13, 14],
            names=["number", "H", "pV", "e_pV"],
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["number", "pV"])
    df = df[(df.pV > 0) & (df.e_pV > 0)]
    df["number"] = df.number.astype(int)
    # inverse-variance weighted mean across epoch fits
    df["w"] = 1.0 / df.e_pV**2
    agg = df.groupby("number").apply(
        lambda g: pd.Series(
            {"pV": np.average(g.pV, weights=g.w),
             "e_pV": np.sqrt(1.0 / g.w.sum()),
             "H": g.H.mean(), "n_fits": len(g)}
        ),
        include_groups=False,
    )
    return agg.reset_index()


def load_al18() -> pd.DataFrame:
    with gzip.open(AL18, "rt") as f:
        df = pd.read_fwf(
            f, colspecs=[(0, 8), (9, 14), (37, 44), (45, 52)],
            names=["packed", "H", "pV", "e_pV"],
        )
    df["number"] = df.packed.astype(str).map(_unpack_mpc)
    df = df.dropna(subset=["number", "pV"])
    df = df[(df.pV > 0) & (df.e_pV > 0)]
    df["number"] = df.number.astype(int)
    df["w"] = 1.0 / df.e_pV**2
    agg = df.groupby("number").apply(
        lambda g: pd.Series(
            {"pV": np.average(g.pV, weights=g.w),
             "e_pV": np.sqrt(1.0 / g.w.sum()),
             "H": g.H.mean()}
        ),
        include_groups=False,
    )
    return agg.reset_index()


def load_acua() -> pd.DataFrame:
    with gzip.open(ACUA, "rt") as f:
        df = pd.read_fwf(
            f, colspecs=[(0, 6), (37, 42), (66, 71), (72, 77)],
            names=["number", "H", "pV", "e_pV"],
        )
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["number", "pV"])
    df = df[(df.pV > 0) & (df.e_pV > 0)]
    df["number"] = df.number.astype(int)
    return df[["number", "pV", "e_pV", "H"]]


def build(cache: Path | None = CACHE) -> pd.DataFrame:
    """Merged per-object scalar table with source priority."""
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    sources = [("neowise", load_neowise()), ("al18", load_al18()),
               ("acua", load_acua())]
    merged: pd.DataFrame | None = None
    for name, df in sources:
        df = df.rename(columns={c: f"{c}_{name}" for c in df.columns if c != "number"})
        merged = df if merged is None else merged.merge(df, on="number", how="outer")
    out = merged
    out["pV"] = np.nan
    out["e_pV"] = np.nan
    out["H"] = np.nan
    out["pV_source"] = ""
    for name, _ in sources:
        take = out.pV.isna() & out[f"pV_{name}"].notna()
        for col in ("pV", "e_pV", "H"):
            out.loc[take, col] = out.loc[take, f"{col}_{name}"]
        out.loc[take, "pV_source"] = name
    out = out.sort_values("number").reset_index(drop=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out
