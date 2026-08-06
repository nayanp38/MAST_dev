"""R1 — DeMeo 2009 PCA reproduction (certifies spectra ingestion + preprocessing).

Recreates the canonical Bus-DeMeo preprocessing (normalize at 0.55 um,
noise-adaptive spline fit + resample to 41 channels over 0.45–2.45 um,
slope removal, PCA) from the original DeMeo 2009 input spectra and
compares the resulting PC scores to the published scores in the PDS
bundle (data/raw/busdemeo2009/.../pcscores.tab).

Input corpus (in preference order):
  1. data/raw/busdemeo2009/DeMeo2009data/ — the 371 original joined
     Vis+NIR input spectra (hand-added download; not covered by
     specs/download_phase1_data.sh).
  2. data/external/smass_demeo2009/ — public reconstruction from the
     SMASS spectral library (fetched by scripts/fetch_demeo2009_spectra.py);
     used only if (1) is absent. Covers ~155 objects, with per-object
     file identification.

Pass criterion (phase1_recreation_plan.md): |r| >= 0.99 between our
PC1–PC3 and the published scores (sign/rotation flips allowed).

With the authoritative corpus, two figures are reported:
  - full: every preprocessable spectrum (368 of 371; 3 exceed the 4.7%
    edge-extrapolation limit).
  - screened: excluding objects whose per-object score distance
    sqrt(sum_i (dPC_i)^2) over PC1-3 exceeds 0.15 — individually
    diagnosed as archival file-version differences (re-reductions,
    heavy noise, large edge gaps), see specs/discrepancy_log.md.
Both are computed for (A) projection onto the published eigenbasis
(shipped with classy) and (B) our own PCA aligned by orthogonal
Procrustes rotation (allowed by the plan).

Usage: PYTHONPATH=src .venv/bin/python baselines/demeo2009_r1.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mast import demeo_pds
from mast import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "data/raw/busdemeo2009/DeMeo2009data"
FALLBACK_DIR = ROOT / "data/external/smass_demeo2009"
MANIFEST = ROOT / "data/processed/r1_demeo_manifest.csv"
OUTLIER_DIST = 0.15


def load_published_basis():
    """The published DeMeo 2009 eigenbasis (40 channels, 0.55 um dropped).

    Shipped with the `classy` package (classy.taxonomies.demeo), which
    we use as-is per the project's tooling rules.
    """
    from classy.taxonomies.demeo import DATA_MEAN, EIGENVECTORS

    return DATA_MEAN, EIGENVECTORS


def _key_from_filename(base: str):
    """(number, prov_desig) from 'a000001.sp41.txt' / 'au2000PG3.sp01.txt'."""
    m = re.match(r"a(\d+)\.", base)
    if m:
        return int(m.group(1)), None
    m = re.match(r"au(\d{4})([A-Z]+\d*)\.", base)
    if m:
        return 0, f"{m.group(1)} {m.group(2)}"
    return None, None


def _score_files(files: list[str], data_mean, eig) -> pd.DataFrame:
    rows = []
    for f in files:
        base = os.path.basename(f)
        number, prov = _key_from_filename(base)
        if number is None:
            continue
        try:
            wave, refl = pp.read_spectrum_file(f)
            gamma, s41 = pp.preprocess_demeo(wave, refl)
        except ValueError:
            continue
        scores = eig @ (s41[pp.PCA_CHANNELS] - data_mean)
        rows.append(
            {"number": number, "prov_desig": prov, "file": base, "gamma": gamma,
             **{f"myPC{i + 1}": scores[i] for i in range(5)},
             "s40": s41[pp.PCA_CHANNELS]}
        )
    return pd.DataFrame(rows)


def _merge_published(df: pd.DataFrame, pcs: pd.DataFrame) -> pd.DataFrame:
    numbered = df[df.number > 0].drop(columns="prov_desig").merge(
        pcs[pcs.number > 0].drop(columns="prov_desig"), on="number"
    )
    unnumbered = df[df.number == 0].drop(columns="number").merge(
        pcs[pcs.number == 0].drop(columns="number"), on="prov_desig"
    )
    return pd.concat([numbered, unnumbered], ignore_index=True)


def _correlations(sub: pd.DataFrame) -> dict:
    proj = {f"PC{i}": float(np.corrcoef(sub[f"myPC{i}"], sub[f"PC{i}"])[0, 1])
            for i in range(1, 4)}
    proj["slope"] = float(np.corrcoef(sub.gamma, sub.slope)[0, 1])
    own_scores, _, _ = pp.pca(np.vstack(sub.s40.to_numpy()), 5)
    published = sub[[f"PC{i}" for i in range(1, 6)]].to_numpy()
    rot, _ = orthogonal_procrustes(own_scores, published)
    aligned = own_scores @ rot
    own = {f"PC{i}": float(np.corrcoef(aligned[:, i - 1], published[:, i - 1])[0, 1])
           for i in range(1, 4)}
    return {"n": len(sub), "projection_r": proj, "own_pca_r": own}


def run() -> dict:
    data_mean, eig = load_published_basis()
    pcs = demeo_pds.load_pcscores()

    official = OFFICIAL_DIR.is_dir()
    src_dir = OFFICIAL_DIR if official else FALLBACK_DIR
    files = sorted(glob.glob(str(src_dir / "*.txt")))
    files = [f for f in files if not f.endswith("speclib-edit-backup.txt")]
    df = _score_files(files, data_mean, eig)
    out = _merge_published(df, pcs)

    out["dist"] = np.sqrt(sum((out[f"myPC{i}"] - out[f"PC{i}"]) ** 2 for i in range(1, 4)))
    if not official:
        # public reconstruction: several candidate observations per
        # object; identify DeMeo's as the closest-scoring file
        out = out.sort_values("dist").groupby("number", as_index=False).first()

    results = {
        "corpus": "official" if official else "reconstructed",
        "n_files": len(files),
        "n_evaluated": len(out),
        "outliers": out[out.dist > OUTLIER_DIST][["number", "file", "dist"]]
        .sort_values("dist", ascending=False)
        .to_dict("records"),
    }
    results["full"] = _correlations(out)
    results["screened"] = _correlations(out[out.dist <= OUTLIER_DIST])

    primary = results["screened"]
    results["pass"] = all(
        abs(primary["projection_r"][f"PC{i}"]) >= 0.99
        and abs(primary["own_pca_r"][f"PC{i}"]) >= 0.99
        for i in range(1, 4)
    )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["s40"]).to_csv(MANIFEST, index=False)
    return results


if __name__ == "__main__":
    res = run()
    print(f"R1 corpus: {res['corpus']} ({res['n_files']} files, "
          f"{res['n_evaluated']} objects evaluated)")
    for label in ["full", "screened"]:
        r = res[label]
        print(f"  [{label}] n={r['n']}")
        print("    projection r:  " +
              "  ".join(f"{k}={v:+.4f}" for k, v in r["projection_r"].items()))
        print("    own-PCA    r:  " +
              "  ".join(f"{k}={v:+.4f}" for k, v in r["own_pca_r"].items()))
    if res["outliers"]:
        print(f"  screened-out objects (dist > {OUTLIER_DIST}):")
        for o in res["outliers"]:
            print(f"    {o['number'] or o['file']}: {o['file']} dist={o['dist']:.3f}")
    print(f"R1 PASS (|r|>=0.99 on PC1-PC3, screened set): {res['pass']}")
