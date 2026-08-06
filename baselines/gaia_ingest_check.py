"""R4 — Gaia DR3 ingestion check against Delbo et al. 2026 (certifies
Gaia parsing + the transfer test set). Deliberately NOT a full recreation
of their KDE classifier.

Checks (phase1_recreation_plan.md):
  1. Parse all 20 bulk chunks -> 60,518 objects. Pass: exact.
  2. Reference-set cut: S/N >= 50 + literature spectral labels.
     Pass: ~2,653 objects recovered (+/-5%). Our literature-label pool
     is the union of label_table_v1 (Mahlke 2022 compilation +
     Bus-DeMeo tiers) and the PDS ast_taxonomy compilation (Tholen,
     Bus, S3OS2, Bus-DeMeo columns; fetched by
     scripts/fetch_ast_taxonomy.py) — the closest public analog of
     Delbo's MP3C aggregation.
  3. Usability cut S/N > 20. Pass: ~36,566 (+/-2%).
     S/N = mean(R/sigma) over the 12 interior bands, calibrated in
     src/mast/gaia.py against their supplement (corr 1.000).
  4. Cross-match our labeled overlap with their dr3class1 (S/N > 20):
     headline agreement on the easy classes S and V (they report >92%
     and 99%); our overlap's per-class counts within ~5% of theirs.

Usage: PYTHONPATH=src .venv/bin/python baselines/gaia_ingest_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mast import gaia
from mast.labels import canonical_bdm

ROOT = Path(__file__).resolve().parents[1]
DELBO_CSV = ROOT / "data/raw/gaia_labels_delbo/data sheet 1.csv"
LABEL_TABLE = ROOT / "data/processed/label_table_v1.parquet"
AST_TAXONOMY = ROOT / "data/external/ast_taxonomy/data/taxonomy10.tab"


def load_ast_taxonomy() -> pd.DataFrame:
    """PDS ast_taxonomy compilation: per-object literature VIS classes."""
    colspecs = [(0, 7), (37, 43), (72, 75), (80, 83), (88, 91), (92, 95), (96, 99)]
    names = ["number", "tholen", "smass_xu", "bus", "s3os2_th", "s3os2_bb", "bdm"]
    df = pd.read_fwf(AST_TAXONOMY, colspecs=colspecs, names=names)
    df["number"] = pd.to_numeric(df.number, errors="coerce")
    df = df.dropna(subset=["number"])
    df["number"] = df.number.astype(int)
    for c in names[1:]:
        df[c] = df[c].fillna("-").astype(str).str.strip().replace("-", "")
    return df


def literature_letter(row) -> str:
    """One-letter literature class for the Delbo comparison.

    Priority: curated Bus-DeMeo > Bus > S3OS2 > Tholen > Mahlke —
    subclasses map to their leading letter (Sq -> S, Xk -> X, Ch -> C),
    matching the granularity of Delbo's dr3class labels.
    """
    cls = canonical_bdm(row.bdm_class) if pd.notna(row.bdm_class) else ""
    for col in ["bdm_lit", "bus", "s3os2_bb", "s3os2_th", "tholen", "smass_xu"]:
        if not cls and getattr(row, col, ""):
            cls = str(getattr(row, col))
    if not cls and pd.notna(getattr(row, "mahlke_classsf", None)):
        cls = str(row.mahlke_classsf)
    return cls[0].upper() if cls else ""


def run() -> dict:
    results = {}
    wide = gaia.load_wide(cache=ROOT / "data/processed/gaia_wide.parquet")

    # Check 1 — object count
    results["check1"] = {"n_objects": int(len(wide)), "target": 60518,
                         "pass": len(wide) == 60518}

    # Check 3 — usability cut (computed before 2 for reuse)
    n_use = int((wide.snr > 20).sum())
    results["check3"] = {"n_snr_gt20": n_use, "target": 36566,
                         "pass": abs(n_use - 36566) <= 0.02 * 36566}

    # Check 2 — reference set: S/N >= 50 + literature label
    table = pd.read_parquet(LABEL_TABLE)
    table["number"] = pd.to_numeric(table.object_id, errors="coerce")
    table = table.dropna(subset=["number"]).assign(number=lambda d: d.number.astype(int))
    ast_tax = load_ast_taxonomy().rename(columns={"bdm": "bdm_lit"})
    merged = table.merge(ast_tax, on="number", how="outer")
    for c in ["tholen", "smass_xu", "bus", "s3os2_th", "s3os2_bb", "bdm_lit"]:
        merged[c] = merged[c].fillna("")
    merged["letter"] = merged.apply(literature_letter, axis=1)
    labeled_num = merged[merged.letter != ""]

    g = wide.dropna(subset=["number_mp"]).copy()
    g["number"] = g.number_mp.astype(int)
    ref = g[g.snr >= 50].merge(labeled_num[["number", "letter"]], on="number")
    results["check2"] = {"n_reference": int(len(ref)), "target": 2653,
                         "pass": abs(len(ref) - 2653) <= 0.05 * 2653}

    # Check 4 — agreement with Delbo classifications (S/N > 20)
    delbo = pd.read_csv(DELBO_CSV, comment="#")
    delbo = delbo[delbo.snr > 20][["number", "dr3class1"]].dropna()
    overlap = g[g.snr > 20].merge(labeled_num[["number", "letter"]], on="number")
    overlap = overlap.merge(delbo, on="number")
    check4 = {"n_overlap": int(len(overlap))}
    for cls in ["S", "V"]:
        ours = overlap[overlap.letter == cls]
        theirs = overlap[overlap.dr3class1 == cls]
        agree = float((ours.dr3class1 == cls).mean()) if len(ours) else np.nan
        count_dev = abs(len(ours) - len(theirs)) / max(len(theirs), 1)
        # Pass bar (plan): per-class counts within ~5%. The agreement
        # rate (they report S > 92%, V 99% vs *their* reference labels)
        # is reported as context, not gated.
        check4[cls] = {
            "n_ours": int(len(ours)), "n_theirs": int(len(theirs)),
            "count_deviation": float(count_dev), "agreement": agree,
            "pass": bool(count_dev <= 0.05),
        }
    check4["pass"] = check4["S"]["pass"] and check4["V"]["pass"]
    results["check4"] = check4

    results["pass"] = all(results[c]["pass"] for c in
                          ["check1", "check2", "check3", "check4"])
    return results


if __name__ == "__main__":
    res = run()
    c1, c2, c3, c4 = (res[f"check{i}"] for i in (1, 2, 3, 4))
    print(f"R4 check 1 — objects parsed: {c1['n_objects']} "
          f"(target {c1['target']}) -> {'PASS' if c1['pass'] else 'FAIL'}")
    print(f"R4 check 2 — reference set (S/N>=50 + literature label): "
          f"{c2['n_reference']} (target ~{c2['target']} ±5%) -> "
          f"{'PASS' if c2['pass'] else 'FAIL'}")
    print(f"R4 check 3 — usability cut (S/N>20): {c3['n_snr_gt20']} "
          f"(target ~{c3['target']} ±2%) -> {'PASS' if c3['pass'] else 'FAIL'}")
    print(f"R4 check 4 — overlap n={c4['n_overlap']}:")
    for cls in ["S", "V"]:
        r = c4[cls]
        print(f"    {cls}: ours {r['n_ours']} vs theirs {r['n_theirs']} "
              f"(dev {r['count_deviation']:.1%}), agreement {r['agreement']:.1%} "
              f"-> {'PASS' if r['pass'] else 'FAIL'}")
    print(f"R4 PASS: {res['pass']}")
