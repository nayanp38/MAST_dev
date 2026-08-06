"""Build label table v1 and the B1 split manifest (plan §2.2, §2.4).

Outputs:
  data/processed/label_table_v1.parquet (+ .csv)
  data/processed/mithneos_demeo_per_spectrum.csv   (tier-2 provenance)
  data/processed/splits/b1_folds_k10_seed42.{csv,json}

Usage: PYTHONPATH=src .venv/bin/python scripts/build_label_table.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mast import labels, splits

SEED = 42
K = 10


def main() -> None:
    per_spectrum_path = ROOT / "data/processed/mithneos_demeo_per_spectrum.csv"
    per_spectrum = labels.classify_mithneos_demeo(progress_path=per_spectrum_path)
    print(f"tier-2: {len(per_spectrum)} classified VIS+NIR spectra, "
          f"{per_spectrum.object_id.nunique()} objects")

    table = labels.build_label_table(per_spectrum)
    out = ROOT / "data/processed/label_table_v1.parquet"
    table.to_parquet(out, index=False)
    table.to_csv(out.with_suffix(".csv"), index=False)
    print(f"label table v1: {len(table)} objects "
          f"(BDM: {table.bdm_class.notna().sum()}, "
          f"Mahlke: {table.mahlke_class.notna().sum()}, "
          f"disputed: {int(table.disputed.sum())})")
    print("BDM class counts:")
    print(table.bdm_class.value_counts().to_string())

    # B1: 10-fold object-stratified CV on the BDM-labeled pool.
    # Disputed objects receive folds too (they are excluded from training
    # but kept in test sets downstream).
    pool = table[table.bdm_class.notna()].copy()
    folds = splits.make_object_folds(pool, k=K, seed=SEED, stratify_col="bdm_class")
    table_hash = hashlib.sha256(
        table.to_csv(index=False).encode()
    ).hexdigest()
    record = splits.write_manifest(
        folds,
        name=f"b1_folds_k{K}_seed{SEED}",
        meta={
            "seed": SEED, "k": K, "stratify": "bdm_class",
            "pool": "label_table_v1 objects with bdm_class",
            "label_table_sha256": table_hash,
            "inner_val_rule": "inner 20% of each training fold, carved at "
                              "runtime with the same seed, for tuning/early "
                              "stopping (plan §2.4 B1)",
            "note": "frozen at P1 exit; never regenerate",
        },
    )
    print(f"B1 manifest: {record['name']} n={record['n_objects']} "
          f"sha256={record['sha256'][:16]}…")
    per_fold = folds.fold.value_counts().sort_index()
    print("fold sizes:", per_fold.tolist())


if __name__ == "__main__":
    main()
