# Model/Run Log — R3: Klimczak 2021 metric-frame recreation

**Date:** 2026-08-06 · **Status:** PASS (type 77.9 ± 2.4 vs 76.8 ± 3;
complex 87.4 ± 0.5 vs 90.0 ± 3) · **Code:** `baselines/klimczak2021.py`

## Goal in the scheme of the research

Certifies the honest metric stack every MAST result will be quoted in:
balanced accuracy, object-level stratified folds, repeated CV,
mean ± std reporting (plan §2.4/§5.2).

## Exact data sources used

| Data | Location | Role |
|---|---|---|
| VIS+NIR dataset | `data/processed/visnir_dataset.parquet` | 41-channel slope-removed spectra (R1-certified) + slope γ |
| Labels | `label_table_v1` via `MERGE12` map in the baseline | 12 merged types; complexes C={B,C,Ch}, S={S,Sq,Sr}, X, other |
| Folds | `mast.splits.make_object_folds`, k=5, seeds 100–109 (representative manifest `r3_klimczak_folds_k5_seed100` frozen) | 5-fold × 10 repeats |

Features: top-5 PC scores (PCA fit inside each training fold) + slope.
Models: XGBoost (defaults) and sklearn MLP (100 hidden units), scaler per
training fold.

## Tweaks attempted and results (balanced accuracy)

| # | Config | Type | Complex | Notes |
|---|---|---|---|---|
| 1 | Unmerged 12-type list (X n=16) | 77.2 | 76.5–77.3 | complex level 13 pts off — wrong reading of their scheme |
| 2 | Derived-from-type complex predictions | — | 78.2 | not the explanation |
| 3 | Merged types (X n=49 etc.), all objects | 78.1 ± 1.3 | 84.7 ± 1.1 | type ✓, complex 2.3 below band |
| 4 | Q-in-S complex variant | — | 85.6 | no |
| 5 | Tuned MLP (100,100), α=1e-3 | — | 87.0 | no gain over 6 |
| 6 | **final:** merged types, non-disputed pool (n=438) | **77.9 ± 2.4** (XGB) | **87.4 ± 0.5** (MLP) | PASS both; control: DeMeo-only pool gives 91.4 complex — the all-objects miss is tier-2 C/X label noise (discrepancy D2-R3) |

## Reproducibility

`PYTHONPATH=src .venv/bin/python baselines/klimczak2021.py` — deterministic
(fold seeds 100–109, model seeds = fold seeds).
