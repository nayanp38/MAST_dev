# Phase 1 Baseline Recreation Results

**Status: P1 complete — all four recreations PASS** (2026-08-06). Frozen
artifacts hashed in `data/processed/P1_FREEZE.json`. Detailed model logs in
`logs/`; deviations in `specs/discrepancy_log.md`.

## R1 — DeMeo 2009 PCA reproduction: **PASS**

Certifies preprocessing (`src/mast/preprocessing.py`); permanent test
`tests/test_preprocessing_demeo2009.py`. Inputs: the 371 original spectra
(`data/raw/busdemeo2009/DeMeo2009data/`); 368 evaluable, screened set
excludes 12 diagnosed archival-version outliers (log D2-R1).

| vs published pcscores.tab | PC1 | PC2 | PC3 | slope |
|---|---|---|---|---|
| Screened (n=356), projection **and** own PCA | **+0.995** | **+0.997** | **+0.998** | +0.9997 |
| Full (n=368) | +0.988 | +0.994 | +0.997 | +0.9994 |

Bar: |r| ≥ 0.99 on PC1–PC3 → met (PC2/PC3 clear it unscreened).

## R2 — Penttilä 2021 FFNN: **PASS**

`baselines/penttila2021.py` — 474-object rebuilt set (vs their 586
spectra; log D1-R2), 11 collapsed classes, 30-tanh-unit net, 10-fold
object CV, 5 seeds. **Accuracy 88.2 ± 0.4** (target 90.6 ± 3 ✓). Top
confusions C/X (83), Q/S (42) — S/Q and C/B/X dominant as published ✓.
MAST protocol (disputed excluded from training, kept in test): 83.8 ± 0.8.

## R3 — Klimczak 2021 metric frame: **PASS**

`baselines/klimczak2021.py` — 12 merged types / 4 complexes, top-5 PCs +
slope, XGBoost + MLP, 5-fold CV × 10 fold-seeds, balanced accuracy.
Primary pool = non-disputed labels (n=438; log D2-R3).

| Balanced accuracy (best model) | result | target | |
|---|---|---|---|
| Type level (12) | **77.9 ± 2.4** (XGBoost) | 76.8 ± 3 | ✓ |
| Complex level (4) | **87.4 ± 0.5** (MLP) | 90.0 ± 3 | ✓ (marginal) |

All-objects pool (with tier-2 label noise): 78.1 type ✓ / 84.7 complex ✗
— logged, attributed to classy-tree C/X label noise (log D2-R3).

## R4 — Gaia ingestion vs Delbo 2026: **PASS**

`baselines/gaia_ingest_check.py` (ingestion in `src/mast/gaia.py`; S/N =
mean R/σ over the 12 interior bands, calibrated to their supplement,
corr 1.000).

| Check | ours | target | |
|---|---|---|---|
| 1. Objects parsed from 20 chunks | 60,518 | 60,518 exact | ✓ |
| 2. Reference set (S/N≥50 + literature label) | 2,524 | ~2,653 ± 5% | ✓ (marginal; log D1-R4) |
| 3. Usability cut (S/N>20) | 36,560 | ~36,566 ± 2% | ✓ |
| 4. Overlap counts: S / V | dev 4.5% / 2.1% | ≤ ~5% | ✓ (agreement 93.8% / 89.8%) |

## Frozen at P1 exit

Label table v1 (4,531 objects; 479 Bus-DeMeo, 27 disputed = 5.6%), split
manifests `b1_folds_k10_seed42` (479), `r2_penttila_folds_k10_seed42`
(474), `r3_klimczak_folds_k5_seed100` (438), R1 manifest, VIS+NIR dataset
— sha256 in `P1_FREEZE.json` / `splits/SHA256SUMS`. Verify anytime with
`.venv/bin/python scripts/freeze_p1.py`. **P2 may start.**
