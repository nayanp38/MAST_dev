# Model/Run Log — R2: Penttilä 2021 FFNN recreation

**Date:** 2026-08-06 · **Status:** PASS (88.2 ± 0.4 vs target 90.6 ± 3) ·
**Code:** `baselines/penttila2021.py` · **Verdict:** `specs/baseline_results.md`

## Goal in the scheme of the research

First end-to-end certification that the label table, class-collapse
mapping, dedup, and supervised training/eval loop are sound (plan §5.2).
Also the baseline that the paper's headline comparison quotes.

## Exact data sources used

| Data | Location | Role |
|---|---|---|
| VIS+NIR dataset (474 objects after O/R drop) | `data/processed/visnir_dataset.parquet` (built by `src/mast/visnir_dataset.py` from `data/raw/busdemeo2009/DeMeo2009data/` + classy mithneos cache) | features: 200 channels, 0.45–2.45 µm, normalized at 0.55 µm, slope retained |
| Labels | `data/processed/label_table_v1.parquet`, collapsed via `PENTTILA_11` (labels.py) | 11 classes |
| Folds | `data/processed/splits/r2_penttila_folds_k10_seed42.csv` (frozen) | 10-fold object-level stratified |

## Model

sklearn `MLPClassifier(hidden_layer_sizes=(30,), activation='tanh')`
(their Matlab patternnet analog), StandardScaler fit per training fold.
Seeds 0–4; reported mean ± std over seeds.

## Tweaks attempted and results

| # | Config | Overall acc | Notes |
|---|---|---|---|
| 1 | Penttilä protocol (all labels at face value), single net, no augmentation | **88.2 ± 0.4** | PASS; top confusions C/X (83), Q/S (42), L/S (24), K/X (20), B/C (15) — S/Q + C/B/X dominant as published |
| 2 | MAST protocol (disputed excluded from train, kept in test) | 83.8 ± 0.8 | not a pass-gate run; measures label-noise cost (~4.4 pts), C/X confusion grows to 115 |

No tuning beyond the paper's published architecture (per plan: "no tuning
beyond each paper's published settings"). Deviations (10-fold vs LOO,
single net vs ensemble, 474 vs 586 spectra, no synthetic augmentation)
in `specs/discrepancy_log.md` D1-R2.

## Reproducibility

`PYTHONPATH=src .venv/bin/python baselines/penttila2021.py` — deterministic
given frozen folds + seeds 0–4.
