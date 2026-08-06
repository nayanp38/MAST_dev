# Model/Run Log — R1: DeMeo 2009 PCA reproduction

**Date:** 2026-08-06 (updated same day after the official input corpus was
added) · **Status:** PASS · **Verdict summary:** `specs/baseline_results.md`

## Goal in the scheme of the research

R1 certifies the canonical spectral preprocessing that every MAST component
downstream consumes (plan §: preprocessing → tokenizer input; R2/R3 reuse the
same 41-channel pipeline). The published PC scores of the 371-object Bus-DeMeo
set are the only available ground truth for the preprocessing itself. No
learned model is trained here; the "model" is the deterministic pipeline
`src/mast/preprocessing.py` + PCA.

## Exact data sources used

| Data | Location | Role |
|---|---|---|
| Published PC scores (slope, PC1–PC5, 371 objects) | `data/raw/busdemeo2009/ast.bus-demeo.taxonomy/data/pcscores.tab` | answer key |
| Bus-DeMeo classes per object | `data/raw/busdemeo2009/.../demeotax.tab` | object list |
| **Original 371 input spectra** | `data/raw/busdemeo2009/DeMeo2009data/` (hand-added download, 2026-08-06; not covered by the download script) | pipeline input (primary) |
| Joined Vis+NIR spectra (sp*/dm* runs) | `data/external/smass_demeo2009/` (fetched from smass.mit.edu via `scripts/fetch_demeo2009_spectra.py`) | pipeline input (fallback; used before the official corpus existed) |
| Published eigenbasis (40-ch mean + 5 eigenvectors) | `classy` package, `classy.taxonomies.demeo` | projection comparison |
| Observation dates per spectrum file | classy index (cache at `~/Library/Caches/classy`) | pre-2009 selection |
| SMASS2 / MITHNEOS PDS bundles | `data/raw/smass`, `data/raw/mithneos` | evaluated as inputs; jointly cover only 27/371 objects → not used |

## Tweaks attempted and results

Comparison metric: Pearson r between our PC1–PC3 scores and `pcscores.tab`
(projection onto published eigenbasis unless stated). "pre-2009" = objects
whose chosen file has an observation date ≤ 2008 (classy index).

| # | Pipeline variant | Selection | n | PC1 / PC2 / PC3 r | Notes |
|---|---|---|---|---|---|
| 1 | Spline w/ scipy default smoothing (s≈len(w)) → resample → slope fit on grid (scaled line) | prefer sp, lowest run | 152 | +0.95 / +0.20 / +0.53 | default `UnivariateSpline` smoothing destroys band shapes — the main bug |
| 2 | same | latest pre-2009 obs | 113 | +0.96 / +0.21 / +0.48 | file choice was not the problem |
| 3 | classy semantics: normalize at nearest 0.55 → slope fit on native grid (translated line) → interpolating-spline resample | latest pre-2009 | 113 | +0.983 / +0.985 / +0.993 | fixing smoothing + translated slope line is the big jump |
| 4 | as 3 | file identification (min score distance) | 155/113 | +0.975–0.986 / +0.976–0.988 / +0.99 | post-2009-only objects drag the full set |
| 5 | slope fit restricted to 0.45–2.45 on native grid | identification, pre-2009 | 113 | +0.988 / +0.990 / +0.994 | small gain |
| 6 | normalize → resample to 41 channels → slope fit on the resampled grid (translated to (0.55, 1)) → divide | identification, pre-2009 (reconstructed corpus) | 114 | +0.992 / +0.992 / +0.993 | matches the paper's stated order; halves median per-object score distance (0.080 → 0.041) |

### After the official 371-file corpus was added (same day)

| # | Tweak | n | PC1 / PC2 / PC3 r | Notes |
|---|---|---|---|---|
| 7 | variant 6 on official corpus, interpolating spline, hard full-coverage requirement | 291 | +0.993 / +0.995 / +0.995 | 80 files rejected (ragged rows; short edges) |
| 8 | + robust line parser, + constant-value edge extrapolation ≤ 4.7% of range (classy's limit) | 363 | +0.987 / +0.994 / +0.994 | new noisy/extrapolated files drag PC1; linear edge extrapolation tested and much worse (PC3 0.885) |
| 9 | **final:** + noise-adaptive smoothing spline (σ̂ from second differences, s = nσ̂²; smooth files ≈ interpolated) + unnumbered-object matching | 368 | full: +0.988 / +0.994 / +0.997 · screened (dist ≤ 0.15, −12 objs): **+0.995 / +0.997 / +0.998** | outliers are 4–6× noisier archival re-reductions; slope r = +0.9994 (full) |

Own-PCA comparison (full pipeline incl. our SVD PCA, orthogonal-Procrustes
aligned, allowed by plan) matches the projection numbers to ±0.001 in every
configuration.

Key implementation facts locked into `src/mast/preprocessing.py`:

1. Normalize at the **nearest native sample** to 0.55 µm.
2. Spline-fit and resample with a **noise-adaptive smoothing** cubic spline
   (s = n·σ̂², σ̂ from second differences); edges missing ≤ 4.7% of the grid
   range are extended with **constant** values, larger gaps are rejected.
3. Fit the slope line **on the 41 resampled channels**, **translate** (not
   scale) it to pass through (0.55, 1), divide it out; γ = line slope.
4. PCA on **40 channels** (0.55 µm channel dropped), channel-mean subtracted.

## Reproducibility

- Run: `PYTHONPATH=src .venv/bin/python baselines/demeo2009_r1.py`
- Permanent test: `PYTHONPATH=src .venv/bin/pytest tests/test_preprocessing_demeo2009.py` (5 passed)
- Per-object file identification frozen in `data/processed/r1_demeo_manifest.csv`
- Environment: `.venv` (Python 3.12.11; numpy/scipy/pandas/sklearn/xgboost;
  space-classy 0.8.8 with populated cache — re-fetch via `classy status` → option 2)
- Pipeline is deterministic (no seeds involved).

## Discrepancies

See `specs/discrepancy_log.md` D1 (155/371 public coverage) and D2
(file identification + pre-2009 primary set; full-set r 0.98–0.99).
