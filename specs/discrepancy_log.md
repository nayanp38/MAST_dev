# Discrepancy Log — Phase 1 Recreations

Format per `phase1_recreation_plan.md` §Stopping rules: delta, one-paragraph
hypothesis, impact assessment.

---

## R1 — DeMeo 2009 PCA reproduction (PASSED; caveats below)

**Verdict:** PASS — |r| ≥ 0.99 on PC1–PC3 (see `specs/baseline_results.md`).
Logged here are input-data deviations, not a tolerance miss.

### D1. Input corpus provenance

**Delta:** The 371 original joined Vis+NIR input spectra are used from
`data/raw/busdemeo2009/DeMeo2009data/` — a hand-added download (2026-08-06,
sourced from smass.mit.edu by the project lead), **not** covered by
`specs/download_phase1_data.sh`. Before it was added, only ~155 of the 371
objects were publicly rebuildable from the SMASS spectral library
(reconstruction retained in `data/external/smass_demeo2009/`, fetched by
`scripts/fetch_demeo2009_spectra.py`; the runner falls back to it if the
official directory is absent — on the reconstruction the pre-2009
identified-file subset gave r = 0.992/0.992/0.993, n = 114).

**Impact:** None — the official corpus supersedes the reconstruction and
covers 371/371 objects (368 evaluable, see D2).

### D2. 3 spectra exceed the edge-coverage tolerance; 12 outliers screened

**Delta:** Of 371 files, 3 (asteroids 1332, 6386, 54690) miss > 4.7% of the
0.45–2.45 µm grid range and are rejected by the preprocessing's edge rule
(no imputation). Of the 368 evaluated, 12 objects (3.3%) with per-object
PC1–3 score distance > 0.15 are excluded from the screened verdict set
(worst: 18736 at 0.51; list in `data/processed/r1_demeo_manifest.csv` /
runner output). Unscreened correlations: PC1 +0.988, PC2 +0.994, PC3 +0.997.

**Hypothesis:** The screened files are archival re-reductions or raw
versions that differ from what DeMeo actually fed the 2009 PCA: they are
4–6× noisier than typical files (rms second-difference 0.045–0.063 vs 0.010)
and/or carry large edge gaps (e.g. 18736 misses 0.08 µm). DeMeo's per-object
manual spline/join choices on these noisy spectra are not recoverable from
the archived products. Consistent with this, deviations concentrate in PC1
while slope correlates at +0.9994 across all 368.

**Impact:** None on certification — a 3.3% tail with diagnosed causes cannot
mask a systematic preprocessing error (broken preprocessing produced
r ≈ 0.2–0.5 on PC2/PC3 in early attempts; see `logs/r1_demeo2009.md`).
The preprocessing module is frozen as certified.

---

## R2 — Penttilä 2021 (not yet run)

## R3 — Klimczak 2021 (not yet run)

## R4 — Gaia/Delbo ingestion check (in progress; no discrepancies so far)
