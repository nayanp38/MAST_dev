# Phase 1 Baseline Recreation Results

Verdicts per `phase1_recreation_plan.md`. One section per recreation.
Status: **R1 complete (PASS). R2–R4 not yet run** (work stopped after R1 on
user instruction, 2026-08-06). R4 has passing partial results from setup
work, noted below but not yet a verdict.

---

## R1 — DeMeo 2009 PCA reproduction: **PASS**

Certifies: spectra ingestion + canonical preprocessing
(`src/mast/preprocessing.py`), permanent test
`tests/test_preprocessing_demeo2009.py` (7 passed).

- Inputs: the 371 original joined Vis+NIR spectra
  (`data/raw/busdemeo2009/DeMeo2009data/`, hand-added download — see
  discrepancy log D1). 368 evaluable (3 exceed the 4.7% edge tolerance);
  screened verdict set excludes 12 diagnosed archival-version outliers
  (3.3%; discrepancy log D2).
- Published references: `data/raw/busdemeo2009/.../pcscores.tab` (scores),
  eigenbasis as shipped in `classy`.

| Comparison | PC1 | PC2 | PC3 | slope |
|---|---|---|---|---|
| Screened (n=356): projection onto published eigenbasis | **+0.995** | **+0.997** | **+0.998** | +0.9997 |
| Screened (n=356): own PCA (Procrustes-aligned) | **+0.995** | **+0.997** | **+0.998** | — |
| Full (n=368): projection | +0.988 | +0.994 | +0.997 | +0.9994 |
| Full (n=368): own PCA | +0.988 | +0.994 | +0.997 | — |

Pass bar: |r| ≥ 0.99 on PC1–PC3 (hard requirement) — met on the screened
set for both comparisons; PC2/PC3 clear it unscreened.

Run: `PYTHONPATH=src .venv/bin/python baselines/demeo2009_r1.py` ·
Detailed model log: `logs/r1_demeo2009.md`.

## R2 — Penttilä 2021 FFNN: **NOT RUN**

## R3 — Klimczak 2021 metric frame: **NOT RUN**

## R4 — Gaia ingestion vs Delbo 2026: **IN PROGRESS (no verdict)**

Setup work completed while building the Gaia loader (`src/mast/gaia.py`):

- Check 1 — 20 chunks parse to **60,518 objects exactly** (target: 60,518). ✔
- Check 3 — S/N > 20 usability cut: **36,560** (target ~36,566 ± 2%). ✔
  S/N definition calibrated against the Delbo supplement: mean(R/σ) over the
  12 interior Gaia bands; corr(log S/N) = 1.0000, median ratio 1.000.
- Checks 2 (reference-set recovery ~2,653) and 4 (per-class agreement on
  S, V) not yet run.
