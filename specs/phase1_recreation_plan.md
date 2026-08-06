# Phase 1 Recreation Plan — Minimal Validation Before Real Work

**Prepared:** August 6, 2026 · Data already on disk under `MAST/data/raw/` (verified).
**Governing principle:** every recreation exists to certify a pipeline component we need for MAST — not to do literature archaeology. If a recreation doesn't validate something we reuse, it's deferred. Total budget: **~1 focused week** (2 calendar weeks max), then P1 exits regardless.

---

## What we recreate (4 items) and what each one certifies

### R1. DeMeo 2009 PCA reproduction — *certifies: spectra ingestion + preprocessing*
**Effort: ~half a day. Do this first; everything depends on it.**

The PDS bundle already on disk contains the answer key: `busdemeo2009/.../data/pcscores.tab` (published PC scores for the 371 objects) and `meanspectra.tab` (24 class templates). Recreate the canonical preprocessing — spline fit, slope removal, resample to 41 channels (0.45–2.45 µm, 0.05 µm steps), PCA — from the raw SMASS/MITHNEOS spectra and compare our PC scores to theirs.

- **Pass:** |r| ≥ 0.99 between our PC1–PC3 and published scores (sign/rotation flips allowed).
- **Why it matters:** this is the only ground truth available for the *preprocessing* itself. If this fails, nothing downstream is interpretable. It also becomes a permanent unit test (`test_preprocessing_demeo2009.py`).

### R2. Penttilä 2021 FFNN — *certifies: label harmonization + supervised training loop*
**Effort: 1–2 days.**

Rebuild the 586-spectrum set (DeMeo 2009 + MITHNEOS, deduplicated), collapse to their 11 classes, train their single-hidden-layer net (30 tanh units, softmax).

- **Minimal deviations allowed (documented):** 10-fold object-level CV instead of leave-one-out; single network instead of the 5-vote ensemble. Both cut compute ~15× and shift accuracy by ≲1 pt.
- **Pass:** overall accuracy within **90.6 ± 3 pts**, and per-class confusion structure qualitatively matching theirs (S/Q and C/B/X confusions dominant).
- **Why it matters:** first end-to-end check that our label table, class-collapse mapping, dedup, and training/eval loop are sound. This is also the baseline the paper's headline comparison uses.

### R3. Klimczak 2021 metric frame — *certifies: the honest metric stack + splits*
**Effort: 1–2 days. Runs in parallel with R2 (shares the label table).**

Their setup: 504 objects (drop classes with <10 members), 12 types and 4 complexes, features = top-5 PCs + slope, models = XGBoost and MLP, 5-fold stratified CV, **balanced accuracy**.

- **Pass:** type-level balanced accuracy within **76.8 ± 3** and complex-level within **90.0 ± 3** for the better of the two models.
- **Why it matters:** this validates the exact metric/reporting machinery (balanced acc, macro-F1, per-class tables, object-level stratified folds) that every MAST result will be quoted in. Getting the *hard, honest* number right matters more than the headline numbers.

### R4. Gaia-side ingestion check against Delbo 2026 — *certifies: Gaia parsing + the transfer test set*
**Effort: 1 day. Deliberately NOT a full recreation.**

We do not rebuild their KDE classifier. We validate our Gaia pipeline against their published intermediate numbers, which are already on disk (`gaia_labels_delbo/data sheet 1.csv`):

1. Parse all 20 chunks → 60,518 objects with 16-band spectra, flags, S/N. **Pass: count = 60,518 exactly.**
2. Apply their reference-set cuts (S/N ≥ 50 + literature spectral labels via MP3C/classy). **Pass: ~2,653 objects recovered (±5%).**
3. Apply the S/N > 20 usability cut. **Pass: ~36,566 (±2%).**
4. Cross-match our labeled-overlap table with their supplement's classifications; check headline agreement on the easy classes only (S, V — they report >92% and 99%). **Pass: our overlap table reproduces their per-class counts within ~5%.**
- **Why it matters:** these counts certify the ingestion, quality-flag, and label-crossmatch code that defines benchmark B2 (the paper's primary benchmark). A simple 3-PC + KDE spot-check is a *stretch goal only* — skip if the counts pass.

---

## What we deliberately do NOT recreate in Phase 1

| Skipped | Why | When it happens instead |
|---|---|---|
| Tang 2025 ASC-Net reimplementation | Comparison baseline, not a pipeline validator; expensive (augmentation machinery) | P4, when the baseline table is assembled |
| Mahlke MCFA | Released code — run as-is, never rebuild | P4 (and `classy` already used in P1 for labels) |
| DeMeo decision-tree classifier | `classy` implements it; we run it, not rebuild it | Used in P1 label harmonization directly |
| Klimczak 2022 survey simulations | Validates nothing we reuse in P2–P3 | Only if reviewers ask |
| Sullivan transfer experiment | Its *protocol* is absorbed into benchmark B2; the thesis models aren't needed | B2 design (P4) |
| Full Delbo KDE classifier | Counts + agreement checks certify what we need | P4 baseline table |

---

## Stopping rules (anti-stall mechanics)

1. **Time-box:** each R-item gets a hard cap — R1: 1 day, R2: 3 days, R3: 3 days, R4: 2 days — including debugging. Caps are generous (~2× estimates).
2. **On a miss:** spend at most the remaining cap diagnosing. Then run the internal-validity checklist: fixed seeds reproduce; no object appears in both train and test; confusion matrix physically sensible; class counts match the source paper's table. If all pass, **log the discrepancy and proceed** — record in `specs/discrepancy_log.md` with the delta, a one-paragraph hypothesis, and impact assessment.
3. **Why proceeding is defensible:** every comparison in the eventual paper is run under *our* protocol on *identical splits* (plan §5.2). Literature reproduction to the decimal is a courtesy, not a dependency — what G1 and the paper require is internal validity. A ±3-pt tolerance already absorbs the known sources of drift (LOO→10-fold, ensemble→single net, label-version differences, seed variance).
4. **Hard failure only if:** R1 fails (preprocessing wrong — must fix, everything depends on it) or R2/R3 land >10 pts off (indicates a label-table or leakage bug, not drift). These block; nothing else does.

---

## Concrete execution order

```
Day 1        R1  (preprocessing vs pcscores.tab)  → freeze preprocessing module
Days 2–4     R2 + R3 in parallel (shared label table built on Day 2 with classy)
Day 5        R4  (Gaia counts + Delbo cross-match)
Day 5–6      Freeze: splits (object-level, hashed manifest), label table v1,
             baseline_results.md, discrepancy_log.md
```

Suggested layout in the `MAST/` folder:

```
src/mast/preprocessing.py     # R1-certified: spline, slope removal, resampling, PCA
src/mast/labels.py            # label harmonization (classy-backed), collapse maps
src/mast/splits.py            # object-level folds; writes hashed split manifests
baselines/penttila2021.py     # R2
baselines/klimczak2021.py     # R3
baselines/gaia_ingest_check.py# R4
tests/test_preprocessing_demeo2009.py   # R1 as a permanent unit test
specs/baseline_results.md     # the four pass/fail verdicts + numbers
specs/discrepancy_log.md      # any logged misses
data/processed/               # parquet: spectra table, label table, split manifests
```

## Exit criteria (P1 done, P2 pretraining starts)

1. R1 passes (hard requirement); R2–R4 pass or have logged, internally-valid discrepancies.
2. Split manifests frozen and hashed (these exact splits are used for the rest of the project — no regeneration later).
3. Label table v1 frozen: per-object (Bus-DeMeo type, complex, Mahlke class, disputed flag).
4. `baseline_results.md` records the four verdicts in ≤1 page.

Nothing else. No extra baselines, no early architecture experiments, no tuning beyond each paper's published settings. P2 (pretraining) starts the day the checklist closes.
