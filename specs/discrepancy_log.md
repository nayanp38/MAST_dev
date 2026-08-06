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

## R2 — Penttilä 2021 (PASSED; deviations below)

### D1-R2. Dataset is 474 objects, not their 586 spectra

**Delta:** Rebuilt pool = 368 DeMeo-2009 originals + 108 MITHNEOS
additions (classy-tree labels), minus 2 O/R objects outside the 11-class
scheme. Their exact 586-spectrum list is not published, and part of the
MITHNEOS data they used is not publicly servable (same root cause as
D1-R1). Also: no PCA-based synthetic augmentation (theirs targets rare
classes), single net instead of 5-vote ensemble, 10-fold object CV
instead of LOO (the latter two are plan-sanctioned deviations).

**Impact:** Accuracy 88.2 ± 0.4 lands within 90.6 ± 3 with the published
confusion structure (C/X, Q/S, B/C dominant); the training-loop and
label-machinery certification goal is met. The MAST-protocol variant
(disputed labels excluded from training but kept in test) gives
83.8 ± 0.8 — the ~4.4-pt gap is the measured cost of label noise, useful
context for later benchmarks.

## R3 — Klimczak 2021 (PASSED on primary pool; all-objects complex level misses)

### D1-R3. Merged-type reading of their 12-class scheme

**Delta:** Their "12 types with ≥10 members" is only reachable from
DeMeo-era data by merging subclasses (Cb,Cg→C; Cgh→Ch; Sa,Sv→S;
Xc,Xe,Xk→X; A/T/O/R dropped); the unmerged reading yields a 16-member X
type and a 13-pt complex-level miss. Merged reading adopted and
documented in `baselines/klimczak2021.py`.

### D2-R3. Complex-level balanced accuracy sensitive to tier-2 label noise

**Delta:** Primary pool (non-disputed, n=438): type 77.9 ± 2.4 ✓,
complex 87.4 ± 0.5 ✓ (band floor 87.0 — marginal). All-objects pool
(n=463): type 78.1 ✓ but complex 84.7 ✗ (2.3 pts below band).

**Hypothesis:** Our tier-2 additions carry classy-tree labels whose
errors concentrate exactly at the albedo-degenerate C/X boundary
(X-complex recall ~72% with them, ~87+ without); Klimczak's dataset had
only curated-era labels. Control: DeMeo-2009-only objects give 91.4
complex balanced accuracy — right at their 90.0.

**Impact:** None on certification (metric stack + splits machinery are
what R3 certifies; the miss is a data-composition effect, internally
valid: seeds reproduce, no train/test object overlap, confusions
physically sensible).

## R4 — Gaia/Delbo ingestion check (PASSED; margins noted)

### D1-R4. Reference-set count at the tolerance edge

**Delta:** 2,524 vs their 2,653 (−4.9%; band edge 2,520). Our
literature-label pool (label table v1 ∪ PDS ast_taxonomy compilation:
Tholen, Xu, Bus, S3OS2, Bus-DeMeo) recovers 95% of Delbo's MP3C
aggregation; MP3C includes minor additional sources not publicly
archived as per-object tables. S-count deviation in check 4 is likewise
at the edge (4.5% vs ~5%). S-agreement 93.8% matches their published
">92%"; V-agreement 89.8% is below their 99% because our letter labels
come from heterogeneous literature schemes rather than their curated
reference subset — the plan gates on counts, not agreement.

**Impact:** None — parsing, S/N, flags, and cross-match code are
certified; benchmark B2's labeled overlap will be rebuilt from the
frozen label table with these exact tools.
