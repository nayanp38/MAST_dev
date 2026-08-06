# Model/Run Log — Label table v1 + object-level splits

**Date:** 2026-08-06 · **Status:** built (formal freeze at P1 exit) ·
**Machinery:** `src/mast/labels.py`, `src/mast/splits.py`,
`scripts/build_label_table.py`, tests in `tests/test_labels_splits.py`.

## Goal in the scheme of the research

Plan §2.2 requires a per-object three-way label table — (Bus-DeMeo class,
Bus-DeMeo complex, Mahlke class + probability) with a disputed flag — that
R2/R3 train against and that later phases fine-tune on. Plan §2.4 requires
object-level, stratified, hashed split manifests (B1: 10-fold CV) frozen for
the entire project. No learned model here; the deliverables are data
artifacts + deterministic machinery.

## Exact data sources used

| Data | Location | Role |
|---|---|---|
| DeMeo 2009 classes (371 obj) | `data/raw/busdemeo2009/.../demeotax.tab` | tier-1 Bus-DeMeo labels (curated, VIS+NIR) |
| MITHNEOS/SMASS joined VIS+NIR spectra | classy cache `~/Library/Caches/classy/mithneos` (mirrors smass.mit.edu sp*/dm* products) | tier-2 inputs, classified with classy's released DeMeo tree |
| classy index | classy cache | observation dates for tier-2 recency |
| Mahlke 2022 aggregated classes | `data/raw/mahlke2022_vizier/asteroid.dat.gz` (frozen VizieR snapshot) | Mahlke class + probability per object |

## Harmonization rules (documented deviations from plan §2.2)

- All Bus-DeMeo sources used are VIS+NIR, so rule (1) [VIS+NIR > VIS-only]
  never discriminates; VIS-only schemes (Bus, Tholen, S3OS2) are excluded
  from v1 by design.
- Curated literature labels (DeMeo 2009) outrank our classy-tree-derived
  labels; the plan's "most recent wins" applies within a tier (tier-2
  majority vote with latest-observation tiebreak). Rationale: an automated
  tree run must not overwrite a curated published label by recency alone.
- disputed = complex-level disagreement between tiers, or an unresolvable
  complex tie within tier 2 for objects with no tier-1 label. Subclass
  wobble within a complex (S vs Sq) is not flagged.
- Labels are canonicalized to the 24 Bus-DeMeo classes ('Sw' → 'S',
  'Sq:' → 'Sq'); raw published strings are kept in `bdm_demeo2009_raw`.

## Tweaks attempted and results

| # | Attempt | Result |
|---|---|---|
| 1 | Tier-2 via manually constructed `classy.Spectrum(wave, refl)` + `classify(taxonomy='demeo')` | **Broken** — mis-classifies: (2) Pallas → V with PC2 = +0.55 (published −0.06); (3) Juno → V; S-types → Q/V. Root cause in classy's handling of manually built spectra (bypasses loader-side preparation). 226–395 spectra classified, 78/122 tier-overlap objects conflicting — all spurious. |
| 2 | Same spectra through classy's own `Spectra` loader per object + released tree | Correct: Pallas → B, scores match the loader path and our certified R1 pipeline (−0.43, −0.10 vs published −0.57, −0.06). Slow (~SsODNet lookup per object) but run once with incremental progress cache (`data/processed/mithneos_demeo_per_spectrum.csv`). |
| 3 | Disputed rule v1: tier conflict OR any tier-2 complex tie | 84/429 disputed (20%) — overcounts: ties on objects that tier 1 resolves. Changed to: tier conflict OR (tie AND no tier-1 label). |

Split machinery: deterministic stratified round-robin over
sha256(seed:object_id) ordering (no library-version dependence); manifests
written once with sha256 sidecars (`data/processed/splits/`), re-runs verify
byte-identity and refuse regeneration. B1 = 10-fold, seed 42, stratified by
`bdm_class`; inner-val rule (20% of each training fold) recorded in the
manifest metadata.

## Final numbers

- Label table v1: **4,531 objects** — 479 with a harmonized Bus-DeMeo class
  (371 tier-1 DeMeo 2009 + 108 tier-2 classy-tree), 4,526 with a Mahlke
  class, **27 disputed (5.6%** of the BDM pool — inside the plan's expected
  5–10%).
- Tier-2 inputs: 369 classified VIS+NIR MITHNEOS spectra over 275 objects
  (`data/processed/mithneos_demeo_per_spectrum.csv`).
- Tier-1/tier-2 overlap (165 objects): 64.8% class-level, 84.2%
  complex-level agreement — consistent with published DeMeo-tree
  self-consistency; complex-level conflicts are the disputed set.
- BDM complexes: S 249 · end 110 · C 71 · X 49.
- B1 manifest `b1_folds_k10_seed42`: 479 objects, folds 48×9 + 47,
  sha256 `93756cbe3b18e284156c55480709dd7b…` (full hash in the JSON sidecar
  and `data/processed/splits/SHA256SUMS`).

## Reproducibility

- `PYTHONPATH=src .venv/bin/python scripts/build_label_table.py` (idempotent;
  tier-2 progress cache resumes; manifest write verifies instead of
  regenerating)
- Tests: `PYTHONPATH=src .venv/bin/pytest tests/test_labels_splits.py` (9 passed)
