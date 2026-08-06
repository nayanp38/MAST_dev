# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MAST (Masked Asteroid Spectral Transformer): an ML research project building a self-supervised, missing-data-native, calibrated asteroid taxonomic classifier, targeting a peer-reviewed paper (A&A). The repo currently contains only specs and raw data — no source code yet. Phase 1 (baseline recreations) is the next work to be done.

## Authoritative documents (read before starting related work)

The `specs/` directory is the general project plan 

- `specs/research_plan_novel_asteroid_classifier.md` — the master plan: hypotheses H1–H5, architecture (tokenizer, encoder, heads), HPO search spaces, benchmarks B1–B5, ablation matrix A1–A14, decision gates. Section numbers (e.g. "plan §2.4") are cross-referenced from the other docs.
- `specs/ml_asteroid_classification_field_review.md` — literature review (cited as "Review §N"); includes a citation-correction list (§8) for the eventual paper.
- `specs/phase1_recreation_plan.md` — the current work plan: four recreations (R1–R4), pass/fail tolerances, time-boxes, stopping rules, and the intended `src/mast/` layout.
- `specs/phase1_data_acquisition_guide.md` — every dataset's source URL and its role (unlabeled corpus A1–A4, labeled pool B1–B7, auxiliary scalars, filter curves).

## Data rules

- `data/raw/` is a download cache, already populated — treat it as **read-only**. Never edit, regenerate, or reorganize it by hand; re-fetch via `bash specs/download_phase1_data.sh [target_dir]` (defaults to `./data/raw`). Three sources are not covered by the script (Delbo 2026 supplement, MP3C TAP, SVO filter curves) — see the script header.
- Derived outputs go to `data/processed/` (parquet: spectra table, label table, split manifests).
- Once split manifests are frozen and hashed (P1 exit), they are never regenerated — every later experiment uses those exact splits.

## Non-negotiable methodology constraints (from the plan)

- **Document everything.** When logging each model, summarize the goal of the model in the scheme of the research goal, log the exact data sources used from the `data/` directory, include the different tweaks to that model attempted, and for each tweak the results found. The key here is organization, ease of comparison, and scientific reproducibility. 
- **Object-level splits everywhere.** All observations of one asteroid stay on one side of every split; observation-level random splits are leakage.
- **No imputation.** Missing modalities are absent tokens, never mean/k-NN filled (imputation appears only as ablation A4 arms).
- **Orbital elements excluded** from primary models (leakage ablation A10 only).
- **Tolerances, not exactness, for recreations:** R1 requires |r| ≥ 0.99 vs published PC scores (hard requirement); R2/R3/R4 pass within stated ±3 pt / count tolerances. On a miss, run the internal-validity checklist, log to `specs/discrepancy_log.md`, and proceed — only R1 failure or >10 pt misses block.
- All reported numbers are mean ± std over seeds; no single-split results.

## Planned code layout (from phase1_recreation_plan.md)

```
src/mast/preprocessing.py      # spline fit, slope removal, resample to 41 channels, PCA
src/mast/labels.py             # label harmonization (classy-backed), collapse maps
src/mast/splits.py             # object-level folds; writes hashed split manifests
baselines/                     # penttila2021.py, klimczak2021.py, gaia_ingest_check.py
tests/test_preprocessing_demeo2009.py   # R1 becomes a permanent unit test
logs/baseline_results.md      # pass/fail verdicts (created at P1 exit)
logs/discrepancy_log.md       # logged recreation misses
```

## Key external tools

- `classy` (`pip install space-classy`) — label harmonization backbone and DeMeo decision-tree baseline; run released tools (classy, Mahlke's MCFA) as-is, never rebuild them.
- `rocks` (`pip install space-rocks`) — per-object albedo/H/G best estimates via SsODNet.
- `astroquery` — Gaia TAP (`gaiadr3.sso_reflectance_spectrum`) and SVO filter curves.

## Repo state notes

- Dataset directory names in `data/raw/` differ slightly from the acquisition guide's suggested layout (e.g. `movis_photometry`/`movis_taxonomy` vs `movis`, `gaia_labels_delbo`/`gaia_labels_tinautruano` vs `gaia_labels`) — trust what's on disk.
