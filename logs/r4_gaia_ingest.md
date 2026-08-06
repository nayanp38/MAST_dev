# Model/Run Log — R4: Gaia ingestion check vs Delbo 2026

**Date:** 2026-08-06 · **Status:** PASS (all 4 checks) ·
**Code:** `baselines/gaia_ingest_check.py`, `src/mast/gaia.py`

## Goal in the scheme of the research

Certifies the Gaia DR3 parsing, quality-flag/S-N, and label-crossmatch
code that defines benchmark B2 (the paper's primary transfer benchmark)
and the P2 pretraining corpus ingestion. Deliberately not a rebuild of
Delbo's KDE classifier.

## Exact data sources used

| Data | Location | Role |
|---|---|---|
| Gaia DR3 SSO reflectance chunks (20) | `data/raw/gaia_dr3_sso/` | ingestion input |
| Delbo et al. 2026 supplement | `data/raw/gaia_labels_delbo/data sheet 1.csv` | published intermediate numbers + dr3class1 |
| Label table v1 | `data/processed/label_table_v1.parquet` | literature labels (BDM + Mahlke) |
| PDS ast_taxonomy compilation | `data/external/ast_taxonomy/` (fetched by `scripts/fetch_ast_taxonomy.py`) | Tholen/Xu/Bus/S3OS2/BDM literature classes (MP3C analog) |

## Tweaks attempted and results

| # | Item | Result |
|---|---|---|
| 1 | S/N definition — quadrature over 16 bands | all 60,518 pass S/N>20: wrong |
| 2 | S/N calibration vs their snr column: mean16 / median16 / quad16 / interior means | **mean(R/σ) over the 12 interior bands**: corr(log)=1.0000, median ratio 1.000, max rel dev 6e-2 → adopted in `gaia.compute_snr` |
| 3 | Check 2 pool = label table only (BDM+Mahlke) | 2,327 (−12%): literature pool too narrow |
| 4 | + ast_taxonomy Bus/S3OS2/Tholen | 2,520 — misses band edge (2,520.35) by 0.35 |
| 5 | **+ Xu 1995 (SMASS I)** | **2,524 ✓** (Howell/Barucci/Tedesco add nothing further) |
| 6 | Check 4 letter priority: recent-first (Mahlke early) | S dev 4.7% but agreement 85% — Mahlke scheme mismatch; rejected |
| 7 | **Priority curated-BDM > lit-BDM > Bus > S3OS2 > Tholen > Xu > Mahlke** | S: dev 4.5%, agreement 93.8% (their ">92%" ✓); V: dev 2.1%, agreement 89.8% |

Final: check 1 = 60,518 exact ✓ · check 2 = 2,524 (~2,653 ± 5%) ✓ ·
check 3 = 36,560 (~36,566 ± 2%) ✓ · check 4 counts ✓. Margins logged as
D1-R4.

## Reproducibility

`PYTHONPATH=src .venv/bin/python baselines/gaia_ingest_check.py`
(rebuilds `data/processed/gaia_wide.parquet` if absent; deterministic).
