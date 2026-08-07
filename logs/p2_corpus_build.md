# Run Log — P2 Pretraining Corpus Build

**Date:** 2026-08-06 · **Status:** built + frozen manifests ·
**Code:** `src/mast/{filters,scalars,corpus,tokenize,pretrain_data}.py`,
`scripts/{fetch_svo_filters,build_corpus}.py`

## Goal in the scheme of the research

The unlabeled multi-survey corpus (plan §2.1) that MAST pretrains on, in the
common quadruplet representation (λ_eff, Δλ, R, σ) + scalar tokens (§2.3).

## Exact data sources used

| Source | Input | Output records | Notes |
|---|---|---|---|
| Gaia DR3 SSO | `data/processed/gaia_wide.parquet` (R4-certified ingest of `data/raw/gaia_dr3_sso`) | 60,518 | 12 interior bands kept (spec band-drop); flags carried as token inputs; S/N>20 tagged |
| SDSS (Sergeyev & Carry 2021) | `data/raw/sdss_sergeyev2021/sso.dat.gz` | 1,031,859 (spec ~1.036M ✓) | `bknown=1` linked SSOs only; per-band quality flags; phase angle kept |
| SkyMapper (Sergeyev 2022) | `data/raw/skymapper_sergeyev2022/skm-obs.dat.gz` | 265,411 | 880,528 per-filter rows → per-(object, night) multi-band records; single-band nights dropped (reflectance needs ≥2 bands) |
| MOVIS (Popescu 2018) | `data/raw/movis_photometry/movis-m.dat.gz` | 43,241 | per-night Y/J/H/Ks magnitude sets; phase angle kept |
| Albedo/H scalars | NEOWISE V2.0 + Alí-Lagoa 2018 + AcuA (`data/raw/{neowise,akari*}`) | 128,397 objects | priority NEOWISE > AL18 > AcuA; inverse-variance epoch aggregation; sanity: Vesta pV 0.355, Nysa 0.482, Hygiea 0.058 |
| Filter metadata | SVO FPS → `data/external/filters_svo/` (`scripts/fetch_svo_filters.py`) | 15 filters + curves | λ_eff, FWHM, ZeroPoint, Fsun |

**Total: 1,401,029 records / 519,468 objects** (~"1.2M" spec scale ✓).
Token cache: `data/processed/corpus/tokens_v1/` (70 MB compressed), per-source
dense arrays + `record_index.parquet` + `meta.json` (source parquet SHA256s).

## Documented physics/provenance decisions

1. **Reflectance conversion:** R_b = 10^(−0.4[(m_b−m_pivot) − (m_b,⊙−m_pivot,⊙)]);
   σ_R = 0.4·ln10·R·√(σ_b²+σ_pivot²) (band–pivot covariance ignored).
2. **Solar magnitudes derived from SVO Fsun** (f_ν = Fsun·λ_pivot²/c;
   m = −2.5log₁₀(f_ν/ZP); ZP = 3631 Jy for AB surveys [SDSS, SkyMapper],
   SVO Vega ZP for VISTA). Validated on SDSS vs Willmer 2018/Holmberg 2006:
   u−g 1.25 (lit 1.28–1.43), g−r 0.48 (0.44–0.46), r−i 0.11 (0.11),
   i−z 0.02 (0.03) — ±0.03 mag; constant per band, absorbed by instrument
   embeddings. Chosen over hand-collected literature values for
   single-source consistency across all three surveys (Sergeyev 2022 used
   untabulated "Holmberg adapted to SkyMapper" — not reproducible exactly).
   **Gotcha: SVO's default ZeroPoint is Vega even for SDSS filters.**
3. **Pivot band** = nearest 0.55 µm present (SDSS r, SkyMapper g/v, MOVIS Y);
   pivot-offset scalar token = log(λ_pivot/550 nm); Gaia identity pivot (no
   token).
4. **Gaia Δλ:** λ/R with R(λ) read approximately from Montegriffo et al.
   2023 resolution curves (table in `filters.py`); A7 ablation supersedes.
5. **σ_H = 0.2 mag** fixed (catalogs give no H uncertainty).
6. **Held-out manifest** `corpus_holdout_k20_seed42`: object-level k=20,
   stratified by number-of-sources; fold 0 = 5% held-out (25,974 objects);
   folds 1–5 = frozen 25% HPO subsample (350,188 records).

## Verification

- Record counts vs spec table (§2.1): SDSS within 0.4%; totals ✓
  (`tests/test_tokenize.py`).
- Physical sanity: median SDSS R_g 0.86 / R_i 1.06 / R_z 0.98 (redward
  slope + 0.9 µm band ✓).
- Solar-color pins: `tests/test_filters.py` (5 tests).
- No-imputation invariants: absent tokens zeroed + model padding-invariance
  test (`tests/test_model.py::test_padding_invariance`).
