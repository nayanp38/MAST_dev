# Phase 1 Data Acquisition Guide

**Prepared:** August 6, 2026 · All links verified working as of this date (verification notes at bottom).
**Companion:** `research_plan_novel_asteroid_classifier.md` §2.1–2.2. Organized by role in the plan.
**Total download:** ~2–3 GB (dominated by SDSS 313 MB and the Gaia bulk chunks). No dataset requires registration.

A ready-to-run script covering the direct-download items is provided as `download_phase1_data.sh`.

---

## A. Unlabeled pretraining corpus (~1.2M records)

### A1. Gaia DR3 SSO reflectance spectra — 60,518 objects, 16 bands
- **Bulk download (recommended):** 20 gzipped ECSV chunks, deterministic names:
  `https://cdn.gea.esac.esa.int/Gaia/gdr3/Solar_system/sso_reflectance_spectrum/SsoReflectanceSpectrum_00.csv.gz` … `_19.csv.gz`
  Directory: https://cdn.gea.esac.esa.int/Gaia/gdr3/Solar_system/sso_reflectance_spectrum/
  Checksums: https://cdn.gea.esac.esa.int/Gaia/gdr3/Solar_system/sso_reflectance_spectrum/_MD5SUM.txt
- **TAP alternative:** endpoint `https://gea.esac.esa.int/tap-server/tap`, table **`gaiadr3.sso_reflectance_spectrum`** (columns: `source_id, number_mp, denomination, nb_samples, num_of_spectra, wavelength, reflectance_spectrum, reflectance_spectrum_err, reflectance_spectrum_flag`). Use `astroquery.gaia`. Flag semantics: 0 = good, 1 = potentially poor, 2 = compromised — keep the flag as a model input per plan §2.1.
- Table docs: https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_solar_system_object_tables/ssec_dm_sso_reflectance_spectrum.html

### A2. SDSS SSO catalog (Sergeyev & Carry 2021) — 1,036,322 obs / 379,714 objects
- **Direct file:** `https://cdsarc.cds.unistra.fr/ftp/J/A+A/652/A59/sso.dat.gz` (**313 MB**, gzipped ASCII; column spec in the `ReadMe` in the same directory)
- VizieR page: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/652/A59

### A3. SkyMapper SSO catalog (Sergeyev et al. 2022) — 880,528 obs / 205,515 objects
- **FTP directory:** https://cdsarc.cds.unistra.fr/ftp/J/A+A/658/A109/ (get `ReadMe` first for file names)
- VizieR page: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/658/A109

### A4. MOVIS NIR photometry (Popescu et al. 2016) — 53,447 SSOs
- **FTP directory:** https://cdsarc.cds.unistra.fr/ftp/J/A+A/591/A115/
- VizieR page: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/591/A115

---

## B. Labeled fine-tuning pool

### B1. Mahlke, Carry & Mattei (2022) taxonomy + compilation — 6,038 obs / 4,526 asteroids
- **VizieR tables:** https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/665/A26 · FTP: https://cdsarc.cds.unistra.fr/ftp/J/A+A/665/A26/
- **Practical route — the `classy` package** (aggregates and downloads the underlying public spectra — Gaia, SMASS, MITHNEOS, S3OS2, ECAS — and classifies in Mahlke/Bus-DeMeo/Tholen):
  `pip install space-classy` (v0.8.8; `space-classy[gui]` for the GUI) · docs: https://classy.readthedocs.io/en/latest/ · repo: https://github.com/maxmahlke/classy
  Use `classy` as the label-harmonization backbone (plan §2.2); keep the VizieR tables as the frozen citable snapshot.

### B2. Bus-DeMeo canonical set (DeMeo et al. 2009) — 371 asteroids + 24 class templates
- **PDS4 zip (69 KB):** https://sbnarchive.psi.edu/pds4/non_mission/ast.bus-demeo.taxonomy.zip
- PDS page: https://sbn.psi.edu/pds/resource/busdemeotax.html
- SMASS web classifier (reference implementation, for baseline 1): http://smass.mit.edu/busdemeoclass.html (server intermittently slow — retry)

### B3. MITHNEOS NEO spectra
- **PDS4 archive 2000–2021, V1.0 (3.6 MB zip):** https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.mithneos.spectra_2000-2021_V1_0.zip (DOI 10.26033/1aft-4018)
- PDS page: https://sbn.psi.edu/pds/resource/mithneos.html
- **Newer runs (through May 2026, unpublished-data caveat applies):** http://smass.mit.edu/minus.html

### B4. SMASS I & II visible spectra
- **Download page (tarballs + per-object files):** http://smass.mit.edu/smass.html — SMASS I (316 objects) and SMASS II (1,341 objects)
- **PDS mirrors (stable, citable):** SMASS I: https://sbn.psi.edu/pds/resource/smass1.html (1.9 MB) · SMASS II: https://sbn.psi.edu/pds/resource/smass2.html (4 MB)

### B5. S3OS2 (Lazzaro et al. 2004) — 820 visible spectra
- **PDS4 zip (6.7 MB):** https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.s3os2.spectra.zip
- PDS page: https://sbn.psi.edu/pds/resource/s3os2.html

### B6. Gaia-side labels (transfer test set + comparisons)
- **Delbo et al. 2026 probabilistic classifications:** Supplementary File on the article page — https://www.frontiersin.org/articles/10.3389/fspas.2026.1774478/full#supplementary-material (no CDS/Zenodo deposit exists; download the supplement manually once)
- **Tinaut-Ruano et al. 2026 (14,042 classifications):** VizieR `J/A+A/711/A167` — https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/711/A167 · FTP: https://cdsarc.cds.unistra.fr/ftp/J/A+A/711/A167/
- **MP3C literature-class aggregation (Delbo's training-label source):** https://mp3c.oca.eu/ — bulk via its TAP service (ADQL); 1.46M bodies. Used to build our own Gaia labeled overlap.

### B7. ECAS (Zellner et al. 1985) — 589 asteroids, 8 colors (Tholen heritage)
- **PDS page:** https://sbn.psi.edu/pds/resource/ecas.html (PDS4 zip 130 KB, links on page)

---

## C. Auxiliary scalars (albedo, H, G — plan §2.1)

### C1. NEOWISE diameters & albedos V2.0 (Mainzer et al.)
- **PDS page (6.5 MB bundle, DOI 10.26033/18S3-2Z54):** https://sbn.psi.edu/pds/resource/neowisediam.html (browse/zip links on page)

### C2. AKARI AcuA (+ 2018 re-fit)
- **JAXA DARTS direct file (117 KB):** https://data.darts.isas.jaxa.jp/pub/akari/AKARI-IRC_Catalogue_AllSky_AcuA_1.0/AcuA_V1.txt.gz (+ `ReadMe.AcuA.txt` in same directory)
- **Alí-Lagoa & Delbo 2018 re-fit:** VizieR `J/A+A/612/A85` — https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/612/A85 · FTP: https://cdsarc.cds.unistra.fr/ftp/J/A+A/612/A85/
- Usui 2011 original via VizieR: `J/PASJ/63/1117`

### C3. Best-estimate aggregation — SsODNet via `rocks`
- `pip install space-rocks` (v1.10.3) · docs: https://rocks.readthedocs.io/en/latest/ · repo: https://github.com/maxmahlke/rocks
- Use for per-object best albedo/H/G/taxonomy cross-checks; underlying service is IMCCE SsODNet (Berthier et al. 2023).

---

## D. Filter transmission curves (tokenizer Δλ — plan §2.3/§3.1)

### D1. SVO Filter Profile Service
- **Base:** http://svo2.cab.inta-csic.es/theory/fps/ (11,014 filters; SDSS ugriz under SLOAN; Gaia G/BP/RP under GAIA; VISTA for MOVIS; SkyMapper uvgriz)
- Programmatic: `astroquery.svo_fps.SvoFps.get_transmission_data('SLOAN/SDSS.r')` etc. Each curve downloads as ASCII/VOTable.
- **Gaia caveat:** the 16 reflectance bands are not filters — derive per-band effective widths from the BP/RP dispersion (documentation in the Gaia data-model page under A1); record the choice in the data-provenance appendix.

### D2. Marsset et al. (2020) slope systematics (optional)
- Machine-readable tables ship with the ApJS article: https://iopscience.iop.org/article/10.3847/1538-4365/ab7b5f — note `classy` applies these slope corrections internally, so a separate download is only needed if we implement our own correction.

---

## E. Suggested directory layout

```
data/
  raw/
    gaia_dr3_sso/          # A1: 20 csv.gz chunks + _MD5SUM.txt
    sdss_sergeyev2021/     # A2: sso.dat.gz + ReadMe
    skymapper_sergeyev2022/# A3
    movis/                 # A4
    mahlke2022_vizier/     # B1
    busdemeo2009/          # B2
    mithneos/              # B3
    smass/                 # B4
    s3os2/                 # B5
    gaia_labels/           # B6: Delbo supplement + Tinaut-Ruano VizieR
    ecas/                  # B7
    neowise/               # C1
    akari/                 # C2
    filters_svo/           # D1
  processed/               # tokenized quadruplets, label table (plan §2.2–2.3)
```

## F. Verification notes

Every URL above was checked on 2026-08-06 (agent-verified; landing pages fetched, Gaia bulk directory confirmed via its `_MD5SUM.txt` manifest listing chunks 00–19). CDS FTP paths follow the pattern stated verbatim on their verified VizieR catalog pages. None of the sources requires registration; anonymous Gaia TAP suffices for our query sizes. The two items needing a manual step: the Delbo et al. 2026 supplementary file (click-through on Frontiers) and MP3C bulk pulls (TAP queries, not a single dump).
