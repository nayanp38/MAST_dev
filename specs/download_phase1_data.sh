#!/usr/bin/env bash
# Phase 1 data download — MAST project (v2, verified 2026-08-06)
# Downloads AND organizes every Phase-1 dataset into data/raw/<dataset>/.
# - PDS zip archives are unzipped in place (zip kept for provenance).
# - CDS/VizieR catalogs: fetches the ReadMe, parses its "File Summary" for data
#   files, and downloads each (tries gzipped variant first).
# - Gaia bulk chunks are checksum-verified (md5sum on Linux, md5 on macOS).
# NOT covered (see phase1_data_acquisition_guide.md):
#   * Delbo 2026 classifications — manual click-through on Frontiers supplement
#   * MP3C — TAP/ADQL queries, no file dump
#   * SVO filter curves — fetched programmatically in the pipeline (astroquery)
# Usage: bash download_phase1_data.sh [target_dir]   (default: ./data/raw)
set -euo pipefail

ROOT="${1:-./data/raw}"
mkdir -p "$ROOT"
CURL="curl -L --fail --retry 3 --retry-delay 5 -C -"

log () { printf '\n=== %s ===\n' "$*"; }

get () { # get <subdir> <url>  -> downloads into $ROOT/<subdir>/
  local dir="$ROOT/$1"; mkdir -p "$dir"
  echo ">>> $(basename "$2")"
  $CURL -o "$dir/$(basename "$2")" "$2"
}

get_zip () { # get_zip <subdir> <url> -> download + unzip in place
  get "$1" "$2"
  local dir="$ROOT/$1" f
  f="$dir/$(basename "$2")"
  unzip -o -q "$f" -d "$dir" && echo "    unzipped -> $dir/"
}

cds_fetch () { # cds_fetch <subdir> <catalog_path e.g. J/A+A/652/A59>
  local sub="$1" cat="$2" base dir names n
  base="https://cdsarc.cds.unistra.fr/ftp/$cat"
  dir="$ROOT/$sub"; mkdir -p "$dir"
  echo ">>> $cat ReadMe"
  $CURL -o "$dir/ReadMe" "$base/ReadMe"
  # Parse the File Summary table for data file names (.dat/.fits/.txt/.csv),
  # excluding the ReadMe itself. Handles both plain and pre-gzipped listings.
  names=$(awk '{print $1}' "$dir/ReadMe" \
          | grep -E '\.(dat|dat\.gz|fits|fit|txt|csv)$' \
          | grep -vi '^readme' | sort -u || true)
  if [ -z "$names" ]; then
    echo "    WARN: no data files parsed from ReadMe — inspect $dir/ReadMe and fetch from $base/ manually"
    return 0
  fi
  for n in $names; do
    echo ">>> $cat/$n"
    if [[ "$n" == *.gz ]]; then
      $CURL -o "$dir/$n" "$base/$n"
    else
      # CDS usually serves a gzipped variant; try it first, fall back to plain.
      $CURL -o "$dir/$n.gz" "$base/$n.gz" 2>/dev/null \
        || $CURL -o "$dir/$n" "$base/$n"
    fi
  done
}

# ---------------------------------------------------------------- A. Unlabeled corpus
log "A1. Gaia DR3 SSO reflectance spectra (20 chunks, ~60k objects)"
GAIA_BASE="https://cdn.gea.esac.esa.int/Gaia/gdr3/Solar_system/sso_reflectance_spectrum"
get gaia_dr3_sso "$GAIA_BASE/_MD5SUM.txt"
i=0
while [ $i -le 19 ]; do
  n=$(printf '%02d' $i)
  get gaia_dr3_sso "$GAIA_BASE/SsoReflectanceSpectrum_${n}.csv.gz"
  i=$((i+1))
done
log "A1. Verifying Gaia checksums"
(
  cd "$ROOT/gaia_dr3_sso"
  if command -v md5sum >/dev/null 2>&1; then
    grep 'SsoReflectanceSpectrum' _MD5SUM.txt | md5sum -c -
  else # macOS
    ok=1
    while read -r want file; do
      case "$file" in SsoReflectanceSpectrum*) ;; *) continue ;; esac
      have=$(md5 -q "$file")
      if [ "$want" = "$have" ]; then echo "$file: OK"; else echo "$file: MISMATCH"; ok=0; fi
    done < _MD5SUM.txt
    [ $ok -eq 1 ] || { echo "CHECKSUM FAILURE — re-run script (resume supported)"; exit 1; }
  fi
)

log "A2. SDSS SSO catalog — Sergeyev & Carry 2021 (~313 MB)"
cds_fetch sdss_sergeyev2021 "J/A+A/652/A59"

log "A3. SkyMapper SSO catalog — Sergeyev et al. 2022"
cds_fetch skymapper_sergeyev2022 "J/A+A/658/A109"

log "A4. MOVIS photometry (2016) + taxonomy (2018)"
cds_fetch movis_photometry "J/A+A/591/A115"
cds_fetch movis_taxonomy   "J/A+A/617/A12"

# ---------------------------------------------------------------- B. Labeled pool
log "B1. Mahlke, Carry & Mattei 2022 (VizieR snapshot)"
cds_fetch mahlke2022_vizier "J/A+A/665/A26"

log "B2. Bus-DeMeo 2009 taxonomy (371 objects + 24 templates)"
get_zip busdemeo2009 "https://sbnarchive.psi.edu/pds4/non_mission/ast.bus-demeo.taxonomy.zip"

log "B3. MITHNEOS spectra 2000-2021 V1.0"
get_zip mithneos "https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.mithneos.spectra_2000-2021_V1_0.zip"

log "B4. SMASS I & II (PDS mirrors)"
get_zip smass "https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.smass.spectra.zip"    # SMASS I
get_zip smass "https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.smass2.spectra.zip"   # SMASS II

log "B5. S3OS2 (820 visible spectra)"
get_zip s3os2 "https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.s3os2.spectra.zip"

log "B6. Tinaut-Ruano 2026 Gaia taxonomy (14,042 classifications)"
cds_fetch gaia_labels_tinautruano "J/A+A/711/A167"

log "B7. ECAS (Tholen-heritage 8-color photometry)"
get_zip ecas "https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.ecas.phot.zip"

# ---------------------------------------------------------------- C. Auxiliary scalars
log "C1. NEOWISE diameters & albedos V2.0"
get_zip neowise "https://sbnarchive.psi.edu/pds4/non_mission/neowise_diameters_albedos_V2_0.zip"

log "C2. AKARI AcuA (+ 2018 re-fit)"
get akari "https://data.darts.isas.jaxa.jp/pub/akari/AKARI-IRC_Catalogue_AllSky_AcuA_1.0/AcuA_V1.txt.gz"
get akari "https://data.darts.isas.jaxa.jp/pub/akari/AKARI-IRC_Catalogue_AllSky_AcuA_1.0/ReadMe.AcuA.txt"
cds_fetch akari_alilagoa2018 "J/A+A/612/A85"

# ---------------------------------------------------------------- Done
log "Download complete"
du -sh "$ROOT"/* 2>/dev/null || true
cat <<'EOF'

Remaining manual/programmatic items:
  1. Delbo 2026 Gaia classifications (Supplementary File, one click):
     https://www.frontiersin.org/articles/10.3389/fspas.2026.1774478/full#supplementary-material
     -> save into data/raw/gaia_labels_delbo/
  2. Python tools + filter curves:
     pip install space-classy space-rocks astroquery
     python -c "from astroquery.svo_fps import SvoFps; print(SvoFps.get_transmission_data('SLOAN/SDSS.r')[:3])"
  3. MP3C label pulls happen later via TAP (plan section 2.2).
EOF
