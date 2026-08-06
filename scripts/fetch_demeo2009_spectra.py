"""Fetch joined Vis+NIR spectra for the DeMeo 2009 object set (R1 inputs).

The 371 DeMeo 2009 input spectra are not archived as a PDS bundle; the
joined Vis+NIR products are served per-object from smass.mit.edu under
/data/spex/<run>/a<number>.<run>.txt (runs 'spNN' and 'dmNN'). This
script uses the SMASS spectral-library dump (speclib) to find every
full-range (0.45–2.45 um) file for a DeMeo 2009 object and mirrors it
into data/external/smass_demeo2009/.

Files already present in the local classy cache are copied instead of
re-downloaded. Re-runnable; skips existing files.

Usage: .venv/bin/python scripts/fetch_demeo2009_spectra.py
"""

from __future__ import annotations

import re
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/external/smass_demeo2009"
SPECLIB_URL = "http://smass.mit.edu/data/spex/dm09/speclib-edit-backup.txt"
BASE = "http://smass.mit.edu/data/spex"
CLASSY_CACHE = Path.home() / "Library/Caches/classy/mithneos"
DEMEOTAX = ROOT / "data/raw/busdemeo2009/ast.bus-demeo.taxonomy/data/demeotax.tab"


def demeo_numbers() -> set[str]:
    nums = set()
    for line in open(DEMEOTAX):
        if line.strip():
            n = line.split()[0]
            if n != "0":
                nums.add(n)
    return nums


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    speclib = DEST / "speclib-edit-backup.txt"
    if not speclib.exists():
        urllib.request.urlretrieve(SPECLIB_URL, speclib)

    targets = demeo_numbers()
    rows = []
    for line in open(speclib):
        if line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        _, number, name, kind, run, pub, date, subdir, fname = parts[:9]
        if number in targets and re.match(r"spex/(sp|dm)\d+/$", subdir):
            rows.append((number, subdir, fname))

    print(f"{len(rows)} candidate spex files for {len(targets)} numbered DeMeo objects")
    n_copied = n_downloaded = n_failed = 0
    for number, subdir, fname in rows:
        out = DEST / fname
        if out.exists():
            continue
        cached = CLASSY_CACHE / subdir.replace("spex/", "") / fname
        if cached.exists():
            shutil.copy(cached, out)
            n_copied += 1
            continue
        url = f"{BASE}/{subdir.replace('spex/', '')}{fname}"
        try:
            urllib.request.urlretrieve(url, out)
            n_downloaded += 1
            time.sleep(0.2)
        except Exception as exc:
            print(f"  FAILED {url}: {exc}")
            n_failed += 1
    print(f"copied {n_copied} from classy cache, downloaded {n_downloaded}, failed {n_failed}")


if __name__ == "__main__":
    main()
