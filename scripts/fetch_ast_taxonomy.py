"""Fetch the PDS asteroid-taxonomy compilation (ast_taxonomy V1.1).

Contains per-object Tholen, Bus, S3OS2 (Lazzaro) and other literature
classifications — used in R4 as the literature-label pool analog of
Delbo's MP3C aggregation (the frozen label table v1 keeps only
Bus-DeMeo + Mahlke; this bundle is an R4 counting aid).

Usage: .venv/bin/python scripts/fetch_ast_taxonomy.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://sbnarchive.psi.edu/pds4/non_mission/ast_taxonomy_v1.1.zip"
DEST = Path(__file__).resolve().parents[1] / "data/external"


def main() -> None:
    out = DEST / "ast_taxonomy_v1.1"
    if out.exists():
        print(f"{out} already present")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} …")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (MAST data fetch)"})
    data = urllib.request.urlopen(req, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(DEST)
    print(f"extracted to {out}")


if __name__ == "__main__":
    main()
