"""P1-exit freeze: record sha256 of every frozen Phase-1 artifact.

Writes data/processed/P1_FREEZE.json. Once written, re-running verifies
hashes instead of overwriting (mismatch = frozen artifact was modified —
hard error).

Usage: .venv/bin/python scripts/freeze_p1.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "data/processed/P1_FREEZE.json"

ARTIFACTS = [
    "data/processed/label_table_v1.parquet",
    "data/processed/label_table_v1.csv",
    "data/processed/mithneos_demeo_per_spectrum.csv",
    "data/processed/visnir_dataset.parquet",
    "data/processed/r1_demeo_manifest.csv",
    "data/processed/splits/b1_folds_k10_seed42.csv",
    "data/processed/splits/r2_penttila_folds_k10_seed42.csv",
    "data/processed/splits/r3_klimczak_folds_k5_seed100.csv",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    hashes = {rel: sha256(ROOT / rel) for rel in ARTIFACTS}
    if FREEZE.exists():
        stored = json.loads(FREEZE.read_text())["artifacts"]
        bad = [rel for rel, h in hashes.items() if stored.get(rel) != h]
        if bad:
            sys.exit(f"FROZEN ARTIFACTS MODIFIED: {bad}")
        print(f"P1 freeze verified: {len(hashes)} artifacts unchanged")
        return
    FREEZE.write_text(json.dumps({
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "P1 exit freeze — these artifacts are never regenerated; "
                "every later experiment uses these exact splits/labels.",
        "artifacts": hashes,
    }, indent=2) + "\n")
    print(f"P1 freeze written: {len(hashes)} artifacts")
    for rel, h in hashes.items():
        print(f"  {h[:16]}…  {rel}")


if __name__ == "__main__":
    main()
