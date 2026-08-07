"""P2 freeze: record sha256 of the pretraining-corpus artifacts.

Analogous to scripts/freeze_p1.py; verifies instead of overwriting.
Model checkpoints are NOT frozen here (they carry their own config +
cache hashes); this freezes the data the runs consumed.

Usage: .venv/bin/python scripts/freeze_p2.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "data/processed/P2_FREEZE.json"

ARTIFACTS = [
    "data/external/filters_svo/filter_metadata.csv",
    "data/processed/corpus/scalars.parquet",
    "data/processed/corpus/sdss_records.parquet",
    "data/processed/corpus/skymapper_records.parquet",
    "data/processed/corpus/movis_records.parquet",
    "data/processed/corpus/tokens_v1/meta.json",
    "data/processed/corpus/tokens_v1/record_index.parquet",
    "data/processed/splits/corpus_holdout_k20_seed42.csv",
    "logs/optuna/stage1_proxy_formula.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    hashes = {rel: sha256(ROOT / rel) for rel in ARTIFACTS}
    if FREEZE.exists():
        stored = json.loads(FREEZE.read_text())["artifacts"]
        bad = [rel for rel, h in hashes.items() if stored.get(rel) != h]
        if bad:
            sys.exit(f"FROZEN P2 ARTIFACTS MODIFIED: {bad}")
        print(f"P2 freeze verified: {len(hashes)} artifacts unchanged")
        return
    FREEZE.write_text(json.dumps({
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "P2 pretraining-corpus freeze — every pretraining run and "
                "ablation uses these exact records/splits.",
        "artifacts": hashes,
    }, indent=2) + "\n")
    print(f"P2 freeze written: {len(hashes)} artifacts")


if __name__ == "__main__":
    main()
