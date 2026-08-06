"""Object-level splits with hashed, frozen manifests (plan §2.4).

Every split in the project is object-level: all observations of one
asteroid live on one side of every split. Folds are stratified by a
label column and fully deterministic given (objects, labels, k, seed):
within each stratum, objects are ordered by sha256(f"{seed}:{object_id}")
and dealt round-robin to folds. This avoids library-version dependence
(sklearn shuffling) and handles rare classes (a stratum with fewer than
k members still spreads across distinct folds).

Manifests are written as CSV plus a JSON sidecar carrying the
parameters and the manifest's own sha256; hashes are also appended to
data/processed/splits/SHA256SUMS. Once frozen at P1 exit, manifests are
never regenerated (re-running verifies byte-identity instead).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data/processed/splits"


def _order_key(seed: int, object_id: str) -> str:
    return hashlib.sha256(f"{seed}:{object_id}".encode()).hexdigest()


def make_object_folds(
    objects: pd.DataFrame,
    k: int,
    seed: int,
    stratify_col: str,
    id_col: str = "object_id",
) -> pd.DataFrame:
    """Deterministic stratified object-level k-fold assignment.

    objects: one row per object; duplicated ids raise. Returns a
    DataFrame (id_col, stratum, fold) sorted by id.
    """
    if objects[id_col].duplicated().any():
        raise ValueError("duplicate object ids — splits must be object-level")
    df = objects[[id_col, stratify_col]].copy()
    df[id_col] = df[id_col].astype(str)
    df["_key"] = df[id_col].map(lambda o: _order_key(seed, o))

    # Strata are processed in a fixed order; fold numbering continues
    # across strata so small strata do not all pile into fold 0.
    df = df.sort_values([stratify_col, "_key"], kind="mergesort")
    df["fold"] = -1
    offset = 0
    for _, group in df.groupby(stratify_col, sort=True):
        idx = group.index
        df.loc[idx, "fold"] = [(offset + i) % k for i in range(len(idx))]
        offset += len(idx)
    out = df.rename(columns={stratify_col: "stratum"})[[id_col, "stratum", "fold"]]
    return out.sort_values(id_col, kind="mergesort").reset_index(drop=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(
    folds: pd.DataFrame,
    name: str,
    meta: dict,
    splits_dir: Path = SPLITS_DIR,
) -> dict:
    """Write <name>.csv + <name>.json; append hash to SHA256SUMS.

    If the manifest already exists, verify byte-identity instead of
    overwriting (frozen-manifest rule) — raises on mismatch.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    csv_path = splits_dir / f"{name}.csv"
    json_path = splits_dir / f"{name}.json"
    payload = folds.to_csv(index=False).encode()
    digest = _sha256_bytes(payload)

    if csv_path.exists():
        existing = csv_path.read_bytes()
        if _sha256_bytes(existing) != digest:
            raise RuntimeError(
                f"{csv_path} exists with different content — frozen manifests "
                "are never regenerated (delete manually only with explicit "
                "project-lead approval)."
            )
        return json.loads(json_path.read_text())

    csv_path.write_bytes(payload)
    record = {
        "name": name,
        "sha256": digest,
        "n_objects": int(len(folds)),
        "n_folds": int(folds.fold.nunique()),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        **meta,
    }
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    with open(splits_dir / "SHA256SUMS", "a") as f:
        f.write(f"{digest}  {name}.csv\n")
    return record


def verify_manifest(name: str, splits_dir: Path = SPLITS_DIR) -> bool:
    """Check a manifest's CSV against the hash recorded in its sidecar."""
    csv_path = splits_dir / f"{name}.csv"
    meta = json.loads((splits_dir / f"{name}.json").read_text())
    return _sha256_bytes(csv_path.read_bytes()) == meta["sha256"]
