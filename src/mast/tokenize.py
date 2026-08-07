"""Compile observation records into the dense token cache the trainer
consumes (plan §2.3/§3.1; storage design per the P2 plan).

Per source, arrays of shape [N, L_src, ...] with a validity mask — the
whole cache (~0.5 GB) is loaded to the training device once.

Token layout per record:
  - spectral tokens (type 0): one per present band — value = relative
    reflectance, sigma; positional (lambda_eff, delta_lambda) from
    mast.filters; per-band quality flag.
  - albedo token (type 1): (pV, e_pV) from mast.scalars when the object
    is numbered and covered (absent otherwise — never imputed).
  - H token (type 2): (H, 0.2) — sigma_H fixed at 0.2 mag (documented;
    source catalogs give no H uncertainty).
  - pivot-offset token (type 3): value log(lambda_pivot / 550 nm),
    sigma 0 — records the normalization frame (§2.3). Omitted for Gaia
    (already normalized at 550 nm; identity pivot).

Gaia records use the 12 interior bands (spec §2.1 band-drop policy);
per-band flags carried. Instrument ids: gaia 0, sdss 1, skymapper 2,
movis 3, spex 4, ground-vis 5.

Output: data/processed/corpus/tokens_v1/<source>.npz +
record_index.parquet (row, source, object_id, epoch) + meta.json
(build config + source parquet SHA256s).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mast import gaia as gaia_mod
from mast import scalars as scalars_mod
from mast.filters import registry

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data/processed/corpus"
TOKENS_DIR = CORPUS_DIR / "tokens_v1"

INSTRUMENT_IDS = {"gaia": 0, "sdss": 1, "skymapper": 2, "movis": 3,
                  "spex": 4, "groundvis": 5}
TYPE_SPECTRAL, TYPE_ALBEDO, TYPE_H, TYPE_PIVOT = 0, 1, 2, 3
SIGMA_H = 0.2

SOURCE_BANDS = {
    "sdss": ["u", "g", "r", "i", "z"],
    "skymapper": ["u", "v", "g", "r", "i", "z"],
    "movis": ["Y", "J", "H", "Ks"],
}
# Gaia: interior 12 of the 16 bands (drop 1-2, 15-16)
GAIA_KEEP = [int(w) for w in gaia_mod.GAIA_WAVELENGTHS[2:14]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _alloc(n: int, length: int):
    return {
        "values": np.zeros((n, length, 2), dtype=np.float32),
        "lam": np.zeros((n, length), dtype=np.float32),
        "dlam": np.zeros((n, length), dtype=np.float32),
        "token_type": np.zeros((n, length), dtype=np.int8),
        "flags": np.zeros((n, length), dtype=np.int8),
        "valid": np.zeros((n, length), dtype=bool),
    }


def _add_scalar_tokens(arr, i, j, number, scalar_map):
    if number is None or number < 0 or number not in scalar_map:
        return j
    pv, e_pv, h = scalar_map[number]
    if np.isfinite(pv):
        arr["values"][i, j] = (pv, e_pv)
        arr["token_type"][i, j] = TYPE_ALBEDO
        arr["valid"][i, j] = True
        j += 1
    if np.isfinite(h):
        arr["values"][i, j] = (h, SIGMA_H)
        arr["token_type"][i, j] = TYPE_H
        arr["valid"][i, j] = True
        j += 1
    return j


def compile_photometry(source: str, records: pd.DataFrame, scalar_map) -> dict:
    reg = registry()
    bands = SOURCE_BANDS[source]
    length = len(bands) + 3  # + albedo + H + pivot
    arr = _alloc(len(records), length)
    lam = {b: reg.lambda_eff(source, b) for b in bands}
    dlam = {b: reg.delta_lambda(source, b) for b in bands}
    numbers = records.number.to_numpy()
    pivots = records.pivot_band.to_numpy()
    R = {b: records[f"R_{b}"].to_numpy(np.float32) for b in bands}
    E = {b: records[f"e_{b}"].to_numpy(np.float32) for b in bands}
    F = {b: (records[f"flag_{b}"].to_numpy() if f"flag_{b}" in records
             else np.zeros(len(records), dtype=np.int8)) for b in bands}
    for i in range(len(records)):
        j = 0
        for b in bands:
            if np.isfinite(R[b][i]) and np.isfinite(E[b][i]):
                arr["values"][i, j] = (R[b][i], E[b][i])
                arr["lam"][i, j] = lam[b]
                arr["dlam"][i, j] = dlam[b]
                arr["flags"][i, j] = min(int(F[b][i]), 7)
                arr["token_type"][i, j] = TYPE_SPECTRAL
                arr["valid"][i, j] = True
                j += 1
        arr["values"][i, j] = (np.log(lam[pivots[i]] / 550.0), 0.0)
        arr["token_type"][i, j] = TYPE_PIVOT
        arr["valid"][i, j] = True
        j += 1
        j = _add_scalar_tokens(arr, i, j, int(numbers[i]), scalar_map)
    return arr


def compile_gaia(wide: pd.DataFrame, scalar_map) -> dict:
    reg = registry()
    length = len(GAIA_KEEP) + 2  # + albedo + H (no pivot token)
    arr = _alloc(len(wide), length)
    lam = {w: reg.lambda_eff("gaia", str(w)) for w in GAIA_KEEP}
    dlam = {w: reg.delta_lambda("gaia", str(w)) for w in GAIA_KEEP}
    refl = {w: wide[f"refl_{w}"].to_numpy(np.float32) for w in GAIA_KEEP}
    err = {w: wide[f"err_{w}"].to_numpy(np.float32) for w in GAIA_KEEP}
    flg = {w: wide[f"flag_{w}"].to_numpy() for w in GAIA_KEEP}
    numbers = wide.number_mp.fillna(-1).astype(int).to_numpy()
    for i in range(len(wide)):
        j = 0
        for w in GAIA_KEEP:
            r, e = refl[w][i], err[w][i]
            if np.isfinite(r) and np.isfinite(e) and e > 0:
                arr["values"][i, j] = (r, e)
                arr["lam"][i, j] = lam[w]
                arr["dlam"][i, j] = dlam[w]
                arr["flags"][i, j] = min(int(flg[w][i]) if not pd.isna(flg[w][i]) else 0, 7)
                arr["token_type"][i, j] = TYPE_SPECTRAL
                arr["valid"][i, j] = True
                j += 1
        j = _add_scalar_tokens(arr, i, j, int(numbers[i]), scalar_map)
    return arr


def build(out_dir: Path = TOKENS_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scal = scalars_mod.build()
    scalar_map = {int(r.number): (r.pV, r.e_pV, r.H)
                  for r in scal.itertuples(index=False)}

    index_rows = []
    meta = {"sources": {}, "sigma_H": SIGMA_H, "gaia_bands_kept": GAIA_KEEP}

    # Gaia
    wide = gaia_mod.load_wide(cache=ROOT / "data/processed/gaia_wide.parquet")
    arr = compile_gaia(wide, scalar_map)
    np.savez_compressed(out_dir / "gaia.npz", **arr,
                        instrument=np.full(len(wide), INSTRUMENT_IDS["gaia"], dtype=np.int8))
    for i, rec in enumerate(wide.itertuples(index=False)):
        oid = str(int(rec.number_mp)) if pd.notna(rec.number_mp) else str(rec.denomination)
        index_rows.append({"source": "gaia", "row": i, "object_id": oid,
                           "epoch_jd": np.nan, "snr": rec.snr})
    meta["sources"]["gaia"] = {"n": len(wide)}

    for source in ["sdss", "skymapper", "movis"]:
        path = CORPUS_DIR / f"{source}_records.parquet"
        records = pd.read_parquet(path)
        arr = compile_photometry(source, records, scalar_map)
        np.savez_compressed(
            out_dir / f"{source}.npz", **arr,
            instrument=np.full(len(records), INSTRUMENT_IDS[source], dtype=np.int8),
        )
        for i, (oid, epoch) in enumerate(zip(records.object_id, records.epoch_jd)):
            index_rows.append({"source": source, "row": i, "object_id": str(oid),
                               "epoch_jd": epoch, "snr": np.nan})
        meta["sources"][source] = {"n": len(records), "parquet_sha256": _sha256(path)}

    index = pd.DataFrame(index_rows)
    index.to_parquet(out_dir / "record_index.parquet", index=False)
    meta["n_records_total"] = len(index)
    meta["n_objects_total"] = index.object_id.nunique()
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
