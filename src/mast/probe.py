"""Linear probe of a (frozen) MAST encoder on the Phase-1 labeled set
(P2 plan W5; HPO proxy metric and Gate-G1 arm A).

The 476-object VIS+NIR dataset (mast.visnir_dataset) is tokenized to
match the pretraining distribution: the 200-channel normalized spectrum
(slope retained, unity at 0.55 um) is averaged into 8-channel patches
-> 25 spectral tokens (value = patch mean, lambda_eff = patch center,
delta_lambda = patch span, sigma = per-spectrum noise estimate), plus
albedo/H scalar tokens where the object is numbered and covered
(instrument id 'groundvis'; identity pivot, no pivot token — like
Gaia).

Probe = multinomial logistic regression on standardized pooled
embeddings, evaluated over the frozen b1_folds_k10_seed42 manifest
(object-level; disputed objects excluded from training folds, kept in
test — plan §2.2). Metrics: balanced accuracy and type-level macro-F1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mast import preprocessing as pp
from mast import scalars as scalars_mod
from mast import visnir_dataset as V
from mast.tokenize import INSTRUMENT_IDS, SIGMA_H, TYPE_ALBEDO, TYPE_H
from mast.labels import canonical_bdm

ROOT = Path(__file__).resolve().parents[2]
PATCH = 8
GRID200_NM = np.linspace(450.0, 2450.0, 200)

_cache: dict | None = None


def _tokenize_probe_set() -> dict:
    """Token arrays + labels/folds for the labeled VIS+NIR pool."""
    global _cache
    if _cache is not None:
        return _cache
    df = V.build()
    folds = pd.read_csv(ROOT / "data/processed/splits/b1_folds_k10_seed42.csv",
                        dtype={"object_id": str})
    df = df.merge(folds[["object_id", "fold"]], on="object_id", how="inner")
    df["label"] = df.bdm_class.map(canonical_bdm)
    df = df[df.label != ""].reset_index(drop=True)

    scal = scalars_mod.build()
    scalar_map = {int(r.number): (r.pV, r.e_pV, r.H)
                  for r in scal.itertuples(index=False)}

    x200, _, _ = V.matrices(df)
    n_patches = 200 // PATCH
    n, length = len(df), n_patches + 2
    arr = {
        "values": np.zeros((n, length, 2), dtype=np.float32),
        "lam": np.zeros((n, length), dtype=np.float32),
        "dlam": np.zeros((n, length), dtype=np.float32),
        "token_type": np.zeros((n, length), dtype=np.int64),
        "flags": np.zeros((n, length), dtype=np.int64),
        "valid": np.zeros((n, length), dtype=bool),
    }
    patch_vals = x200[:, : n_patches * PATCH].reshape(n, n_patches, PATCH)
    centers = GRID200_NM[: n_patches * PATCH].reshape(n_patches, PATCH).mean(1)
    span = float(GRID200_NM[PATCH - 1] - GRID200_NM[0])
    for i in range(n):
        sigma = max(pp.estimate_noise_sigma(x200[i]), 1e-3)
        arr["values"][i, :n_patches, 0] = patch_vals[i].mean(1)
        arr["values"][i, :n_patches, 1] = sigma
        arr["lam"][i, :n_patches] = centers
        arr["dlam"][i, :n_patches] = span
        arr["valid"][i, :n_patches] = True
        j = n_patches
        oid = df.object_id.iloc[i]
        number = int(oid) if oid.isdigit() else -1
        if number in scalar_map:
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
    _cache = {
        "arrays": arr,
        "labels": df.label.to_numpy(),
        "folds": df.fold.to_numpy(),
        "disputed": df.disputed.to_numpy(),
    }
    return _cache


def encode_probe_set(model, device: str) -> np.ndarray:
    data = _tokenize_probe_set()
    arr = data["arrays"]
    batch = {
        "values": torch.from_numpy(arr["values"]).to(device),
        "lam": torch.from_numpy(arr["lam"]).to(device),
        "dlam": torch.from_numpy(arr["dlam"]).to(device),
        "token_type": torch.from_numpy(arr["token_type"]).to(device),
        "flags": torch.from_numpy(arr["flags"]).to(device),
        "valid": torch.from_numpy(arr["valid"]).to(device),
        "instrument": torch.full((len(arr["lam"]),), INSTRUMENT_IDS["groundvis"],
                                 dtype=torch.long, device=device),
    }
    model.eval()
    with torch.no_grad():
        emb = model.encode(batch).float().cpu().numpy()
    model.train()
    return emb


def probe_features(features: np.ndarray, seed: int = 0) -> dict:
    """CV logistic regression on given per-object features (frozen folds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    data = _tokenize_probe_set()
    y, folds, disputed = data["labels"], data["folds"], data["disputed"]
    preds = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        test = folds == f
        train = ~test & ~disputed
        scaler = StandardScaler().fit(features[train])
        clf = LogisticRegression(max_iter=3000, random_state=seed)
        clf.fit(scaler.transform(features[train]), y[train])
        preds[test] = clf.predict(scaler.transform(features[test]))
    return {
        "balanced_acc": float(balanced_accuracy_score(y, preds.astype(str)) * 100),
        "macro_f1": float(f1_score(y, preds.astype(str), average="macro") * 100),
    }


def linear_probe(model, device: str, seed: int = 0) -> dict:
    return probe_features(encode_probe_set(model, device), seed=seed)
