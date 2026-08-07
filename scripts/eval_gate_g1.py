"""Gate G1 (plan §8, P2 exit): linear probe on the frozen pretrained
encoder vs the same classifier on PCA features.

Pre-registered comparison:
  - classifier: multinomial logistic regression (StandardScaler + LR),
    identical for both arms;
  - folds: frozen b1_folds_k10_seed42 (object-level); disputed labels
    excluded from training folds, kept in test (plan §2.2);
  - metric: type-level macro-F1 (canonical Bus-DeMeo classes), plus
    balanced accuracy; 5 classifier seeds, mean +/- std;
  - Arm A: pooled MAST embedding (frozen encoder, best checkpoint);
  - Arm B: top-5 PCs + slope (R3-certified feature set), PCA fit inside
    training folds only;
  - PASS: mean(A) - mean(B) >= +3 macro-F1.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/eval_gate_g1.py \
      --checkpoint logs/pretrain/final/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from mast import preprocessing as pp
from mast import probe as probe_mod
from mast import visnir_dataset as V
from mast.model import MAST, MASTConfig
from mast.train_pretrain import get_device

SEEDS = [0, 1, 2, 3, 4]
GATE_MARGIN = 3.0


def pca_slope_features() -> np.ndarray:
    """Raw inputs for Arm B; PCA itself is fit per training fold."""
    data = probe_mod._tokenize_probe_set()
    df = V.build()
    import pandas as pd
    folds = pd.read_csv(ROOT / "data/processed/splits/b1_folds_k10_seed42.csv",
                        dtype={"object_id": str})
    df = df.merge(folds[["object_id", "fold"]], on="object_id", how="inner")
    from mast.labels import canonical_bdm
    df["label"] = df.bdm_class.map(canonical_bdm)
    df = df[df.label != ""].reset_index(drop=True)
    _, x41, gamma = V.matrices(df)
    return x41[:, pp.PCA_CHANNELS], gamma, df


def arm_b_scores(seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    x40, gamma, df = pca_slope_features()
    y = df.label.to_numpy()
    folds = df.fold.to_numpy()
    disputed = df.disputed.to_numpy()
    preds = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        test = folds == f
        train = ~test & ~disputed
        mean = x40[train].mean(axis=0)
        _, _, vt = np.linalg.svd(x40[train] - mean, full_matrices=False)
        feats = np.column_stack([(x40 - mean) @ vt[:5].T, gamma])
        scaler = StandardScaler().fit(feats[train])
        clf = LogisticRegression(max_iter=3000, random_state=seed)
        clf.fit(scaler.transform(feats[train]), y[train])
        preds[test] = clf.predict(scaler.transform(feats[test]))
    return {
        "balanced_acc": float(balanced_accuracy_score(y, preds.astype(str)) * 100),
        "macro_f1": float(f1_score(y, preds.astype(str), average="macro") * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    device = get_device()

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MAST(MASTConfig(**{
        k: v for k, v in state["cfg"]["model"].items()
        if k in MASTConfig.__dataclass_fields__
    })).to(device)
    model.load_state_dict(state["model"])
    emb = probe_mod.encode_probe_set(model, device)

    arm_a = [probe_mod.probe_features(emb, seed=s) for s in SEEDS]
    arm_b = [arm_b_scores(seed=s) for s in SEEDS]

    def stats(runs, key):
        vals = [r[key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    a_f1, a_f1_std = stats(arm_a, "macro_f1")
    b_f1, b_f1_std = stats(arm_b, "macro_f1")
    a_ba, a_ba_std = stats(arm_a, "balanced_acc")
    b_ba, b_ba_std = stats(arm_b, "balanced_acc")
    delta = a_f1 - b_f1
    verdict = {
        "checkpoint": args.checkpoint,
        "arm_a_macro_f1": [a_f1, a_f1_std], "arm_b_macro_f1": [b_f1, b_f1_std],
        "arm_a_balanced_acc": [a_ba, a_ba_std], "arm_b_balanced_acc": [b_ba, b_ba_std],
        "delta_macro_f1": delta,
        "gate_margin": GATE_MARGIN,
        "pass": bool(delta >= GATE_MARGIN),
    }
    out = ROOT / "logs/gate_g1.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"Arm A (MAST probe):  macro-F1 {a_f1:.1f} ± {a_f1_std:.1f}  "
          f"bal-acc {a_ba:.1f} ± {a_ba_std:.1f}")
    print(f"Arm B (PCA+slope):   macro-F1 {b_f1:.1f} ± {b_f1_std:.1f}  "
          f"bal-acc {b_ba:.1f} ± {b_ba_std:.1f}")
    print(f"Δ macro-F1 = {delta:+.1f} (gate: ≥ +{GATE_MARGIN})")
    print(f"GATE G1: {'PASS' if verdict['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
