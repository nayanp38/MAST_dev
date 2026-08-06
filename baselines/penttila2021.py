"""R2 — Penttilä et al. 2021 FFNN recreation (certifies labels + training loop).

Their setup: 586 VIS+NIR spectra (DeMeo 2009 + MITHNEOS), resampled to
200 points on 0.45–2.45 um, 11 collapsed Bus-DeMeo classes, single
hidden layer of 30 tanh units + softmax, 5-network voting ensemble,
leave-one-out CV -> 90.6% overall accuracy.

Documented deviations (phase1_recreation_plan.md allows the first two):
  - 10-fold object-level stratified CV instead of leave-one-out.
  - Single network instead of the 5-vote ensemble.
  - Dataset is the rebuildable 476-object set (368 DeMeo-2009 originals
    + 108 MITHNEOS additions with classy-tree labels) instead of their
    586 spectra — the extra MITHNEOS data they used is not fully
    publicly servable; see specs/discrepancy_log.md.
  - No PCA-based synthetic augmentation (their 200 samples/class); the
    effect on *overall* accuracy is small since it targets rare classes.

Pass: overall accuracy within 90.6 +/- 3 pts; S/Q and C/B/X confusions
dominant in the confusion matrix.

Two protocols are reported:
  - "penttila": all labels at face value (their convention; primary for
    the pass check).
  - "mast": disputed labels excluded from training folds but kept in
    test (our plan §2.2 convention).

Usage: PYTHONPATH=src .venv/bin/python baselines/penttila2021.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from mast import splits
from mast import visnir_dataset as V
from mast.labels import PENTTILA_11

SEEDS = [0, 1, 2, 3, 4]
K = 10
FOLD_SEED = 42
TARGET = 90.6
TOL = 3.0


def load_dataset() -> pd.DataFrame:
    df = V.build()
    df = df[df.bdm_class.map(PENTTILA_11).notna()].copy()
    df["label"] = df.bdm_class.map(PENTTILA_11)
    return df.reset_index(drop=True)


def get_folds(df: pd.DataFrame) -> pd.Series:
    folds = splits.make_object_folds(
        df.rename(columns={"label": "cls"}), k=K, seed=FOLD_SEED, stratify_col="cls"
    )
    splits.write_manifest(
        folds, name=f"r2_penttila_folds_k{K}_seed{FOLD_SEED}",
        meta={"seed": FOLD_SEED, "k": K, "stratify": "penttila_11",
              "pool": "visnir_dataset objects with a Penttilä-11 class"},
    )
    return df.merge(folds, on="object_id").fold


def run_cv(df: pd.DataFrame, protocol: str, seed: int):
    X, _, _ = V.matrices(df)
    y = df.label.to_numpy()
    fold = df.fold.to_numpy()
    disputed = df.disputed.to_numpy()

    y_true, y_pred = [], []
    for f in range(K):
        test = fold == f
        train = ~test
        if protocol == "mast":
            train = train & ~disputed
        scaler = StandardScaler().fit(X[train])
        clf = MLPClassifier(
            hidden_layer_sizes=(30,), activation="tanh", max_iter=3000,
            random_state=seed,
        )
        clf.fit(scaler.transform(X[train]), y[train])
        y_true.extend(y[test])
        y_pred.extend(clf.predict(scaler.transform(X[test])))
    return np.array(y_true), np.array(y_pred)


def confusion_report(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            if i < j:
                pairs.append((off[i, j] + off[j, i], f"{a}/{b}"))
    pairs.sort(reverse=True)
    return cm, pairs[:6]


def run() -> dict:
    df = load_dataset()
    df["fold"] = get_folds(df)
    classes = sorted(df.label.unique())
    results = {"n_objects": len(df), "classes": classes}

    for protocol in ["penttila", "mast"]:
        accs = []
        agg_true, agg_pred = [], []
        for seed in SEEDS:
            y_true, y_pred = run_cv(df, protocol, seed)
            accs.append(accuracy_score(y_true, y_pred) * 100)
            agg_true.extend(y_true)
            agg_pred.extend(y_pred)
        cm, top_pairs = confusion_report(agg_true, agg_pred, classes)
        results[protocol] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "top_confusions": top_pairs, "confusion_matrix": cm.tolist(),
        }
    prim = results["penttila"]
    results["pass_accuracy"] = abs(prim["acc_mean"] - TARGET) <= TOL
    top3 = {p for _, p in prim["top_confusions"][:3]}
    results["pass_confusions"] = bool(
        top3 & {"Q/S", "S/Q"} or any("S" in p and "Q" in p for p in top3)
    ) and any(set(p.split("/")) <= {"C", "B", "X"} for _, p in prim["top_confusions"])
    results["pass"] = results["pass_accuracy"] and results["pass_confusions"]
    return results


if __name__ == "__main__":
    res = run()
    print(f"R2 dataset: {res['n_objects']} objects, {len(res['classes'])} classes")
    for protocol in ["penttila", "mast"]:
        r = res[protocol]
        print(f"  [{protocol}] accuracy = {r['acc_mean']:.1f} ± {r['acc_std']:.1f} "
              f"(target {TARGET} ± {TOL})")
        print(f"    top confusions: "
              + ", ".join(f"{p} ({n})" for n, p in r["top_confusions"]))
    print(f"R2 PASS: {res['pass']} (accuracy: {res['pass_accuracy']}, "
          f"confusion structure: {res['pass_confusions']})")
