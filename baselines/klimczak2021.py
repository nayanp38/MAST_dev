"""R3 — Klimczak et al. 2021 metric-frame recreation (certifies the honest
metric stack + splits).

Their setup: 504 asteroids (classes with <10 members dropped -> 12
Bus-DeMeo types), features = top-5 PCs + slope from DeMeo-style
preprocessed spectra, models incl. XGBoost and MLP, 5-fold stratified CV
repeated 10x, balanced accuracy as headline metric. Best model:
76.8 balanced accuracy at type level, 90.0 at complex level.

Type scheme (merged, the only reading that yields 12 classes with >=10
members from DeMeo-era data): B; C(+Cb,Cg); Ch(+Cgh); D; K; L; Q;
S(+Sa,Sv); Sq; Sr; V; X(+Xc,Xe,Xk). A/T/O/R dropped. Complexes:
C = {B,C,Ch}, S = {S,Sq,Sr}, X = {X}, other = {D,K,L,Q,V}.

Pass: better-of-two-models balanced accuracy within 76.8 +/- 3 (types)
and 90.0 +/- 3 (complexes) on the primary pool.

Documented deviations:
  - Dataset is the rebuildable set; their exact 504-object list is not
    published.
  - Primary pool excludes disputed-label objects (plan §2.2): their
    dataset carried only curated-era labels, whereas our tier-2
    additions include classy-tree labels whose noise concentrates
    exactly in the C/X boundary; the all-objects numbers are reported
    alongside (and logged in specs/discrepancy_log.md).
  - Folds are object-level and deterministic via mast.splits;
    repetition = 10 fold seeds (100-109). PCA + scaler fit inside
    training folds only.

Usage: PYTHONPATH=src .venv/bin/python baselines/klimczak2021.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mast import preprocessing as pp
from mast import splits
from mast import visnir_dataset as V

MERGE12 = {
    "B": "B", "C": "C", "Cb": "C", "Cg": "C", "Ch": "Ch", "Cgh": "Ch",
    "D": "D", "K": "K", "L": "L", "Q": "Q",
    "S": "S", "Sa": "S", "Sv": "S", "Sq": "Sq", "Sr": "Sr", "V": "V",
    "X": "X", "Xc": "X", "Xe": "X", "Xk": "X",
}
COMPLEX4 = {
    "B": "C", "C": "C", "Ch": "C", "S": "S", "Sq": "S", "Sr": "S",
    "X": "X", "D": "other", "K": "other", "L": "other", "Q": "other",
    "V": "other",
}
K = 5
REPEAT_SEEDS = list(range(100, 110))
TARGETS = {"type": (76.8, 3.0), "complex": (90.0, 3.0)}


def load_dataset() -> pd.DataFrame:
    df = V.build()
    df = df[df.bdm_class.isin(MERGE12)].copy()
    df["type"] = df.bdm_class.map(MERGE12)
    df["complex"] = df["type"].map(COMPLEX4)
    return df.reset_index(drop=True)


def fold_features(df, train, n_pcs=5):
    """Top-5 PC scores (fit on train only) + slope."""
    _, x41, gamma = V.matrices(df)
    x40 = x41[:, pp.PCA_CHANNELS]
    mean = x40[train].mean(axis=0)
    _, _, vt = np.linalg.svd(x40[train] - mean, full_matrices=False)
    scores = (x40 - mean) @ vt[:n_pcs].T
    return np.column_stack([scores, gamma])


def models(seed):
    return {
        "XGBoost": XGBClassifier(random_state=seed, n_jobs=4,
                                 eval_metric="mlogloss"),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000,
                             random_state=seed),
    }


def _cv(df: pd.DataFrame, level: str) -> dict:
    df = df.reset_index(drop=True)
    y_all = df[level].to_numpy()
    codes = {c: i for i, c in enumerate(sorted(set(y_all)))}
    y = np.array([codes[v] for v in y_all])
    scores: dict[str, list[float]] = {m: [] for m in models(0)}
    for seed in REPEAT_SEEDS:
        folds = splits.make_object_folds(
            df.assign(cls=y_all), k=K, seed=seed, stratify_col="cls"
        )
        fold = df.merge(folds, on="object_id").fold.to_numpy()
        preds = {m: np.empty(len(df), dtype=int) for m in scores}
        for f in range(K):
            test = fold == f
            train = ~test
            X = fold_features(df, train)
            scaler = StandardScaler().fit(X[train])
            Xs = scaler.transform(X)
            for name, clf in models(seed).items():
                clf.fit(Xs[train], y[train])
                preds[name][test] = clf.predict(Xs[test])
        for name in scores:
            scores[name].append(balanced_accuracy_score(y, preds[name]) * 100)
    return {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for name, v in scores.items()}


def run() -> dict:
    full = load_dataset()
    primary = full[~full.disputed].reset_index(drop=True)
    results = {
        "n_primary": len(primary), "n_all": len(full),
        "type_counts": primary["type"].value_counts().to_dict(),
    }
    for pool_name, pool in [("primary", primary), ("all_objects", full)]:
        results[pool_name] = {level: _cv(pool, level) for level in ["type", "complex"]}

    passes = {}
    for level, (target, tol) in TARGETS.items():
        best = max(results["primary"][level].values(), key=lambda r: r["mean"])
        passes[level] = abs(best["mean"] - target) <= tol
        results["primary"][level]["best_mean"] = best["mean"]
    results["pass"] = all(passes.values())
    results["passes"] = passes

    folds = splits.make_object_folds(
        primary.assign(cls=primary["type"]), k=K, seed=REPEAT_SEEDS[0],
        stratify_col="cls",
    )
    splits.write_manifest(
        folds, name=f"r3_klimczak_folds_k{K}_seed{REPEAT_SEEDS[0]}",
        meta={"seed": REPEAT_SEEDS[0], "k": K, "stratify": "klimczak_type_12",
              "pool": "visnir_dataset, Klimczak 12 merged types, non-disputed",
              "note": "repetitions use seeds 100-109, regenerable "
                      "deterministically via mast.splits"},
    )
    return results


if __name__ == "__main__":
    res = run()
    print(f"R3 pools: primary n={res['n_primary']} (non-disputed), "
          f"all n={res['n_all']}")
    print("  type counts (primary):", res["type_counts"])
    for pool in ["primary", "all_objects"]:
        for level in ["type", "complex"]:
            t, tol = TARGETS[level]
            line = "  ".join(
                f"{name} {r['mean']:.1f}±{r['std']:.1f}"
                for name, r in res[pool][level].items() if isinstance(r, dict)
            )
            print(f"  [{pool}/{level}] target {t}±{tol}: {line}")
    print(f"R3 PASS: {res['pass']} ({res['passes']})")
