"""Unit tests for label harmonization and object-level splits."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mast import labels, splits


# ---------------------------------------------------------------------------
# labels

def test_canonical_bdm():
    assert labels.canonical_bdm("Sw") == "S"
    assert labels.canonical_bdm("Sqw") == "Sq"
    assert labels.canonical_bdm("Sq:") == "Sq"
    assert labels.canonical_bdm("Ch") == "Ch"
    assert labels.canonical_bdm("CX") == ""
    assert labels.canonical_bdm("") == ""


def test_complex_maps_cover_schemes():
    assert len(labels.DEMEO_COMPLEX) == 24
    assert set(labels.DEMEO_COMPLEX.values()) == {"C", "S", "X", "end"}
    assert len(labels.MAHLKE_COMPLEX) == 17
    assert set(labels.MAHLKE_COMPLEX.values()) == {"C", "M", "S"}


def test_load_demeo2009():
    df = labels.load_demeo2009()
    assert len(df) == 371
    assert df.bdm_demeo2009.notna().all()
    assert df.object_id.is_unique


@pytest.mark.skipif(not labels.MAHLKE_DAT.exists(), reason="Mahlke snapshot absent")
def test_load_mahlke():
    df = labels.load_mahlke()
    assert len(df) == 4526
    assert df.object_id.is_unique
    assert df.mahlke_classsf.isin(labels.MAHLKE_COMPLEX).all()
    # probabilities present for the vast majority and within [0, 1]
    probs = df.mahlke_prob.dropna()
    assert len(probs) > 4000
    assert probs.between(0, 1).all()


def test_aggregate_majority_and_tie():
    per = pd.DataFrame(
        {"object_id": ["1", "1", "1", "2", "2"],
         "file": list("abcde"),
         "year": [2005, 2006, 2007, 2005, 2010],
         "bdm": ["S", "S", "Q", "C", "X"]}
    )
    agg = labels.aggregate_mithneos(per).set_index("object_id")
    assert agg.loc["1", "bdm_mithneos"] == "S"          # majority
    assert not agg.loc["1", "mithneos_complex_tie"]      # S-complex + end: S wins 2-1
    assert agg.loc["2", "bdm_mithneos"] == "X"           # tie -> latest year
    assert agg.loc["2", "mithneos_complex_tie"]          # C vs X complexes tied 1-1


# ---------------------------------------------------------------------------
# splits

def _pool(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"object_id": [str(i) for i in range(n)],
         "cls": rng.choice(["S", "C", "X", "V"], p=[0.5, 0.3, 0.15, 0.05], size=n)}
    )


def test_folds_deterministic_and_stratified():
    pool = _pool()
    f1 = splits.make_object_folds(pool, k=5, seed=42, stratify_col="cls")
    f2 = splits.make_object_folds(pool.sample(frac=1, random_state=1), k=5, seed=42,
                                  stratify_col="cls")
    pd.testing.assert_frame_equal(f1, f2)  # input order must not matter
    f3 = splits.make_object_folds(pool, k=5, seed=43, stratify_col="cls")
    assert not f1.fold.equals(f3.fold)     # seed changes assignment
    # stratification: each class spread across folds within ±1
    for _, g in f1.groupby("stratum"):
        counts = g.fold.value_counts().reindex(range(5), fill_value=0)
        assert counts.max() - counts.min() <= 1
    # every object appears exactly once (object-level, no leakage)
    assert f1.object_id.is_unique and len(f1) == len(pool)


def test_folds_rare_class_spread():
    pool = pd.DataFrame({"object_id": [str(i) for i in range(3)], "cls": ["R"] * 3})
    f = splits.make_object_folds(pool, k=10, seed=0, stratify_col="cls")
    assert f.fold.nunique() == 3  # 3 members land in 3 distinct folds


def test_duplicate_ids_raise():
    pool = pd.DataFrame({"object_id": ["1", "1"], "cls": ["S", "S"]})
    with pytest.raises(ValueError):
        splits.make_object_folds(pool, k=2, seed=0, stratify_col="cls")


def test_manifest_freeze(tmp_path):
    pool = _pool(30)
    folds = splits.make_object_folds(pool, k=3, seed=7, stratify_col="cls")
    rec = splits.write_manifest(folds, "t_folds", {"seed": 7}, splits_dir=tmp_path)
    assert splits.verify_manifest("t_folds", splits_dir=tmp_path)
    # identical rewrite: fine, returns stored record
    rec2 = splits.write_manifest(folds, "t_folds", {"seed": 7}, splits_dir=tmp_path)
    assert rec2["sha256"] == rec["sha256"]
    # changed content: refuses
    other = splits.make_object_folds(pool, k=3, seed=8, stratify_col="cls")
    with pytest.raises(RuntimeError):
        splits.write_manifest(other, "t_folds", {"seed": 8}, splits_dir=tmp_path)
