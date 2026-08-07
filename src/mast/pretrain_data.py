"""Device-resident corpus batching for pretraining (P2 plan W3).

Loads the compiled token cache (mast.tokenize) fully onto the training
device (~0.5 GB) and serves homogeneous per-source batches as pure
index-gathers — no DataLoader, no workers (MPS-safe).

Splits: record membership comes from the frozen object-level manifests
(corpus_holdout_k20_seed42: fold 0 = held-out 5%; corpus subsampling
for HPO selects objects whose fold is in the first 5 of the remaining
19 — both derived from the same manifest, so train/held-out never mix).

Contrastive pairing: for each batch row, a second record of the same
object from the same source (if one exists) — pair indices precomputed
per source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mast import splits as splits_mod
from mast.tokenize import TOKENS_DIR

HOLDOUT_MANIFEST = "corpus_holdout_k20_seed42"
SOURCES = ["gaia", "sdss", "skymapper", "movis"]


def build_holdout_manifest() -> pd.DataFrame:
    """Object-level k=20 folds over the corpus (fold 0 = held-out 5%)."""
    index = pd.read_parquet(TOKENS_DIR / "record_index.parquet")
    objects = (
        index.groupby("object_id")["source"].nunique().rename("n_sources").reset_index()
    )
    objects["stratum"] = objects.n_sources.astype(str)
    folds = splits_mod.make_object_folds(
        objects, k=20, seed=42, stratify_col="stratum"
    )
    splits_mod.write_manifest(
        folds, name=HOLDOUT_MANIFEST,
        meta={"seed": 42, "k": 20, "stratify": "n_sources",
              "pool": "pretraining corpus object union",
              "rule": "fold 0 = held-out 5% (never trained on); "
                      "folds 1-5 = 25% HPO subsample"},
    )
    return folds


class CorpusTensors:
    """All sources' token arrays on one device + batch sampling."""

    def __init__(self, device: str = "cpu", subset: str = "full",
                 tokens_dir: Path = TOKENS_DIR):
        """subset: 'full' (folds 1-19), 'hpo25' (folds 1-5), 'holdout' (fold 0)."""
        self.device = torch.device(device)
        self.meta = json.loads((tokens_dir / "meta.json").read_text())
        index = pd.read_parquet(tokens_dir / "record_index.parquet")

        manifest_path = splits_mod.SPLITS_DIR / f"{HOLDOUT_MANIFEST}.csv"
        if not manifest_path.exists():
            build_holdout_manifest()
        folds = pd.read_csv(manifest_path, dtype={"object_id": str})
        index = index.merge(folds[["object_id", "fold"]], on="object_id", how="left")
        index["fold"] = index.fold.fillna(-1).astype(int)
        if subset == "holdout":
            keep = index.fold == 0
        elif subset == "hpo25":
            keep = index.fold.between(1, 5)
        elif subset == "full":
            keep = index.fold >= 1
        else:
            raise ValueError(subset)

        self.data: dict[str, dict[str, torch.Tensor]] = {}
        self.pair_index: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        for source in SOURCES:
            rows = index[(index.source == source) & keep]
            if not len(rows):
                continue
            row_ids = rows.row.to_numpy()
            npz = np.load(tokens_dir / f"{source}.npz")
            tensors = {
                "values": torch.from_numpy(npz["values"][row_ids]),
                "lam": torch.from_numpy(npz["lam"][row_ids]),
                "dlam": torch.from_numpy(npz["dlam"][row_ids]),
                "token_type": torch.from_numpy(npz["token_type"][row_ids]).long(),
                "flags": torch.from_numpy(npz["flags"][row_ids]).long(),
                "valid": torch.from_numpy(npz["valid"][row_ids]),
                "instrument": torch.from_numpy(npz["instrument"][row_ids]).long(),
            }
            self.data[source] = {k: v.to(self.device) for k, v in tensors.items()}
            self.counts[source] = len(row_ids)
            # same-object same-source pair: next record of the object (or self)
            oid = rows.object_id.to_numpy()
            order = np.argsort(oid, kind="mergesort")
            pair = np.arange(len(oid))
            sorted_oid = oid[order]
            nxt = np.roll(order, -1)
            same = sorted_oid == np.roll(sorted_oid, -1)
            pair[order[same]] = nxt[same]
            self.pair_index[source] = torch.from_numpy(pair).to(self.device)

        self.n_total = sum(self.counts.values())
        self._probs = np.array([self.counts[s] for s in self.sources], dtype=float)
        self._probs /= self._probs.sum()

    @property
    def sources(self) -> list[str]:
        return list(self.data)

    def sample_batch(self, batch_size: int, rng: np.random.Generator,
                     with_pairs: bool = False):
        source = rng.choice(self.sources, p=self._probs)
        n = self.counts[source]
        idx = torch.from_numpy(rng.integers(0, n, size=min(batch_size, n))).to(self.device)
        batch = {k: v[idx] for k, v in self.data[source].items()}
        batch["source"] = source
        if with_pairs:
            pidx = self.pair_index[source][idx]
            batch["pair"] = {k: v[pidx] for k, v in self.data[source].items()}
        return batch

    def iter_all(self, source: str, batch_size: int):
        n = self.counts[source]
        for start in range(0, n, batch_size):
            idx = torch.arange(start, min(start + batch_size, n), device=self.device)
            yield {k: v[idx] for k, v in self.data[source].items()}
