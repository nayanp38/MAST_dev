"""Readers for the DeMeo 2009 PDS4 bundle (data/raw/busdemeo2009)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE = Path(__file__).resolve().parents[2] / "data/raw/busdemeo2009/ast.bus-demeo.taxonomy/data"


def load_pcscores(path: Path | None = None) -> pd.DataFrame:
    """pcscores.tab: published slope + PC1–PC5 for the 371 objects."""
    path = path or BUNDLE / "pcscores.tab"
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        number = int(line[0:7])
        prov_desig = line[7:18].strip()
        vals = [float(v) for v in line[18:].split()]
        rows.append([number, prov_desig, *vals])
    return pd.DataFrame(rows, columns=["number", "prov_desig", "slope", "PC1", "PC2", "PC3", "PC4", "PC5"])


def load_taxonomy(path: Path | None = None) -> pd.DataFrame:
    """demeotax.tab: number, name, prov. designation, Bus-DeMeo class."""
    path = path or BUNDLE / "demeotax.tab"
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        rows.append(
            {
                "number": int(line[0:7]),
                "name": line[7:25].strip(),
                "prov_desig": line[25:36].strip(),
                "bdm_class": line[36:40].strip(),
            }
        )
    return pd.DataFrame(rows)


def load_meanspectra(path: Path | None = None) -> pd.DataFrame:
    """meanspectra.tab: 24 class templates on the 41-channel grid (wide)."""
    path = path or BUNDLE / "meanspectra.tab"
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    classes = raw.iloc[:, 0].tolist()
    values = raw.iloc[:, 1:42].to_numpy(dtype=float)
    return pd.DataFrame(values, index=classes)
