"""Label harmonization for the Phase-1 labeled pool (plan §2.2).

Produces label table v1: one row per object with
  (Bus-DeMeo class, Bus-DeMeo complex, Mahlke class + probability,
   disputed flag) plus full per-source provenance columns.

Bus-DeMeo label sources (all VIS+NIR):
  tier 1 — DeMeo et al. 2009 published classes (demeotax.tab, 371 objects).
  tier 2 — classy end-to-end (its own Spectra loader + released DeMeo
           decision tree) on the joined VIS+NIR MITHNEOS spectra;
           per-object majority vote across spectra with
           latest-observation tiebreak.

Harmonization rules (documented deviation-free from plan §2.2):
  - All sources here are VIS+NIR, so rule (1) [VIS+NIR over VIS-only]
    never discriminates; VIS-only sources are deliberately excluded
    from label table v1.
  - A curated literature label (tier 1) outranks our tree-derived
    labels (tier 2); tier 2 fills objects absent from tier 1.
  - disputed = True when the two tiers disagree at the *complex* level
    (subclass wobble within a complex, e.g. S vs Sq, is expected
    tree behavior and not flagged), or when tier 2's per-spectrum
    complexes have no majority. Disputed objects are excluded from
    training but kept in test sets (enforced downstream).

Mahlke labels come from the frozen VizieR snapshot
(data/raw/mahlke2022_vizier/asteroid.dat.gz), aggregated per asteroid,
with the probability of the assigned class.
"""

from __future__ import annotations

import glob
import gzip
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mast import demeo_pds

ROOT = Path(__file__).resolve().parents[2]
MAHLKE_DAT = ROOT / "data/raw/mahlke2022_vizier/asteroid.dat.gz"
CLASSY_MITHNEOS = Path.home() / "Library/Caches/classy/mithneos"

# ---------------------------------------------------------------------------
# Complex mappings, fixed a priori (DeMeo 2009; Mahlke 2022 Fig. 6)

DEMEO_COMPLEX = {
    "B": "C", "C": "C", "Cb": "C", "Cg": "C", "Cgh": "C", "Ch": "C",
    "S": "S", "Sa": "S", "Sq": "S", "Sr": "S", "Sv": "S",
    "X": "X", "Xc": "X", "Xe": "X", "Xk": "X",
    "A": "end", "D": "end", "K": "end", "L": "end", "O": "end",
    "Q": "end", "R": "end", "T": "end", "V": "end",
}

MAHLKE_COMPLEX = {
    "B": "C", "C": "C", "Ch": "C", "P": "C", "D": "C", "Z": "C",
    "E": "M", "M": "M", "X": "M",
    "A": "S", "K": "S", "L": "S", "O": "S", "Q": "S", "R": "S",
    "S": "S", "V": "S",
}


# Penttilä et al. 2021 collapse: 24 Bus-DeMeo classes -> 11.
# Subclasses merge into their complex parent; end members stay. O and R
# (1-2 objects each) have no home in the 11-class scheme and are dropped.
PENTTILA_11 = {
    "B": "B",
    "C": "C", "Cb": "C", "Cg": "C", "Cgh": "C", "Ch": "C",
    "S": "S", "Sa": "S", "Sq": "S", "Sr": "S", "Sv": "S",
    "X": "X", "Xc": "X", "Xe": "X", "Xk": "X",
    "A": "A", "D": "D", "K": "K", "L": "L", "Q": "Q", "T": "T", "V": "V",
}

# Klimczak et al. 2021 4-complex scheme: C, S, X, other ("end").
KLIMCZAK_COMPLEX = {**{k: v for k, v in DEMEO_COMPLEX.items()}}


def canonical_bdm(cls: str) -> str:
    """Canonicalize a published/derived Bus-DeMeo label to the 24 classes.

    Drops the high-slope 'w' suffix (e.g. 'Sw' -> 'S', used both by
    DeMeo 2009 and classy) and the ':' uncertainty marker in
    demeotax.tab. Returns '' if the result is not a known class.
    """
    cls = str(cls).strip().rstrip(":")
    if cls.endswith("w") and len(cls) > 1:
        cls = cls[:-1]
    return cls if cls in DEMEO_COMPLEX else ""


def demeo_complex(cls: str) -> str:
    return DEMEO_COMPLEX.get(canonical_bdm(cls), "")


# ---------------------------------------------------------------------------
# Sources

def load_demeo2009() -> pd.DataFrame:
    """Tier-1 Bus-DeMeo labels: DeMeo et al. 2009 published classes."""
    tax = demeo_pds.load_taxonomy()
    tax = tax.rename(columns={"bdm_class": "bdm_demeo2009_raw"})
    tax["bdm_demeo2009"] = tax.bdm_demeo2009_raw.map(canonical_bdm).replace("", np.nan)
    tax["object_id"] = np.where(
        tax.number > 0, tax.number.astype(str), tax.prov_desig
    )
    return tax[["object_id", "number", "name", "bdm_demeo2009", "bdm_demeo2009_raw"]]


def _mithneos_files() -> list[str]:
    return sorted(glob.glob(str(CLASSY_MITHNEOS / "*" / "a*.txt")))


def _full_range_object_numbers() -> list[int]:
    """Object numbers with at least one full VIS+NIR spectrum in the cache."""
    from mast import preprocessing as pp

    numbers = set()
    for f in _mithneos_files():
        m = re.match(r"a(\d+)\.", os.path.basename(f))
        if not m or int(m.group(1)) in numbers:
            continue
        wave, _ = pp.read_spectrum_file(f)
        if len(wave) >= 20 and wave.min() <= 0.46 and wave.max() >= 2.44:
            numbers.add(int(m.group(1)))
    return sorted(numbers)


def classify_mithneos_demeo(
    numbers: list[int] | None = None, progress_path: Path | None = None
) -> pd.DataFrame:
    """Tier-2: classy end-to-end DeMeo classification of MITHNEOS spectra.

    Spectra are loaded per object through classy's own Spectra loader
    and classified with its released DeMeo decision tree — manually
    constructed Spectrum objects mis-classify (verified on (2) Pallas:
    loader path gives B, manual path gives V), so the released loader
    is mandatory here despite being slower (per-object SsODNet lookups).

    Returns one row per classifiable spectrum: object_id, file, year,
    class ('indeterminate' for CX/empty tree results, kept for
    provenance). If progress_path is given, results are appended there
    incrementally and already-done objects are skipped on re-run.
    """
    import classy

    if numbers is None:
        numbers = _full_range_object_numbers()

    done = set()
    rows = []
    if progress_path is not None and Path(progress_path).exists():
        prev = pd.read_csv(progress_path, dtype={"object_id": str})
        rows = prev.to_dict("records")
        done = set(prev.object_id)

    for num in numbers:
        if str(num) in done:
            continue
        new = []
        try:
            specs = classy.Spectra(num, source="MITHNEOS")
        except Exception:
            specs = []
        for spec in specs:
            if spec.wave.min() > 0.46 or spec.wave.max() < 2.44:
                continue
            try:
                spec.classify(taxonomy="demeo")
            except Exception:
                continue
            cls = canonical_bdm(str(getattr(spec, "class_demeo", "") or ""))
            date = str(getattr(spec, "date_obs", "") or "")
            year = int(date[:4]) if date[:4].isdigit() else -1
            new.append(
                {"object_id": str(num), "file": os.path.basename(spec.filename),
                 "year": year, "bdm": cls if cls else "indeterminate"}
            )
        if not new:  # record objects with no classifiable spectrum
            new = [{"object_id": str(num), "file": "", "year": -1, "bdm": "none"}]
        rows.extend(new)
        if progress_path is not None:
            pd.DataFrame(rows).to_csv(progress_path, index=False)
    df = pd.DataFrame(rows)
    return df[df.bdm != "none"].reset_index(drop=True)


def aggregate_mithneos(per_spectrum: pd.DataFrame) -> pd.DataFrame:
    """Per-object majority vote (latest-year tiebreak) over tier-2 spectra."""
    per_spectrum = per_spectrum[per_spectrum.bdm != "indeterminate"]
    rows = []
    for oid, g in per_spectrum.groupby("object_id"):
        counts = g.bdm.value_counts()
        top = counts[counts == counts.max()].index.tolist()
        if len(top) == 1:
            cls = top[0]
        else:  # tie: latest observation among tied classes
            cls = g[g.bdm.isin(top)].sort_values("year").iloc[-1].bdm
        complexes = g.bdm.map(DEMEO_COMPLEX)
        cmax = complexes.value_counts()
        no_majority = (cmax == cmax.max()).sum() > 1
        rows.append(
            {"object_id": oid, "bdm_mithneos": cls,
             "n_mithneos_spectra": len(g),
             "mithneos_year": int(g.year.max()),
             "mithneos_complex_tie": bool(no_majority)}
        )
    return pd.DataFrame(rows)


def load_mahlke() -> pd.DataFrame:
    """Mahlke 2022 per-asteroid class + probability of the assigned class."""
    classes = list("ABCDEKLMOPQRSVXZ")  # probability columns (Ch has none)
    colspecs = [(0, 17), (18, 24), (27, 30), (31, 33)]
    names = ["name", "number", "mahlke_class", "mahlke_classsf"]
    prob_start = 34
    for i, c in enumerate(classes):
        colspecs.append((prob_start + 8 * i, prob_start + 8 * i + 7))
        names.append(f"p{c}")
    with gzip.open(MAHLKE_DAT, "rt") as f:
        df = pd.read_fwf(f, colspecs=colspecs, names=names)
    df["number"] = pd.to_numeric(df.number, errors="coerce").astype("Int64")
    df["object_id"] = np.where(
        df.number.notna(), df.number.astype("Int64").astype(str), df.name.str.strip()
    )
    df["mahlke_class"] = df.mahlke_class.str.strip()
    df["mahlke_classsf"] = df.mahlke_classsf.str.strip()

    def prob_of_assigned(row):
        col = f"p{row.mahlke_classsf}"
        return float(row[col]) if col in row.index and pd.notna(row[col]) else np.nan

    df["mahlke_prob"] = df.apply(prob_of_assigned, axis=1)
    df["mahlke_complex"] = df.mahlke_classsf.map(MAHLKE_COMPLEX)
    return df[["object_id", "name", "number", "mahlke_class", "mahlke_classsf",
               "mahlke_prob", "mahlke_complex"]]


# ---------------------------------------------------------------------------
# Harmonization

def build_label_table(per_spectrum: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble label table v1 (one row per object)."""
    demeo = load_demeo2009()
    if per_spectrum is None:
        per_spectrum = classify_mithneos_demeo()
    mith = aggregate_mithneos(per_spectrum)
    mahlke = load_mahlke()

    table = demeo.merge(mith, on="object_id", how="outer")
    table = table.merge(
        mahlke.drop(columns=["name", "number"]), on="object_id", how="outer"
    )

    # harmonized Bus-DeMeo label: curated tier 1 first, tier 2 fills
    table["bdm_class"] = table.bdm_demeo2009.fillna(table.bdm_mithneos)
    table["bdm_source"] = np.select(
        [table.bdm_demeo2009.notna(), table.bdm_mithneos.notna()],
        ["DeMeo+ 2009", "classy-demeo-tree (MITHNEOS)"], default="",
    )
    table["bdm_complex"] = table.bdm_class.map(lambda c: demeo_complex(c) if pd.notna(c) else "")

    both = table.bdm_demeo2009.notna() & table.bdm_mithneos.notna()
    tier_conflict = both & (
        table.bdm_demeo2009.map(demeo_complex) != table.bdm_mithneos.map(demeo_complex)
    )
    # a tier-2 internal tie only disputes objects that have no curated
    # tier-1 label (otherwise tier 1 resolves it)
    tier2_tie = table.mithneos_complex_tie.fillna(False) & table.bdm_demeo2009.isna()
    table["disputed"] = tier_conflict | tier2_tie

    table = table.sort_values(
        "object_id", key=lambda s: s.map(lambda x: (0, int(x)) if str(x).isdigit() else (1, 0))
    ).reset_index(drop=True)
    cols = ["object_id", "number", "name", "bdm_class", "bdm_complex", "bdm_source",
            "disputed", "bdm_demeo2009", "bdm_demeo2009_raw", "bdm_mithneos",
            "n_mithneos_spectra", "mithneos_year", "mithneos_complex_tie",
            "mahlke_class", "mahlke_classsf", "mahlke_prob", "mahlke_complex"]
    return table[cols]
