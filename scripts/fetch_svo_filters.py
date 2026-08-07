"""Fetch filter metadata + transmission curves from the SVO Filter
Profile Service for every photometric band in the pretraining corpus.

Writes:
  data/external/filters_svo/filter_metadata.csv   (id, lambda_eff_nm, fwhm_nm, ...)
  data/external/filters_svo/curves/<id>.csv       (wavelength_nm, transmission)

`src/mast/filters.py` reads filter_metadata.csv; the curves support the
plan's A7 transmission-curve tokenization ablation later.

Usage: .venv/bin/python scripts/fetch_svo_filters.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEST = Path(__file__).resolve().parents[1] / "data/external/filters_svo"

FILTER_IDS = [
    # SDSS (survey-telescope response)
    "SLOAN/SDSS.u", "SLOAN/SDSS.g", "SLOAN/SDSS.r", "SLOAN/SDSS.i", "SLOAN/SDSS.z",
    # SkyMapper
    "SkyMapper/SkyMapper.u", "SkyMapper/SkyMapper.v", "SkyMapper/SkyMapper.g",
    "SkyMapper/SkyMapper.r", "SkyMapper/SkyMapper.i", "SkyMapper/SkyMapper.z",
    # VISTA (MOVIS)
    "Paranal/VISTA.Y", "Paranal/VISTA.J", "Paranal/VISTA.H", "Paranal/VISTA.Ks",
]


def main() -> None:
    from astroquery.svo_fps import SvoFps

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "curves").mkdir(exist_ok=True)

    facilities = sorted({fid.split("/")[0] for fid in FILTER_IDS})
    meta = {}
    for fac in facilities:
        table = SvoFps.get_filter_list(facility=fac)
        for row in table:
            meta[str(row["filterID"])] = row

    rows = []
    for fid in FILTER_IDS:
        rec = meta[fid]
        rows.append(
            {"filter_id": fid,
             "lambda_eff_nm": float(rec["WavelengthEff"]) / 10.0,  # Å → nm
             "lambda_mean_nm": float(rec["WavelengthMean"]) / 10.0,
             "lambda_pivot_nm": float(rec["WavelengthPivot"]) / 10.0,
             "fwhm_nm": float(rec["FWHM"]) / 10.0,
             "width_eff_nm": float(rec["WidthEff"]) / 10.0,
             "mag_sys": str(rec["MagSys"]),
             "zeropoint_jy": float(rec["ZeroPoint"]),
             # Fsun: solar flux through the filter (erg/cm2/s/A per SVO) —
             # lets solar colors be derived from one consistent source
             "fsun": float(rec["Fsun"]) if rec["Fsun"] else float("nan"),
             }
        )
        curve = SvoFps.get_transmission_data(fid)
        pd.DataFrame(
            {"wavelength_nm": curve["Wavelength"].data / 10.0,
             "transmission": curve["Transmission"].data}
        ).to_csv(DEST / "curves" / f"{fid.replace('/', '_')}.csv", index=False)
        print(f"  {fid}: λ_eff={rows[-1]['lambda_eff_nm']:.1f} nm "
              f"FWHM={rows[-1]['fwhm_nm']:.1f} nm Fsun={rows[-1]['fsun']:.4g}")
    pd.DataFrame(rows).to_csv(DEST / "filter_metadata.csv", index=False)
    print(f"wrote {DEST/'filter_metadata.csv'} ({len(rows)} filters)")


if __name__ == "__main__":
    main()
