"""Canonical DeMeo 2009 spectral preprocessing (R1-certified).

Pipeline (DeMeo et al. 2009, Icarus 202, 160 — §3; numerics verified in
R1 against the published PC scores of the 371-object set; the
resample-then-slope-fit order below halves the per-object score
distance versus fitting the slope on the native grid):
  1. Normalize reflectance to unity at 0.55 um (nearest native sample).
  2. Spline-fit and resample to the 41-channel grid: 0.45–2.45 um in
     0.05 um steps. The spline is a smoothing spline whose smoothing
     target adapts to the spectrum's noise level (sigma estimated from
     second differences; s = n * sigma^2) — already-smooth archival
     "spfit" products are effectively interpolated, while raw noisy
     spectra are smoothed as in DeMeo's spline fits. Edges missing up
     to 4.7% of the grid range are extended with constant values.
  3. Fit a line to the 41 resampled channels (least squares), translate
     it to pass through (0.55 um, 1), and divide it out ("slope
     removal"). The translated line's slope gamma is the reported
     slope feature.
  4. PCA on the 40 channels that exclude the 0.55 um normalization
     channel, after subtracting the channel mean.

No imputation anywhere — spectra that do not cover the full grid are
rejected, never filled.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

# The canonical 41-channel Bus-DeMeo wavelength grid (um).
DEMEO_GRID = np.round(np.arange(0.45, 2.451, 0.05), 3)
NORM_WAVELENGTH = 0.55
# Channel indices retained for PCA (0.55 um channel dropped).
PCA_CHANNELS = np.array([i for i, w in enumerate(DEMEO_GRID) if abs(w - NORM_WAVELENGTH) > 1e-9])


def _clean(wave: np.ndarray, refl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by wavelength, drop non-finite / non-positive samples and duplicates."""
    wave = np.asarray(wave, dtype=float)
    refl = np.asarray(refl, dtype=float)
    ok = np.isfinite(wave) & np.isfinite(refl) & (refl > 0)
    wave, refl = wave[ok], refl[ok]
    order = np.argsort(wave)
    wave, refl = wave[order], refl[order]
    uniq, inv = np.unique(wave, return_inverse=True)
    if len(uniq) < len(wave):
        summed = np.zeros(len(uniq))
        counts = np.zeros(len(uniq))
        np.add.at(summed, inv, refl)
        np.add.at(counts, inv, 1)
        wave, refl = uniq, summed / counts
    return wave, refl


def normalize_at(wave: np.ndarray, refl: np.ndarray, at: float = NORM_WAVELENGTH) -> np.ndarray:
    """Divide by the reflectance at the native sample closest to `at`."""
    idx = int(np.argmin(np.abs(wave - at)))
    return refl / refl[idx]


def remove_slope(
    wave: np.ndarray, refl: np.ndarray, translate_to: float = NORM_WAVELENGTH
) -> tuple[float, np.ndarray]:
    """Fit a line to (wave, refl), translate to (translate_to, 1), divide.

    Canonically applied on the resampled 41-channel grid. Returns
    (gamma, slope-removed reflectance); gamma is the slope of the
    translated line — the slope feature reported by DeMeo 2009.
    """
    a, b = np.polyfit(wave, refl, 1)
    b = 1.0 - a * translate_to
    continuum = a * wave + b
    return float(a), refl / continuum


# Maximum fraction of the grid range that may be extrapolated at the
# edges (matches classy's EXTRAPOLATION_LIMIT of 4.7%).
MAX_EXTRAPOLATION_FRACTION = 0.047


def estimate_noise_sigma(refl: np.ndarray) -> float:
    """Per-point noise estimate from second differences.

    For iid noise the second difference has variance 6*sigma^2; smooth
    spectral structure contributes negligibly at native sampling.
    """
    return float(np.sqrt(np.mean(np.diff(refl, 2) ** 2) / 6.0))


def resample_to_demeo_grid(
    wave: np.ndarray,
    refl: np.ndarray,
    grid: np.ndarray = DEMEO_GRID,
    smoothing: float | str = "auto",
    max_extrapolation_fraction: float = MAX_EXTRAPOLATION_FRACTION,
) -> np.ndarray:
    """Spline-fit and sample on the DeMeo grid.

    smoothing: "auto" (default) sets the smoothing target to
    n * sigma^2 with sigma from `estimate_noise_sigma` — noisy raw
    spectra are smoothed (as in DeMeo's spline fits), already-smooth
    products are effectively interpolated. Pass 0 for a strictly
    interpolating spline, or a float for an explicit target.

    Edges missing up to `max_extrapolation_fraction` of the grid range
    (summed over both ends) are extended with constant values; spectra
    missing more raise ValueError (no imputation beyond this canonical
    edge tolerance).
    """
    missing = max(wave.min() - grid.min(), 0.0) + max(grid.max() - wave.max(), 0.0)
    if missing > max_extrapolation_fraction * (grid.max() - grid.min()) + 1e-12:
        raise ValueError(
            f"spectrum covers {wave.min():.3f}-{wave.max():.3f} um, "
            f"needs {grid.min():.2f}-{grid.max():.2f} um "
            f"(missing {missing:.3f} um > extrapolation limit)"
        )
    if smoothing == "auto":
        smoothing = len(wave) * estimate_noise_sigma(refl) ** 2
    spline = UnivariateSpline(wave, refl, k=3, s=smoothing)
    inside = (grid >= wave.min()) & (grid <= wave.max())
    out = np.empty(len(grid))
    out[inside] = spline(grid[inside])
    out[grid < wave.min()] = refl[0]
    out[grid > wave.max()] = refl[-1]
    return out


def read_spectrum_file(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a SMASS/MITHNEOS ASCII spectrum robustly.

    Takes the first two float tokens per line (wavelength um,
    reflectance); tolerates ragged rows and comment lines.
    """
    waves, refls = [], []
    for line in open(path):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            w, r = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        waves.append(w)
        refls.append(r)
    return np.array(waves), np.array(refls)


def preprocess_demeo(wave: np.ndarray, refl: np.ndarray) -> tuple[float, np.ndarray]:
    """Full single-spectrum pipeline.

    Returns (gamma, 41-channel slope-removed spectrum on DEMEO_GRID).
    Use `spectrum[PCA_CHANNELS]` for the 40-channel PCA input.
    """
    wave, refl = _clean(wave, refl)
    refl = normalize_at(wave, refl)
    resampled = resample_to_demeo_grid(wave, refl)
    gamma, resampled = remove_slope(DEMEO_GRID, resampled)
    return gamma, resampled


def pca(spectra: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain PCA via SVD on channel-mean-subtracted spectra.

    `spectra` should already be restricted to PCA_CHANNELS (40 columns).
    Returns (scores [n_obj, n_comp], components [n_comp, n_chan],
    mean_spectrum [n_chan]). Component sign is fixed so that each
    component's largest-|loading| channel is positive (sign is
    arbitrary; comparisons must allow flips).
    """
    spectra = np.asarray(spectra, dtype=float)
    mean = spectra.mean(axis=0)
    centered = spectra - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    for i, comp in enumerate(components):
        if comp[np.argmax(np.abs(comp))] < 0:
            components[i] = -comp
    scores = centered @ components.T
    return scores, components, mean
