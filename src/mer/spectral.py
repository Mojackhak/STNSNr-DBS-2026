# src/mer/spectral.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np


BandpowerMode = Literal["mean", "sum", "relative"]
NanPolicy = Literal["zero", "propagate"]


def psd_to_bandpower(
    psds: np.ndarray,
    freqs: np.ndarray,
    bands: Mapping[str, Sequence[float]],
    *,
    fmin_psd: float | None = None,
    fmax_psd: float | None = None,
    mode: BandpowerMode = "mean",
    nan_policy: NanPolicy = "zero",
    eps: float = 1e-30,
) -> dict[str, np.ndarray]:
    """
    Convert PSDs to bandpower per channel.

    This function treats each bandpower as the integral of PSD over frequency.
    - mode="sum": returns integrated power within each band.
    - mode="mean": returns integrated power divided by band width (power per Hz).
    - mode="relative": returns band integrated power divided by total integrated power
      across the selected PSD frequency range.

    Parameters
    ----------
    psds
        PSD array of shape (n_channels, n_freqs).
    freqs
        Frequency vector of shape (n_freqs,). Must be finite and strictly increasing.
    bands
        Mapping from band name to (fmin, fmax). Endpoints are inclusive.
    fmin_psd, fmax_psd
        Optional overall frequency range to crop PSD before computing bandpower.
    mode
        "mean", "sum", or "relative".
    nan_policy
        "zero": treat NaNs in PSD as 0.0 (your original behavior).
        "propagate": keep NaNs (any integration over NaNs will produce NaN).
    eps
        Small positive number to stabilize division (mainly for relative mode).

    Returns
    -------
    dict[str, np.ndarray]
        Bandpower values per band, each an array of shape (n_channels,).
    """
    psds = np.asarray(psds, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if psds.ndim != 2:
        raise ValueError(f"psds must be 2D (n_channels, n_freqs), got shape={psds.shape}")
    if freqs.ndim != 1:
        raise ValueError(f"freqs must be 1D (n_freqs,), got shape={freqs.shape}")
    if psds.shape[1] != freqs.shape[0]:
        raise ValueError(
            f"psds.shape[1] must equal len(freqs); got {psds.shape[1]} vs {freqs.shape[0]}"
        )
    if freqs.size == 0:
        raise ValueError("freqs is empty")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("freqs contains non-finite values")
    if np.any(np.diff(freqs) <= 0):
        raise ValueError("freqs must be strictly increasing")

    # Crop to the PSD frequency range of interest
    lo = -np.inf if fmin_psd is None else float(fmin_psd)
    hi = np.inf if fmax_psd is None else float(fmax_psd)
    main_mask = (freqs >= lo) & (freqs <= hi)
    freqs_sel = freqs[main_mask]
    psds_sel = psds[:, main_mask]

    n_channels = psds.shape[0]
    if freqs_sel.size == 0:
        # Return zeros: no bins after cropping
        return {name: np.zeros(n_channels, dtype=float) for name in bands.keys()}

    if nan_policy == "zero":
        psds_sel = np.nan_to_num(psds_sel, nan=0.0)
    elif nan_policy != "propagate":
        raise ValueError(f"Invalid nan_policy={nan_policy!r}")

    # Estimate a representative frequency resolution for the single-bin case.
    # For multi-bin integration we always use x=freqs, so this is only fallback.
    if freqs_sel.size >= 2:
        df_rep = float(np.median(np.diff(freqs_sel)))
    else:
        # If the whole PSD selection is only a single bin, fall back to 1 Hz nominal width.
        df_rep = 1.0

    # Total power across the selected PSD range (for relative mode)
    power_total = np.trapezoid(y=psds_sel, x=freqs_sel, axis=1)

    out: dict[str, np.ndarray] = {}

    for band_name, band in bands.items():
        if len(band) != 2:
            raise ValueError(f"Band {band_name!r} must be (fmin, fmax), got {band!r}")

        f_low, f_high = float(band[0]), float(band[1])
        if not np.isfinite(f_low) or not np.isfinite(f_high):
            raise ValueError(f"Band {band_name!r} has non-finite limits: {band!r}")
        if f_high <= f_low:
            raise ValueError(f"Band {band_name!r} must satisfy fmax > fmin, got {band!r}")

        band_width = f_high - f_low
        band_mask = (freqs_sel >= f_low) & (freqs_sel <= f_high)
        freqs_band = freqs_sel[band_mask]
        psds_band = psds_sel[:, band_mask]

        if freqs_band.size == 0:
            band_power = np.zeros(n_channels, dtype=float)
        elif freqs_band.size == 1:
            # Single-bin approximation: integral ≈ PSD(f) * df
            band_power = psds_band[:, 0] * df_rep
        else:
            band_power = np.trapezoid(y=psds_band, x=freqs_band, axis=1)

        if mode == "sum":
            out[band_name] = band_power
        elif mode == "mean":
            out[band_name] = band_power / band_width
        elif mode == "relative":
            denom = np.maximum(np.abs(power_total), eps)
            out[band_name] = band_power / denom
        else:
            raise ValueError(f"Invalid mode={mode!r}")

    return out
