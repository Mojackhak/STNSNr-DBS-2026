"""mer.stn_boundaries

Main purpose
------------
Estimate STN (subthalamic nucleus) entry/exit depths from depth-vs-feature
curves derived from intraoperative MER.

The functions here operate on per-side dictionaries of numpy arrays and support
both manual overrides and simple automatic heuristics.
"""

from __future__ import annotations
from typing import Literal

import numpy as np

ExitMethod = Literal["min", "turning"]
EntryMethod = Literal["crossing", "median"]


def _to_1d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def _mask_range(x: np.ndarray, y: np.ndarray, depth_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x1 = _to_1d_float(x)
    y1 = _to_1d_float(y)
    if x1.size != y1.size:
        raise ValueError(f"x and y must have the same length (got {x1.size} vs {y1.size})")

    lo = float(min(depth_range))
    hi = float(max(depth_range))

    mask = np.isfinite(x1) & np.isfinite(y1) & (x1 >= lo) & (x1 <= hi)
    xm = x1[mask]
    ym = y1[mask]

    if xm.size < 2:
        raise ValueError(f"Not enough points within range [{lo}, {hi}] (n={xm.size}).")

    # Sort by depth for consistent interpolation/gradient.
    order = np.argsort(xm)
    return xm[order], ym[order]


def _linear_crossings(x: np.ndarray, y: np.ndarray, threshold: float) -> list[float]:
    """Find all x where y crosses threshold using linear interpolation."""
    x1 = _to_1d_float(x)
    y1 = _to_1d_float(y)

    d = y1 - float(threshold)

    crossings: list[float] = []
    for i in range(d.size - 1):
        d0 = d[i]
        d1 = d[i + 1]

        if not np.isfinite(d0) or not np.isfinite(d1):
            continue

        # Exact hit
        if d0 == 0.0:
            crossings.append(float(x1[i]))
            continue

        # Sign change indicates a crossing between i and i+1.
        if d0 * d1 < 0.0:
            x0 = float(x1[i])
            x1_ = float(x1[i + 1])
            y0 = float(y1[i])
            y1__ = float(y1[i + 1])
            denom = (y1__ - y0)
            if denom == 0.0:
                continue
            xc = x0 + (threshold - y0) * (x1_ - x0) / denom
            crossings.append(float(xc))

    # Handle an exact threshold hit at the final sample.
    if d.size > 0 and np.isfinite(d[-1]) and d[-1] == 0.0:
        crossings.append(float(x1[-1]))

    return crossings


def get_stn_exit_depth(
    x_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    exit_range_dict: dict[str, tuple[float, float]],
    manual_exit_depths: dict[str, float] | None = None,
    *,
    method: ExitMethod = "min",
) -> dict[str, float]:
    """Determine STN exit (ventral boundary) for each side.

    Parameters
    ----------
    x_dict, y_dict:
        Per-side arrays of depth (x) and feature values (y).
    exit_range_dict:
        Per-side (min_depth, max_depth) within which to search.
    manual_exit_depths:
        Optional dict of manual overrides. If provided and a side is present,
        that value is returned directly.
    method:
        * ``min``: choose depth at minimum y within exit_range
        * ``turning``: choose depth where dy/dx crosses 0 (last negative slope)

    Returns
    -------
    dict[str, float]
        Mapping side -> exit depth.
    """
    manual = manual_exit_depths or {}
    out: dict[str, float] = {}

    for side in x_dict.keys():
        if side in manual:
            out[side] = float(manual[side])
            continue

        if side not in y_dict or side not in exit_range_dict:
            raise KeyError(f"Missing y or range for side '{side}'")

        x, y = _mask_range(x_dict[side], y_dict[side], exit_range_dict[side])

        if method == "min":
            idx = int(np.argmin(y))
            out[side] = float(x[idx])
            continue

        if method == "turning":
            # Use dy/dx to find the turning point where slope switches sign.
            dy = np.gradient(y, x)

            # Find the last index where dy < 0 and next dy >= 0.
            candidates = np.where((dy[:-1] < 0.0) & (dy[1:] >= 0.0))[0]
            if candidates.size == 0:
                raise ValueError(f"No turning point found for side '{side}' within the specified range.")

            i = int(candidates[-1])
            x0, x1_ = float(x[i]), float(x[i + 1])
            dy0, dy1_ = float(dy[i]), float(dy[i + 1])

            if dy1_ == dy0:
                out[side] = x0
            else:
                out[side] = x0 + (0.0 - dy0) * (x1_ - x0) / (dy1_ - dy0)
            continue

        raise ValueError("method must be one of {'min', 'turning'}")

    return out


def get_stn_entry_depth(
    x_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    entry_range_dict: dict[str, tuple[float, float]],
    manual_entry_depths: dict[str, float] | None,
    n_rms_threshold: float,
    *,
    method: EntryMethod = "crossing",
) -> dict[str, float]:
    """Determine STN entry (dorsal boundary) for each side.

    Parameters
    ----------
    x_dict, y_dict:
        Per-side arrays of depth (x) and feature values (y). In many workflows
        these are smoothed curves (e.g., LOESS).
    entry_range_dict:
        Per-side (min_depth, max_depth) within which to search.
    manual_entry_depths:
        Optional dict of manual overrides.
    n_rms_threshold:
        Threshold used when ``method='crossing'``.
    method:
        * ``crossing``: find depth where y crosses the threshold
        * ``median``: choose depth whose y is closest to median(y) within range

    Returns
    -------
    dict[str, float]
        Mapping side -> entry depth.
    """
    manual = manual_entry_depths or {}
    out: dict[str, float] = {}

    for side in x_dict.keys():
        if side in manual:
            out[side] = float(manual[side])
            continue

        if side not in y_dict or side not in entry_range_dict:
            raise KeyError(f"Missing y or range for side '{side}'")

        x, y = _mask_range(x_dict[side], y_dict[side], entry_range_dict[side])

        if method == "crossing":
            crossings = _linear_crossings(x, y, float(n_rms_threshold))
            if not crossings:
                raise ValueError(
                    f"No threshold crossing found for side '{side}' within the specified range "
                    f"(threshold={n_rms_threshold})."
                )
            # Choose the most dorsal crossing (largest depth).
            out[side] = float(max(crossings))
            continue

        if method == "median":
            med = float(np.median(y))
            idx = int(np.argmin(np.abs(y - med)))
            out[side] = float(x[idx])
            continue

        raise ValueError("method must be one of {'crossing', 'median'}")

    return out


# ----------------------------------------------------------------------------
# Legacy aliases (to minimize breaking changes)
# ----------------------------------------------------------------------------

def get_STN_exit(
    x_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    exit_range_dict: dict[str, tuple[float, float]],
    STN_exit_dict_manual: dict[str, float],
    method: ExitMethod = "min",
) -> dict[str, float]:
    """Legacy alias for :func:`get_stn_exit_depth`."""
    return get_stn_exit_depth(
        x_dict,
        y_dict,
        exit_range_dict,
        manual_exit_depths=STN_exit_dict_manual,
        method=method,
    )


def get_STN_entry(
    x_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    entry_range_dict: dict[str, tuple[float, float]],
    STN_entry_dict_manual: dict[str, float],
    nRMS_threshold: float,
    method: EntryMethod = "crossing",
) -> dict[str, float]:
    """Legacy alias for :func:`get_stn_entry_depth`."""
    return get_stn_entry_depth(
        x_dict,
        y_dict,
        entry_range_dict,
        manual_entry_depths=STN_entry_dict_manual,
        n_rms_threshold=nRMS_threshold,
        method=method,
    )
