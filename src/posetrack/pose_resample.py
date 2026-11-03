"""
Resample PoseTrack per-frame DataFrames to a new FPS using 1D interpolation.

Each column is interpolated independently along the frame/time axis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, UnivariateSpline, interp1d


def _warn(msg: str) -> None:
    print(f"[pose_fps_interp] {msg}")


def _frame_values_from_df(
    df: pd.DataFrame,
    *,
    frame_col: str | None,
    frame0: float | None,
) -> np.ndarray:
    if frame_col is not None and frame_col in df.columns:
        frame_vals = pd.to_numeric(df[frame_col], errors="coerce").to_numpy(dtype=float)
    else:
        try:
            frame_vals = pd.to_numeric(df.index, errors="raise").to_numpy(dtype=float)
        except Exception:
            frame_vals = np.arange(len(df), dtype=float)

    if not np.all(np.isfinite(frame_vals)):
        raise ValueError("Frame values contain NaN/Inf; provide a clean frame_col or numeric index.")

    if frame0 is None:
        frame0 = float(frame_vals[0])

    return frame_vals - float(frame0)


def _time_axis_from_frames(
    frame_vals: np.ndarray,
    *,
    fps: float,
    t0_offset_sec: float,
) -> np.ndarray:
    return frame_vals / float(fps) + float(t0_offset_sec)


def _build_time_axis_new(t_old: np.ndarray, fps_new: float) -> np.ndarray:
    if t_old.size == 0:
        return t_old.copy()
    t_start = float(t_old[0])
    t_end = float(t_old[-1])
    if t_end < t_start:
        raise ValueError("Time axis must be non-decreasing.")
    step = 1.0 / float(fps_new)
    # Include end point (within floating tolerance).
    return np.arange(t_start, t_end + 0.5 * step, step, dtype=float)


def _akima_with_extrap(t_valid: np.ndarray, y_valid: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    ak = Akima1DInterpolator(t_valid, y_valid)
    t0, t1 = float(t_valid[0]), float(t_valid[-1])
    y0, y1 = float(y_valid[0]), float(y_valid[-1])
    slope_lo = (y_valid[1] - y_valid[0]) / (t_valid[1] - t_valid[0])
    slope_hi = (y_valid[-1] - y_valid[-2]) / (t_valid[-1] - t_valid[-2])

    def f(t_new: np.ndarray) -> np.ndarray:
        out = ak(t_new).astype(float)
        lo_mask = t_new < t0
        hi_mask = t_new > t1
        if np.any(lo_mask):
            out[lo_mask] = y0 + slope_lo * (t_new[lo_mask] - t0)
        if np.any(hi_mask):
            out[hi_mask] = y1 + slope_hi * (t_new[hi_mask] - t1)
        return out

    return f


def _make_interpolator(
    t_valid: np.ndarray,
    y_valid: np.ndarray,
    method: str,
    *,
    spline_order: int,
) -> Callable[[np.ndarray], np.ndarray]:
    method_l = str(method).lower()
    if method_l == "akima":
        return _akima_with_extrap(t_valid, y_valid)
    if method_l == "spline":
        k = int(spline_order)
        k = max(1, min(k, len(t_valid) - 1))
        spl = UnivariateSpline(t_valid, y_valid, k=k, s=0)

        def f(t_new: np.ndarray) -> np.ndarray:
            return spl(t_new).astype(float)

        return f

    # Fall back to scipy interp1d kinds (linear, nearest, cubic, ...)
    interp_kind = method_l
    min_points = 2
    if interp_kind in ("cubic", "quadratic"):
        min_points = 4 if interp_kind == "cubic" else 3
    if len(t_valid) < min_points:
        _warn(
            f"Not enough points for method '{method_l}' (need {min_points}, got {len(t_valid)}). "
            "Falling back to linear."
        )
        interp_kind = "linear"
    f0 = interp1d(t_valid, y_valid, kind=interp_kind, fill_value="extrapolate", bounds_error=False)

    def f(t_new: np.ndarray) -> np.ndarray:
        return f0(t_new).astype(float)

    return f


def _remsaple_values(
    values: np.ndarray,
    t_old: np.ndarray,
    t_new: np.ndarray,
    method: str,
    *,
    spline_order: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & np.isfinite(t_old)
    if mask.sum() == 0:
        return np.full_like(t_new, np.nan, dtype=float)
    if mask.sum() == 1:
        return np.full_like(t_new, float(values[mask][0]), dtype=float)
    t_valid = t_old[mask]
    y_valid = values[mask]
    interp_fn = _make_interpolator(t_valid, y_valid, method, spline_order=spline_order)
    return interp_fn(t_new)


def remsaple_pose_df_fps(
    pose_df: pd.DataFrame,
    fps: float,
    fps_new: float,
    *,
    method: str = "akima",
    frame_col: str | None = None,
    frame0: float | None = None,
    t0_offset_sec: float = 0.0,
    index_name: str | None = "frame",
    spline_order: int = 3,
) -> pd.DataFrame:
    """
    remsaple a PoseTrack dataframe to a new FPS.

    Parameters
    ----------
    pose_df
        Per-frame pose dataframe.
    fps
        Original FPS of the dataframe.
    fps_new
        Target FPS.
    method
        Interpolation method: "akima" (default), "linear", "nearest", "cubic",
        "quadratic", "spline", etc.
    frame_col
        Optional column name to use as frame index instead of df.index.
    frame0
        Frame value treated as time 0. Defaults to the first frame value.
    t0_offset_sec
        Optional time offset (sec) added to frame time.
    index_name
        Name of the output index. Set to None to keep default.
    spline_order
        Order for spline interpolation when method="spline".
    """
    if pose_df.shape[0] == 0:
        out_empty = pose_df.copy()
        if index_name is not None:
            out_empty.index.name = index_name
        return out_empty

    fps = float(fps)
    fps_new = float(fps_new)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and > 0, got: {fps}")
    if not np.isfinite(fps_new) or fps_new <= 0:
        raise ValueError(f"fps_new must be finite and > 0, got: {fps_new}")

    frame_vals = _frame_values_from_df(pose_df, frame_col=frame_col, frame0=frame0)
    t_old = _time_axis_from_frames(frame_vals, fps=fps, t0_offset_sec=t0_offset_sec)
    if np.any(np.diff(t_old) < 0):
        raise ValueError("Time axis must be non-decreasing.")

    t_new = _build_time_axis_new(t_old, fps_new)
    n_new = len(t_new)

    out_cols: dict[str, np.ndarray] = {}
    for col in pose_df.columns:
        if frame_col is not None and col == frame_col:
            continue
        series = pose_df[col]
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if not np.any(np.isfinite(values)):
            _warn(f"Column '{col}' is non-numeric or all-NaN; filled with NaN.")
            out_cols[col] = np.full(n_new, np.nan, dtype=float)
            continue
        out_cols[col] = _remsaple_values(values, t_old, t_new, method, spline_order=spline_order)

    # Rebuild in original column order.
    out_data: dict[str, np.ndarray] = {}
    for col in pose_df.columns:
        if frame_col is not None and col == frame_col:
            out_data[col] = np.arange(n_new, dtype=float)
        else:
            out_data[col] = out_cols[col]

    out_df = pd.DataFrame(out_data)
    if index_name is not None:
        out_df.index.name = index_name
    return out_df
