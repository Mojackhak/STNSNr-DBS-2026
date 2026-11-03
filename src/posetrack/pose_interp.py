"""
Interpolate high-error 3D pose points in Anipose pose-3d CSV files.

Assumptions about CSV columns:
- For each bodypart `bp`, columns exist:
  `{bp}_x`, `{bp}_y`, `{bp}_z`, `{bp}_error`, `{bp}_score`
- Interpolated points are marked by setting `{bp}_error` to 99.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import Akima1DInterpolator, interp1d

from .track_io import load_csv_3d


def create_interpolator(
    valid_indices: np.ndarray,
    valid_points: np.ndarray,
    method: str,
) -> Callable[[int], float]:
    """
    Create a 1D interpolator with extrapolation.

    Parameters
    ----------
    valid_indices:
        1D array of integer indices (positions) where points are valid.
    valid_points:
        1D array of values corresponding to valid_indices.
    method:
        'akima' or any scipy interp1d kind (e.g., 'linear', 'nearest', 'cubic').

    Returns
    -------
    A function f(i)->float for interpolated/extrapolated value at position i.
    """
    valid_indices = np.asarray(valid_indices, dtype=int)
    valid_points = np.asarray(valid_points, dtype=float)

    if valid_indices.size < 2:
        # Not enough points to build an interpolator; caller should skip interpolation.
        return lambda i: float("nan")

    if method == "akima":
        ak = Akima1DInterpolator(valid_indices, valid_points)

        def f(i: int) -> float:
            i = int(i)
            if i < valid_indices[0]:
                slope = (valid_points[1] - valid_points[0]) / (valid_indices[1] - valid_indices[0])
                return float(valid_points[0] + slope * (i - valid_indices[0]))
            if i > valid_indices[-1]:
                slope = (valid_points[-1] - valid_points[-2]) / (valid_indices[-1] - valid_indices[-2])
                return float(valid_points[-1] + slope * (i - valid_indices[-1]))
            return float(ak(i))

        return f

    f0 = interp1d(valid_indices, valid_points, kind=method, fill_value="extrapolate", bounds_error=False)

    def f(i: int) -> float:
        return float(f0(int(i)))

    return f


def interpolate_bodypart(
    data_subset: np.ndarray,
    valid_indices: np.ndarray,
    gap: float,
    interpolators: dict[str, Callable[[int], float]],
) -> np.ndarray:
    """
    Interpolate one bodypart's [x, y, z, error, score] array in-place-like.

    Logic preserved:
    - Only indices NOT in valid_indices are candidates.
    - Skip interpolation if BOTH nearest-valid gaps (lower and upper) exceed `gap`.
    - Mark interpolated points by setting error=99.
    - If score is NaN, set it to 0.99 for interpolated points.
    """
    out = np.array(data_subset, copy=True, dtype=float)
    n = out.shape[0]

    valid_indices = np.asarray(valid_indices, dtype=int)
    if valid_indices.size < 2:
        return out

    valid_indices.sort()
    valid_set = set(valid_indices.tolist())

    for i in range(n):
        if i in valid_set:
            continue

        pos = bisect_left(valid_indices, i)
        lower = valid_indices[pos - 1] if pos > 0 else None
        upper = valid_indices[pos] if pos < valid_indices.size else None

        lower_gap = (i - lower) if lower is not None else np.inf
        upper_gap = (upper - i) if upper is not None else np.inf

        if lower_gap > gap and upper_gap > gap:
            continue

        out[i, 0] = interpolators["x"](i)
        out[i, 1] = interpolators["y"](i)
        out[i, 2] = interpolators["z"](i)
        out[i, 3] = 99.0  # mark as interpolated

        if np.isnan(out[i, 4]):
            out[i, 4] = 0.99

    return out


def interpolate_errors_parallel(
    csv_path: str | Path,
    interp_method: str = "akima",
    threshold: float = 80,
    gap: float = np.inf,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Interpolate 3D points for all bodyparts in one Anipose pose-3d CSV.

    A point is considered "valid" if `{bp}_error < threshold`.
    Only invalid points are considered for interpolation.

    Returns
    -------
    data, metadata
    """
    csv_path = Path(csv_path)
    data, metadata, *_ = load_csv_3d(str(csv_path), config_file_path=None, bp_side=False, cs_info=True, cam_info=True)
    bodyparts: list[str] = list(metadata["bodyparts"])

    def _process_one(bp: str) -> tuple[str, np.ndarray]:
        cols = [f"{bp}_x", f"{bp}_y", f"{bp}_z", f"{bp}_error", f"{bp}_score"]
        subset = data[cols].to_numpy(dtype=float, copy=True)

        valid_mask = subset[:, 3] < float(threshold)
        valid_idx = np.flatnonzero(valid_mask).astype(int)

        if valid_idx.size < 2:
            return bp, subset

        interpolators = {
            "x": create_interpolator(valid_idx, subset[valid_idx, 0], interp_method),
            "y": create_interpolator(valid_idx, subset[valid_idx, 1], interp_method),
            "z": create_interpolator(valid_idx, subset[valid_idx, 2], interp_method),
        }
        return bp, interpolate_bodypart(subset, valid_idx, float(gap), interpolators)

    results = Parallel(n_jobs=n_jobs)(delayed(_process_one)(bp) for bp in bodyparts)

    for bp, arr in results:
        data[[f"{bp}_x", f"{bp}_y", f"{bp}_z", f"{bp}_error", f"{bp}_score"]] = arr

    return data, metadata


def summarize_interpolation(data: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """
    Summarize interpolation statistics per bodypart based on `{bp}_error == 99`.
    """
    bodyparts: list[str] = list(metadata["bodyparts"])
    n = len(data)

    rows: list[dict[str, Any]] = []
    for bp in bodyparts:
        err_col = f"{bp}_error"
        if err_col not in data.columns:
            continue

        interpolated_idx = np.flatnonzero(data[err_col].to_numpy() == 99)
        total = int(interpolated_idx.size)
        percent = (100.0 * total / n) if n > 0 else 0.0

        if total == 0:
            longest = 0
        else:
            gaps = np.diff(interpolated_idx)
            split_points = np.where(gaps != 1)[0] + 1
            chunks = np.split(interpolated_idx, split_points)
            longest = int(max((len(c) for c in chunks), default=0))

        rows.append(
            {
                "bodypart": bp,
                "total_interpolated": total,
                "percent_interpolated": percent,
                "longest_continuous": longest,
            }
        )

    return pd.DataFrame(rows).set_index("bodypart")


def interp_anipose_dir(
    source_dir: str | Path,
    target_dir: str | Path | None = None,
    interp_method: str = "akima",
    suffix: str = "_interpolated",
    threshold: float = 80,
    gap: float = np.inf,
    save_evaluation: bool = False,
    n_jobs: int = -1,
) -> list[Path]:
    """
    Process all pose-3d CSV files in a directory.

    Output structure (inside target_dir):
    - pose-3d-interpolated/<stem>{suffix}.csv
    - pose-3d-evaluation/<stem>_evaluation.csv (optional)
    """
    source_dir = Path(source_dir)
    if target_dir is None:
        target_dir = source_dir
    target_dir = Path(target_dir)

    out_pose_dir = target_dir / "pose-3d-interpolated"
    out_pose_dir.mkdir(parents=True, exist_ok=True)

    out_eval_dir = target_dir / "pose-3d-evaluation"
    if save_evaluation:
        out_eval_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(source_dir.glob("*.csv"))  # non-recursive to avoid reprocessing outputs
    outputs: list[Path] = []

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")

        data, metadata = interpolate_errors_parallel(
            csv_file,
            interp_method=interp_method,
            threshold=threshold,
            gap=gap,
            n_jobs=n_jobs,
        )

        out_csv = out_pose_dir / f"{csv_file.stem}{suffix}.csv"
        data.to_csv(out_csv, index=False)
        outputs.append(out_csv)

        if save_evaluation:
            eval_df = summarize_interpolation(data, metadata)
            out_eval = out_eval_dir / f"{csv_file.stem}_evaluation.csv"
            eval_df.to_csv(out_eval, index=True)

    return outputs
