from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Literal, Set

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


SmoothingMethod = Literal["savgol", "moving_average", "lowess"]
PeakSelection = Literal["max"]
BoundaryMode = Literal["threshold", "min"]


class PainEvokedPeakDetectionError(RuntimeError):
    """Raised when pain-evoked peak detection fails due to invalid inputs or settings."""


@dataclass
class PainEvokedPeakDetectionConfig:
    """Configuration for pain-evoked peak detection on a time-indexed band-power trace."""

    # Peak search windows (in the same time unit as the input series index; typically seconds)
    start_win: tuple[float, float] = (3.5, 5.5)
    start_t: float = 4.0
    end_win: tuple[float, float] = (13.5, 15.5)
    end_t: float = 14.0

    # Smoothing
    method: SmoothingMethod = "lowess"
    lowess_frac: float | None = 0.05
    smooth_win_s: float = 0.50
    polyorder: int = 3

    # Boundary detection (threshold mode)
    slope_thresh: float | None = None
    slope_thresh_factor: float = 1.0

    # Baseline windows for derivative noise estimation and value thresholds.
    # NOTE: baseline_window is DEPRECATED. If set, it will override left/right windows.
    baseline_window: tuple[float, float] | None = None
    baseline_window_left: tuple[float, float] = (0.0, 2.5)
    baseline_window_right: tuple[float, float] = (6.5, 12.5)

    # Value thresholds for boundaries: boundary value must be < mean + n_sd * sd
    n_sd: float = -0.5
    min_sd: float = 1e-12

    # Window expansion for boundary search
    expand_left_s: float = 1.00
    expand_right_s: float = 2.50

    # Peak selection and boundary mode
    peak_selection: PeakSelection = "max"
    boundary_mode: BoundaryMode = "min"

    # Fallback behavior
    fallback_baseline_local_s: float = 0.5  # seconds around the peak if baseline window has no samples
    min_samples: int = 5

    def validate(self) -> None:
        """Validate configuration values and raise ValueError on invalid settings."""
        for name, win in [("start_win", self.start_win), ("end_win", self.end_win)]:
            if win[0] >= win[1]:
                raise ValueError(f"{name} must satisfy (t0 < t1); got {win}.")

        for name, win in [
            ("baseline_window_left", self.baseline_window_left),
            ("baseline_window_right", self.baseline_window_right),
        ]:
            if win[0] >= win[1]:
                raise ValueError(f"{name} must satisfy (t0 < t1); got {win}.")

        if self.baseline_window is not None and self.baseline_window[0] >= self.baseline_window[1]:
            raise ValueError(f"baseline_window must satisfy (t0 < t1); got {self.baseline_window}.")

        if self.smooth_win_s <= 0:
            raise ValueError(f"smooth_win_s must be > 0; got {self.smooth_win_s}.")
        if self.polyorder < 0:
            raise ValueError(f"polyorder must be >= 0; got {self.polyorder}.")
        if self.slope_thresh_factor <= 0:
            raise ValueError(f"slope_thresh_factor must be > 0; got {self.slope_thresh_factor}.")
        if self.min_sd <= 0:
            raise ValueError(f"min_sd must be > 0; got {self.min_sd}.")
        if self.expand_left_s < 0 or self.expand_right_s < 0:
            raise ValueError("expand_left_s and expand_right_s must be >= 0.")
        if self.lowess_frac is not None and not (0 < float(self.lowess_frac) <= 1.0):
            raise ValueError(f"lowess_frac must be in (0, 1]; got {self.lowess_frac}.")
        if self.min_samples < 3:
            raise ValueError("min_samples must be >= 3.")


def _time_index_to_seconds(index: pd.Index) -> np.ndarray:
    """Convert a pandas Index to float seconds.

    Supported:
      - numeric index (float/int)
      - TimedeltaIndex (converted via total_seconds)
      - DatetimeIndex (converted to seconds relative to first timestamp)
    """
    if isinstance(index, pd.TimedeltaIndex):
        return index.total_seconds().to_numpy(dtype=float)

    if isinstance(index, pd.DatetimeIndex):
        dt = (index - index[0]).total_seconds()
        return np.asarray(dt, dtype=float)

    return np.asarray(index.to_numpy(), dtype=float)


def _prepare_xy(series: pd.Series, *, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract x/y arrays, sort by time, drop NaNs, and coalesce duplicate timestamps."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"`series` must be a pandas Series, got {type(series)}.")

    x = _time_index_to_seconds(series.index)
    y = np.asarray(series.to_numpy(dtype=float), dtype=float)

    if x.size != y.size:
        raise PainEvokedPeakDetectionError("Index and values have mismatched lengths.")

    if x.size < min_samples:
        raise PainEvokedPeakDetectionError(f"Not enough samples: {x.size} < {min_samples}.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if x.size < min_samples:
        raise PainEvokedPeakDetectionError(f"Not enough finite samples: {x.size} < {min_samples}.")

    # Coalesce duplicate timestamps (np.gradient + dt estimation require strictly increasing time).
    if np.any(np.diff(x) == 0):
        tmp = pd.DataFrame({"time": x, "value": y}).groupby("time", sort=True)["value"].mean()
        x = tmp.index.to_numpy(dtype=float)
        y = tmp.to_numpy(dtype=float)

    if x.size < min_samples:
        raise PainEvokedPeakDetectionError("Too few samples after coalescing duplicates.")

    if np.any(np.diff(x) <= 0):
        raise PainEvokedPeakDetectionError("Time index must be strictly increasing after cleaning.")

    return x, y


def _estimate_dt(x: np.ndarray) -> float:
    """Estimate the sampling interval (seconds) as the median positive difference."""
    dx = np.diff(x)
    pos = dx[dx > 0]
    if pos.size == 0:
        raise PainEvokedPeakDetectionError("Non-positive or invalid sampling interval detected.")
    dt = float(np.median(pos))
    if not np.isfinite(dt) or dt <= 0:
        raise PainEvokedPeakDetectionError("Non-positive or invalid sampling interval detected.")
    return dt


def _odd_at_least(n: int) -> int:
    """Return the smallest odd integer >= n."""
    n = int(n)
    return n if (n % 2 == 1) else (n + 1)


def _savgol_window_length(n_samples: int, *, dt: float, smooth_win_s: float, polyorder: int) -> int:
    """Compute a valid Savitzky-Golay window length."""
    if n_samples < 3:
        raise PainEvokedPeakDetectionError("Not enough samples for Savitzky-Golay smoothing.")

    raw = max(5, int(round(smooth_win_s / dt)))
    win_len = _odd_at_least(raw)

    # Savitzky-Golay requires window_length > polyorder
    min_len = polyorder + 2  # slightly conservative
    win_len = _odd_at_least(max(win_len, min_len))

    max_len = n_samples if (n_samples % 2 == 1) else (n_samples - 1)
    win_len = min(win_len, max_len)

    if win_len <= polyorder:
        raise PainEvokedPeakDetectionError(
            f"Savitzky-Golay window_length ({win_len}) must be > polyorder ({polyorder})."
        )
    return win_len


def _smooth_signal(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dt: float,
    method: SmoothingMethod,
    smooth_win_s: float,
    polyorder: int,
    lowess_frac: float | None,
    min_samples: int,
) -> tuple[np.ndarray, str]:
    """Smooth the signal according to `method` and return (y_smooth, description)."""
    if method == "savgol":
        win_len = _savgol_window_length(len(y), dt=dt, smooth_win_s=smooth_win_s, polyorder=polyorder)
        y_smooth = savgol_filter(y, window_length=win_len, polyorder=polyorder, mode="interp")
        return y_smooth, f"savgol(window={win_len}, polyorder={polyorder})"

    if method == "moving_average":
        win_len = max(1, int(round(smooth_win_s / dt)))
        kernel = np.ones(win_len, dtype=float) / float(win_len)
        y_smooth = np.convolve(y, kernel, mode="same")
        return y_smooth, f"moving_average(window={win_len})"

    if method == "lowess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "LOWESS smoothing requires statsmodels. Install it or use method='savgol'/'moving_average'."
            ) from exc

        win_len = max(min_samples, int(round(smooth_win_s / dt)))
        win_len = min(win_len, len(y))

        if lowess_frac is None:
            min_frac = float(min_samples / max(len(y), 1))
            frac = float(np.clip(win_len / max(len(y), 1), min_frac, 0.99))
        else:
            frac = float(lowess_frac)

        if not (0 < frac <= 1.0):
            raise ValueError(f"lowess_frac must be in (0, 1]; got {frac}.")

        y_smooth = lowess(y, x, frac=frac, it=0, is_sorted=True, return_sorted=False)
        return y_smooth, f"lowess(frac={frac:.3f}, win~{win_len})"

    raise ValueError("Invalid smoothing method. Use 'savgol', 'moving_average', or 'lowess'.")


def _robust_mad(a: np.ndarray) -> float:
    """Median absolute deviation (MAD) around the median."""
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def _mask_for_window(x: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    t0, t1 = float(win[0]), float(win[1])
    return (x >= t0) & (x <= t1)


def _fallback_baseline_mask(
    x: np.ndarray, *, side: Literal["left", "right"], frac: float = 0.2, min_n: int = 5
) -> np.ndarray:
    n = x.size
    k = max(min_n, int(frac * n))
    mask = np.zeros(n, dtype=bool)
    if side == "left":
        mask[:k] = True
    else:
        mask[-k:] = True
    return mask


def _estimate_slope_thresholds(
    x: np.ndarray,
    dydt: np.ndarray,
    cfg: PainEvokedPeakDetectionConfig,
) -> tuple[float, float, tuple[float, float], tuple[float, float], dict[str, float]]:
    """Estimate derivative thresholds for rise/fall and return debug info."""
    if cfg.slope_thresh is not None:
        thr = float(cfg.slope_thresh)
        return (
            thr,
            thr,
            cfg.baseline_window_left,
            cfg.baseline_window_right,
            {"noise_left": float("nan"), "noise_right": float("nan")},
        )

    if cfg.baseline_window is not None:
        bmask = _mask_for_window(x, cfg.baseline_window)
        if not np.any(bmask):
            bmask = _fallback_baseline_mask(x, side="left")
        mad = _robust_mad(dydt[bmask])
        noise = 1.4826 * mad
        thr = float(cfg.slope_thresh_factor * noise)
        return thr, thr, cfg.baseline_window, cfg.baseline_window, {
            "noise_left": float(noise),
            "noise_right": float(noise),
        }

    lmask = _mask_for_window(x, cfg.baseline_window_left)
    if not np.any(lmask):
        lmask = _fallback_baseline_mask(x, side="left")

    rmask = _mask_for_window(x, cfg.baseline_window_right)
    if not np.any(rmask):
        rmask = _fallback_baseline_mask(x, side="right")

    mad_l = _robust_mad(dydt[lmask])
    mad_r = _robust_mad(dydt[rmask])

    noise_l = 1.4826 * mad_l
    noise_r = 1.4826 * mad_r

    thr_rise = float(cfg.slope_thresh_factor * noise_l)
    thr_fall = float(cfg.slope_thresh_factor * noise_r)

    return (
        thr_rise,
        thr_fall,
        cfg.baseline_window_left,
        cfg.baseline_window_right,
        {"noise_left": float(noise_l), "noise_right": float(noise_r)},
    )


def _window_indices(x: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    """Return indices for x within [t0, t1], or nearest range if empty."""
    wmask = _mask_for_window(x, win)
    idx = np.where(wmask)[0]
    if idx.size:
        return idx

    left = int(np.searchsorted(x, win[0], side="left"))
    right = int(np.searchsorted(x, win[1], side="right") - 1)
    left = int(np.clip(left, 0, x.size - 1))
    right = int(np.clip(right, 0, x.size - 1))
    lo, hi = (left, right) if left <= right else (right, left)
    return np.arange(lo, hi + 1)


def _pick_peak(y_smooth: np.ndarray, idx_range: np.ndarray, peak_selection: PeakSelection) -> int:
    if idx_range.size == 0:
        raise PainEvokedPeakDetectionError("Empty peak-search window.")
    if peak_selection == "max":
        return int(idx_range[np.argmax(y_smooth[idx_range])])
    raise NotImplementedError("Only peak_selection='max' is implemented.")


def _mean_sd_in_time_window(
    x: np.ndarray,
    y_smooth: np.ndarray,
    win: tuple[float, float],
    *,
    side_constraint: Literal["left", "right"] | None,
    ref_time: float | None,
    fallback_local_s: float,
) -> tuple[float, float, int, bool]:
    """Compute mean and SD in a time window, with optional left/right constraint and fallbacks."""
    t0, t1 = float(win[0]), float(win[1])
    mask = (x >= t0) & (x <= t1)

    if side_constraint == "left" and ref_time is not None:
        mask &= x < float(ref_time)
    elif side_constraint == "right" and ref_time is not None:
        mask &= x > float(ref_time)

    def _stats(arr: np.ndarray) -> tuple[float, float, int]:
        if arr.size == 0:
            return float("nan"), float("nan"), 0
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
        return mu, sd, int(arr.size)

    if np.any(mask):
        mu, sd, n = _stats(y_smooth[mask])
        return mu, sd, n, False

    # Fallback: local neighborhood around ref_time on the requested side.
    if ref_time is not None and fallback_local_s > 0 and side_constraint in {"left", "right"}:
        ref_time = float(ref_time)
        if side_constraint == "left":
            t0_loc = max(float(x[0]), ref_time - fallback_local_s)
            t1_loc = ref_time
        else:
            t0_loc = ref_time
            t1_loc = min(float(x[-1]), ref_time + fallback_local_s)

        mask_loc = (x >= t0_loc) & (x <= t1_loc)
        if np.any(mask_loc):
            mu, sd, n = _stats(y_smooth[mask_loc])
            return mu, sd, n, True

    # Last resort: use boundary sample.
    if side_constraint == "left":
        mu, sd, n = _stats(np.asarray([y_smooth[0]], dtype=float))
    else:
        mu, sd, n = _stats(np.asarray([y_smooth[-1]], dtype=float))
    return mu, sd, n, True


def _auc(xseg: np.ndarray, yseg: np.ndarray) -> float:
    """Area under curve using the trapezoidal rule (NumPy 1.x/2.x compatible)."""
    try:
        return float(np.trapezoid(yseg, xseg))  # NumPy 2.x
    except AttributeError:  # pragma: no cover
        return float(np.trapz(yseg, xseg))  # NumPy 1.x


def _enforce_value_thresholds(
    y_smooth: np.ndarray,
    *,
    idx_on: int,
    idx_off: int,
    left_idx: int,
    right_idx: int,
    thr_left: float,
    thr_right: float,
) -> tuple[int, int, bool, bool]:
    """Adjust boundaries to satisfy value thresholds where possible."""
    start_thr_ok = True
    offset_thr_ok = True

    if not (y_smooth[idx_on] < thr_left):
        candidates = np.where(y_smooth[left_idx : idx_on + 1] < thr_left)[0]
        if candidates.size:
            idx_on = int(candidates[-1] + left_idx)
        else:
            start_thr_ok = False

    if not (y_smooth[idx_off] < thr_right):
        candidates = np.where(y_smooth[idx_off : right_idx + 1] < thr_right)[0]
        if candidates.size:
            idx_off = int(candidates[0] + idx_off)
        else:
            offset_thr_ok = False

    return idx_on, idx_off, bool(start_thr_ok), bool(offset_thr_ok)


def _local_minima_indices(y_smooth: np.ndarray, lo: int, hi: int) -> list[int]:
    """Return indices of local minima in y_smooth[lo:hi+1], excluding endpoints."""
    lo = int(lo)
    hi = int(hi)
    if hi - lo < 2:
        return []
    out: list[int] = []
    for i in range(lo + 1, hi):
        if (y_smooth[i] <= y_smooth[i - 1]) and (y_smooth[i] < y_smooth[i + 1]):
            out.append(i)
    return out


def _boundaries_threshold(
    x: np.ndarray,
    y_smooth: np.ndarray,
    dydt: np.ndarray,
    *,
    idx_peak: int,
    left_idx: int,
    right_idx: int,
    slope_thresh_rise: float,
    slope_thresh_fall: float,
    baseline_left_win: tuple[float, float],
    baseline_right_win: tuple[float, float],
    n_sd: float,
    min_sd: float,
    fallback_local_s: float,
) -> tuple[int, int, float, float, float, float, float, float, bool, bool]:
    """Boundary detection using derivative thresholds with a half-height fallback."""
    t_peak = float(x[idx_peak])

    mu_left, sd_left, _, _ = _mean_sd_in_time_window(
        x, y_smooth, baseline_left_win,
        side_constraint="left", ref_time=t_peak, fallback_local_s=fallback_local_s
    )
    mu_right, sd_right, _, _ = _mean_sd_in_time_window(
        x, y_smooth, baseline_right_win,
        side_constraint="right", ref_time=t_peak, fallback_local_s=fallback_local_s
    )

    sd_left = max(float(sd_left if np.isfinite(sd_left) else 0.0), float(min_sd))
    sd_right = max(float(sd_right if np.isfinite(sd_right) else 0.0), float(min_sd))

    thr_left = float(mu_left) + float(n_sd) * sd_left
    thr_right = float(mu_right) + float(n_sd) * sd_right

    i = int(idx_peak) - 1
    while i > left_idx and dydt[i] > slope_thresh_rise:
        i -= 1
    idx_on = max(left_idx, i + 1)

    j = int(idx_peak) + 1
    while j < right_idx and dydt[j] < -slope_thresh_fall:
        j += 1
    idx_off = min(right_idx, j - 1)

    has_rise = bool(np.any(dydt[left_idx:idx_peak] > slope_thresh_rise))
    has_fall = bool(np.any(dydt[idx_peak:right_idx] < -slope_thresh_fall))

    v_peak = float(y_smooth[idx_peak])
    half_left = float(mu_left) + 0.5 * (v_peak - float(mu_left))
    half_right = float(mu_right) + 0.5 * (v_peak - float(mu_right))

    if (not has_rise) or idx_on >= idx_peak:
        left_candidates = np.where(y_smooth[left_idx:idx_peak] <= half_left)[0]
        idx_on = left_idx if left_candidates.size == 0 else int(left_candidates[-1] + left_idx)

    if (not has_fall) or idx_off <= idx_peak:
        right_candidates = np.where(y_smooth[idx_peak : right_idx + 1] <= half_right)[0]
        idx_off = right_idx if right_candidates.size == 0 else int(right_candidates[0] + idx_peak)

    idx_on = int(np.clip(idx_on, left_idx, idx_peak))
    idx_off = int(np.clip(idx_off, idx_peak, right_idx))

    idx_on, idx_off, start_thr_ok, offset_thr_ok = _enforce_value_thresholds(
        y_smooth,
        idx_on=idx_on,
        idx_off=idx_off,
        left_idx=left_idx,
        right_idx=right_idx,
        thr_left=thr_left,
        thr_right=thr_right,
    )

    return (
        int(idx_on),
        int(idx_off),
        float(mu_left),
        float(sd_left),
        float(mu_right),
        float(sd_right),
        float(thr_left),
        float(thr_right),
        bool(start_thr_ok),
        bool(offset_thr_ok),
    )


def _boundaries_min(
    x: np.ndarray,
    y_smooth: np.ndarray,
    *,
    idx_peak: int,
    left_idx: int,
    right_idx: int,
    baseline_left_win: tuple[float, float],
    baseline_right_win: tuple[float, float],
    n_sd: float,
    min_sd: float,
    fallback_local_s: float,
) -> tuple[int, int, float, float, float, float, float, float, bool, bool]:
    """Boundary detection using nearest local minima on each side of the peak."""
    t_peak = float(x[idx_peak])

    mu_left, sd_left, _, _ = _mean_sd_in_time_window(
        x, y_smooth, baseline_left_win,
        side_constraint="left", ref_time=t_peak, fallback_local_s=fallback_local_s
    )
    mu_right, sd_right, _, _ = _mean_sd_in_time_window(
        x, y_smooth, baseline_right_win,
        side_constraint="right", ref_time=t_peak, fallback_local_s=fallback_local_s
    )

    sd_left = max(float(sd_left if np.isfinite(sd_left) else 0.0), float(min_sd))
    sd_right = max(float(sd_right if np.isfinite(sd_right) else 0.0), float(min_sd))

    thr_left = float(mu_left) + float(n_sd) * sd_left
    thr_right = float(mu_right) + float(n_sd) * sd_right

    left_minima = _local_minima_indices(y_smooth, left_idx, idx_peak)
    right_minima = _local_minima_indices(y_smooth, idx_peak, right_idx)

    if left_minima:
        left_ok = [i for i in left_minima if float(y_smooth[i]) < thr_left]
        idx_on = int(left_ok[-1] if left_ok else left_minima[-1])
        start_thr_ok = bool(left_ok)
    else:
        seg = slice(left_idx, idx_peak + 1)
        idx_on = int(np.argmin(y_smooth[seg]) + left_idx)
        start_thr_ok = bool(float(y_smooth[idx_on]) < thr_left)

    if right_minima:
        right_ok = [i for i in right_minima if float(y_smooth[i]) < thr_right]
        idx_off = int(right_ok[0] if right_ok else right_minima[0])
        offset_thr_ok = bool(right_ok)
    else:
        seg = slice(idx_peak, right_idx + 1)
        idx_off = int(np.argmin(y_smooth[seg]) + idx_peak)
        offset_thr_ok = bool(float(y_smooth[idx_off]) < thr_right)

    idx_on = int(np.clip(idx_on, left_idx, idx_peak))
    idx_off = int(np.clip(idx_off, idx_peak, right_idx))

    return (
        int(idx_on),
        int(idx_off),
        float(mu_left),
        float(sd_left),
        float(mu_right),
        float(sd_right),
        float(thr_left),
        float(thr_right),
        bool(start_thr_ok),
        bool(offset_thr_ok),
    )


def _analyze_window(
    x: np.ndarray,
    y_smooth: np.ndarray,
    dydt: np.ndarray,
    *,
    win: tuple[float, float],
    shift_t: float,
    cfg: PainEvokedPeakDetectionConfig,
    slope_thresh_rise: float,
    slope_thresh_fall: float,
    baseline_left_win: tuple[float, float],
    baseline_right_win: tuple[float, float],
) -> dict[str, Any]:
    """Analyze a single peak-search window and return metrics."""
    idx_win = _window_indices(x, win)
    if idx_win.size == 0:
        return {"ok": False, "reason": "empty_window"}

    idx_peak = _pick_peak(y_smooth, idx_win, cfg.peak_selection)

    lbound = int(np.searchsorted(x, float(win[0]) - float(cfg.expand_left_s), side="left"))
    rbound = int(np.searchsorted(x, float(win[1]) + float(cfg.expand_right_s), side="right") - 1)

    lbound = int(np.clip(lbound, 0, x.size - 1))
    rbound = int(np.clip(rbound, 0, x.size - 1))
    if rbound <= lbound:
        lbound = int(max(0, int(np.min(idx_win))))
        rbound = int(min(x.size - 1, int(np.max(idx_win))))

    if cfg.boundary_mode == "threshold":
        (
            idx_on,
            idx_off,
            mu_left,
            sd_left,
            mu_right,
            sd_right,
            thr_left,
            thr_right,
            start_thr_ok,
            offset_thr_ok,
        ) = _boundaries_threshold(
            x,
            y_smooth,
            dydt,
            idx_peak=idx_peak,
            left_idx=lbound,
            right_idx=rbound,
            slope_thresh_rise=slope_thresh_rise,
            slope_thresh_fall=slope_thresh_fall,
            baseline_left_win=baseline_left_win,
            baseline_right_win=baseline_right_win,
            n_sd=cfg.n_sd,
            min_sd=cfg.min_sd,
            fallback_local_s=cfg.fallback_baseline_local_s,
        )
    elif cfg.boundary_mode == "min":
        (
            idx_on,
            idx_off,
            mu_left,
            sd_left,
            mu_right,
            sd_right,
            thr_left,
            thr_right,
            start_thr_ok,
            offset_thr_ok,
        ) = _boundaries_min(
            x,
            y_smooth,
            idx_peak=idx_peak,
            left_idx=lbound,
            right_idx=rbound,
            baseline_left_win=baseline_left_win,
            baseline_right_win=baseline_right_win,
            n_sd=cfg.n_sd,
            min_sd=cfg.min_sd,
            fallback_local_s=cfg.fallback_baseline_local_s,
        )
    else:  # pragma: no cover
        raise ValueError("boundary_mode must be 'threshold' or 'min'.")

    t_on, t_pk, t_off = float(x[idx_on]), float(x[idx_peak]), float(x[idx_off])
    v_on, v_pk, v_off = float(y_smooth[idx_on]), float(y_smooth[idx_peak]), float(y_smooth[idx_off])

    amplitude_edge = float(v_pk - min(v_on, v_off))
    amplitude_vs_baseline = float(v_pk - min(mu_left, mu_right))

    auc = _auc(x[idx_on : idx_off + 1], y_smooth[idx_on : idx_off + 1])
    y_rebased = y_smooth[idx_on : idx_off + 1] - float(mu_left)
    auc_rel_left = _auc(x[idx_on : idx_off + 1], y_rebased)

    rise_dt = float(max(t_pk - t_on, 0.0))
    fall_dt = float(max(t_off - t_pk, 0.0))
    mean_slope_rise = float((v_pk - v_on) / rise_dt) if rise_dt > 0 else float("nan")
    mean_slope_fall_pos = float((v_pk - v_off) / fall_dt) if fall_dt > 0 else float("nan")
    duration = float(max(t_off - t_on, 0.0))

    return {
        "onset_time": float(t_on) - float(shift_t),
        "peak_time": float(t_pk) - float(shift_t),
        "offset_time": float(t_off) - float(shift_t),
        "onset_value": float(v_on),
        "peak_value": float(v_pk),
        "offset_value": float(v_off),
        "amplitude_edge": amplitude_edge,
        "amplitude_vs_baseline": amplitude_vs_baseline,
        "auc": float(auc),
        "auc_rel_left": float(auc_rel_left),
        "mean_slope_rise": float(mean_slope_rise),
        "mean_slope_fall": float(mean_slope_fall_pos),
        "duration": float(duration),
        "baseline_left_mean": float(mu_left),
        "baseline_left_sd": float(sd_left),
        "baseline_right_mean": float(mu_right),
        "baseline_right_sd": float(sd_right),
        "thr_left": float(thr_left),
        "thr_right": float(thr_right),
        "start_thr_ok": bool(start_thr_ok),
        "offset_thr_ok": bool(offset_thr_ok),
        "ok": True,
    }


def _apply_cfg_overrides(cfg: PainEvokedPeakDetectionConfig, overrides: dict[str, Any]) -> PainEvokedPeakDetectionConfig:
    """Return a new config with overrides applied (unknown keys raise a helpful error)."""
    if not overrides:
        return cfg
    try:
        return replace(cfg, **overrides)
    except TypeError as exc:
        valid = {f.name for f in cfg.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        bad = sorted(set(overrides) - valid)
        hint = f"Unknown config fields: {bad}. Valid fields: {sorted(valid)}" if bad else "Invalid config override."
        raise TypeError(hint) from exc


def detect_evoked_peaks(
    series: pd.Series,
    *,
    cfg: PainEvokedPeakDetectionConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Detect stimulation-locked peaks in a band-power time series.

    This function supports two calling styles:

      1) Config-driven (recommended):
         detect_evoked_peaks(series, cfg=PainEvokedPeakDetectionConfig(...))

      2) Keyword overrides (backward-friendly):
         detect_evoked_peaks(series, method="savgol", boundary_mode="threshold", ...)

    Returns
    -------
    results : dict
        Nested dictionary with keys:
          - 'start': metrics for the start window
          - 'end': metrics for the end window
          - 'smoothed': pd.Series of the smoothed trace
          - 'debug': metadata and parameter tracing
    """
    base_cfg = cfg or PainEvokedPeakDetectionConfig()
    cfg = _apply_cfg_overrides(base_cfg, overrides)
    cfg.validate()

    x, y = _prepare_xy(series, min_samples=cfg.min_samples)
    dt = _estimate_dt(x)

    y_smooth, smooth_desc = _smooth_signal(
        x,
        y,
        dt=dt,
        method=cfg.method,
        smooth_win_s=cfg.smooth_win_s,
        polyorder=cfg.polyorder,
        lowess_frac=cfg.lowess_frac,
        min_samples=cfg.min_samples,
    )

    dydt = np.gradient(y_smooth, x)

    slope_thresh_rise, slope_thresh_fall, baseline_left_win, baseline_right_win, slope_debug = _estimate_slope_thresholds(
        x, dydt, cfg
    )

    start_res = _analyze_window(
        x,
        y_smooth,
        dydt,
        win=cfg.start_win,
        shift_t=cfg.start_t,
        cfg=cfg,
        slope_thresh_rise=slope_thresh_rise,
        slope_thresh_fall=slope_thresh_fall,
        baseline_left_win=baseline_left_win,
        baseline_right_win=baseline_right_win,
    )
    end_res = _analyze_window(
        x,
        y_smooth,
        dydt,
        win=cfg.end_win,
        shift_t=cfg.end_t,
        cfg=cfg,
        slope_thresh_rise=slope_thresh_rise,
        slope_thresh_fall=slope_thresh_fall,
        baseline_left_win=baseline_left_win,
        baseline_right_win=baseline_right_win,
    )

    # Match prior behavior: in "min" mode, derivative thresholds are not part of the boundary logic.
    if cfg.slope_thresh is not None:
        out_slope_rise = float(cfg.slope_thresh)
        out_slope_fall = float(cfg.slope_thresh)
    elif cfg.boundary_mode == "min":
        out_slope_rise = float("nan")
        out_slope_fall = float("nan")
    else:
        out_slope_rise = float(slope_thresh_rise)
        out_slope_fall = float(slope_thresh_fall)

    return {
        "start": {**start_res, "slope_thresh_rise": out_slope_rise, "slope_thresh_fall": out_slope_fall},
        "end": {**end_res, "slope_thresh_rise": out_slope_rise, "slope_thresh_fall": out_slope_fall},
        "smoothed": pd.Series(y_smooth, index=pd.Index(x, name="time"), name="value"),
        "debug": {
            "dt": float(dt),
            "smoothing": smooth_desc,
            "baseline_window_left": baseline_left_win,
            "baseline_window_right": baseline_right_win,
            "baseline_window_deprecated": cfg.baseline_window,
            "computed_slope_thresh_rise": float(slope_thresh_rise),
            "computed_slope_thresh_fall": float(slope_thresh_fall),
            **slope_debug,
            "params": {
                "n_sd": float(cfg.n_sd),
                "min_sd": float(cfg.min_sd),
                "expand_left_s": float(cfg.expand_left_s),
                "expand_right_s": float(cfg.expand_right_s),
                "boundary_mode": str(cfg.boundary_mode),
            },
        },
    }


def _flatten_result_dict(
    d: Dict[str, Any],
    *,
    sep: str = "-",
    exclude_keys: Iterable[str] = ("smoothed",),
    include_smoothed: bool = False,
) -> Dict[str, Any]:
    """Flatten a nested dict into a single-level dict with keys joined by `sep`."""
    exclude: Set[str] = set(exclude_keys or ())
    flat: Dict[str, Any] = {}

    if include_smoothed and "smoothed" in d:
        flat["smoothed"] = d["smoothed"]

    def _recurse(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if prefix == "" and k in exclude:
                    continue
                new_key = f"{prefix}{sep}{k}" if prefix else str(k)
                _recurse(v, new_key)
            return

        if isinstance(obj, (pd.Series, pd.DataFrame, np.ndarray)):
            return

        flat[prefix] = obj

    _recurse(d, "")
    return flat


def apply_detect_evoked_peaks_to_df(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    detect_func: Callable[..., Dict[str, Any]] = detect_evoked_peaks,
    sep: str = "-",
    exclude_keys: Iterable[str] = ("smoothed",),
    include_smoothed: bool = False,
    passthrough_errors: bool = True,
    **detect_kwargs: Any,
) -> pd.DataFrame:
    """Apply an evoked-peak detection function row-wise to a DataFrame of time-series values.

    This is a thin, reusable wrapper that:
      1) Iterates over rows in `df`.
      2) Expects each row to contain a time-indexed `pd.Series` in `value_col`.
      3) Calls `detect_func(series, **detect_kwargs)` to compute evoked-peak features.
      4) Flattens the (possibly nested) result dict into a 1-level mapping with keys joined by `sep`.
      5) Joins the flattened feature columns back onto the original DataFrame.

    The goal is to keep paradigm-specific peak logic inside `detect_func`, while centralizing
    the common "apply + flatten + error-handling" mechanics in one place.

    Parameters
    ----------
    df:
        Input DataFrame. Must contain `value_col`. Each entry of `df[value_col]` should be a
        `pd.Series` whose index represents time (typically seconds) and whose values represent
        the band-power (or similar) trace to analyze.
    value_col:
        Name of the column containing per-row `pd.Series` objects to be passed into `detect_func`.
    detect_func:
        Callable implementing the detection logic, e.g. `detect_evoked_peaks`. It must accept
        a `pd.Series` as its first positional argument and return a dict (optionally nested)
        containing scalar features (and optionally debug objects like a smoothed trace).
    sep:
        Separator used when flattening nested dict outputs into column names.
        Example: {"start": {"peak_time": 1.2}} -> column "start-peak_time" if sep="-".
    exclude_keys:
        Top-level keys in the result dict to exclude from flattening. This is typically used to
        avoid storing large array-like objects (e.g., "smoothed" as a full `pd.Series`) in the
        output DataFrame.
    include_smoothed:
        If True and the result dict contains a "smoothed" entry, keep it as an object column
        in the output DataFrame (one `pd.Series` per row). If False (default), "smoothed" is
        excluded by default via `exclude_keys`.
        Note: object columns can be inconvenient for serialization (CSV/Parquet) and downstream
        stats; enable only when needed for QC/debugging.
    passthrough_errors:
        Controls error behavior in batch processing:
          - True (default): catch per-row exceptions, continue processing other rows, and record
            "error" (bool) and "error_msg" (str) columns for that row.
          - False: raise immediately on the first error.
    **detect_kwargs:
        Additional keyword arguments forwarded directly to `detect_func`. This allows you to tune
        detection parameters (e.g., smoothing method, windows, thresholds) without changing this
        wrapper's signature.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with additional flattened feature columns appended.
        When `passthrough_errors=True`, the output includes per-row "error" and "error_msg"
        columns (unless `detect_func` already provides them).

    Raises
    ------
    KeyError
        If `value_col` is not present in `df`.
    TypeError
        If any row in `value_col` is not a `pd.Series` (when the row is processed).
    Exception
        Propagates exceptions from `detect_func` when `passthrough_errors=False`.
    """
    
    if value_col not in df.columns:
        raise KeyError(f"`{value_col}` column not found in DataFrame.")

    flattened_rows: list[Dict[str, Any]] = []
    for idx, series in df[value_col].items():
        try:
            if not isinstance(series, pd.Series):
                raise TypeError(f"Row {idx}: `{value_col}` must be a pd.Series, got {type(series)}.")
            res = detect_func(series, **detect_kwargs)
            flat = _flatten_result_dict(
                res, sep=sep, exclude_keys=exclude_keys, include_smoothed=include_smoothed
            )
            if passthrough_errors:
                flat.setdefault("error", False)
                flat.setdefault("error_msg", "")
        except Exception as exc:
            if not passthrough_errors:
                raise
            flat = {"error": True, "error_msg": str(exc)}
        flattened_rows.append(flat)

    flat_df = pd.DataFrame(flattened_rows, index=df.index)
    return df.join(flat_df)
