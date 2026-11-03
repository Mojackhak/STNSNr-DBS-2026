"""sync.py

Synchronization utilities for aligning LFP (or other time series) to behavioral audio beeps.

This module is designed for the following workflow:

1) Detect beep times in an audio track (e.g., video audio).
2) Detect corresponding markers in the LFP recording:
   - either from digital tags (sample indices from a tag column / annotations),
   - or from peaks in a chosen analog channel.
3) Fit a linear mapping between audio time (seconds) and signal sample index (samples):

       sample_index ~= sfreq_est * audio_time + intercept_samples

   where:
     - sfreq_est is the estimated sampling frequency (samples / second)
     - lag_s = intercept_samples / sfreq_est is the lag in seconds

4) Apply the lag by cutting or zero-padding the *signal* to align signal time zero to audio time zero:
   - lag_s > 0: signal leads audio -> cut the first lag_s seconds of the signal
   - lag_s < 0: signal lags audio -> pad -lag_s seconds of zeros at the beginning

Optionally, correct sampling frequency drift using the regression slope (sfreq_est).

All code is written to be:
- deterministic (no randomness),
- reasonably robust to dtype quirks (hex strings, floats),
- usable in both interactive and batch pipelines (plots are optional).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import linregress, t as t_dist

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import mne
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class SyncFitResult:
    """Result of fitting audio times to signal sample indices."""
    lag_s: float
    sfreq_hz: float
    rmse_ms: float
    r2: float
    p_value: float
    intercept_samples: float
    n_events: int


def load_audio_mono(audio_path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float array.

    This function tries `librosa` first (supports many formats). If it is unavailable,
    it falls back to `scipy.io.wavfile` (WAV only).

    Parameters
    ----------
    audio_path:
        Path to an audio file.
    sr:
        Target sampling rate. If None, keep original sampling rate (librosa behavior).

    Returns
    -------
    audio:
        1D float array in range approximately [-1, 1] (librosa) or raw scaled (wavfile fallback).
    sr:
        Sampling rate in Hz.
    """
    try:
        import librosa  # type: ignore

        audio, audio_sr = librosa.load(audio_path, sr=sr, mono=True)
        return audio.astype(float, copy=False), int(audio_sr)
    except Exception as exc_librosa:
        logger.debug("librosa load failed (%s). Falling back to scipy.io.wavfile.", exc_librosa)

    try:
        from scipy.io import wavfile

        audio_sr, audio = wavfile.read(audio_path)
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Convert integer PCM to float in [-1, 1]
        if np.issubdtype(audio.dtype, np.integer):
            max_val = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(float) / max_val
        else:
            audio = audio.astype(float)
        return audio, int(audio_sr)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file: {audio_path}") from exc


def _validate_1d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    return x


def detect_peaks(
    x: np.ndarray,
    sfreq_hz: float,
    search_range_samples: tuple[int, int] | None,
    n_peaks: int | None,
    min_distance_s: float = 1.0,
    *,
    use_abs: bool = True,
    plot: bool = False,
    title: str = "",
    **find_peaks_kwargs: Any,
) -> np.ndarray:
    """Detect peaks and return peak sample indices (absolute indices).

    Parameters
    ----------
    x:
        1D signal.
    sfreq_hz:
        Sampling frequency in Hz.
    search_range_samples:
        (start, stop) sample indices to search within `x`. If None, use full range.
    n_peaks:
        If provided, keep only the top-N peaks by amplitude (after abs if use_abs=True).
    min_distance_s:
        Minimum distance between peaks in seconds.
    use_abs:
        If True, peak detection is done on abs(x) to handle biphasic beeps.
    plot:
        If True, plot the detection window and selected peaks.
    title:
        Plot title.
    **find_peaks_kwargs:
        Forwarded to scipy.signal.find_peaks (e.g., prominence=..., height=...).

    Returns
    -------
    peaks_abs:
        1D array of peak indices in samples (sorted in ascending time).
    """
    x = _validate_1d(np.asarray(x, dtype=float), "x")
    if sfreq_hz <= 0:
        raise ValueError(f"sfreq_hz must be > 0, got {sfreq_hz}")

    if search_range_samples is None:
        start, stop = 0, x.shape[0]
    else:
        start, stop = int(search_range_samples[0]), int(search_range_samples[1])

    start = max(0, start)
    stop = min(x.shape[0], stop)
    if stop <= start:
        raise ValueError(f"Invalid search_range_samples={search_range_samples} for x length {x.shape[0]}")

    x_slice = x[start:stop]
    x_detect = np.abs(x_slice) if use_abs else x_slice

    distance = max(1, int(round(min_distance_s * sfreq_hz)))
    peaks, props = find_peaks(x_detect, distance=distance, **find_peaks_kwargs)

    if peaks.size == 0:
        raise ValueError("No peaks detected. Consider relaxing thresholds or expanding the search range.")

    # Rank by amplitude in the detection signal (always defined).
    scores = x_detect[peaks]
    order = np.argsort(scores)[::-1]  # descending
    peaks = peaks[order]

    if n_peaks is not None:
        n_peaks = int(n_peaks)
        if n_peaks <= 0:
            raise ValueError(f"n_peaks must be positive, got {n_peaks}")
        peaks = peaks[:n_peaks]

    # Convert to absolute indices and sort in time.
    peaks_abs = np.sort(peaks + start)

    if plot:
        import matplotlib.pyplot as plt  # local import

        t = np.arange(start, stop) / sfreq_hz
        plt.figure(figsize=(16, 4))
        plt.plot(t, x_detect, color="black", linewidth=1)
        plt.plot(peaks_abs / sfreq_hz, x_detect[peaks_abs - start], "x", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude" if use_abs else "Signal")
        plt.title(title or "Peak detection")
        plt.tight_layout()
        plt.show()

    return peaks_abs


def detect_peaks_times(
    x: np.ndarray,
    sfreq_hz: float,
    search_range_samples: tuple[int, int] | None,
    n_peaks: int | None,
    min_distance_s: float = 1.0,
    *,
    use_abs: bool = True,
    plot: bool = False,
    title: str = "",
    **find_peaks_kwargs: Any,
) -> np.ndarray:
    """Detect peaks and return peak times in seconds (sorted)."""
    peaks_abs = detect_peaks(
        x=x,
        sfreq_hz=sfreq_hz,
        search_range_samples=search_range_samples,
        n_peaks=n_peaks,
        min_distance_s=min_distance_s,
        use_abs=use_abs,
        plot=plot,
        title=title,
        **find_peaks_kwargs,
    )
    return peaks_abs / sfreq_hz


def fit_audio_to_signal_samples(
    tag_sample_indices: np.ndarray,
    audio_peak_times_s: np.ndarray,
    nominal_sfreq_hz: float,
    *,
    correct_sfreq: bool = True,
    plot: bool = False,
    title: str = "Audio ↔ Signal sync fit",
) -> tuple[SyncFitResult, "Figure | None"]:
    """Fit lag (and optionally sampling frequency) using linear regression.

    Model:
        tag_sample_indices ~= sfreq_est * audio_peak_times_s + intercept_samples

    Parameters
    ----------
    tag_sample_indices:
        Sample indices (in the *signal* timeline) for each tag/marker.
    audio_peak_times_s:
        Times (seconds) of each beep/marker in the *audio* timeline.
    nominal_sfreq_hz:
        Nominal signal sampling frequency (used if correct_sfreq=False).
    correct_sfreq:
        If True, estimate sfreq from the regression slope. If False, keep nominal_sfreq_hz.
    plot:
        If True, return a matplotlib figure visualizing the fit.
    title:
        Figure title.

    Returns
    -------
    result:
        SyncFitResult with lag and quality metrics.
    fig:
        Optional matplotlib Figure (or None if plot=False).
    """
    y = _validate_1d(np.asarray(tag_sample_indices, dtype=float), "tag_sample_indices")
    x = _validate_1d(np.asarray(audio_peak_times_s, dtype=float), "audio_peak_times_s")

    if y.size != x.size:
        raise ValueError(f"tag_sample_indices and audio_peak_times_s must have same length, got {y.size} vs {x.size}")
    if y.size < 2:
        raise ValueError("Need at least 2 events to estimate lag (and drift).")

    if correct_sfreq:
        lr = linregress(x, y)
        sfreq_est = float(lr.slope)
        intercept = float(lr.intercept)
        r_value = float(lr.rvalue)
        p_value = float(lr.pvalue)
    else:
        if nominal_sfreq_hz <= 0:
            raise ValueError(f"nominal_sfreq_hz must be > 0, got {nominal_sfreq_hz}")
        sfreq_est = float(nominal_sfreq_hz)
        intercept = float(np.mean(y - sfreq_est * x))
        pred = sfreq_est * x + intercept
        r_value = float(np.corrcoef(pred, y)[0, 1]) if y.size >= 2 else float("nan")

        # Approximate p-value from correlation coefficient (two-sided)
        df = int(y.size - 2)
        if df > 0 and np.isfinite(r_value) and abs(r_value) < 1:
            t_stat = r_value * math.sqrt(df / (1 - r_value**2))
            p_value = float(2 * t_dist.sf(abs(t_stat), df))
        else:
            p_value = float("nan")

    if not np.isfinite(sfreq_est) or sfreq_est <= 0:
        raise ValueError(f"Estimated sfreq is invalid: {sfreq_est}")

    lag_s = intercept / sfreq_est
    pred = sfreq_est * x + intercept
    residuals_s = (pred - y) / sfreq_est
    rmse_ms = float(np.sqrt(np.mean(residuals_s**2)) * 1e3)
    r2 = float(r_value**2) if np.isfinite(r_value) else float("nan")

    fig = None
    if plot:
        import matplotlib.pyplot as plt  # local import

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, s=20, alpha=0.8, label="Events")
        ax.plot(x, pred, linewidth=2, label="Fit")
        ax.set_xlabel("Audio time (s)")
        ax.set_ylabel("Signal sample index (samples)")
        ax.set_title(title)
        text = f"lag = {lag_s:.3f} s\nsfreq = {sfreq_est:.3f} Hz\nRMSE = {rmse_ms:.3f} ms\nR² = {r2:.4f}\np = {p_value:.3e}"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.legend()
        fig.tight_layout()

    result = SyncFitResult(
        lag_s=float(lag_s),
        sfreq_hz=float(sfreq_est),
        rmse_ms=rmse_ms,
        r2=r2,
        p_value=p_value,
        intercept_samples=float(intercept),
        n_events=int(y.size),
    )
    return result, fig


def apply_lag_to_array(
    data: np.ndarray,
    sfreq_hz: float,
    lag_s: float,
    *,
    time_axis: int = -1,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Apply lag to an array by cutting/padding along the time axis.

    Parameters
    ----------
    data:
        Array containing time series. Can be 1D or multi-D.
    sfreq_hz:
        Sampling frequency in Hz used to convert seconds -> samples.
    lag_s:
        Lag in seconds. Positive cuts from start, negative pads at start.
    time_axis:
        Axis that represents time.
    pad_value:
        Fill value for padded samples.

    Returns
    -------
    shifted:
        Array with adjusted length along time_axis.
    """
    if sfreq_hz <= 0:
        raise ValueError(f"sfreq_hz must be > 0, got {sfreq_hz}")

    arr = np.asarray(data)
    if arr.ndim == 0:
        raise ValueError("data must be at least 1D")

    # Normalize axis
    time_axis = int(time_axis)
    if time_axis < 0:
        time_axis = arr.ndim + time_axis
    if time_axis < 0 or time_axis >= arr.ndim:
        raise ValueError(f"Invalid time_axis={time_axis} for data.ndim={arr.ndim}")

    n_times = arr.shape[time_axis]
    shift_samples = int(round(lag_s * sfreq_hz))

    if shift_samples == 0:
        return arr.copy()

    moved = np.moveaxis(arr, time_axis, -1)  # time is last
    if shift_samples > 0:
        if shift_samples >= n_times:
            raise ValueError(f"Cannot cut {shift_samples} samples from length {n_times}")
        moved = moved[..., shift_samples:]
    else:
        pad_n = -shift_samples
        pad_shape = list(moved.shape)
        pad_shape[-1] = pad_n
        pad = np.full(pad_shape, pad_value, dtype=moved.dtype)
        moved = np.concatenate([pad, moved], axis=-1)

    shifted = np.moveaxis(moved, -1, time_axis)
    return shifted


def apply_lag_to_dataframe(
    df: pd.DataFrame,
    sfreq_hz: float,
    lag_s: float,
    *,
    pad_values: dict[str, Any] | None = None,
    recompute_time: bool = True,
    time_col: str = "Time",
    time_bjs_col: str = "Time_BJS",
    start_time_bjs: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Apply lag to a DataFrame by shifting rows (pad/cut), optionally recomputing time columns.

    Notes
    -----
    - For padding, values are filled using `pad_values` where provided; otherwise:
        - numeric columns -> 0
        - non-numeric columns -> NA
    - If `recompute_time=True`, the function overwrites/creates `time_col` and `time_bjs_col`.
      `time_bjs_col` requires a valid `start_time_bjs` (or a parseable value in the first row).

    Parameters
    ----------
    df:
        Input DataFrame whose rows correspond to time samples.
    sfreq_hz:
        Sampling frequency (used for row shift and time reconstruction).
    lag_s:
        Lag (seconds). Positive cuts rows at the beginning; negative pads rows at the beginning.
    pad_values:
        Column-specific fill values for padded rows.
    recompute_time:
        If True, regenerate `time_col` and `time_bjs_col`.
    time_col:
        Name of the time-in-seconds column.
    time_bjs_col:
        Name of the absolute time column (Beijing time).
    start_time_bjs:
        Start time corresponding to the first row of the *original* df (before shifting).
        If None, the function tries to parse df[time_bjs_col].iloc[0].

    Returns
    -------
    df_shifted:
        Shifted DataFrame.
    """
    if sfreq_hz <= 0:
        raise ValueError(f"sfreq_hz must be > 0, got {sfreq_hz}")

    shift_samples = int(round(lag_s * sfreq_hz))
    pad_values = pad_values or {}

    if shift_samples > 0:
        df_shifted = df.iloc[shift_samples:].copy()
        df_shifted.reset_index(drop=True, inplace=True)
    elif shift_samples < 0:
        pad_n = -shift_samples
        pad_dict: dict[str, Any] = {}
        for col in df.columns:
            if col in pad_values:
                pad_dict[col] = pad_values[col]
            elif col in (time_col, time_bjs_col):
                pad_dict[col] = np.nan
            elif pd.api.types.is_numeric_dtype(df[col].dtype):
                pad_dict[col] = 0
            else:
                pad_dict[col] = pd.NA
        pad_df = pd.DataFrame({col: [pad_dict[col]] * pad_n for col in df.columns})
        df_shifted = pd.concat([pad_df, df], ignore_index=True)
    else:
        df_shifted = df.copy()

    if recompute_time:
        n = len(df_shifted)
        df_shifted[time_col] = np.arange(n, dtype=float) / float(sfreq_hz)

        if start_time_bjs is None and time_bjs_col in df.columns:
            try:
                start_time_bjs = pd.to_datetime(df[time_bjs_col].iloc[0])
            except Exception:
                start_time_bjs = None

        if start_time_bjs is not None:
            new_start = start_time_bjs + pd.to_timedelta(lag_s, unit="s")
            df_shifted[time_bjs_col] = new_start + pd.to_timedelta(df_shifted[time_col], unit="s")

    return df_shifted


def apply_lag_to_raw(
    raw: "mne.io.BaseRaw",
    lag_s: float,
    *,
    new_sfreq_hz: float | None = None,
    pad_value: float = 0.0,
    scale_annotations_for_sfreq: bool = True,
) -> "mne.io.Raw":
    """Apply lag (and optionally sfreq correction) to an MNE Raw object.

    This function does NOT resample. If `new_sfreq_hz` differs from the current `raw.info['sfreq']`,
    it only updates the metadata and (optionally) scales existing annotations so they remain aligned
    to sample indices.

    Annotation handling
    -------------------
    - If sfreq changes: onsets/durations are scaled by (sfreq_old / sfreq_new)
      so that an event at sample index i stays at time i/sfreq_new.
    - Then lag is applied: onset_new = onset_scaled - lag_s
      (this matches the pad/cut convention used throughout this module).

    Parameters
    ----------
    raw:
        Input raw object.
    lag_s:
        Lag in seconds. Positive cuts from the start; negative pads zeros at start.
    new_sfreq_hz:
        If provided, use this as the new sampling frequency.
    pad_value:
        Value used for zero-padding when lag_s < 0.
    scale_annotations_for_sfreq:
        If True, scale annotations when sfreq changes (recommended when annotations are sample-index-based).

    Returns
    -------
    raw_shifted:
        A new RawArray containing shifted data and updated metadata/annotations.
    """
    import mne  # local import

    sfreq_old = float(raw.info["sfreq"])
    sfreq_new = float(new_sfreq_hz) if new_sfreq_hz is not None else sfreq_old
    if sfreq_new <= 0:
        raise ValueError(f"new_sfreq_hz must be > 0, got {sfreq_new}")

    data = raw.get_data()  # (n_ch, n_times)
    data_shifted = apply_lag_to_array(data, sfreq_new, lag_s, time_axis=-1, pad_value=pad_value)

    info_new = mne.create_info(
        ch_names=list(raw.ch_names),
        sfreq=sfreq_new,
        ch_types=raw.get_channel_types(),
    )
    raw_shifted = mne.io.RawArray(data_shifted, info_new)

    # Copy & adjust annotations if present
    ann = raw.annotations
    if ann is not None and len(ann) > 0:
        onsets = ann.onset.astype(float)
        durations = ann.duration.astype(float)
        descriptions = np.array(ann.description, dtype=object)

        if scale_annotations_for_sfreq and sfreq_new != sfreq_old:
            factor = sfreq_old / sfreq_new
            onsets = onsets * factor
            durations = durations * factor

        onsets = onsets - float(lag_s)

        max_time = (data_shifted.shape[1] - 1) / sfreq_new if data_shifted.shape[1] > 0 else 0.0
        keep = (onsets >= 0) & (onsets <= max_time)

        if keep.any():
            ann_new = mne.Annotations(
                onset=onsets[keep],
                duration=durations[keep],
                description=list(descriptions[keep]),
            )
            raw_shifted.set_annotations(ann_new)

    return raw_shifted


# -------------------------------------------------------------------------
# Backwards-compatible wrappers (keep old API used by existing notebooks)
# -------------------------------------------------------------------------

def find_peaks_custom(
    signal: np.ndarray,
    sf: float,
    signal_peak_range: tuple[int, int],
    signal_peak_num: int,
    tap_dist: float,
    title: str,
    **kwargs: Any,
) -> np.ndarray:
    """Legacy wrapper: detect peaks and return peak times in seconds."""
    return detect_peaks_times(
        x=signal,
        sfreq_hz=float(sf),
        search_range_samples=signal_peak_range,
        n_peaks=int(signal_peak_num),
        min_distance_s=float(tap_dist),
        plot=True,
        title=title,
        **kwargs,
    )


def sync_signal_tag(
    signal_t: np.ndarray,
    t: np.ndarray,
    sr: float,
    correct_sf: bool = True,
) -> tuple[dict[str, float], "Figure | None"]:
    """Legacy wrapper matching the original return type (dict + figure)."""
    tag_samples = np.asarray(signal_t, dtype=float) * float(sr)
    result, fig = fit_audio_to_signal_samples(
        tag_sample_indices=tag_samples,
        audio_peak_times_s=np.asarray(t, dtype=float),
        nominal_sfreq_hz=float(sr),
        correct_sfreq=bool(correct_sf),
        plot=True,
        title="Sync fit",
    )
    lag_dict = {
        "Lag": float(result.lag_s),
        "Sr": float(result.sfreq_hz),
        "RMSE": float(result.rmse_ms),
        "R²": float(result.r2),
        "p-value": float(result.p_value),
    }
    return lag_dict, fig


def sync_by_lag(signal: np.ndarray, sf: float, lag: float) -> np.ndarray:
    """Legacy wrapper applying lag to a signal whose time axis is the first axis."""
    shifted = apply_lag_to_array(signal, float(sf), float(lag), time_axis=0, pad_value=0.0)
    return np.asarray(shifted).squeeze()


def sync_df_by_lag(df: pd.DataFrame, sf: float, lag: float) -> pd.DataFrame:
    """Legacy wrapper: shift rows only (no time recomputation)."""
    return apply_lag_to_dataframe(
        df=df,
        sfreq_hz=float(sf),
        lag_s=float(lag),
        recompute_time=False,
    )


def sync_taps(
    audio: np.ndarray,
    sr: int,
    signal: np.ndarray,
    sf: int,
    audio_peak_range: tuple[int, int],
    audio_peak_num: int,
    signal_peak_range: tuple[int, int],
    signal_peak_num: int,
    tap_dist: float = 1.0,
    save_path: str | None = None,
    correct_sf: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, float], "Figure | None"]:
    """Legacy wrapper to synchronize audio taps and signal taps.

    Notes
    -----
    - Peak detection is performed on the mean across channels (if multi-channel),
      but the lag is applied to the full `signal` array.
    """
    audio_arr = np.asarray(audio)
    sig_arr = np.asarray(signal)

    # Flatten to 1D for peak detection
    if audio_arr.ndim > 1:
        audio_1d = audio_arr.mean(axis=-1)
    else:
        audio_1d = audio_arr

    if sig_arr.ndim > 1:
        signal_1d = sig_arr.mean(axis=-1)
    else:
        signal_1d = sig_arr

    peaks_audio_t = detect_peaks_times(
        x=audio_1d,
        sfreq_hz=float(sr),
        search_range_samples=audio_peak_range,
        n_peaks=int(audio_peak_num),
        min_distance_s=float(tap_dist),
        use_abs=True,
        plot=True,
        title="Find peaks in audio",
        **kwargs,
    )

    peaks_signal_samp = detect_peaks(
        x=signal_1d,
        sfreq_hz=float(sf),
        search_range_samples=signal_peak_range,
        n_peaks=int(signal_peak_num),
        min_distance_s=float(tap_dist),
        use_abs=True,
        plot=True,
        title="Find peaks in signal",
        **kwargs,
    )

    result, fig = fit_audio_to_signal_samples(
        tag_sample_indices=peaks_signal_samp.astype(float),
        audio_peak_times_s=peaks_audio_t.astype(float),
        nominal_sfreq_hz=float(sf),
        correct_sfreq=bool(correct_sf),
        plot=True,
        title="Sync fit",
    )

    lag_dict = {
        "Lag": float(result.lag_s),
        "Sr": float(result.sfreq_hz),
        "RMSE": float(result.rmse_ms),
        "R²": float(result.r2),
        "p-value": float(result.p_value),
    }

    signal_synced = sync_by_lag(sig_arr, float(result.sfreq_hz), float(result.lag_s))

    if save_path and fig is not None:
        fig.savefig(save_path)

    return np.asarray(signal_synced), lag_dict, fig
