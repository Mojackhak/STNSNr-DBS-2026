"""tag.py

Convert video-frame-based event tags (stored in pandas DataFrames) into
MNE-Python ``Annotations`` attached to an ``mne.io.Raw`` object.

This module covers two common tag formats:

1) Long format (ranges): one row per event segment

   - columns: ``condition`` / ``start`` / ``end`` (frame indices)
   - example:

     >>> df = pd.DataFrame({
     ...     "condition": ["rest", "move"],
     ...     "start": [0, 120],
     ...     "end": [119, 240],
     ... })

2) Wide format (frame-wise booleans): one row per frame

   - index: frame number (usually consecutive integers)
   - columns: event labels; values are booleans (or 0/1)

The key outcome is an ``mne.Annotations`` object whose onsets/durations are
computed from frame indices using the provided video FPS and aligned to the
Raw recording using ``video_start_sec``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import mne


def df_to_boolean_dataframe(
    df: pd.DataFrame,
    expected_order: Sequence[str] | None = None,
    *,
    condition_col: str = "condition",
    start_col: str = "start",
    end_col: str = "end",
    start_at_zero: bool = True,
    inclusive_end: bool = True,
) -> pd.DataFrame:
    """Convert a long-format tag DataFrame (start/end ranges) into a boolean frame table.

    Parameters
    ----------
    df
        Long-format DataFrame containing at least:
        - ``condition_col``: event label
        - ``start_col``: start frame index
        - ``end_col``: end frame index
    expected_order
        Optional explicit ordering of output columns. Missing labels will be
        created and filled with False.
    condition_col, start_col, end_col
        Column names for label/start/end.
    start_at_zero
        If True, output index starts at frame 0. If False, output index starts
        at the minimum start frame.
    inclusive_end
        If True, interpret ``end`` as inclusive (the common convention for
        frame ranges). If False, interpret ``end`` as exclusive.

    Returns
    -------
    pd.DataFrame
        Wide-format boolean DataFrame:
        - index: frame numbers
        - columns: conditions
        - values: True/False indicating whether each condition is active at
          each frame.
    """
    _validate_ranges_df(df, condition_col=condition_col, start_col=start_col, end_col=end_col)

    # Convert start/end to integer frame indices (allow numeric strings/floats that are integer-like).
    starts = _coerce_integer_series(df[start_col], name=start_col)
    ends = _coerce_integer_series(df[end_col], name=end_col)

    # Validate ranges.
    # - inclusive_end=True: require end >= start
    # - inclusive_end=False (half-open [start, end)): require end >= start
    if (ends < starts).any():
        bad = df.loc[ends < starts, [condition_col, start_col, end_col]].head(5)
        raise ValueError(
            "Found rows where end < start. Example rows:\n"
            f"{bad}"
        )

    min_start = int(starts.min())
    if inclusive_end:
        max_end = int(ends.max())
    else:
        # The last included frame is end-1. If an interval is empty (start==end),
        # treat it as contributing no True frames but keep a reasonable index length.
        max_end = int(max(int(ends.max()) - 1, int(starts.max())))

    if max_end < 0:
        raise ValueError("All end frames are < 0. Frame indices must be non-negative.")

    index_start = 0 if start_at_zero else min_start
    if index_start < 0:
        raise ValueError("start_at_zero=False produced a negative index_start.")

    # Use unique labels as discovered (order of appearance), unless expected_order is provided.
    labels = pd.unique(df[condition_col].astype(str))
    bool_df = pd.DataFrame(
        False,
        index=pd.RangeIndex(start=index_start, stop=max_end + 1, step=1, name="frame"),
        columns=labels,
        dtype=bool,
    )

    # Populate ranges.
    for label, start_frame, end_frame in zip(
        df[condition_col].astype(str).to_numpy(),
        starts.to_numpy(),
        ends.to_numpy(),
        strict=True,
    ):
        if inclusive_end:
            last_frame = int(end_frame)
        else:
            last_frame = int(end_frame) - 1

        if last_frame < int(start_frame):
            # When inclusive_end=False, a row with start==end means an empty interval.
            continue

        bool_df.loc[int(start_frame) : last_frame, label] = True

    if expected_order is not None:
        # Reindex columns to a provided order, creating missing columns if needed.
        expected_order = [str(x) for x in expected_order]
        bool_df = bool_df.reindex(columns=expected_order, fill_value=False)

    return bool_df


def visualize_boolean_df(
    df: pd.DataFrame,
    expected_order: Sequence[str] | None = None,
    title: str | None = None,
    *,
    show: bool = True,
):
    """Visualize a frame-wise boolean DataFrame as a raster plot.

    Parameters
    ----------
    df
        Wide-format boolean DataFrame with frames on the index and labels as columns.
    expected_order
        Optional explicit ordering of labels (columns).
    title
        Plot title.
    show
        If True, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib figure.
    """
    # Import lazily so non-plotting use cases do not require a GUI backend.
    import matplotlib.pyplot as plt

    if expected_order is not None:
        df = df.reindex(columns=[str(x) for x in expected_order], fill_value=False)

    mat = df.astype(int).to_numpy().T
    n_states, n_frames = mat.shape

    cmap = plt.get_cmap("tab20", n_states)
    img = np.ones((n_states, n_frames, 3), dtype=float)
    for i in range(n_states):
        img[i, mat[i].astype(bool)] = cmap(i)[:3]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(df.columns)
    ax.set_xlabel("Frame index")
    ax.set_title(title or "Event raster plot")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def frames_df_to_annotations(
    df: pd.DataFrame,
    raw: mne.io.BaseRaw,
    video_fps: float,
    video_start_sec: float = 0.0,
    treat_single_sample_as: Literal["point", "epoch"] = "point",
    *,
    copy: bool = True,
    keep_existing: bool = True,
    sort_by_onset: bool = True,
) -> mne.io.BaseRaw:
    """Convert a frame-wise boolean DataFrame into Raw annotations.

    This function assumes the DataFrame rows correspond to consecutive video
    frames. The DataFrame **index is used as the actual frame number**.
    Therefore, an index offset (e.g., starting at frame 100) will correctly
    shift onsets by 100 / video_fps.

    Parameters
    ----------
    df
        Wide-format boolean DataFrame:
        - columns: event labels
        - index: video frame numbers (should be consecutive integers)
    raw
        The recording to receive the annotations.
    video_fps
        Frames per second of the video.
    video_start_sec
        Time (in Raw seconds) at which video frame 0 occurred.
    treat_single_sample_as
        - ``"point"``: a 1-frame run becomes instantaneous (duration=0)
        - ``"epoch"``: a 1-frame run becomes a 1/FPS epoch
    copy
        If True, return a copy of ``raw`` with annotations attached.
        If False, modify the passed ``raw`` in place and return it.
    keep_existing
        If True, keep existing Raw annotations and append new ones.
    sort_by_onset
        If True, sort newly created annotations by onset time.

    Returns
    -------
    mne.io.BaseRaw
        Raw with the new annotations attached.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df)!r}")
    if df.empty:
        raise ValueError("df is empty; cannot create annotations.")
    if df.shape[1] == 0:
        raise ValueError("df has no columns; cannot create annotations.")

    if not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError(f"video_fps must be a positive finite float, got {video_fps!r}")

    if treat_single_sample_as not in ("point", "epoch"):
        raise ValueError(
            "treat_single_sample_as must be either 'point' or 'epoch', "
            f"got {treat_single_sample_as!r}"
        )

    frame_numbers = _coerce_consecutive_frame_index(df.index)
    time_per_frame = 1.0 / float(video_fps)

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    for label in df.columns.astype(str):
        mask = df[label].astype(bool).to_numpy()

        # Find contiguous True runs using edge detection on the padded mask.
        padded = np.pad(mask.astype(int), (1, 1))
        starts = np.where(np.diff(padded) == 1)[0]  # first True index
        ends = np.where(np.diff(padded) == -1)[0]  # one past last True index

        for start_pos, end_pos in zip(starts, ends, strict=True):
            n_frames = int(end_pos - start_pos)
            if n_frames <= 0:
                continue

            if n_frames == 1 and treat_single_sample_as == "point":
                dur_sec = 0.0
            else:
                dur_sec = n_frames * time_per_frame

            start_frame = int(frame_numbers[start_pos])
            onset_sec = (start_frame * time_per_frame) + float(video_start_sec)

            onsets.append(float(onset_sec))
            durations.append(float(dur_sec))
            descriptions.append(str(label))

    if len(onsets) == 0:
        ann = mne.Annotations([], [], [], orig_time=_pick_annotations_orig_time(raw, keep_existing))
    else:
        if sort_by_onset:
            order = np.argsort(np.asarray(onsets))
            onsets = [onsets[i] for i in order]
            durations = [durations[i] for i in order]
            descriptions = [descriptions[i] for i in order]

        ann = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=_pick_annotations_orig_time(raw, keep_existing),
        )

    target = raw.copy() if copy else raw
    if keep_existing and len(raw.annotations) > 0:
        target.set_annotations(raw.annotations + ann)
    else:
        target.set_annotations(ann)
    return target


def ranges_df_to_annotations(
    df: pd.DataFrame,
    raw: mne.io.BaseRaw,
    video_fps: float,
    video_start_sec: float = 0.0,
    treat_single_sample_as: Literal["point", "epoch"] = "point",
    *,
    condition_col: str = "condition",
    start_col: str = "start",
    end_col: str = "end",
    inclusive_end: bool = True,
    copy: bool = True,
    keep_existing: bool = True,
    sort_by_onset: bool = True,
) -> mne.io.BaseRaw:
    """Convert a long-format ranges DataFrame directly into Raw annotations.

    This avoids expanding ranges into a potentially huge frame-by-frame table.

    Parameters
    ----------
    df
        Long-format ranges DataFrame.
    raw
        The recording to receive the annotations.
    video_fps
        Frames per second of the video.
    video_start_sec
        Time (in Raw seconds) at which video frame 0 occurred.
    treat_single_sample_as
        How to handle a segment that spans exactly 1 frame.
    condition_col, start_col, end_col
        Column names.
    inclusive_end
        Whether ``end`` is inclusive.
    copy, keep_existing, sort_by_onset
        See ``frames_df_to_annotations``.

    Returns
    -------
    mne.io.BaseRaw
        Raw with the new annotations attached.
    """
    _validate_ranges_df(df, condition_col=condition_col, start_col=start_col, end_col=end_col)

    if not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError(f"video_fps must be a positive finite float, got {video_fps!r}")
    if treat_single_sample_as not in ("point", "epoch"):
        raise ValueError(
            "treat_single_sample_as must be either 'point' or 'epoch', "
            f"got {treat_single_sample_as!r}"
        )

    starts = _coerce_integer_series(df[start_col], name=start_col)
    ends = _coerce_integer_series(df[end_col], name=end_col)
    labels = df[condition_col].astype(str).to_numpy()

    time_per_frame = 1.0 / float(video_fps)

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    for label, start_frame, end_frame in zip(labels, starts.to_numpy(), ends.to_numpy(), strict=True):
        start_frame_i = int(start_frame)
        end_frame_i = int(end_frame)

        if inclusive_end:
            n_frames = (end_frame_i - start_frame_i) + 1
        else:
            n_frames = end_frame_i - start_frame_i

        if n_frames <= 0:
            continue

        if n_frames == 1 and treat_single_sample_as == "point":
            dur_sec = 0.0
        else:
            dur_sec = n_frames * time_per_frame

        onset_sec = (start_frame_i * time_per_frame) + float(video_start_sec)

        onsets.append(float(onset_sec))
        durations.append(float(dur_sec))
        descriptions.append(str(label))

    if len(onsets) == 0:
        ann = mne.Annotations([], [], [], orig_time=_pick_annotations_orig_time(raw, keep_existing))
    else:
        if sort_by_onset:
            order = np.argsort(np.asarray(onsets))
            onsets = [onsets[i] for i in order]
            durations = [durations[i] for i in order]
            descriptions = [descriptions[i] for i in order]

        ann = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=_pick_annotations_orig_time(raw, keep_existing),
        )

    target = raw.copy() if copy else raw
    if keep_existing and len(raw.annotations) > 0:
        target.set_annotations(raw.annotations + ann)
    else:
        target.set_annotations(ann)
    return target


def _validate_ranges_df(df: pd.DataFrame, *, condition_col: str, start_col: str, end_col: str) -> None:
    """Validate that a long-format ranges DataFrame has the required columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df)!r}")

    missing = [c for c in (condition_col, start_col, end_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    if df.empty:
        raise ValueError("df is empty.")


def _coerce_integer_series(series: pd.Series, *, name: str) -> pd.Series:
    """Coerce a pandas Series into integer values with validation."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad_idx = numeric[numeric.isna()].index[:5].to_list()
        raise ValueError(f"Column '{name}' contains non-numeric values (examples at index {bad_idx}).")

    as_float = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(as_float)):
        raise ValueError(f"Column '{name}' contains non-finite values.")

    # Accept floats only if they are integer-like (e.g., 12.0).
    if not np.allclose(as_float, np.round(as_float), atol=0, rtol=0):
        raise ValueError(f"Column '{name}' must contain integer frame indices.")

    return pd.Series(np.round(as_float).astype(int), index=series.index, name=name)


def _coerce_consecutive_frame_index(index: pd.Index) -> np.ndarray:
    """Ensure the index is a consecutive integer frame sequence and return it as an array."""
    if isinstance(index, pd.RangeIndex):
        if index.step != 1:
            raise ValueError(
                "DataFrame index must have step=1 (consecutive frames). "
                f"Got RangeIndex(step={index.step})."
            )
        frames = np.arange(index.start, index.stop, index.step, dtype=int)
    else:
        numeric = pd.to_numeric(index.to_numpy(), errors="coerce")
        if np.isnan(numeric).any():
            raise ValueError("DataFrame index must be numeric (frame numbers).")
        if not np.allclose(numeric, np.round(numeric), atol=0, rtol=0):
            raise ValueError("DataFrame index must be integer-like frame numbers.")
        frames = np.round(numeric).astype(int)

    if frames.size == 0:
        raise ValueError("DataFrame index is empty.")

    if np.any(frames < 0):
        raise ValueError("Frame numbers must be non-negative.")

    diffs = np.diff(frames)
    if frames.size > 1 and not np.all(diffs == 1):
        raise ValueError(
            "DataFrame index must be consecutive frame numbers (difference of 1). "
            "If you have gaps, reindex to a full frame range and fill missing frames with False."
        )

    return frames


def _pick_annotations_orig_time(raw: mne.io.BaseRaw, keep_existing: bool) -> object | None:
    """Choose an orig_time compatible with existing annotations (if any)."""
    if keep_existing and hasattr(raw, "annotations") and raw.annotations is not None:
        if raw.annotations.orig_time is not None:
            return raw.annotations.orig_time
    # Fall back to meas_date (may be None).
    return raw.info.get("meas_date", None)
