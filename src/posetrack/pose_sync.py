from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, Union

import numpy as np
import pandas as pd
import mne

def _normalize_patterns(
    bad_descs: Union[str, Sequence[str]],
    *,
    case_sensitive: bool,
) -> tuple[str, ...]:
    patterns = (bad_descs,) if isinstance(bad_descs, str) else tuple(bad_descs)
    if len(patterns) == 0:
        raise ValueError("bad_descs cannot be empty; provide at least one BAD identifier.")
    if case_sensitive:
        return tuple(str(p) for p in patterns)
    return tuple(str(p).upper() for p in patterns)


def _make_is_bad_desc(
    patterns_norm: tuple[str, ...],
    *,
    match_mode: str,
    case_sensitive: bool,
):
    if match_mode not in ("substring", "prefix"):
        raise ValueError(f"match_mode must be 'substring' or 'prefix', got: {match_mode}")

    def _norm_desc(desc: str) -> str:
        return desc if case_sensitive else desc.upper()

    def is_bad_desc(desc: str) -> bool:
        d = _norm_desc(str(desc))
        if match_mode == "substring":
            return any(p in d for p in patterns_norm)
        return any(d.startswith(p) for p in patterns_norm)

    return is_bad_desc


def build_bad_sample_mask_like_filter(
    raw: mne.io.BaseRaw,
    *,
    bad_annotations: Optional[mne.Annotations] = None,
    bad_descs: Union[str, Sequence[str]] = ("BAD",),
    match_mode: str = "substring",
    case_sensitive: bool = False,
) -> np.ndarray:
    """
    Build a boolean bad-sample mask following the same onset/duration -> samples
    logic used by filter_lfp_with_bad_annotations.

    Notes
    -----
    - This function does NOT modify `raw`.
    - The returned mask is in the original raw buffer sample coordinates
      (length == raw.n_times).
    """
    sfreq = float(raw.info["sfreq"])
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid raw.info['sfreq']: {sfreq}")

    n_times = int(raw.n_times)
    if n_times <= 0:
        return np.zeros(0, dtype=bool)

    patterns_norm = _normalize_patterns(bad_descs, case_sensitive=case_sensitive)
    is_bad_desc = _make_is_bad_desc(
        patterns_norm,
        match_mode=match_mode,
        case_sensitive=case_sensitive,
    )

    sample_shift = int(getattr(raw, "first_samp", 0))
    bad_mask = np.zeros(n_times, dtype=bool)

    def iter_ann(ann: Optional[mne.Annotations]):
        if ann is None:
            return
        for o, d, s in zip(ann.onset, ann.duration, ann.description):
            yield float(o), float(d), str(s)

    # Same order as the filter function: bad_annotations first, then raw.annotations
    combined = list(iter_ann(bad_annotations)) + list(iter_ann(raw.annotations))

    for onset, duration, desc in combined:
        if not is_bad_desc(desc):
            continue

        start_samp = int(round(onset * sfreq)) - sample_shift
        stop_samp = int(round((onset + duration) * sfreq)) - sample_shift

        start_samp = max(start_samp, 0)
        stop_samp = min(stop_samp, n_times)

        if stop_samp > start_samp:
            bad_mask[start_samp:stop_samp] = True

    return bad_mask


def crop_posetrack_df_to_filtered_raw(
    posetrack_df: pd.DataFrame,
    posetrack_fps: float,
    raw: mne.io.BaseRaw,
    *,
    bad_annotations: Optional[mne.Annotations] = None,
    bad_descs: Union[str, Sequence[str]] = ("BAD",),
    overlap_policy: str = "split",
    match_mode: str = "substring",
    case_sensitive: bool = False,
    frame0: Optional[float] = None,
    t0_offset_sec: float = 0.0,
    reset_index: bool = True,
    index_name: str = "frame",
    keep_original_index_col: Optional[str] = None,
    add_lfp_good_time_sec_col: Optional[str] = None,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Crop a PoseTrack per-frame DataFrame so it stays synchronized with an MNE Raw
    object after applying filter_lfp_with_bad_annotations-like BAD cutting.

    Parameters
    ----------
    posetrack_df
        Input DataFrame. Index represents frame numbers; each row is a frame.
    posetrack_fps
        PoseTrack frame rate (Hz).
    raw
        The original (pre-cut) electrophysiology raw object (or Raw-like).
        Its start must be synchronized with posetrack_df start.
    bad_annotations, bad_descs, match_mode, case_sensitive
        BAD cutting rules consistent with filter_lfp_with_bad_annotations.
    overlap_policy
        Kept only for signature compatibility. It does NOT affect which samples are removed.
    frame0
        Frame index treated as time 0 of the raw. If None, uses posetrack_df.index[0].
    t0_offset_sec
        Optional time offset (sec) added to frame time to account for known alignment delay.
        Positive means PoseTrack is later than raw.
    reset_index
        If True, reset output index to 0..N-1 (recommended for a "cut video" timeline).
    keep_original_index_col
        If provided, store the original frame index in a new column with this name.
    add_lfp_good_time_sec_col
        If provided, add a column containing each kept frame's time (sec) in the
        compressed "good-only" raw timeline.
    return_report
        If True, return (df_out, report_dict).

    Returns
    -------
    df_out (and optionally report)
        Cropped DataFrame synchronized to the same BAD removal as the raw.
    """
    if overlap_policy not in ("split", "drop"):
        raise ValueError(f"overlap_policy must be 'split' or 'drop', got: {overlap_policy}")

    fps = float(posetrack_fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"posetrack_fps must be finite and > 0, got: {posetrack_fps}")

    if posetrack_df.shape[0] == 0:
        out_empty = posetrack_df.copy()
        if return_report:
            return out_empty, {
                "n_frames_in": 0,
                "n_frames_out": 0,
                "n_frames_dropped_bad": 0,
                "n_frames_dropped_out_of_range": 0,
            }
        return out_empty

    # Frame index -> seconds (relative to raw start)
    idx = np.asarray(posetrack_df.index)
    try:
        frame_vals = idx.astype(float)
    except Exception as e:
        raise TypeError(
            "posetrack_df.index must be numeric (frame numbers). "
            "If your DataFrame index is not numeric, set it to frame numbers first."
        ) from e

    if not np.all(np.isfinite(frame_vals)):
        raise ValueError("posetrack_df.index contains non-finite frame numbers (NaN/Inf).")

    if not np.all(np.diff(frame_vals) >= 0):
        raise ValueError("posetrack_df.index must be non-decreasing (monotonic).")

    if frame0 is None:
        frame0 = float(frame_vals[0])

    t_sec = (frame_vals - float(frame0)) / fps + float(t0_offset_sec)

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)

    # Map each frame time to a sample index in the original raw buffer.
    samp_idx = np.round(t_sec * sfreq).astype(np.int64)
    in_range = (samp_idx >= 0) & (samp_idx < n_times)

    bad_mask = build_bad_sample_mask_like_filter(
        raw,
        bad_annotations=bad_annotations,
        bad_descs=bad_descs,
        match_mode=match_mode,
        case_sensitive=case_sensitive,
    )
    good_mask = ~bad_mask

    keep = np.zeros(len(posetrack_df), dtype=bool)
    keep[in_range] = good_mask[samp_idx[in_range]]

    # Optional: compute frame time in compressed good-only timeline.
    lfp_good_time_sec: Optional[np.ndarray] = None
    if add_lfp_good_time_sec_col is not None:
        good_cum = np.cumsum(good_mask.astype(np.int64))
        lfp_good_time_sec = np.full(len(posetrack_df), np.nan, dtype=float)
        valid = in_range & keep
        lfp_good_time_sec[valid] = (good_cum[samp_idx[valid]] - 1) / sfreq

    out = posetrack_df.loc[keep].copy()

    if keep_original_index_col is not None:
        out.insert(0, keep_original_index_col, out.index.astype(frame_vals.dtype, copy=False))

    if add_lfp_good_time_sec_col is not None and lfp_good_time_sec is not None:
        out[add_lfp_good_time_sec_col] = lfp_good_time_sec[keep]

    if reset_index:
        out.index = pd.RangeIndex(start=0, stop=out.shape[0], step=1, name=index_name)

    report: dict[str, Any] = {
        "posetrack_fps": fps,
        "sfreq": sfreq,
        "n_times_raw": n_times,
        "raw_duration_sec": n_times / sfreq if sfreq > 0 else float("nan"),
        "n_frames_in": int(posetrack_df.shape[0]),
        "n_frames_out": int(out.shape[0]),
        "n_frames_dropped_bad": int(np.sum(in_range & ~keep)),
        "n_frames_dropped_out_of_range": int(np.sum(~in_range)),
        "frame0_used": float(frame0),
        "t0_offset_sec": float(t0_offset_sec),
        "bad_descs": bad_descs if isinstance(bad_descs, str) else list(bad_descs),
        "match_mode": match_mode,
        "case_sensitive": bool(case_sensitive),
    }

    if return_report:
        return out, report
    return out
