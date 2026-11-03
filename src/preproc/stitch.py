"""Stitch multiple MNE Raw objects into a new RawArray while preserving annotations.

This module provides a safe alternative to `mne.concatenate_raws` / `Raw.append` for
workflows where annotation timing must remain stable.

Why not use the official concatenation?
    In some pipelines, built-in concatenation can lead to unexpected annotation onset
    shifts due to interactions between:
      - `raw.first_time` (derived from `first_samp / sfreq`),
      - `meas_date` / absolute time handling,
      - internal remapping performed during concatenation.

What this module does instead:
    1) Collects each segment's data via `raw.get_data()` and concatenates into a single
       preallocated NumPy array.
    2) Creates a brand-new `mne.io.RawArray` starting at `first_samp=0`, ensuring the
       stitched time axis is purely relative (stitched `first_time == 0`).
    3) Preserves all annotations from each input raw by remapping onsets onto the new
       stitched timeline using robust rules for both relative and absolute annotations.
    4) Optionally inserts a boundary annotation (e.g., "STITCH_EDGE") at every stitch edge.

Key guarantees:
    - Annotation positions are preserved on the stitched time axis.
    - A new raw object is created (no in-place modification of the inputs).

Compatibility requirements:
    - All input raws must have identical channel names in the same order.
    - All input raws must share the same sampling frequency (`info['sfreq']`).

Performance notes:
    - This implementation loads all segments into memory and creates a single dense array.
      For very large recordings, consider a chunked or memmap-based approach.

Public API:
    - `stitch_raws(...)` is the main entry point.
    - `RawStitchConfig` controls boundary annotations and meas_date handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

import mne

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawStitchConfig:
    """Configuration for stitching multiple MNE Raw objects."""

    add_edge_annotations: bool = True
    edge_description: str = "STITCH_EDGE"
    edge_duration_sec: float = 0.0

    # meas_date handling:
    # - True  -> require all raws have identical meas_date; keep it
    # - False -> set stitched meas_date=None
    # - None  -> auto: keep if identical else None
    keep_meas_date: bool | None = None

    # RawArray copy behavior
    copy_data: str | bool = "auto"


def stitch_raws(
    raws: Sequence[mne.io.BaseRaw],
    *,
    config: RawStitchConfig | None = None,
) -> mne.io.RawArray:
    """Stitch multiple MNE Raw objects into a new RawArray while preserving annotations.

    This function deliberately avoids `mne.concatenate_raws` / `Raw.append`, and instead:
      1) concatenates data into a new array,
      2) constructs a brand-new `mne.io.RawArray`,
      3) remaps every annotation onto the stitched time axis,
      4) optionally inserts a boundary annotation at each stitch edge.

    Parameters
    ----------
    raws
        Raw objects to stitch, in order.
        All raws must share the same channel names (same order) and sampling rate.
    config
        RawStitchConfig controlling edge annotations, meas_date handling, and copy behavior.

    Returns
    -------
    stitched
        A new RawArray with concatenated data and correctly placed annotations.

    Raises
    ------
    ValueError
        If `raws` is empty or incompatible (sfreq/channel mismatch).
    """
    if config is None:
        config = RawStitchConfig()

    if len(raws) == 0:
        raise ValueError("`raws` must be a non-empty sequence of mne.io.BaseRaw objects.")

    _validate_raws_compatible(raws)

    sfreq = float(raws[0].info["sfreq"])
    n_channels = len(raws[0].ch_names)
    total_samples = int(sum(r.n_times for r in raws))

    meas_date = _decide_meas_date(raws, keep_meas_date=config.keep_meas_date)

    # Build output info from first raw.
    info = raws[0].info.copy()

    # Union bad channels across raws.
    bads_union: set[str] = set()
    for r in raws:
        bads_union.update(r.info.get("bads", []))
    info["bads"] = sorted(bads_union)

    # Preallocate and fill data.
    data = np.empty((n_channels, total_samples), dtype=np.float64)
    cursor = 0
    for idx, r in enumerate(raws):
        block = r.get_data()  # (n_channels, n_times)
        if block.shape[0] != n_channels:
            raise ValueError(
                f"Raw #{idx} has {block.shape[0]} channels, expected {n_channels}."
            )
        n_t = block.shape[1]
        data[:, cursor : cursor + n_t] = block
        cursor += n_t

    # Create a brand-new RawArray. Start at first_samp=0 so stitched first_time=0.
    stitched = mne.io.RawArray(
        data,
        info,
        first_samp=0,
        copy=config.copy_data,
        verbose=False,
    )

    # IMPORTANT: MNE forbids setting info['meas_date'] directly in newer versions.
    # Use the official setter on the Raw instance.
    stitched.set_meas_date(meas_date)

    # Remap and merge annotations (including boundary markers).
    annotations = _merge_annotations_with_offsets(
        raws,
        sfreq=sfreq,
        add_edge_annotations=config.add_edge_annotations,
        edge_description=config.edge_description,
        edge_duration_sec=config.edge_duration_sec,
    )
    stitched.set_annotations(annotations)

    return stitched


def _validate_raws_compatible(raws: Sequence[mne.io.BaseRaw]) -> None:
    """Validate that raws can be stitched safely."""
    base = raws[0]
    base_sfreq = float(base.info["sfreq"])
    base_ch_names = list(base.ch_names)

    for idx, r in enumerate(raws):
        if float(r.info["sfreq"]) != base_sfreq:
            raise ValueError(
                f"Sampling rate mismatch at raw #{idx}: {r.info['sfreq']} != {base_sfreq}"
            )
        if list(r.ch_names) != base_ch_names:
            raise ValueError(
                "Channel mismatch detected. All raws must have identical `ch_names` "
                "(same names and same order) to stitch safely."
            )


def _decide_meas_date(
    raws: Sequence[mne.io.BaseRaw],
    *,
    keep_meas_date: bool | None,
) -> Any:
    """Decide the meas_date for the stitched Raw."""
    meas_dates = [r.info.get("meas_date", None) for r in raws]
    all_equal = all(md == meas_dates[0] for md in meas_dates)

    if keep_meas_date is True:
        if not all_equal:
            raise ValueError(
                "keep_meas_date=True but raws have different info['meas_date']. "
                "Either set keep_meas_date=None/False, or make meas_date consistent."
            )
        return meas_dates[0]

    if keep_meas_date is False:
        return None

    # auto
    if all_equal:
        return meas_dates[0]

    logger.warning(
        "Raw objects have different meas_date values; using meas_date=None for stitched Raw."
    )
    return None


def _to_posix_seconds(t: Any) -> float:
    """Convert various time representations to POSIX seconds.

    Supports:
      - datetime (naive treated as UTC)
      - (sec, usec) tuple stamps
      - float/int seconds
    """
    if t is None:
        raise ValueError("Time value is None; cannot convert to POSIX seconds.")

    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return float(t.timestamp())

    if isinstance(t, tuple) and len(t) >= 2:
        sec = float(t[0])
        usec = float(t[1])
        return sec + usec * 1e-6

    if isinstance(t, (int, float, np.integer, np.floating)):
        return float(t)

    raise TypeError(f"Unsupported time type for conversion: {type(t)!r}")


def _annotation_onsets_relative_to_raw_start(
    raw: mne.io.BaseRaw,
    ann: mne.Annotations,
) -> np.ndarray:
    """Compute annotation onsets relative to the start of `raw` (raw.times[0] == 0).

    Rules:
      - If ann.orig_time is None:
            ann.onset is already relative to raw start (time=0 at first sample of this Raw object).
      - If ann.orig_time is not None:
            treat ann.onset as seconds after ann.orig_time (absolute timeline).
            Convert to seconds after raw start using raw.info['meas_date'] and raw.first_time.

    Returns
    -------
    onset_rel : ndarray, shape (n_annotations,)
        Onsets in seconds relative to the start of the Raw object.
    """
    if len(ann) == 0:
        return np.asarray([], dtype=float)

    if ann.orig_time is None:
        return np.asarray(ann.onset, dtype=float)

    meas_date = raw.info.get("meas_date", None)
    if meas_date is None:
        raise ValueError(
            "Annotations have orig_time set but raw.info['meas_date'] is None. "
            "Cannot convert absolute annotation times to raw-relative times."
        )

    orig_posix = _to_posix_seconds(ann.orig_time)
    meas_posix = _to_posix_seconds(meas_date)

    raw_start_posix = meas_posix + float(raw.first_time)
    onset_abs_posix = orig_posix + np.asarray(ann.onset, dtype=float)
    return onset_abs_posix - raw_start_posix


def _merge_annotations_with_offsets(
    raws: Sequence[mne.io.BaseRaw],
    *,
    sfreq: float,
    add_edge_annotations: bool,
    edge_description: str,
    edge_duration_sec: float,
) -> mne.Annotations:
    """Merge annotations from each raw into the stitched timeline."""
    all_onset: list[float] = []
    all_duration: list[float] = []
    all_desc: list[str] = []
    all_ch_names: list[tuple[str, ...]] = []

    cumulative_samples = 0

    for i, r in enumerate(raws):
        offset_sec = cumulative_samples / sfreq

        ann = r.annotations
        if len(ann) > 0:
            onset_rel = _annotation_onsets_relative_to_raw_start(r, ann)
            onset_global = onset_rel + offset_sec

            all_onset.extend(onset_global.tolist())
            all_duration.extend(np.asarray(ann.duration, dtype=float).tolist())
            all_desc.extend([str(d) for d in ann.description])

            # MNE stores channel-specific annotation mapping as tuples.
            for ch_tuple in ann.ch_names:
                all_ch_names.append(tuple(ch_tuple))

        cumulative_samples += int(r.n_times)

        # Add boundary marker after this segment unless it's the last.
        if add_edge_annotations and i < (len(raws) - 1):
            boundary_onset = cumulative_samples / sfreq
            all_onset.append(float(boundary_onset))
            all_duration.append(float(edge_duration_sec))
            all_desc.append(str(edge_description))
            all_ch_names.append(tuple())

    if len(all_onset) == 0:
        return mne.Annotations(onset=[], duration=[], description=[], orig_time=None)

    order = np.argsort(np.asarray(all_onset, dtype=float))
    all_onset_sorted = [all_onset[j] for j in order]
    all_duration_sorted = [all_duration[j] for j in order]
    all_desc_sorted = [all_desc[j] for j in order]
    all_ch_names_sorted = [all_ch_names[j] for j in order]

    return mne.Annotations(
        onset=all_onset_sorted,
        duration=all_duration_sorted,
        description=all_desc_sorted,
        orig_time=None,
        ch_names=all_ch_names_sorted,
    )
