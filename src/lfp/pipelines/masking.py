"""High-level masking pipelines for nested tensor bundles.

This module provides pipeline-level APIs that operate on a *nested tensors bundle*
computed from a single MNE Raw object.

The intended workflow is:

    compute tensors  ->  mask (keep or drop)  ->  warp/crop

Masking is applied **before** any warping so that the mask is defined on the
original (unwarped) time axis.

Supported modes
---------------
- :func:`mask_tensors_keep` keeps values only inside selected annotation-covered
  intervals and sets everything else to NaN.
- :func:`mask_tensors_drop` does the inverse: it sets values inside selected
  annotation-covered intervals to NaN and keeps everything else.
- :func:`mask_tensors_dynamic` drops (NaNs) matched annotation intervals but
  expands each interval by a **frequency-dependent** margin specified by the
  user via a per-frequency `time` radius (seconds).
- :func:`get_custom_warped_tensors` runs the high-level keep-mask -> warp -> QC
  -> save flow for a single annotation label.

All functions also return a ``raw_masked`` where Raw annotations are filtered
according to the same keep/drop logic. The Raw signal data are never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ...io.pkl_io import save_pkl
from ...qc.tensor_qc import export_tensor_qc
from ...utils import fileops
from ...utils.transforms import apply_transform_array
from ..common import infer_sfreq_from_times
from ..mask.annotations import (
    MatchMode,
    OverlapPolicy,
    annotation_duration_sec,
    drop_raw_annotations,
    filter_raw_annotations,
    time_mask_by_annotations,
)
from ..mask.mask import apply_time_mask_nan
from .warping import warp_tensors


def _validate_group(group_name: str, group: Mapping[str, Any]) -> None:
    if not isinstance(group, Mapping):
        raise ValueError(f"Group '{group_name}' must be a mapping, got {type(group)}.")
    if "meta" not in group:
        raise ValueError(f"Group '{group_name}' is missing required key 'meta'.")
    meta = group["meta"]
    if not isinstance(meta, Mapping):
        raise ValueError(f"Group '{group_name}' meta must be a mapping, got {type(meta)}.")
    axes = meta.get("axes", {})
    if not isinstance(axes, Mapping) or "time" not in axes:
        raise ValueError(f"Group '{group_name}' meta must contain axes['time'].")


def _mask_group_tensors(
    group: Mapping[str, Any],
    *,
    keep_mask: np.ndarray,
) -> dict[str, Any]:
    """Return a masked copy of all tensors in a group.

    Notes:
        - Only keys other than 'meta' are treated as tensors.
        - ``None`` values are skipped.
        - Time axis must be the last dimension.
    """
    out: dict[str, Any] = {}
    for name, tensor in group.items():
        if name == "meta":
            continue
        if tensor is None:
            continue
        out[name] = apply_time_mask_nan(np.asarray(tensor), keep_mask)
    return out


def _apply_keep_mask_freq_time(
    tensor: np.ndarray,
    keep_mask_ft: np.ndarray,
) -> np.ndarray:
    """Apply a per-frequency time keep-mask to a tensor.

    Conventions:
        - Time axis must be the **last** dimension.
        - A frequency axis must exist and have length == keep_mask_ft.shape[0].

    Args:
        tensor: Input array.
        keep_mask_ft: Boolean keep-mask of shape (n_freqs, n_times).

    Returns:
        A masked copy of ``tensor`` with values set to NaN where keep_mask_ft is False.
    """
    x = np.asarray(tensor)
    m = np.asarray(keep_mask_ft, dtype=bool)

    if m.ndim != 2:
        raise ValueError("`keep_mask_ft` must be 2D: (n_freqs, n_times).")
    n_freqs, n_times = int(m.shape[0]), int(m.shape[1])

    if x.ndim < 2:
        raise ValueError("`tensor` must be at least 2D to apply a frequency-aware mask.")
    if x.shape[-1] != n_times:
        raise ValueError(
            "Tensor time axis length does not match keep_mask_ft. "
            f"Got tensor_time={int(x.shape[-1])} vs n_times={n_times}."
        )

    # Prefer the convention freq axis == -2, otherwise fall back to a unique match.
    if int(x.shape[-2]) == n_freqs:
        freq_axis = x.ndim - 2
    else:
        candidates = [ax for ax in range(x.ndim - 1) if int(x.shape[ax]) == n_freqs]
        if len(candidates) == 0:
            raise ValueError(
                "Could not find a frequency axis matching keep_mask_ft.shape[0]. "
                f"Expected n_freqs={n_freqs} somewhere other than the last axis."
            )
        if len(candidates) > 1:
            raise ValueError(
                "Ambiguous frequency axis: multiple axes match the expected n_freqs. "
                f"Candidate axes={candidates}, tensor_shape={x.shape}."
            )
        freq_axis = int(candidates[0])

    # Move freq axis to -2 for straightforward indexing.
    x2 = np.moveaxis(x, freq_axis, -2) if freq_axis != (x.ndim - 2) else x

    if int(x2.shape[-2]) != n_freqs or int(x2.shape[-1]) != n_times:
        raise RuntimeError("Internal axis move produced unexpected shape.")

    # Ensure dtype can represent NaNs.
    if np.iscomplexobj(x2):
        out = x2.astype(np.complex64, copy=True)
        fill_value = np.nan + 1j * np.nan
    else:
        out = x2.astype(np.float64, copy=True)
        fill_value = np.nan

    # Apply per-frequency mask without broadcasting a full boolean array.
    for fi in range(n_freqs):
        keep_t = m[fi, :]
        out[..., fi, ~keep_t] = fill_value

    out0 = np.moveaxis(out, -2, freq_axis) if freq_axis != (x.ndim - 2) else out
    return out0


def _coerce_float_or_none(x: Any) -> float | None:
    """Best-effort conversion to float.

    Returns None if conversion fails or yields a non-finite number.
    """
    try:
        f = float(x)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    return float(f)


def _freq_items_equal(a: Any, b: Any) -> bool:
    """Return True if two freq-axis items should be treated as equal.

    - If both can be coerced to float, compare numerically with tolerance.
    - Otherwise compare as case-insensitive, stripped strings.
    """
    fa = _coerce_float_or_none(a)
    fb = _coerce_float_or_none(b)
    if fa is not None and fb is not None:
        return bool(np.isclose(fa, fb, rtol=1e-8, atol=1e-12))
    return str(a).strip().lower() == str(b).strip().lower()




def mask_tensors_keep(
    raw: "mne.io.BaseRaw",
    tensors: Mapping[str, Mapping[str, Any]],
    *,
    keep: Sequence[str],
    mode: MatchMode = "substring",
    pad_s: float = 0.0,
    clip_to_raw: bool = True,
    require_match: bool = False,
    overlap_policy: OverlapPolicy = "split",
) -> tuple["mne.io.BaseRaw", dict[str, dict[str, Any]]]:
    """Keep only values inside selected Raw annotation intervals.

    Args:
        raw: MNE Raw providing annotations.
        tensors: Nested mapping where each group has a shared ``meta`` dict with
            ``meta['axes']['time']``.
        keep: Annotation descriptions to keep (e.g., ['sit', 'gait', 'pain']).
        mode: 'substring' (default) or 'exact'.
        pad_s: Optional padding (seconds) applied to each kept interval.
        clip_to_raw: Clip matched intervals to the Raw time span.
        require_match: If True, raise if nothing matches `keep`.
        overlap_policy: How to handle non-keep annotations that overlap the
            keep-interval union in ``raw_masked``:

            - 'split': keep only the overlapping portion(s).
            - 'drop': remove the annotation entirely.

    Returns:
        raw_masked: Shallow-copied Raw with filtered annotations (signal unchanged).
        tensors_masked: Same nested structure, but tensors have NaNs outside kept
            annotation intervals.
    """
    raw_masked, raw_mask_info = filter_raw_annotations(
        raw,
        keep=keep,
        mode=mode,
        pad_s=pad_s,
        clip_to_raw=clip_to_raw,
        require_match=require_match,
        overlap_policy=overlap_policy,
    )

    out: dict[str, dict[str, Any]] = {}

    for group_name, group in tensors.items():
        _validate_group(str(group_name), group)

        meta = dict(group["meta"])
        axes = dict(meta.get("axes", {}))
        times = axes["time"]

        keep_mask, info = time_mask_by_annotations(
            raw,
            times_s=times,
            keep=keep,
            mode=mode,
            pad_s=pad_s,
            clip_to_raw=clip_to_raw,
            require_match=require_match,
        )

        group_out = _mask_group_tensors(group, keep_mask=keep_mask)

        mask_info = dict(info)
        mask_info["raw_annotations"] = raw_mask_info
        meta["mask"] = mask_info
        group_out["meta"] = meta
        out[str(group_name)] = group_out

    return raw_masked, out


def mask_tensors_drop(
    raw: "mne.io.BaseRaw",
    tensors: Mapping[str, Mapping[str, Any]],
    *,
    drop: Sequence[str],
    mode: MatchMode = "substring",
    pad_s: float = 0.0,
    clip_to_raw: bool = True,
    require_match: bool = False,
    overlap_policy: OverlapPolicy = "split",
) -> tuple["mne.io.BaseRaw", dict[str, dict[str, Any]]]:
    """Set values to NaN inside selected Raw annotation intervals.

    This is the logical inverse of :func:`mask_tensors_keep`.

    Args:
        raw: MNE Raw providing annotations.
        tensors: Nested mapping where each group has a shared ``meta`` dict with
            ``meta['axes']['time']``.
        drop: Annotation descriptions whose covered intervals should be dropped.
        mode: 'substring' (default) or 'exact'.
        pad_s: Optional padding (seconds) applied to each dropped interval.
        clip_to_raw: Clip matched intervals to the Raw time span.
        require_match: If True, raise if nothing matches `drop`.
        overlap_policy: How to handle annotations that overlap the drop-interval
            union in ``raw_masked``:

            - 'split': remove only the overlapping portion(s).
            - 'drop': remove the annotation entirely.

    Returns:
        raw_masked: Shallow-copied Raw with filtered annotations (signal unchanged).
        tensors_masked: Same nested structure, but tensors have NaNs inside dropped
            annotation intervals.
    """
    raw_masked, raw_mask_info = drop_raw_annotations(
        raw,
        drop=drop,
        mode=mode,
        pad_s=pad_s,
        clip_to_raw=clip_to_raw,
        require_match=require_match,
        overlap_policy=overlap_policy,
    )

    out: dict[str, dict[str, Any]] = {}

    for group_name, group in tensors.items():
        _validate_group(str(group_name), group)

        meta = dict(group["meta"])
        axes = dict(meta.get("axes", {}))
        times = np.asarray(axes["time"], dtype=float)

        drop_mask, info0 = time_mask_by_annotations(
            raw,
            times_s=times,
            keep=drop,
            mode=mode,
            pad_s=pad_s,
            clip_to_raw=clip_to_raw,
            require_match=require_match,
        )

        # Keep everything that is not explicitly inside a drop interval.
        finite = np.isfinite(times)
        keep_mask = finite & (~np.asarray(drop_mask, dtype=bool))

        group_out = _mask_group_tensors(group, keep_mask=keep_mask)

        # Rewrite the info dict so callers do not need to mentally invert it.
        mask_info: dict[str, Any] = dict(info0)
        mask_info["drop"] = mask_info.pop("keep")
        mask_info["matched_intervals"] = mask_info.get("matched_intervals", [])
        mask_info["n_drop"] = int(mask_info.pop("n_keep"))
        mask_info["n_keep"] = int(np.sum(keep_mask))
        mask_info["kind"] = "drop"
        mask_info["raw_annotations"] = raw_mask_info

        meta["mask"] = mask_info
        group_out["meta"] = meta
        out[str(group_name)] = group_out

    return raw_masked, out


def mask_tensors_dynamic(
    raw: "mne.io.BaseRaw",
    tensors: Mapping[str, Mapping[str, Any]],
    *,
    drop: Sequence[str] = ("edge",),
    freqs: Sequence[float | str] | None = None,
    time: Sequence[float] | None = None,
    mode: MatchMode = "substring",
    clip_to_raw: bool = True,
    require_match: bool = False,
) -> tuple["mne.io.BaseRaw", dict[str, dict[str, Any]]]:
    """Frequency-aware drop-masking using user-provided per-frequency time radii.

    Overview
    --------
    Wavelet-style TFR/PSI/burst estimates are temporally smeared. Instead of
    computing the smear length internally, this function accepts an explicit
    per-frequency (or per-band) *time radius* in seconds.

    For each Raw annotation interval ``[onset, onset+duration]`` that matches
    ``drop``, and for each frequency item ``f`` on a tensor's ``axes['freq']``,
    we expand the interval by ``± time_radius_s[f]`` and set tensor values within
    the expanded window to NaN.

    Mapping rule
    ------------
    ``freqs`` and ``time`` define a shared lookup table:

        freqs[i]  <->  time[i]

    For each tensor group, each entry in ``meta['axes']['freq']`` is matched
    against entries in ``freqs`` using the same matching rule as
    :func:`_freq_items_equal`:

    - If both items are numeric, compare with ``np.isclose``.
    - Otherwise, compare as case-insensitive stripped strings.

    If a group's frequency axis cannot be fully mapped to the provided ``freqs``
    list, the function falls back to a conservative **time-only** mask using the
    maximum radius ``max(time)``.

    Notes
    -----
    Raw annotations cannot represent per-frequency padding, so the returned
    ``raw_masked`` is filtered using the conservative maximum padding.

    Args:
        raw: MNE Raw providing annotations.
        tensors: Nested mapping where each group has a shared ``meta`` dict with
            ``meta['axes']['time']``.
        drop: Annotation descriptions to drop (default: ('edge',)).
        freqs: Lookup keys for the per-frequency radii. Must be provided.
            Elements may be floats/ints (Hz) or strings (band/label names).
        time: Per-frequency time radius in seconds, same length as ``freqs``.
            Each matched annotation interval is expanded by ``± time[i]`` for the
            corresponding ``freqs[i]``.
        mode: 'substring' (default) or 'exact'.
        clip_to_raw: Clip matched intervals to the Raw time span.
        require_match: If True, raise if nothing matches `drop`.

    Returns:
        raw_masked: Shallow-copied Raw with annotations removed in the
            (conservative) max-radius drop union.
        tensors_masked: Same nested structure, but tensors have NaNs inside the
            dynamically expanded drop intervals.
    """
    if freqs is None or time is None:
        raise ValueError("`freqs` and `time` must be provided explicitly.")

    freq_items = list(freqs)
    time_radius = np.asarray(list(time), dtype=float).ravel()

    if len(freq_items) < 1:
        raise ValueError("`freqs` must contain at least 1 element.")
    if time_radius.ndim != 1 or int(time_radius.size) != int(len(freq_items)):
        raise ValueError("`time` must be 1D and have the same length as `freqs`.")
    if np.any(~np.isfinite(time_radius)):
        raise ValueError("`time` must contain only finite values (seconds).")
    if np.any(time_radius < 0):
        raise ValueError("`time` values must be >= 0 (seconds).")

    pad_s_max = float(np.max(time_radius))

    # Raw annotations cannot represent per-frequency padding, so we apply a conservative
    # maximum padding when filtering the Raw annotations.
    raw_masked, raw_mask_info = drop_raw_annotations(
        raw,
        drop=drop,
        mode=mode,
        pad_s=pad_s_max,
        clip_to_raw=clip_to_raw,
        require_match=require_match,
        overlap_policy="split",
    )
    raw_mask_info = dict(raw_mask_info)
    raw_mask_info["kind"] = "drop_dynamic"
    raw_mask_info["pad_s_dynamic_max"] = float(pad_s_max)

    t_min = float(raw.times[0])
    t_max = float(raw.times[-1])

    # Collect base matched intervals (without padding) for provenance.
    drop_lower = [str(x).strip().lower() for x in drop if str(x).strip()]
    matched_base: list[dict[str, Any]] = []
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        d = str(desc)
        d_l = d.lower()
        is_match = (d_l in drop_lower) if mode == "exact" else any(x in d_l for x in drop_lower)
        if not is_match:
            continue
        matched_base.append(
            dict(
                description=d,
                onset_s=float(onset),
                duration_s=float(dur),
            )
        )

    if require_match and len(matched_base) == 0:
        raise ValueError("No Raw annotations matched the requested `drop` labels.")

    out: dict[str, dict[str, Any]] = {}

    def _lookup_index(item: Any) -> int | None:
        for i, key in enumerate(freq_items):
            if _freq_items_equal(item, key):
                return int(i)
        return None

    for group_name, group in tensors.items():
        _validate_group(str(group_name), group)

        meta_in = group["meta"]
        meta = dict(meta_in)
        axes = dict(meta.get("axes", {}))
        times = np.asarray(axes["time"], dtype=float)
        if times.ndim != 1:
            raise ValueError(f"Group '{group_name}' meta axes['time'] must be 1D.")

        finite = np.isfinite(times)

        has_freq_axis = ("freq" in axes) and (axes.get("freq") is not None)

        if has_freq_axis:
            freq_axis_items = list(np.asarray(axes["freq"], dtype=object).ravel())

            idx_map: list[int | None] = []
            missing: list[Any] = []
            for item in freq_axis_items:
                idx = _lookup_index(item)
                idx_map.append(idx)
                if idx is None:
                    missing.append(item)

            if len(missing) > 0:
                # Fallback: apply a conservative time-only mask using max radius.
                drop_mask_t, info0 = time_mask_by_annotations(
                    raw,
                    times_s=times,
                    keep=drop,
                    mode=mode,
                    pad_s=pad_s_max,
                    clip_to_raw=clip_to_raw,
                    require_match=require_match,
                )
                keep_mask_t = finite & (~np.asarray(drop_mask_t, dtype=bool))

                group_out = _mask_group_tensors(group, keep_mask=keep_mask_t)

                mask_info = dict(info0)
                mask_info["drop"] = mask_info.pop("keep")
                mask_info["n_drop"] = int(mask_info.pop("n_keep"))
                mask_info["n_keep"] = int(np.sum(keep_mask_t))
                mask_info["kind"] = "drop_dynamic_time_only"
                mask_info["pad_s_dynamic_max"] = float(pad_s_max)
                mask_info["freq_axis_mapped"] = False
                mask_info["freq_axis"] = [str(x) for x in freq_axis_items]
                mask_info["missing_freq_items"] = [str(x) for x in missing]
                mask_info["raw_annotations"] = raw_mask_info

                meta["mask"] = mask_info
                group_out["meta"] = meta
                out[str(group_name)] = group_out
                continue

            # Per-frequency radii for this group, in the group's freq-axis order.
            radii = np.asarray([float(time_radius[int(i)]) for i in idx_map if i is not None], dtype=float)

            keep_mask_ft = np.ones((int(len(freq_axis_items)), int(times.size)), dtype=bool)
            keep_mask_ft &= finite[None, :]

            for itv in matched_base:
                onset_s = float(itv["onset_s"])
                dur_s = float(itv["duration_s"])
                base_start = float(onset_s)
                base_end = float(onset_s + dur_s)

                for fi in range(int(len(freq_axis_items))):
                    pad = float(radii[fi])
                    start = base_start - pad
                    end = base_end + pad
                    if clip_to_raw:
                        start = max(start, t_min)
                        end = min(end, t_max)
                    if end < start:
                        continue
                    keep_mask_ft[fi, :] &= ~(finite & (times >= start) & (times <= end))

            group_out2: dict[str, Any] = {}
            for name, tensor in group.items():
                if name == "meta" or tensor is None:
                    continue
                group_out2[name] = _apply_keep_mask_freq_time(np.asarray(tensor), keep_mask_ft)

            mask_info2: dict[str, Any] = dict(
                kind="drop_dynamic",
                drop=[str(x) for x in drop],
                mode=str(mode),
                clip_to_raw=bool(clip_to_raw),
                require_match=bool(require_match),
                n_times=int(times.size),
                n_freqs=int(len(freq_axis_items)),
                freqs_lookup=[
                    float(_coerce_float_or_none(x)) if _coerce_float_or_none(x) is not None else str(x)
                    for x in freq_items
                ],
                time_radius_s=time_radius.astype(float).tolist(),
                freq_axis=[str(x) for x in freq_axis_items],
                freq_index_map=[int(i) for i in idx_map if i is not None],
                radii_s=radii.astype(float).tolist(),
                pad_s_dynamic_max=float(pad_s_max),
                matched_intervals=matched_base,
                raw_annotations=raw_mask_info,
            )
            meta["mask"] = mask_info2
            group_out2["meta"] = meta
            out[str(group_name)] = group_out2

        else:
            # Fallback: apply a conservative time-only mask using the maximum radius.
            drop_mask_t, info0 = time_mask_by_annotations(
                raw,
                times_s=times,
                keep=drop,
                mode=mode,
                pad_s=pad_s_max,
                clip_to_raw=clip_to_raw,
                require_match=require_match,
            )
            keep_mask_t = finite & (~np.asarray(drop_mask_t, dtype=bool))

            group_out = _mask_group_tensors(group, keep_mask=keep_mask_t)

            mask_info = dict(info0)
            mask_info["drop"] = mask_info.pop("keep")
            mask_info["n_drop"] = int(mask_info.pop("n_keep"))
            mask_info["n_keep"] = int(np.sum(keep_mask_t))
            mask_info["kind"] = "drop_dynamic_time_only"
            mask_info["pad_s_dynamic_max"] = float(pad_s_max)
            mask_info["raw_annotations"] = raw_mask_info

            meta["mask"] = mask_info
            group_out["meta"] = meta
            out[str(group_name)] = group_out

    return raw_masked, out


def get_custom_warped_tensors(
    raw: "mne.io.BaseRaw",
    tensors: Mapping[str, Mapping[str, Any]],
    anno: str,
    warper: Callable[[Any], tuple[dict[str, Any], Any]],
    warped_dir: Path,
    *,
    mask_cfg: Mapping[str, Any],
    n_samples_warp: Mapping[str, int] | None = None,
    overlap_policy: OverlapPolicy = "split",
    save_name: str | None = None,
    overwrite: bool = True,
) -> None:
    """Mask, warp, QC, and save tensors for a single annotation label."""
    sr_map = {
        key: infer_sfreq_from_times(group["meta"]["axes"]["time"])
        for key, group in tensors.items()
    }

    save_name = anno if save_name is None else save_name

    if anno not in raw.annotations.description:
        print(f"No '{anno}' annotations found; skipping {save_name} warping.")
        return

    anno_dir = warped_dir / save_name
    anno_pkl = anno_dir / "tensors_warped.pkl"

    if anno_pkl.exists() and not overwrite:
        print(f"Warped {save_name} tensors already exist: {anno_pkl}")
        return

    raw_masked, tensors_masked = mask_tensors_keep(
        raw,
        tensors,
        keep=[anno],
        overlap_policy=overlap_policy,
        **dict(mask_cfg),
    )

    if anno not in raw_masked.annotations.description:
        print(f"No '{anno}' annotations found after masking; skipping {save_name} warping.")
        return

    anno_epochs_by_label, anno_warp_fn = warper(raw_masked)
    all_epochs = anno_epochs_by_label.get("ALL", [])
    if len(all_epochs) < 1:
        print(f"No valid epochs found after warping for {save_name}; skipping saving tensors.")
        return

    n_samples: int | Mapping[str, int]
    if n_samples_warp is None:
        anno_duration = annotation_duration_sec(raw_masked, anno)
        n_samples = {
            key: int(np.round(anno_duration * sr))
            for key, sr in sr_map.items()
        }
    else:
        group_n_samples = n_samples_warp.get(save_name)
        if group_n_samples is None:
            anno_duration = annotation_duration_sec(raw_masked, anno)
            n_samples = {
                key: int(np.round(anno_duration * sr))
                for key, sr in sr_map.items()
            }
        else:
            n_samples = int(group_n_samples)

    tensors_anno = warp_tensors(
        tensors_masked,
        warp_fn=anno_warp_fn,
        n_samples=n_samples,
    )

    anno_dir = fileops.create_subfolder(warped_dir, save_name)
    anno_warp_fn_pkl = anno_dir / "warp_fn.pkl"
    anno_warp_labels_pkl = anno_dir / "warp_labels.pkl"
    qc_dir = fileops.create_subfolder(anno_dir, "qc")

    for key, group in tensors_anno.items():
        group_meta = group["meta"]
        qc_key_dir = fileops.create_subfolder(qc_dir, key)
        for subkey, tensor_data in group.items():
            if subkey == "meta":
                continue
            qc_subkey_dir = fileops.create_subfolder(qc_key_dir, subkey)
            export_tensor_qc(
                apply_transform_array(tensor_data, mode="dB" if key == "tfr" else "none"),
                group_meta,
                out_dir=qc_subkey_dir,
                agg_epoch=True,
                y_scale_heatmap="linear",
                x_scale_spectrum="linear",
            )

    save_pkl(tensors_anno, anno_pkl)
    save_pkl(anno_warp_fn, anno_warp_fn_pkl)
    save_pkl(anno_epochs_by_label, anno_warp_labels_pkl)
    print(f"Saved warped {save_name} tensors: {anno_pkl}")
