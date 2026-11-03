"""Warping helpers for time-aligned tensors.

This module provides two levels of API:

- :func:`warp_tensor` for a single tensor (time axis is last).
- :func:`warp_tensors` for a *nested* tensors bundle where each group stores
  multiple tensors plus a shared ``meta`` dict.

Both functions rely on a paradigm-specific ``warp_fn`` returned by warper
builders (e.g., gait_warper, pad_warper, concat_warper).

Conventions
-----------
Unwarped group meta must contain:

    meta['axes'] = {'epoch', 'channel', 'freq', 'time', ...}

Warped group meta will be rewritten to:

    meta['axes'] = {'annotation', 'epoch', 'channel', 'freq', 'time', 'percent', ...}

Notes
-----
- ``axes['freq']`` may be numeric (Hz) or strings (band/parameter names).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..common import infer_sfreq_from_times
from ..warp.metadata import build_warped_tensor_metadata


def warp_tensor(
    tensor: np.ndarray,
    *,
    warp_fn,
    source_meta: Dict[str, Any],
    n_samples: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Warp a single tensor and build warped metadata.

    Args:
        tensor: Array whose last axis is time.
        warp_fn: Callable returned by a warper builder. Must accept
            ``(tensor, sr=<Hz>, n_samples=<int>)`` and return
            ``(warped, percent_axis, meta_epochs)``.
        source_meta: Unwarped metadata dict containing at least ``source_meta['axes']``.
        n_samples: Number of samples in the warped time axis.

    Returns:
        warped: Warped tensor with shape ``(n_epochs, ...)``.
        meta_warped: Warped metadata dict.
    """
    if "axes" not in source_meta:
        raise ValueError("`source_meta` must contain an 'axes' dict.")

    axes = source_meta["axes"]
    if "time" not in axes:
        raise ValueError("`source_meta['axes']` must contain a 'time' axis.")

    sr = infer_sfreq_from_times(axes["time"])

    warped, percent_axis, meta_epochs = warp_fn(np.asarray(tensor), sr=float(sr), n_samples=int(n_samples))

    meta_warped = build_warped_tensor_metadata(
        axes,
        np.asarray(percent_axis, dtype=float),
        meta_epochs,
        source_meta=source_meta,
    )
    return np.asarray(warped, dtype=np.float64), meta_warped


def warp_tensors(
    tensors: Mapping[str, Mapping[str, Any]],
    *,
    warp_fn,
    n_samples: int | Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Warp an entire nested tensors bundle.

    The expected input structure is:

    .. code-block:: python

        tensors = {
            'tfr': {'raw': <ndarray>, 'aperiod': <ndarray>, 'meta': <dict>},
            'conn': {'coh': <ndarray>, 'plv': <ndarray>, 'meta': <dict>},
            ...
        }

    Each group must have a shared ``meta`` dict. All tensors in a group are
    warped using the group's time axis to infer the effective sampling rate.

    Args:
        tensors: Nested mapping ``group_name -> {tensor_name: ndarray, ..., 'meta': dict}``.
        warp_fn: Warper callable.
        n_samples: Warped time-axis length.

    Returns:
        tensors_warped: Same nested structure, but tensors are warped and each
        group's ``meta`` is replaced by warped metadata.
    """
    out: dict[str, dict[str, Any]] = {}
    
    if isinstance(n_samples, int):
        n_samples = {group_name: n_samples for group_name in tensors.keys()}
    
    if isinstance(n_samples, Mapping):
        missing_groups = set(tensors.keys()) - set(n_samples.keys())
        if missing_groups:
            raise ValueError(f"`n_samples` is missing groups: {missing_groups}")

    for group_name, group in tensors.items():
        if not isinstance(group, Mapping):
            raise ValueError(f"Group '{group_name}' must be a mapping, got {type(group)}.")

        if "meta" not in group:
            raise ValueError(f"Group '{group_name}' is missing required key 'meta'.")

        src_meta = dict(group["meta"])
        if "axes" not in src_meta or "time" not in src_meta["axes"]:
            raise ValueError(f"Group '{group_name}' meta must contain axes['time'].")

        # Infer sampling rate from this group's time axis (ignore NaNs).
        sr = infer_sfreq_from_times(src_meta["axes"]["time"])

        group_out: dict[str, Any] = {}
        meta_warped: Dict[str, Any] | None = None

        for name, tensor in group.items():
            if name == "meta":
                continue
            if tensor is None:
                continue

            warped, percent_axis, meta_epochs = warp_fn(np.asarray(tensor), sr=float(sr), n_samples=int(n_samples[group_name]))
            group_out[name] = np.asarray(warped, dtype=np.float64)

            if meta_warped is None:
                meta_warped = build_warped_tensor_metadata(
                    src_meta["axes"],
                    np.asarray(percent_axis, dtype=float),
                    meta_epochs,
                    source_meta=src_meta,
                )

        if meta_warped is None:
            # No tensors in this group; keep meta as-is.
            meta_warped = src_meta

        group_out["meta"] = meta_warped
        out[str(group_name)] = group_out

    return out
