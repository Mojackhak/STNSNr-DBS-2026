"""Tests for the pad+concat warper."""

from __future__ import annotations

import numpy as np
import mne

from ..warp.concat import concat_warper
from ..warp.pad_concat import pad_warper
from ..warp.stack import stack_warper


def _build_mock_raw(duration_s: float = 20.0, sfreq: float = 100.0) -> mne.io.BaseRaw:
    n_times = int(duration_s * sfreq)
    info = mne.create_info(ch_names=["misc"], sfreq=sfreq, ch_types=["misc"])
    t = np.arange(n_times) / sfreq
    sig = np.sin(2 * np.pi * 1.0 * t)[None, :]
    raw = mne.io.RawArray(sig, info)

    # One annotation at 10s, duration 2s
    ann = mne.Annotations(onset=[10.0], duration=[2.0], description=["L26 pain"])
    raw.set_annotations(ann)
    return raw


def test_pad_concat_warp_resample():
    raw = _build_mock_raw()
    epochs_by_label, warp_fn = pad_warper(
        raw,
        anno_allowed={"L26": (4.0, 5.0, 5.0, 6.0)},
    )
    assert len(epochs_by_label["ALL"]) == 1

    # Build a time-aligned tensor at a different sampling rate
    sr = 50.0
    n_times = int(20.0 * sr)
    data = np.random.RandomState(0).randn(2, 3, n_times).astype(np.float64)

    warped, pct, meta = warp_fn(data, sr=sr, n_samples=400)
    assert warped.shape == (1, 2, 3, 400)
    assert pct.shape == (400,)
    assert len(meta) == 1


def _build_raw_with_overlap() -> mne.io.BaseRaw:
    info = mne.create_info(ch_names=["misc"], sfreq=100.0, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, 200), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.50, 0.55],
            duration=[0.20, 0.10],
            description=["event", "bad-zone"],
        )
    )
    return raw


def test_stack_warper_keep_and_drop_modes_are_independent() -> None:
    raw = _build_raw_with_overlap()

    epochs_exact_keep, _ = stack_warper(
        raw,
        keep=["eve"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
        require_match=False,
    )
    assert len(epochs_exact_keep["ALL"]) == 0

    epochs_sub_keep, _ = stack_warper(
        raw,
        keep=["eve"],
        mode="substring",
        anno_drop=["bad"],
        drop_mode="exact",
        require_match=False,
    )
    assert len(epochs_sub_keep["ALL"]) == 1

    epochs_drop_sub, _ = stack_warper(
        raw,
        keep=["event"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
        require_match=False,
    )
    assert len(epochs_drop_sub["ALL"]) == 0


def test_concat_warper_uses_drop_mode_for_blacklist_matching() -> None:
    raw = _build_raw_with_overlap()

    epochs_drop_exact, _ = concat_warper(
        raw,
        keep=["event"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="exact",
        require_match=False,
    )
    assert len(epochs_drop_exact["ALL"][0].intervals_s) == 1

    epochs_drop_sub, _ = concat_warper(
        raw,
        keep=["event"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
        require_match=False,
    )
    assert epochs_drop_sub["ALL"][0].intervals_s == []


def test_pad_warper_splits_keep_and_drop_matching_modes() -> None:
    raw = _build_raw_with_overlap()
    anno_allowed = {"eve": (0.0, 0.05, 0.05, 0.0)}

    epochs_exact_keep, warp_fn_exact_keep = pad_warper(
        raw,
        anno_allowed=anno_allowed,
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
    )
    assert len(epochs_exact_keep["ALL"]) == 0
    assert warp_fn_exact_keep is None

    epochs_sub_keep, warp_fn_sub_keep = pad_warper(
        raw,
        anno_allowed=anno_allowed,
        mode="substring",
        anno_drop=["bad"],
        drop_mode="exact",
    )
    assert len(epochs_sub_keep["ALL"]) == 1
    assert warp_fn_sub_keep is not None

    epochs_drop_sub, warp_fn_drop_sub = pad_warper(
        raw,
        anno_allowed={"event": (0.0, 0.05, 0.05, 0.0)},
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
    )
    assert len(epochs_drop_sub["ALL"]) == 0
    assert warp_fn_drop_sub is None
