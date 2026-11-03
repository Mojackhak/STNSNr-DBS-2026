"""Tests for annotation-based masking.

These tests validate that masking keeps only time bins covered by selected
Raw annotations, setting all other values to NaN.
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

from ..mask.annotations import (
    drop_raw_annotations,
    filter_raw_annotations,
    mask_by_annotations,
    time_mask_by_annotations,
)


def _make_raw_with_state_annotations(sfreq: float = 10.0, duration_s: float = 10.0) -> mne.io.BaseRaw:
    n_times = int(duration_s * sfreq)
    info = mne.create_info(ch_names=["x"], sfreq=sfreq, ch_types=["misc"])
    data = np.zeros((1, n_times), dtype=float)
    raw = mne.io.RawArray(data, info)

    ann = mne.Annotations(
        onset=[1.0, 4.0, 7.0],
        duration=[2.0, 1.0, 1.0],
        description=["sit", "gait", "noise"],
    )
    raw.set_annotations(ann)
    return raw


def test_time_mask_by_annotations_substring() -> None:
    raw = _make_raw_with_state_annotations()
    times = raw.times
    keep_mask, info = time_mask_by_annotations(raw, times, keep=["sit", "gait"], mode="substring")

    expected = ((times >= 1.0) & (times <= 3.0)) | ((times >= 4.0) & (times <= 5.0))
    assert keep_mask.shape == times.shape
    assert np.array_equal(keep_mask, expected)
    assert info["n_keep"] == int(np.sum(expected))


def test_mask_by_annotations_sets_nan_outside_kept_intervals() -> None:
    raw = _make_raw_with_state_annotations()
    times = raw.times

    tensor = np.ones((2, 3, times.size), dtype=float)
    masked, keep_mask, _ = mask_by_annotations(raw, tensor, times, keep=["sit", "gait"])

    assert masked.shape == tensor.shape
    assert np.all(np.isfinite(masked[..., keep_mask]))
    assert np.all(np.isnan(masked[..., ~keep_mask]))


def test_time_mask_require_match_raises() -> None:
    raw = _make_raw_with_state_annotations()
    times = raw.times

    with pytest.raises(ValueError, match="No Raw annotations matched"):
        time_mask_by_annotations(raw, times, keep=["does_not_exist"], require_match=True)


def _make_raw_for_overlap_tests(sfreq: float = 10.0) -> mne.io.BaseRaw:
    """Create a Raw with one keep interval and several overlapping annotations."""
    n_times = int(12.0 * sfreq)
    info = mne.create_info(ch_names=["x"], sfreq=sfreq, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, n_times), dtype=float), info)

    # Keep interval: [1, 5]
    # Fully-covered annotation: inside keep
    # Partially-overlapping annotation: crosses keep boundary
    # Outside annotation: disjoint
    ann = mne.Annotations(
        onset=[1.0, 2.0, 4.0, 7.0],
        duration=[4.0, 1.0, 2.0, 1.0],
        description=["keep", "inside", "partial", "outside"],
    )
    raw.set_annotations(ann)
    return raw


def test_filter_raw_annotations_overlap_policy_drop_keeps_fully_covered() -> None:
    raw = _make_raw_for_overlap_tests()

    raw_masked, info = filter_raw_annotations(raw, keep=["keep"], overlap_policy="drop")
    desc = list(raw_masked.annotations.description)

    assert "keep" in desc
    assert "inside" in desc  # fully covered by keep union -> must survive
    assert "partial" not in desc  # partial overlap -> dropped when overlap_policy='drop'
    assert "outside" not in desc  # outside keep -> dropped

    assert info["n_keep_exact"] == 1
    assert info["n_keep_covered"] == 1


def test_filter_raw_annotations_overlap_policy_split_splits_partial() -> None:
    raw = _make_raw_for_overlap_tests()

    raw_masked, _ = filter_raw_annotations(raw, keep=["keep"], overlap_policy="split")
    anns = raw_masked.annotations

    # Partial annotation [4, 6] should be trimmed to [4, 5] (keep union is [1, 5])
    idx = [i for i, d in enumerate(anns.description) if d == "partial"]
    assert len(idx) == 1
    i = idx[0]
    assert anns.onset[i] == pytest.approx(4.0)
    assert anns.duration[i] == pytest.approx(1.0)


def test_drop_raw_annotations_split_removes_only_overlap() -> None:
    sfreq = 10.0
    n_times = int(10.0 * sfreq)
    info = mne.create_info(ch_names=["x"], sfreq=sfreq, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, n_times), dtype=float), info)

    # Drop interval: [2, 4]
    # 'a' overlaps -> should keep [1,2] after splitting
    # 'b' outside -> unchanged
    # 'c' fully inside drop -> removed
    ann = mne.Annotations(
        onset=[2.0, 1.0, 5.0, 2.5],
        duration=[2.0, 2.0, 1.0, 1.0],
        description=["drop", "a", "b", "c"],
    )
    raw.set_annotations(ann)

    raw_masked, _ = drop_raw_annotations(raw, drop=["drop"], overlap_policy="split")
    anns = raw_masked.annotations

    # drop annotation itself removed
    assert "drop" not in list(anns.description)

    # 'a' should be trimmed to [1,2]
    idx_a = [i for i, d in enumerate(anns.description) if d == "a"]
    assert len(idx_a) == 1
    i = idx_a[0]
    assert anns.onset[i] == pytest.approx(1.0)
    assert anns.duration[i] == pytest.approx(1.0)

    # 'b' should remain
    assert "b" in list(anns.description)

    # 'c' should be removed
    assert "c" not in list(anns.description)
