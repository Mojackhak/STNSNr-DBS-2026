"""Tests for burst grid computation."""

from __future__ import annotations

import numpy as np
import mne

from ..burst.grid import grid as burst_grid


def _make_raw_with_sine_burst(
    *,
    sfreq: float = 100.0,
    duration_s: float = 10.0,
    f_hz: float = 20.0,
    burst_onset_s: float = 2.0,
    burst_offset_s: float = 4.0,
    base_amp: float = 0.1,
    burst_amp: float = 1.0,
) -> mne.io.BaseRaw:
    n_times = int(round(duration_s * sfreq))
    times = np.arange(n_times) / sfreq

    amp = np.full(times.shape, base_amp, dtype=float)
    amp[(times >= burst_onset_s) & (times <= burst_offset_s)] = burst_amp

    sig = amp * np.sin(2.0 * np.pi * float(f_hz) * times)
    data = sig.reshape(1, -1)

    info = mne.create_info(ch_names=["x"], sfreq=sfreq, ch_types=["misc"])
    return mne.io.RawArray(data, info)


def test_burst_grid_shape_and_time_axis_length() -> None:
    raw = _make_raw_with_sine_burst()

    hop_s = 0.05
    tensor, meta = burst_grid(
        raw,
        bands={"beta": (13.0, 30.0)},
        percentile=75.0,
        min_cycles=2.0,
        hop_s=hop_s,
    )

    sfreq = float(raw.info["sfreq"])
    decim_eff = int(round(sfreq * hop_s))
    expected_n_times = int(np.asarray(raw.times[::decim_eff]).size)

    assert tensor.shape == (1, 1, 1, expected_n_times)
    assert int(meta["axes"]["time"].size) == expected_n_times


def test_burst_grid_masks_outside_burst_with_nan() -> None:
    raw = _make_raw_with_sine_burst(burst_onset_s=2.0, burst_offset_s=4.0)

    tensor, meta = burst_grid(
        raw,
        bands={"beta": (13.0, 30.0)},
        percentile=75.0,
        min_cycles=2.0,
        hop_s=0.05,
    )

    y = tensor[0, 0, 0]
    t = np.asarray(meta["axes"]["time"], dtype=float)

    inside = (t >= 2.0) & (t <= 4.0)

    # Filtering can smear slightly; use tolerant assertions.
    frac_nan_outside = float(np.mean(np.isnan(y[~inside])))
    frac_finite_inside = float(np.mean(np.isfinite(y[inside])))

    assert frac_nan_outside > 0.8
    assert frac_finite_inside > 0.6

    # Non-NaN samples should reflect the larger burst amplitude.
    if np.any(np.isfinite(y)):
        assert float(np.nanmedian(y)) > 0.3
