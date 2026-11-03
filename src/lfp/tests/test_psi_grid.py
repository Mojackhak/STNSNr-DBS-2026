"""Tests for PSI grid computation.

These tests focus on shape/time-axis alignment and basic pair selection.
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

pytest.importorskip("mne_connectivity")

from ..psi.grid import grid as psi_grid


def _make_raw_two_channel_delay(
    *,
    sfreq: float = 200.0,
    duration_s: float = 5.0,
    f_hz: float = 20.0,
    delay_s: float = 0.01,
) -> mne.io.BaseRaw:
    n_times = int(round(duration_s * sfreq))
    times = np.arange(n_times) / sfreq

    x = np.sin(2.0 * np.pi * float(f_hz) * times)
    delay_samples = int(round(delay_s * sfreq))
    y = np.roll(x, delay_samples)

    data = np.vstack([x, y])
    info = mne.create_info(ch_names=["a", "b"], sfreq=sfreq, ch_types=["misc", "misc"])
    return mne.io.RawArray(data, info)


def test_psi_grid_shape_and_pair_selection() -> None:
    raw = _make_raw_two_channel_delay()

    hop_s = 0.05
    tensor, meta = psi_grid(
        raw,
        bands={"beta": (13.0, 30.0)},
        time_resolution_s=0.10,
        pairs=[("a", "b")],
        ordered_pairs=True,
        hop_s=hop_s,
        n_jobs=1,
    )

    sfreq = float(raw.info["sfreq"])
    decim_eff = int(round(sfreq * hop_s))
    expected_n_times = int(np.asarray(raw.times[::decim_eff]).size)

    assert tensor.shape == (1, 1, 1, expected_n_times)
    assert int(meta["axes"]["time"].size) == expected_n_times

    # Pair axis is stored as the "channel" axis (list of (seed, target) tuples)
    assert list(meta["axes"]["channel"]) == [("a", "b")]
