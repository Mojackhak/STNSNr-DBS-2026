"""Tests for the event-anchored (gait) piecewise-linear warper."""

from __future__ import annotations

import mne
import numpy as np

from ..warp.event_anchored_linear import gait_warper, linear_warper


def _build_mock_raw(duration_s: float = 5.0, sfreq: float = 100.0) -> mne.io.BaseRaw:
    n_times = int(duration_s * sfreq)
    info = mne.create_info(ch_names=["misc"], sfreq=sfreq, ch_types=["misc"])
    t = np.arange(n_times) / sfreq
    sig = np.sin(2 * np.pi * 3.0 * t)[None, :]
    raw = mne.io.RawArray(sig, info)

    strike_L = [1.0, 2.2, 3.4]
    strike_R = [1.6, 2.8, 4.0]
    off_R = [1.144, 2.344, 3.544]
    off_L = [1.744, 2.944]

    onsets, descs, durs = [], [], []
    for t0 in strike_L:
        onsets.append(t0)
        descs.append("strike_L")
        durs.append(0.0)
    for t0 in strike_R:
        onsets.append(t0)
        descs.append("strike_R")
        durs.append(0.0)
    for t0 in off_L:
        onsets.append(t0)
        descs.append("off_L")
        durs.append(0.0)
    for t0 in off_R:
        onsets.append(t0)
        descs.append("off_R")
        durs.append(0.0)

    # irrelevant annotations outside gait windows
    onsets += [0.2, 4.2]
    durs += [0.2, 0.3]
    descs += ["standing", "turning"]

    ann = mne.Annotations(onset=onsets, duration=durs, description=descs)
    raw.set_annotations(ann)
    return raw


def _build_mock_tensor(duration_s: float = 5.0, sr: float = 250.0, n_ch: int = 2, n_f: int = 3):
    n_times = int(duration_s * sr)
    t = np.arange(n_times) / sr
    data = np.zeros((n_ch, n_f, n_times), dtype=np.float64)
    base_freqs = [5.0, 11.0, 23.0]
    for c in range(n_ch):
        for f in range(n_f):
            data[c, f] = np.sin(2 * np.pi * base_freqs[f] * t + 0.3 * c)
    return data, sr


def test_gait_event_anchored_warp_shapes():
    raw = _build_mock_raw()
    cycles, warp_fn = gait_warper(raw)

    assert len(cycles["L"]) == 2
    assert len(cycles["R"]) == 2
    assert len(cycles["ALL"]) == 4

    tensor, sr = _build_mock_tensor()
    warped, pct, meta = warp_fn(tensor, sr=sr, n_samples=500)

    assert warped.shape == (4, 2, 3, 500)
    assert pct.shape == (500,)
    assert len(meta) == 4


def test_linear_warper_uses_drop_mode_and_keeps_multiple_epochs() -> None:
    info = mne.create_info(ch_names=["misc"], sfreq=100.0, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, 200), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.10, 0.20, 0.50, 0.80, 1.20],
            duration=[0.00, 0.10, 0.00, 0.00, 0.00],
            description=["start", "bad-zone", "end", "start", "end"],
        )
    )

    epochs_exact, _ = linear_warper(
        raw,
        anchors_percent={0.0: "start", 100.0: "end"},
        mode="exact",
        anno_drop=["bad"],
        drop_mode="exact",
    )
    assert len(epochs_exact["ALL"]) == 2

    epochs_sub, _ = linear_warper(
        raw,
        anchors_percent={0.0: "start", 100.0: "end"},
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
    )
    assert len(epochs_sub["ALL"]) == 1
