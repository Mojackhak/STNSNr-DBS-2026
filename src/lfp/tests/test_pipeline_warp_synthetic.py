"""Integration-style tests: synthetic Raw -> timefreq/connectivity -> mask/drop -> warp.

These tests are intentionally small, but they exercise the same public API
pieces that the real pipeline (e.g. lfp.py) uses.
"""

from __future__ import annotations

import numpy as np
import pytest

import mne

from .synthetic_raw import make_synthetic_lfp_raw

from ..tfr.grid import grid as tfr_grid
from ..pipelines.masking import mask_tensors, drop_tensors
from ..pipelines.warping import warp_tensors
from ..warp.concat import concat_warper
from ..warp.pad_concat import pad_warper
from ..warp.event_anchored_linear import gait_warper


def _beta_freq_index(freqs: np.ndarray, beta_hz: float) -> int:
    freqs_f = np.asarray(freqs, dtype=float)
    return int(np.argmin(np.abs(freqs_f - float(beta_hz))))


def _compute_tfr_tensor(raw: mne.io.BaseRaw):
    out = tfr_grid(
        raw,
        method="morlet",
        fmin=5.0,
        fmax=80.0,
        n_freqs=76,
        log_grid=False,
        average=False,
        time_resolution_s=0.30,
        hop_s=0.05,
        decim=None,
        min_cycles=3.0,
        max_cycles=None,
        n_jobs=1,
    )
    tfr_raw = np.asarray(out["power"][0], dtype=np.float64)  # (ch, freq, time)
    meta = out["metadata"]
    return tfr_raw, meta


def test_concat_warp_recovers_sit_beta_power_level():
    raw, truth = make_synthetic_lfp_raw(sfreq=200.0, n_channels=3, rng_seed=0)
    tfr_raw, tfr_meta = _compute_tfr_tensor(raw)

    tensors = {"tfr": {"raw": tfr_raw, "meta": tfr_meta}}

    # Mask+drop exactly like the pipeline.
    raw_m, ten_m = mask_tensors(raw, tensors, keep=["sit"], mode="exact", pad_s=0.0, clip_to_raw=True, overlap_policy="split")
    raw_m, ten_m = drop_tensors(raw_m, ten_m, drop=["bad", "edge"], mode="substring", pad_s=0.0, clip_to_raw=True, overlap_policy="split")

    _, warp_fn = concat_warper(raw_m, keep=["sit"], mode="exact")
    warped = warp_tensors(ten_m, warp_fn=warp_fn, n_samples=201)

    freqs = np.asarray(tfr_meta["axes"]["freq"], dtype=float)
    beta_idx = _beta_freq_index(freqs, truth.beta_freq_hz)
    ch0 = 0

    # Compare warped beta power to the median beta power in the original sit intervals.
    # Build a keep-mask per interval using the TFR time axis.
    tfr_times = np.asarray(tfr_meta["axes"]["time"], dtype=float)
    keep = np.zeros(tfr_times.shape, dtype=bool)
    for s0, s1 in truth.sit_intervals:
        keep |= (tfr_times >= float(s0)) & (tfr_times < float(s1))

    sit_median = float(np.nanmedian(tfr_raw[ch0, beta_idx, keep]))
    warped_median = float(np.nanmedian(warped["tfr"]["raw"][0, ch0, beta_idx, :]))

    assert np.isfinite(sit_median) and sit_median > 0
    assert np.isfinite(warped_median)
    # Warping should preserve the level up to wavelet smoothing tolerance.
    assert abs(warped_median - sit_median) / sit_median < 0.35


def test_pad_warp_preserves_left_right_power_contrast():
    raw, truth = make_synthetic_lfp_raw(sfreq=200.0, n_channels=3, rng_seed=1)
    tfr_raw, tfr_meta = _compute_tfr_tensor(raw)
    tensors = {"tfr": {"raw": tfr_raw, "meta": tfr_meta}}

    raw_m, ten_m = mask_tensors(raw, tensors, keep=["pain"], mode="exact", pad_s=0.0, clip_to_raw=True, overlap_policy="split")
    raw_m, ten_m = drop_tensors(raw_m, ten_m, drop=["bad", "edge"], mode="substring", pad_s=0.0, clip_to_raw=True, overlap_policy="split")

    # Only allow the labels present in the synthetic spec.
    pain_allowed = {ep.label: [ep.pad_left_s, ep.anno_left_s, ep.anno_right_s, ep.pad_right_s] for ep in truth.pain_epochs}

    _, warp_fn = pad_warper(raw_m, anno_allowed=pain_allowed, duration_range=(0.0, float("inf")))
    warped = warp_tensors(ten_m, warp_fn=warp_fn, n_samples=200)

    freqs = np.asarray(tfr_meta["axes"]["freq"], dtype=float)
    beta_idx = _beta_freq_index(freqs, truth.beta_freq_hz)
    ch0 = 0

    y = warped["tfr"]["raw"][0, ch0, beta_idx, :]
    assert y.shape[-1] == 200

    # For the default spec, left and right segments have equal durations.
    left = float(np.nanmean(y[:100]))
    right = float(np.nanmean(y[100:]))
    assert np.isfinite(left) and np.isfinite(right)
    assert left > right * 1.5


def test_gait_warp_recovers_global_beta_ramp_shape():
    raw, truth = make_synthetic_lfp_raw(sfreq=200.0, n_channels=3, rng_seed=2)
    tfr_raw, tfr_meta = _compute_tfr_tensor(raw)
    tensors = {"tfr": {"raw": tfr_raw, "meta": tfr_meta}}

    raw_m, ten_m = mask_tensors(raw, tensors, keep=["gait"], mode="exact", pad_s=0.0, clip_to_raw=True, overlap_policy="split")
    raw_m, ten_m = drop_tensors(raw_m, ten_m, drop=["bad", "edge"], mode="substring", pad_s=0.0, clip_to_raw=True, overlap_policy="split")

    cycles, warp_fn = gait_warper(
        raw_m,
        cycle_duration_range=(0.6, 2.0),
        anchors_percent=(0.0, 18.0, 50.0, 68.0, 100.0),
        percent_tolerance=10.0,
        anno_allowed=["gait", "walk", "tag", "gait_dual"],
    )
    assert len(cycles["ALL"]) >= 1

    warped = warp_tensors(ten_m, warp_fn=warp_fn, n_samples=101)

    freqs = np.asarray(tfr_meta["axes"]["freq"], dtype=float)
    beta_idx = _beta_freq_index(freqs, truth.beta_freq_hz)
    ch0 = 0

    meta_w = warped["tfr"]["meta"]
    percent = np.asarray(meta_w["axes"]["percent"], dtype=float)
    assert percent.shape == (101,)

    # Expected global ramp envelope (linear in absolute time), composed with a linear warp.
    gait0, gait1 = truth.gait_interval
    gait_span = float(gait1 - gait0)
    assert gait_span > 0

    y_all = warped["tfr"]["raw"][:, ch0, beta_idx, :]
    assert y_all.ndim == 2  # (n_epochs, n_times)

    # Per-epoch correlation with expected trend.
    corrs: list[float] = []
    for ep_i, ep_meta in enumerate(meta_w.get("warp_epochs", [])):
        start_t = float(ep_meta.get("start_t", np.nan))
        end_t = float(ep_meta.get("end_t", np.nan))
        if not np.isfinite(start_t) or not np.isfinite(end_t) or end_t <= start_t:
            continue

        # If anchors match, time mapping is effectively linear.
        t = start_t + (end_t - start_t) * (percent / 100.0)
        env = truth.gait_beta_amp_start + (truth.gait_beta_amp_end - truth.gait_beta_amp_start) * ((t - gait0) / gait_span)
        exp = env**2

        obs = np.asarray(y_all[ep_i], dtype=float)
        mask = np.isfinite(obs) & np.isfinite(exp)
        if mask.sum() < 10:
            continue

        o = obs[mask]
        e = exp[mask]
        # Normalize to compare shape.
        o = (o - o.mean()) / (o.std() + 1e-12)
        e = (e - e.mean()) / (e.std() + 1e-12)
        corr = float(np.corrcoef(o, e)[0, 1])
        corrs.append(corr)

    assert len(corrs) >= 1
    assert float(np.nanmean(corrs)) > 0.6
