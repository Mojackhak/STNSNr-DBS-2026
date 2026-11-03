"""Unit tests for the synthetic Raw generator used across the test suite."""

from __future__ import annotations

import numpy as np
import mne

from .synthetic_raw import (
    iter_annotation_descriptions,
    make_synthetic_lfp_raw,
    make_synthetic_lfp_raw_bundle,
)


def test_synthetic_raw_channel_count_and_names():
    raw, truth = make_synthetic_lfp_raw(sfreq=100.0, n_channels=4, ch_names=["A", "B", "C", "D"], rng_seed=1)
    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.info["sfreq"] == truth.sfreq
    assert raw.info["nchan"] == 4
    assert raw.ch_names == ["A", "B", "C", "D"]


def test_synthetic_raw_required_annotations_present():
    raw, _ = make_synthetic_lfp_raw(sfreq=100.0, n_channels=3, rng_seed=0)
    descs = set(iter_annotation_descriptions(raw))

    # High-level paradigms.
    assert "sit" in descs
    assert "pain" in descs
    assert "gait" in descs

    # Nested gait structure.
    assert "walk" in descs
    for key in ("strike_L", "strike_R", "off_L", "off_R"):
        assert key in descs

    # Random BAD annotations.
    assert any("bad" in d.lower() for d in descs)


def test_synthetic_raw_has_reasonable_signal_scale():
    raw, _ = make_synthetic_lfp_raw(sfreq=200.0, n_channels=3, rng_seed=0)
    data = raw.get_data()
    assert data.shape[0] == 3
    # Volts scale: should be on the order of microvolts.
    assert np.nanmax(np.abs(data)) < 1e-3


def test_synthetic_raw_has_expected_spectral_signatures_by_paradigm():
    from scipy.signal import welch

    raw, truth = make_synthetic_lfp_raw(sfreq=200.0, ch_names=["A", "B", "C"], rng_seed=0)
    sfreq = float(raw.info["sfreq"])

    x = raw.get_data(picks=[0])[0]

    def psd_on_interval(t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        i0 = int(round(t0 * sfreq))
        i1 = int(round(t1 * sfreq))
        seg = x[i0:i1]
        assert seg.size > 100, "Segment too short for a stable PSD in tests"
        nperseg = int(min(512, seg.size))
        freqs, pxx = welch(seg, fs=sfreq, nperseg=nperseg, detrend=False)
        return freqs, pxx

    def power_at(freqs: np.ndarray, pxx: np.ndarray, f_hz: float) -> float:
        idx = int(np.argmin(np.abs(freqs - f_hz)))
        return float(pxx[idx])

    sit0, sit1 = truth.sit_intervals[0]
    pain0 = min(ep.pain_interval[0] for ep in truth.pain_epochs)
    pain1 = max(ep.pain_interval[1] for ep in truth.pain_epochs)
    gait0, gait1 = truth.gait_interval

    f_sit, p_sit = psd_on_interval(sit0, sit1)
    f_pain, p_pain = psd_on_interval(pain0, pain1)
    f_gait, p_gait = psd_on_interval(gait0, gait1)

    # These are intentionally qualitative checks (ratios), not exact values.
    sit_beta = power_at(f_sit, p_sit, truth.beta_freq_hz)
    sit_theta = power_at(f_sit, p_sit, truth.theta_freq_hz)
    sit_gamma = power_at(f_sit, p_sit, truth.gamma_freq_hz)

    pain_beta = power_at(f_pain, p_pain, truth.beta_freq_hz)
    pain_gamma = power_at(f_pain, p_pain, truth.gamma_freq_hz)

    gait_theta = power_at(f_gait, p_gait, truth.theta_freq_hz)

    # Sit: beta clearly present; little theta/gamma.
    assert sit_beta > 3.0 * sit_theta
    assert sit_beta > 3.0 * sit_gamma

    # Pain: gamma clearly present.
    assert pain_gamma > 2.0 * pain_beta
    assert pain_gamma > 5.0 * sit_gamma

    # Gait: theta is elevated versus sit.
    assert gait_theta > 3.0 * sit_theta


def test_synthetic_raw_bundle_has_three_variants():
    bundle = make_synthetic_lfp_raw_bundle(
        sfreq=100.0,
        ch_names=["A", "B", "C"],
        rng_seed=0,
    )

    assert set(bundle.keys()) == {"concat", "pad", "gait"}

    for raw, _truth in bundle.values():
        descs = set(iter_annotation_descriptions(raw))
        assert {"sit", "pain", "gait", "walk"}.issubset(descs)