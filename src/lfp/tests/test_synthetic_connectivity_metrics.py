"""Connectivity/interaction metrics on synthetic data.

These tests verify that the *analysis* code produces values consistent with the
known synthetic ground truth:

- Channel-1 is a delayed version of Channel-0 -> high coupling + directionality
- Other channels are independent noise -> low coupling
"""

from __future__ import annotations

import numpy as np
import pytest

import mne

pytest.importorskip("mne_connectivity")

from .synthetic_raw import make_synthetic_lfp_raw

from ..connectivity.grid import grid as conn_grid
from ..psi.grid import grid as psi_grid
from ..tfr.grid import grid as tfr_grid


def _compute_freqs_and_n_times(raw: mne.io.BaseRaw) -> tuple[np.ndarray, int]:
    out = tfr_grid(
        raw,
        method="morlet",
        fmin=5.0,
        fmax=40.0,
        n_freqs=36,
        log_grid=False,
        average=False,
        time_resolution_s=0.30,
        hop_s=0.05,
        decim=None,
        min_cycles=3.0,
        max_cycles=None,
        n_jobs=1,
    )
    freqs = np.asarray(out["metadata"]["axes"]["freq"], dtype=float)
    n_times = int(np.asarray(out["power"], dtype=np.float64)[0].shape[-1])
    return freqs, n_times


def test_undirected_connectivity_is_high_for_coupled_pair_and_low_for_noise_pair():
    raw, truth = make_synthetic_lfp_raw(sfreq=200.0, n_channels=3, rng_seed=0)
    freqs, target_n_times = _compute_freqs_and_n_times(raw)

    pairs = [(truth.ch_names[0], truth.ch_names[1]), (truth.ch_names[0], truth.ch_names[2])]
    methods = ["coh", "plv", "pli", "ciplv", "wpli"]

    for method in methods:
        tensor4d, meta = conn_grid(
            raw,
            freqs=freqs,
            target_n_times=target_n_times,
            method=method,
            pairs=pairs,
            ordered_pairs=False,
            multivariate=False,
            time_resolution_s=0.30,
            hop_s=0.05,
            decim=None,
            min_cycles=3.0,
            max_cycles=None,
        )
        con = np.asarray(tensor4d[0], dtype=float)  # (n_pairs, n_freq, n_time)

        # Focus on a narrow beta neighborhood around the synthetic beta.
        f = np.asarray(meta["axes"]["freq"], dtype=float)
        beta_band = (f >= truth.beta_freq_hz - 2.0) & (f <= truth.beta_freq_hz + 2.0)
        assert beta_band.any()

        ab = float(np.nanmedian(con[0, beta_band, :]))
        ac = float(np.nanmedian(con[1, beta_band, :]))

        assert np.isfinite(ab) and np.isfinite(ac)
        assert ab > 0.5
        assert ac < ab


def test_granger_causality_prefers_true_direction():
    raw, truth = make_synthetic_lfp_raw(sfreq=200.0, n_channels=3, rng_seed=1)
    freqs, target_n_times = _compute_freqs_and_n_times(raw)

    pairs = [(truth.ch_names[0], truth.ch_names[1]), (truth.ch_names[1], truth.ch_names[0])]
    tensor4d, meta = conn_grid(
        raw,
        freqs=freqs,
        target_n_times=target_n_times,
        method="gc",
        pairs=pairs,
        ordered_pairs=True,
        multivariate=True,
        time_resolution_s=0.30,
        hop_s=0.05,
        decim=None,
        min_cycles=3.0,
        max_cycles=None,
        gc_n_lags=10,
    )
    gc = np.asarray(tensor4d[0], dtype=float)  # (n_pairs, n_freq, n_time)
    f = np.asarray(meta["axes"]["freq"], dtype=float)
    beta_band = (f >= truth.beta_freq_hz - 2.0) & (f <= truth.beta_freq_hz + 2.0)

    gc_ab = float(np.nanmedian(gc[0, beta_band, :]))
    gc_ba = float(np.nanmedian(gc[1, beta_band, :]))
    assert np.isfinite(gc_ab) and np.isfinite(gc_ba)
    assert gc_ab > gc_ba


def test_phase_slope_index_has_opposite_sign_across_directions():
    raw, truth = make_synthetic_lfp_raw(sfreq=200.0, n_channels=3, rng_seed=2)

    bands = {"beta": (truth.beta_freq_hz - 4.0, truth.beta_freq_hz + 4.0)}
    pairs = [(truth.ch_names[0], truth.ch_names[1]), (truth.ch_names[1], truth.ch_names[0])]

    psi4d, meta = psi_grid(
        raw,
        bands=bands,
        pairs=pairs,
        ordered_pairs=True,
        time_resolution_s=0.30,
        hop_s=0.05,
        decim=None,
        min_cycles=3.0,
        max_cycles=None,
    )
    psi = np.asarray(psi4d[0], dtype=float)  # (n_pairs, n_bands, n_time)
    assert psi.shape[0] == 2
    assert psi.shape[1] == 1

    ab = float(np.nanmean(psi[0, 0, :]))
    ba = float(np.nanmean(psi[1, 0, :]))
    assert np.isfinite(ab) and np.isfinite(ba)
    # The two directions should have opposite sign (or at least be clearly different).
    assert ab * ba < 0