"""Synthetic MNE Raw generator for deterministic, testable DBS-LFP pipelines.

This module intentionally lives under ``lfp.tests`` so unit tests can build
compact Raw objects with:

- Fully controlled spectral content (band power via known oscillations)
- Controlled coupling between channels (coherence / PLV / PLI / wPLI etc.)
- Controlled directed structure (GC / PSI via delayed coupling)
- A realistic annotation structure for downstream masking + warping

The goal is not to perfectly emulate real DBS-LFP, but to provide a stable,
reproducible *ground-truth* sandbox for unit/integration tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import mne


@dataclass(frozen=True)
class PainEpochSpec:
    """Specification for one pain-epoch annotation used by pad_warper."""

    label: str
    onset_s: float
    duration_s: float
    pad_left_s: float
    anno_left_s: float
    anno_right_s: float
    pad_right_s: float

    beta_amp_left: float
    beta_amp_mid: float
    beta_amp_right: float

    @property
    def end_s(self) -> float:
        return float(self.onset_s + self.duration_s)

    @property
    def left_window(self) -> tuple[float, float]:
        """[onset - pad_left, onset + anno_left]."""
        return (float(self.onset_s - self.pad_left_s), float(self.onset_s + self.anno_left_s))

    @property
    def right_window(self) -> tuple[float, float]:
        """[end - anno_right, end + pad_right]."""
        return (float(self.end_s - self.anno_right_s), float(self.end_s + self.pad_right_s))

    @property
    def pain_interval(self) -> tuple[float, float]:
        """A 'pain' interval that fully covers pad_warper windows."""
        return (float(self.left_window[0]), float(self.right_window[1]))


@dataclass(frozen=True)
class SyntheticRawTruth:
    """Ground-truth parameters used by tests."""

    sfreq: float
    ch_names: list[str]
    beta_freq_hz: float
    delay_s: float

    sit_intervals: list[tuple[float, float]]
    pain_epochs: list[PainEpochSpec]
    gait_interval: tuple[float, float]

    gait_beta_amp_start: float
    gait_beta_amp_end: float


def _sine(times_s: np.ndarray, freq_hz: float, phase_rad: float = 0.0) -> np.ndarray:
    return np.sin(2.0 * np.pi * float(freq_hz) * times_s + float(phase_rad))


def _mark_interval(mask: np.ndarray, times_s: np.ndarray, start_s: float, end_s: float) -> None:
    start_f = float(start_s)
    end_f = float(end_s)
    if end_f <= start_f:
        return
    mask[(times_s >= start_f) & (times_s < end_f)] = True


def _choose_bad_intervals(
    *,
    rng: np.random.Generator,
    total_dur_s: float,
    protected: Sequence[tuple[float, float]],
    n_bad: int,
    bad_dur_s: float,
) -> list[tuple[float, float]]:
    """Pick non-overlapping-ish BAD intervals outside protected windows."""

    def _overlaps_any(a0: float, a1: float, intervals: Sequence[tuple[float, float]]) -> bool:
        for b0, b1 in intervals:
            if a0 < b1 and a1 > b0:
                return True
        return False

    out: list[tuple[float, float]] = []
    attempts = 0
    while len(out) < int(n_bad) and attempts < 10000:
        attempts += 1
        onset = float(rng.uniform(0.0, max(0.0, float(total_dur_s) - float(bad_dur_s))))
        end = onset + float(bad_dur_s)
        if _overlaps_any(onset, end, protected):
            continue
        if _overlaps_any(onset, end, out):
            continue
        out.append((onset, end))
    return out


def make_synthetic_lfp_raw(
    *,
    sfreq: float = 200.0,
    ch_names: Sequence[str] | None = None,
    n_channels: int = 3,
    rng_seed: int = 0,
    beta_freq_hz: float = 20.0,
    theta_freq_hz: float = 6.0,
    gamma_freq_hz: float = 60.0,
    delay_s: float = 0.01,
    noise_std: float = 0.05,
    sit_intervals: Sequence[tuple[float, float]] | None = None,
    pain_epochs: Sequence[PainEpochSpec] | None = None,
    gait_start_s: float | None = None,
    n_gait_cycles: int = 2,
    gait_cycle_dur_s: float = 1.0,
    anchors_percent: Sequence[float] = (0.0, 18.0, 50.0, 68.0, 100.0),
    gait_beta_amp_start: float = 0.5,
    gait_beta_amp_end: float = 1.0,
    gait_theta_amp: float = 0.35,
    pain_gamma_amp: float = 0.8,
    n_bad: int = 2,
    bad_dur_s: float = 0.15,
) -> tuple[mne.io.BaseRaw, SyntheticRawTruth]:
    """Create a compact synthetic Raw with a realistic annotation scaffold.

    The returned Raw contains at least these annotation descriptions:

    - 'sit'
    - 'pain' + one or more label annotations like '{LABEL} pain'
    - 'gait' containing 'walk'
    - gait events: 'strike_L', 'strike_R', 'off_L', 'off_R'
    - random 'BAD_*' annotations placed outside protected windows

    Notes
    -----
    - Channel-0 is the "source".
    - Channel-1 is a delayed mixture of Channel-0 (directional coupling).
    - Remaining channels are independent noise (weak coupling).
    """

    if float(sfreq) <= 0:
        raise ValueError("sfreq must be > 0")
    if int(n_channels) < 1:
        raise ValueError("n_channels must be >= 1")

    if ch_names is None:
        ch_names_use = [f"ch{i:02d}" for i in range(int(n_channels))]
    else:
        ch_names_use = [str(x) for x in ch_names]
        if len(ch_names_use) != int(n_channels):
            raise ValueError("len(ch_names) must equal n_channels")

    rng = np.random.default_rng(int(rng_seed))

    # --- Default schedules ---
    sit_intervals_use = list(sit_intervals) if sit_intervals is not None else [(0.0, 2.0), (3.0, 5.0)]

    if pain_epochs is None:
        # Small, test-friendly default pad windows (total warped length = 4 s).
        #
        # IMPORTANT: choose the default pain onset *after* the last sit interval so
        # callers can freely override ``sit_intervals`` without creating overlap.
        last_sit_end = 0.0
        for _s0, _s1 in sit_intervals_use:
            last_sit_end = max(last_sit_end, float(_s1))

        # By construction: pain interval starts at (onset - pad_left).
        # Put that one second after the last sit.
        default_pad_left = 1.0
        pain_interval_start = last_sit_end + 1.0
        default_label_onset = pain_interval_start + default_pad_left

        pain_epochs_use = [
            PainEpochSpec(
                label="L26",
                onset_s=default_label_onset,
                duration_s=4.0,
                pad_left_s=default_pad_left,
                anno_left_s=1.0,
                anno_right_s=1.0,
                pad_right_s=1.0,
                beta_amp_left=1.1,
                beta_amp_mid=0.6,
                beta_amp_right=0.25,
            )
        ]
    else:
        pain_epochs_use = list(pain_epochs)

    # Place gait after pain by default.
    last_end = 0.0
    for a, b in sit_intervals_use:
        last_end = max(last_end, float(b))
    for ep in pain_epochs_use:
        last_end = max(last_end, float(ep.pain_interval[1]))
    gait_start = float(gait_start_s) if gait_start_s is not None else float(last_end + 1.0)

    # Gait duration must cover all events (for n cycles, need (n_cycles+0.5) seconds if strikes alternate).
    n_cycles = int(n_gait_cycles)
    if n_cycles < 1:
        raise ValueError("n_gait_cycles must be >= 1")
    cycle_dur = float(gait_cycle_dur_s)
    if cycle_dur <= 0:
        raise ValueError("gait_cycle_dur_s must be > 0")

    gait_end = gait_start + (n_cycles + 0.5) * cycle_dur
    gait_interval = (gait_start, gait_end)

    total_dur_s = max(gait_end + 0.5, last_end + 0.5)
    n_times = int(np.round(total_dur_s * float(sfreq)))
    times_s = np.arange(n_times, dtype=float) / float(sfreq)

    # --- Build amplitude envelopes ---
    beta_env = np.full(times_s.shape, 0.05, dtype=float)
    theta_env = np.zeros(times_s.shape, dtype=float)
    gamma_env = np.zeros(times_s.shape, dtype=float)

    # Sit: strong beta.
    sit_mask = np.zeros(times_s.shape, dtype=bool)
    for s0, s1 in sit_intervals_use:
        _mark_interval(sit_mask, times_s, float(s0), float(s1))
    beta_env[sit_mask] = 1.0

    # Pain: beta changes across left/mid/right windows; gamma present throughout pain intervals.
    protected: list[tuple[float, float]] = []
    protected.extend([(float(a), float(b)) for a, b in sit_intervals_use])

    for ep in pain_epochs_use:
        p0, p1 = ep.pain_interval
        protected.append((p0, p1))

    # Apply pain envelopes explicitly (avoid boolean view tricks).
    for ep in pain_epochs_use:
        p0, p1 = ep.pain_interval
        in_pain = (times_s >= p0) & (times_s < p1)
        gamma_env[in_pain] = float(pain_gamma_amp)

        l0, l1 = ep.left_window
        r0, r1 = ep.right_window

        in_left = (times_s >= l0) & (times_s < l1)
        in_right = (times_s >= r0) & (times_s < r1)
        in_mid = in_pain & (~in_left) & (~in_right)

        beta_env[in_left] = float(ep.beta_amp_left)
        beta_env[in_mid] = float(ep.beta_amp_mid)
        beta_env[in_right] = float(ep.beta_amp_right)

    # Gait: linear ramp in beta + theta.
    protected.append((gait_interval[0], gait_interval[1]))
    # NOTE:
    # Include the end sample in the gait envelope.
    # The event-anchored warper can place a cycle end anchor exactly at
    # ``gait_interval[1]`` (e.g. ``strike_R``). If the envelope is defined as
    # half-open [start, end), that final anchor sample falls back to baseline,
    # introducing an artificial drop right at 100% of the gait cycle.
    in_gait = (times_s >= gait_interval[0]) & (times_s <= gait_interval[1])
    if np.any(in_gait):
        gait_span = float(gait_interval[1] - gait_interval[0])
        frac = (times_s[in_gait] - float(gait_interval[0])) / (gait_span + 1e-12)
        frac = np.clip(frac, 0.0, 1.0)
        beta_env[in_gait] = float(gait_beta_amp_start) + frac * (float(gait_beta_amp_end) - float(gait_beta_amp_start))
        theta_env[in_gait] = float(gait_theta_amp)

    # --- Build signals ---
    x_beta = beta_env * _sine(times_s, beta_freq_hz)
    x_theta = theta_env * _sine(times_s, theta_freq_hz, phase_rad=0.2)
    x_gamma = gamma_env * _sine(times_s, gamma_freq_hz, phase_rad=-0.1)
    x0 = x_beta + x_theta + x_gamma + float(noise_std) * rng.standard_normal(size=n_times)

    delay_samples = int(np.round(float(delay_s) * float(sfreq)))
    if delay_samples < 0:
        raise ValueError("delay_s must be >= 0")

    if int(n_channels) >= 2:
        # Channel-1 is a *causally* delayed version of Channel-0, plus independent noise.
        # Avoid np.roll() here: circular wrap-around leaks "future" samples into the
        # beginning of the signal and can bias directed metrics (e.g. Granger causality).
        if delay_samples == 0:
            x0_delayed = x0
        else:
            pad_val = float(x0[0])
            x0_delayed = np.concatenate(
                [
                    np.full(delay_samples, pad_val, dtype=float),
                    x0[:-delay_samples],
                ]
            )

        x1 = x0_delayed + float(noise_std) * rng.standard_normal(size=n_times)
    else:
        x1 = None

    data = np.zeros((int(n_channels), n_times), dtype=float)
    data[0] = x0
    if x1 is not None:
        data[1] = x1
    for ci in range(2, int(n_channels)):
        data[ci] = float(noise_std) * rng.standard_normal(size=n_times)

    # Scale to Volts (typical DBS-LFP is in microvolts).
    data *= 1e-6

    info = mne.create_info(ch_names=list(ch_names_use), sfreq=float(sfreq), ch_types=["misc"] * int(n_channels))
    raw = mne.io.RawArray(data, info)

    # --- Build annotations ---
    onsets: list[float] = []
    durs: list[float] = []
    descs: list[str] = []

    # Sit blocks.
    for s0, s1 in sit_intervals_use:
        onsets.append(float(s0))
        durs.append(float(s1 - s0))
        descs.append("sit")

    # Pain blocks + label epochs.
    for ep in pain_epochs_use:
        p0, p1 = ep.pain_interval
        onsets.append(float(p0))
        durs.append(float(p1 - p0))
        descs.append("pain")

        onsets.append(float(ep.onset_s))
        durs.append(float(ep.duration_s))
        descs.append(f"{ep.label} pain")

    # Gait + walk blocks.
    onsets.append(float(gait_interval[0]))
    durs.append(float(gait_interval[1] - gait_interval[0]))
    descs.append("gait")

    onsets.append(float(gait_interval[0]))
    durs.append(float(gait_interval[1] - gait_interval[0]))
    descs.append("walk")

    # Gait events (strike/off). We build alternating strikes with a 50% step offset.
    anchors = np.asarray(list(anchors_percent), dtype=float)
    if anchors.ndim != 1 or anchors.size != 5:
        raise ValueError("anchors_percent must contain 5 values")
    if anchors[0] != 0.0 or anchors[-1] != 100.0:
        raise ValueError("anchors_percent must start at 0 and end at 100")

    # L strikes define L cycles, R strikes define R cycles.
    strike_L = gait_start + np.arange(0, n_cycles + 1) * cycle_dur
    strike_R = gait_start + 0.5 * cycle_dur + np.arange(0, n_cycles + 1) * cycle_dur

    off_R = gait_start + (anchors[1] / 100.0) * cycle_dur + np.arange(0, n_cycles + 1) * cycle_dur
    off_L = gait_start + (anchors[3] / 100.0) * cycle_dur + np.arange(0, n_cycles + 1) * cycle_dur

    # Drop events outside gait interval (last partial).
    def _keep_in_interval(arr: np.ndarray) -> np.ndarray:
        return arr[(arr >= gait_interval[0]) & (arr <= gait_interval[1])]

    strike_L = _keep_in_interval(strike_L)
    strike_R = _keep_in_interval(strike_R)
    off_L = _keep_in_interval(off_L)
    off_R = _keep_in_interval(off_R)

    # Add as point annotations.
    for t0 in strike_L:
        onsets.append(float(t0))
        durs.append(0.0)
        descs.append("strike_L")
    for t0 in strike_R:
        onsets.append(float(t0))
        durs.append(0.0)
        descs.append("strike_R")
    for t0 in off_L:
        onsets.append(float(t0))
        durs.append(0.0)
        descs.append("off_L")
    for t0 in off_R:
        onsets.append(float(t0))
        durs.append(0.0)
        descs.append("off_R")

    # Random BAD annotations outside protected windows.
    bad_intervals = _choose_bad_intervals(
        rng=rng,
        total_dur_s=float(total_dur_s),
        protected=protected,
        n_bad=int(n_bad),
        bad_dur_s=float(bad_dur_s),
    )
    for i, (b0, b1) in enumerate(bad_intervals):
        onsets.append(float(b0))
        durs.append(float(b1 - b0))
        descs.append(f"BAD_artifact_{i:02d}")

    ann = mne.Annotations(onset=onsets, duration=durs, description=descs)
    raw.set_annotations(ann)

    truth = SyntheticRawTruth(
        sfreq=float(sfreq),
        ch_names=list(ch_names_use),
        beta_freq_hz=float(beta_freq_hz),
        delay_s=float(delay_s),
        sit_intervals=[(float(a), float(b)) for a, b in sit_intervals_use],
        pain_epochs=pain_epochs_use,
        gait_interval=(float(gait_interval[0]), float(gait_interval[1])),
        gait_beta_amp_start=float(gait_beta_amp_start),
        gait_beta_amp_end=float(gait_beta_amp_end),
    )
    return raw, truth


def iter_annotation_descriptions(raw: mne.io.BaseRaw) -> Iterable[str]:
    """Yield annotation descriptions as plain strings."""
    for d in raw.annotations.description:
        yield str(d)


def make_synthetic_lfp_raw_bundle(
    *,
    sfreq: float = 200.0,
    ch_names: Sequence[str] | None = None,
    rng_seed: int = 0,
) -> dict[str, tuple[mne.io.BaseRaw, SyntheticRawTruth]]:
    """Create a small bundle of Raw objects that stress different warpers.

    The returned dict has three keys:

    - ``"concat"``: richer ``sit`` structure (multiple segments)
    - ``"pad"``: multiple pain epochs with different labels
    - ``"gait"``: more gait cycles

    Each Raw still contains the minimum required annotation vocabulary.
    """

    # NOTE: The generator auto-places pain/gait after the last sit interval
    # when pain_epochs / gait_start_s are not provided.

    # 1) Concat-heavy
    sit_concat = [(0.0, 1.5), (2.0, 3.5), (4.0, 5.5)]
    raw_concat, truth_concat = make_synthetic_lfp_raw(
        sfreq=sfreq,
        ch_names=ch_names,
        rng_seed=rng_seed,
        sit_intervals=sit_concat,
        n_gait_cycles=1,
    )

    # 2) Pad-heavy (two pain epochs)
    sit_pad = [(0.0, 2.0)]
    # Place pain epochs sequentially with a gap so pad windows don't overlap.
    # First epoch: pain interval starts at 3.0 -> onset at 4.0.
    ep1 = PainEpochSpec(
        label="L26",
        onset_s=4.0,
        duration_s=4.0,
        pad_left_s=1.0,
        anno_left_s=1.0,
        anno_right_s=1.0,
        pad_right_s=1.0,
        beta_amp_left=1.1,
        beta_amp_mid=0.6,
        beta_amp_right=0.25,
    )
    # Second epoch starts after ep1 pain interval end + 1 s.
    ep2_onset = (ep1.pain_interval[1] + 1.0) + ep1.pad_left_s
    ep2 = PainEpochSpec(
        label="R26",
        onset_s=ep2_onset,
        duration_s=4.0,
        pad_left_s=1.0,
        anno_left_s=1.0,
        anno_right_s=1.0,
        pad_right_s=1.0,
        beta_amp_left=1.1,
        beta_amp_mid=0.6,
        beta_amp_right=0.25,
    )
    raw_pad, truth_pad = make_synthetic_lfp_raw(
        sfreq=sfreq,
        ch_names=ch_names,
        rng_seed=rng_seed,
        sit_intervals=sit_pad,
        pain_epochs=[ep1, ep2],
        n_gait_cycles=1,
    )

    # 3) Gait-heavy
    raw_gait, truth_gait = make_synthetic_lfp_raw(
        sfreq=sfreq,
        ch_names=ch_names,
        rng_seed=rng_seed,
        n_gait_cycles=4,
    )

    return {
        "concat": (raw_concat, truth_concat),
        "pad": (raw_pad, truth_pad),
        "gait": (raw_gait, truth_gait),
    }
