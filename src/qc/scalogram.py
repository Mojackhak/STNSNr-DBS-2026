from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str, max_len: int = 180) -> str:
    """Make a filesystem-friendly filename stem."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[\\/:\*\?\"<>\|\n\r\t]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:max_len] if len(name) > max_len else name


def _coerce_channel_indices(
    *,
    raw_ch_names: Sequence[str],
    picks: Sequence[str] | Sequence[int] | None,
) -> list[int]:
    """
    Convert flexible picks into channel indices.

    Supported:
      - None or "all": all channels
      - "data": all non-stim channels (robust default for LFP-like data)
      - Sequence[str]: channel names
      - Sequence[int]: channel indices
    """
    import mne

    n_chan = len(raw_ch_names)

    if picks is None or picks == "all":
        return list(range(n_chan))

    if isinstance(picks, (list, tuple)) and len(picks) > 0 and isinstance(picks[0], str):
        return mne.pick_channels(raw_ch_names, include=list(picks), ordered=True).tolist()

    if isinstance(picks, (list, tuple)) and (len(picks) == 0 or isinstance(picks[0], int)):
        idxs = [int(x) for x in picks]
        for i in idxs:
            if i < 0 or i >= n_chan:
                raise IndexError(f"Channel index out of range: {i} (n_channels={n_chan}).")
        return idxs

    raise TypeError(
        "picks must be None, 'all', 'data', a sequence of channel names, or a sequence of indices."
    )


def save_channel_scalograms_from_raw(
    raw,
    out_dir: str | Path,
    *,
    freqs_hz: np.ndarray | None = None,
    n_cycles: float | np.ndarray = 7.0,
    tmin_s: float | None = None,
    tmax_s: float | None = None,
    picks: Sequence[str] | Sequence[int] | None = None,
    decim: int = 1,
    power_scale: str = "db",
    eps: float = 1e-120,
    dpi: int = 200,
    image_format: str = "png",
    overwrite: bool = True,
    save_metadata_json: bool = True,
    n_jobs: int | None = None,
) -> list[Path]:
    """
    Generate and save a scalogram (time-frequency power) image per channel from an MNE Raw object.

    This uses Morlet wavelets via mne.time_frequency.tfr_array_morlet on a single "epoch"
    (the selected raw segment) per channel.

    Args:
        raw: mne.io.BaseRaw-like instance.
        out_dir: Output folder to store images (and optional JSON sidecars).
        freqs_hz: Frequencies of interest in Hz. If None, defaults to 2..120 Hz (60 bins).
        n_cycles: Morlet n_cycles (float or array-like per frequency).
        tmin_s: Start time in seconds (relative to raw start). If None, start at 0.
        tmax_s: Stop time in seconds (relative to raw start). If None, stop at raw end.
        picks: Which channels to plot:
            - None: all channels
            - list of channel names
            - list of channel indices
        decim: Decimation factor (>=1) to speed up computation and reduce image width.
        power_scale: "db" (10*log10), "log10", or "linear".
        eps: Small constant to avoid log(0).
        dpi: Saved image DPI.
        image_format: "png", "jpg", "pdf", ... supported by matplotlib.
        overwrite: If False, skip channels whose output image already exists.
        save_metadata_json: If True, save a JSON sidecar per channel with parameters.
        n_jobs: Parallel jobs for wavelet computation (passed to MNE).

    Returns:
        List of saved image paths (one per channel actually saved).

    Notes:
        - For long recordings, computing full-resolution scalograms can be expensive.
          Use tmin_s/tmax_s and/or decim to control runtime and output size.
        - Raw data are assumed to be in Volts (MNE convention). Power units are V^2
          before log scaling.
    """
    import mne
    from mne.time_frequency import tfr_array_morlet

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if freqs_hz is None:
        freqs_hz = np.linspace(2.0, 120.0, 60, dtype=float)
    else:
        freqs_hz = np.asarray(freqs_hz, dtype=float)

    if freqs_hz.ndim != 1 or freqs_hz.size < 2:
        raise ValueError("freqs_hz must be a 1D array with at least 2 frequencies.")

    if decim < 1:
        raise ValueError("decim must be >= 1.")

    power_scale = power_scale.lower().strip()
    if power_scale not in {"db", "log10", "linear"}:
        raise ValueError("power_scale must be one of: 'db', 'log10', 'linear'.")

    image_format = image_format.lower().strip().lstrip(".")
    if not image_format:
        raise ValueError("image_format must be a non-empty string (e.g., 'png').")

    sfreq = float(raw.info["sfreq"])
    n_total = int(raw.n_times)

    seg_tmin = 0.0 if tmin_s is None else float(tmin_s)
    seg_tmax = (n_total / sfreq) if tmax_s is None else float(tmax_s)

    if seg_tmin < 0:
        raise ValueError("tmin_s must be >= 0.")
    if seg_tmax <= seg_tmin:
        raise ValueError("tmax_s must be > tmin_s.")
    if seg_tmin > (n_total / sfreq):
        raise ValueError("tmin_s is beyond the end of the Raw object.")

    start_idx = int(np.round(seg_tmin * sfreq))
    stop_idx = int(np.round(seg_tmax * sfreq))
    start_idx = max(0, min(start_idx, n_total))
    stop_idx = max(0, min(stop_idx, n_total))
    if stop_idx <= start_idx:
        raise ValueError("Selected segment is empty after index conversion.")

    pick_indices = _coerce_channel_indices(
        raw_ch_names=raw.ch_names,
        picks=picks,
    )

    if len(pick_indices) == 0:
        raise ValueError("No channels selected by picks; nothing to plot.")

    saved: list[Path] = []

    logger.info(
        "Saving scalograms: n_channels=%d, segment=[%.3fs, %.3fs), sfreq=%.3f Hz, decim=%d",
        len(pick_indices),
        seg_tmin,
        seg_tmax,
        sfreq,
        decim,
    )

    # Precompute time axis for the selected segment (decimated)
    seg_len = stop_idx - start_idx
    times_s = (np.arange(seg_len, dtype=float) + start_idx) / sfreq
    times_s = times_s[::decim]

    for ch_idx in pick_indices:
        ch_name = raw.ch_names[ch_idx]
        safe_name = _sanitize_filename(ch_name)

        img_name = (
            f"{safe_name}__scalogram_{freqs_hz.min():.1f}-{freqs_hz.max():.1f}Hz."
            f"{image_format}"
        )
        img_path = out_path / img_name

        if img_path.exists() and not overwrite:
            logger.info("Skipping existing file (overwrite=False): %s", img_path)
            continue

        # Read only this channel + this time segment (works without raw.preload=True)
        x = raw.get_data(picks=[ch_idx], start=start_idx, stop=stop_idx)  # shape (1, n_times)
        if x.shape[1] != seg_len:
            raise RuntimeError(
                f"Unexpected segment length for {ch_name}: got {x.shape[1]}, expected {seg_len}."
            )

        data = x[np.newaxis, :, :]  # (n_epochs=1, n_channels=1, n_times)

        power = tfr_array_morlet(
            data,
            sfreq=sfreq,
            freqs=freqs_hz,
            n_cycles=n_cycles,
            output="power",
            decim=decim,
            n_jobs=n_jobs,
        )  # (1, 1, n_freqs, n_times_out)

        power_2d = np.asarray(power[0, 0, :, :], dtype=float)  # (n_freqs, n_times_out)

        if power_scale == "db":
            z = 10.0 * np.log10(power_2d + eps)
            z_label = "Power (dB)"
        elif power_scale == "log10":
            z = np.log10(power_2d + eps)
            z_label = "log10(Power)"
        else:
            z = power_2d
            z_label = "Power (V^2)"

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(times_s, freqs_hz, z, shading="auto")
        ax.set_title(f"Scalogram (Morlet) - {ch_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(z_label)

        fig.tight_layout()
        fig.savefig(img_path, dpi=dpi)
        plt.close(fig)

        saved.append(img_path)

        if save_metadata_json:
            meta = {
                "channel_name": ch_name,
                "channel_index": int(ch_idx),
                "sfreq_hz": sfreq,
                "tmin_s": seg_tmin,
                "tmax_s": seg_tmax,
                "start_sample": int(start_idx),
                "stop_sample": int(stop_idx),
                "decim": int(decim),
                "freqs_hz": freqs_hz.tolist(),
                "n_cycles": (float(n_cycles) if np.isscalar(n_cycles) else np.asarray(n_cycles).tolist()),
                "power_scale": power_scale,
                "image_format": image_format,
                "mne_version": getattr(mne, "__version__", "unknown"),
            }
            json_path = img_path.with_suffix(".json")
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        logger.info("Saved scalogram: %s", img_path)

    return saved
