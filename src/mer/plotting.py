"""mer.plotting

Main purpose
------------
Convenience plotting utilities for MER/spike analysis.

These functions are intended for interactive use (e.g., Jupyter) and therefore
favor simple, explicit figures.

SpikeInterface is an optional dependency; imports are performed lazily.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def plot_waveforms(
    analyzer: object,
    *,
    unit_ids: Iterable[int] | None = None,
    zscore_amplitude: bool = True,
    show: bool = True,
) -> list["matplotlib.figure.Figure"]:
    """Plot waveforms and templates for units in a SpikeInterface analyzer.

    Parameters
    ----------
    analyzer:
        A SpikeInterface SortingAnalyzer with waveform + template extensions.
    unit_ids:
        Units to plot. If None, plots all units.
    zscore_amplitude:
        If True, z-score waveform amplitudes per unit before plotting.
    show:
        If True, call ``plt.show()`` per figure.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One figure per unit.

    Raises
    ------
    ImportError
        If matplotlib or SpikeInterface is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plot_waveforms().") from exc

    try:
        # Only to ensure the analyzer API is present; not used directly.
        import spikeinterface.full as si  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for plot_waveforms().") from exc

    _ = si  # silence linter

    if unit_ids is None:
        unit_ids = getattr(analyzer, "unit_ids")

    unit_ids_list = list(unit_ids)

    sr_hz = float(analyzer.recording.get_sampling_frequency())

    figures: list["matplotlib.figure.Figure"] = []

    for unit_id in unit_ids_list:
        sub = analyzer.select_units(unit_ids=[unit_id])

        waveforms = sub.get_extension("waveforms").get_data().squeeze()
        template = sub.get_extension("templates").get_data(operator="average").squeeze()

        waveforms = np.asarray(waveforms, dtype=float)
        template = np.asarray(template, dtype=float)

        # Keep plotting robust when a unit has only one extracted spike.
        waveforms = np.atleast_2d(waveforms)
        template = template.reshape(-1)

        if zscore_amplitude:
            mean_amp = float(np.mean(waveforms))
            std_amp = float(np.std(waveforms))
            if std_amp > 0.0:
                waveforms = (waveforms - mean_amp) / std_amp
                template = (template - mean_amp) / std_amp
            else:
                waveforms = waveforms - mean_amp
                template = template - mean_amp

        t_ms = np.arange(template.size, dtype=float) / sr_hz * 1000.0

        fig, ax = plt.subplots(figsize=(4, 3))
        for wf in waveforms:
            ax.plot(t_ms, wf, color="gray", alpha=0.5, linewidth=0.7)
        ax.plot(t_ms, template, color="red", linewidth=1.5, label="mean")

        ax.set_title(f"Unit {unit_id}  (n={len(waveforms)} spikes)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (z-score)" if zscore_amplitude else "Amplitude (uV)")
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()

        figures.append(fig)

    return figures
