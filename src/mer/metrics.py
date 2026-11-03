"""mer.metrics

Main purpose
------------
Compute commonly used intraoperative MER features and spike-train metrics.

The functions here are intentionally small and composable so they can be used
from notebooks, scripts, or larger pipelines.

Some metrics depend on SpikeInterface objects (e.g., SortingAnalyzer). Those
imports are performed lazily to keep this package import-safe when
SpikeInterface is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_rms(
    rms_dict: dict[str, pd.DataFrame],
    *,
    depth_col: str = "depth",
    rms_col: str = "RMS",
    nrms_col: str = "nRMS",
    n_baseline: int = 5,
) -> dict[str, pd.DataFrame]:
    """Add a normalized RMS column (nRMS) to each DataFrame in a dict.

    The baseline is defined as the mean RMS of the top-N largest depth rows.

    Parameters
    ----------
    rms_dict:
        Mapping from side (e.g. "L", "R") to a DataFrame with at least
        ``depth_col`` and ``rms_col``.
    depth_col:
        Name of the depth column (numeric; larger = more dorsal).
    rms_col:
        Name of the raw RMS column.
    nrms_col:
        Name of the output normalized RMS column.
    n_baseline:
        How many rows with the largest depth to average for baseline.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A new dict with updated DataFrames.
    """
    if n_baseline <= 0:
        raise ValueError("n_baseline must be >= 1")

    out: dict[str, pd.DataFrame] = {}
    for side, df in rms_dict.items():
        if depth_col not in df.columns or rms_col not in df.columns:
            raise KeyError(f"{side}: expected columns '{depth_col}' and '{rms_col}'")

        df2 = df.copy()
        top_n = df2.nlargest(n_baseline, depth_col)
        baseline = float(top_n[rms_col].mean())

        if not np.isfinite(baseline) or baseline == 0.0:
            raise ValueError(f"{side}: cannot compute baseline (value={baseline})")

        df2[nrms_col] = df2[rms_col] / baseline
        out[side] = df2

    return out


def compute_lvr(isis_s: np.ndarray, *, refractory_s: float = 0.002) -> float:
    """Compute LvR (local variation with refractoriness) from an ISI series.

    LvR = (3/(n-1)) * sum_{i=1..n-1} [ (1 - 4 I_i I_{i+1}/(I_i+I_{i+1})^2)
                                      * (1 + 4R/(I_i+I_{i+1})) ]

    Parameters
    ----------
    isis_s:
        Inter-spike intervals in seconds.
    refractory_s:
        Refractoriness constant R (seconds). Default = 2 ms.

    Returns
    -------
    float
        LvR value (np.nan if fewer than 2 ISIs).
    """
    isis = np.asarray(isis_s, dtype=float)
    n = isis.size
    if n < 2:
        return float("nan")

    i = isis[:-1]
    j = isis[1:]
    denom = (i + j)

    term1 = 1.0 - (4.0 * i * j) / (denom**2)
    term2 = 1.0 + (4.0 * refractory_s) / denom

    return float((3.0 / (n - 1)) * np.sum(term1 * term2))


def compute_lvr_from_analyzer(analyzer: object, *, refractory_s: float = 0.002) -> dict[int, float]:
    """Compute LvR for each unit in a SpikeInterface analyzer.

    Parameters
    ----------
    analyzer:
        A SpikeInterface SortingAnalyzer (or compatible) with a ``sorting``
        attribute.
    refractory_s:
        Refractoriness constant in seconds.

    Returns
    -------
    dict[int, float]
        Mapping unit_id -> LvR.

    Raises
    ------
    ImportError
        If SpikeInterface is not installed.
    """
    try:
        import spikeinterface.full as si  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for compute_lvr_from_analyzer().") from exc

    _ = si  # suppress unused import warning for type checkers

    unit_ids = list(getattr(analyzer, "unit_ids"))
    lvr: dict[int, float] = {}

    for unit_id in unit_ids:
        sub = analyzer.select_units(unit_ids=[unit_id])
        sorting = sub.sorting
        fs = float(sorting.get_sampling_frequency())
        spikes = sorting.get_unit_spike_train(unit_id)
        isis_s = np.diff(spikes) / fs
        lvr[int(unit_id)] = compute_lvr(isis_s, refractory_s=refractory_s)

    return lvr


def compute_burst_index(analyzer: object, *, window_ms: float = 1000.0) -> dict[int, float]:
    """Compute a Poisson-surprise-based Burst Index (BI) per unit.

    Definition used in the legacy implementation:

    BI = (1 / modal_ISI) / mean_firing_rate

    Where modal_ISI is computed from an ISI histogram.

    Parameters
    ----------
    analyzer:
        SpikeInterface SortingAnalyzer with sorting and recording.
    window_ms:
        ISI histogram window in milliseconds.

    Returns
    -------
    dict[int, float]
        Mapping unit_id -> burst index.

    Raises
    ------
    ImportError
        If SpikeInterface is not installed.
    """
    try:
        import spikeinterface.full as si  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for compute_burst_index().") from exc

    unit_ids = list(getattr(analyzer, "unit_ids"))
    out: dict[int, float] = {}

    for unit_id in unit_ids:
        sub = analyzer.select_units(unit_ids=[unit_id])
        recording = sub.recording
        sorting = sub.sorting

        counts, bins = si.compute_isi_histograms(sub, window_ms=window_ms)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        # For a single selected unit, counts has shape (1, n_bins).
        modal_idx = int(np.argmax(counts, axis=1)[0])
        modal_isi_ms = float(bin_centers[modal_idx])

        duration_s = float(recording.get_duration())
        if duration_s <= 0:
            raise ValueError("Recording duration must be > 0")

        mean_rate_hz = len(sorting.get_unit_spike_train(unit_id)) / duration_s
        if mean_rate_hz <= 0:
            out[int(unit_id)] = float("nan")
            continue

        # Units cancel out as long as modal_isi and mean_rate are consistent.
        out[int(unit_id)] = (1.0 / modal_isi_ms) / (mean_rate_hz / 1000.0)

    return out
