"""mer.waveclus_pipeline

Main purpose
------------
Reusable helpers for the Wave_clus-based intraoperative MER pipeline.

This module is designed to keep the interactive notebook/script (wc_sorting.py)
short and readable by factoring out repetitive steps:

- Convert/organize Wave_clus input/output files
- Read NeuroExplorer/NEX streams into SpikeInterface recordings
- Parse and standardize filename-based metadata (side/tract/depth)
- Create SpikeInterface analyzers and compute standard extensions
- Export QC figures and summary tables

Notes
-----
This module targets interactive use and prioritizes clarity and robustness over
micro-optimizations. Heavy dependencies (SpikeInterface, statsmodels, etc.) are
imported lazily inside functions when possible.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .io_utils import ensure_dir
from .metrics import compute_burst_index, compute_lvr_from_analyzer
from .plotting import plot_waveforms
from .spikeinterface_adapter import recording_to_mat

logger = logging.getLogger(__name__)


_DEPTH_PATTERN = re.compile(r"D([-+]?\d*\.\d+|\d+)F")


def parse_base_metadata(base: str) -> tuple[str, str, float]:
    """Parse (side, tract, depth) from a NEX/recording base name.

    The legacy pipeline assumed the naming convention:
        <side><tract>...D<depth>F...

    Examples
    --------
    - "L02D6.5F"  -> side="L", tract="02", depth=6.5
    - "R01D-1F"   -> side="R", tract="01", depth=-1.0

    Parameters
    ----------
    base:
        File stem without extension (e.g., "L02D6.5F").

    Returns
    -------
    tuple[str, str, float]
        side, tract, depth.

    Raises
    ------
    ValueError
        If depth cannot be parsed.
    """
    base = str(base)
    side = base[0] if len(base) >= 1 else ""
    tract = base[1:3] if len(base) >= 3 else ""

    match = _DEPTH_PATTERN.search(base)
    if match is None:
        raise ValueError(
            f"Cannot parse depth from base='{base}'. Expected a token like 'D6.5F'. "
            "If your filenames use a different convention, update _DEPTH_PATTERN."
        )
    depth = float(match.group(1))
    return side, tract, depth


def annotate_with_base_metadata(obj: Any, base: str) -> None:
    """Annotate a SpikeInterface object with standardized base/side/tract/depth."""
    side, tract, depth = parse_base_metadata(base)
    obj.annotate(base=base, side=side, tract=tract, depth=depth)


def assert_single_tract_per_side(objs: Sequence[Any], *, label: str = "objects") -> dict[str, str]:
    """Ensure each side maps to exactly one tract.

    This is useful for downstream STN boundary workflows that aggregate by
    ``side`` only and would otherwise mix multiple trajectories together.
    """
    tract_map: dict[str, dict[str, list[str]]] = {}

    for i, obj in enumerate(objs):
        try:
            side = str(obj.get_annotation("side"))
            tract = str(obj.get_annotation("tract"))
            base = str(obj.get_annotation("base"))
        except Exception as exc:
            raise ValueError(
                f"{label}[{i}] is missing required annotations 'side'/'tract'/'base'."
            ) from exc

        if not side or not tract:
            raise ValueError(
                f"{label}[{i}] has an empty side/tract annotation (base='{base}', "
                f"side='{side}', tract='{tract}')."
            )

        tract_map.setdefault(side, {}).setdefault(tract, []).append(base)

    multi_tract_sides = {side: tracts for side, tracts in tract_map.items() if len(tracts) > 1}
    if multi_tract_sides:
        details = []
        for side, tracts in sorted(multi_tract_sides.items()):
            tract_summary = ", ".join(
                f"{tract}: {len(sorted(bases))} files" for tract, bases in sorted(tracts.items())
            )
            details.append(f"{side} -> {tract_summary}")
        raise ValueError(
            "Multiple tracts detected within the same side. "
            "This STN boundary workflow assumes a single tract per side. "
            f"{label}: " + "; ".join(details)
        )

    return {side: next(iter(tracts.keys())) for side, tracts in tract_map.items()}


def read_neuroexplorer_recordings(
    file_paths: Sequence[str | Path],
    *,
    stream_name: str,
    bandpass_hz: tuple[float, float] | None = None,
    notch_freqs_hz: Sequence[float] | None = None,
    all_annotations: bool = True,
    use_names_as_ids: bool = True,
    show_traces: bool = False,
) -> list[Any]:
    """Read a NeuroExplorer/NEX stream into SpikeInterface recordings.

    Parameters
    ----------
    file_paths:
        List of ``.nex`` files.
    stream_name:
        Stream to read, e.g. "SPK 02" or "LFP 01".
    bandpass_hz:
        Optional (fmin, fmax) bandpass filter.
        REF: https://pubmed.ncbi.nlm.nih.gov/16763982
        REF: https://www.sciencedirect.com/science/article/pii/S0969996122001395
    notch_freqs_hz:
        Optional list of notch frequencies (Hz), applied sequentially.
    all_annotations, use_names_as_ids:
        Passed through to ``si.read_neuroexplorer``.
    show_traces:
        If True, plot traces for each recording.

    Returns
    -------
    list[spikeinterface.core.BaseRecording]
        One recording per file.

    Raises
    ------
    ImportError
        If SpikeInterface is not installed.
    """
    try:
        import spikeinterface.full as si  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for read_neuroexplorer_recordings().") from exc

    recordings: list[Any] = []
    for fp in file_paths:
        path = Path(fp)
        base = path.stem

        rec = si.read_neuroexplorer(
            file_path=str(path),
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
        )

        if bandpass_hz is not None:
            rec = si.bandpass_filter(rec, freq_min=bandpass_hz[0], freq_max=bandpass_hz[1])

        if notch_freqs_hz is not None:
            for f in notch_freqs_hz:
                rec = si.notch_filter(rec, freq=float(f))

        annotate_with_base_metadata(rec, base)
        recordings.append(rec)

        if show_traces:
            si.plot_traces(rec)

    return recordings


def export_recordings_to_waveclus_mat(
    recordings: Sequence[Any],
    output_dir: str | Path,
    *,
    segment_index: int = 0,
) -> list[Path]:
    """Export recordings to Wave_clus-compatible ``.mat`` files.

    The output ``.mat`` contains fields:
    - data: ndarray (n_samples, n_channels)
    - sr: float (Hz)

    Parameters
    ----------
    recordings:
        SpikeInterface recordings.
    output_dir:
        Destination folder.
    segment_index:
        Segment index to export (default: 0).

    Returns
    -------
    list[pathlib.Path]
        Written ``.mat`` file paths.
    """
    out_dir = ensure_dir(output_dir)
    written: list[Path] = []

    for rec in recordings:
        base = rec.get_annotation("base")
        out_path = out_dir / f"{base}.mat"
        recording_to_mat(rec, out_path, segment_index=segment_index)
        written.append(out_path)

    return written


def organize_waveclus_outputs(
    mat_dir: str | Path,
    sorting_dir: str | Path,
) -> dict[str, list[Path]]:
    """Move Wave_clus output files into per-base subfolders.

    The original notebook moved only a subset of the Wave_clus outputs. In
    practice, Wave_clus may generate multiple sidecar files (e.g.,
    ``times_*.mat``, ``cluster_class_*.mat``, ``spikes_*.mat``, ``*.txt``,
    ``*.png``). For reproducibility and debugging, this helper moves **all**
    files that contain the recording base token, except the original input
    ``<base>.mat`` file.

    Parameters
    ----------
    mat_dir:
        Folder where Wave_clus wrote its outputs (often the same as inputs).
    sorting_dir:
        Destination folder with per-recording subfolders.

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Mapping base -> moved file paths.
    """
    mat_dir_p = Path(mat_dir)
    sorting_dir_p = ensure_dir(sorting_dir)

    moved: dict[str, list[Path]] = {}

    # Raw input files are typically "<base>.mat".
    for raw_mat in sorted(mat_dir_p.glob("*.mat")):
        if raw_mat.name.startswith("times_"):
            # Skip output files when determining bases.
            continue

        base = raw_mat.stem
        dest = ensure_dir(sorting_dir_p / base)
        moved.setdefault(base, [])

        # Move every file that contains the base token, except the raw input itself.
        for p in sorted(mat_dir_p.glob(f"*{base}*")):
            if p == raw_mat:
                continue

            target = dest / p.name
            ensure_dir(target.parent)
            shutil.move(str(p), str(target))
            moved[base].append(target)

    return moved


def load_waveclus_sortings(
    recordings: Sequence[Any],
    sorting_dir: str | Path,
    *,
    keep_good_only: bool = True,
) -> list[Any]:
    """Read Wave_clus sorting results into SpikeInterface Sorting objects.

    Parameters
    ----------
    recordings:
        Recording list (used only for metadata + base names).
    sorting_dir:
        Folder with per-base subfolders produced by :func:`organize_waveclus_outputs`.
    keep_good_only:
        Passed to WaveClusSortingExtractor.

    Returns
    -------
    list[spikeinterface.core.BaseSorting]
        Sortings with annotations:
        - base, side, tract, depth, recording_i
    """
    try:
        from spikeinterface.extractors import WaveClusSortingExtractor as WCSX  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for load_waveclus_sortings().") from exc

    sorting_dir_p = Path(sorting_dir)
    sortings: list[Any] = []

    for i, rec in enumerate(recordings):
        base = rec.get_annotation("base")
        subfolder = sorting_dir_p / base
        times_file = subfolder / f"times_{base}.mat"
        if not times_file.exists():
            logger.warning("Missing Wave_clus times file: %s", times_file)
            continue

        sorting = WCSX(file_path=str(times_file), keep_good_only=keep_good_only)
        annotate_with_base_metadata(sorting, base)
        sorting.annotate(recording_i=i)
        sortings.append(sorting)

    return sortings


def make_single_contact_probe(
    *,
    radius: float = 5.0,
    ypitch: float = 20.0,
) -> Any:
    """Create a 1-contact probe for single-channel MER recordings."""
    try:
        from probeinterface import generate_linear_probe  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("probeinterface is required for make_single_contact_probe().") from exc

    probe = generate_linear_probe(
        num_elec=1,
        ypitch=ypitch,
        contact_shapes="circle",
        contact_shape_params={"radius": radius},
    )
    probe.set_device_channel_indices(np.arange(1))
    return probe


def create_sorting_analyzers(
    recordings: Sequence[Any],
    sortings: Sequence[Any],
    probe: Any,
) -> list[Any]:
    """Create SpikeInterface analyzers for matching recording/sorting pairs."""
    try:
        import spikeinterface.full as si  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for create_sorting_analyzers().") from exc

    analyzers: list[Any] = []

    for sorting in sortings:
        recording_i = int(sorting.get_annotation("recording_i"))
        rec = recordings[recording_i].set_probe(probe)

        base = sorting.get_annotation("base")
        logger.info("Creating analyzer for base=%s", base)

        analyzer = si.create_sorting_analyzer(sorting=sorting, recording=rec)
        analyzers.append(analyzer)

    return analyzers


def compute_default_analyzer_extensions(
    analyzers: Sequence[Any],
    *,
    ms_before: float = 1.0,
    ms_after: float = 2.0,
    peak_relative_threshold: float | None = 0.2,
    min_thresh_detect_peaks_troughs: float | None = None,
) -> None:
    """Compute a standard set of SpikeInterface extensions for each analyzer.

    REF: https://spikeinterface.readthedocs.io/en/stable/modules/postprocessing.html
    """
    try:
        import spikeinterface.full as si  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for compute_default_analyzer_extensions().") from exc

    try:
        from spikeinterface.metrics.template import (  # type: ignore
            get_default_template_metrics_params as get_default_tm_params,
        )
    except Exception:
        try:
            from spikeinterface.metrics.template import get_default_tm_params  # type: ignore
        except Exception:
            try:
                from spikeinterface.postprocessing.template_metrics import get_default_tm_params  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "Unable to import SpikeInterface template metrics defaults. "
                    "Your installed SpikeInterface version may use an unsupported API."
                ) from exc

    for analyzer in analyzers:
        analyzer.compute("random_spikes", method="uniform")
        analyzer.compute("waveforms", ms_before=ms_before, ms_after=ms_after)
        analyzer.compute("templates", operators=["average", "median", "std"])

        analyzer.compute("noise_levels")
        analyzer.compute("correlograms")
        analyzer.compute("spike_amplitudes")
        analyzer.compute("spike_locations")
        analyzer.compute("unit_locations")
        analyzer.compute("template_similarity")
        analyzer.compute("amplitude_scalings")
        analyzer.compute("isi_histograms")
        analyzer.compute("principal_components")

        peak_detection_threshold = min_thresh_detect_peaks_troughs
        if peak_detection_threshold is None:
            peak_detection_threshold = peak_relative_threshold

        metric_params = get_default_tm_params(None)
        for params in metric_params.values():
            if isinstance(params, dict) and "peak_relative_threshold" in params:
                params["peak_relative_threshold"] = float(peak_detection_threshold)

        template_metrics_kwargs = {"metric_params": metric_params}
        if peak_detection_threshold is not None:
            template_metrics_kwargs["min_thresh_detect_peaks_troughs"] = float(peak_detection_threshold)

        try:
            analyzer.compute("template_metrics", **template_metrics_kwargs)
        except TypeError:
            # Older SpikeInterface releases use only per-metric params and do not
            # accept the newer top-level threshold keyword.
            template_metrics_kwargs.pop("min_thresh_detect_peaks_troughs", None)
            analyzer.compute("template_metrics", **template_metrics_kwargs)

        analyzer.compute("quality_metrics")


def save_analyzer_qc_figures(
    analyzers: Sequence[Any],
    output_dir: str | Path,
    *,
    close_figures: bool = True,
) -> None:
    """Save common QC figures (waveforms + basic raster/ISI/correlograms)."""
    try:
        import matplotlib.pyplot as plt
        import spikeinterface.full as si  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("matplotlib and SpikeInterface are required for save_analyzer_qc_figures().") from exc

    out_dir = ensure_dir(output_dir)

    for analyzer in analyzers:
        rec = analyzer.recording
        sorting = analyzer.sorting
        base = rec.get_annotation("base")

        fig_subdir = ensure_dir(out_dir / base)

        # Waveforms (one figure per unit)
        figs = plot_waveforms(analyzer, show=False)
        for j, fig in enumerate(figs):
            fig.savefig(fig_subdir / f"waveforms_unit_{j}.pdf")
            if close_figures:
                plt.close(fig)

        # Base characteristics
        w_ts = si.plot_timeseries(rec)
        w_ts.figure.savefig(fig_subdir / f"timeseries_{base}.pdf")
        if close_figures:
            plt.close(w_ts.figure)

        w_rs = si.plot_rasters(sorting)
        w_rs.figure.savefig(fig_subdir / f"rasters_{base}.pdf")
        if close_figures:
            plt.close(w_rs.figure)

        w_isi = si.plot_isi_distribution(sorting, window_ms=150.0, bin_ms=3.0)
        w_isi.figure.savefig(fig_subdir / f"isi_distribution_{base}.pdf")
        if close_figures:
            plt.close(w_isi.figure)

        w_ach = si.plot_autocorrelograms(sorting, window_ms=150.0, bin_ms=3.0)
        w_ach.figure.savefig(fig_subdir / f"autocorrelograms_{base}.pdf")
        if close_figures:
            plt.close(w_ach.figure)

        w_cch = si.plot_crosscorrelograms(sorting, window_ms=150.0, bin_ms=3.0)
        w_cch.figure.savefig(fig_subdir / f"crosscorrelograms_{base}.pdf")
        if close_figures:
            plt.close(w_cch.figure)

        w_pr = si.plot_unit_presence(sorting)
        w_pr.figure.savefig(fig_subdir / f"unit_presence_{base}.pdf")
        if close_figures:
            plt.close(w_pr.figure)


def compute_rms_df_from_recordings(recordings: Sequence[Any]) -> pd.DataFrame:
    """Compute RMS per recording (single-channel expected).

    Returns a DataFrame with columns: side, depth, RMS, base.
    """
    rows: list[dict[str, Any]] = []
    for rec in recordings:
        base = rec.get_annotation("base")
        side = rec.get_annotation("side")
        depth = float(rec.get_annotation("depth"))
        data = np.asarray(rec.get_traces()).squeeze()
        rms = float(np.sqrt(np.mean(np.square(data))))
        rows.append({"side": side, "depth": depth, "RMS": rms, "base": base})
    return pd.DataFrame(rows)


def add_side_baseline_and_normalize(
    df: pd.DataFrame,
    *,
    side_col: str,
    depth_col: str,
    value_col: str,
    baseline_top_n: int = 5,
    baseline_stat: str = "median",
    baseline_col: str = "baseline",
    normalized_col: str = "normalized",
) -> pd.DataFrame:
    """Add per-side baseline + normalized column to a DataFrame.

    Baseline is computed from the top-N largest depth rows per side.

    REF (baseline region selection idea): https://pubmed.ncbi.nlm.nih.gov/16763982/
    """
    if baseline_top_n <= 0:
        raise ValueError("baseline_top_n must be >= 1")

    if baseline_stat not in {"mean", "median"}:
        raise ValueError("baseline_stat must be 'mean' or 'median'")

    df2 = df.copy()
    df2[depth_col] = pd.to_numeric(df2[depth_col], errors="coerce")
    df2[value_col] = pd.to_numeric(df2[value_col], errors="coerce")

    def _baseline(g: pd.DataFrame) -> float:
        top = g.sort_values(depth_col, ascending=False).head(baseline_top_n)
        if baseline_stat == "mean":
            return float(top[value_col].mean())
        return float(top[value_col].median())

    baseline_per_side = df2.groupby(side_col, group_keys=False).apply(_baseline)
    df2[baseline_col] = df2[side_col].map(baseline_per_side)
    df2[normalized_col] = df2[value_col] / df2[baseline_col]

    return df2


def loess_smooth_by_side(
    df: pd.DataFrame,
    *,
    side_col: str,
    x_col: str,
    y_col: str,
    frac: float = 0.2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute LOESS-smoothed curves per side.

    Parameters
    ----------
    df:
        Input DataFrame.
    side_col:
        Side column name (e.g., "L"/"R").
    x_col, y_col:
        Columns to smooth.
    frac:
        LOESS smoothing parameter.

    Returns
    -------
    (x_dict, y_dict):
        Dicts mapping side -> smoothed arrays.
    """
    try:
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover
        raise ImportError("statsmodels is required for LOESS smoothing.") from exc

    x_dict: dict[str, np.ndarray] = {}
    y_dict: dict[str, np.ndarray] = {}

    for side, g in df.groupby(side_col):
        x = pd.to_numeric(g[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)

        # statsmodels.lowess expects x as the second argument.
        smooth = sm.nonparametric.lowess(y, x, frac=float(frac))
        x_dict[str(side)] = smooth[:, 0]
        y_dict[str(side)] = smooth[:, 1]

    return x_dict, y_dict


def compute_burst_lvr_df(analyzers: Sequence[Any]) -> pd.DataFrame:
    """Compute Burst Index and LvR per unit for a list of analyzers.

    REF: https://www.nature.com/articles/s41598-020-74196-5
    """
    rows: list[dict[str, Any]] = []

    for analyzer in analyzers:
        rec = analyzer.recording
        base = rec.get_annotation("base")
        side = rec.get_annotation("side")
        depth = float(rec.get_annotation("depth"))

        bi = compute_burst_index(analyzer)
        lvr = compute_lvr_from_analyzer(analyzer)

        for unit_id, burst_index in bi.items():
            rows.append(
                {
                    "side": side,
                    "depth": depth,
                    "base": base,
                    "unit_id": int(unit_id),
                    "burst_index": float(burst_index),
                    "lvr": float(lvr.get(int(unit_id), np.nan)),
                }
            )

    return pd.DataFrame(rows)


def export_waveform_properties_csv(
    analyzers: Sequence[Any],
    burst_df: pd.DataFrame,
    output_csv_path: str | Path,
) -> pd.DataFrame:
    """Export per-unit waveform/quality metrics to a single CSV.

    The output table includes:
    - side, depth, base, unit_id, label (if available)
    - noise_levels (single-channel assumed)
    - burst_index, lvr (joined from burst_df)
    - all template_metrics and quality_metrics columns from SpikeInterface

    Parameters
    ----------
    analyzers:
        List of SpikeInterface analyzers with computed extensions.
    burst_df:
        DataFrame from :func:`compute_burst_lvr_df`.
    output_csv_path:
        Destination CSV path.

    Returns
    -------
    pandas.DataFrame
        The exported DataFrame.
    """
    out_path = Path(output_csv_path)
    ensure_dir(out_path.parent)

    rows: list[dict[str, Any]] = []

    # Prepare a lookup for burst/lvr (join by base + unit_id).
    burst_lookup = (
        burst_df[["base", "unit_id", "burst_index", "lvr"]]
        .copy()
        .assign(unit_id=lambda d: pd.to_numeric(d["unit_id"], errors="coerce").astype("Int64"))
        .set_index(["base", "unit_id"])
    )

    for analyzer in analyzers:
        rec = analyzer.recording
        base = rec.get_annotation("base")
        side = rec.get_annotation("side")
        depth = float(rec.get_annotation("depth"))

        label_list = {}
        try:
            label_list = rec.get_annotation("label_list")
        except Exception:
            label_list = {}

        noise_levels = analyzer.get_extension("noise_levels").get_data()
        template_metrics = analyzer.get_extension("template_metrics").get_data()
        quality_metrics = analyzer.get_extension("quality_metrics").get_data()

        for unit_id in analyzer.unit_ids:
            unit_id_int = int(unit_id)

            row: dict[str, Any] = {
                "side": side,
                "depth": depth,
                "base": base,
                "unit_id": unit_id_int,
                "label": label_list.get(unit_id, "unknown"),
                "noise_levels": float(np.asarray(noise_levels).reshape(-1)[0]),
            }

            # Burst/LvR join (base, unit_id)
            try:
                burst_row = burst_lookup.loc[(base, unit_id_int)]
                row["burst_index"] = float(burst_row["burst_index"])
                row["lvr"] = float(burst_row["lvr"])
            except Exception:
                row["burst_index"] = float("nan")
                row["lvr"] = float("nan")

            # Add template/quality metrics
            for col in template_metrics.columns:
                row[str(col)] = template_metrics.loc[unit_id, col]
            for col in quality_metrics.columns:
                row[str(col)] = quality_metrics.loc[unit_id, col]

            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    return df_out
