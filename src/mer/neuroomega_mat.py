"""mer.neuroomega_mat

Main purpose
------------
Read NeuroOmega (Alpha Omega) intraoperative MER exports in MATLAB ``.mat``
format and standardize them into Python-friendly structures.

This module focuses on two NeuroOmega signal types commonly exported per depth:

* CLFP: low-pass (<~300 Hz) channel intended for LFP
* CSPK: high-pass (>~300 Hz) channel intended for spikes

It can optionally export per-signal CSV files and also pickle the extracted
metadata + samples for convenient downstream analysis.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .io_utils import ensure_dir, list_files

logger = logging.getLogger(__name__)


@dataclass
class MerMatReadConfig:
    """Configuration for reading NeuroOmega MER ``.mat`` exports."""

    mer_noise_range_uv: tuple[float, float] = (2.0, 7.0)
    """Expected background noise (microvolts) used to infer amplifier gain."""

    csv_column_style: Literal["legacy", "standard"] = "legacy"
    """CSV column naming scheme.

    * ``legacy``: ``Time(s)``, ``Voltage(uV)``
    * ``standard``: ``time_s``, ``voltage_uv``
    """


@dataclass
class MerSignal:
    """A single MER signal channel extracted from a NeuroOmega ``.mat`` file."""

    base: str
    channel_type: str
    channel_index: int
    time_s: np.ndarray
    signal_uv: np.ndarray
    sampling_frequency_hz: float
    bit_resolution_uv: float
    gain: float
    gain_source: Literal["declared", "inferred"]

    # Optional filename-derived metadata
    side: str | None = None
    tract: str | None = None
    depth_label: str | None = None
    depth_mm: float | None = None

    # Provenance
    source_file: str | None = None

    unit_time: str = "s"
    unit_signal: str = "uV"
    unit_sampling_frequency: str = "Hz"

    def to_dataframe(self, *, column_style: Literal["legacy", "standard"] = "legacy") -> pd.DataFrame:
        """Convert to a 2-column DataFrame (time, voltage)."""
        if column_style == "legacy":
            return pd.DataFrame({"Time(s)": self.time_s, "Voltage(uV)": self.signal_uv})
        if column_style == "standard":
            return pd.DataFrame({"time_s": self.time_s, "voltage_uv": self.signal_uv})
        raise ValueError(f"Unknown column_style: {column_style}")

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a dict compatible with the legacy ``read_OA_MER`` output."""
        return {
            "base": self.base,
            "time": self.time_s,
            "data": self.signal_uv,
            "bit_resolution": self.bit_resolution_uv,
            "gain": self.gain,
            "sr": self.sampling_frequency_hz,
            "unit_sr": self.unit_sampling_frequency,
            "unit_time": self.unit_time,
            "unit_voltage": self.unit_signal,
            "side": self.side,
            "tract": self.tract,
            "depth": self.depth_label,
        }


@dataclass
class MerReadResult:
    """Container for NeuroOmega MER signals read from a folder."""

    patient_name: str
    clfp: dict[str, MerSignal]
    cspk: dict[str, MerSignal]

    def to_legacy_dicts(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Return two legacy dictionaries: (CLFP_dict, CSPK_dict)."""
        clfp_legacy = {k: v.to_legacy_dict() for k, v in self.clfp.items()}
        cspk_legacy = {k: v.to_legacy_dict() for k, v in self.cspk.items()}
        return clfp_legacy, cspk_legacy


def _mat_scalar(x: Any) -> float:
    """Convert a MATLAB scalar (often a 1x1 array) to float."""
    arr = np.asarray(x).squeeze()
    try:
        return float(arr)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Cannot convert to float scalar: type={type(x)} shape={getattr(arr, 'shape', None)}") from exc


def _mat_1d(x: Any) -> np.ndarray:
    """Convert a MATLAB vector-like object to a 1D numpy array."""
    arr = np.asarray(x).squeeze()
    if arr.ndim == 0:
        return np.asarray([arr], dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(float, copy=False)


def _infer_gain(
    ad_samples: np.ndarray,
    bit_resolution_uv: float,
    declared_gain: float | None,
    mer_noise_range_uv: tuple[float, float],
) -> tuple[float, Literal["declared", "inferred"]]:
    """Infer gain if missing or outside plausible noise-based range."""
    if ad_samples.size == 0:
        return (declared_gain if declared_gain is not None else 1.0, "declared" if declared_gain is not None else "inferred")

    noise_lo, noise_hi = mer_noise_range_uv
    if noise_lo <= 0 or noise_hi <= 0 or noise_lo >= noise_hi:
        raise ValueError(f"Invalid mer_noise_range_uv: {mer_noise_range_uv}")

    # Range of plausible gain values given an assumed noise amplitude.
    # This follows the heuristic used in the original script.
    gain_range = np.mean(np.abs(ad_samples)) * bit_resolution_uv / np.array([noise_lo, noise_hi], dtype=float)
    gain_min = float(np.min(gain_range))
    gain_max = float(np.max(gain_range))

    if declared_gain is not None and gain_min <= declared_gain <= gain_max:
        return float(declared_gain), "declared"

    # Heuristic defaults from common MER systems.
    if gain_max < 500:
        return 20.0, "inferred"  # Neuro Omega
    if gain_min > 500:
        return 1385.0, "inferred"  # NeuroNav
    return 1.0, "inferred"


def _parse_filename_metadata(basename: str) -> tuple[str | None, str | None, str | None, float | None]:
    """Best-effort metadata parsing from a filename stem.

    The original script assumed a strict convention (side/tract/depth by fixed
    string slicing). In practice, intraoperative naming varies widely.

    This parser is deliberately conservative:
    * side: first character if it is alphabetic (e.g. "L" or "R")
    * tract: first two digits after side (e.g. "01") if present
    * depth: last numeric token in the basename (float if possible)

    Returns
    -------
    (side, tract, depth_label, depth_mm)
    """
    side: str | None = None
    tract: str | None = None

    if basename and basename[0].isalpha():
        side = basename[0]
        m = re.match(r"^[A-Za-z](\d{2})", basename)
        if m:
            tract = m.group(1)

    # Extract the last signed number as a depth hint.
    nums = re.findall(r"[-+]?\d*\.?\d+", basename)
    depth_label: str | None = nums[-1] if nums else None
    depth_mm: float | None = None
    if depth_label is not None:
        try:
            depth_mm = float(depth_label)
        except ValueError:
            depth_mm = None

    return side, tract, depth_label, depth_mm


def _find_channel_indices(mat_dict: dict[str, Any], channel_type: str) -> list[str]:
    """Return all channel indices available for a given channel_type.

    Example: keys like "CLFP_1", "CLFP_2" -> returns ["1", "2"].
    """
    pat = re.compile(rf"^{re.escape(channel_type)}_(\d+)$")
    indices: list[str] = []
    for key in mat_dict.keys():
        m = pat.match(key)
        if m:
            indices.append(m.group(1))
    return sorted(indices, key=lambda s: int(s))


def _extract_mer_signal(
    mat_dict: dict[str, Any],
    *,
    channel_type: str,
    channel_index: str,
    basename: str,
    source_file: Path,
    config: MerMatReadConfig,
) -> MerSignal:
    """Extract a single (channel_type, channel_index) from the MATLAB dict."""
    ad_key = f"{channel_type}_{channel_index}"
    bit_key = f"{channel_type}_{channel_index}_BitResolution"
    gain_key = f"{channel_type}_{channel_index}_Gain"
    sr_key = f"{channel_type}_{channel_index}_KHz"

    if ad_key not in mat_dict:
        raise KeyError(f"Missing key '{ad_key}' in {source_file.name}")
    if bit_key not in mat_dict:
        raise KeyError(f"Missing key '{bit_key}' in {source_file.name}")
    if sr_key not in mat_dict:
        raise KeyError(f"Missing key '{sr_key}' in {source_file.name}")

    ad_samples = _mat_1d(mat_dict[ad_key])
    bit_resolution_uv = _mat_scalar(mat_dict[bit_key])
    sr_khz = _mat_scalar(mat_dict[sr_key])
    sampling_frequency_hz = float(sr_khz * 1000.0)

    declared_gain: float | None = None
    if gain_key in mat_dict:
        try:
            declared_gain = _mat_scalar(mat_dict[gain_key])
        except Exception:
            declared_gain = None

    gain, gain_source = _infer_gain(
        ad_samples,
        bit_resolution_uv,
        declared_gain,
        config.mer_noise_range_uv,
    )

    # Convert samples to microvolts.
    signal_uv = ad_samples * bit_resolution_uv / gain

    # Time vector in seconds.
    if sampling_frequency_hz <= 0:
        raise ValueError(f"Invalid sampling frequency in {source_file.name}: {sampling_frequency_hz}")
    time_s = np.arange(signal_uv.size, dtype=float) / sampling_frequency_hz

    side, tract, depth_label, depth_mm = _parse_filename_metadata(basename)

    return MerSignal(
        base=basename,
        channel_type=channel_type,
        channel_index=int(channel_index),
        time_s=time_s,
        signal_uv=signal_uv,
        sampling_frequency_hz=sampling_frequency_hz,
        bit_resolution_uv=bit_resolution_uv,
        gain=gain,
        gain_source=gain_source,
        side=side,
        tract=tract,
        depth_label=depth_label,
        depth_mm=depth_mm,
        source_file=str(source_file),
    )


def read_neuroomega_mer_mat_folder(
    folder_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    export_csv: bool = True,
    export_pickle: bool = True,
    recursive: bool = False,
    config: MerMatReadConfig | None = None,
) -> MerReadResult:
    """Read all NeuroOmega MER ``.mat`` files in a folder.

    Parameters
    ----------
    folder_path:
        Directory containing NeuroOmega ``.mat`` files.
    output_dir:
        Output directory for CSV / pickle exports. If None, nothing is written.
    export_csv:
        If True, export one CSV per extracted signal.
    export_pickle:
        If True, export two pickle files in the legacy dict format.
    recursive:
        If True, also search in subfolders.
    config:
        Reader configuration. If None, defaults are used.

    Returns
    -------
    MerReadResult
        A structured container with CLFP and CSPK signals.
    """
    cfg = config or MerMatReadConfig()

    folder = Path(folder_path)
    patient_name = folder.name

    mat_files = list_files(folder, extensions=(".mat",), recursive=recursive)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under: {folder}")

    clfp: dict[str, MerSignal] = {}
    cspk: dict[str, MerSignal] = {}

    for mat_file in mat_files:
        mat_dict = loadmat(mat_file)
        basename = mat_file.stem

        for channel_type, out_dict in (("CLFP", clfp), ("CSPK", cspk)):
            indices = _find_channel_indices(mat_dict, channel_type)
            if not indices:
                logger.debug("No %s channels found in %s", channel_type, mat_file.name)
                continue

            for idx_str in indices:
                try:
                    sig = _extract_mer_signal(
                        mat_dict,
                        channel_type=channel_type,
                        channel_index=idx_str,
                        basename=basename,
                        source_file=mat_file,
                        config=cfg,
                    )
                except Exception as exc:
                    logger.warning("Failed to extract %s_%s from %s: %s", channel_type, idx_str, mat_file.name, exc)
                    continue

                # Use basename as the default key unless multiple channels exist.
                key = basename if len(indices) == 1 else f"{basename}_ch{idx_str}"
                out_dict[key] = sig

                if output_dir is not None and export_csv:
                    out_dir = ensure_dir(output_dir)
                    df = sig.to_dataframe(column_style=cfg.csv_column_style)
                    csv_name = f"{patient_name}_{basename}_{channel_type}{idx_str}.csv"
                    df.to_csv(out_dir / csv_name, index=False)

    result = MerReadResult(patient_name=patient_name, clfp=clfp, cspk=cspk)

    if output_dir is not None and export_pickle:
        out_dir = ensure_dir(output_dir)
        clfp_legacy, cspk_legacy = result.to_legacy_dicts()
        with open(out_dir / f"{patient_name}_CLFP_dict.pickle", "wb") as f:
            pickle.dump(clfp_legacy, f)
        with open(out_dir / f"{patient_name}_CSPK_dict.pickle", "wb") as f:
            pickle.dump(cspk_legacy, f)

    return result


# -----------------------------------------------------------------------------
# Legacy compatibility wrappers
# -----------------------------------------------------------------------------

def read_oa_mer(folder_path: str | Path, save_path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Legacy wrapper compatible with the original ``read_OA_MER`` function.

    This wrapper preserves the legacy return type (two dictionaries), while
    fixing critical issues (e.g., time vector computation).

    Parameters
    ----------
    folder_path:
        Folder containing NeuroOmega ``.mat`` files.
    save_path:
        Folder where CSV and pickle files are written.

    Returns
    -------
    (CLFP_dict, CSPK_dict)
        Legacy dicts matching the historical structure.
    """
    result = read_neuroomega_mer_mat_folder(
        folder_path,
        output_dir=save_path,
        export_csv=True,
        export_pickle=True,
        recursive=False,
        config=MerMatReadConfig(csv_column_style="legacy"),
    )
    return result.to_legacy_dicts()

