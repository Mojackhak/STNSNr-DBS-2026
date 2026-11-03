"""mer.spikeinterface_adapter

Main purpose
------------
Adapters between NeuroOmega MER data structures and SpikeInterface objects.

This module is optional: it requires SpikeInterface to be installed.
All SpikeInterface imports are performed lazily so importing dbslfp_mer does not
fail on machines without SpikeInterface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from .io_utils import ensure_dir

logger = logging.getLogger(__name__)


def waveclus_mat_to_recording(file_path: str | Path) -> "spikeinterface.core.BaseRecording":
    """Load a Wave_clus-compatible recording ``.mat`` into a SpikeInterface Recording.

    Expected MATLAB fields
    ----------------------
    data:
        1D or 2D array of samples (microvolts), typically (n_samples, 1).
    sr:
        Sampling rate (Hz).

    Optional fields (passed to recording annotations if present)
    ------------------------------------------------------------
    side, depth, base, unit_sr, unit_time, unit_voltage

    Parameters
    ----------
    file_path:
        Path to Wave_clus ``.mat``.

    Returns
    -------
    spikeinterface.core.BaseRecording
        A single-channel NumpyRecording.

    Raises
    ------
    ImportError
        If SpikeInterface is not installed.
    """
    try:
        from scipy.io import loadmat
        import spikeinterface.extractors as se  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for waveclus_mat_to_recording().") from exc

    mat = loadmat(file_path)
    data = np.asarray(mat["data"], dtype=float)
    data = data.squeeze()
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError(f"Expected single-channel data, got shape={data.shape}")

    sr = float(np.asarray(mat["sr"]).squeeze())

    recording = se.NumpyRecording(
        traces_list=[data],
        sampling_frequency=sr,
        t_starts=None,
        channel_ids=[0],
    )

    # Optional bookkeeping
    annotations: dict[str, Any] = {
        "signal_type": "SPK",
        "is_filtered": True,
    }
    for k in ("side", "depth", "base", "unit_sr", "unit_time", "unit_voltage"):
        if k in mat:
            annotations[k] = np.asarray(mat[k]).squeeze().tolist()

    recording.annotate(**annotations)
    return recording


def mer_dict_to_recordings(
    sig_dict: dict[str, dict[str, Any]],
    *,
    signal_tag: str = "SPK",
) -> list["spikeinterface.core.BaseRecording"]:
    """Convert a legacy MER signal dict to a list of SpikeInterface recordings.

    Parameters
    ----------
    sig_dict:
        Legacy dict produced by ``read_oa_mer`` / ``read_OA_MER``.
    signal_tag:
        A label written into recording annotations.

    Returns
    -------
    list[spikeinterface.core.BaseRecording]
        One single-channel recording per dict entry.

    Raises
    ------
    ImportError
        If SpikeInterface is not installed.
    """
    try:
        import spikeinterface.extractors as se  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for mer_dict_to_recordings().") from exc

    recordings: list["spikeinterface.core.BaseRecording"] = []

    for key, item in sig_dict.items():
        data = np.asarray(item["data"], dtype=float).reshape(-1, 1)
        sr = float(item["sr"])

        recording = se.NumpyRecording(
            traces_list=[data],
            sampling_frequency=sr,
            t_starts=None,
            channel_ids=[0],
        )

        # Standardized annotations
        recording.annotate(
            signal_type=signal_tag,
            is_filtered=True,
            base=item.get("base", key),
            side=item.get("side"),
            depth=item.get("depth"),
            unit_sr=item.get("unit_sr", "Hz"),
            unit_time=item.get("unit_time", "s"),
            unit_voltage=item.get("unit_voltage", "uV"),
        )

        recordings.append(recording)

    return recordings


def recording_to_mat(
    recording: "spikeinterface.core.BaseRecording",
    save_path: str | Path,
    *,
    segment_index: int = 0,
) -> Path:
    """Export a SpikeInterface Recording to MATLAB ``.mat``.

    The resulting ``.mat`` contains:
    - ``data``: ndarray (n_samples, n_channels)
    - ``sr``: float (Hz)

    Parameters
    ----------
    recording:
        Any SpikeInterface recording.
    save_path:
        Output file path (should end with ``.mat``).
    segment_index:
        Recording segment to export.

    Returns
    -------
    pathlib.Path
        Written file path.
    """
    out = Path(save_path)
    if out.suffix.lower() != ".mat":
        raise ValueError("save_path must end with '.mat'")
    ensure_dir(out.parent)

    data = recording.get_traces(segment_index=segment_index)
    sr = float(recording.get_sampling_frequency())

    sio.savemat(str(out), {"data": data, "sr": sr})
    return out


def _is_padlen_boundary_error(exc: Exception) -> bool:
    """Return True when SciPy filtfilt padding fails on a short save chunk."""
    return isinstance(exc, ValueError) and "greater than padlen" in str(exc)


def _should_use_full_segment_chunk_save(recording: Any) -> bool:
    """Return True when a recording is likely to hit filtered chunk edge errors."""
    get_annotation = getattr(recording, "get_annotation", None)
    if not callable(get_annotation):
        return False

    try:
        return bool(recording.get_annotation("is_filtered"))
    except Exception:
        return False


def _get_max_segment_frames(recording: Any) -> int | None:
    """Get the longest segment length for a SpikeInterface recording."""
    get_num_segments = getattr(recording, "get_num_segments", None)
    get_num_frames = getattr(recording, "get_num_frames", None)
    if not callable(get_num_segments) or not callable(get_num_frames):
        return None

    try:
        num_segments = int(recording.get_num_segments())
        segment_lengths = [
            int(recording.get_num_frames(segment_index=segment_index))
            for segment_index in range(num_segments)
        ]
    except Exception:
        return None

    if not segment_lengths:
        return None
    return max(segment_lengths)


def _get_backup_path(path: Path) -> Path:
    """Return a non-existing backup path for an existing output path."""
    candidate = path.with_name(path.name + "_old")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_old_{index}")
        index += 1
    return candidate


def _prepare_save_target(
    target_path: Path,
    *,
    overwrite: bool,
    send2trash: Any,
) -> None:
    """Prepare the target path according to the overwrite policy."""
    if not target_path.exists():
        return

    if not overwrite:
        raise FileExistsError(f"Save target already exists: {target_path}")

    if send2trash is not None:
        send2trash(str(target_path))
        logger.info("Moved existing save target to trash: %s", target_path)
        return

    backup = _get_backup_path(target_path)
    target_path.rename(backup)
    logger.info("Renamed existing save target: %s -> %s", target_path, backup)


def save_spikeinterface_objects(
    objects: list[object],
    output_dir: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    """Save SpikeInterface objects (Recording/Sorting/Analyzer) to disk.

    This is a convenience wrapper around SpikeInterface ``save``/``save_as``.

    Parameters
    ----------
    objects:
        A list of SpikeInterface objects.
    output_dir:
        Folder to write outputs.
    overwrite:
        If True, replace an existing save target before writing. If False,
        raise ``FileExistsError`` when the target already exists.

    Notes
    -----
    Existing save targets are replaced when ``overwrite=True``. They are moved
    to trash when ``send2trash`` is available; otherwise they are renamed to a
    unique backup path.
    For filtered recordings, SpikeInterface can fail on the final binary-writing
    chunk with a SciPy ``padlen`` error. To avoid this boundary condition, this
    helper saves filtered recordings using a chunk size equal to the longest
    segment length so each segment is written in one chunk.
    """
    try:
        import spikeinterface.full as si  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SpikeInterface is required for save_spikeinterface_objects().") from exc

    try:
        from send2trash import send2trash
    except Exception:  # pragma: no cover
        send2trash = None

    out_dir = ensure_dir(output_dir)

    for obj in objects:
        # Determine a basename.
        base: str | None = None
        if hasattr(obj, "recording") and hasattr(obj.recording, "get_annotation"):
            try:
                base = obj.recording.get_annotation("base")
            except Exception:
                base = None
        if base is None and hasattr(obj, "get_annotation"):
            try:
                base = obj.get_annotation("base")
            except Exception:
                base = None
        if base is None:
            base = "spikeinterface_object"

        target_root = out_dir / str(base)
        is_sorting_analyzer = isinstance(obj, getattr(si, "SortingAnalyzer", ()))
        target_path = Path(str(target_root) + ".zarr") if is_sorting_analyzer else target_root

        _prepare_save_target(target_path, overwrite=overwrite, send2trash=send2trash)

        if is_sorting_analyzer:
            obj.save_as(folder=str(target_path), format="zarr")
        else:
            save_kwargs: dict[str, Any] = {}
            chunk_size = None
            if _should_use_full_segment_chunk_save(obj):
                chunk_size = _get_max_segment_frames(obj)
                if chunk_size is not None:
                    save_kwargs["chunk_size"] = chunk_size
                    logger.info(
                        "Saving filtered SpikeInterface recording base=%s with full-segment "
                        "chunk_size=%s to avoid padlen boundary failures.",
                        base,
                        chunk_size,
                    )
            try:
                obj.save(folder=str(target_path), **save_kwargs)
            except Exception as exc:
                if chunk_size is None or not _is_padlen_boundary_error(exc):
                    raise
                raise ValueError(
                    "Unable to save filtered recording even with full-segment chunking. "
                    f"base={base!r}, chunk_size={chunk_size}. "
                    "The recording itself may be shorter than the filter padding requirement."
                ) from exc
