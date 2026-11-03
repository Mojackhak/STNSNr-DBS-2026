from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
from collections.abc import Sequence as AbcSequence

import matplotlib.pyplot as plt
import numpy as np
from src.tabular.text import join_sequence

logger = logging.getLogger(__name__)


CmapName = str
SaveMode = Literal["heatmap", "spectrum"]


def _sanitize_filename(name: str, max_len: int = 180) -> str:
    """Make a filesystem-friendly filename stem."""
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"[\\/:\*\?\"<>\|\n\r\t]+", "_", s)
    s = re.sub(r"_+", "_", s)
    if not s:
        s = "unnamed"
    return s[:max_len] if len(s) > max_len else s


def _safe_str_label(x: Any, fallback: str) -> str:
    """Convert metadata label to a safe string for titles/filenames."""
    if x is None:
        return fallback
    try:
        s = str(x)
    except Exception:
        return fallback
    s = s.strip()
    return s if s else fallback


def _get_axis(meta: dict[str, Any] | None, key: str | Sequence[str], n: int) -> np.ndarray:
    """Fetch an axis from meta['axes']; fall back to 0..n-1.

    Supports:
      - key as a single str, e.g. "freq"
      - key as a sequence of str, e.g. ["freq", "frequency_hz"] (first hit wins)

    Also supports per-element labels that are Sequence[str] (e.g. multi-part labels).
    In that case, each element will be formatted with join_sequence().
    """

    def _fallback() -> np.ndarray:
        return np.arange(n, dtype=np.float64)

    def _is_str_sequence(x: Any) -> bool:
        return isinstance(x, AbcSequence) and not isinstance(x, (str, bytes))

    def _coerce_labels(values: Any) -> np.ndarray:
        """Coerce a length-n iterable into a 1D axis array with joined str-sequences."""
        try:
            items = list(values)
        except Exception:
            return _fallback()
        if len(items) != int(n):
            return _fallback()

        out: list[Any] = []
        for v in items:
            # If v is something like ("Left", "STN"), join into a single label string.
            if _is_str_sequence(v) and all(isinstance(s, str) for s in v):
                out.append(join_sequence(v))
            else:
                out.append(v)
        return np.asarray(out, dtype=object)

    if meta is None:
        return _fallback()

    axes = meta.get("axes", None)
    if not isinstance(axes, dict):
        return _fallback()

    # Allow multiple candidate keys (first non-None wins).
    keys = [key] if isinstance(key, str) else list(key)
    axis_value: Any | None = None
    for k in keys:
        if isinstance(k, str) and k in axes and axes[k] is not None:
            axis_value = axes[k]
            break
    if axis_value is None:
        return _fallback()

    # Fast path: numeric / string axis already 1D length-n.
    try:
        arr = np.asarray(axis_value)
    except Exception:
        return _fallback()

    # If it didn't become a clean 1D length-n array, fall back to coercing per-element labels.
    if arr.ndim != 1 or int(arr.shape[0]) != int(n):
        return _coerce_labels(axis_value)

    # If it's 1D but object elements are themselves sequences of strings, join them.
    if arr.dtype == object and arr.size == int(n):
        needs_join = False
        for v in arr.tolist():
            if _is_str_sequence(v) and all(isinstance(s, str) for s in v):
                needs_join = True
                break
        if needs_join:
            return _coerce_labels(arr.tolist())

    return arr



def _is_numeric_axis(axis: np.ndarray) -> bool:
    """Return True if axis can be treated as numeric."""
    try:
        arr = np.asarray(axis)
    except Exception:
        return False
    if arr.size == 0:
        return False
    return np.issubdtype(arr.dtype, np.number) and np.all(np.isfinite(arr.astype(np.float64)))


def _validate_tensor_4d(tensor: np.ndarray) -> tuple[int, int, int, int]:
    x = np.asarray(tensor)
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor (epoch, channel, freq, time). Got shape={x.shape}.")
    n_ep, n_ch, n_f, n_t = map(int, x.shape)
    if n_ep <= 0 or n_ch <= 0 or n_f <= 0 or n_t <= 0:
        raise ValueError(f"Tensor shape must be all positive. Got shape={x.shape}.")
    return n_ep, n_ch, n_f, n_t


def _iter_epoch_channel_pairs(n_ep: int, n_ch: int) -> Iterable[tuple[int, int]]:
    for e in range(n_ep):
        for c in range(n_ch):
            yield e, c


def export_tensor_qc_heatmaps(
    tensor: np.ndarray,
    meta: dict[str, Any] | None,
    out_dir: str | Path,
    *,
    cmap: CmapName = "viridis",
    overwrite: bool = True,
    dpi: int = 300,
    title_prefix: str | None = None,
    skip_all_nan: bool = True,
    subdir: str = "heatmap",
    x_scale: str = "linear",
    y_scale: str = "log",
    agg_epoch: bool = True,
) -> list[Path]:
    """Export one heatmap PDF per (epoch, channel).

    Heatmap is a 2D image of shape (n_freq, n_time).

    Args:
        tensor: 4D tensor (epoch, channel, freq, time).
        meta: Metadata dict containing meta['axes'] with keys like 'epoch', 'channel', 'freq', 'time'.
        out_dir: Output folder root.
        cmap: Matplotlib colormap name (default: 'viridis').
        overwrite: Overwrite existing PDFs.
        dpi: Figure DPI.
        title_prefix: Optional title prefix.
        skip_all_nan: If True, skip pages that are entirely NaN.
        subdir: Subfolder under out_dir to store heatmap PDFs.
        agg_epoch: If True, aggregate over epochs before plotting.
    Returns:
        List of saved PDF paths.
    """
    if agg_epoch:
        tensor = np.nanmean(tensor, axis=0, keepdims=True)  # (1, channel, freq, time)
    n_ep, n_ch, n_f, n_t = _validate_tensor_4d(tensor)

    out_root = Path(out_dir)
    out_path = out_root / subdir
    out_path.mkdir(parents=True, exist_ok=True)

    epoch_axis = _get_axis(meta, "epoch", n_ep)
    ch_axis = _get_axis(meta, "channel", n_ch)
    f_axis = _get_axis(meta, "freq", n_f)
    t_axis = _get_axis(meta, "time", n_t)

    use_numeric_f = _is_numeric_axis(f_axis)
    use_numeric_t = _is_numeric_axis(t_axis)

    saved: list[Path] = []

    for e, c in _iter_epoch_channel_pairs(n_ep, n_ch):
        z = np.asarray(tensor[e, c, :, :], dtype=np.float64)  # (freq, time)

        if skip_all_nan and (not np.any(np.isfinite(z))):
            continue

        ep_label = _safe_str_label(epoch_axis[e], fallback=f"epoch_{e:04d}")
        ch_label = _safe_str_label(ch_axis[c], fallback=f"ch_{c:04d}")

        safe_ep = _sanitize_filename(ep_label)
        safe_ch = _sanitize_filename(ch_label)

        fname = f"heatmap_e{e:04d}_ch{c:04d}__{safe_ep}__{safe_ch}.pdf"
        fpath = out_path / fname

        if fpath.exists() and not overwrite:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=int(dpi))

        if use_numeric_t and use_numeric_f:
            # Use imshow with an extent for fast rendering.
            extent = [np.float64(t_axis[0]), np.float64(t_axis[-1]), np.float64(f_axis[0]), np.float64(f_axis[-1])]
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                extent=extent,
                interpolation="nearest",
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")

        else:
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                interpolation="nearest",
            )
            ax.set_xlabel("Time (index)")
            ax.set_ylabel("Frequency (index)")
            
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)    

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Value")

        title = f"Heatmap | e={e:04d} ({ep_label}) | ch={c:04d} ({ch_label})"
        if title_prefix:
            title = f"{title_prefix} | {title}"
        ax.set_title(title)

        fig.tight_layout()
        fig.savefig(fpath, format="pdf")
        plt.close(fig)

        saved.append(fpath)

    logger.info("Saved %d heatmap QC PDFs to %s", len(saved), out_path)
    return saved


def export_tensor_qc_spectra(
    tensor: np.ndarray,
    meta: dict[str, Any] | None,
    out_dir: str | Path,
    *,
    overwrite: bool = True,
    dpi: int = 200,
    title_prefix: str | None = None,
    skip_all_nan: bool = True,
    subdir: str = "spectrum",
    x_scale: str = "log",
    y_scale: str = "linear",
    agg_epoch: bool = True,
) -> list[Path]:
    """Export one spectrum PDF per (epoch, channel).

    Spectrum is computed as nanmean over the time axis:
        spectrum[f] = nanmean(tensor[e, c, f, :])

    Args:
        tensor: 4D tensor (epoch, channel, freq, time).
        meta: Metadata dict with axes.
        out_dir: Output folder root.
        overwrite: Overwrite existing PDFs.
        dpi: Figure DPI.
        title_prefix: Optional title prefix.
        skip_all_nan: If True, skip pages that are entirely NaN after nanmean.
        subdir: Subfolder under out_dir to store spectrum PDFs.

    Returns:
        List of saved PDF paths.
    """
    if agg_epoch:
        tensor = np.nanmean(tensor, axis=0, keepdims=True)  # (1, channel, freq, time)
    n_ep, n_ch, n_f, n_t = _validate_tensor_4d(tensor)

    out_root = Path(out_dir)
    out_path = out_root / subdir
    out_path.mkdir(parents=True, exist_ok=True)

    epoch_axis = _get_axis(meta, "epoch", n_ep)
    ch_axis = _get_axis(meta, "channel", n_ch)
    f_axis = _get_axis(meta, "freq", n_f)

    use_numeric_f = _is_numeric_axis(f_axis)
    x_freq = np.asarray(f_axis, dtype=np.float64) if use_numeric_f else np.arange(n_f, dtype=np.float64)

    saved: list[Path] = []

    for e, c in _iter_epoch_channel_pairs(n_ep, n_ch):
        z = np.asarray(tensor[e, c, :, :], dtype=np.float64)  # (freq, time)

        with np.errstate(invalid="ignore", divide="ignore"):
            spec = np.nanmean(z, axis=-1)  # (freq,)

        if skip_all_nan and (not np.any(np.isfinite(spec))):
            continue

        ep_label = _safe_str_label(epoch_axis[e], fallback=f"epoch_{e:04d}")
        ch_label = _safe_str_label(ch_axis[c], fallback=f"ch_{c:04d}")

        safe_ep = _sanitize_filename(ep_label)
        safe_ch = _sanitize_filename(ch_label)

        fname = f"spectrum_e{e:04d}_ch{c:04d}__{safe_ep}__{safe_ch}.pdf"
        fpath = out_path / fname

        if fpath.exists() and not overwrite:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=int(dpi))

        ax.plot(x_freq, spec, linewidth=1.5)

        if use_numeric_f:
            ax.set_xlabel("Frequency (Hz)")
        else:
            ax.set_xlabel("Frequency (index)")
        ax.set_ylabel("nanmean over time")

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        title = f"Spectrum | e={e:04d} ({ep_label}) | ch={c:04d} ({ch_label})"
        if title_prefix:
            title = f"{title_prefix} | {title}"
        ax.set_title(title)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fpath, format="pdf")
        plt.close(fig)

        saved.append(fpath)

    logger.info("Saved %d spectrum QC PDFs to %s", len(saved), out_path)
    return saved


def export_tensor_qc(
    tensor: np.ndarray,
    meta: dict[str, Any] | None,
    out_dir: str | Path,
    *,
    cmap: CmapName = "viridis",
    overwrite: bool = True,
    dpi: int = 200,
    title_prefix: str | None = None,
    skip_all_nan: bool = True,
    x_scale_heatmap: str = "linear",
    y_scale_heatmap: str = "log",
    x_scale_spectrum: str = "log",
    y_scale_spectrum: str = "linear",
    agg_epoch: bool = True,
) -> dict[str, list[Path]]:
    """Convenience wrapper: export both heatmaps and spectra."""
    heat = export_tensor_qc_heatmaps(
        tensor,
        meta,
        out_dir,
        cmap=cmap,
        overwrite=overwrite,
        dpi=dpi,
        title_prefix=title_prefix,
        skip_all_nan=skip_all_nan,
        x_scale=x_scale_heatmap,
        y_scale=y_scale_heatmap,
        agg_epoch=agg_epoch,
    )
    spec = export_tensor_qc_spectra(
        tensor,
        meta,
        out_dir,
        overwrite=overwrite,
        dpi=dpi,
        title_prefix=title_prefix,
        skip_all_nan=skip_all_nan,
        x_scale=x_scale_spectrum,
        y_scale=y_scale_spectrum,
        agg_epoch=agg_epoch,
    )
    return {"heatmap": heat, "spectrum": spec}
