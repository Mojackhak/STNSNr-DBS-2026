"""Axis-wise smoothing utilities for LFP feature tensors.

This module provides a small, focused API to smooth a tensor along a single
axis (typically time or frequency) while optionally applying an element-wise
transform prior to smoothing.

Why this exists
--------------
Many pipelines compute time-frequency features using a hop size that is much
smaller than the effective support of the estimator (for example Morlet
wavelets in time, or narrowband spectra in frequency). The resulting tensors
can be oversampled along one axis, and bin-to-bin noise can destabilize
downstream steps such as spectral parameterization.

Smoothing is therefore treated as an explicit post-processing step:
  1) (optional) transform -> 2) smooth -> 3) (optional) inverse transform

All code comments and docstrings are in English (per project rules).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from scipy.ndimage import convolve1d, gaussian_filter1d, median_filter

from ...utils.transforms import TransformMode, apply_inverse_transform_array, apply_transform_array


SmoothMethod = Literal["median", "mean", "gaussian"]
NanPolicy = Literal["omit", "propagate"]
PadMode = Literal["nearest", "reflect", "mirror", "constant", "wrap"]


def _normalize_kernel_size(kernel_size: int, *, enforce_odd: bool = True) -> int:
    k = int(kernel_size)
    if k < 1:
        raise ValueError("kernel_size must be >= 1")
    if enforce_odd and (k % 2 == 0):
        k += 1
    return k


def _axis_to_positive(axis: int, ndim: int) -> int:
    ax = int(axis)
    if ax < 0:
        ax += int(ndim)
    if ax < 0 or ax >= int(ndim):
        raise ValueError("axis is out of bounds for tensor")
    return ax


def _validate_sigma(sigma: float | None) -> float:
    if sigma is None:
        raise ValueError("sigma must be provided when method='gaussian'")
    sigma_f = float(sigma)
    if not np.isfinite(sigma_f) or sigma_f <= 0:
        raise ValueError("sigma must be a positive, finite number")
    return sigma_f


def _validate_truncate(truncate: float | None) -> float | None:
    if truncate is None:
        return None
    truncate_f = float(truncate)
    if not np.isfinite(truncate_f) or truncate_f <= 0:
        raise ValueError("truncate must be a positive, finite number when provided")
    return truncate_f


def smooth_axis(
    tensor: np.ndarray,
    *,
    kernel_size: int | None = None,
    method: SmoothMethod = "median",
    axis: int = -1,
    sigma: float | None = None,
    truncate: float | None = None,
    transform_mode: TransformMode | None = None,
    return_in_original_domain: bool = True,
    eps: float = 1e-120,
    pad_mode: PadMode = "nearest",
    cval: float = 0.0,
    nan_policy: NanPolicy = "omit",
    enforce_odd_kernel: bool = True,
) -> np.ndarray:
    """Smooth a tensor along one axis.

    Args:
        tensor: Input tensor (commonly 4D: epochs x channels x freqs x time).
        kernel_size: Smoothing window length in *samples* along `axis` for
            `method="median"` and `method="mean"`. Ignored for
            `method="gaussian"`.
        method: "median" (robust to outliers), "mean" (moving average), or
            "gaussian" (Gaussian-weighted moving average).
        axis: Axis index to smooth along. Default is -1 (last axis).
        sigma: Standard deviation of the Gaussian kernel, in bins along `axis`.
            Required when `method="gaussian"`.
        truncate: Optional Gaussian truncation radius in units of sigma, passed
            to SciPy. If None, SciPy's default truncation is used.
        transform_mode: Optional element-wise transform to apply *before*
            smoothing (e.g., "log" or "dB" for power; "fisherz" for coherence).
        return_in_original_domain: If True and `transform_mode` is not None,
            apply the inverse transform after smoothing.
        eps: Small positive constant used by transforms for clipping.
        pad_mode: Boundary handling mode passed to SciPy.
        cval: Constant value used when pad_mode == "constant".
        nan_policy:
            - "omit": ignore NaNs for mean smoothing (nan-aware normalization).
              For median smoothing, NaNs are currently **propagated** (see
              notes). For Gaussian smoothing, NaNs are omitted via weighted
              normalization.
            - "propagate": propagate NaNs (SciPy default behavior).
        enforce_odd_kernel: If True, even kernel sizes are rounded up to the
            next odd integer (recommended for symmetric smoothing). Applies only
            to methods that use `kernel_size`.

    Returns:
        Smoothed tensor with the same shape as the input.

    Notes:
        - For median smoothing, SciPy's `median_filter` does not natively ignore
          NaNs. If you need nan-robust median smoothing, pre-impute / mask
          NaNs, or use mean smoothing with `nan_policy="omit"`.
        - When `transform_mode` is not None, smoothing is performed in the
          transformed domain.
        - `method="gaussian"` ignores `kernel_size` and uses `sigma` /
          `truncate` to define the kernel support.
    """
    x_in = np.asarray(tensor)
    if x_in.ndim < 1:
        raise ValueError("tensor must have at least 1 dimension")

    ax = _axis_to_positive(axis, x_in.ndim)
    method_l = str(method).lower()

    if method_l in {"median", "mean"}:
        if kernel_size is None:
            raise ValueError("kernel_size must be provided for method='median' and method='mean'")
        k = _normalize_kernel_size(kernel_size, enforce_odd=enforce_odd_kernel)
    elif method_l == "gaussian":
        sigma_f = _validate_sigma(sigma)
        truncate_f = _validate_truncate(truncate)
    else:
        raise ValueError("method must be 'median', 'mean', or 'gaussian'")

    # k == 1 is an intentional no-op. We do NOT apply transforms in this case,
    # to avoid unexpected changes due to clipping (eps) in log-like transforms.
    if method_l in {"median", "mean"} and k == 1:
        return np.array(x_in, copy=True)

    # Use float for all operations; cast back at the end.
    out_dtype = x_in.dtype if np.issubdtype(x_in.dtype, np.floating) else np.float64
    x = np.asarray(x_in, dtype=float)

    if transform_mode is not None:
        x = apply_transform_array(x, mode=transform_mode, eps=eps)

    if method_l == "median":
        # Size tuple: smooth only along the requested axis.
        size = [1] * x.ndim
        size[ax] = k
        x_smooth = median_filter(x, size=tuple(size), mode=pad_mode, cval=float(cval))

    elif method_l == "mean":
        if nan_policy == "omit":
            # Nan-aware moving average:
            #   y = conv(x * mask) / conv(mask)
            # where mask = isfinite(x)
            mask = np.isfinite(x)
            x0 = np.where(mask, x, 0.0)
            w = np.ones(k, dtype=float) / float(k)

            num = convolve1d(x0, w, axis=ax, mode=pad_mode, cval=float(cval))
            den = convolve1d(mask.astype(float), w, axis=ax, mode=pad_mode, cval=float(cval))

            with np.errstate(invalid="ignore", divide="ignore"):
                x_smooth = num / den
            x_smooth = np.where(den > 0, x_smooth, np.nan)

        elif nan_policy == "propagate":
            w = np.ones(k, dtype=float) / float(k)
            x_smooth = convolve1d(x, w, axis=ax, mode=pad_mode, cval=float(cval))
        else:
            raise ValueError("nan_policy must be 'omit' or 'propagate'")

    else:  # method_l == "gaussian"
        gaussian_kwargs = dict(
            sigma=sigma_f,
            axis=ax,
            mode=pad_mode,
            cval=float(cval),
        )
        if truncate_f is not None:
            gaussian_kwargs["truncate"] = truncate_f

        if nan_policy == "omit":
            mask = np.isfinite(x)
            x0 = np.where(mask, x, 0.0)
            num = gaussian_filter1d(x0, **gaussian_kwargs)
            den = gaussian_filter1d(mask.astype(float), **gaussian_kwargs)

            with np.errstate(invalid="ignore", divide="ignore"):
                x_smooth = num / den
            x_smooth = np.where(den > 0, x_smooth, np.nan)

        elif nan_policy == "propagate":
            x_smooth = gaussian_filter1d(x, **gaussian_kwargs)
        else:
            raise ValueError("nan_policy must be 'omit' or 'propagate'")

    if transform_mode is not None and return_in_original_domain:
        x_smooth = apply_inverse_transform_array(x_smooth, mode=transform_mode, eps=eps)

    return np.asarray(x_smooth, dtype=out_dtype)
