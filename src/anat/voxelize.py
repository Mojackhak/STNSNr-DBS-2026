# -*- coding: utf-8 -*-
"""voxelize

Map point-based electrophysiology features (MNI coordinates + values) into a volumetric
representation (NIfTI) in a template/atlas space.

This module intentionally stays as a single file for portability. The public API is
kept small and stable:

- map_property_to_mni:
    Interpolated voxelization inside an ROI mask (template probability threshold
    intersected with a point-cloud convex hull, optionally inflated/smoothed).
- pointcloud_to_voxel_mean_nifti:
    Non-interpolated voxelization by per-voxel aggregation (mean + counts).
- voxelize:
    Convenience wrapper to call either strategy.

The module also includes optional QC helpers (sampling, metrics, LOOCV, grid search).

Notes on coordinate conventions
------------------------------
NIfTI affines map *voxel indices* to world (MNI) coordinates. Some pipelines treat
integer indices as voxel centers, others treat them as voxel corners. This code keeps
a configurable voxel-center offset (default: 0.5) to preserve legacy behavior in
convex-hull masking and interpolation queries.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Literal

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This module requires nibabel for NIfTI IO. Install with `pip install nibabel`."
    ) from e
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import pearsonr, spearmanr

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover
    DBSCAN = None  # type: ignore[assignment]

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as SKRBF, WhiteKernel
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None  # type: ignore[assignment]
    SKRBF = None  # type: ignore[assignment]
    WhiteKernel = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
BoolArray = npt.NDArray[np.bool_]


# -----------------------------------------------------------------------------
# Small dataclasses (optional; used mainly for clearer internal code)
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _DbscanInfo:
    used: bool
    min_samples: int
    eps_mm: float | None
    n_kept: int
    n_removed: int


@dataclass(frozen=True, slots=True)
class _InterpInfo:
    method: str
    details: dict[str, Any]


DEFAULT_VOXEL_CENTER_OFFSET_VOX = 0.5


def _inflate_mask_mm(
    mask: BoolArray,
    voxel_sizes_mm: FloatArray,
    radius_mm: float,
) -> BoolArray:
    """Dilate a binary mask by an exact Euclidean radius in physical space."""
    radius = float(max(radius_mm, 0.0))
    if radius == 0.0 or not mask.any():
        return mask.copy()
    distance_mm = distance_transform_edt(
        ~mask,
        sampling=np.asarray(voxel_sizes_mm, dtype=float),
    )
    return distance_mm <= (radius + 1e-6)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _as_xyz_array(xyz: Any, *, name: str = "xyz") -> FloatArray:
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an (N,3) array in MNI space (mm). Got shape {arr.shape}.")
    return arr


def _as_1d_float(values: Any, *, name: str = "values") -> FloatArray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr.astype(float)
    if not np.any(np.isfinite(arr)):
        raise ValueError(f"{name} contains no finite values.")
    return arr


def _validate_template_img(template_img: nib.spatialimages.SpatialImage) -> None:
    if not hasattr(template_img, "get_fdata") or not hasattr(template_img, "affine"):
        raise TypeError("template_img must be a nibabel SpatialImage (e.g., Nifti1Image).")
    data = np.asanyarray(template_img.get_fdata())
    if data.ndim != 3:
        raise ValueError(f"template_img must be 3D. Got data ndim={data.ndim} with shape {data.shape}.")
    affine = np.asarray(template_img.affine, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError(f"template_img.affine must be 4x4. Got {affine.shape}.")


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def compute_radial_scale(kernel_type: str, decay_spec: tuple[float, float]) -> float:
    """Compute a spatial scale parameter for a radial kernel given a decay target.

    Parameters
    ----------
    kernel_type
        Either "gaussian" or "exponential".

        - "gaussian": φ(r) = exp(-(r/ε)^2)
        - "exponential": φ(r) = exp(-r/λ)

    decay_spec
        Tuple (distance, alpha) where alpha is the desired kernel value at that distance.
        Example: (1.0, 0.1) means φ(1mm) = 0.1.

    Returns
    -------
    scale
        For Gaussian returns ε, for exponential returns λ.
    """
    d, alpha = decay_spec
    if d <= 0:
        raise ValueError("Distance d must be > 0.")
    if not (0 < alpha < 1):
        raise ValueError("Decay alpha must be in (0, 1), e.g., 0.5 or 0.1.")

    k = kernel_type.lower()
    if k in ("gaussian", "gauss"):
        return d / math.sqrt(-math.log(alpha))
    if k in ("exponential", "exp"):
        return d / -math.log(alpha)
    raise ValueError("kernel_type must be 'gaussian' or 'exponential'.")


def compute_norm_bounds(
    vals: FloatArray,
    method: str = "p02_p98",
    pmin: float = 2.0,
    pmax: float = 98.0,
    mad_clip: float = 3.0,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """Compute robust normalization bounds (lo, hi) from a value array."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0

    if method.lower().startswith("p"):
        lo, hi = np.percentile(v, [pmin, pmax])
    else:
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        scale = 1.4826 * (mad + eps)
        lo, hi = med - mad_clip * scale, med + mad_clip * scale

    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def robust_scale(vals: FloatArray, pmin: float = 5.0, pmax: float = 95.0) -> float:
    """Robust scale estimate using an inter-percentile range."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    lo, hi = np.percentile(v, [pmin, pmax])
    scale = float(hi - lo)
    return scale if scale > 0 else float(np.ptp(v) + 1e-8)


# -----------------------------------------------------------------------------
# Coloring helper
# -----------------------------------------------------------------------------
def colorize_points(
    df: pd.DataFrame,
    value_col: str = "Value",
    colormap: str = "viridis",
    alpha: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> pd.DataFrame:
    """Attach RGBA hex color to each row based on `value_col` using a Matplotlib colormap."""
    try:
        from matplotlib import colormaps, colors as mpl_colors
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "colorize_points requires matplotlib. Install with `pip install matplotlib`."
        ) from e
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame.")

    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(vals)) if np.any(np.isfinite(vals)) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else (vmin + 1.0)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = float(vmin + 1.0)

    cmap = colormaps.get_cmap(colormap)
    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(vals))
    rgba[:, 3] = float(np.clip(alpha, 0.0, 1.0))

    def _to_hex(row: FloatArray) -> str:
        r, g, b, a = np.round(row * 255).astype(np.uint8)
        return f"#{r:02X}{g:02X}{b:02X}{a:02X}"

    out = df.copy()
    out["color"] = [_to_hex(r) for r in rgba]
    return out


# -----------------------------------------------------------------------------
# Point filtering (DBSCAN-like)
# -----------------------------------------------------------------------------
def _estimate_eps_mm_knn(xyz_mni: FloatArray, k: int) -> float:
    """Heuristic eps from median distance to the k-th nearest neighbor."""
    xyz = _as_xyz_array(xyz_mni, name="xyz_mni")
    if xyz.shape[0] <= 1:
        return 1.0
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=max(2, int(k)))
    kth = d[:, -1]
    med = float(np.median(kth))
    return 1.2 * med if med > 0 else 1.0


def filter_points_dbscan_indices(
    xyz_mni: FloatArray,
    *,
    min_samples: int = 2,
    eps_mm: float | None = None,
) -> tuple[BoolArray, float]:
    """Compute a boolean mask of inlier points using DBSCAN (or a simple fallback).

    Returns
    -------
    keep_mask
        Boolean array (N,) marking points to keep.
    eps_used
        The eps value that was used (mm).
    """
    xyz = _as_xyz_array(xyz_mni, name="xyz_mni")
    n = int(xyz.shape[0])
    if n == 0:
        eps_used = float(eps_mm if eps_mm is not None else 1.0)
        return np.zeros((0,), dtype=bool), eps_used

    if eps_mm is None:
        eps_mm = _estimate_eps_mm_knn(xyz, k=min_samples)
    eps_used = float(eps_mm)

    if DBSCAN is None:  # pragma: no cover
        # Fallback: keep points with at least `min_samples` neighbors in radius eps.
        tree = cKDTree(xyz)
        try:
            neighbor_lists = tree.query_ball_point(xyz, r=eps_used)
            counts = np.array([len(nl) for nl in neighbor_lists], dtype=int)
        except Exception:
            counts = np.ones((n,), dtype=int)
        keep = counts >= int(min_samples)
        return keep.astype(bool), eps_used

    labels = DBSCAN(eps=eps_used, min_samples=int(min_samples), metric="euclidean").fit_predict(xyz)
    keep = labels != -1
    return keep.astype(bool), eps_used


# -----------------------------------------------------------------------------
# Masking: template ROI ∩ convex hull (with optional inflation/smoothing)
# -----------------------------------------------------------------------------
def convex_hull_mask_in_template(
    template_img: nib.Nifti1Image,
    xyz_mni: FloatArray,
    *,
    prob_threshold: float = 0.05,
    inflate_mm: float = 0.0,
    smooth_mm: float = 0.0,
    smooth_threshold: float = 0.5,
    voxel_center_offset_vox: float = DEFAULT_VOXEL_CENTER_OFFSET_VOX,
) -> tuple[BoolArray, IntArray, dict[str, Any]]:
    """Build a final mask in template space.

    final_mask = (template > prob_threshold) ∩ InflatedSmoothedHull(points)

    The hull is computed in *world (MNI) space* using a Delaunay triangulation, then
    evaluated at voxel centers (with an optional `voxel_center_offset_vox`).

    Parameters
    ----------
    template_img
        3D template NIfTI defining the space and providing a probability map.
    xyz_mni
        (N,3) point cloud in MNI space (mm).
    prob_threshold
        Threshold on template probability map to define the anatomical ROI.
    inflate_mm
        Isotropic dilation radius (mm) applied to the hull mask in voxel space.
    smooth_mm
        Gaussian smoothing sigma (mm) applied to the dilated hull mask before thresholding.
    smooth_threshold
        Threshold (0..1) applied after smoothing.
    voxel_center_offset_vox
        Offset added to voxel indices before mapping to MNI coordinates.

        Default (0.5) preserves legacy behavior in this repository. Set to 0.0 if your
        affine convention treats integer voxel indices as voxel centers.

    Returns
    -------
    final_mask
        Boolean volume in template voxel space.
    coords_vox_inside
        (K,3) integer voxel indices where final_mask is True.
    hull_info
        Diagnostics, including voxel counts and masking method.
    """
    _validate_template_img(template_img)

    tpl = np.asanyarray(template_img.get_fdata(), dtype=float)
    tmask = tpl > float(prob_threshold)
    if not tmask.any():
        # Fallback: at least keep all non-zero voxels
        tmask = tpl > 0

    shape = tmask.shape
    affine = np.asarray(template_img.affine, dtype=float)
    voxel_sizes = np.array(nib.affines.voxel_sizes(affine), dtype=float)
    voxel_sizes[voxel_sizes <= 0] = 1.0  # safety

    candidates = np.column_stack(np.nonzero(tmask)).astype(int)
    if candidates.size == 0:
        final_mask = np.zeros(shape, dtype=bool)
        return final_mask, candidates, {
            "method": "empty_template_mask",
            "template_mask_voxels": 0,
            "hull_voxels": 0,
            "final_mask_voxels": 0,
            "inflate_mm": float(inflate_mm),
            "smooth_mm": float(smooth_mm),
            "smooth_threshold": float(smooth_threshold),
            "voxel_center_offset_vox": float(voxel_center_offset_vox),
        }

    centers_vox = candidates.astype(float) + float(voxel_center_offset_vox)
    centers_mni = nib.affines.apply_affine(affine, centers_vox)

    xyz = _as_xyz_array(xyz_mni, name="xyz_mni")
    hull_mask = np.zeros(shape, dtype=bool)
    hull_info: dict[str, Any] = {
        "method": None,
        "template_mask_voxels": int(tmask.sum()),
        "voxel_center_offset_vox": float(voxel_center_offset_vox),
    }

    # 1) Convex hull inside template mask via Delaunay
    if xyz.shape[0] >= 4:
        try:
            tri = Delaunay(xyz.astype(float))
            inside = tri.find_simplex(centers_mni) >= 0
            hull_info["method"] = "delaunay"
        except Exception as e:
            logger.debug("Delaunay failed: %s", e)
            inside = np.zeros(centers_mni.shape[0], dtype=bool)
            hull_info["method"] = "fallback_none"
    else:
        inside = np.zeros(centers_mni.shape[0], dtype=bool)
        hull_info["method"] = "insufficient_points"

    hull_mask[candidates[inside, 0], candidates[inside, 1], candidates[inside, 2]] = True

    # 2) Optional inflation (3D dilation in mm)
    inflate_mm = float(max(inflate_mm, 0.0))
    if inflate_mm > 0.0 and hull_mask.any():
        min_vs = float(np.min(voxel_sizes))
        max_r_vox = int(np.ceil(inflate_mm / max(min_vs, 1e-6)))
        xs = np.arange(-max_r_vox, max_r_vox + 1)
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        dx_mm = X * voxel_sizes[0]
        dy_mm = Y * voxel_sizes[1]
        dz_mm = Z * voxel_sizes[2]
        dist_mm = np.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2)
        struct = dist_mm <= (inflate_mm + 1e-6)

        hull_mask = _inflate_mask_mm(hull_mask, voxel_sizes, inflate_mm)

        hull_info.update(
            {
                "inflate_mm": inflate_mm,
                "inflate_vox_radius": int(max_r_vox),
                "inflate_struct_voxels": int(struct.sum()),
            }
        )
    else:
        hull_info.update(
            {"inflate_mm": 0.0, "inflate_vox_radius": 0, "inflate_struct_voxels": 0}
        )

    # 3) Optional smoothing for curved boundaries
    smooth_mm = float(max(smooth_mm, 0.0))
    smooth_threshold = float(smooth_threshold)
    if smooth_mm > 0.0 and hull_mask.any():
        sigma_vox = smooth_mm / np.maximum(voxel_sizes, 1e-6)
        sm = gaussian_filter(hull_mask.astype(float), sigma=sigma_vox, mode="constant", cval=0.0)
        hull_mask = sm >= smooth_threshold
        hull_info.update(
            {
                "smooth_mm": smooth_mm,
                "smooth_sigma_vox": tuple(float(x) for x in sigma_vox),
                "smooth_threshold": smooth_threshold,
            }
        )
    else:
        hull_info.update(
            {
                "smooth_mm": 0.0,
                "smooth_sigma_vox": (0.0, 0.0, 0.0),
                "smooth_threshold": smooth_threshold,
            }
        )

    # 4) Final mask = template ROI ∩ inflated+smoothed hull
    final_mask = tmask & hull_mask
    coords = np.column_stack(np.nonzero(final_mask)).astype(int)

    hull_info["hull_voxels"] = int(hull_mask.sum())
    hull_info["final_mask_voxels"] = int(final_mask.sum())

    return final_mask, coords, hull_info


# -----------------------------------------------------------------------------
# Interpolation helpers
# -----------------------------------------------------------------------------
def _deduplicate_points_mean(xyz: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Average y for duplicate xyz rows."""
    xyz = _as_xyz_array(xyz, name="xyz_train")
    y = _as_1d_float(y, name="y_train")
    if xyz.shape[0] != y.shape[0]:
        raise ValueError(f"xyz_train and y_train length mismatch: {xyz.shape[0]} vs {y.shape[0]}")
    if xyz.shape[0] == 0:
        return xyz, y

    uniq, inv = np.unique(xyz, axis=0, return_inverse=True)
    y_sum = np.zeros((uniq.shape[0],), dtype=float)
    y_cnt = np.zeros((uniq.shape[0],), dtype=int)
    np.add.at(y_sum, inv, y)
    np.add.at(y_cnt, inv, 1)
    y_mean = y_sum / np.maximum(y_cnt, 1)
    return uniq.astype(float), y_mean.astype(float)


def _interpolate_idw(
    xyz_train: FloatArray,
    y_train: FloatArray,
    xyz_query: FloatArray,
    *,
    k: int = 16,
    power: float = 2.0,
    eps: float = 1e-8,
) -> FloatArray:
    """Inverse Distance Weighting using k nearest neighbors."""
    X, y = _deduplicate_points_mean(xyz_train, y_train)
    Q = _as_xyz_array(xyz_query, name="xyz_query")
    if X.shape[0] == 0 or Q.shape[0] == 0:
        return np.full((Q.shape[0],), np.nan, dtype=float)

    tree = cKDTree(X)
    k_eff = int(min(max(1, k), X.shape[0]))
    d, idx = tree.query(Q, k=k_eff)

    if k_eff == 1:
        d = d[:, None]
        idx = idx[:, None]

    w = 1.0 / np.maximum(d, eps) ** float(power)
    exact = d <= eps

    out = np.sum(w * y[idx], axis=1) / np.sum(w, axis=1)

    # If a query point coincides with a training point, return the exact value.
    if np.any(exact.any(axis=1)):
        rows = np.where(exact.any(axis=1))[0]
        for r in rows:
            out[r] = y[idx[r, exact[r]].flatten()[0]]

    return out.astype(float)


def _interpolate_linear(
    xyz_train: FloatArray, y_train: FloatArray, xyz_query: FloatArray
) -> tuple[FloatArray, dict[str, Any]]:
    """Linear (Delaunay) interpolation; NaN outside hull with an IDW edge fallback."""
    X, y = _deduplicate_points_mean(xyz_train, y_train)
    Q = _as_xyz_array(xyz_query, name="xyz_query")
    if X.shape[0] == 0 or Q.shape[0] == 0:
        return np.full((Q.shape[0],), np.nan, dtype=float), {"method": "linear", "empty": True}

    lin = LinearNDInterpolator(X, y, fill_value=np.nan, rescale=False)
    pred = np.asarray(lin(Q), dtype=float)

    info: dict[str, Any] = {"method": "linear", "fill_value": np.nan, "idw_edge_fallback": False}
    nan_mask = ~np.isfinite(pred)
    if np.any(nan_mask):
        pred[nan_mask] = _interpolate_idw(X, y, Q[nan_mask], k=min(8, X.shape[0]))
        info["idw_edge_fallback"] = True
    return pred, info


def _interpolate_rbf_safe(
    xyz_train: FloatArray,
    y_train: FloatArray,
    xyz_query: FloatArray,
    *,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
    neighbors: int | None = None,
    epsilon: float | None = None,
    degree: int | None = None,
) -> tuple[FloatArray, dict[str, Any]]:
    """RBF interpolation with fallbacks that avoid common singular-matrix failures."""
    X, y = _deduplicate_points_mean(xyz_train, y_train)
    Q = _as_xyz_array(xyz_query, name="xyz_query")

    info: dict[str, Any] = {
        "method": "rbf",
        "kernel": kernel,
        "degree": degree,
        "smoothing": float(smoothing),
        "neighbors": neighbors,
        "epsilon": epsilon,
        "fallback": None,
        "n_train_used": int(X.shape[0]),
    }

    if X.shape[0] == 0 or Q.shape[0] == 0:
        return np.full((Q.shape[0],), np.nan, dtype=float), {**info, "empty": True}

    try:
        rbf = RBFInterpolator(
            X,
            y,
            kernel=kernel,
            degree=degree,
            smoothing=float(smoothing),
            neighbors=neighbors,
            epsilon=epsilon,
        )
        return np.asarray(rbf(Q), dtype=float), info
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback: Gaussian + degree=0 + heuristic epsilon based on median NN distance
        logger.debug("RBFInterpolator failed (%s). Falling back to gaussian degree=0.", e)

        if X.shape[0] > 1:
            tree = cKDTree(X)
            d, _ = tree.query(X, k=min(2, X.shape[0]))
            med = float(np.median(d[:, -1]))
        else:
            med = 1.0
        eps0 = (1.0 / med) if med > 0 else 1.0

        rbf = RBFInterpolator(
            X,
            y,
            kernel="gaussian",
            degree=0,
            epsilon=float(eps0),
            smoothing=0.0,
            neighbors=neighbors,
        )
        info.update({"kernel": "gaussian", "degree": 0, "epsilon": float(eps0), "fallback": "gaussian_degree0"})
        return np.asarray(rbf(Q), dtype=float), info


def _interpolate_gpr(
    xyz_train: FloatArray,
    y_train: FloatArray,
    xyz_query: FloatArray,
    *,
    length_scale: float | None = None,
    noise: float = 1e-3,
) -> tuple[FloatArray, dict[str, Any]]:
    """Gaussian Process regression (RBF + WhiteKernel)."""
    if GaussianProcessRegressor is None or SKRBF is None or WhiteKernel is None:  # pragma: no cover
        raise ImportError("scikit-learn is required for method='gpr'.")

    X, y = _deduplicate_points_mean(xyz_train, y_train)
    Q = _as_xyz_array(xyz_query, name="xyz_query")
    if X.shape[0] == 0 or Q.shape[0] == 0:
        return np.full((Q.shape[0],), np.nan, dtype=float), {"method": "gpr", "empty": True}

    if length_scale is None:
        if X.shape[0] > 1:
            tree = cKDTree(X)
            d, _ = tree.query(X, k=min(2, X.shape[0]))
            length_scale = float(np.median(d[:, -1]))
        else:
            length_scale = 1.0

    kernel = SKRBF(length_scale=float(length_scale), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        noise_level=float(noise)
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=2,
    )
    gpr.fit(X, y)
    pred, _ = gpr.predict(Q, return_std=True)
    info = {"method": "gpr", "length_scale": float(length_scale), "noise": float(noise)}
    return np.asarray(pred, dtype=float), info


# -----------------------------------------------------------------------------
# Main API: interpolated voxelization (returns params dict)
# -----------------------------------------------------------------------------
def map_property_to_mni(
    template_img: nib.Nifti1Image,
    df: pd.DataFrame,
    *,
    x_col: str = "MNI_x",
    y_col: str = "MNI_y",
    z_col: str = "MNI_z",
    value_col: str = "Value",
    colormap: str = "viridis",
    alpha: float = 1.0,
    prob_threshold: float = 0.05,
    inflate_mm: float = 0.0,
    smooth_mm: float = 0.0,
    smooth_threshold: float = 0.5,
    # pre-filter
    use_dbscan: bool = True,
    dbscan_min_samples: int = 2,
    dbscan_eps_mm: float | None = None,
    # interpolation
    method: Literal["rbf", "linear", "idw", "gpr"] = "rbf",
    rbf_kernel: str = "thin_plate_spline",
    rbf_smoothing: float = 0.0,
    rbf_neighbors: int | None = None,
    rbf_degree: int | None = None,
    idw_k: int = 16,
    idw_power: float = 2.0,
    gpr_length_scale: float | None = None,
    # optional post-smoothing
    sigma_mm: float | None = None,
    # normalization
    norm: bool = False,
    norm_method: str = "p02_p98",
    norm_pmin: float = 2.0,
    norm_pmax: float = 98.0,
    norm_mad_clip: float = 3.0,
    clip_range: tuple[float, float] | None = None,
    # coordinate convention
    voxel_center_offset_vox: float = DEFAULT_VOXEL_CENTER_OFFSET_VOX,
) -> tuple[pd.DataFrame, nib.Nifti1Image, dict[str, Any]]:
    """Voxelize pointwise features into a NIfTI heatmap (template space).

    Returns
    -------
    df_colored
        Copy of df (after dropna) with an added "color" column (#RRGGBBAA).
    heat_img
        NIfTI image (float32) with NaNs outside the final mask.
    params
        Dictionary capturing parameters and diagnostics for reproducibility.
    """
    _validate_template_img(template_img)
    _require_columns(df, [x_col, y_col, z_col, value_col])

    df0_n = int(len(df))
    df_clean = df.dropna(subset=[x_col, y_col, z_col, value_col]).copy()
    df1_n = int(len(df_clean))
    if df1_n == 0:
        raise ValueError("No valid rows after dropna on coordinate/value columns.")

    # Ensure numeric
    for c in (x_col, y_col, z_col, value_col):
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    df_clean = df_clean.dropna(subset=[x_col, y_col, z_col, value_col]).copy()
    if len(df_clean) == 0:
        raise ValueError("No valid numeric rows after coercing coordinate/value columns.")

    df_colored = colorize_points(df_clean, value_col=value_col, colormap=colormap, alpha=alpha)

    xyz = df_clean[[x_col, y_col, z_col]].to_numpy(dtype=float)
    y = df_clean[value_col].to_numpy(dtype=float)
    xyz = _as_xyz_array(xyz, name="xyz")
    y = _as_1d_float(y, name="y")

    n_unique = int(np.unique(xyz, axis=0).shape[0])
    vmin, vmax = float(np.nanmin(y)), float(np.nanmax(y))

    # Numeric rank diagnostics (raw and centered/scaled)
    if xyz.shape[0] > 0:
        P_raw = np.c_[np.ones(len(xyz)), xyz]
        rank_raw = int(np.linalg.matrix_rank(P_raw))
        shift = xyz.mean(axis=0)
        scale = np.ptp(xyz, axis=0)
        scale[scale == 0] = 1.0
        Z = (xyz - shift) / scale
        s = np.linalg.svd(np.c_[np.ones(len(Z)), Z], compute_uv=False)
        tol = s.max() * np.finfo(s.dtype).eps * max(len(Z), 4)
        rank_c = int((s > tol).sum())
        singvals = s.tolist()
    else:
        rank_raw, rank_c, singvals = 0, 0, []

    # DBSCAN filtering (optional)
    if use_dbscan and xyz.shape[0] >= int(dbscan_min_samples):
        keep_mask, eps_used = filter_points_dbscan_indices(
            xyz, min_samples=int(dbscan_min_samples), eps_mm=dbscan_eps_mm
        )
        xyz_f = xyz[keep_mask]
        y_f = y[keep_mask]
        dbscan_info = _DbscanInfo(
            used=True,
            min_samples=int(dbscan_min_samples),
            eps_mm=float(eps_used),
            n_kept=int(keep_mask.sum()),
            n_removed=int((~keep_mask).sum()),
        )
        xyz, y = xyz_f, y_f
    else:
        dbscan_info = _DbscanInfo(
            used=False,
            min_samples=int(dbscan_min_samples),
            eps_mm=float(dbscan_eps_mm) if dbscan_eps_mm is not None else None,
            n_kept=int(len(xyz)),
            n_removed=0,
        )

    # Final mask = template-threshold ∩ convex hull
    final_mask, coords, hull_info = convex_hull_mask_in_template(
        template_img,
        xyz,
        prob_threshold=float(prob_threshold),
        inflate_mm=float(inflate_mm),
        smooth_mm=float(smooth_mm),
        smooth_threshold=float(smooth_threshold),
        voxel_center_offset_vox=float(voxel_center_offset_vox),
    )

    shape = tuple(final_mask.shape)
    affine = np.asarray(template_img.affine, dtype=float)
    zooms = np.array(nib.affines.voxel_sizes(affine), dtype=float)

    heat = np.zeros(shape, dtype=np.float32)
    interpolated = bool(coords.size > 0 and xyz.shape[0] > 0)

    if interpolated:
        centers_vox = coords.astype(float) + float(voxel_center_offset_vox)
        centers_mni = nib.affines.apply_affine(affine, centers_vox)

        if method == "rbf":
            pred, info = _interpolate_rbf_safe(
                xyz,
                y,
                centers_mni,
                kernel=rbf_kernel,
                smoothing=float(rbf_smoothing),
                neighbors=rbf_neighbors,
                epsilon=None,
                degree=rbf_degree,
            )
            interp_info = _InterpInfo(method="rbf", details=info)
        elif method == "linear":
            pred, info = _interpolate_linear(xyz, y, centers_mni)
            interp_info = _InterpInfo(method="linear", details=info)
        elif method == "idw":
            pred = _interpolate_idw(xyz, y, centers_mni, k=int(idw_k), power=float(idw_power))
            interp_info = _InterpInfo(method="idw", details={"method": "idw", "k": int(idw_k), "power": float(idw_power)})
        elif method == "gpr":
            pred, info = _interpolate_gpr(xyz, y, centers_mni, length_scale=gpr_length_scale)
            interp_info = _InterpInfo(method="gpr", details=info)
        else:
            raise ValueError("method must be one of {'rbf','linear','idw','gpr'}")

        heat[coords[:, 0], coords[:, 1], coords[:, 2]] = pred.astype(np.float32, copy=False)

        # Optional Gaussian smoothing
        if sigma_mm is not None and float(sigma_mm) > 0:
            sigma_vox = np.maximum(float(sigma_mm) / np.maximum(zooms, 1e-6), 0.0)
            heat = gaussian_filter(heat, sigma=sigma_vox, mode="nearest")
            heat[~final_mask] = 0.0
        else:
            sigma_vox = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        interp_info = _InterpInfo(method=str(method), details={"method": str(method), "executed": False})
        sigma_vox = np.array([0.0, 0.0, 0.0], dtype=float)
        # Avoid silently returning zeros as if they were meaningful.
        heat[final_mask] = np.nan

    # Ensure outside-mask voxels are NaN (regardless of normalization)
    heat[~final_mask] = np.nan

    # Optional clipping
    if clip_range is not None:
        lo_clip, hi_clip = float(clip_range[0]), float(clip_range[1])
        if not (np.isfinite(lo_clip) and np.isfinite(hi_clip) and hi_clip >= lo_clip):
            raise ValueError("clip_range must be (lo, hi) with finite values and hi>=lo.")
        heat[final_mask] = np.clip(heat[final_mask], lo_clip, hi_clip)

    vals_in_mask = heat[final_mask].astype(float)

    # Optional robust normalization to [0, 1]
    if norm and np.any(np.isfinite(vals_in_mask)):
        lo, hi = compute_norm_bounds(
            vals_in_mask,
            method=norm_method,
            pmin=float(norm_pmin),
            pmax=float(norm_pmax),
            mad_clip=float(norm_mad_clip),
        )
        heat_out = np.full_like(heat, np.nan, dtype=np.float32)
        clipped = np.clip(vals_in_mask, lo, hi)
        heat_out[final_mask] = ((clipped - lo) / (hi - lo + 1e-12)).astype(np.float32)
        norm_enabled = True
        norm_method_used: str | None = str(norm_method)
    else:
        heat_out = heat.astype(np.float32, copy=False)
        if np.any(np.isfinite(vals_in_mask)):
            lo = float(np.nanmin(vals_in_mask))
            hi = float(np.nanmax(vals_in_mask))
        else:
            lo, hi = float("nan"), float("nan")
        norm_enabled = False
        norm_method_used = None

    # Pack NIfTI
    header = template_img.header.copy()
    header.set_data_dtype(np.float32)
    heat_img = nib.Nifti1Image(heat_out, affine, header)

    params: dict[str, Any] = {
        "template": {
            "shape": tuple(int(x) for x in shape),
            "voxel_sizes_mm": tuple(float(z) for z in zooms),
            "prob_threshold": float(prob_threshold),
        },
        "input_points": {
            "n_points_raw": int(df0_n),
            "n_points_after_dropna": int(df1_n),
            "n_unique_points": int(n_unique),
            "n_points_used": int(len(xyz)),
            "value_min": float(vmin),
            "value_max": float(vmax),
        },
        "diagnostics": {
            "rankP_raw": int(rank_raw),
            "rankP_centered": int(rank_c),
            "singular_values_centered": singvals,
        },
        "dbscan": {
            "used": dbscan_info.used,
            "min_samples": dbscan_info.min_samples,
            "eps_mm": dbscan_info.eps_mm,
            "n_kept": dbscan_info.n_kept,
            "n_removed": dbscan_info.n_removed,
        },
        "masking": {
            "template_mask_voxels": int(hull_info.get("template_mask_voxels", 0)),
            "hull_method": hull_info.get("method", None),
            "hull_voxels": int(hull_info.get("hull_voxels", 0)),
            "final_mask_voxels": int(hull_info.get("final_mask_voxels", 0)),
            "inflate_mm": float(hull_info.get("inflate_mm", 0.0)),
            "smooth_mm": float(hull_info.get("smooth_mm", 0.0)),
            "smooth_threshold": float(hull_info.get("smooth_threshold", smooth_threshold)),
            "voxel_center_offset_vox": float(hull_info.get("voxel_center_offset_vox", voxel_center_offset_vox)),
        },
        "interpolation": {"method": interp_info.method, **interp_info.details},
        "smoothing": {"sigma_mm": float(sigma_mm) if sigma_mm is not None else 0.0, "sigma_vox": tuple(float(x) for x in sigma_vox)},
        "normalization": {
            "enabled": bool(norm_enabled),
            "method": norm_method_used,
            "pmin": float(norm_pmin),
            "pmax": float(norm_pmax),
            "mad_clip": float(norm_mad_clip),
            "lo": float(lo),
            "hi": float(hi),
        },
        "coloring": {"colormap": str(colormap), "alpha": float(alpha)},
    }

    return df_colored, heat_img, params


# -----------------------------------------------------------------------------
# Alternative API: non-interpolated voxel mean (mean + count)
# -----------------------------------------------------------------------------
def pointcloud_to_voxel_mean_nifti(
    template_img: nib.Nifti1Image,
    df: pd.DataFrame,
    *,
    x_col: str = "MNI_x",
    y_col: str = "MNI_y",
    z_col: str = "MNI_z",
    value_col: str = "Value",
    prob_threshold: float | None = None,
    apply_mask: bool = False,
    min_count: int = 1,
    fill_value: float = 0.0,
    dtype: Any = np.float32,
    snap: Literal["nearest", "floor", "ceil"] = "nearest",
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Aggregate pointwise values into voxels by mean (no interpolation).

    Returns
    -------
    mean_img
        NIfTI volume storing the per-voxel mean of `value_col`.
    count_img
        NIfTI volume storing the per-voxel sample count (int16).
    """
    _validate_template_img(template_img)
    _require_columns(df, [x_col, y_col, z_col, value_col])

    df_clean = df.dropna(subset=[x_col, y_col, z_col, value_col]).copy()
    for c in (x_col, y_col, z_col, value_col):
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    df_clean = df_clean.dropna(subset=[x_col, y_col, z_col, value_col]).copy()

    # Template geometry
    shape = tuple(int(x) for x in template_img.shape[:3])
    affine = np.asarray(template_img.affine, dtype=float)
    inv_affine = np.linalg.inv(affine)

    # Optional mask from template probability threshold
    if apply_mask and prob_threshold is not None:
        tpl = np.asanyarray(template_img.get_fdata(), dtype=float)
        if tpl.ndim != 3:
            raise ValueError("template_img must be 3D.")
        mask = tpl > float(prob_threshold)
    else:
        mask = None

    xyz = df_clean[[x_col, y_col, z_col]].to_numpy(dtype=float)
    vals = df_clean[value_col].to_numpy(dtype=float)

    ijk_f = nib.affines.apply_affine(inv_affine, xyz)  # float voxel coordinates

    if snap == "nearest":
        ijk = np.rint(ijk_f).astype(int)
    elif snap == "floor":
        ijk = np.floor(ijk_f).astype(int)
    elif snap == "ceil":
        ijk = np.ceil(ijk_f).astype(int)
    else:
        raise ValueError("snap must be one of {'nearest','floor','ceil'}")

    inside = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < shape[0])
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < shape[1])
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < shape[2])
    )
    ijk = ijk[inside]
    vals = vals[inside]

    sum_vol = np.zeros(shape, dtype=np.float64)
    cnt_vol = np.zeros(shape, dtype=np.int32)
    if ijk.size > 0:
        np.add.at(sum_vol, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), vals)
        np.add.at(cnt_vol, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1)

    mean_vol = np.full(shape, float(fill_value), dtype=np.float64)
    valid = cnt_vol >= int(min_count)
    mean_vol[valid] = (sum_vol[valid] / np.maximum(cnt_vol[valid], 1)).astype(np.float64)

    if mask is not None:
        mean_vol = np.where(mask, mean_vol, float(fill_value))
        cnt_vol = np.where(mask, cnt_vol, 0)

    header = template_img.header.copy()
    header.set_data_dtype(np.float32)
    mean_img = nib.Nifti1Image(mean_vol.astype(dtype), affine, header)

    cnt_header = template_img.header.copy()
    cnt_header.set_data_dtype(np.int16)
    count_img = nib.Nifti1Image(cnt_vol.astype(np.int16), affine, cnt_header)

    return mean_img, count_img


# -----------------------------------------------------------------------------
# Convenience wrapper (single entry point)
# -----------------------------------------------------------------------------
def voxelize(
    template_img: nib.Nifti1Image,
    df: pd.DataFrame,
    *,
    strategy: Literal["interpolate", "mean"] = "interpolate",
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Convenience wrapper for voxelization.

    Parameters
    ----------
    strategy
        - "interpolate": calls map_property_to_mni
        - "mean": calls pointcloud_to_voxel_mean_nifti

    kwargs
        Passed through to the underlying function.

    Returns
    -------
    tuple
        For "interpolate": (df_colored, heat_img, params)
        For "mean": (mean_img, count_img)
    """
    if strategy == "interpolate":
        return map_property_to_mni(template_img, df, **kwargs)
    if strategy == "mean":
        return pointcloud_to_voxel_mean_nifti(template_img, df, **kwargs)
    raise ValueError("strategy must be 'interpolate' or 'mean'.")


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------
def sample_nifti_at_mni(
    img: nib.Nifti1Image,
    xyz_mni: FloatArray,
    *,
    mode: Literal["nearest", "trilinear"] = "nearest",
    outside_value: float = np.nan,
) -> FloatArray:
    """Sample a NIfTI volume at given MNI coordinates.

    - mode='nearest': round to nearest voxel index.
    - mode='trilinear': first-order interpolation via scipy.ndimage.map_coordinates.

    Values outside the image are returned as `outside_value`.
    """
    vol = np.asarray(img.get_fdata(), dtype=float)
    if vol.ndim != 3:
        raise ValueError(f"img must be 3D. Got vol.ndim={vol.ndim}.")
    affine = np.asarray(img.affine, dtype=float)
    inv_aff = np.linalg.inv(affine)

    xyz = _as_xyz_array(xyz_mni, name="xyz_mni")
    ijk_f = nib.affines.apply_affine(inv_aff, xyz)

    shape = vol.shape
    if mode == "nearest":
        ijk = np.rint(ijk_f).astype(int)
        inside = (
            (ijk[:, 0] >= 0)
            & (ijk[:, 0] < shape[0])
            & (ijk[:, 1] >= 0)
            & (ijk[:, 1] < shape[1])
            & (ijk[:, 2] >= 0)
            & (ijk[:, 2] < shape[2])
        )
        out = np.full((ijk.shape[0],), float(outside_value), dtype=float)
        if np.any(inside):
            out[inside] = vol[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
        return out.astype(float)

    if mode == "trilinear":
        coords = ijk_f.T  # (3, N)
        sampled = map_coordinates(
            vol,
            coords,
            order=1,
            mode="constant",
            cval=float(outside_value),
        )
        return np.asarray(sampled, dtype=float)

    raise ValueError("mode must be 'nearest' or 'trilinear'.")


# -----------------------------------------------------------------------------
# Metrics / QC helpers
# -----------------------------------------------------------------------------
def concordance_correlation_coefficient(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Lin's concordance correlation coefficient (CCC)."""
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    if x.size < 2:
        return float("nan")

    mx, my = float(np.mean(x)), float(np.mean(y))
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0:
        return float("nan")
    return float(2 * cov / denom)


def bland_altman(y_true: FloatArray, y_pred: FloatArray) -> dict[str, float]:
    """Bland–Altman statistics: mean bias and 95% limits of agreement."""
    diff = np.asarray(y_pred, float) - np.asarray(y_true, float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"bias": float("nan"), "loa_low": float("nan"), "loa_high": float("nan"), "sd_diff": float("nan")}

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    return {"bias": bias, "loa_low": loa_low, "loa_high": loa_high, "sd_diff": sd}


def compute_loyalty_metrics(
    heat_img: nib.Nifti1Image,
    df_points: pd.DataFrame,
    *,
    x_col: str = "MNI_x",
    y_col: str = "MNI_y",
    z_col: str = "MNI_z",
    value_col: str = "Value",
    sample_mode: Literal["nearest", "trilinear"] = "nearest",
    weights_img: nib.Nifti1Image | None = None,
    weight_kind: Literal["nearest", "trilinear"] = "nearest",
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compare pointwise original values vs sampled heatmap values."""
    _require_columns(df_points, [x_col, y_col, z_col, value_col])
    df = df_points.dropna(subset=[x_col, y_col, z_col, value_col]).copy()
    for c in (x_col, y_col, z_col, value_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[x_col, y_col, z_col, value_col]).copy()

    xyz = df[[x_col, y_col, z_col]].to_numpy(float)
    d_true = df[value_col].to_numpy(float)

    h_pred = sample_nifti_at_mni(heat_img, xyz, mode=sample_mode)

    if weights_img is not None:
        w = sample_nifti_at_mni(weights_img, xyz, mode=weight_kind)
        w = np.clip(np.asarray(w, float), 0.0, np.inf)
    else:
        w = np.ones_like(h_pred, dtype=float)

    mask = np.isfinite(d_true) & np.isfinite(h_pred) & (w > 0)
    d_true = d_true[mask]
    h_pred = h_pred[mask]
    w = w[mask]
    xyz = xyz[mask]

    err = h_pred - d_true
    abs_err = np.abs(err)
    sq_err = err**2

    rng = float(np.ptp(d_true)) if d_true.size else 1.0
    rng = rng if rng > 0 else 1.0
    rscale = robust_scale(d_true) if d_true.size else 1.0

    wsum = float(np.sum(w))
    if wsum <= 0:
        wsum = 1.0

    def wmean(x: FloatArray) -> float:
        return float(np.sum(w * x) / max(wsum, eps))

    mae = wmean(abs_err)
    rmse = float(np.sqrt(wmean(sq_err)))
    medae = float(np.median(abs_err)) if abs_err.size else float("nan")
    nmae_range = mae / (rng + eps)
    nrmse_range = rmse / (rng + eps)
    nmae_robust = mae / (rscale + eps)
    nrmse_robust = rmse / (rscale + eps)

    pr = pearsonr(d_true, h_pred)[0] if d_true.size > 1 else float("nan")
    sr = spearmanr(d_true, h_pred)[0] if d_true.size > 1 else float("nan")

    ccc = concordance_correlation_coefficient(d_true, h_pred)

    if d_true.size > 1:
        X = np.c_[np.ones_like(d_true), d_true]
        beta, *_ = np.linalg.lstsq(X, h_pred, rcond=None)
        intercept, slope = float(beta[0]), float(beta[1])
    else:
        intercept, slope = float("nan"), float("nan")

    ba = bland_altman(d_true, h_pred)

    per_point = pd.DataFrame(
        {
            x_col: xyz[:, 0],
            y_col: xyz[:, 1],
            z_col: xyz[:, 2],
            "Value_true": d_true,
            "Heat_pred": h_pred,
            "Error": err,
            "AbsError": abs_err,
            "Weight": w,
        }
    )

    loyalty_score = 0.5 * (1.0 - nmae_robust) + 0.5 * ((ccc + 1.0) / 2.0)
    loyalty_score = float(np.clip(loyalty_score, 0.0, 1.0))

    return {
        "N": int(d_true.size),
        "sampling": {"mode": sample_mode},
        "weights": {"used": weights_img is not None, "mode": weight_kind},
        "accuracy": {
            "MAE": mae,
            "RMSE": rmse,
            "MedAE": medae,
            "NMAE_range": nmae_range,
            "NRMSE_range": nrmse_range,
            "NMAE_robust": nmae_robust,
            "NRMSE_robust": nrmse_robust,
        },
        "correlation": {"Pearson_r": float(pr), "Spearman_rho": float(sr)},
        "agreement": {"CCC": float(ccc), **ba},
        "calibration": {"slope": float(slope), "intercept": float(intercept)},
        "loyalty_score": loyalty_score,
        "per_point": per_point,
    }


# -----------------------------------------------------------------------------
# DataFrame normalization helper
# -----------------------------------------------------------------------------
def normalize_col(
    df: pd.DataFrame,
    col: str,
    *,
    method: str = "p02_p98",
    pmin: float = 2.0,
    pmax: float = 98.0,
    mad_clip: float = 3.0,
    eps: float = 1e-8,
    clip: bool = True,
    inplace: bool = False,
    new_col: str | None = None,
    return_normalizer: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Callable[[Any], FloatArray | float]]:
    """Robust-normalize df[col] to [0, 1] with stored (lo, hi) bounds."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in DataFrame.")

    out = df if inplace else df.copy()

    x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(x)
    vals = x[valid_mask]
    lo, hi = compute_norm_bounds(vals, method=method, pmin=pmin, pmax=pmax, mad_clip=mad_clip, eps=eps)

    denom = hi - lo if np.isfinite(hi - lo) and (hi - lo) > 0 else 1.0

    def _normalizer(values: Any) -> FloatArray | float:
        arr = np.asarray(values, dtype=np.float64)
        normed = (arr - lo) / denom
        if clip:
            normed = np.clip(normed, 0.0, 1.0, out=normed)
        return float(normed) if np.ndim(values) == 0 else normed

    # attach metadata for reproducibility
    _normalizer.lo = float(lo)  # type: ignore[attr-defined]
    _normalizer.hi = float(hi)  # type: ignore[attr-defined]
    _normalizer.denom = float(denom)  # type: ignore[attr-defined]
    _normalizer.clip = bool(clip)  # type: ignore[attr-defined]

    x_norm = (x - lo) / denom
    if clip:
        x_norm = np.clip(x_norm, 0.0, 1.0, out=x_norm)

    target_col = col if inplace else (new_col or f"{col}_norm")
    out[target_col] = x_norm

    if return_normalizer:
        return out, _normalizer
    return out


# -----------------------------------------------------------------------------
# Internal helpers for LOOCV / grid search (optional)
# -----------------------------------------------------------------------------
def _normalize_like_params(values: FloatArray, norm_params: dict[str, Any]) -> FloatArray:
    """Apply the SAME normalization as a heatmap (using lo/hi in params)."""
    lo = float(norm_params["lo"])
    hi = float(norm_params["hi"])
    v = np.asarray(values, float)
    v = np.clip(v, lo, hi)
    return (v - lo) / (hi - lo + 1e-12)


def _metrics_from_arrays(y_true: FloatArray, y_pred: FloatArray) -> dict[str, float]:
    """Compute loyalty-style metrics from arrays."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "NMAE_robust": float("nan"),
            "Pearson_r": float("nan"),
            "Spearman_rho": float("nan"),
            "CCC": float("nan"),
            "bias": float("nan"),
            "loa_low": float("nan"),
            "loa_high": float("nan"),
            "score": float("nan"),
        }

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    rscale = robust_scale(yt)
    nmae_robust = mae / (rscale + 1e-12)

    pr = pearsonr(yt, yp)[0] if yt.size > 1 else float("nan")
    sr = spearmanr(yt, yp)[0] if yt.size > 1 else float("nan")

    ccc = concordance_correlation_coefficient(yt, yp)
    ba = bland_altman(yt, yp)

    score = 0.6 * float(ccc) + 0.2 * float(pr) + 0.2 * float(1.0 - nmae_robust)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "NMAE_robust": float(nmae_robust),
        "Pearson_r": float(pr),
        "Spearman_rho": float(sr),
        "CCC": float(ccc),
        "bias": float(ba["bias"]),
        "loa_low": float(ba["loa_low"]),
        "loa_high": float(ba["loa_high"]),
        "score": float(score),
    }


def loocv_loyalty_for_params(
    template_img: nib.Nifti1Image,
    df: pd.DataFrame,
    combo: dict[str, Any],
    *,
    x_col: str = "MNI_x",
    y_col: str = "MNI_y",
    z_col: str = "MNI_z",
    value_col: str = "Value",
    sample_mode: Literal["nearest", "trilinear"] = "trilinear",
    verbose: bool = False,
) -> dict[str, Any]:
    """Leave-one-out CV wrapper for quick parameter sweeps."""
    _require_columns(df, [x_col, y_col, z_col, value_col])
    N = int(len(df))
    if N == 0:
        raise ValueError("df is empty.")

    y_true_all: list[float] = []
    y_pred_all: list[float] = []

    for i in range(N):
        df_train = df.drop(df.index[i], axis=0)
        df_test = df.iloc[[i]]

        _, heat_img, params = map_property_to_mni(
            template_img,
            df_train,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            value_col=value_col,
            colormap=combo.get("colormap", "viridis"),
            alpha=combo.get("alpha", 1.0),
            prob_threshold=combo.get("prob_threshold", 0.05),
            inflate_mm=combo.get("inflate_mm", 0.0),
            smooth_mm=combo.get("smooth_mm", 0.0),
            smooth_threshold=combo.get("smooth_threshold", 0.5),
            use_dbscan=combo.get("use_dbscan", False),
            dbscan_min_samples=combo.get("dbscan_min_samples", 2),
            dbscan_eps_mm=combo.get("dbscan_eps_mm", None),
            method=combo.get("method", "rbf"),
            rbf_kernel=combo.get("rbf_kernel", "thin_plate_spline"),
            rbf_smoothing=combo.get("rbf_smoothing", 0.0),
            rbf_neighbors=combo.get("rbf_neighbors", None),
            rbf_degree=combo.get("rbf_degree", None),
            idw_k=combo.get("idw_k", 16),
            idw_power=combo.get("idw_power", 2.0),
            sigma_mm=combo.get("sigma_mm", 0.0),
            norm=combo.get("norm", False),
            norm_method=combo.get("norm_method", "p02_p98"),
            norm_pmin=combo.get("norm_pmin", 2.0),
            norm_pmax=combo.get("norm_pmax", 98.0),
            norm_mad_clip=combo.get("norm_mad_clip", 3.0),
            voxel_center_offset_vox=combo.get("voxel_center_offset_vox", DEFAULT_VOXEL_CENTER_OFFSET_VOX),
        )

        xyz = df_test[[x_col, y_col, z_col]].to_numpy(float)
        pred = float(sample_nifti_at_mni(heat_img, xyz, mode=sample_mode)[0])

        if combo.get("norm", False):
            y_true_norm = _normalize_like_params(
                df_test[value_col].to_numpy(float),
                params["normalization"],
            )
            y_true_all.append(float(y_true_norm[0]))
        else:
            y_true_all.append(float(df_test[value_col].to_numpy(float)[0]))

        y_pred_all.append(float(pred))

        if verbose:
            logger.info("LOOCV %d/%d done", i + 1, N)

    metrics = _metrics_from_arrays(np.asarray(y_true_all, float), np.asarray(y_pred_all, float))
    metrics["N"] = int(N)
    return {"combo": combo, **metrics}


def grid_search_loyalty(
    template_img: nib.Nifti1Image,
    df: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    *,
    x_col: str = "MNI_x",
    y_col: str = "MNI_y",
    z_col: str = "MNI_z",
    value_col: str = "Value",
    sample_mode: Literal["nearest", "trilinear"] = "trilinear",
    top_k: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """Exhaustive grid search using LOOCV; returns a DataFrame sorted by score."""
    keys = list(param_grid.keys())
    results: list[dict[str, Any]] = []

    for values in product(*[param_grid[k] for k in keys]):
        combo = {k: v for k, v in zip(keys, values)}
        res = loocv_loyalty_for_params(
            template_img,
            df,
            combo,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            value_col=value_col,
            sample_mode=sample_mode,
            verbose=verbose,
        )
        results.append(res)
        if verbose:
            logger.info(
                "Done combo %s -> score=%.3f, CCC=%.3f, NMAE_r=%.3f",
                combo,
                res.get("score", float("nan")),
                res.get("CCC", float("nan")),
                res.get("NMAE_robust", float("nan")),
            )

    df_res = pd.DataFrame(results).sort_values(by="score", ascending=False)

    try:
        from pprint import pprint

        print("\nTop candidates:")
        for _, row in df_res.head(int(top_k)).iterrows():
            c = row["combo"]
            pprint(
                {
                    **c,
                    "score": round(float(row["score"]), 3),
                    "CCC": round(float(row["CCC"]), 3),
                    "Pearson_r": round(float(row["Pearson_r"]), 3),
                    "NMAE_robust": round(float(row["NMAE_robust"]), 3),
                }
            )
    except Exception:
        pass

    return df_res


# -----------------------------------------------------------------------------
# Public symbols
# -----------------------------------------------------------------------------
__all__ = [
    "voxelize",
    "map_property_to_mni",
    "pointcloud_to_voxel_mean_nifti",
    "convex_hull_mask_in_template",
    "sample_nifti_at_mni",
    "compute_loyalty_metrics",
    "normalize_col",
    "grid_search_loyalty",
    "loocv_loyalty_for_params",
    "compute_radial_scale",
    "compute_norm_bounds",
    "robust_scale",
]
