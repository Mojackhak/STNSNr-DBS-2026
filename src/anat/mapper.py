"""mapper.py

Map DBS contact coordinates (MNI space) to atlas regions.

This module takes contact coordinates (typically exported by Lead-DBS) and maps each contact
to user-provided atlas regions stored as NIfTI masks. For every contact and every region, it
computes:
- whether the contact falls inside the region mask (boolean),
- relative (x, y, z) offset from the region's center of gravity (CoG),
- a summary label `region`: "EXT" (outside all regions), region name (inside exactly one),
  or "Mid" (inside more than one region).

Design constraints (kept intentionally simple):
- Lead-DBS `.mat` structure is treated as fixed (hard-coded indexing).
- During a single mapping call, each atlas NIfTI file is loaded at most once, and each
  region CoG is computed at most once (cached) to avoid repeated IO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, Literal

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io


# -----------------------------------------------------------------------------
# Public low-level helpers (no caching across calls)
# -----------------------------------------------------------------------------
def points_in_region(coords: np.ndarray, region_nii_path: str, threshold: float) -> np.ndarray:
    """Check whether points (MNI) fall inside a region mask.

    Note: This function loads the NIfTI on every call. For batch mapping, prefer
    `get_points_loc()` / `get_points_loc_df()` which cache NIfTI volumes per call.
    """
    img = nib.load(region_nii_path)
    data = img.get_fdata()
    inv_affine = np.linalg.inv(img.affine)

    coords_arr = np.asarray(coords, dtype=float)
    hom = np.hstack([coords_arr, np.ones((coords_arr.shape[0], 1))])
    vox = (inv_affine @ hom.T).T[:, :3]
    ijk = np.round(vox).astype(int)

    x_max, y_max, z_max = data.shape
    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < x_max)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < y_max)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < z_max)
    )

    inside = np.zeros(coords_arr.shape[0], dtype=bool)
    if np.any(valid):
        vi, vj, vk = ijk[valid].T
        inside[valid] = data[vi, vj, vk] > threshold
    return inside


def get_region_mid(region_nii_path: str, threshold: float = 0.0) -> np.ndarray:
    """Compute the center of gravity (CoG) of a region in MNI space.

    Note: This function loads the NIfTI on every call. For batch mapping, prefer
    `get_points_loc()` / `get_points_loc_df()` which cache NIfTI volumes per call.
    """
    img = nib.load(region_nii_path)
    data = img.get_fdata()
    affine = np.asarray(img.affine, dtype=float)

    mask = data > threshold
    if not np.any(mask):
        return np.zeros(3, dtype=float)

    ijk = np.column_stack(np.nonzero(mask))
    weights = data[mask].astype(float)
    total = float(np.sum(weights))
    if total == 0.0:
        return np.zeros(3, dtype=float)

    hom = np.hstack([ijk, np.ones((ijk.shape[0], 1))])
    xyz = (affine @ hom.T).T[:, :3]
    cog = np.sum(xyz * weights[:, None], axis=0) / total
    return np.asarray(cog, dtype=float)


def get_mid_coords(coords: np.ndarray) -> np.ndarray:
    """Compute midpoints between successive points."""
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape[0] < 2:
        return np.empty((0, 3), dtype=float)
    return (coords_arr[:-1, :] + coords_arr[1:, :]) / 2.0


# -----------------------------------------------------------------------------
# Internal per-call caching utilities
# -----------------------------------------------------------------------------
_NiftiCacheValue = tuple[np.ndarray, np.ndarray, np.ndarray]  # data, affine, inv_affine


def _load_region_volume(region_path: str, nifti_cache: dict[str, _NiftiCacheValue]) -> _NiftiCacheValue:
    """Load a NIfTI region volume with per-call caching."""
    cached = nifti_cache.get(region_path)
    if cached is not None:
        return cached

    img = nib.load(region_path)
    data = img.get_fdata()  # keep dtype as-is; atlas files are typically small masks
    affine = np.asarray(img.affine, dtype=float)
    inv_affine = np.linalg.inv(affine)

    nifti_cache[region_path] = (data, affine, inv_affine)
    return nifti_cache[region_path]


def _points_in_region_cached(
    coords: np.ndarray,
    data: np.ndarray,
    inv_affine: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Vectorized point-in-region check using a preloaded volume."""
    coords_arr = np.asarray(coords, dtype=float)

    hom = np.hstack([coords_arr, np.ones((coords_arr.shape[0], 1))])
    vox = (inv_affine @ hom.T).T[:, :3]
    ijk = np.round(vox).astype(int)

    x_max, y_max, z_max = data.shape
    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < x_max)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < y_max)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < z_max)
    )

    inside = np.zeros(coords_arr.shape[0], dtype=bool)
    if np.any(valid):
        vi, vj, vk = ijk[valid].T
        inside[valid] = data[vi, vj, vk] > threshold
    return inside


def _region_cog_cached(
    region_path: str,
    threshold: float,
    nifti_cache: dict[str, _NiftiCacheValue],
    cog_cache: dict[tuple[str, float], np.ndarray],
) -> np.ndarray:
    """Compute (and cache) the region CoG for a given threshold."""
    key = (region_path, float(threshold))
    cached = cog_cache.get(key)
    if cached is not None:
        return cached

    data, affine, _ = _load_region_volume(region_path, nifti_cache)
    mask = data > threshold
    if not np.any(mask):
        cog = np.zeros(3, dtype=float)
        cog_cache[key] = cog
        return cog

    ijk = np.column_stack(np.nonzero(mask))
    weights = data[mask].astype(float)
    total = float(np.sum(weights))
    if total == 0.0:
        cog = np.zeros(3, dtype=float)
        cog_cache[key] = cog
        return cog

    hom = np.hstack([ijk, np.ones((ijk.shape[0], 1))])
    xyz = (affine @ hom.T).T[:, :3]
    cog = np.sum(xyz * weights[:, None], axis=0) / total

    cog_cache[key] = np.asarray(cog, dtype=float)
    return cog_cache[key]


def _map_coords_to_regions(
    coords: np.ndarray,
    index: pd.Index,
    side: str,
    region_list: list[str],
    atlas_path: str,
    threshold: float,
    nifti_cache: dict[str, _NiftiCacheValue],
    cog_cache: dict[tuple[str, float], np.ndarray],
) -> pd.DataFrame:
    """Map a coordinate array to all regions for a single hemisphere/side."""
    coords_arr = np.asarray(coords, dtype=float)
    out_frames: list[pd.DataFrame] = []

    for region in region_list:
        # Keep hard-coded atlas layout: {atlas_path}/{side}/{region}.nii.gz
        region_path = os.path.join(atlas_path, side, f"{region}.nii.gz")

        data, _, inv_affine = _load_region_volume(region_path, nifti_cache)
        region_mid = _region_cog_cached(region_path, threshold, nifti_cache, cog_cache)

        inside = _points_in_region_cached(coords_arr, data, inv_affine, threshold)
        rel = coords_arr - region_mid.reshape(1, 3)

        out_frames.append(
            pd.DataFrame(
                {
                    f"{region}_in": inside,
                    f"{region}_x": rel[:, 0],
                    f"{region}_y": rel[:, 1],
                    f"{region}_z": rel[:, 2],
                },
                index=index,
            )
        )

    if not out_frames:
        return pd.DataFrame(index=index)

    return pd.concat(out_frames, axis=1)


def _compute_region(mapped_df: pd.DataFrame, region_list: list[str]) -> np.ndarray:
    """Compute region from multiple region membership columns."""
    if not region_list:
        return np.array([], dtype=object)

    in_cols = [f"{r}_in" for r in region_list]
    in_mat = mapped_df[in_cols].to_numpy(dtype=bool)
    counts = in_mat.sum(axis=1)

    loc = np.full(mapped_df.shape[0], "EXT", dtype=object)

    one = counts == 1
    if np.any(one):
        region_arr = np.asarray(region_list, dtype=object)
        idx = in_mat[one].argmax(axis=1)
        loc[one] = region_arr[idx]

    loc[counts > 1] = "Mid"
    return loc


# -----------------------------------------------------------------------------
# Public mapping APIs
# -----------------------------------------------------------------------------
def get_mni_coords(d):
    return d["coords"]["mni"][0][0][0]

def get_mni_reco(d):
    return d["reco"]["mni"][0][0][0][0][0][0]

COORDS_MODE = Literal['coords', 'reco']
POINT_NAMES_MODE = Literal['single', 'pair']

def get_points_loc(
    coordsFile: str | Path,
    region_list: list[str],
    atlas_path: str | Path,
    threshold: float,
    side_dict: dict[str, int | Sequence[int]],
    coords_mode: COORDS_MODE = 'coords',
    point_names_mode: POINT_NAMES_MODE = 'pair'
) -> pd.DataFrame:
    """Map Lead-DBS exported contact coordinates to atlas regions.

    Parameters
    ----------
    coordsFile:
        Path to Lead-DBS .mat export (hard-coded structure).
    region_list:
        List of atlas region names (NIfTI basenames).
    atlas_path:
        Atlas root folder. Expected layout: {atlas_path}/{side}/{region}.nii.gz
    threshold:
        Threshold used for region membership and CoG computation.
    side_dict:
        Mapping from side folder name (e.g. 'lh', 'rh') to Lead-DBS index (e.g. 0/1).
        If multiple leads per side, pass a sequence of indices. E.g. {'lh': [0, 2], 'rh': 1}

    Returns
    -------
    DataFrame indexed by contact label with columns:
        side, MNI_x, MNI_y, MNI_z,
        {region}_in, {region}_x/{region}_y/{region}_z for each region,
        region
    """
    coords_raw: dict[str, Any] = scipy.io.loadmat(str(coordsFile))

    # Per-call caches (avoid repeated disk IO within this call)
    nifti_cache: dict[str, _NiftiCacheValue] = {}
    cog_cache: dict[tuple[str, float], np.ndarray] = {}

    points_num = 0
    side_frames: list[pd.DataFrame] = []

    for side, idxs in side_dict.items():
        # Allow multiple electrodes/leads per hemisphere by passing a sequence of indices.
        if isinstance(idxs, (list, tuple, np.ndarray)):
            idx_list = [int(i) for i in idxs]
        else:
            idx_list = [int(idxs)]

        for lead_num, idx in enumerate(idx_list):
            # Keep hard-coded Lead-DBS structure access.
            if coords_mode == 'coords':
                coords = get_mni_coords(coords_raw)[idx]
            else:  # coords_mode == 'reco'
                coords = get_mni_reco(coords_raw)[idx]
            if point_names_mode == 'pair':
                point_names = [
                    f"{i + points_num}_{i + points_num + 1}" for i in range(coords.shape[0])
                ]
                points_num += coords.shape[0] + 1
            else:  # point_names_mode == 'single'
                point_names = [
                    f"{i + points_num}" for i in range(coords.shape[0])
                ]
                points_num += coords.shape[0]
            
            points_index = pd.Index(point_names, name="contact")

            basic_df = pd.DataFrame(
                {
                    "side": np.array([side] * coords.shape[0]),
                    "lead_idx": np.array([idx] * coords.shape[0]),  
                    "lead_num": np.array([lead_num] * coords.shape[0]), 
                    "MNI_x": coords[:, 0],
                    "MNI_y": coords[:, 1],
                    "MNI_z": coords[:, 2],
                },
                index=points_index,
            )

            mapped_df = _map_coords_to_regions(
                coords=coords,
                index=points_index,
                side=side,
                region_list=region_list,
                atlas_path=atlas_path,
                threshold=threshold,
                nifti_cache=nifti_cache,
                cog_cache=cog_cache,
            )

            side_frames.append(pd.concat([basic_df, mapped_df], axis=1))


    if not side_frames:
        return pd.DataFrame()

    points_loc_df = pd.concat(side_frames, axis=0)
    points_loc_df["region"] = _compute_region(points_loc_df, region_list)
    return points_loc_df


def get_points_loc_batch(
    coordsFiles: list[Path],
    region_list: list[str],
    atlas_path: str | Path,
    threshold: float,
    side_dict: dict[str, int | Sequence[int]],  
    save_name: str,
    coords_mode: COORDS_MODE = 'coords',
    point_names_mode: POINT_NAMES_MODE = 'pair',      
) -> list[Path]:
    """Batch process multiple Lead-DBS .mat files and save results as CSV."""
    saved_paths: list[Path] = []
    for coordsFile in coordsFiles:
        coordsFile = Path(coordsFile)
        print(f"Processing {coordsFile} ...")
        points_loc_df = get_points_loc(
            coordsFile, region_list, atlas_path, threshold, 
            side_dict, coords_mode, point_names_mode)
        save_path = coordsFile.parent / save_name
        points_loc_df.to_csv(save_path, index=True)
        print(f"Saved results to {save_path}")
        saved_paths.append(save_path)

    return saved_paths


def get_points_loc_df(
    df: pd.DataFrame,
    region_list: list[str],
    atlas_path: str | Path,
    threshold: float,
) -> pd.DataFrame:
    """Map a DataFrame of contact coordinates to atlas regions.

    The input DataFrame must include columns:
        - side (values 'L'/'R' or 'lh'/'rh')
        - MNI_x, MNI_y, MNI_z
    """
    if df.empty:
        return df.copy()

    required_cols = {"side", "MNI_x", "MNI_y", "MNI_z"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Per-call caches (avoid repeated disk IO within this call)
    nifti_cache: dict[str, _NiftiCacheValue] = {}
    cog_cache: dict[tuple[str, float], np.ndarray] = {}

    # Keep the original 'side' column; only normalize for atlas folder lookup.
    side_norm = df["side"].replace({"L": "lh", "R": "rh"}).astype(str)

    mapped_frames: list[pd.DataFrame] = []
    for side in side_norm.unique():
        mask = side_norm == side
        idx = df.index[mask]
        coords = df.loc[idx, ["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)

        mapped = _map_coords_to_regions(
            coords=coords,
            index=pd.Index(idx),
            side=side,
            region_list=region_list,
            atlas_path=atlas_path,
            threshold=threshold,
            nifti_cache=nifti_cache,
            cog_cache=cog_cache,
        )
        mapped_frames.append(mapped)

    mapped_df = pd.concat(mapped_frames, axis=0).reindex(df.index)
    mapped_df["region"] = _compute_region(mapped_df, region_list)

    return pd.concat([df, mapped_df], axis=1)


def make_paired_df(
    df: pd.DataFrame,
    subject_col: str = "subject",
    contact_col: str = "contact",
) -> pd.DataFrame:
    """Generate all ordered contact pairs (with replacement) within each subject.

    For contact_col and every other column return a tuple: (val1, val2)

    Returns a DataFrame with:
        - subject_col
        - one tuple type column for each other column in `df`.
    """
    required = {subject_col, contact_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    other_cols = [c for c in df.columns if c not in (subject_col, contact_col)]

    out_frames: list[pd.DataFrame] = []
    for subj, g in df.groupby(subject_col, sort=False):
        g = g[[contact_col] + other_cols].copy().reset_index(drop=True)

        g1 = g.copy()
        g2 = g.copy()
        g1["__key__"] = 1
        g2["__key__"] = 1
        pairs = g1.merge(g2, on="__key__", suffixes=("1", "2")).drop(columns="__key__")

        res = pd.DataFrame({subject_col: [subj] * len(pairs)})
        res[contact_col] = list(zip(pairs[f"{contact_col}1"].astype(str), 
                                              pairs[f"{contact_col}2"].astype(str)))
        
        for col in other_cols:
            s1 = pairs[f"{col}1"]
            s2 = pairs[f"{col}2"]
            res[col] = list(zip(s1, s2))

        out_frames.append(res)

    if not out_frames:
        return pd.DataFrame(columns=[subject_col, contact_col] + other_cols)

    return pd.concat(out_frames, ignore_index=True)
