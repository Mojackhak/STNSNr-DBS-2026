"""
Merge 2D pose H5 files with matched coordinate-frame markers (o, x, y).
"""

import re
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.posetrack.track_io import load_h5_2d
from src.utils.paths import filter_files_by_re


def find_matched_cam(
    file: str | Path,
    files: Iterable[str | Path],
    contain: str = r"(.*)(cam\d)(.*)",
    verbose: bool = False,
) -> Path | None:
    """Find the first file in `files` that corresponds to the same camera token (e.g., cam1)."""
    file = Path(file)
    basename = file.name

    m = re.search(contain, basename)
    if m is None:
        if verbose:
            print(f"No camera token matched in: {file}")
        return None

    # Keep original intent: default regex puts 'cam\\d' in group(2).
    cam_token = m.group(2) if (m.lastindex or 0) >= 2 else m.group(1)

    candidates = filter_files_by_re([str(Path(f)) for f in files], cam_token)
    if not candidates:
        if verbose:
            print(f"No matched files found for {file}")
        return None

    if len(candidates) > 1 and verbose:
        print(f"Multiple matched files found for {file}, returning the first one.")

    return Path(candidates[0])


def merge_world_xyz(
    pose_2d_files: Iterable[str | Path],
    cs_files: Iterable[str | Path],
    contain: str = r"(.*)(cam\d)(.*)",
    zoom: float | None = 1.0,
) -> None:
    """
    For each pose 2D file:
    - Optionally scale x/y by `zoom`
    - Add coordinate-frame bodyparts: o, x, y (each has x/y/likelihood)
    - Fill those markers with the first row of the matched cs file (if found)
    - Overwrite the original H5 file
    """
    pose_2d_files = [Path(p) for p in pose_2d_files]
    cs_files = [Path(p) for p in cs_files]

    idx = pd.IndexSlice

    for pose_file in pose_2d_files:
        cs_file = find_matched_cam(pose_file, cs_files, contain=contain, verbose=False)

        data, metadata = load_h5_2d(str(pose_file))
        if "o" in metadata["bodyparts"]:
            print(f"{pose_file} already contains 'o' bodypart. Skipping... (processed?)")
            continue

        if zoom is not None and zoom != 1:
            data.loc[:, idx[:, ["x", "y"]]] *= zoom

        # Create empty coordinate-frame markers: bodyparts=['o','x','y'], coords=['x','y','likelihood']
        cs_cols = pd.MultiIndex.from_product(
            [["o", "x", "y"], ["x", "y", "likelihood"]],
            names=["bodyparts", "coords"],
        )
        data_cs_new = pd.DataFrame(index=data.index, columns=cs_cols)

        if cs_file is not None:
            data_cs, _ = load_h5_2d(str(cs_file))
            data_cs_new = data_cs_new.fillna(data_cs.iloc[0])

        data_new = pd.concat([data, data_cs_new], axis=1)

        scorer = metadata["scorer"]
        data_new.columns = pd.MultiIndex.from_tuples(
            [(scorer, bp, coord) for bp, coord in data_new.columns],
            names=["scorer", "bodyparts", "coords"],
        )

        out_file = pose_file  # overwrite original

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            data_new = data_new.infer_objects()  # keep float dtype where possible
            data_new.to_hdf(str(out_file), key="df_with_missing", mode="w")

        print(f"Processed {pose_file}")
