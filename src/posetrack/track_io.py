from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd



def load_h5_2d(fname: str, scorer_name: str = "built_in"):
    """
    Load 2D pose data from a DeepLabCut-style HDF5 file.

    Behavior (kept the same as the original):
    - If the DataFrame has a column level named 'scorer', use the first scorer and slice to it.
    - Otherwise, treat the DataFrame as already sliced, and set scorer to `scorer_name`.

    Parameters
    ----------
    fname : str
        Path to the .h5 file.
    scorer_name : str
        Fallback scorer name when 'scorer' is not present in column level names.

    Returns
    -------
    data : pd.DataFrame
        Pose data table (possibly sliced by scorer).
    metadata : dict
        Basic metadata including bodyparts/coords/frame counts.
    """
    data_orig = pd.read_hdf(fname)

    col_names = list(data_orig.columns.names) if data_orig.columns is not None else []
    if "scorer" in col_names:
        scorer = data_orig.columns.levels[0][0]
        data = data_orig.loc[:, scorer]
    else:
        scorer = scorer_name
        data = data_orig

    bp_index = data.columns.names.index("bodyparts")
    coord_index = data.columns.names.index("coords")

    bodyparts = list(data.columns.get_level_values(bp_index).unique())
    coords = list(data.columns.get_level_values(coord_index).unique())

    n_frames = len(data)
    n_joints = len(bodyparts)

    metadata = {
        "bodyparts": bodyparts,
        "scorer": scorer,
        "coords": coords,
        "index": data.index,
        "n_frames": n_frames,
        "n_joints": n_joints,
    }
    return data, metadata



def load_csv_3d(
    pose_3d_file_path: str | Path,
    config_file_path: str | Path | None = None,
    bp_side: bool = False,
    cs_info: bool = False,
    cam_info: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load 3D pose data from a CSV file.

    Behavior kept the same as the original:
    - If cam_info is False and "M_00" exists, truncate columns before "M_00".
    - If cs_info is False and "o_x" exists (after the above truncation), truncate columns before "o_x".
    - Parse columns by splitting on '_' to infer bodyparts/coords (and optional side).

    Returns
    -------
    data : pd.DataFrame
        The loaded (and possibly truncated) pose table.
    metadata : dict
        Keys: bodyparts, coords, side, index, scheme, n_frames, n_joints.
    """
    pose_3d_file_path = Path(pose_3d_file_path)

    scheme = None
    if config_file_path is not None:
        config_file_path = Path(config_file_path)
        # Keep logic: only used to read labeling.scheme when config exists
        import toml  # local import to avoid hard dependency when config_file_path is None

        with config_file_path.open("r", encoding="utf-8") as f:
            config_data = toml.load(f)
        scheme = config_data["labeling"]["scheme"]

    data = pd.read_csv(pose_3d_file_path)

    # Keep original truncation logic
    if not cam_info and "M_00" in data.columns:
        m_pos = data.columns.get_loc("M_00")
        data = data.iloc[:, :m_pos]

        if not cs_info and "o_x" in data.columns:
            o_pos = data.columns.get_loc("o_x")
            data = data.iloc[:, :o_pos]

    # Parse bodyparts / coords / side
    bodyparts: list[str] = []
    coords: list[str] = []
    side: list[str] | None

    if bp_side:
        side = []
        for col in data.columns:
            if "_" not in col:
                continue

            parts = col.rsplit("_", 2)
            if len(parts) == 3:
                bp, sd, coord = parts
                bodyparts.append(bp)
                side.append(sd)
                coords.append(coord)
            else:
                bp, coord = col.rsplit("_", 1)
                bodyparts.append(bp)
                coords.append(coord)
    else:
        side = None
        # Preserve ordering while removing duplicates
        bodyparts = list(dict.fromkeys(col.rsplit("_", 1)[0] for col in data.columns if "_" in col))
        coords = list(dict.fromkeys(col.rsplit("_", 1)[1] for col in data.columns if "_" in col))

    n_frames = int(len(data))
    n_coords = max(1, len(coords))  # prevent ZeroDivision if malformed CSV
    n_joints = int(len(data.columns) / n_coords)

    metadata: dict[str, Any] = {
        "bodyparts": bodyparts,
        "coords": coords,
        "side": side,
        "index": data.index,
        "scheme": scheme,
        "n_frames": n_frames,
        "n_joints": n_joints,
    }
    return data, metadata


