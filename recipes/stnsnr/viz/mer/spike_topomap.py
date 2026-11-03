#%%
import os
from pathlib import Path
import sys

import nibabel as nib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[4]))
import src.utils.fileops as fileops
import plugin.default.viz_defaults as cfg
from src.anat.voxelize import map_property_to_mni
from src.io.pkl_io import load_pkl, save_pkl

#%% Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])
summary_dir = project / "summary"
stats_dir = summary_dir / "stats"
spike_dir = stats_dir / "mer" / "spike"
coords_path = spike_dir / "spike_unit_coords.pkl"
filebase = "spike_unit_cover"

#%% Load atlas
_, mask_dict, _ = cfg.load_atlas(project)


def prepare_spike_cover_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid SNr spike unit coordinates with a constant coverage value."""
    required_cols = [
        "ID",
        "Region",
        "Side",
        "Hemi",
        "MNI_x_flip",
        "MNI_y_flip",
        "MNI_z_flip",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required coordinate columns: {missing}")

    df_cover = df.loc[df["Region"] == "SNr", required_cols].copy()
    df_cover = df_cover.dropna(subset=["MNI_x_flip", "MNI_y_flip", "MNI_z_flip"])
    df_cover = df_cover.drop_duplicates(
        subset=["MNI_x_flip", "MNI_y_flip", "MNI_z_flip"],
        keep="first",
    )
    df_cover = df_cover.reset_index(drop=True)
    df_cover["Value"] = 1.0
    return df_cover


def write_spike_cover_topomap() -> Path:
    """Voxelize MER spike unit coordinate coverage and write topomap artifacts."""
    topomap_dir = fileops.create_subfolder(spike_dir, "topomap")
    cover_dir = fileops.create_subfolder(topomap_dir, "cover")

    df_coords = load_pkl(coords_path)
    df_cover = prepare_spike_cover_dataframe(df_coords)
    if df_cover.empty:
        raise ValueError("No valid SNr spike unit coordinates were found.")

    region = "SNr"
    save_dir = fileops.create_subfolder(cover_dir, region)
    csv_path = save_dir / f"{filebase}.csv"
    pkl_path = save_dir / f"{filebase}.pkl"
    log_path = save_dir / f"{filebase}_log.pkl"
    nii_path = save_dir / f"{filebase}.nii.gz"

    df_cover.to_csv(csv_path, index=False)
    save_pkl(df_cover, pkl_path)

    _, img, log = map_property_to_mni(
        mask_dict[region],
        df_cover,
        **cfg.voxelization_cfg,
    )
    save_pkl(log, log_path)
    nib.save(img, nii_path)

    print(f"Wrote {len(df_cover)} spike unit coverage coordinates to: {save_dir}")
    print(f"Wrote coverage NIfTI: {nii_path}")
    return save_dir


#%%
if __name__ == "__main__":
    write_spike_cover_topomap()

