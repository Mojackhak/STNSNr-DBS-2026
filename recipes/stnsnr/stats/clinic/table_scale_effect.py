#%%
import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))

import src.utils.paths as paths
import src.utils.fileops as fileops
from src.tabular.text import normalize_side
from src.stats.preproc.aggregate import summarize_df
from src.stats.preproc.normalize import normalize_df
import plugin.default.viz_defaults as cfg
import src.stats.stats as stats

import importlib
importlib.reload(cfg)

#%% 1) Project paths and cohort-level metadata

project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS

summary_dir = project / 'summary'
clinic_dir = summary_dir / 'stats' / 'clinic'
scale_dir = clinic_dir / 'effect_old'
feature = "3m"
data_dir = scale_dir / feature
df_paths = paths.get_files_list(data_dir, filetypes='.xlsx', 
                                contain='df', recursive=True)

clinic_dir = summary_dir / 'stats' / 'clinic'
pca_dir = fileops.create_subfolder(clinic_dir, "pca")
wk_dir = fileops.create_subfolder(pca_dir, feature)

#%%
df_long_path = clinic_dir / "pca" / feature / f'{feature}_long.pkl'
df_long = pd.read_pickle(df_long_path)

#%%
df_effect = df_long.copy().rename(columns={"Phase": "Condition"})
df_effect["Condition"] = df_effect["Condition"].str.replace("3m", "3 m")
condition_map = {
    "Pre-op": "A",
    "STN (immediate)": "B",
    "STN+SNr (immediate)": "C",
    "STN (3 m)": "D",
    "STN+SNr (3 m)": "E",
}

df_effect["Group"] = df_effect["Condition"].map(condition_map)
df_effect.sort_values(["Feature", "Group", "ID"], inplace=True)
save_path = clinic_dir / "effect" / feature / f'scale_subject.pkl'
save_path.parent.mkdir(parents=True, exist_ok=True)
df_effect.to_pickle(save_path)
df_effect.to_excel(save_path.with_suffix(".xlsx"), index=False)

# %%
