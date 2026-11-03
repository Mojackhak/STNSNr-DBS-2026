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

subj_dir = summary_dir / 'cohort' / 'subj'
df_scale_path = subj_dir / 'subj_delta_effect.xlsx'
df_scale = pd.read_excel(df_scale_path)
df_meta = df_scale[["ID"]].copy()

clinic_dir = summary_dir / 'stats' / 'clinic'
pca_dir = fileops.create_subfolder(clinic_dir, "pca")
wk_dir = fileops.create_subfolder(pca_dir, feature)

#%%
dfs = []

for df_path in df_paths:
    df = pd.read_excel(df_path)
    df = pd.concat([df_meta, df], axis=1)
    df = pd.melt(
        df,
        id_vars=["ID"],            
        var_name="Phase",     
        value_name="Value" 
    )
    df.insert(1, 'Feature', df_path.parent.name)
    df["Phase"] = (
        df["Phase"]
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    ) 
    
    df = df[["ID", "Feature", "Phase", "Value"]]
    dfs.append(df)

df_long = pd.concat(dfs, axis=0, ignore_index=True)
df_long = df_long.sort_values(["Feature", "Phase", "ID"])
df_long = df_long.reset_index(drop=True)
save_path = wk_dir / f'{feature}_long.pkl'
df_long.to_pickle(save_path)
df_long.to_excel(save_path.with_suffix(".xlsx"), index=False)


#%%
df_wide = df_long.pivot_table(
    index=["ID", "Phase"],
    columns="Feature",
    values="Value",
    aggfunc="mean"
)

save_path = wk_dir / f'{feature}_wide.pkl'
df_wide.to_pickle(save_path)
df_wide.to_excel(save_path.with_suffix(".xlsx"), index=False)

#%%
df_agg = summarize_df(
    df_long, group_cols=["Feature", "Phase"],
    mode = 'mean',
    value_col="Value",
    )

#%%
df_agg.rename(columns={"Phase": "Protocol_Phase"}, inplace=True)
df_agg[["Protocol", "Phase"]] = (
    df_agg["Protocol_Phase"].str
    .extract(r"^\s*(.*?)\s*\((.*?)\)\s*$")
)
df_agg = df_agg[["Protocol_Phase", "Protocol", "Phase", "Feature", "Value"]]
df_agg = df_agg.reset_index(drop=True)

df_agg["p"] = 1.0
df_agg_pre = df_agg[df_agg["Protocol_Phase"] == "Pre-op"].copy().reset_index(drop=True)
df_agg_stim = df_agg[df_agg["Protocol_Phase"] != "Pre-op"].copy().reset_index(drop=True)
df_agg_3m = df_agg_stim[df_agg_stim["Phase"] == "3m"].copy().reset_index(drop=True)

df_agg_pre["Protocol"] = "Pre-op"
df_agg_pre["Phase"] = "Pre-op"

# %%
df_stim = df_long.copy().reset_index(drop=True)
df_stim.rename(columns={"Phase": "Protocol_Phase"}, inplace=True)
df_stim[["Protocol", "Phase"]] = (
    df_stim["Protocol_Phase"].str
    .extract(r"^\s*(.*?)\s*\((.*?)\)\s*$")
)
df_stim = df_stim[["ID", "Protocol_Phase", "Protocol", "Phase", "Feature", "Value"]]

df_3m = df_stim[df_stim["Phase"] == "3m"].copy().reset_index(drop=True)

df_pre = df_stim[df_stim["Phase"].isna()].copy().reset_index(drop=True)
df_pre["Protocol"] = "Pre-op"
df_pre["Phase"] = "Pre-op"

#%%
group_cols = ["Feature", "Phase"]
for keys, df_sub in df_3m.groupby(group_cols, observed=True):
    results = stats.single_var_stat_pair_test(
        df_sub, var="Protocol", var_pair="ID", value="Value")
    mask_dict = {g:k for g, k in zip(group_cols, keys)}
    mask = pd.Series(True, index=df_agg_3m.index)
    for col, val in mask_dict.items():
        mask &= (df_agg_3m[col] == val)   
    df_agg_3m.loc[mask, "p"] = results['p_value'] 

#%%
df_agg = pd.concat([df_agg_pre, df_agg_3m], axis=0, ignore_index=True)
df_agg = df_agg.reset_index(drop=True)
df_agg["Value_raw"] = df_agg["Value"]
df_agg[["Value",]] = df_agg.groupby(
    ["Feature",], observed=True)[["Value",]].transform(cfg.r_scale_like)


save_path = wk_dir / f'{feature}_agg.pkl'
df_agg.to_pickle(save_path)
df_agg.to_excel(save_path.with_suffix(".xlsx"), index=False)


# %%
