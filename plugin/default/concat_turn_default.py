"""
# Necessary parameters for tabular/concat pipeline
1. make_features_dict()
2. paradigm_dict
3. time_axis
4. normalize_func_dict

"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.tabular.text import normalize_side, normalize_fog, capitalize_df, tuple_1st
from src.tabular.selection import slice_df
from plugin.default.tabular_default import standardize_df
from typing import Any


def make_features_dict(): # create by function to avoid shared references
    return {
        'tfr': {'raw': [], 'aperiod': [], 'period': []},
        'params': {'params': []},
        'conn': {'coh': [], 'ciplv': [], 'wpli': []},
        'gc': {'delta_net_gc': []},
        'psi': {'psi': []},
        'burst': {'burst': []},
    }

paradigm_dict = {
    'turn_feature':(
        r"turn_L_feature", r"turn_R_feature", 
        r"turn_dual_L_feature", r"turn_dual_R_feature"), 
    }

time_axis = {
    'default': 'percent',
}

slices_default = {'Region': ['STN', 'SNr', ("SNr", "STN"),]}


is_direction_list = ['turn_L_feature', 'turn_R_feature', 
                     'turn_dual_L_feature', 'turn_dual_R_feature',]

is_dual_list = ['turn_dual_L_feature', 'turn_dual_R_feature',]

def normalize_pose(df: pd.DataFrame) -> pd.DataFrame:
    df["Epoch"] = df["Epoch"].str[1].astype(int)
    
    mask_lat = df["Paradigm"].astype("string").str.lower().isin({x.lower() for x in is_direction_list})
    df["Pose"] = df["Paradigm"].astype("string")
    df["Side_pose"] = "Bilat"
    df.loc[mask_lat, ["Pose", "Side_pose"]] = (
        df.loc[mask_lat, "Paradigm"]
        .astype("string")
        .str.replace("_feature", "", regex=False)
        .str.rsplit("_", n=1, expand=True)
        .to_numpy()
    )

    df["Side_pose"] = df["Side_pose"].map(normalize_side)
    df["Lat"] = np.where(
        df["Side_pose"] == "Bilat", 
        "Bilat",
        np.where(
            df["Side_pose"] == df["Side"].map(tuple_1st), 
            "Ipsi", "Contra"))
    
    mask_dual = df["Paradigm"].astype("string").str.lower().isin({x.lower() for x in is_dual_list})
    df["Task"] = "Single"
    df.loc[mask_dual, "Task"] = "Dual"
    
    df['Pose'] = df['Pose'].str.split('_').str[0]
    df = standardize_df(df)
    df = slice_df(df, slices_default)
        
    return df


normalize_func_dict = {
    'turn_feature': normalize_pose,
    'default': standardize_df,
}
