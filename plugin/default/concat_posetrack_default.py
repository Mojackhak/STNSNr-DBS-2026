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
from src.tabular.reshape import shiftTFR
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
    'gait_feature':(
        'gait', 'gait_dual', 'gait_feature', 'gait_dual_feature'), 
    }

time_axis = {
    'default': 'percent',
}

slices_default = {'Region': ['STN', 'SNr', ("SNr", "STN"),]}

def flip_side(side):
    if side == 'Left':
        return 'Right'
    elif side == 'Right':
        return 'Left'
    else:
        return side

def normalize_gait(df: pd.DataFrame) -> pd.DataFrame:
    df["Side_cycle"] = (
        df["Epoch"]
        .str[0]
        .map(normalize_side)
    )
    df["Epoch"] = df["Epoch"].str[1].astype(int)
    df["Lat"] = np.where(
        df["Side_cycle"] == df["Side"].map(tuple_1st), 
        "Ipsi", "Contra")
    mask = df["Lat"] == "Ipsi"
    df.loc[mask, "Value"] = df.loc[mask, "Value"].map(shiftTFR)
    df.loc[mask, "Side_cycle"] = df.loc[mask, "Side_cycle"].map(flip_side)
    df["Lat"] = "Contra"
    mask_single = df["Paradigm"].astype("string").str.lower().isin({"gait", "gait_feature"})
    df["Task"] = np.where(mask_single, "Single", "Dual")
    df = standardize_df(df)
    df = slice_df(df, slices_default)
    
    return df

normalize_func_dict = {
    'gait_feature': normalize_gait,
    'default': standardize_df,
}
