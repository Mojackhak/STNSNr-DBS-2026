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
    'pain':('pain',), 
    'gait':('gait', 'gait_dual'), 
    'fog_episode':('fog_episode',),
    'pose':
    ('sit', 'stand', 'walk', 'turn_L', 'turn_R', 'fog',
     'sit_dual', 'stand_dual', 'walk_dual', 
     'turn_dual_L', 'turn_dual_R', 'fog_dual',
     'sit_medon', 'stand_medon', 'walk_medon',
     'turn_medon_L', 'turn_medon_R', 'fog_medon'),
    }

time_axis = {
    'gait': 'percent',
    'gait_dual': 'percent',
    'default': 'time',
}

slices_default = {'Region': ['STN', 'SNr', ("SNr", "STN"),]}

def flip_side(side):
    if side == 'Left':
        return 'Right'
    elif side == 'Right':
        return 'Left'
    else:
        return side


def normalize_pain(df: pd.DataFrame) -> pd.DataFrame:
    df[["Side_stim", "Force"]] = (
        df["Epoch"]
        .str[0]
        .str.extract(r"^(L|R)(\d+)$")
    )
    df["Side_stim"] = df["Side_stim"].map(normalize_side)
    df["Force"] = df["Force"].astype(int)
    df["LogForce"] = np.log10(df["Force"])
    df["Lat"] = np.where(
        df["Side_stim"] == df["Side"].map(tuple_1st), 
        "Ipsi", "Contra")
    df["Epoch"] = df["Epoch"].str[1].astype(int)
    df.drop(columns=['Paradigm'], inplace=True)
    df = standardize_df(df)
    df = slice_df(df, slices_default)
    
    return df

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
    df["Task"] = np.where(
        df["Paradigm"] == "Gait", 
        "Single", "Dual")
    df.drop(columns=['Paradigm'], inplace=True)
    df = standardize_df(df)
    df = slice_df(df, slices_default)
    
    return df

def normalize_fog_episode(df: pd.DataFrame) -> pd.DataFrame:
    df["Epoch"] = df["Epoch"].str[1].astype(int)
    df["Lat"] = "Bilat"
    df.drop(columns=['Paradigm'], inplace=True)
    df = standardize_df(df)
    df = slice_df(df, slices_default)

    return df
    
is_direction_list = ['turn_L', 'turn_R', 'turn_dual_L', 'turn_dual_R',
                     'turn_medon_L', 'turn_medon_R']

is_medon_list = ['sit_medon', 'stand_medon', 'walk_medon',
                 'turn_medon_L', 'turn_medon_R', 'fog_medon']

is_dual_list = ['sit_dual', 'stand_dual', 'walk_dual',
                'turn_dual_L', 'turn_dual_R', 'fog_dual']

def normalize_pose(df: pd.DataFrame) -> pd.DataFrame:
    df["Epoch"] = df["Epoch"].str[1].astype(int)
    
    mask_lat = df["Paradigm"].astype("string").str.lower().isin({x.lower() for x in is_direction_list})
    df["Pose"] = df["Paradigm"].astype("string")
    df["Side_pose"] = "Bilat"
    df.loc[mask_lat, ["Pose", "Side_pose"]] = ( # pose without side, side
        df.loc[mask_lat, "Paradigm"]
        .astype("string")
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
    
    mask_med = df["Paradigm"].astype("string").str.lower().isin({x.lower() for x in is_medon_list})
    df["Med"] = "Off"
    df.loc[mask_med, "Med"] = "On"
    
    mask_dual = df["Paradigm"].astype("string").str.lower().isin({x.lower() for x in is_dual_list})
    df["Task"] = "Single"
    df.loc[mask_dual, "Task"] = "Dual"
    
    df['Pose'] = df['Pose'].str.split('_').str[0]
    df['Pose'] = df['Pose'].map(normalize_fog)
    df.drop(columns=['Paradigm'], inplace=True)
    df = standardize_df(df)
    df = slice_df(df, slices_default)
        
    return df


normalize_func_dict = {
    'pain': normalize_pain,
    'gait': normalize_gait,
    'fog_episode': normalize_fog_episode,
    'pose': normalize_pose,
    'default': standardize_df,
}
