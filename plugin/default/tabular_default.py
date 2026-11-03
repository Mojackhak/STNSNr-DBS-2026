"""
# Necessary parameters for tabular standardization
1. cols_coords
2. cols_basic
3. cols_order
4. df_category
5. standardize_df()
6. transform_mode_cfg

"""

import pandas as pd
from src.tabular.text import capitalize_df
from typing import Sequence

cols_coords = [
    'MNI_x',
    'MNI_y',
    'MNI_z',
    'SNr_in',
    'SNr_x',
    'SNr_y',
    'SNr_z',
    'STN_in',
    'STN_x',
    'STN_y',
    'STN_z',
    'MNI_x_flip',
    'MNI_y_flip',
    'MNI_z_flip',    
]

cols_basic = [
    'ID',
    'Subject',
    'Contact',
    'Region',
    'Region_bak',
    'Side',
    'Side_pose',
    'Side_cycle',
    'Side_stim',
    'Hemi',
    'Lat',
    'Loc_lat',
    'Pose',
    'Med',
    'Task',
    'Force',
    'LogForce',
    'Polarity',
    'Phase',
    'Band',    
    'Epoch',
    'Label',
    'Scale',
    'Param',
    'Paradigm',
]


cols_order = [
    *cols_basic,
    *cols_coords,
    'Value',
]

region = ['SNr', 'STN', ]
region_pair_str = [f'{a}-{b}' for a in region for b in region]
region_pair_tuple = [(a, b) for a in region for b in region]
regions_all = region + region_pair_str + region_pair_tuple

side = ['Left', 'Right', 'Bilat', ]
side_pair_str = [f'{a}-{b}' for a in side for b in side]
side_pair_tuple = [(a, b) for a in side for b in side]
sides_all = side + side_pair_str + side_pair_tuple

hemi = ['LAH', 'MAH', 'Bilat', ]
hemi_pair_str = [f'{a}-{b}' for a in hemi for b in hemi]
hemi_pair_tuple = [(a, b) for a in hemi for b in hemi]
hemis_all = hemi + hemi_pair_str + hemi_pair_tuple

lat = ['Ipsi', 'Contra', 'Bilat', ]
lat_pair_str = [f'{a}-{b}' for a in lat for b in lat]
lat_pair_tuple = [(a, b) for a in lat for b in lat]
lats_all = lat + lat_pair_str + lat_pair_tuple


df_category = {
    'Region': regions_all,
    'Side': sides_all,
    'Side_pose': sides_all,
    'Side_cycle': sides_all,
    'Side_stim': sides_all,
    'Hemi': hemis_all,
    'Lat': lats_all,
    'Loc_lat': lats_all,
    'Pose': ['Sit', 'Stand', 'Walk', 'Turn', 'Gait', 'FoG'],
    'Med': ['Off', 'On'],
    'Task': ['Single', 'Dual'],
    'Polarity': ['Anodal', 'Cathodal', 'Bipolar'],
    'Phase': ['BL', 'Pre', 'ONS', 'SUS', 'OFS', 'PST', 'HS', 'TO', 'Stance', 'Swing', 'All'],
    'Band': ['Delta', 'Theta', 'Alpha', 'Beta', 'Beta_low', 'Beta_high', 'Gamma', 
             'Exponent', 'Offset', 'All'],
}

def standardize_df(df: pd.DataFrame, cols_order: Sequence[str] = cols_order) -> pd.DataFrame:
    df = capitalize_df(df)
    cols_order = [col for col in cols_order if col in df.columns]
    cols_order += [col for col in df.columns if col not in cols_order]
    df = df[cols_order]
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    return df