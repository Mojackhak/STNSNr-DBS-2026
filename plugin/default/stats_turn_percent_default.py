"""
# Necessary parameters for stats/stats pipeline
1. sections
2. normalize_df_category_cfg
3. to_extra_format_cfg
4. slices_cfg
5. cols_basic
6. cols_coords
7. cols_filter_cfg
8. filter_outliers_cfg
9. normalize_df_func
10. summarize_df_cfg
11. cols_contact_agg_cfg
12. cols_region_agg_cfg
13. cols_subject_agg_cfg (if used)
14. split_features_cfg (if used) # split paired connectivity features
15. split_col_bak (if used)
16. split_cfg (if used)

"""

import pandas as pd
from plugin.default.tensors_default import bands
from plugin.default.tabular_default import cols_basic, cols_coords, df_category, standardize_df
import plugin.default.derive_default as cfg
from src.utils.config import get_param
from src.tabular.text import tuple_1st, join_sequence
from src.stats.preproc.normalize import normalize_df, normalize_df_by_baseline
from typing import Sequence, Any


contain = 'turn_feature_merged'

cols_basic = cols_basic + ['Feature',]

sections = {
    'turn_feature_merged': ['turn_feature_percent',],
}

slices_cfg = {
    'turn_feature_merged': {
        'turn_feature_percent': {'Task': ['Single',],},
    },
    'default': {
        'default': None,
    }
}


filter_outliers_cfg = {
    'value_col': ['Value', 'Value_feature', ],
    'mode': 'mcd',
    'alpha': 0.99,               
    'support_fraction': 0.95,
    'pca': True, 'pca_max_components': 20, 'pca_explained_var': 0.99,
    'min_n': 8, 'min_n_over_p': 1.2,
    'return_scores': False,
    'random_state': 42,
}


cols_filter_cfg = {
    'default': {
        'default': ['ID', 'Contact', 'Epoch',],
    }
}


normalize_df_cfg = {
    'value_col': 'Value',
    'mode': 'percent',
    'mode_baseline': 'mean',
    'out_col': None,
    'eps': 1e-120,
    'on_missing_baseline': 'drop',
    'drop_empty': True,
    'align': 'force',
}

normalize_df_baseline_cfg = {
    'turn_feature_merged': {
        'turn_feature': {"Pose": "Stand"},
    },
    'fog_episode': {
        'fog_episode': {"Phase": "BL"},
    },
    'pose': {
        'rest': None,
        'med': {"Med": "Off"},
        'walk': {"Pose": "Sit"},
        'turn': {"Lat": "Ipsi"},
        'fog': {"Pose": "Stand"},
    },
    'default': {
        'default': None,
    },
}

baseline_normalize_cfg = {
    'mode_baseline': 'percent',
    'mode': 'mean',
    'eps': 1e-120,
    'slice_mode': 'percent'
}

baseline_normalize_baseline_cfg = {
    'default': {
        'default': None,
    },
}

normalize_func_selector = {
    'raw': 'normalize_df_by_baseline',
    'spectral': 'normalize_df',
    'trace': 'normalize_df_by_baseline',
    'scalar': 'normalize_df',
    'default': 'normalize_df',
}

normalize_baselize_selector = {
    'raw': baseline_normalize_baseline_cfg,
    'spectral': normalize_df_baseline_cfg,
    'trace': baseline_normalize_baseline_cfg,
    'scalar': normalize_df_baseline_cfg,
    'default': normalize_df_baseline_cfg,
}

cols_extra_normalize_cfg = {
    'default': {
        'default': [],
    }
}


def normalize_df_func(df: pd.DataFrame, paradigm: str, section: str, feature: str) -> pd.DataFrame:
    
    norm_base = normalize_baselize_selector.get(
        feature, normalize_baselize_selector['default'])
    
    baseline = get_param(norm_base, paradigm, section)
    
    if baseline is None:
        return df.copy()
    
    norm_fun = normalize_func_selector.get(
        feature, normalize_func_selector['default'])
    
    if norm_fun == 'normalize_df_by_baseline':
        df_normalized = normalize_df_by_baseline(
            df,
            baseline=baseline,
            **baseline_normalize_cfg
        )
    elif norm_fun == 'normalize_df':
        cols_baseline = list(baseline.keys()) if baseline is not None else []
        cols_extra = get_param(cols_extra_normalize_cfg, paradigm, section) 
        cols_norm = [col for col in df.columns if (col in cols_basic) \
                and (col not in cols_baseline + cols_extra)]
        print(f"Normalizing columns: {cols_baseline} gourp by {cols_norm}")
        
        df_normalized = normalize_df(
            df,
            group_cols=cols_norm,
            baseline=baseline,
            **normalize_df_cfg
        )
        if cols_baseline:
            df_normalized = df_normalized[~df_normalized[cols_baseline].eq(pd.Series(baseline)).all(axis=1)] # drop baseline rows
        df_normalized.reset_index(drop=True, inplace=True)   
    else:
        raise ValueError(f"Unknown normalization function: {norm_fun}")
           
    return df_normalized


cols_contact_agg_cfg = {
    'default': {
        'default': []
    }
}

cols_region_agg_cfg = {
    'default': {
        'default': ['Contact', ]
    }
}

cols_subject_agg_cfg = {
    'default': {
        'default': ['Contact', 'Side', 
                    'Side_pose', 'Side_cycle', 'Side_stim', 'Hemi',
                    'Lat', 'Loc_lat', 'Force', 'LogForce',]
    }
}

summarize_df_cfg = {
    'value_col': ['Value', 'Value_feature', ],
    'mode': 'mean',
    'drop_empty': True,
    'align': 'force',
}


split_cfg = {
    'cols_to_split': ['Contact', 'Region', *cols_coords],
    'strict': False,
}

split_col_bak = {'Region': 'Region_bak',}

split_features_cfg = {
    'raw': [],
    'spectral': [],
    'trace': [],
    'scalar': ['conn', 'gc', 'psi'],
    'default': ['conn', 'gc', 'psi'],
    }

normalize_df_category_cfg = {
    'df_category': df_category,
    'ordered': True,
    'mode': 'append',
    'dropna': True,
    'inplace': False,
}

to_extra_format_cfg = {
    'raw': False,
    'spectral': False,
    'trace': False,
    'scalar': True,
    'default': True,
}

def standardize_df_stats(df: pd.DataFrame) -> pd.DataFrame:
    
    tuple_1st_cols = ['Side', 'Side_pose', 'Side_cycle', 'Side_stim', 'Hemi',
                      'Lat', 'Loc_lat',]
    join_sequence_cols = ['Contact', 'Region', 'Region_bak',]
    
    for cols in df.columns:
        if cols in tuple_1st_cols:
            df[cols] = df[cols].map(tuple_1st)
        if cols in join_sequence_cols:
            df[cols] = df[cols].map(lambda x: join_sequence(x, sep='-'))
    
    df = standardize_df(df)    
    
    return df