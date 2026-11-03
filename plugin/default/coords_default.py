"""
# Necessary parameters for tabular/concat pipeline
1. label_slice_cfg
2. properties_cfg
3. zscore_cols

"""

import numpy as np
import pandas as pd
from pathlib import Path
from plugin.default.stats_default import filter_outliers_cfg
from plugin.default.tabular_default import cols_order
from plugin.default.derive_default import transform_func_cfg
from plugin.default.stats_default import summarize_df_cfg
from typing import Any


label_slice_cfg = {'label': ["GABA"]}

properties_cfg = [
    'ID', 'subject', 'side', 'depth', 'offset', 'label', 
    'burst_index', 'LvR', 'firing_rate', 'peak_to_valley', 'peak_trough_ratio',
    'half_width', 'repolarization_slope', 'recovery_slope',]

zscore_cols = [
    'peak_to_valley', 'peak_trough_ratio',
    'repolarization_slope', 'recovery_slope',]

derive_slice_cfg = {'Region': ['SNr',],
                    'Hemi': ['LAH', 'MAH'],}

transform_mode_cfg = {
    'burst_index': 'log',
    'LvR': 'none',
    'peak_to_valley': 'log',
    'peak_trough_ratio': 'log',
    'half_width': 'log',
    'repolarization_slope': 'log',
    'recovery_slope': 'log',
    'firing_rate': 'asinh',
} 