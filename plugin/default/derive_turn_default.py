"""
# Necessary parameters for tabular/derive pipeline
1. freqs_cfg
2. time_cfg
3. derive_paradigm_cfg
4. pre_transform_mode_cfg
5. is_derivable()
6. collapse_base_cfg
7. reducer_cfg
8. post_transform_mode_cfg

"""

import pandas as pd
from pathlib import Path
from plugin.default.tensors_default import bands
from src.tabular.grid import GridResultColumns
from typing import Any
from src.utils.config import get_param

contain = 'turn_feature'

transform_func_cfg = {
    'value_col': 'Value',
    'out_col': None,
    'eps': 1e-120,
    'drop_empty': True,
}

pre_transform_mode_cfg = {
    'tfr': {
        'raw': 'dB',
        'aperiod': 'dB',
        'period': 'dB',
    },
    'burst': {
        'burst': 'log',
    },
    'default': {
        'default': 'none',
    }
}

post_transform_mode_cfg = {
    'burst': { # subparam
        'rate': 'asinh', # reducer
        'duration': 'log',
        'occupation': 'none',
        'mean': 'none',
        'default': 'none',
    },
    'coh': {
        'default': 'fisherz_sqrt',
    },
    'ciplv': {
        'default': 'logit',
    },    
    'wpli': {
        'default': 'logit',
    },    
    'plv': {
        'default': 'logit',
    },    
    'pli': {
        'default': 'logit',
    },
    'default': {
        'default': 'none',
    },    
}
    

bands = {k: [v] for k, v in bands.items()}


turn_feature = {
    "All": [[0, 100]]
}


turn_feature_times = {
    "Contra": # signal side 
        {"Contra": # "limb side"
            turn_feature,
        },
    "Ipsi":
        {"Ipsi":
            turn_feature,
        },
}


collapse_base_cfg = {
    "value_col": "Value",
    "lat_col": "Lat",
    "time_interval_mode": "percent",
    "freq_interval_mode": "absolute",
    "phase_order": None,
    "lat_limb_order": None,
    "inclusive": False,
    "drop_value": True,
    "keep_full_dim_cols": False,
    "on_missing": "skip",
    "out_cols": GridResultColumns(
        band="Band", phase="Phase", lat_limb="Lat", value="Value")
}

reducer_cfg = {
    'burst': {
        'burst': ['mean', 'occupation', 'rate', 'duration'],
    },
    'default': {
        'default': ['mean',],
    }
}
    
bands_freqs = [[b,] for b in bands.keys()]
params_freqs = [['offset',], ['exponent',],]

freqs_cfg = {
    'params': params_freqs,
    'psi': bands_freqs,
    'burst': bands_freqs,
    'default': bands,
}

time_cfg = {
    'turn_feature': turn_feature_times,
    'default': None,
}

# only save derivated dataframe if paradigm/param supports derivation or 
# it is needed in the downstream analysis

derive_paradigm_cfg = { # does this paradigm need derivation?
    'turn_feature': {
        'raw': False,
        'spectral': False,
        'trace': False,
        'scalar': True,
    },
    'default': {
        'raw': True,
        'spectral': True,
        'trace': True,
        'scalar': True,
    },
}

derive_param_cfg = { # does this param support derivation?
    'params': {
        'raw': False,
        'spectral': False,
        'trace': True,
        'scalar': True,
    },                    
    'psi': {
        'raw': False,
        'spectral': False,
        'trace': True,
        'scalar': True,
    },
    'burst': {
        'raw': False,
        'spectral': False,
        'trace': False,
        'scalar': True,
    },
    'default': {
        'raw': True,
        'spectral': True,
        'trace': True,
        'scalar': True,
    },
}


def is_derivable(paradigm: str, param: str, derive: str) -> bool:
    derive_paradigm = get_param(derive_paradigm_cfg, paradigm, derive)
    
    derive_param = get_param(derive_param_cfg, param, derive)
    
    to_derive = derive_paradigm and derive_param
    
    return to_derive
