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
from src.tabular.grid import GridResultColumns, abs_to_percent_intervals
from typing import Any
from src.utils.config import get_param

contain = None

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


pain_bilat = {
    "BL" : [[0.0, 2.5]],
    "ONS": [[3.5, 5.5]],
    "SUS": [[6.5, 12.5]],
    "OFS": [[13.5, 15.5]],
    "PST": [[16.5, 20.0]],
}

pain_bilat_percent = abs_to_percent_intervals(
    pain_bilat,
    axis_start=0.0,
    axis_end=20.0,
)

pain_times = {
    "Contra": # signal side 
        {"Contra": # "limb side"
            pain_bilat_percent,
        },
    "Ipsi":
        {"Ipsi":
            pain_bilat_percent,
        }
}


cycle_anchor = [0, 18, 50, 68, 100] # HS_ipsi, TO_ipsi, HS_contra, TO_contra, HS_ipsi
tolerance = 5
gait_ipsi = {
    "TO": [[cycle_anchor[3]-tolerance, cycle_anchor[3]+tolerance]],
    "HS": [[cycle_anchor[0], cycle_anchor[0]+tolerance],
           [cycle_anchor[4]-tolerance, cycle_anchor[4]]],
    "Swing": [[cycle_anchor[3], cycle_anchor[4]]],
    "Stance": [[cycle_anchor[0], cycle_anchor[3]]],
    # "All": [[cycle_anchor[0], cycle_anchor[4]]]
}

gait_contra = {
    "TO": [[cycle_anchor[1]-tolerance, cycle_anchor[1]+tolerance]],
    "HS": [[cycle_anchor[2]-tolerance, cycle_anchor[2]+tolerance]],
    "Swing": [[cycle_anchor[1], cycle_anchor[2]]],
    "Stance": [[cycle_anchor[0], cycle_anchor[1]],
               [cycle_anchor[2], cycle_anchor[4]]],
    # "All": [[cycle_anchor[0], cycle_anchor[4]]]
}



gait_times = {
    "Contra": # signal side 
        {"Contra": # "limb side"
            gait_ipsi,
        "Ipsi":
            gait_contra,
        },
    "Ipsi":
        {"Contra":
            gait_contra,
        "Ipsi":
            gait_ipsi,
        }
}

fog_bilat = {
    "BL" : [[0.0, 1.0]],
    "Pre": [[1.0, 2.0]],
    "ONS": [[2.0, 3.0]],
    # "OFS": [[4.5, 5.5]],
    # "PST": [[6.0, 7.0]],
}

fog_bilat_percent = abs_to_percent_intervals(
    fog_bilat,
    axis_start=0.0,
    axis_end=3.0,
)

fog_times = {
    "Bilat": # signal side 
        {"Bilat": # "limb side"
            fog_bilat_percent,
        },
}


pose_bilat_percent = {
    "All": [[0.0, 100.0]],
}

pose_time = {
    "Bilat": # signal side 
        {"Bilat": # "limb side"
            pose_bilat_percent,
        },
    "Contra":
        {"Contra":
            pose_bilat_percent,
        },
    "Ipsi":
        {"Ipsi":
            pose_bilat_percent,
        }
}


def custom_split_times_fn(key):
    if key == 'pain':
        
        return pain_bilat
    if key in gait_times:
        return gait_times[key]
    elif key in fog_times:
        return fog_times[key]
    else:
        raise ValueError(f"Unknown annotation key for time splits: {key!r}")


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
    'pain': pain_times,
    'gait': gait_times,
    'fog_episode': fog_times,
    'pose': pose_time,
    'default': None,
}

# only save derivated dataframe if paradigm/param supports derivation or 
# it is needed in the downstream analysis

derive_paradigm_cfg = { # does this paradigm need derivation?
    'pose': {
        'raw': False,
        'spectral': True,
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
