"""
# Necessary parameters for lfp/tensors pipeline
1. morlet_cfg
2. bands
3. k_sigma
4. dynamic_mask_cfg
5. burst_qc_cfg
6. mask_cfg
7. n_samples_warp
8. pain_warp_cfg (if used)
9. fog_warp_cfg (if used)
10. gait_warp_cfg (if used)
11. concat_warp_cfg (if used)
12. warp_jobs
13. build_warper

"""

import numpy as np
from plugin.default.tensors_default import grid_base_cfg, bands
from src.lfp.warp.event_anchored_linear import gait_warper
from src.lfp.warp.stack import stack_warper
from src.lfp.warp.pad_concat import pad_warper
from src.lfp.warp.concat import concat_warper

morlet_cfg = {
    'fwhm_time_s': grid_base_cfg['time_resolution_s'],
    'min_cycles': grid_base_cfg['min_cycles'],
    'max_cycles': grid_base_cfg['max_cycles'],
}
    
burst_qc_cfg = {
    "metric": "masked_away_fraction_of_burst_bins",
    "threshold": 0.2,
    "min_burst_bins": 10,
    "channels": None,          # all channels
    "channel_reduce": "max",     
}


mask_cfg = {
    "mode": "exact",
    "pad_s": 0.0,
    "clip_to_raw": True,
}

drop_cfg = {
    "drop": ["bad", "edge",],
    "mode": "substring",
    "pad_s": 0.0,
    "clip_to_raw": True,
}

k_sigma = 5.0  # wavelets of mne extend to ±5 standard deviations

dynamic_mask_cfg = {
    "drop": ["bad", "edge"],
    "mode": "substring",
    "clip_to_raw": True,
}

n_samples_warp = {
    'pain': 1001,
    'fog_episode': 301,
    'gait': 101,
    'gait_dual': 101,
    'gait_feature': 101,
    'gait_dual_feature': 101,
    'turn_L_feature': 501,
    'turn_R_feature': 501,
    'turn_dual_L_feature': 501,
    'turn_dual_R_feature': 501,
}

pain_labels = [
    "L26", "R26",
    "L60", "R60",
    "L100", "R100",
    "L180", "R180",
    "L300", "R300",
]

pain_pad_list = (4.0, 5.0, 5.0, 6.0)
pain_allowed = {lbl: pain_pad_list for lbl in pain_labels}
pain_warp_cfg = {
    "anno_allowed": pain_allowed,
    "duration_range": (7.0, np.inf), # (half of duration of SUS) + (other stage duration)
    "anno_drop": ["edge", "bad",],
}

fog_pad_list = (2.2, 0.8, 0.0, 0.0)
fog_allowed = {lbl: fog_pad_list for lbl in ['fog',]}
fog_warp_cfg = {
    "anno_allowed": fog_allowed,
    "duration_range": (0.8, np.inf), # (half of duration of SUS) + (other stage duration)
    "anno_drop": ["edge", "bad",],
}


gait_warp_cfg = {
    "cycle_duration_range": (0.6, 2.0),
    "anchors_percent": (0.0, 18.0, 50.0, 68.0, 100.0),
    "percent_tolerance": 10.0,
    "anno_allowed": ["gait", "gait_dual", "tag", "walk",],
}

gait_feature_warp_cfg = {
    "cycle_duration_range": (0.6, 2.0),
    "anchors_percent": (0.0, 18.0, 50.0, 68.0, 100.0),
    "linear_warp": False,    
    "percent_tolerance": 10.0,
    "anno_allowed": ["gait", "gait_dual", "tag", "walk",],
}

concat_warp_cfg = {
    "mode": "exact",
    "pad_s": 0.0,
    "clip_to_raw": True,
}

turn_feature_warp_cfg = {
    "mode": "substring",
    "duration_range": (1.0, 10.0),
}


def build_warper(job):
    warper_type = str(job["warper_type"]).strip().lower()
    warper_cfg = globals()[job["warper_cfg_key"]]

    if warper_type == "pad":
        return lambda raw: pad_warper(raw, **warper_cfg)
    if warper_type == "gait":
        return lambda raw: gait_warper(raw, **warper_cfg)
    if warper_type == "concat":
        keep = job["keep"]
        keep = [] if keep is None else list(keep)
        return lambda raw: concat_warper(raw, keep=keep, **warper_cfg)
    if warper_type == "stack":
        keep = job["keep"]
        keep = [] if keep is None else list(keep)
        return lambda raw: stack_warper(raw, keep=keep, **warper_cfg)

    raise ValueError(f"Unsupported warper_type: {warper_type}")


warp_jobs = [
    {
        "anno": "pain",
        "warper_type": "pad",
        "warper_cfg_key": "pain_warp_cfg",
        "keep": None,
        "overlap_policy": "drop",
        "save_name": "pain",
        "overwrite": True,
    },
    {
        "anno": "gait",
        "warper_type": "pad",
        "warper_cfg_key": "fog_warp_cfg",
        "keep": None,
        "overlap_policy": "drop",
        "save_name": "fog_episode",
        "overwrite": True,
    },
    {
        "anno": "gait",
        "warper_type": "gait",
        "warper_cfg_key": "gait_warp_cfg",
        "keep": None,
        "overlap_policy": "split",
        "save_name": "gait",
        "overwrite": True,
    },
    {
        "anno": "gait_dual",
        "warper_type": "gait",
        "warper_cfg_key": "gait_warp_cfg",
        "keep": None,
        "overlap_policy": "split",
        "save_name": "gait_dual",
        "overwrite": True,
    },
    {
        "anno": "gait",
        "warper_type": "gait",
        "warper_cfg_key": "gait_feature_warp_cfg",
        "keep": None,
        "overlap_policy": "split",
        "save_name": "gait_feature",
        "overwrite": True,
    },
    {
        "anno": "gait_dual",
        "warper_type": "gait",
        "warper_cfg_key": "gait_feature_warp_cfg",
        "keep": None,
        "overlap_policy": "split",
        "save_name": "gait_dual_feature",
        "overwrite": True,
    },
    {
        "anno": "sit",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["sit"],
        "overlap_policy": "split",
        "save_name": "sit",
        "overwrite": True,
    },
    {
        "anno": "stand",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["stand"],
        "overlap_policy": "split",
        "save_name": "stand",
        "overwrite": True,
    },
    {
        "anno": "walk",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["walk"],
        "overlap_policy": "split",
        "save_name": "walk",
        "overwrite": True,
    },
    {
        "anno": "turn_L",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["turn_L"],
        "overlap_policy": "split",
        "save_name": "turn_L",
        "overwrite": True,
    },
    {
        "anno": "turn_R",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["turn_R"],
        "overlap_policy": "split",
        "save_name": "turn_R",
        "overwrite": True,
    },
    {
        "anno": "fog",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["fog"],
        "overlap_policy": "split",
        "save_name": "fog",
        "overwrite": True,
    },
    {
        "anno": "sit_dual",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["sit_dual"],
        "overlap_policy": "split",
        "save_name": "sit_dual",
        "overwrite": True,
    },
    {
        "anno": "stand_dual",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["stand_dual"],
        "overlap_policy": "split",
        "save_name": "stand_dual",
        "overwrite": True,
    },
    {
        "anno": "walk_dual",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["walk_dual"],
        "overlap_policy": "split",
        "save_name": "walk_dual",
        "overwrite": True,
    },
    {
        "anno": "turn_dual_L",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["turn_dual_L"],
        "overlap_policy": "split",
        "save_name": "turn_dual_L",
        "overwrite": True,
    },
    {
        "anno": "turn_dual_R",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["turn_dual_R"],
        "overlap_policy": "split",
        "save_name": "turn_dual_R",
        "overwrite": True,
    },
    {
        "anno": "fog_dual",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["fog_dual"],
        "overlap_policy": "split",
        "save_name": "fog_dual",
        "overwrite": True,
    },
    {
        "anno": "sit_medon",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["sit_medon"],
        "overlap_policy": "split",
        "save_name": "sit_medon",
        "overwrite": True,
    },
    {
        "anno": "stand_medon",
        "warper_type": "concat",
        "warper_cfg_key": "concat_warp_cfg",
        "keep": ["stand_medon"],
        "overlap_policy": "split",
        "save_name": "stand_medon",
        "overwrite": True,
    },
    {
        "anno": "turn_L",
        "warper_type": "stack",
        "warper_cfg_key": "turn_feature_warp_cfg",
        "keep": ["turn_L"],
        "overlap_policy": "drop",
        "save_name": "turn_L_feature",
        "overwrite": True,
    },
    {
        "anno": "turn_R",
        "warper_type": "stack",
        "warper_cfg_key": "turn_feature_warp_cfg",
        "keep": ["turn_R"],
        "overlap_policy": "drop",
        "save_name": "turn_R_feature",
        "overwrite": True,
    },
    {
        "anno": "turn_dual_L",
        "warper_type": "stack",
        "warper_cfg_key": "turn_feature_warp_cfg",
        "keep": ["turn_dual_L"],
        "overlap_policy": "drop",
        "save_name": "turn_dual_L_feature",
        "overwrite": True,
    },
    {
        "anno": "turn_dual_R",
        "warper_type": "stack",
        "warper_cfg_key": "turn_feature_warp_cfg",
        "keep": ["turn_dual_R"],
        "overlap_policy": "drop",
        "save_name": "turn_dual_R_feature",
        "overwrite": True,
    },
]
