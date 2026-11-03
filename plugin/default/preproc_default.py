"""
# Necessary parameters for preproc/preproc pipeline
1. events
2. notches_base_hz
3. sfreq_base_hz
4. bandpass_cfg
5. notch_cfg
6. bad_annot_cfg
7. bad_annot_filter_cfg
8. add_annot_cfg
9. add_end_annot_cfg
10. ecg_remove_cfg (if used)
11. montage (if used)

"""

import numpy as np

montage = {
    'anode':   ['0_7', '4_7', '8_15', '12_15',],
    'cathode': ['1_7', '5_7', '9_15', '13_15',],
    'ch_name': ['0_1', '4_5',  '8_9', '12_13',],
}

# montage = {
#     'anode':   ['2_0', '6_0', '10_8', '13_8',],
#     'cathode': ['3_0', '7_0', '11_8', '14_8',],
#     'ch_name': ['2_3', '6_7',  '10_11', '13_14',],
# }

events = {"pain": ['L26', 'R26', 
                   'L60', 'R60', 
                   'L100', 'R100', 
                   'L180', 'R180', 
                   'L300', 'R300'],
          "gait": ['sit', 'stand', 'walk', 'turn_L', 'turn_R', 'fog',
                   'strike_L', 'strike_R', 'off_L', 'off_R'],
          "gait_dual": ['sit_dual', 'stand_dual', 'walk_dual', 'turn_dual_L', 'turn_dual_R', 'fog_dual',
                        'strike_L', 'strike_R', 'off_L', 'off_R'],
          }

notches_base_hz = np.array([41.75, 83.5, 125.25, 167.0], dtype=float)
sfreq_base_hz = 418.4

bandpass_cfg = {
    "l_freq": 1.0,
    "h_freq": 200.0,
}

notch_cfg = {
    "notches": notches_base_hz,
    "notch_widths": 1.0,
}

bad_annot_cfg = {
    "epoch_dur": 1.0,
    "overlap": 0.0,
    "p2p_thresh": (1e-6, 1e-2),
    "autoreject_correct_factor": 1.5,
    "min_good_len_sec": 3.0,
    "merge_gap_sec": 0.5,
}

bad_annot_filter_cfg = {
    "concat_duration": 0, 
}

add_annot_cfg = {
    "avoid_buffer_sec": 0.0, 
}

add_end_annot_cfg = {
    "head_duration_sec": 0, 
    "tail_duration_sec": 0,
    "description": "EDGE_concat",
}

ecg_remove_cfg = {
    'method': 'svd',
    "components": 2,
    "window_ms": 150,  
    "peak_height_range": (2.0, 10.0),
    "min_interpeak_ms": 500, 
    "pre_ms": 200,
    "post_ms": 200,
    "qrs_duration_ms": 80,  
    "pqrst": False,
    "force_orientation": None,
    "return_figure": True,
}