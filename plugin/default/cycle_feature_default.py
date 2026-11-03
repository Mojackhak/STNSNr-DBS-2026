from turtle import pd


fps = 25

resample_cfg = {
    'fps': fps,
    'method': 'akima',
}

pose_sync_cfg = {
    'bad_descs': ("BAD",),
    'overlap_policy': "split",
    'match_mode': "substring",
    'case_sensitive': False,
    'reset_index': True,
    'return_report': True,
}

gait_feature_warp_cfg = {
    "cycle_duration_range": (0.6, 2.0),
    "anchors_percent": (0.0, 18.0, 50.0, 68.0, 100.0),
    "linear_warp": False,    
    "percent_tolerance": 10.0,
    "anno_allowed": ["gait", "gait_dual", "tag", "walk",],
}

gait_features = [
    "stride_length",
    "cycle_duration",
    "walk_speed",]