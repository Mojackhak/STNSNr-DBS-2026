"""
Interactive step-by-step pipeline for LFP time-frequency features on a stitched MNE Raw.

This script is intentionally written as a sequence of IPython/VSCode "#%%" cells to
run incrementally. Workflow:
1) Load stitched Raw (.fif) and tensors
2) Warp paradigm-specific tensors and save to stitch/warped/{paradigm}/
"""

#%% 0) Imports & interactive setup (paths, configs, matplotlib backend)
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path

import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
    
import src.utils.paths as paths
import src.utils.fileops as fileops
import plugin.default.warp_default as cfg
from src.lfp.pipelines.masking import get_custom_warped_tensors
from src.io.pkl_io import load_pkl


import importlib
importlib.reload(cfg)

#%% 1) Define project paths and dataset identifiers (subject/paradigm/sync settings)
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

deri = Path(r"derivatives/mne")  # Lead-DBS derivatives folder (BIDS-style)
deri = project / deri

subject = os.environ["MYLFP_SUBJECT"]  # must start with "sub-"
wd = deri / subject

stitch_dir = wd / "stitch"
preproc_dir = stitch_dir / "preproc"
tensor_dir = stitch_dir / "tensor"

fifpath = preproc_dir / f"{subject}_stitched_bi_raw.fif"
print("Reading:", fifpath)

tensor_dir = fileops.create_subfolder(stitch_dir, "tensor")
tensor_pkl = tensor_dir / "tensors_raw.pkl"
print("tensor_pkl:", tensor_pkl)

#%% 2.1) Load stitched Raw and tensors
raw = mne.io.read_raw_fif(fifpath, preload=True)
print(raw)
print("sfreq (Hz):", raw.info["sfreq"])
print("n_times:", raw.n_times)
print("Duration (s):", raw.times[-1] - raw.times[0])
print("n_annotations:", len(raw.annotations))

tensors = load_pkl(tensor_pkl) # load original tensors
warped_dir = fileops.create_subfolder(stitch_dir, "warped")

#%% 5.1) Warp and save warped outputs 
for job in cfg.warp_jobs:
    print(
        "Running warp job:",
        job["save_name"],
        f"(anno={job['anno']}, warper={job['warper_type']})",
    )
    custom_warper = cfg.build_warper(job)
    get_custom_warped_tensors(
        raw=raw,
        tensors=tensors,
        anno=job["anno"],
        warper=custom_warper,
        warped_dir=warped_dir,
        mask_cfg=cfg.mask_cfg,
        n_samples_warp=cfg.n_samples_warp,
        overlap_policy=job["overlap_policy"],
        save_name=job["save_name"],
        overwrite=job["overwrite"],
    )

# %%
