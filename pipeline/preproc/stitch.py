#%% 0) Imports & interactive setup (paths, configs, matplotlib backend)
import os
import mne
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.utils.fileops as fileops
import src.preproc.stitch as stitch

%matplotlib qt

#%% 1) Define project paths and dataset identifiers (subject/paradigm/sync settings)
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

deri = Path(r"derivatives/mne") # leaddbs project folder (Brain Imaging Data Structure (BIDS))
deri = project / deri
subject = os.environ["MYLFP_SUBJECT"] # start with "sub-"
wd = deri / subject
stitch_dir = fileops.create_subfolder(wd, "test")
paradigm_list = ["standing", "gait"]

#%% 2) Loop through paradigms
raw_list = []
for paradigm in paradigm_list:
    print(f"Processing paradigm: {paradigm}")
    fifpath = wd / paradigm / 'preproc' / f"{subject}_bi_raw.fif"
    if not os.path.exists(fifpath):
        continue
    raw = mne.io.read_raw_fif(fifpath, preload=True)
    raw_list.append(raw)

#%% 3) Stitching the paradigms together
raw_stitched = stitch.stitch_raws(raw_list)

#%% 4) QC: check the stitched raw
raw_stitched.plot()

#%% 5) Save the stitched raw
preproc_dir = fileops.create_subfolder(stitch_dir, "preproc")
save_path = preproc_dir / f"{subject}_stitched_bi_raw.fif"
raw_stitched.save(save_path, overwrite=True)

# %%
