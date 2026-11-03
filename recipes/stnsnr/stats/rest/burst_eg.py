

#%% 0) Imports & interactive setup (paths, configs, matplotlib backend)
import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
    
import src.utils.paths as paths
import src.utils.fileops as fileops
from src.lfp.burst.grid import grid as burst_grid

#%%
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

summary_dir = project / 'summary'
stats_dir = summary_dir / 'stats'
eg_dir = stats_dir / 'eg' / 'burst'
fifpath = eg_dir / r"sub-SNr008_rest_bi_raw.fif"

bands = {
    "beta_low": (13.0, 20.0),
}

burst_grid_cfg = {
    "bands": bands,
    "min_cycles": 2,
    "hop_s": None,
    "decim": 1,
    "baseline_match": "exact",
    "baseline_fallback": "raise",
}

#%%
raw = mne.io.read_raw_fif(fifpath, preload=True)
print(raw)
print("sfreq (Hz):", raw.info["sfreq"])
print("n_times:", raw.n_times)
print("Duration (s):", raw.times[-1] - raw.times[0])
print("n_annotations:", len(raw.annotations))

#%%
raw_value = raw.get_data()[0, :] * 1e6 # to uV
time = raw.times

#%%
burst_tensor_4d, _ = burst_grid(
    raw,
    thresholds=[-1,],
    **burst_grid_cfg,
)
burst_value = burst_tensor_4d[0, 0, 0, :] * 1e6 # to uV

burst_tensor_valid, burst_meta = burst_grid(
    raw,
    percentile=75,
    **burst_grid_cfg,
)
burst_value_valid = burst_tensor_valid[0, 0, 0, :] * 1e6 # to uV
threshold = float(burst_meta["qc"]["thresholds"][0, 0]) * 1e6 # to uV

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time, burst_value, color="black", linewidth=1, label="burst_value")
ax.plot(time, burst_value_valid, color="#0E6AAF", linewidth=2, label="burst_value_valid")
ax.axhline(
    threshold,
    color="#F2000E",
    linestyle="--",
    linewidth=1,
    label=f"threshold={threshold:.2f}",
)
ax.fill_between(
    time,
    threshold,
    burst_value,
    where=burst_value > threshold,
    color="#F2000E",
    alpha=0.30,
    interpolate=True,
    label="burst_value > threshold",
)
ax.set_xlabel("time (s)")
ax.set_ylabel("Amplitude (uV)")
ax.set_title("Burst curve with threshold-highlighted region")
ax.legend(loc="upper right")
ax.grid(False)
plt.tight_layout()
plt.show()
fig.savefig(eg_dir / "hilbert.pdf", dpi=600)


# %%
