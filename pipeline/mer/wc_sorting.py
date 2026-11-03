#%%
"""
Main purpose
------------
Wave_clus-based MER sorting and analysis pipeline (interactive #%% script).

This file is intended to be executed cell-by-cell in IPython/Jupyter/VScode.

Pipeline outline
----------------
1) Convert NeuroOmega MAP/MPX to NEX (requires NeuroExplorer + `nex` Python API)
2) Read spike-band stream from NEX -> export Wave_clus-compatible .mat
3) Run Wave_clus in MATLAB (recommended) or via SpikeInterface sorter (optional)
4) Read Wave_clus results -> SpikeInterface sortings/analyzers
5) Compute QC metrics + STN boundaries + unit labeling
6) Export figures and summary tables
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spikeinterface.extractors.neoextractors.neuroexplorer import NeuroExplorerRecordingExtractor

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.utils.fileops as fileops
import src.utils.paths as paths
from src.io.converter import matrix2mne
from src.mer.spectral import psd_to_bandpower
import src.mer as mer
import src.mer.waveclus_pipeline as wc
from src.mer.spikeinterface_adapter import save_spikeinterface_objects
import plugin.default.mer_default as cfg


import importlib
importlib.reload(cfg)

#%%#################################################################################
############# Section1: Create configuration #######################################
####################################################################################

#%% Step1: set the project folder
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

deri = project / "derivatives" / "spikeinterface"  # Lead-DBS BIDS derivatives folder
sourcedata = project / Path("sourcedata")
rawdata = project / Path("rawdata")

subject = os.environ["MYLFP_SUBJECT"]  # start with "sub-"
paradigm = r"surgery"

#%% Step2: create the source folders
wd_parent = fileops.create_subfolder(sourcedata, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
raw_folder = fileops.create_subfolder(wd, "raw")
converted_folder = fileops.create_subfolder(wd, "converted")

#%% Step3: convert the raw MAP/MPX file to NEX (requires full NeuroExplorer)
# Copy the MAP/MPX files to `raw_folder` in advance.
# Note: delete files with size < 100 kb, which are usually not valid MAP/MPX files.
_ = mer.convert_map_mpx_to_nex(raw_folder, output_dir=converted_folder)

#%% Step4: backup the raw data into `rawdata/`
nexfiles = paths.get_files_list(converted_folder, filetypes=".nex")

wd_parent = fileops.create_subfolder(rawdata, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
raw_folder_bak = fileops.create_subfolder(wd, "raw")

nexfiles_bak = []
for f in nexfiles:
    nexfiles_bak.append(fileops.copy_file_to_new_folder(f, raw_folder_bak))


#%% Step5: create SpikeInterface working folder in derivatives
wd_parent = fileops.create_subfolder(deri, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
raw_folder_deri = fileops.create_subfolder(wd, "raw")

raw_files = []
for f in nexfiles_bak:
    raw_files.append(fileops.copy_file_to_new_folder(f, raw_folder_deri))

raw_files = paths.get_files_list(raw_folder_deri, filetypes=".nex")


#%% Step6: inspect available streams in the first NEX file
streams = NeuroExplorerRecordingExtractor.get_streams(file_path=raw_files[0])
print("Available streams:", streams[0])


#%% Step7: read spike-band stream and export Wave_clus input files
# REF (MER spike band): https://pubmed.ncbi.nlm.nih.gov/16763982
# REF (MER spike band): https://www.sciencedirect.com/science/article/pii/S0969996122001395
bandpass_spk = cfg.band_filter  # Frequency band for spike filtering
name_selected_spk = "SPK 01"

mat_folder = fileops.create_subfolder(wd, "raw_mat")

recording_list = wc.read_neuroexplorer_recordings(
    raw_files,
    stream_name=name_selected_spk,
    bandpass_hz=bandpass_spk,
    show_traces=False,
)
_ = wc.export_recordings_to_waveclus_mat(recording_list, mat_folder)


#%%#################################################################################
############# Section2: Sorting ####################################################
####################################################################################

#%% Way1: Run Wave_clus in MATLAB to sort the data manually (recommended)
# You can also run Wave_clus directly in MATLAB without matlab.engine.
# import matlab.engine

# eng = matlab.engine.start_matlab()  # Start MATLAB (this may take a few seconds)
# wave_clus_path = Path(cfg.wave_clus_path).expanduser()
# if not wave_clus_path.exists():
#     raise FileNotFoundError(
#         "wave_clus.m was not found. Please update cfg.wave_clus_path. "
#         f"Current value: {wave_clus_path}"
#     )

# wave_clus_dir = wave_clus_path.parent
# wave_clus_fig_candidates = [
#     wave_clus_dir / "wave_clus.fig",
#     wave_clus_dir / "figs" / "wave_clus.fig",
# ]
# wave_clus_fig = next((p for p in wave_clus_fig_candidates if p.exists()), None)
# if wave_clus_fig is None:
#     raise FileNotFoundError(
#         "wave_clus.fig was not found. Checked: "
#         + ", ".join(str(p) for p in wave_clus_fig_candidates)
#     )

# # Wave_clus GUI needs the repo subfolders on the MATLAB path so GUIDE can
# # find resources such as ``figs/wave_clus.fig``.
# eng.addpath(eng.genpath(str(wave_clus_dir)), nargout=0)
# eng.cd(str(wave_clus_dir), nargout=0)
# print("Using Wave_clus:", wave_clus_path)
# print("Using Wave_clus GUI:", wave_clus_fig)


#%% Create per-recording folders for Wave_clus outputs
sorting_folder = os.path.join(wd, "waveclus_sorting")
for rec in recording_list:
    base = rec.get_annotation("base")
    _ = fileops.create_subfolder(sorting_folder, base)


# #%% Run Wave_clus interactively (it will ask for files / produce outputs)
# eng.wave_clus(nargout=0)


#%% Move Wave_clus output files into per-base folders
_ = wc.organize_waveclus_outputs(mat_folder, sorting_folder)

# # %% When done, quit MATLAB
# eng.quit()  


#%% Read Wave_clus outputs back into SpikeInterface Sorting objects
sorting_WC_list = wc.load_waveclus_sortings(recording_list, sorting_folder, keep_good_only=True)
for s in sorting_WC_list:
    print(s)
    for unit_id in s.get_unit_ids():
        st = s.get_unit_spike_train(unit_id)
        print(f"Num. spikes for unit {unit_id}: {len(st)}")


#%% Way2: Run wave_clus via SpikeInterface sorter (optional, disabled by default)
# import spikeinterface.full as si
# from spikeinterface.sorters import WaveClusSorter
# Configure the Wave_clus installation before using a SpikeInterface sorter.
# custom_params = dict(min_clus=10)
# sorting_WC_list = []
# for i, recording in enumerate(recording_list):
#     base = recording.get_annotation("base")
#     subfolder = os.path.join(sorting_folder, base)
#     try:
#         sorting_WC = si.run_sorter(
#             sorter_name="waveclus",
#             recording=recording,
#             folder=subfolder,
#             verbose=True,
#             **custom_params,
#         )
#         sorting_WC.annotate(base=base, recording_i=i)
#     except Exception as e:
#         print(f"WaveClus failed on {base}: {e}")
#         sorting_WC = None
#     sorting_WC_list.append(sorting_WC)


#%% Generate a probe with one contact (single-channel MER)
probe = wc.make_single_contact_probe(radius=5, ypitch=20)


#%% Analyze the sorted data
analyzer_list = wc.create_sorting_analyzers(recording_list, sorting_WC_list, probe)
wc.compute_default_analyzer_extensions(
    analyzer_list,
    **cfg.compute_default_analyzer_extensions_cfg)                                      

#%% Save QC figures for each analyzer
fig_folder = fileops.create_subfolder(wd, "figures")
wc.save_analyzer_qc_figures(analyzer_list, fig_folder, close_figures=True)


#%%#################################################################################
############# Section3: Label the units in the sorting (Glu, GABA, DA) ##############
####################################################################################

#%% Step1: read the LFP stream
name_selected_lfp = "LFP 01"
LFP_list = wc.read_neuroexplorer_recordings(
    raw_files,
    stream_name=name_selected_lfp,
    bandpass_hz=cfg.lfp_filter,
    # 50 Hz powerline and harmonics
    notch_freqs_hz=cfg.notch_filter,
    show_traces=False,
)


#%% STN boundaries via nRMS
# nRMS entry threshold for STN
nRMS_threshold = 1.2  # REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC6500885
# nRMS_threshold = 2.0 # REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC10255000
# nRMS_threshold = 1.5 # REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC10664412

_ = wc.assert_single_tract_per_side(
    recording_list,
    label="recording_list used for nRMS/STN boundaries",
)

RMS_df = wc.compute_rms_df_from_recordings(recording_list)

# Use pre-STN RMS as baseline (top 5 most dorsal points)
# REF: https://pubmed.ncbi.nlm.nih.gov/16763982/
RMS_df = wc.add_side_baseline_and_normalize(
    RMS_df,
    side_col="side",
    depth_col="depth",
    value_col="RMS",
    baseline_top_n=5,
    baseline_stat="median",
    baseline_col="baseline",
    normalized_col="nRMS",
)
RMS_df = RMS_df.sort_values(by=["side", "depth"], ascending=[True, False])

x_loess_dict, y_loess_dict = wc.loess_smooth_by_side(
    RMS_df, side_col="side", x_col="depth", y_col="nRMS", frac=0.2
)

#%% Visualize nRMS curves (raw + LOESS)
for side, g in RMS_df.groupby("side"):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x_loess_dict[side], y_loess_dict[side], marker="o", linestyle="-", color="blue", label="nRMS (smoothed)")
    plt.plot(g["depth"], g["nRMS"], marker="o", linestyle="", color="gray", alpha=0.5, label="nRMS")
    plt.axhline(y=nRMS_threshold, color="r", linestyle="--", label="nRMS threshold")
    plt.title(f"nRMS for {side} side")
    plt.xlabel("Depth (mm)")
    plt.ylabel("nRMS")
    plt.legend()


#%% Calculate STN entry boundaries
# STN_entry_dict_manual = {}  # manual override has the highest priority eg: STN_entry_dict_manual = {"L": 6, "R": 7}
# entry_range_dict = {"L": [4, 8], "R": [5, 6]}  # monotonic range for crossing

# STN_entry_dict = mer.get_STN_entry(
#     x_dict=x_loess_dict,
#     y_dict=y_loess_dict,
#     entry_range_dict=entry_range_dict,
#     STN_entry_dict_manual=STN_entry_dict_manual,
#     nRMS_threshold=nRMS_threshold,
#     method="crossing",  # "crossing" or "median"
# )


#%% Calculate STN exit boundaries using the minimum of nRMS curve
# STN_exit_dict_manual = {}
# exit_range_dict = {"L": [-2, 1], "R": [-2, 1]}

# STN_exit_dict = mer.get_STN_exit(
#     x_dict=x_loess_dict,
#     y_dict=y_loess_dict,
#     exit_range_dict=exit_range_dict,
#     STN_exit_dict_manual=STN_exit_dict_manual,
#     method="min",
# )


#%% STN exit via 100-150 Hz / 5-25 Hz bandpower ratio
# REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC5478927
_ = wc.assert_single_tract_per_side(
    LFP_list,
    label="LFP_list used for STN bandpower boundaries",
)

bands = {"high": [100, 150], "low": [5, 25]}
bandpower_ratio_rows = []

for LFP in LFP_list:
    side = LFP.get_annotation("side")
    depth = float(LFP.get_annotation("depth"))

    data = LFP.get_traces()
    sr = LFP.get_sampling_frequency()
    ch_names = list(LFP.get_channel_ids())

    LFP_mne = matrix2mne(data, sfreq=sr, ch_names=ch_names)
    psd_mne = LFP_mne.compute_psd(fmin=1, fmax=200, method="welch")
    psd = psd_mne.get_data()
    freqs = psd_mne.freqs

    bp_results = psd_to_bandpower(psd, freqs=freqs, bands=bands)
    bp_ratio = (bp_results["high"] / bp_results["low"])[0]

    bandpower_ratio_rows.append(
        {
            "side": side,
            "depth": depth,
            "bandpower_100_150": bp_results["high"][0],
            "bandpower_5_25": bp_results["low"][0],
            "bandpower_ratio": bp_ratio,
        }
    )

bandpower_ratio_df = pd.DataFrame(bandpower_ratio_rows)

#%% Normalize bandpower ratio per side + LOESS smoothing
STN_boundaries_folder = fileops.create_subfolder(wd, "STN_boundaries")

nbandpower_ratio_threshold = 1  # REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC5478927/

bandpower_ratio_df = wc.add_side_baseline_and_normalize(
    bandpower_ratio_df,
    side_col="side",
    depth_col="depth",
    value_col="bandpower_ratio",
    baseline_top_n=5,
    baseline_stat="median",
    baseline_col="baseline",
    normalized_col="nbandpower_ratio",
)
bandpower_ratio_df = bandpower_ratio_df.sort_values(by=["side", "depth"], ascending=[True, False])

x_loess_dict0, y_loess_dict0 = wc.loess_smooth_by_side(
    bandpower_ratio_df, side_col="side", x_col="depth", y_col="nbandpower_ratio", frac=0.2
)

for side, g in bandpower_ratio_df.groupby("side"):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x_loess_dict0[side], y_loess_dict0[side], marker="o", linestyle="-", color="blue", label="nbandpower_ratio (smoothed)")
    plt.plot(g["depth"], g["nbandpower_ratio"], marker="o", linestyle="", color="gray", alpha=0.5, label="nbandpower_ratio")
    plt.axhline(y=nbandpower_ratio_threshold, color="r", linestyle="--", label="nbandpower_ratio threshold")
    plt.title(f"nbandpower_ratio for {side} side")
    plt.xlabel("Depth (mm)")
    plt.ylabel("nbandpower_ratio")
    plt.legend()

    fig.savefig(os.path.join(STN_boundaries_folder, f"nbandpower_ratio_{side}_side.png"))
    g.to_csv(os.path.join(STN_boundaries_folder, f"nbandpower_ratio_{side}_side.csv"), index=False)


#%% STN exit boundaries using the minimum of the bandpower curve
# exit_range_dict0 = {"L": [-2, 0], "R": [-2, 1]}

# STN_exit_dict = mer.get_STN_exit(
#     x_dict=x_loess_dict0,
#     y_dict=y_loess_dict0,
#     exit_range_dict=exit_range_dict0,
#     STN_exit_dict_manual=STN_exit_dict_manual,
#     method="min",
# )

#%% Set boundary manually
STN_entry_dict = {"L":6, "R": 7}
STN_exit_dict = {"L": -1, "R": -2}

#%% Visualize nRMS and STN entry/exit boundaries
for side in RMS_df["side"].unique():
    g = RMS_df[RMS_df["side"] == side]

    fig = plt.figure(figsize=(6, 4))
    plt.plot(x_loess_dict[side], y_loess_dict[side], marker="o", linestyle="-", color="blue", label="nRMS (smoothed)")
    plt.plot(g["depth"], g["nRMS"], marker="o", linestyle=":", color="gray", alpha=0.5, label="nRMS")

    stn_entry = STN_entry_dict[side]
    stn_exit = STN_exit_dict[side]

    plt.axhline(y=nRMS_threshold, color="r", linestyle="--", label="nRMS threshold")
    plt.axvline(x=stn_entry, color="green", linestyle="--", label="STN Entry")
    plt.axvline(x=stn_exit, color="orange", linestyle="--", label="STN Exit")

    plt.title(f"nRMS for {side} side")
    plt.xlabel("Depth (mm)")
    plt.ylabel("nRMS")
    plt.legend()

    fig.savefig(os.path.join(STN_boundaries_folder, f"nRMS_{side}_side.png"))
    g.to_csv(os.path.join(STN_boundaries_folder, f"nRMS_{side}_side.csv"), index=False)


#%% Burst Index (BI) and LvR per unit
# REF: https://www.nature.com/articles/s41598-020-74196-5
burst_df = wc.compute_burst_lvr_df(analyzer_list)

burst_folder = fileops.create_subfolder(wd, "burst")
for side in burst_df["side"].unique():
    g = burst_df[burst_df["side"] == side]

    fig = plt.figure(figsize=(6, 4))
    plt.scatter(g["depth"], g["burst_index"], c=g["unit_id"], cmap="viridis", alpha=0.7)
    plt.title(f"Burst Index for {side} side")
    plt.xlabel("Depth (mm)")
    plt.ylabel("Burst Index")
    fig.savefig(os.path.join(burst_folder, f"BI_{side}_side.png"))

    fig = plt.figure(figsize=(6, 4))
    plt.scatter(g["depth"], g["lvr"], c=g["unit_id"], cmap="viridis", alpha=0.7)
    plt.title(f"LvR for {side} side")
    plt.xlabel("Depth (mm)")
    plt.ylabel("LvR")
    fig.savefig(os.path.join(burst_folder, f"LvR_{side}_side.png"))

    g.to_csv(os.path.join(burst_folder, f"burst_{side}_side.csv"), index=False)


#%% Label units in the sorting (Glu, GABA, DA)
label_rows = []
for analyzer in analyzer_list:
    template_metrics = analyzer.get_extension("template_metrics").get_data()
    quality_metrics = analyzer.get_extension("quality_metrics").get_data()

    recording = analyzer.recording
    sorting = analyzer.sorting
    depth = float(recording.get_annotation("depth"))
    side = recording.get_annotation("side")

    stn_exit = STN_exit_dict[side]
    stn_entry = STN_entry_dict[side]

    label_list = {}
    for unit_id in analyzer.unit_ids:
        # Outside STN (ventral)
        if depth <= stn_exit:
            # GABA-like waveform
            # REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC12053694
            if (template_metrics.loc[unit_id, "peak_to_valley"] * 1000 < 0.8) and (quality_metrics.loc[unit_id, "firing_rate"] > 15):
                label = "GABA"
            # DA-like waveform
            # REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC12053694
            elif (template_metrics.loc[unit_id, "peak_to_valley"] * 1000 > 0.8) and (quality_metrics.loc[unit_id, "firing_rate"] < 15):
                label = "DA"
            else:
                label = "unknown"

        # Inside STN
        elif stn_exit < depth < stn_entry:
            # Glu-like waveform
            # REF: https://pubmed.ncbi.nlm.nih.gov/18378458/
            # REF: https://pmc.ncbi.nlm.nih.gov/articles/PMC4073906/
            if (template_metrics.loc[unit_id, "peak_to_valley"] * 1000 > 0.3) and (quality_metrics.loc[unit_id, "firing_rate"] > 15):
                label = "Glu"
            else:
                label = "Glu"
        else:
            label = "unknown"

        label_list[unit_id] = label
        label_rows.append({"side": side, "depth": depth, "unit_id": unit_id, "label": label})

    recording.annotate(label_list=label_list)
    sorting.annotate(label_list=label_list)

label_df = pd.DataFrame(label_rows)

label_folder = fileops.create_subfolder(wd, "labels")
label_df.to_csv(os.path.join(label_folder, "labels_side.csv"), index=False)

label_counts = label_df["label"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
plt.bar(label_counts.index, label_counts.values)


#%% Save SpikeInterface objects
recording_folder = fileops.create_subfolder(wd, "recording")
save_spikeinterface_objects(recording_list, recording_folder)

sorting_folder_out = fileops.create_subfolder(wd, "sorting")
save_spikeinterface_objects(sorting_WC_list, sorting_folder_out)

analyzer_folder = fileops.create_subfolder(wd, "analyzer")
save_spikeinterface_objects(analyzer_list, analyzer_folder)

LFP_folder = fileops.create_subfolder(wd, "LFP")
save_spikeinterface_objects(LFP_list, LFP_folder)


#%% Export waveform properties table
waveform_folder = fileops.create_subfolder(wd, "waveform")
_ = wc.export_waveform_properties_csv(
    analyzer_list,
    burst_df,
    os.path.join(waveform_folder, "waveform_properties.csv"),
)

# %%
