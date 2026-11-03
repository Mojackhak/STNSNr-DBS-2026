#%% 0) Imports & interactive setup (paths, configs, matplotlib backend)
import os
import numpy as np
import pandas as pd
import mne
import json
from pathlib import Path

import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.utils.paths as paths
import src.utils.fileops as fileops
import src.tabular.df_schema as df_schema
import src.io.converter as converter
import src.io.montage as montage
import src.preproc.channels as channels
import src.preproc.filter as filter
import src.preproc.ecg_remover as ecg_remover
import src.preproc.tag as tag
import src.qc.scalogram as scalogram
import plugin.default.preproc_default as cfg

%matplotlib qt

import importlib
importlib.reload(cfg)

#%% 1) Define project paths and dataset identifiers (subject/paradigm/sync settings)
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

deri = Path(r"derivatives/mne") # leaddbs project folder (Brain Imaging Data Structure (BIDS))
deri = project / deri
sourcedata = Path(r"sourcedata")
sourcedata = project / sourcedata
rawdata = Path(r"rawdata")
rawdata = project / rawdata

subject = os.environ["MYLFP_SUBJECT"] # start with "sub-"
paradigm = r"gait" # paradigm name, e.g., 'rest', 'standing', 'rest_medon', 'standing_medon'
annotations = r'gait'

sync_by_audio = True # if True, the audio sync file will be used to correct the time of the raw data

FPS_VIDEO = 25.0 # fps of the video recording

#%% 2) Create sourcedata working directories (raw/converted [+ audio/video if needed])
wd_parent = fileops.create_subfolder(sourcedata, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
raw_dir = fileops.create_subfolder(wd, "raw")
converted_dir = fileops.create_subfolder(wd, "converted")
if sync_by_audio:
    audio_dir = fileops.create_subfolder(wd, "audio")
    video_dir = fileops.create_subfolder(wd, "video")

#%% 3) Convert raw CSV -> channel-labeled CSV (pre-conversion step)
raw_file = paths.get_files_list(raw_dir, filetypes=(".csv"))[0]
csv_path = converter.uvcsv2vt(raw_file, correct_fs=False)
converted_file = fileops.move_files_to_folder([csv_path], converted_dir)[0]

#%% 4) Optional sync: align CSV timestamps to audio taps, then convert to MNE Raw (.fif)
tap_dist = 1 # the distance between two taps in seconds
tag_selected = [6,7,8,9,10]

if sync_by_audio:
    audio_file = paths.get_files_list(audio_dir, filetypes=(".wav", ".mp3"))[0]
    save_path = converted_file[:-4] + '_sync.csv' # '_reorg.csv' -> '_reorg_sync.csv'
    synced_file = converter.sync_csv_audio(converted_file, audio_file, 
                                           tag_selected, save_path, 
                                           tap_dist=tap_dist, correct_sf=False)
    mne_file = converter.csv2mne(synced_file)
else:
    mne_file = converter.csv2mne(converted_file)

#%% 4.1) Reset reference (optional)
if cfg.montage is not None:
    raw = mne.io.read_raw_fif(mne_file, preload=False)
    raw = montage.reset_ref(raw, cfg.montage, drop_refs=True)
    raw.save(mne_file, overwrite=True)
    
#%% 5) Backup the original raw .fif into rawdata/ (immutable copy)
wd_parent = fileops.create_subfolder(rawdata, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
raw_dir = fileops.create_subfolder(wd, "raw")
mne_file = fileops.copy_file_to_new_folder(mne_file, raw_dir, overwrite=True)
mne_basename = subject + "_raw.fif"
mne_file = fileops.rename_file_or_folder(mne_file, mne_basename, overwrite=True)

#%% 6) Copy raw .fif into derivatives/mne (working copy for preprocessing)

wd_parent = fileops.create_subfolder(deri, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
raw_dir = fileops.create_subfolder(wd, "raw")
raw_file = fileops.copy_file_to_new_folder(mne_file, raw_dir, overwrite=True)
rawfile_basename = os.path.basename(raw_file).rsplit(".")[0]

#%% 7) Load raw data from derivatives and start preprocessing (MNE)
#### Preprocessing EEG data ###############################################################
###########################################################################################

wd_parent = fileops.create_subfolder(deri, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
raw_dir = fileops.create_subfolder(wd, "raw")
raw_file = paths.get_files_list(raw_dir, filetypes='.fif')[0]
rawfile_basename = os.path.basename(raw_file).rsplit(".")[0]
raw = mne.io.read_raw_fif(raw_file, preload=True)

#%% 8) Derive channel mapping and bipolar contact pairs from Raw
chs = raw.ch_names
type2dbs = {ch: 'dbs' for ch in chs}
type2eeg = {ch: 'eeg' for ch in chs}

# 1) Derive everything (mapping + bipolar pairs)
derived = channels.derive_channels_from_raw(raw, drop_non_adjacent_pairs=True)

# 2) Use mapping
ch2eeg = derived.ch2eeg
eeg2ch = derived.eeg2ch

# 3) Bipolar pairs
anodes = derived.anodes
cathodes = derived.cathodes
new_names = derived.pair_names
dropped_idx = derived.dropped_pair_idx

#%% 9) Set montage for visualization (optional QC of sensor layout)
'''
Using the 10-05 EEG system for electrode positioning, but in DBS, 
the posterior-to-anterior direction of the head corresponds to 
the ventral-to-dorsal orientation of the DBS electrodes
'''
raw.rename_channels(ch2eeg)
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)
raw.rename_channels(eeg2ch)
fig = raw.plot_sensors(show_names=True)
save_path = os.path.join(raw_dir, f"{rawfile_basename}_montage.png")

#%% 10) Quick PSD of raw data (sanity check)
raw.compute_psd().plot()

#%% 11) Create folders for event labeling (posetrack + events)
posetrack_dir = fileops.create_subfolder(wd, 'posetrack')
event_dir = fileops.create_subfolder(wd, 'events')

#%% 12) Create an empty marker CSV template for manual event annotation
events_header = {'condition':str, 'start':int, 'end':int}
marker_path = os.path.join(posetrack_dir, f"markers.csv")
# Write to CSV file
if not os.path.exists(marker_path):
    print(f"Creating marker file at {marker_path}")
    df_posetrack = df_schema.make_typed_dataframe(events_header)
    df_posetrack.to_csv(marker_path, index=False)
else:
    print(f"Marker file already exists at {marker_path}, skipping creation.")

#%% 13) Event detection: convert marker CSV -> boolean events -> QC figure + save
posetrack_file = paths.get_files_list(posetrack_dir, filetypes='.csv')[0]
if sync_by_audio:
    fps = FPS_VIDEO
else:
    fps = raw.info['sfreq']
df_posetrack = pd.read_csv(posetrack_file)
df_posetrack = df_schema.coerce_df_types(df_posetrack, events_header)
event = tag.df_to_boolean_dataframe(df_posetrack, cfg.events[paradigm])
fig = tag.visualize_boolean_df(event, cfg.events[paradigm])
event.to_csv(os.path.join(event_dir, f"events.csv"), index=False)
fig.savefig(os.path.join(event_dir, f"events_manual.png"), dpi=300)

#%% 14) Add detected/annotated gait events to Raw as MNE annotations
raw_labeled = raw.copy()
raw_labeled.set_annotations(raw.annotations + raw_labeled.annotations)
raw_labeled = tag.frames_df_to_annotations(event, raw_labeled, fps)

#%% 15) QC plot: inspect labeled Raw with events/annotations
raw_labeled.plot()

#%% 16) Create preprocessing output folder (derivatives/.../preproc)
preproc_dir = fileops.create_subfolder(wd, "preproc")

#%% 15b) Save labeled raw data for record-keeping (optional)
save_path = os.path.join(preproc_dir, f"{rawfile_basename}_labeled_raw.fif")
raw_labeled.save(save_path, overwrite=True)

#%% 17) Inspect sampling rate and adapt notch frequencies to actual sfreq
# raw
raw.info

correct_factor_fs = raw.info['sfreq'] / cfg.sfreq_base_hz

print(f"Correct factor for the sampling frequency: {correct_factor_fs:.5f}")
notch = cfg.notches_base_hz * correct_factor_fs
cfg.notch_cfg['notches'] = notch

#%% 18) Build artifact/segment rejection configuration (notch + bandpass + rules)
# Example usage inside a pipeline script

bad_cfg = filter.BadAnnotationConfig(
    **cfg.bad_annot_cfg,
    **cfg.notch_cfg,
    **cfg.bandpass_cfg,
)

#%% 19) Detect and annotate bad segments (artifact marking) on labeled Raw
raw_marked, reject_log, summary = filter.mark_lfp_bad_segments(
    raw_labeled,
    bad_cfg,
    eeg_like_channels=raw.ch_names,
    reject_plot_path=Path(preproc_dir) / f"{rawfile_basename}_reject.png",
)

#%% 20) Manual QC: inspect marked bad segments
raw_marked.plot()

#%% 20) Save marked raw data for record-keeping (optional)
save_path = os.path.join(preproc_dir, f"{rawfile_basename}_marked_raw.fif")
raw_marked.save(save_path, overwrite=True)

#%% 20a) Create raw_good working copy (raw_marked -> raw_good)
raw_good = raw_marked.copy()

#%% 21) Filter data and exclude annotated bad segments; save JSON report (Optional)

raw_good, report = filter.filter_lfp_with_bad_annotations(
    raw_good,
    **cfg.bad_annot_filter_cfg,
)
save_path = os.path.join(preproc_dir, f"{rawfile_basename}_report.json")
with open(save_path, 'w') as f:
    f.write(json.dumps(report, indent=4))

#%% 22) Add user-defined task annotation while avoiding segment edges
raw_good, report = filter.add_head_tail_annotations(
    raw_good,
    **cfg.add_end_annot_cfg,
)

raw_good, report = filter.add_annotation_avoiding_edges(
    raw_good,
    description=annotations,
    **cfg.add_annot_cfg,
)

#%% 23) QC plot: inspect cleaned continuous data with annotations
raw_good.plot()

#%% 24) PSD after cleaning (0-200 Hz)
raw_good_psd = raw_good.compute_psd(fmin=0, fmax=200)
raw_good_psd.plot()
#%% 24a) Create raw_ecg branch (raw_good -> raw_ecg) for optional ECG cleaning
raw_ecg = raw_good.copy()

#%% 24b) ECG artifact removal via SVD/template subtraction (optional)
###################################################################################
picks=['0_7'] # select the ECG contaminated channel(s) for ECG removal
remover_params = cfg.ecg_remove_cfg
raw_ecg, figs_ecg = ecg_remover.raw_call_ecgremover(
    raw_good,
    picks=picks,
    **remover_params,
)

#%% 24c) ECG-SVD QC: save ECG templates and remover parameters
for ch, fig in figs_ecg.items():
    save_path = preproc_dir / f"{rawfile_basename}_{ch}_ecg_template.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

saveparams_path = preproc_dir / f"{rawfile_basename}_ecg_remover_params.json"
with open(saveparams_path, 'w') as f:
    json.dump(remover_params, f)
    
#%% 24d) ECG-SVD QC: inspect ECG-cleaned raw_ecg (interactive plot)
raw_ecg.plot()
###################################################################################
#%% 24e) ECG artifact removal via ICA (optional)

ica = mne.preprocessing.ICA(random_state=42, max_iter='auto', method='picard')

ica.fit(raw_ecg, reject_by_annotation=True)

#%% 24f) ECG-ICA QC: plot ICA components (manual inspection)
fig = ica.plot_components()  # Visually inspect
save_path = preproc_dir / f"{rawfile_basename}_ica_components.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight')

#%% 24g) ECG-ICA QC: print explained variance ratios
explained_var_ratio = ica.get_explained_variance_ratio(raw_ecg)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")
    
#%% 24h) ECG-ICA QC: inspect ICA sources (time courses)
ica.plot_sources(raw_ecg)

#%% 24i) ECG-ICA QC: plot IC diagnostic properties
figs = ica.plot_properties(raw_ecg, np.arange(ica.n_components_))
for i, fig in enumerate(figs):
    save_path = preproc_dir / f"{rawfile_basename}_ica_{i:03}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
#%% 24j) ECG-ICA QC: overlay IC exclusion effect on raw signal

fig = ica.plot_overlay(raw_ecg, exclude=[6], picks="eeg")
save_path = preproc_dir / f"{rawfile_basename}_ica_overlay.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight')

#%% 24k) ECG-ICA: select artifact components to exclude (set ica.exclude)
ica.exclude = [6]    # Select artifact-related components

#%% 24l) ECG-ICA: set channel types/names to EEG for ICA application
raw_ecg.set_channel_types(type2eeg)
raw_ecg.rename_channels(ch2eeg)

#%% 24m) ECG-ICA: apply ICA and restore DBS channel names/types

ica.apply(raw_ecg)

raw_ecg.rename_channels(eeg2ch)
raw_ecg.set_channel_types(type2dbs)

#%% 24n) ECG-ICA QC: inspect post-ICA cleaned signal
raw_ecg.plot()

#%% 24o) Finalize raw_good (use ECG-cleaned raw_ecg as raw_good)
raw_good = raw_ecg.copy()

#%% 25) QC plot: inspect time series
raw_good.plot()

#%% 26) PSD of cleaned data + save figure
raw_good_psd = raw_good.compute_psd()
raw_good_psd.plot()
plt.savefig(preproc_dir / f"{rawfile_basename}_raw_psd.png", dpi=300)

#%% 27) Save cleaned continuous data (bi_raw.fif)
save_path = os.path.join(preproc_dir, f"{subject}_bi_raw.fif")
raw_good.save(save_path, overwrite=True)

#%% 28) QC: save per-channel scalograms (time-frequency plots)
freqs_hz = np.logspace(np.log10(cfg.bandpass_cfg['l_freq']), 
                       np.log10(cfg.bandpass_cfg['h_freq']), num=50)
figs = scalogram.save_channel_scalograms_from_raw(raw_good, 
                                                  freqs_hz=freqs_hz,
                                                  out_dir=preproc_dir,
                                                  decim=10,)

#%% 29) End of pipeline
