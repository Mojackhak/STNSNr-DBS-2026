
#%% Importing modules
import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.utils.paths as paths
import src.utils.fileops as fileops
from src.posetrack.merge_pose import merge_world_xyz
import src.posetrack.anipose_api as anipose_api
from src.posetrack.mmpose_tracker import VideoPoseTracker
from src.posetrack.pose_interp import interp_anipose_dir
import plugin.default.gait_track_default as cfg

import importlib
importlib.reload(cfg)

#%%#################################################################################
############# Section1: Create configuration #######################################
####################################################################################

#%% Step1: set the project folder
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

deri = Path(r"derivatives/posetrack")
deri = project / deri
sourcedata = r"sourcedata"
sourcedata = project / sourcedata
rawdata = r"rawdata"
rawdata = project / rawdata
subject = os.environ["MYLFP_SUBJECT"] # start with "sub-"
paradigm = r"gait"

#%% Step2: create the source folders

wd_parent = fileops.create_subfolder(sourcedata, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
# raw_folder = fileops.create_subfolder(wd, "raw")
videos_folder = fileops.create_subfolder(wd, "videos")

# %% Step3: copy the video and lfp files to the videos/raw folder in advance

#%% Step4: backup the raw data 
video_files = paths.get_files_list(videos_folder, filetypes='.mp4')
# raw_files = fileops.get_files_list(raw_folder, filetypes='.csv')
wd_parent = fileops.create_subfolder(rawdata, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
# raw_folder = fileops.create_subfolder(wd, "raw")
videos_folder = fileops.create_subfolder(wd, "videos")
copy_func = np.vectorize(lambda file: fileops.copy_file_to_new_folder(file, videos_folder))
video_files = copy_func(video_files)

#%% Step3: create posetrack related folder in derivatives

wd_parent = fileops.create_subfolder(deri, subject)
wd = fileops.create_subfolder(wd_parent, paradigm)
videos_raw_folder = fileops.create_subfolder(wd, "videos-raw")
copy_func = np.vectorize(lambda file: fileops.copy_file_to_new_folder(file, videos_raw_folder))
raw_files = copy_func(video_files)

 #%% Step4: Create the subfolders for the anipose pipeline
calibration = fileops.create_subfolder(wd, 'calibration')
videos_raw_folder = fileops.create_subfolder(wd, 'videos-raw')

events_folder = fileops.create_subfolder(wd, 'events')

config_template = project / cfg.config_path
config_path = fileops.copy_file_to_new_folder(config_template, wd)
config = anipose_api.anipose_load_config(config_path)
save_csv = config['save_as_csv']

calibration_template = project / cfg.calibration_path
calibration_path = fileops.copy_file_to_new_folder(calibration_template, calibration)

yolo_path = project / cfg.yolo_path

videos_raw_files = paths.get_files_list(videos_raw_folder, filetypes='.mp4')
print(f'Videos-raw path: {videos_raw_files}')


#%% Analyze the videos

# create tracker object
tracker = VideoPoseTracker(yolo_path, mmpose_model='rtmpose-l', scale=1.2, device='cuda:0')

videos_labeled_folder = fileops.create_subfolder(wd, 'videos-labeled')
pose_2d_folder = fileops.create_subfolder(wd, 'pose-2d')

pose_2d_files = []

for video_file in videos_raw_files:
    print(f"Processing video: {video_file}")
    basename = video_file.name
    # Process video and save keypoints
    output_coords = pose_2d_folder / basename.replace('.mp4', '.h5')
    output_video = videos_labeled_folder / basename
    
    df = tracker.process_video(
        video_file,
        output_coords=output_coords,
        visualize=False,
        output_video=output_video,
        skip_frames=1,  # Process every frame, set to higher value to speed up
        dlc_format=True
    )
    
    print(f"Keypoints saved to: {output_coords}")
    print(f"Pose visualization saved to: {output_video}")
    
    pose_2d_files.append(output_coords)
    
#%% Convert the pose data to csv format
pose_2d_folder = wd / 'pose-2d'
pose_2d_files = paths.get_files_list(pose_2d_folder, filetypes='.h5')

#%% Copy the pose data to the pose-2d-filtered folder

pose_2d_files = paths.get_files_list(pose_2d_folder, filetypes='.h5')
pose_2d_filtered_folder = wd / 'pose-2d-filtered'
# anipose_api.anipose_filter(config)
copy_func = np.vectorize(lambda file: fileops.copy_file_to_new_folder(file, pose_2d_filtered_folder))
pose_2d_filtered_files = copy_func(pose_2d_files)

pose_2d_filtered_files = paths.get_files_list(pose_2d_filtered_folder, filetypes='.h5')

#%% Combine the coodinate system data 

cs_folder = project / r"config/posetrack/cs"
# cs_folder = r"C:\Users\Ankang\Desktop\movinghand\model_v1\MovingHandCS-HAK-2024-04-15\labeled-data"
cs_files = paths.get_files_list(cs_folder, filetypes='.h5')
print(f'Coordinate system files: {cs_files}')

merge_world_xyz(pose_2d_filtered_files, cs_files, contain=r'(.*)(cam\d)(.*)', zoom=1)

#%% fill the na/nan values in the 2D pose data to 
# avoid (ValueError: Residuals are not finite in the initial point) in triangulation

for file in pose_2d_filtered_files:
    df = pd.read_hdf(file)
    df = df.fillna(0)
    df.to_hdf(file, key='df_with_missing', mode='w')
    
#%% Visualize the 2D pose data
anipose_api.anipose_label_2d_filter(config)

#%% Triangulate the 3D pose data

anipose_api.anipose_triangulate(config)
pose_3d_path = wd / 'pose-3d'
pose_3d_files = paths.get_files_list(pose_3d_path, filetypes='.csv')

#%% Interpolate the bad points in the 3D pose data (Optional)

pose_3d_interpolated_files = interp_anipose_dir(
    pose_3d_path, target_dir=wd, interp_method='akima', suffix='',
    threshold=99, gap=np.inf, save_evaluation=True, n_jobs=-1)

#%% fill the na/nan values in the 2D pose data to 
# avoid (ValueError: Residuals are not finite in the initial point) in triangulation

for file in pose_3d_interpolated_files:
    df = pd.read_csv(file)
    df = df.fillna(0)
    df.to_csv(file, index=False)
    
#%% filter the 3D pose data

anipose_api.anipose_filter_3d(config)

pose_3d_filtered_path = wd / 'pose-3d-filtered'
pose_3d_filtered_files = paths.get_files_list(pose_3d_filtered_path, filetypes='.csv')

#%% Create combined videos for 3D and 2D pose data

anipose_api.anipose_label_2d_filter(config)

anipose_api.anipose_label_3d_filter(config)

anipose_api.anipose_label_combined(config)

# %%
