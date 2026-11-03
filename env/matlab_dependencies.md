# MATLAB and external analysis resources

MATLAB workflows were retained but not executed during this extraction. Record
the actual MATLAB release, toolbox versions and external repository revisions
from the analysis installation before asserting historical reproducibility.

| Resource | Used for | Required external inputs |
| --- | --- | --- |
| MATLAB | Anatomy transforms and surface rendering | Installed release and any toolboxes required by the selected functions |
| Lead-DBS | Reconstruction coordinates, atlas lookup, nonlinear hemisphere mapping and surface display | Reconstruction files, transforms, atlas installation and Lead-DBS revision |
| Wave_clus | Interactive MER sorting | Acquisition files, sorting configuration, reviewed sorting output and Wave_clus revision |
| Pose inference software | Video-derived gait inputs | Trained detector/pose weights, inference configuration, synchronized videos and camera calibration |
| Illustrator | Final figure assembly and optional layered-PDF export | Original artwork, linked source panels, fonts and saved layout |

For MATLAB scripts, set environment variables before starting MATLAB or use:

```matlab
setenv('MYLFP_REPO_ROOT', '/absolute/path/to/MyLFP');
setenv('MYLFP_PROJECT_ROOT', '/absolute/path/to/STNSNr');
addpath(genpath(fullfile(getenv('MYLFP_REPO_ROOT'), 'src', 'anat', 'leaddbs')));
addpath(genpath(fullfile(getenv('MYLFP_REPO_ROOT'), 'src', 'viz', 'surface')));
```

Add the external Lead-DBS and Wave_clus installations through their documented
setup procedures. The local MATLAB helper files do not replace those toolboxes.
The anatomy scripts reference the selected combined atlas by its recorded name;
that atlas and its source licenses are not bundled.

The MER entrypoint supports manual MATLAB Wave_clus use without Python's MATLAB
Engine. Set `MYLFP_WAVECLUS_SCRIPT` to the installed `wave_clus.m` file if using
the optional Engine route. It is not required for the manual workflow.
The current source workflows do not justify a claim that GPU acceleration or
any particular non-standard hardware is necessary for the two synthetic demos.
