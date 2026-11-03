assert(~isempty(getenv('MYLFP_PROJECT_ROOT')), 'Set MYLFP_PROJECT_ROOT before running this script.');

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(fileparts(script_dir))));
addpath(fullfile(repo_root, 'src', 'viz', 'surface', 'batch'));
addpath(fullfile(repo_root, 'src', 'viz', 'surface', 'batch_helper'));
addpath(fullfile(repo_root, 'src', 'viz', 'surface', 'render'));

% MER uses a wider custom camera than the default STN/SNr batch views.
vSNr1 = struct();
vSNr1.az        = 0.8823;
vSNr1.el        = 0.1224;
vSNr1.camva     = 0.3492;
vSNr1.camup     = [0 0 1];
vSNr1.camproj   = 'orthographic';
vSNr1.camtarget = [10.2192 -16.6493 -11.9080];
vSNr1.campos    = [-841.7542 -1.6139e+03 475.5032];

vSNr2 = struct();
vSNr2.az        = -0.8227;
vSNr2.el        = -0.1286;
vSNr2.camva     = 0.3492;
vSNr2.camup     = [0 0 1];
vSNr2.camproj   = 'orthographic';
vSNr2.camtarget = [9.0846 -16.1730 -12.4099];
vSNr2.campos    = [1.0473e+03 1.4862e+03 411.5260];

vSTN1 = struct();
vSTN1.az        = 0.9190;
vSTN1.el        = 0.1644;
vSTN1.camva     = 0.3175;
vSTN1.camup     = [0 0 1];
vSTN1.camproj   = 'orthographic';
vSTN1.camtarget = [8.6423 -20.0641 -3.4598];
vSTN1.campos    = [-663.1839 -1.5861e+03 778.1860];

vSTN2 = struct();
vSTN2.az        = -0.6810;
vSTN2.el        = 0.3717;
vSTN2.camva     = 0.3175;
vSTN2.camup     = [0 0 1];
vSTN2.camproj   = 'orthographic';
vSTN2.camtarget = [7.0658 -16.3686 -3.6039];
vSTN2.campos    = [1.1899e+03 1.0837e+03 -955.1834];

v = struct();
v.STN = {vSTN1, vSTN2};
v.SNr = {vSNr1, vSNr2};

cfg_nifti2patch = struct();
cfg_nifti2patch.Colormap = 'viridis';
cfg_nifti2patch.CLimMode = 'auto';

cfg_plot = default_plot_patch_config();
cfg_plot.ExportContentType = 'mixed';
cfg_plot.ExportTransparent = false;
cfg_plot.ExportRenderer = 'painters';
cfg_plot.FigureVisible = 'off';
cfg_plot.FigureBackend = 'matlab';
cfg_plot.StrictHeadless = true;
cfg_plot.ColorbarLabelStyle = 'plain';
cfg_plot.ColorbarLabelInterpreter = 'none';
cfg_plot.ColorbarTickLabelInterpreter = 'none';

cfg_batch = struct();
cfg_batch.OutputFormats = "pdf";
cfg_batch.Headless = true;
cfg_batch.HeadlessMode = "strict";
cfg_batch.UseParallel = true;
cfg_batch.NumWorkers = [];
cfg_batch.PathMustContain = "";
cfg_batch.SkipExisting = false;
cfg_batch.CloseFigures = true;

RAW_YLABEL_TPL = containers.Map();
RAW_YLABEL_TPL('spike') = '<key>';

BAND_TPL = containers.Map();
BAND_TPL('Burst_index') = 'ln burst index';
BAND_TPL('LvR') = 'LvR';
BAND_TPL('Peak_to_valley') = 'ln peak-to-valley (s)';
BAND_TPL('Peak_trough_ratio') = 'ln peak/trough ratio';
BAND_TPL('Half_width') = 'ln half-width (s)';
BAND_TPL('Repolarization_slope') = 'ln repolarization slope (V/s)';
BAND_TPL('Recovery_slope') = 'ln recovery slope (V/s)';
BAND_TPL('Firing_rate') = 'arcsinh firing rate (Hz)';

summary = visual_section_surface(fullfile(getenv('MYLFP_PROJECT_ROOT'), 'summary/stats/mer'), 'raw', ...
    v, cfg_nifti2patch, cfg_plot, false, ...
    RAW_YLABEL_TPL, BAND_TPL, [], cfg_batch);
