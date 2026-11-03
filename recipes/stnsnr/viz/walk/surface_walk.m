assert(~isempty(getenv('MYLFP_PROJECT_ROOT')), 'Set MYLFP_PROJECT_ROOT before running this script.');

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(fileparts(script_dir))));
addpath(fullfile(repo_root, 'src', 'viz', 'surface', 'batch'));
addpath(fullfile(repo_root, 'src', 'viz', 'surface', 'batch_helper'));
addpath(fullfile(repo_root, 'src', 'viz', 'surface', 'render'));

v = region_views();

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

summary = visual_section_surface(fullfile(getenv('MYLFP_PROJECT_ROOT'), 'summary/stats/walk'), 'norm', ...
    v, [], cfg_plot, false, [], [], [], cfg_batch);
