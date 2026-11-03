assert(~isempty(getenv('MYLFP_PROJECT_ROOT')), 'Set MYLFP_PROJECT_ROOT before running this script.');

project_folder = getenv('MYLFP_PROJECT_ROOT');
% coords_csv = fullfile(project_folder, 'summary/cohort/lead/contact_space_locs.csv');
% coords_csv = fullfile(project_folder, 'summary/cohort/lead/contact_reco_space_locs.csv');
coords_csv = fullfile(project_folder, 'summary/table/raw/mer/sorted/waveform_properties.csv');
direction = "L2R";
results = flip_coords_lr_nonlinear(...
    coords_csv, direction,...
    "OutCsv",coords_csv,...
    "WriteNewColumns", true);
