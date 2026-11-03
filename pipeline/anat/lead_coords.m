assert(~isempty(getenv('MYLFP_PROJECT_ROOT')), 'Set MYLFP_PROJECT_ROOT before running this script.');

project_folder = getenv('MYLFP_PROJECT_ROOT');
leaddbs_folder = fullfile(project_folder, 'derivatives/leaddbs');
subject_config_json = fullfile(project_folder, 'summary/cohort/lead/subject_coords_config.json');

results = dist2space_coords_batch(leaddbs_folder, subject_config_json);
