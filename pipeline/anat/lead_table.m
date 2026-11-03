assert(~isempty(getenv('MYLFP_PROJECT_ROOT')), 'Set MYLFP_PROJECT_ROOT before running this script.');

project_folder = getenv('MYLFP_PROJECT_ROOT');
leaddbs_folder = fullfile(project_folder, 'derivatives/leaddbs');
dist_table_path = fullfile(project_folder, 'summary/table/raw/mer/sorted/waveform_properties.csv');
dist_table = readtable(dist_table_path) ;

results = dist2space_coords_table(leaddbs_folder, dist_table, ...
    "SubjectColumn","Subject",...
    "OutCsv",dist_table_path);
