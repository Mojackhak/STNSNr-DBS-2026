function results = init_batch_results_table(subject_names, project_folder)
%INIT_BATCH_RESULTS_TABLE Initialize a standard results table for batch processing.
%
% results = init_batch_results_table(subject_names, project_folder)
%
% Output columns:
%   - subject
%   - subject_folder
%   - save_path
%   - status ('ok'|'skipped'|'error')
%   - message
%

arguments
    subject_names string
    project_folder (1,:) char
end

n = numel(subject_names);

results = table('Size', [n, 5], ...
    'VariableTypes', {'string', 'string', 'string', 'string', 'string'}, ...
    'VariableNames', {'subject', 'subject_folder', 'save_path', 'status', 'message'});

for i = 1:n
    subj = subject_names(i);
    results.subject(i) = subj;
    results.subject_folder(i) = string(fullfile(project_folder, char(subj)));
    results.save_path(i) = "";
    results.status(i) = "";
    results.message(i) = "";
end

end
