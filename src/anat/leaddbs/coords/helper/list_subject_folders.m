function subject_names = list_subject_folders(project_folder, options)
%LIST_SUBJECT_FOLDERS List Lead-DBS subject folder names under a project folder.
%
% subject_names = list_subject_folders(project_folder)
% subject_names = list_subject_folders(project_folder, Name, Value)
%
% Inputs
% ------
% project_folder
%   Path that contains subject directories (e.g., .../leaddbs).
%
% Name-Value Options
% ------------------
% SubjectPattern (char) : Folder pattern used by dir() (default: 'sub-*').
% Subjects (string)     : If provided, use exactly these subject folder names.
%
% Output
% ------
% subject_names
%   String array of subject folder names (not full paths).
%

arguments
    project_folder (1,:) char
    options.SubjectPattern (1,:) char = 'sub-*'
    options.Subjects string = string.empty(0, 1)
end

if ~isfolder(project_folder)
    error('project_folder does not exist or is not a folder: %s', project_folder);
end

if ~isempty(options.Subjects)
    subject_names = string(options.Subjects(:));
    subject_names = strip(subject_names);
    subject_names = subject_names(subject_names ~= "");
    subject_names = unique(subject_names, 'stable');
    return;
end

d = dir(fullfile(project_folder, options.SubjectPattern));
d = d([d.isdir]);

names = string({d.name});
names = names(~ismember(names, [".", ".."]));
names = sort(names);

subject_names = names;

end
