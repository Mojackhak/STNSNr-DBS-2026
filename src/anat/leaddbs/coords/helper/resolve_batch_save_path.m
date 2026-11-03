function save_path = resolve_batch_save_path(subject_folder, subject_name, options)
%RESOLVE_BATCH_SAVE_PATH Resolve output save path for a per-subject MAT file.
%
% save_path = resolve_batch_save_path(subject_folder, subject_name, Name, Value)
%
% Name-Value Options
% ------------------
% SaveName (char)             : File name for per-subject output (default: 'contact_space_coords.mat').
% SaveInSubjectFolder (logical): If true, save into subject folder (default: true).
% SaveFolder (char)           : If SaveInSubjectFolder=false, save into this folder.
%

arguments
    subject_folder (1,:) char
    subject_name (1,:) char
    options.SaveName (1,:) char = 'contact_space_coords.mat'
    options.SaveInSubjectFolder (1,1) logical = true
    options.SaveFolder (1,:) char = ''
end

if options.SaveInSubjectFolder
    save_path = fullfile(subject_folder, 'reconstruction', options.SaveName);
    return;
end

if isempty(options.SaveFolder)
    error('When SaveInSubjectFolder=false, options.SaveFolder must be provided.');
end

if ~isfolder(options.SaveFolder)
    mkdir(options.SaveFolder);
end

save_path = fullfile(options.SaveFolder, sprintf('%s_%s', subject_name, options.SaveName));

end
