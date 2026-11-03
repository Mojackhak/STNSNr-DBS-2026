function results = dist2space_coords_batch(project_folder, subject_config_json, options)
%DIST2SPACE_COORDS_BATCH Batch-convert axial distances using a subject coords config JSON.
%
% results = dist2space_coords_batch(project_folder, subject_config_json)
%
% Inputs
% ------
% project_folder
%   Lead-DBS project folder that contains subject directories (e.g., .../leaddbs).
% subject_config_json
%   Path to subject config JSON file, or a decoded MATLAB struct.
%
% Subject Config JSON Schema
% --------------------------
% {
%   "units": "mm",
%   "subjects": {
%     "sub-001": {
%       "R": { "dist": [..], "offset": [x,y,depth], "lead": "Medtronic 3389" },
%       "L": { "dist": [..], "offset": [x,y,depth], "lead": "Medtronic 3389" }
%     }
%   }
% }
%
% Name-Value Options
% ------------------
% SubjectPattern (char)       : Folder pattern for subjects on disk (default: 'sub-*').
% Subjects (string)           : If provided, process only these subjects.
% SaveName (char)             : Per-subject MAT output file name (default: 'contact_space_coords.mat').
% SaveInSubjectFolder (logical): If true, save into each subject folder (default: true).
% SaveFolder (char)           : If SaveInSubjectFolder=false, save into this folder (default: '').
% Overwrite (logical)         : If false, skip existing outputs (default: true).
% ContinueOnError (logical)   : Continue after per-subject errors (default: true).
% Verbose (logical)           : Print progress (default: true).
% OnMissingConfig (char)      : 'error' or 'skip' when a subject has no config entry (default: 'error').
%
% Output
% ------
% results (table)
%   One row per subject with columns:
%     - subject
%     - subject_folder
%     - save_path
%     - status ('ok' | 'skipped' | 'error')
%     - message
%
% Dependencies
% ------------
% - dist2space_coords (same folder)
% - Lead-DBS toolbox on path (used by dist2space_coords)
%

arguments
    project_folder (1,:) char
    subject_config_json
    options.SubjectPattern (1,:) char = 'sub-*'
    options.Subjects string = string.empty(0, 1)
    options.SaveName (1,:) char = 'contact_space_coords.mat'
    options.SaveInSubjectFolder (1,1) logical = true
    options.SaveFolder (1,:) char = ''
    options.Overwrite (1,1) logical = true
    options.ContinueOnError (1,1) logical = true
    options.Verbose (1,1) logical = true
    options.OnMissingConfig (1,:) char {mustBeMember(options.OnMissingConfig, {'error','skip'})} = 'error'
end

if ~isfolder(project_folder)
    error('project_folder does not exist or is not a folder: %s', project_folder);
end

cfg = load_subject_config(subject_config_json);

subject_names = list_subject_folders(project_folder, ...
    'SubjectPattern', options.SubjectPattern, ...
    'Subjects', options.Subjects);

results = init_batch_results_table(subject_names, project_folder);

for i = 1:height(results)
    subj = results.subject(i);
    subj_folder = char(results.subject_folder(i));

    save_path = resolve_batch_save_path(subj_folder, char(subj), ...
        'SaveName', options.SaveName, ...
        'SaveInSubjectFolder', options.SaveInSubjectFolder, ...
        'SaveFolder', options.SaveFolder);

    results.save_path(i) = string(save_path);

    try
        if ~isfolder(subj_folder)
            error('Subject folder not found: %s', subj_folder);
        end

        if ~options.Overwrite && isfile(save_path)
            results.status(i) = "skipped";
            results.message(i) = "Output exists; skipped (Overwrite=false).";
            if options.Verbose
                fprintf('[%s] SKIP  %s (exists): %s\n', string(datetime()), subj, save_path);
            end
            continue;
        end

        subj_cfg = get_subject_cfg(cfg.subjects, char(subj));
        if isempty(subj_cfg)
            msg = sprintf('Subject "%s" not found in subject config JSON.', subj);
            if strcmp(options.OnMissingConfig, 'skip')
                results.status(i) = "skipped";
                results.message(i) = string(msg);
                if options.Verbose
                    fprintf('[%s] SKIP  %s: %s\n', string(datetime()), subj, msg);
                end
                continue;
            end
            error(msg);
        end

        [dist_cell, offset_cell] = subject_coords_config_to_cells(subj_cfg);

        if options.Verbose
            fprintf('[%s] START %s\n', string(datetime()), subj);
        end

        dist2space_coords(subj_folder, dist_cell, offset_cell, ...
            'SavePath', save_path, ...
            'Verbose', false);

        results.status(i) = "ok";
        results.message(i) = "";

        if options.Verbose
            fprintf('[%s] DONE  %s -> %s\n', string(datetime()), subj, save_path);
        end

    catch ME
        results.status(i) = "error";
        results.message(i) = string(ME.message);

        if options.Verbose
            fprintf('[%s] ERROR %s: %s\n', string(datetime()), subj, ME.message);
        end

        if ~options.ContinueOnError
            rethrow(ME);
        end
    end
end

end

% ========================================================================
function cfg = load_subject_config(subject_config_json)

if ischar(subject_config_json) || isstring(subject_config_json)
    cfg_raw = read_json_file(char(subject_config_json));
else
    cfg_raw = subject_config_json;
end

if ~isstruct(cfg_raw)
    error('subject_config_json must be a JSON path or a decoded struct.');
end

if ~isfield(cfg_raw, 'units')
    error('Subject config JSON missing required field "units".');
end
units = string(cfg_raw.units);
if lower(units) ~= "mm"
    error('Subject config units must be "mm". Got: %s', units);
end

if ~isfield(cfg_raw, 'subjects')
    error('Subject config JSON missing required field "subjects".');
end
if ~isstruct(cfg_raw.subjects)
    error('Subject config JSON field "subjects" must be a JSON object.');
end

cfg = struct();
cfg.units = char(units);
cfg.subjects = cfg_raw.subjects;

end

% ========================================================================
function subj_cfg = get_subject_cfg(subjects_struct, subject_name)
% Find a subject config entry using MATLAB's JSON key normalization rules.

subj_cfg = [];

field = matlab.lang.makeValidName(subject_name);
if isfield(subjects_struct, field)
    subj_cfg = subjects_struct.(field);
end

end

% ========================================================================
function [dist_cell, offset_cell] = subject_coords_config_to_cells(subject_cfg, options)
%SUBJECT_COORDS_CONFIG_TO_CELLS Convert one subject config entry into Lead-DBS side-indexed cells.
%
% [dist_cell, offset_cell] = subject_coords_config_to_cells(subject_cfg)
%
% Inputs
% ------
% subject_cfg
%   Struct with optional fields 'R' and/or 'L'. Each side contains:
%     - dist   : numeric scalar/vector (mm)
%     - offset : numeric vector with 3 elements [x y depth] (mm)
%     - lead   : (optional) string
%
% Name-Value Options
% ------------------
% DefaultOffset (1x3 double): default offset for missing sides (default: [0 0 0])
%
% Outputs
% -------
% dist_cell
%   1x2 cell, dist_cell{1}=R distances (Nx1), dist_cell{2}=L distances (Nx1)
% offset_cell
%   1x2 cell, offset_cell{1}=R offset (1x3), offset_cell{2}=L offset (1x3)
%

arguments
    subject_cfg (1,1) struct
    options.DefaultOffset (1,3) double = [0 0 0]
end

dist_cell = {zeros(0,1), zeros(0,1)};
offset_cell = {options.DefaultOffset, options.DefaultOffset};

if isfield(subject_cfg, 'R')
    [dist_cell{1}, offset_cell{1}] = parse_side_entry(subject_cfg.R, 'R');
end

if isfield(subject_cfg, 'L')
    [dist_cell{2}, offset_cell{2}] = parse_side_entry(subject_cfg.L, 'L');
end

end

% ========================================================================
function [dist_mm, offset_mm] = parse_side_entry(entry, side_label)

if ~isstruct(entry)
    error('Subject config side "%s" must be an object.', side_label);
end

if ~isfield(entry, 'dist')
    error('Subject config side "%s" missing required field "dist".', side_label);
end
if ~isfield(entry, 'offset')
    error('Subject config side "%s" missing required field "offset".', side_label);
end

dist = entry.dist;
if isempty(dist)
    dist_mm = zeros(0,1);
elseif isnumeric(dist)
    dist_mm = double(dist(:));
else
    error('Subject config side "%s".dist must be numeric.', side_label);
end

offset = entry.offset;
if ~isnumeric(offset)
    error('Subject config side "%s".offset must be numeric.', side_label);
end
offset_mm = double(offset(:).');
if numel(offset_mm) ~= 3
    error('Subject config side "%s".offset must have 3 elements [x y depth].', side_label);
end

if any(~isfinite(dist_mm))
    error('Subject config side "%s".dist contains NaN/Inf.', side_label);
end
if any(~isfinite(offset_mm))
    error('Subject config side "%s".offset contains NaN/Inf.', side_label);
end

end
