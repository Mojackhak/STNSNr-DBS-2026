function [config, results] = make_subject_coords_config_from_lead_library(project_folder, lead_library_json, out_json, options)
%MAKE_SUBJECT_COORDS_CONFIG_FROM_LEAD_LIBRARY Build a subject coords config JSON based on reco.electrode.
%
% [config, results] = make_subject_coords_config_from_lead_library(project_folder, lead_library_json, out_json)
%
% This function:
%   1) Loads a lead library JSON that maps electrode model -> (dist, offset)
%   2) Iterates subject folders and reads Lead-DBS reconstruction (reco.electrode)
%   3) Assigns dist/offset for each subject-side based on electrode model
%   4) Writes a subject config JSON (units + subjects[sub-xxx][R/L]{dist,offset,lead})
%
% Inputs
% ------
% project_folder
%   Lead-DBS project folder that contains subject directories (e.g., .../leaddbs).
% lead_library_json
%   Path to lead library JSON file, or a decoded MATLAB struct.
% out_json
%   Output JSON file path. If empty, no file is written.
%
% Lead Library JSON Schema
% ------------------------
% {
%   "units": "mm",
%   "lead": {
%     "Medtronic 3389": { "dist": [..], "offset": [x, y, depth] },
%     "Medtronic 3387": { "dist": [..], "offset": [x, y, depth] }
%   }
% }
%
% Name-Value Options
% ------------------
% SubjectPattern (char)      : Folder pattern for subjects (default: 'sub-*').
% Subjects (string)          : If provided, process only these subjects.
% Overwrite (logical)        : Overwrite output JSON if exists (default: true).
% ContinueOnError (logical)  : Continue after per-subject errors (default: true).
% Verbose (logical)          : Print progress (default: true).
% OnMissingLead (char)       : 'error' or 'skip' when a lead model is not found in the library (default: 'error').
%
% Outputs
% -------
% config
%   Struct with fields:
%     - units (char)
%     - subjects_map (containers.Map)
% results
%   Table with per-subject status/messages.
%
% Dependencies
% ------------
% Requires Lead-DBS on MATLAB path:
%   ea_getptopts
%

arguments
    project_folder (1,:) char
    lead_library_json
    out_json (1,:) char = ''
    options.SubjectPattern (1,:) char = 'sub-*'
    options.Subjects string = string.empty(0, 1)
    options.Overwrite (1,1) logical = true
    options.ContinueOnError (1,1) logical = true
    options.Verbose (1,1) logical = true
    options.OnMissingLead (1,:) char {mustBeMember(options.OnMissingLead, {'error','skip'})} = 'error'
end

require_functions({"ea_getptopts"});

lib = load_lead_library(lead_library_json);

subject_names = list_subject_folders(project_folder, ...
    'SubjectPattern', options.SubjectPattern, ...
    'Subjects', options.Subjects);

subjects_map = containers.Map('KeyType', 'char', 'ValueType', 'any');

% Results table (extended for convenience).
n = numel(subject_names);
results = table('Size', [n, 5], ...
    'VariableTypes', {'string', 'string', 'string', 'string', 'string'}, ...
    'VariableNames', {'subject', 'status', 'message', 'lead_R', 'lead_L'});

for i = 1:n
    subj = subject_names(i);
    results.subject(i) = subj;
    results.status(i) = "";
    results.message(i) = "";
    results.lead_R(i) = "";
    results.lead_L(i) = "";

    subj_folder = fullfile(project_folder, char(subj));

    try
        if options.Verbose
            fprintf('[%s] CONFIG  %s\n', string(datetime()), subj);
        end

        if ~isfolder(subj_folder)
            error('Subject folder not found: %s', subj_folder);
        end

        opt = ea_getptopts(subj_folder);
        [reco, reco_path] = load_reco_from_opt(opt);

        if ~isfield(opt, 'sides')
            error('Lead-DBS options missing opt.sides for subject: %s', subj);
        end

        sides = opt.sides;
        subj_entry = struct();

        for side = sides
            side_label = side_index_to_label(side);

            lead_name = get_reco_lead_name(reco, side);
            lead_key = matlab.lang.makeValidName(lead_name);

            if ~isfield(lib.lead, lead_key)
                msg = sprintf('Lead model "%s" not found in lead library JSON.', lead_name);
                if strcmp(options.OnMissingLead, 'skip')
                    if options.Verbose
                        fprintf('[%s] WARN   %s %s: %s\n', string(datetime()), subj, side_label, msg);
                    end
                    continue;
                end
                error(msg);
            end

            entry = lib.lead.(lead_key);
            dist = normalize_dist(entry, sprintf('%s.%s.dist', subj, side_label));
            offset = normalize_offset(entry, sprintf('%s.%s.offset', subj, side_label));

            subj_entry.(side_label) = struct( ...
                'dist', dist, ...
                'offset', offset, ...
                'lead', char(lead_name) ...
            );

            if side_label == 'R'
                results.lead_R(i) = lead_name;
            elseif side_label == 'L'
                results.lead_L(i) = lead_name;
            end
        end

        if isempty(fieldnames(subj_entry))
            error('No valid side entries were generated for %s (check reco.electrode and lead library).', subj);
        end

        subjects_map(char(subj)) = subj_entry;

        results.status(i) = "ok";
        results.message(i) = "";

        if options.Verbose
            fprintf('[%s] OK     %s (reco: %s)\n', string(datetime()), subj, reco_path);
        end

    catch ME
        results.status(i) = "error";
        results.message(i) = string(ME.message);

        if options.Verbose
            fprintf('[%s] ERROR  %s: %s\n', string(datetime()), subj, ME.message);
        end

        if ~options.ContinueOnError
            rethrow(ME);
        end
    end
end

config = struct();
config.units = lib.units;
config.subjects_map = subjects_map;

if ~isempty(out_json)
    write_subject_coords_config_json(out_json, lib.units, subjects_map, 'Overwrite', options.Overwrite);
end

end

% ========================================================================
function lib = load_lead_library(lead_library_json)

if ischar(lead_library_json) || isstring(lead_library_json)
    lib_raw = read_json_file(char(lead_library_json));
else
    lib_raw = lead_library_json;
end

if ~isstruct(lib_raw)
    error('lead_library_json must be a JSON path or a decoded struct.');
end

if ~isfield(lib_raw, 'units')
    error('Lead library JSON missing required field "units".');
end
units = string(lib_raw.units);
if lower(units) ~= "mm"
    error('Lead library units must be "mm". Got: %s', units);
end

if ~isfield(lib_raw, 'lead')
    error('Lead library JSON missing required field "lead".');
end
if ~isstruct(lib_raw.lead)
    error('Lead library JSON field "lead" must be a JSON object.');
end

lib = struct();
lib.units = char(units);
lib.lead = lib_raw.lead;

end

% ========================================================================
function [reco, reco_path] = load_reco_from_opt(opt)

if ~isfield(opt, 'subj') || ~isfield(opt.subj, 'recon') || ~isfield(opt.subj.recon, 'recon')
    error('Lead-DBS options missing opt.subj.recon.recon.');
end

reco_path = opt.subj.recon.recon;
if ~isfile(reco_path)
    error('Reconstruction file not found: %s', reco_path);
end

s = load(reco_path, 'reco');
if ~isfield(s, 'reco')
    error('Reconstruction file does not contain variable "reco": %s', reco_path);
end

reco = s.reco;

end

% ========================================================================
function side_label = side_index_to_label(side_index)

switch side_index
    case 1
        side_label = 'R';
    case 2
        side_label = 'L';
    otherwise
        error('Unsupported Lead-DBS side index: %d (expected 1=R, 2=L).', side_index);
end

end

% ========================================================================
function dist = normalize_dist(entry, context)

if ~isfield(entry, 'dist')
    error('Lead library entry missing "dist": %s', context);
end

dist = entry.dist;
if isempty(dist)
    dist = zeros(0,1);
    return;
end
if ~isnumeric(dist)
    error('Lead library "%s" dist must be numeric.', context);
end

dist = double(dist(:));

if any(~isfinite(dist))
    error('Lead library "%s" dist contains NaN/Inf.', context);
end

end

% ========================================================================
function offset = normalize_offset(entry, context)

if ~isfield(entry, 'offset')
    error('Lead library entry missing "offset": %s', context);
end

offset = entry.offset;
if ~isnumeric(offset)
    error('Lead library "%s" offset must be numeric.', context);
end

offset = double(offset(:).');
if numel(offset) ~= 3
    error('Lead library "%s" offset must have 3 elements [x y depth].', context);
end

if any(~isfinite(offset))
    error('Lead library "%s" offset contains NaN/Inf.', context);
end

end

% ========================================================================
function write_subject_coords_config_json(out_path, units, subjects_map, options)
%WRITE_SUBJECT_COORDS_CONFIG_JSON Write subject coords config JSON with arbitrary subject keys.
%
% write_subject_coords_config_json(out_path, units, subjects_map)
%
% Inputs
% ------
% out_path
%   Output JSON file path.
% units
%   Unit string (e.g., 'mm').
% subjects_map
%   containers.Map where keys are subject IDs (e.g., 'sub-001') and values are
%   structs with side fields ('R','L',...) each containing:
%     - dist   : numeric vector
%     - offset : 1x3 numeric vector [x y depth]
%     - lead   : string/char (electrode model name)
%
% Name-Value Options
% ------------------
% Overwrite (logical) : If false, error when out_path exists (default: true).
%
% Notes
% -----
% This function avoids MATLAB struct field-name restrictions by writing JSON manually.
% It also ensures that numeric vectors remain JSON arrays even when they have length 1.
%

arguments
    out_path (1,:) char
    units (1,:) char
    subjects_map
    options.Overwrite (1,1) logical = true
end

if ~isa(subjects_map, 'containers.Map')
    error('subjects_map must be a containers.Map. Got: %s', class(subjects_map));
end

if isfile(out_path) && ~options.Overwrite
    error('Output JSON already exists and Overwrite=false: %s', out_path);
end

[folder, ~, ~] = fileparts(out_path);
if ~isempty(folder) && ~isfolder(folder)
    mkdir(folder);
end

if subjects_map.Count == 0
    subject_keys = strings(0,1);
else
    subject_keys = string(subjects_map.keys);
    subject_keys = sort(subject_keys);
end

lines = build_lines(units, subjects_map, subject_keys);
txt = strjoin(lines, newline);

fid = fopen(out_path, 'w');
if fid < 0
    error('Failed to open output file for writing: %s', out_path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fwrite(fid, txt, 'char');

end

% ========================================================================
function lines = build_lines(units, subjects_map, subject_keys)

indent1 = "  ";
indent2 = "    ";
indent3 = "      ";

lines = strings(0,1);
lines(end+1,1) = "{";
lines(end+1,1) = indent1 + quote_key("units") + ": " + jsonencode(string(units)) + ",";
lines(end+1,1) = indent1 + quote_key("subjects") + ": {";

for si = 1:numel(subject_keys)
    subj = subject_keys(si);
    subj_value = subjects_map(char(subj));

    if ~isstruct(subj_value)
        error('Subject entry for "%s" must be a struct.', subj);
    end

    side_names = string(fieldnames(subj_value));
    side_names = order_side_names(side_names);

    lines(end+1,1) = indent2 + jsonencode(subj) + ": {";

    for sj = 1:numel(side_names)
        side = side_names(sj);
        entry = subj_value.(char(side));

        validate_side_entry(entry, sprintf('%s.%s', subj, side));

        dist_json = json_array_from_numeric(entry.dist);
        offset_json = json_array_from_numeric(entry.offset);

        lead_str = "";
        if isfield(entry, "lead")
            lead_str = string(entry.lead);
        end

        comma = ",";
        if sj == numel(side_names)
            comma = "";
        end

        lines(end+1,1) = indent3 + jsonencode(side) + ": { " + ...
            quote_key("dist") + ": " + dist_json + ", " + ...
            quote_key("offset") + ": " + offset_json + ", " + ...
            quote_key("lead") + ": " + jsonencode(lead_str) + " }" + comma;
    end

    subj_comma = ",";
    if si == numel(subject_keys)
        subj_comma = "";
    end
    lines(end+1,1) = indent2 + "}" + subj_comma;
end

lines(end+1,1) = indent1 + "}";
lines(end+1,1) = "}";

end

% ========================================================================
function s = quote_key(key)
% Quote a JSON object key (always a string).
s = jsonencode(string(key));
end

% ========================================================================
function arr_json = json_array_from_numeric(x)
% Always encode numeric input as a JSON array (even if scalar).
if isempty(x)
    arr_json = "[]";
    return;
end

if ~isnumeric(x)
    error('Expected numeric input, got %s.', class(x));
end

x = double(x(:).'); % force row
cells = num2cell(x);
arr_json = string(jsonencode(cells));
end

% ========================================================================
function side_names = order_side_names(side_names)
% Prefer R then L; otherwise keep alphabetical.

preferred = ["R", "L"];
ordered = strings(0,1);

for i = 1:numel(preferred)
    if any(side_names == preferred(i))
        ordered(end+1,1) = preferred(i); %#ok<AGROW>
    end
end

rest = setdiff(side_names, preferred, 'stable');
rest = sort(rest);

side_names = [ordered; rest];

end

% ========================================================================
function validate_side_entry(entry, context)
if ~isstruct(entry)
    error('Side entry "%s" must be a struct.', context);
end
if ~isfield(entry, 'dist')
    error('Side entry "%s" missing required field "dist".', context);
end
if ~isfield(entry, 'offset')
    error('Side entry "%s" missing required field "offset".', context);
end

dist = entry.dist;
if ~(isnumeric(dist) || isempty(dist))
    error('Side entry "%s".dist must be numeric.', context);
end

off = entry.offset;
if ~isnumeric(off)
    error('Side entry "%s".offset must be numeric.', context);
end
off = double(off(:).');
if numel(off) ~= 3
    error('Side entry "%s".offset must have 3 elements [x y depth].', context);
end
if any(~isfinite(off))
    error('Side entry "%s".offset contains NaN/Inf.', context);
end
end

% ========================================================================
function lead_name = get_reco_lead_name(reco, electrode_index)
%GET_RECO_LEAD_NAME Extract DBS lead model name from a Lead-DBS reco struct.
%
% lead_name = get_reco_lead_name(reco, electrode_index)
%
% Convention
% ----------
% This project uses a single, explicit convention for lead model name:
%   reco.electrode(i).dbs.elmodel
%
% This project uses a single, strict convention (no fallbacks):
%   reco.electrode(i).dbs.elmodel
%
% Inputs
% ------
% reco
%   Lead-DBS reconstruction struct loaded from opt.subj.recon.recon.
% electrode_index
%   Index into reco.electrode (typically matches Lead-DBS side index: 1=R, 2=L).
%
% Output
% ------
% lead_name
%   Trimmed lead model name (e.g., "Medtronic 3389").
%

arguments
    reco (1,1) struct
    electrode_index (1,1) double {mustBeInteger, mustBePositive, mustBeMember(electrode_index, [1 2])}
end

if ~isfield(reco, 'electrode')
    error('reco is missing required field "electrode".');
end

els = reco.electrode;
if ~isstruct(els)
    error('reco.electrode must be a struct array. Got: %s', class(els));
end

if numel(els) < electrode_index
    error('reco.electrode has only %d elements; requested index %d.', numel(els), electrode_index);
end

e = els(electrode_index);
if ~isfield(e, 'dbs')
    error('reco.electrode(%d) missing required field "dbs".', electrode_index);
end

dbs = e.dbs;
if ~isstruct(dbs)
    error('Expected reco.electrode(%d).dbs to be a struct containing field "elmodel". Got: %s', electrode_index, class(dbs));
end

if ~isfield(dbs, 'elmodel')
    error('reco.electrode(%d).dbs is missing required field "elmodel".', electrode_index);
end

lead_name = string(dbs.elmodel);

lead_name = strtrim(lead_name);
if strlength(lead_name) == 0
    error('Empty lead model name extracted from reco.electrode(%d).dbs.', electrode_index);
end

end
