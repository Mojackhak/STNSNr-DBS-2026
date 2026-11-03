function result_table = dist2space_coords_table(project_folder, dist_table, options)
%DIST2SPACE_COORDS_TABLE convert axial distances to MNI coordinates for a table.
%
% This helper takes a row-wise table (e.g., MER / contact depth samples) and
% appends MNI coordinates computed via dist2space_coords (Lead-DBS dependent).
%
% Required table columns (configurable via name-value options)
% ------------------------------------------------------------
% - subject: subject identifier (e.g., '001' or 'sub-001')
% - side   : 'L' or 'R' (case-insensitive; can also be categorical/string)
% - depth  : axial distance in mm (numeric)
% - offset : [x y depth] in mm (numeric vector or a string containing 3 numbers)
%
% Name-Value Options
% ------------------
% SubjectColumn (char) : column name for subject id (default: 'subject')
% SideColumn    (char) : column name for side label (default: 'side')
% DepthColumn   (char) : column name for distance (default: 'depth')
% OffsetColumn  (char) : column name for offset vector (default: 'offset')
% OutCsv        (char) : if non-empty, write result table to this CSV path
% Verbose       (logical): print progress (default: true)
%
% Outputs
% -------
% result_table
%   Input table plus:
%     - MNI_x, MNI_y, MNI_z : numeric columns (NaN if unavailable)
%     - coord_status        : string ('ok' | 'missing_subject_folder' | 'error')
%     - coord_message       : string (diagnostic message)
%
% Dependencies
% ------------
% - dist2space_coords (this folder)
% - Lead-DBS toolbox on path (used by dist2space_coords)
%

arguments
    project_folder (1,:) char
    dist_table table
    options.SubjectColumn (1,:) char = 'subject'
    options.SideColumn (1,:) char = 'side'
    options.DepthColumn (1,:) char = 'depth'
    options.OffsetColumn (1,:) char = 'offset'
    options.OutCsv (1,:) char = ''
    options.Verbose (1,1) logical = true
end

% Resolve actual variable names (case-insensitive).
subject_var = table_find_var(dist_table, {options.SubjectColumn});
side_var    = table_find_var(dist_table, {options.SideColumn});
depth_var   = table_find_var(dist_table, {options.DepthColumn});
offset_var  = table_find_var(dist_table, {options.OffsetColumn});

n = height(dist_table);
result_table = dist_table;

% Preallocate outputs.
result_table.MNI_x = nan(n, 1);
result_table.MNI_y = nan(n, 1);
result_table.MNI_z = nan(n, 1);
% result_table.mni = cell(n, 1);
result_table.coord_status = strings(n, 1);
result_table.coord_message = strings(n, 1);

for i = 1:n
    try
        subj_id = normalize_subject_id(result_table.(subject_var)(i));
        subj_folder = fullfile(project_folder, subj_id);

        if ~isfolder(subj_folder)
            result_table.coord_status(i) = "missing_subject_folder";
            result_table.coord_message(i) = "Folder not found: " + string(subj_folder);
            continue;
        end

        side_label = normalize_side_label(result_table.(side_var)(i));
        side_index = side_label_to_index(side_label);

        depth_mm = parse_numeric_scalar(result_table.(depth_var)(i), sprintf('%s(row %d)', depth_var, i));
        offset_mm = parse_numeric_rowvector(result_table.(offset_var)(i), 3, sprintf('%s(row %d)', offset_var, i));

        % Build side-indexed cells (Lead-DBS convention: typically 1=R, 2=L).
        dist_cell = {[], []};
        offset_cell = {[0 0 0], [0 0 0]};
        dist_cell{side_index} = depth_mm;
        offset_cell{side_index} = offset_mm;

        coords = dist2space_coords(subj_folder, dist_cell, offset_cell, 'Verbose', false);

        mni_xyz = coords.mni{side_index};
        if isempty(mni_xyz)
            result_table.coord_status(i) = "error";
            result_table.coord_message(i) = "Empty MNI result (check reconstruction/warps).";
            continue;
        end

        mni_row = mni_xyz(1, :);
        result_table.MNI_x(i) = mni_row(1);
        result_table.MNI_y(i) = mni_row(2);
        result_table.MNI_z(i) = mni_row(3);
        % result_table.mni{i} = mni_row;

        result_table.coord_status(i) = "ok";
        result_table.coord_message(i) = "";

        if options.Verbose && mod(i, max(1, floor(n/10))) == 0
            fprintf('[%s] dist2space_coords_table: %d/%d rows processed\n', string(datetime()), i, n);
        end

    catch ME
        result_table.coord_status(i) = "error";
        result_table.coord_message(i) = string(ME.message);
    end
end

if ~isempty(options.OutCsv)
    writetable(result_table, options.OutCsv, ...
        'FileType', 'text', ...
        'Encoding', 'UTF-8', ...
        'WriteVariableNames', true, ...
        'QuoteStrings', true);
end

end

% ========================================================================
function subj_id = normalize_subject_id(v)
% Normalize subject ID to 'sub-XXX' form.

s = string(v);
s = strtrim(s);

if startsWith(lower(s), "sub-")
    subj_id = char(s);
else
    subj_id = char("sub-" + s);
end

end

% ========================================================================
function side = normalize_side_label(v)
% Normalize side label to 'L' or 'R'.

if isnumeric(v)
    if isscalar(v)
        if v == 1
            side = 'R';
            return;
        elseif v == 2
            side = 'L';
            return;
        end
    end
    error('Numeric side labels must be scalar 1 (R) or 2 (L).');
end

s = upper(strtrim(string(v)));
if startsWith(s, "L")
    side = 'L';
elseif startsWith(s, "R")
    side = 'R';
else
    error('Unsupported side label: %s (expected L/R).', s);
end

end

% ========================================================================
function idx = side_label_to_index(side)
% Map 'R'/'L' to Lead-DBS side index (1=R, 2=L).

switch side
    case 'R'
        idx = 1;
    case 'L'
        idx = 2;
    otherwise
        error('Unsupported side: %s', side);
end

end

% ========================================================================
function v = parse_numeric_rowvector(x, expected_len, label)
%LDBS_PARSE_NUMERIC_ROWVECTOR Parse a numeric row vector from heterogeneous inputs.
%
% v = parse_numeric_rowvector(x, expected_len, label)
%
% Supported input types:
% - numeric vector
% - char/string/categorical containing numeric text (e.g. "[1 2 3]", "1,2,3")
% - scalar cell containing any of the above
%
% This function is a safe replacement for str2num/ eval-based parsing.
%
% Inputs
% ------
% x
%   Value to parse.
% expected_len
%   If >0, enforce that the parsed vector has exactly this length.
% label
%   String used for error messages.
%
% Output
% ------
% v
%   1xN double row vector.
%

if nargin < 2 || isempty(expected_len)
    expected_len = 0;
end
if nargin < 3 || isempty(label)
    label = 'value';
end

% Unwrap scalar cell.
if iscell(x)
    if numel(x) ~= 1
        error('%s: expected a scalar cell, got %d elements.', label, numel(x));
    end
    x = x{1};
end

if isnumeric(x)
    v = double(x(:).');
else
    s = string(x);
    s = strtrim(s);

    if ismissing(s) || strlength(s) == 0
        v = [];
    else
        tokens = regexp(char(s), '[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', 'match');
        if isempty(tokens)
            error('%s: could not parse any numeric values from "%s".', label, s);
        end
        v = str2double(tokens);
        v = double(v(:).');
    end
end

if expected_len > 0
    if numel(v) ~= expected_len
        error('%s: expected %d numeric value(s), got %d.', label, expected_len, numel(v));
    end
end

if any(~isfinite(v))
    error('%s: contains NaN/Inf which is not allowed.', label);
end

end

% ========================================================================
function v = parse_numeric_scalar(x, label)
%LDBS_PARSE_NUMERIC_SCALAR Parse a single numeric scalar from heterogeneous inputs.
%
% v = parse_numeric_scalar(x, label)
%
% This is a convenience wrapper around parse_numeric_rowvector(..., 1).
%

if nargin < 2 || isempty(label)
    label = 'value';
end

v = parse_numeric_rowvector(x, 1, label);
v = v(1);

end
