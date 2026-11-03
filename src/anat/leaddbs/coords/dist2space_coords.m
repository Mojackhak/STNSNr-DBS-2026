function coords = dist2space_coords(subj_folder, dist_by_side, offset_by_side, options)
%DIST2SPACE_COORDS Convert DBS lead axial distances into native/SCRF/MNI coordinates.
%
% This function converts distances measured along a reconstructed DBS lead
% (relative to the lead "head" marker) into 3D coordinates.
%
% Inputs
% ------
% subj_folder
%   Subject folder managed by Lead-DBS (e.g., .../leaddbs/sub-001).
%
% dist_by_side
%   Cell array indexed by Lead-DBS side index (typically 1=R, 2=L).
%   Each cell contains a numeric vector of distances in millimeters.
%   Distances are applied along the lead axis defined by (head -> tail).
%
% offset_by_side
%   Cell array indexed by side. Each cell is a 1x3 numeric vector:
%       [x_mm, y_mm, depth_mm]
%   where x/y are uniform native-space shifts (RAS+ convention) and depth
%   is an axial shift applied to all distances (along head->tail).
%
% Name-Value Options
% ------------------
% SavePath (char)             : If non-empty, save the output struct to this path.
% MarkerSpace (char)          : Must be 'native' (kept for API stability; no other spaces supported).
% TransformFolderName (char)  : Folder under opt.subj.subjDir containing deformation fields
%                               for ea_map_coords (default: 'inverseTransform').
% Verbose (logical)           : Print progress messages (default: true).
%
% Output
% ------
% coords (struct)
%   .native     {side} Nx3 native coordinates (mm)
%   .scrf       {side} Nx3 SCRF coordinates (mm) (brainshift affine if available)
%   .mni        {side} Nx3 MNI coordinates (mm)
%   .dist    {side} Nx1 distances provided (mm)
%   .offset  {side} 1x3 offsets used (mm)
%   .axis       struct with head/tail/dir_unit in native space
%   .meta       struct with provenance and file paths
%
% Dependencies
% ------------
% Requires Lead-DBS on MATLAB path:
%   ea_getptopts, ea_load_nii, ea_map_coords
%
% Notes
% -----
% - This function intentionally does NOT support legacy offset formats.
% - All coordinates are assumed to be in millimeters.
%

arguments
    subj_folder (1,:) char
    dist_by_side (1,:) cell
    offset_by_side (1,:) cell = {}
    options.SavePath (1,:) char = ''
    options.MarkerSpace (1,:) char = 'native'
    options.TransformFolderName (1,:) char = 'inverseTransform'
    options.Verbose (1,1) logical = true
end

require_functions({"ea_getptopts", "ea_load_nii", "ea_map_coords"});

if options.Verbose
    fprintf('[%s] dist2space_coords: loading Lead-DBS options...\n', string(datetime()));
end
opt = ea_getptopts(subj_folder);

sides = opt.sides;
n_sides = max(sides);

% Normalize inputs into dense cell arrays indexed by side.
dist_cell = normalize_dist_cell(dist_by_side, n_sides, sides);
offset_cell = normalize_offset_cell(offset_by_side, n_sides, sides);

% Load Lead-DBS reconstruction (lead markers).
[reco, reco_path] = load_reco(opt);

if ~strcmpi(options.MarkerSpace, 'native')
    error('Only MarkerSpace="native" is supported. Got: %s', options.MarkerSpace);
end

marker_space = 'native';
validate_native_markers(reco);

% Compute native coordinates.
[native_cell, axis_info] = distances_to_native_coords(reco, sides, dist_cell, offset_cell);

% Apply SCRF affine (brainshift) if available.
[scrf_cell, scrf_affine_path] = apply_scrf_affine_if_present(native_cell, sides, opt);

% Warp SCRF -> MNI (nonlinear deformation fields).
[mni_cell, anat_nii_path, transform_folder] = warp_scrf_to_mni(scrf_cell, sides, opt, options.TransformFolderName);

% Assemble output.
coords = struct();
coords.native = native_cell;
coords.scrf = scrf_cell;
coords.mni = mni_cell;
coords.dist = dist_cell;
coords.offset = offset_cell;
coords.axis = axis_info;
coords.meta = struct();
coords.meta.created_at = string(datetime());
coords.meta.units = "mm";
coords.meta.subj_folder = subj_folder;
coords.meta.reco_path = reco_path;
coords.meta.marker_space = marker_space;
coords.meta.scrf_affine_path = scrf_affine_path;
coords.meta.anat_nii_path = anat_nii_path;
coords.meta.transform_folder = transform_folder;
coords.meta.leaddbs_sides = sides;
coords.meta.native_axis = "RAS+ (Lead-DBS native space)";

% Optional save.
if ~isempty(options.SavePath)
    if options.Verbose
        fprintf('[%s] dist2space_coords: saving result...\n', string(datetime()));
    end
    save(options.SavePath, 'coords');
end

if options.Verbose
    fprintf('[%s] dist2space_coords: done.\n', string(datetime()));
end

end

% ========================================================================
function dist_cell = normalize_dist_cell(dist_by_side, n_sides, sides)
% Ensure a cell array indexed by side, with Nx1 double vectors.

validateattributes(dist_by_side, {'cell'}, {'vector'}, mfilename, 'dist_by_side');

dist_cell = cell(1, n_sides);
for i = 1:n_sides
    dist_cell{i} = [];
end

for side = sides
    if numel(dist_by_side) >= side && ~isempty(dist_by_side{side})
        v = dist_by_side{side};
        validateattributes(v, {'numeric'}, {'vector', 'real', 'finite'}, mfilename, sprintf('dist_by_side{%d}', side));
        dist_cell{side} = double(v(:));
    else
        dist_cell{side} = zeros(0,1);
    end
end

end

% ========================================================================
function offset_cell = normalize_offset_cell(offset_by_side, n_sides, sides)
% Ensure a cell array indexed by side, with 1x3 double offsets.

validateattributes(offset_by_side, {'cell'}, {'vector'}, mfilename, 'offset_by_side');

offset_cell = cell(1, n_sides);
for i = 1:n_sides
    offset_cell{i} = [0, 0, 0];
end

if isempty(offset_by_side)
    return;
end

for side = sides
    if numel(offset_by_side) >= side && ~isempty(offset_by_side{side})
        v = offset_by_side{side};
        validateattributes(v, {'numeric'}, {'vector', 'real', 'finite'}, mfilename, sprintf('offset_by_side{%d}', side));
        v = double(v(:).');
        if numel(v) ~= 3
            error('offset_by_side{%d} must be a 1x3 numeric vector [x y depth] in mm.', side);
        end
        offset_cell{side} = v;
    end
end

end

% ========================================================================
function [reco, reco_path] = load_reco(opt)
% Load Lead-DBS reconstruction file containing lead markers.

if ~isfield(opt, 'subj') || ~isfield(opt.subj, 'recon') || ~isfield(opt.subj.recon, 'recon')
    error('Lead-DBS options do not contain opt.subj.recon.recon.');
end

reco_path = opt.subj.recon.recon;
if ~isfile(reco_path)
    error('Lead-DBS reconstruction file not found: %s', reco_path);
end

s = load(reco_path, 'reco');
if ~isfield(s, 'reco')
    error('Reconstruction file does not contain variable "reco": %s', reco_path);
end
reco = s.reco;

end

% ========================================================================
function validate_native_markers(reco)
% Validate that reco.native.markers exists.

if ~isfield(reco, 'native')
    error('reco is missing required field "native".');
end
if ~isfield(reco.native, 'markers')
    error('reco.native is missing required field "markers".');
end

end

% ========================================================================
function [native_cell, axis_info] = distances_to_native_coords(reco, sides, dist_cell, offset_cell)
% Convert distances to native coordinates for each side.

native_cell = cell(1, max(sides));
axis_info = struct();
axis_info.head_native_mm = cell(1, max(sides));
axis_info.tail_native_mm = cell(1, max(sides));
axis_info.dir_unit_native = cell(1, max(sides));

for side = sides
    dist_vec = dist_cell{side};
    off = offset_cell{side};

    if isempty(dist_vec)
        native_cell{side} = zeros(0,3);
        axis_info.head_native_mm{side} = [NaN, NaN, NaN];
        axis_info.tail_native_mm{side} = [NaN, NaN, NaN];
        axis_info.dir_unit_native{side} = [NaN, NaN, NaN];
        continue;
    end

    [head_mm, tail_mm] = get_head_tail(reco, side);

    axis_vec = tail_mm - head_mm;
    axis_norm = norm(axis_vec);
    if axis_norm == 0
        error('Lead axis has zero length for side %d. Check markers in reconstruction.', side);
    end
    dir_unit = axis_vec ./ axis_norm; % head -> tail

    dist_with_offset = dist_vec + off(3);
    xyz_native = head_mm + dist_with_offset * dir_unit; % (N x 1) * (1 x 3) => N x 3

    % Uniform planar shifts in native RAS+.
    xyz_native(:,1) = xyz_native(:,1) + off(1);
    xyz_native(:,2) = xyz_native(:,2) + off(2);

    native_cell{side} = xyz_native;
    axis_info.head_native_mm{side} = head_mm;
    axis_info.tail_native_mm{side} = tail_mm;
    axis_info.dir_unit_native{side} = dir_unit;
end

end

% ========================================================================
function [head_mm, tail_mm] = get_head_tail(reco, side)
% Retrieve head/tail markers for a given side.

markers = reco.native.markers;

if numel(markers) < side
    error('Markers array has only %d entries; requested side index %d.', numel(markers), side);
end

m = markers(side);
if ~isfield(m, 'head') || ~isfield(m, 'tail')
    error('Markers for side %d do not contain required fields "head" and "tail".', side);
end

head_mm = double(m.head(:).');
tail_mm = double(m.tail(:).');

if numel(head_mm) ~= 3 || numel(tail_mm) ~= 3
    error('Head and tail markers must be 1x3 vectors for side %d.', side);
end

end

% ========================================================================
function [scrf_cell, scrf_affine_path] = apply_scrf_affine_if_present(native_cell, sides, opt)
% Apply brainshift affine transform (SCRF) if Lead-DBS provides it.

scrf_affine_path = '';
scrf_cell = native_cell;

if isfield(opt, 'subj') && isfield(opt.subj, 'brainshift') && isfield(opt.subj.brainshift, 'transform') && isfield(opt.subj.brainshift.transform, 'scrf')
    scrf_affine_path = opt.subj.brainshift.transform.scrf;
end

if isempty(scrf_affine_path) || ~isfile(scrf_affine_path)
    return;
end

d = load(scrf_affine_path);
if ~isfield(d, 'mat')
    error('SCRF affine file does not contain variable "mat": %s', scrf_affine_path);
end
A = double(d.mat);
if ~isequal(size(A), [4 4])
    error('SCRF affine matrix must be 4x4: %s', scrf_affine_path);
end

for side = sides
    xyz = native_cell{side};
    if isempty(xyz)
        scrf_cell{side} = zeros(0,3);
        continue;
    end
    scrf_cell{side} = apply_affine_4x4(A, xyz);
end

end

% ========================================================================
function out = apply_affine_4x4(A, xyz)
% Apply 4x4 affine matrix to Nx3 coordinates.

n = size(xyz, 1);
tmp = (A * [xyz, ones(n,1)]')';
out = tmp(:,1:3);

end

% ========================================================================
function [mni_cell, anat_nii_path, transform_folder] = warp_scrf_to_mni(scrf_cell, sides, opt, transform_folder_name)
% Warp SCRF coordinates to MNI using Lead-DBS deformation fields.

if ~isfield(opt, 'subj') || ~isfield(opt.subj, 'coreg') || ~isfield(opt.subj.coreg, 'anat') || ~isfield(opt.subj.coreg.anat, 'preop')
    error('Lead-DBS options missing opt.subj.coreg.anat.preop.');
end
if ~isfield(opt.subj, 'AnchorModality')
    error('Lead-DBS options missing opt.subj.AnchorModality.');
end

anchor = opt.subj.AnchorModality;
if ~isfield(opt.subj.coreg.anat.preop, anchor)
    error('Anchor modality "%s" not found in opt.subj.coreg.anat.preop.', anchor);
end

anat_nii_path = opt.subj.coreg.anat.preop.(anchor);
nii = ea_load_nii(anat_nii_path);

if ~isfield(nii, 'mat') || ~isfield(nii, 'fname')
    error('ea_load_nii output missing required fields "mat" and/or "fname".');
end

if ~isfield(opt.subj, 'subjDir')
    error('Lead-DBS options missing opt.subj.subjDir.');
end

transform_folder = fullfile(opt.subj.subjDir, transform_folder_name);
% if ~isfolder(transform_folder)
%     error('Transform folder not found: %s', transform_folder);
% end

mni_cell = scrf_cell;
for side = sides
    xyz = scrf_cell{side};
    if isempty(xyz)
        mni_cell{side} = zeros(0,3);
        continue;
    end
    mni_cell{side} = warp_mm_coords_to_mni_mm(xyz, nii, transform_folder);
end

end

% ========================================================================
function mni_mm = warp_mm_coords_to_mni_mm(xyz_mm, nii, transform_folder)
% Wrapper around ea_map_coords: mm -> vox -> deformation -> mm.

n = size(xyz_mm, 1);
vox = (nii.mat \ [xyz_mm, ones(n,1)]')';
vox = vox(:,1:3);

% ea_map_coords expects 3xN coordinates.
def = ea_map_coords(vox', nii.fname, transform_folder, '');

mni_mm = def';

end
