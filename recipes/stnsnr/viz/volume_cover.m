assert(~isempty(getenv('MYLFP_PROJECT_ROOT')), 'Set MYLFP_PROJECT_ROOT before running this script.');

%% Plot Selection
clear;clc;close("all");
section = "rest";
var_idx = 1;
mode_idx = 2;
region_idx = 2;
view_idx = 2;

%% File config
project = getenv('MYLFP_PROJECT_ROOT');
stats_dir = fullfile(project, 'summary', 'stats');
sections = struct();
sections.rest = ["Sit"];
sections.med = ["On"];
sections.walk = ["Stand", "Walk", "Turn"];
sections.pain = ["Pain"];

modes = ["single", "double"];
regions = ["SNr", "STN"];
vars = sections.(section);
var = vars(var_idx);
mode = modes(mode_idx);
region = regions(region_idx);
data_dir = fullfile(stats_dir, section, 'topomap', 'cover', mode, var, region);
data_path = fullfile(data_dir, 'contact_locs.nii.gz');
df_contact_path = fullfile(data_dir, 'contact_locs.csv');
df_contact = readtable(df_contact_path);

%% func config
cmap = struct();
cmap.SNr = [0.80, 0.80, 0.80;
            0.80, 0.80, 0.80];
cmap.STN = [0.80, 0.80, 0.80;
            0.80, 0.80, 0.80];

patchObj = ea_nifti2patch(data_path, ...
    'TemplateNifti', [], ...
    'SurfaceMode', 'mask', ...
    'MaskThreshold', -9999, ...
    'MaskIsovalue', 0.5, ...
    'VolumeSmoothingSigmaMm', 0.0, ...
    'SurfaceSmoothingIters', 20, ...
    'SurfaceSmoothingMethod', 'taubin', ...
    'SurfaceSmoothingLambda', 0.5, ...
    'SurfaceSmoothingMu', -0.1, ...
    'Alpha', 0.3, ...
    'Colormap', cmap.(region), ...
    'CLimMode', 'auto', ...
    'GeometryUpsampleFactor', 1, ...
    'ColorSampling', 'inside', ...
    'SampleDepthMm', 0.5, ...
    'ReduceFactor', 1);

fprintf('Created patch: %d faces, %d vertices. GeometrySource=%s, MaskThreshold=%.3f\n', ...
    size(patchObj.faces,1), size(patchObj.vertices,1), patchObj.geometry_source, patchObj.mask_threshold);

%% Plot fig
% Original SNr view 
vSNr1 = struct();
vSNr1.az        = 0.8823;
vSNr1.el        = 0.1224;
vSNr1.camva     = 0.5000;
vSNr1.camup     = [0 0 1];  % forced change
vSNr1.camproj   = 'orthographic';
vSNr1.camtarget = [9.6266 -16.6089 -12.8113];
vSNr1.campos    = [-842.3468 -1.6138e+03 474.5999];

% Added SNr view (provided)
vSNr2 = struct();
vSNr2.az        = -0.8227;
vSNr2.el        = -0.1286;
vSNr2.camva     = 0.5000;
vSNr2.camup     = [0 0 1];
vSNr2.camproj   = 'orthographic';
vSNr2.camtarget = [9.1244 -16.2821 -12.1207];
vSNr2.campos    = [1.0473e+03 1.4861e+03 411.8152];

% Original STN view (preserved, but camup forced to [0 0 1])
vSTN1 = struct();
vSTN1.az        = 0.8823;
vSTN1.el        = 0.1224;
vSTN1.camva     = 0.3500;
vSTN1.camup     = [0 0 1];  % forced change
vSTN1.camproj   = 'orthographic';
vSTN1.camtarget = [9.8725 -14.8273 -6.5437];
vSTN1.campos    = [-841.1497 -1.6125e+03 481.0182];

% Added STN view (provided)
vSTN2 = struct();
vSTN2.az        = -0.8227;
vSTN2.el        = -0.1286;
vSTN2.camva     = 0.3500;
vSTN2.camup     = [0 0 1];
vSTN2.camproj   = 'orthographic';
vSTN2.camtarget = [8.5895 -16.9006 -8.6190];
vSTN2.campos    = [1.0468e+03 1.4855e+03 415.3169];


% Store as view lists (cell arrays)
v = struct();
v.STN = {vSTN1, vSTN2};
v.SNr = {vSNr1, vSNr2};
views = [1,2];
view = views(view_idx);
regioni = region + num2str(view);
outPng = fullfile(data_dir, ['cover', num2str(view),  '.png']);
key = '\beta\mbox{-low}';   % shows as "β-low" (not subscript)
clabel = 'test';
clabel = strrep(clabel, '<KEY>', key);


fontsize = 32;
tickfontsize = 27;

[hFig, hAx, hPatch, triad, extrema, hCb] = ea_plot_patch_leaddbs(patchObj, ...
    'MissingDataMode', 'gray', ...
    'MissingDataBlendRings', 0, ...
    'ViewStruct', v.(region){view}, ...
    'AddColorbar', true, ...
    'ColorbarUnits', 'normalized', ...
    'ColorbarPosition', [0.84 0.20 0.03 0.60], ...
    'ColorbarLabel', clabel, ...
    'ColorbarLabelFontSize', fontsize, ...
    'ColorbarTickFontSize', tickfontsize, ...
    'ColorbarLabelOffset', 0.6, ...
    'MarkExtrema', false, ...
    'RASTriadFontSize', tickfontsize, ...
    'RASTriadLineWidth', 5, ...
    'RASTriadHeadSize', 1.0, ...
    'ExportFile', '', ...
    'ExportResolution', 600);

%% plot contact

load([ea_space([],'atlases'),'Custom_Ewert_Zhang_Middlebrooks0.05',filesep,'atlas_index.mat']); % manually load definition of atlas.
atlas_idx = struct();
atlas_idx.SNr = 13;
atlas_idx.STN = 14;
roi=atlases.roi{atlas_idx.(region),1}.fv; % extract the right roi
roi=reducepatch(roi,0.5); % reduce patch a bit.
rh=patch('Faces',roi.faces,'Vertices',roi.vertices, ...
    'facecolor','none','edgecolor',[0.30, 0.30, 0.30], 'EdgeAlpha', 0.4); % visualize the right roi as wireframes.
coords = [df_contact.MNI_x_flip, df_contact.MNI_y_flip, df_contact.MNI_z_flip];
radius = 0.127;
n = 100;
color = [0.95, 0.55, 0.20];
h = add_spheres(coords, radius, n, 'FaceColor', color);

%% save fig
% drawnow;
% ea_export_figure_transparent(hFig, outPng, ...
%     'Transparent', false, ...
%     'BackgroundColor', [1 1 1], ...
%     'Resolution', 600, ...
%     'Renderer', 'opengl');

% Capture the current view parameters (useful after manual interaction):
% try
%     vCaptured = ea_capture_view_struct(hAx);
%     disp('Captured view struct (Lead-DBS ea_view compatible):');
%     disp(vCaptured);
% catch
% end
