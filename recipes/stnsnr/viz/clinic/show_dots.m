assert(~isempty(getenv('MYLFP_PROJECT_ROOT')), 'Set MYLFP_PROJECT_ROOT before running this script.');

ea_mnifigure(); % open up Elvis viewer
hFig = gcf;
set(hFig, 'Color', 'w', 'InvertHardcopy', 'off');
set(findall(hFig, 'Type', 'axes'), 'Color', 'none');
load([ea_space([],'atlases'),'Custom_Ewert_Zhang_Middlebrooks0.05',filesep,'atlas_index.mat']); % manually load definition of atlas.
rSNr=atlases.roi{13,1}.fv; % extract the right SNr.
rSNr=reducepatch(rSNr,0.5); % reduce patch a bit.
rh_rSNr=patch('Faces',rSNr.faces,'Vertices',rSNr.vertices,'facecolor','none','edgecolor',[1,0.5020,0],'EdgeAlpha', 0.4); % visualize the right SNr as wireframes.
rSTN=atlases.roi{14,1}.fv; % do the same for the left SNr.
rSTN=reducepatch(rSTN,0.5);
rh_rSTN=patch('Faces',rSTN.faces,'Vertices',rSTN.vertices,'facecolor','none','edgecolor',[0,0.7843,0],'EdgeAlpha', 0.4);

% add MER coords
df = readtable(fullfile(getenv('MYLFP_PROJECT_ROOT'), 'summary/stats/clinic/coords/scale/scale_contact_only.xlsx'));
coords = [df.MNI_x_flip, df.MNI_y_flip, df.MNI_z_flip];
if ~ismember('ULF', df.Properties.VariableNames)
    error('scale_contact_only.xlsx must contain a ULF column.');
end
ulf = df.ULF;
if islogical(ulf)
    ulf = ulf(:);
elseif isnumeric(ulf)
    ulf = ulf(:) ~= 0;
else
    ulf_text = lower(strtrim(string(ulf(:))));
    ulf = ulf_text == "true" | ulf_text == "1";
end
% mask = strcmp(df.SNr_in, 'TRUE');
% coords(~mask,:) = [];
% coords(any(isnan(coords),2),:) = [];
radius = 0.25;
n = 100;
color_false = [14, 106, 175] / 255; % #0E6AAF
color_true = [242, 0, 14] / 255;    % #F2000E
h = gobjects(0);
if any(~ulf)
    h(end + 1) = add_spheres(coords(~ulf, :), radius, n, 'FaceColor', color_false);
end
if any(ulf)
    h(end + 1) = add_spheres(coords(ulf, :), radius, n, 'FaceColor', color_true);
end


% SNr view 1
vSNr1 = struct();
vSNr1.az        = 0.8823;
vSNr1.el        = 0.1224;
vSNr1.camva     = 0.2886;
vSNr1.camup     = [0 0 1];
vSNr1.camproj   = 'orthographic';
vSNr1.camtarget = [10.0692 -16.3838 -11.3000];
vSNr1.campos    = [-841.9042 -1.6136e+03 476.1112];

% % SNr view 2
% vSNr2 = struct();
% vSNr2.az        = -0.8227;
% vSNr2.el        = -0.1286;
% vSNr2.camva     = 0.2886;
% vSNr2.camup     = [0 0 1];
% vSNr2.camproj   = 'orthographic';
% vSNr2.camtarget = [9.1244 -16.2821 -12.1207];
% vSNr2.campos    = [1.0473e+03 1.4861e+03 411.8152];

ea_view(vSNr1)

% set(h, 'FaceAlpha', 0);   % fully transparent (effectively hidden)
% set(rh, 'edgeAlpha', 0);   % fully transparent (effectively hidden)
% set(lh, 'edgeAlpha', 0);   % fully transparent (effectively hidden)
