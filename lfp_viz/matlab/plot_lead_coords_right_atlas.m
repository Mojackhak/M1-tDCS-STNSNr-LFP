function outputs = plot_lead_coords_right_atlas(csv_file, output_dir, leaddbs_root)
%PLOT_LEAD_COORDS_RIGHT_ATLAS Plot right-normalized coordinates in Lead-DBS.
%   Supply an enriched cohort CSV, a fresh output directory, and Lead-DBS root.

    narginchk(3, 3);
    csv_file = char(string(csv_file));
    output_dir = char(string(output_dir));

    if ~isfile(csv_file)
        error('Input CSV does not exist: %s', csv_file);
    end
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    fig_path = fullfile(output_dir, 'lead_coords_right_atlas.fig');
    png_path = fullfile(output_dir, 'lead_coords_right_atlas.png');
    existing_outputs = string({fig_path, png_path});
    existing_outputs = existing_outputs(isfile(existing_outputs));
    if ~isempty(existing_outputs)
        error('Refusing to overwrite existing outputs: %s', ...
            strjoin(existing_outputs, ', '));
    end

    T = readtable(csv_file, 'VariableNamingRule', 'preserve');
    required_columns = {
        'mni_x_flip', 'mni_y_flip', 'mni_z_flip', 'Region'
    };
    missing_columns = setdiff(required_columns, T.Properties.VariableNames);
    if ~isempty(missing_columns)
        error('Missing required columns: %s', strjoin(missing_columns, ', '));
    end

    coords = [
        ensureNumericColumn(T.mni_x_flip, 'mni_x_flip'), ...
        ensureNumericColumn(T.mni_y_flip, 'mni_y_flip'), ...
        ensureNumericColumn(T.mni_z_flip, 'mni_z_flip')
    ];
    regions = string(T.Region);
    allowed_regions = ["STN", "SNr", "Mid", "EXT"];
    if any(ismissing(regions) | ~ismember(regions, allowed_regions))
        error('Region contains a missing or unsupported value.');
    end
    if any(coords(:, 1) <= 0)
        error('Right-side visualization requires positive mni_x_flip values.');
    end

    addpath(genpath(leaddbs_root));
    addpath(fileparts(mfilename('fullpath')), '-begin');
    atlas_name = 'Custom_Ewert_Zhang_Middlebrooks0.05';
    atlas_path = fullfile(leaddbs_root, 'templates', 'space', ...
        'MNI152NLin2009bAsym', 'atlases', atlas_name, 'atlas_index.mat');
    if ~isfile(atlas_path)
        error('Atlas index does not exist: %s', atlas_path);
    end

    atlas_data = load(atlas_path, 'atlases');
    atlases = atlas_data.atlases;
    if ~strcmp(string(atlases.names{13}), "SNr.nii.gz") || ...
            ~strcmp(string(atlases.names{14}), "STN.nii.gz")
        error('Atlas ROI 13/14 do not resolve to SNr/STN.');
    end

    right_snr = reducepatch(atlases.roi{13, 1}.fv, 0.5);
    right_stn = reducepatch(atlases.roi{14, 1}.fv, 0.5);
    if mean(right_snr.vertices(:, 1)) <= 0 || ...
            mean(right_stn.vertices(:, 1)) <= 0
        error('Atlas ROI surfaces are not in the right hemisphere.');
    end

    stn_color = hexToRgb('#0E6AAF');
    snr_color = hexToRgb('#F2000E');
    other_color = hexToRgb('#E5E5E5');
    snr_wire_color = [1, 0.5020, 0];
    stn_wire_color = [0, 0.7843, 0];
    sphere_radius = 0.25;
    sphere_resolution = 24;

    h_fig = ea_mnifigure(atlas_name);
    h_ax = h_fig.CurrentAxes;
    set(h_fig, ...
        'Color', [1, 1, 1], ...
        'InvertHardcopy', 'off', ...
        'Position', [100, 100, 960, 720], ...
        'Visible', 'on');
    set(h_ax, 'Color', [1, 1, 1]);
    set(findall(h_ax, 'Type', 'patch'), 'Visible', 'off');
    set(findall(h_ax, 'Type', 'surface'), 'Visible', 'off');
    hold(h_ax, 'on');

    patch(h_ax, ...
        'Faces', right_snr.faces, ...
        'Vertices', right_snr.vertices, ...
        'FaceColor', 'none', ...
        'EdgeColor', snr_wire_color, ...
        'EdgeAlpha', 0.4, ...
        'LineWidth', 0.5, ...
        'Tag', 'lead_coords_right_snr_wireframe');
    patch(h_ax, ...
        'Faces', right_stn.faces, ...
        'Vertices', right_stn.vertices, ...
        'FaceColor', 'none', ...
        'EdgeColor', stn_wire_color, ...
        'EdgeAlpha', 0.4, ...
        'LineWidth', 0.5, ...
        'Tag', 'lead_coords_right_stn_wireframe');

    stn_mask = regions == "STN";
    snr_mask = regions == "SNr";
    other_mask = regions == "Mid" | regions == "EXT";
    addSpheres(h_ax, coords(stn_mask, :), sphere_radius, ...
        sphere_resolution, stn_color, 'lead_coords_stn_spheres');
    addSpheres(h_ax, coords(snr_mask, :), sphere_radius, ...
        sphere_resolution, snr_color, 'lead_coords_snr_spheres');
    addSpheres(h_ax, coords(other_mask, :), sphere_radius, ...
        sphere_resolution, other_color, 'lead_coords_other_spheres');

    atlas_view = struct();
    atlas_view.az = 0.8823;
    atlas_view.el = 0.1224;
    atlas_view.camva = 0.5000;
    atlas_view.camup = [0, 0, 1];
    atlas_view.camproj = 'orthographic';
    atlas_view.camtarget = [9.6266, -16.6089, -12.8113];
    atlas_view.campos = [-842.3468, -1.6138e+03, 474.5999];
    set(h_fig, 'CurrentAxes', h_ax);
    ea_view(atlas_view);

    triad = ea_add_ras_triad(h_ax, ...
        'FontSize', 27, ...
        'LineWidth', 5, ...
        'HeadSize', 1.0, ...
        'BackgroundColor', [1, 1, 1], ...
        'BackgroundAlpha', 0);
    triad.refresh();

    drawnow;
    savefig(h_fig, fig_path);
    print(h_fig, png_path, '-dpng', '-r300', '-image');

    outputs = struct( ...
        'figure_path', fig_path, ...
        'png_path', png_path, ...
        'row_count', height(T), ...
        'stn_count', sum(stn_mask), ...
        'snr_count', sum(snr_mask), ...
        'other_count', sum(other_mask), ...
        'sphere_radius_mm', sphere_radius, ...
        'sphere_resolution', sphere_resolution);

    fprintf('Rows: %d\n', outputs.row_count);
    fprintf('STN spheres: %d\n', outputs.stn_count);
    fprintf('SNr spheres: %d\n', outputs.snr_count);
    fprintf('Mid/EXT spheres: %d\n', outputs.other_count);
    fprintf('FIG: %s\n', fig_path);
    fprintf('PNG: %s\n', png_path);
end

function values = ensureNumericColumn(values, column_name)
%ENSURENUMERICCOLUMN Parse one finite numeric CSV column.
    if isnumeric(values)
        values = double(values);
    else
        values = str2double(string(values));
    end
    values = values(:);
    if any(~isfinite(values))
        error('Column %s contains a non-finite numeric value.', column_name);
    end
end

function rgb = hexToRgb(hex_color)
%HEXTORGB Convert one six-digit hexadecimal color to normalized RGB.
    hex_color = char(erase(string(hex_color), '#'));
    if numel(hex_color) ~= 6
        error('Expected a six-digit hexadecimal color.');
    end
    rgb = double(sscanf(hex_color, '%2x%2x%2x', [1, 3])) / 255;
end

function h_patch = addSpheres(h_ax, centers, radius, resolution, color, tag)
%ADDSPHERES Add equally sized sphere meshes to one axes as a single patch.
    [x, y, z] = sphere(resolution);
    [base_faces, base_vertices] = surf2patch( ...
        x * radius, y * radius, z * radius, 'triangles');
    vertex_count = size(base_vertices, 1);
    face_count = size(base_faces, 1);
    sphere_count = size(centers, 1);
    vertices = zeros(vertex_count * sphere_count, 3);
    faces = zeros(face_count * sphere_count, 3);

    for sphere_index = 1:sphere_count
        vertex_rows = (sphere_index - 1) * vertex_count + (1:vertex_count);
        face_rows = (sphere_index - 1) * face_count + (1:face_count);
        vertices(vertex_rows, :) = base_vertices + centers(sphere_index, :);
        faces(face_rows, :) = base_faces + (sphere_index - 1) * vertex_count;
    end

    h_patch = patch(h_ax, ...
        'Faces', faces, ...
        'Vertices', vertices, ...
        'FaceColor', color, ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 1, ...
        'FaceLighting', 'gouraud', ...
        'AmbientStrength', 0.35, ...
        'DiffuseStrength', 0.65, ...
        'SpecularStrength', 0.15, ...
        'Tag', tag);
end
