function new_data = flip_coords_lr_nonlinear(input_data, direction, options)
%FLIP_COORDS_LR_NONLINEAR Nonlinear L/R flip of MNI coordinates using Lead-DBS.
%
% This function flips MNI coordinates across the midline using Lead-DBS's
% ea_flip_lr_nonlinear(). It can operate on a MATLAB table or a CSV file path.
%
% Inputs
% ------
% input_data
%   Either:
%     - table containing MNI coordinate columns, or
%     - path to a CSV file readable by readtable().
%
% direction
%   'L2R' : flip only rows with x <= 0 (left hemisphere) into the right.
%   'R2L' : flip only rows with x  > 0 (right hemisphere) into the left.
%
% Name-Value Options
% ------------------
% OutCsv (char)            : If non-empty, write output table to this CSV path.
% WriteNewColumns (logical): If true, append *_flip columns instead of overwriting.
% Suffix (char)            : Suffix for new columns when WriteNewColumns=true (default: '_flip').
% XColumn (char)           : Preferred x column name (default: 'MNI_x').
% YColumn (char)           : Preferred y column name (default: 'MNI_y').
% ZColumn (char)           : Preferred z column name (default: 'MNI_z').
%
% Output
% ------
% new_data
%   Output table with flipped coordinates.
%
% Dependency
% ----------
% Requires Lead-DBS function ea_flip_lr_nonlinear on MATLAB path.
%

arguments
    input_data
    direction (1,:) char
    options.OutCsv (1,:) char = ''
    options.WriteNewColumns (1,1) logical = false
    options.Suffix (1,:) char = '_flip'
    options.XColumn (1,:) char = 'MNI_x'
    options.YColumn (1,:) char = 'MNI_y'
    options.ZColumn (1,:) char = 'MNI_z'
end

if ~ismember(upper(direction), {'L2R', 'R2L'})
    error('direction must be "L2R" or "R2L".');
end

require_functions({"ea_flip_lr_nonlinear"});

T = read_input_as_table(input_data);

x_var = table_find_var(T, {options.XColumn});
y_var = table_find_var(T, {options.YColumn});
z_var = table_find_var(T, {options.ZColumn});

x = ensure_numeric_vector(T.(x_var), x_var);
y = ensure_numeric_vector(T.(y_var), y_var);
z = ensure_numeric_vector(T.(z_var), z_var);

coords = [x, y, z];

finite_mask = all(isfinite(coords), 2);
switch upper(direction)
    case 'L2R'
        flip_mask = finite_mask & (coords(:,1) <= 0);
    case 'R2L'
        flip_mask = finite_mask & (coords(:,1) > 0);
end

coords_flipped = coords;
if any(flip_mask)
    coords_flipped(flip_mask, :) = ea_flip_lr_nonlinear(coords(flip_mask, :), [], 4);
else
    warning('No rows matched the flip mask; no coordinates were changed.');
end

if options.WriteNewColumns
    T.("MNI_x" + string(options.Suffix)) = coords_flipped(:,1);
    T.("MNI_y" + string(options.Suffix)) = coords_flipped(:,2);
    T.("MNI_z" + string(options.Suffix)) = coords_flipped(:,3);
else
    T.(x_var) = coords_flipped(:,1);
    T.(y_var) = coords_flipped(:,2);
    T.(z_var) = coords_flipped(:,3);
end

new_data = T;

if ~isempty(options.OutCsv)
    writetable(new_data, options.OutCsv, ...
        'FileType', 'text', ...
        'Encoding', 'UTF-8', ...
        'WriteVariableNames', true, ...
        'QuoteStrings', true);
end

end

% ========================================================================
function T = read_input_as_table(input_data)
% Read input as a table.

if istable(input_data)
    T = input_data;
    return;
end

if isstring(input_data) || ischar(input_data)
    csv_path = char(input_data);
    if ~isfile(csv_path)
        error('CSV file not found: %s', csv_path);
    end
    T = readtable(csv_path, 'VariableNamingRule', 'preserve');
    return;
end

error('input_data must be a table or a CSV file path.');

end

% ========================================================================
function v = ensure_numeric_vector(v, label)
% Ensure a column vector is numeric (double). Converts strings/cells/categorical.

if isnumeric(v)
    v = double(v(:));
    return;
end

try
    v = str2double(string(v));
catch
    error('Failed to convert column "%s" to numeric.', label);
end

v = double(v(:));

end
