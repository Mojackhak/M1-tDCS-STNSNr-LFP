function var_name = table_find_var(T, candidates)
%LDBS_TABLE_FIND_VAR Find a table variable name using case-insensitive matching.
%
% var_name = table_find_var(T, candidates)
%
% Inputs
% ------
% T
%   MATLAB table.
% candidates
%   Cell array or string array of candidate column names. The first
%   case-insensitive match is returned.
%
% Output
% ------
% var_name
%   The actual variable name in the table (preserves original capitalization).
%

if ~istable(T)
    error('T must be a table.');
end

cand = string(candidates);
if isempty(cand)
    error('candidates must be non-empty.');
end

names = T.Properties.VariableNames;
for i = 1:numel(cand)
    idx = find(strcmpi(names, cand(i)), 1);
    if ~isempty(idx)
        var_name = names{idx};
        return;
    end
end

error('Could not find any of the candidate columns: %s', strjoin(cand, ', '));

end
