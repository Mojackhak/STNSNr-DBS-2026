function data = read_json_file(json_path)
%READ_JSON_FILE Read and decode a JSON file into MATLAB data.
%
% data = read_json_file(json_path)
%
% Notes
% -----
% - MATLAB's jsondecode converts JSON object keys into valid MATLAB field names.
% - For lookup, use matlab.lang.makeValidName() on the original JSON key string.
%

arguments
    json_path (1,:) char
end

if ~isfile(json_path)
    error('JSON file not found: %s', json_path);
end

txt = fileread(json_path);

% Remove UTF-8 BOM if present.
if ~isempty(txt) && txt(1) == char(65279)
    txt = txt(2:end);
end

try
    data = jsondecode(txt);
catch ME
    error('Failed to decode JSON file "%s": %s', json_path, ME.message);
end

end
