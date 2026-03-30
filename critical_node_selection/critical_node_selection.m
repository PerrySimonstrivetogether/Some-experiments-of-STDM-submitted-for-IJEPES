function critical_node_selection()
% critical_node_selection
% Public-facing compact version for observability-aware node ranking.
%
% Required inputs in the current folder:
%   IEEE33_adjacency.csv
%   IEEE37_adjacency.csv
%
% Required OPP functions on the MATLAB path:
%   OPP_DFS
%   OPP_GThN
%   OPP_RSN
%   OPP_SA
%   OPP_SAB
%   OPP_GThN_forbidden

base_dir = fileparts(mfilename('fullpath'));
addpath(base_dir);

systems = {'IEEE33', 'IEEE37'};

for s = 1:numel(systems)
    system_name = systems{s};
    fprintf('Processing %s...\n', system_name);
    A = readmatrix(fullfile(base_dir, sprintf('%s_adjacency.csv', system_name)));
    A = double(A ~= 0);
    n = size(A, 1);
    ZI_buses = [];

    alg_names = {};
    placements = {};

    [p, ~] = OPP_DFS(A);
    alg_names{end + 1} = 'DFS';
    placements{end + 1} = p(:)';

    [p, ~] = OPP_GThN(A, ZI_buses);
    alg_names{end + 1} = 'GThN';
    placements{end + 1} = p(:)';
    baseline_gthn = p(:)';

    [p, ~] = OPP_RSN(A, ZI_buses);
    alg_names{end + 1} = 'RSN';
    placements{end + 1} = p(:)';

    rng(20260324, 'twister');
    for r = 1:5
        [p, ~] = OPP_SA(A, ZI_buses);
        alg_names{end + 1} = sprintf('SA_%02d', r);
        placements{end + 1} = p(:)';
    end

    rng(20260325, 'twister');
    for r = 1:5
        [p, ~] = OPP_SAB(A, ZI_buses);
        alg_names{end + 1} = sprintf('SAB_%02d', r);
        placements{end + 1} = p(:)';
    end

    placement_count = numel(placements);
    selection_matrix = zeros(n, placement_count);
    for k = 1:placement_count
        selection_matrix(placements{k}, k) = 1;
    end

    select_frequency = mean(selection_matrix, 2);
    degree = sum(A, 2) - diag(A);
    baseline_pmu_count = numel(baseline_gthn);
    min_pmu_without = zeros(n, 1);
    pmu_penalty = zeros(n, 1);
    success_flag = zeros(n, 1);

    for node = 1:n
        [p_forbidden, msg] = OPP_GThN_forbidden(A, ZI_buses, node);
        success_flag(node) = double(msg.success);
        if msg.success
            min_pmu_without(node) = numel(p_forbidden);
            pmu_penalty(node) = numel(p_forbidden) - baseline_pmu_count;
        else
            min_pmu_without(node) = n + 1;
            pmu_penalty(node) = n;
        end
    end

    score = 0.55 * normalize_vector(pmu_penalty) + ...
            0.35 * normalize_vector(select_frequency) + ...
            0.10 * normalize_vector(degree);

    T = table((1:n)', degree, select_frequency, min_pmu_without, pmu_penalty, success_flag, score, ...
        'VariableNames', {'node', 'degree', 'selection_frequency', 'min_pmu_without_node', 'pmu_penalty', 'success_flag', 'importance_score'});
    T = sortrows(T, {'importance_score', 'pmu_penalty', 'selection_frequency', 'degree'}, {'descend', 'descend', 'descend', 'descend'});
    T.rank = (1:height(T))';
    T = movevars(T, 'rank', 'Before', 'node');

    writetable(T, fullfile(base_dir, sprintf('%s_ranking.csv', system_name)));
    fprintf('Saved %s_ranking.csv\n', system_name);
end
end

function y = normalize_vector(x)
x = double(x(:));
if all(abs(x - x(1)) < 1e-12)
    y = zeros(size(x));
else
    y = (x - min(x)) ./ (max(x) - min(x));
end
end
