function [ placement, msg ] = OPP_GThN_forbidden( A, ZI_buses, forbidden_nodes )
% Compact helper used for node criticality analysis.
% It re-runs GThN while forbidding PMU placement on selected nodes.

if nargin() < 2
    ZI_buses = [];
end
if nargin() < 3
    forbidden_nodes = [];
end

msg.method = 'GThN with forbidden nodes';
timer_val = tic;

num_buses = length(A);
X = zeros(num_buses, 1);
in_degree = full(sum(A, 2));
in_degree(forbidden_nodes) = -inf;

adj = cell(num_buses, 1);
for i = 1:num_buses
    A(i, i) = 0;
    adj{i} = find(A(i, :) ~= 0);
    A(i, i) = 1;
end

equ = cell(1, num_buses);
for i = 1:num_buses
    equ{i} = sprintf('f(%d)=X(%d)%s;', i, i, sprintf('+X(%d)', adj{i}));
end

for i = 1:length(ZI_buses)
    equ{ZI_buses(i)} = sprintf('%s+~any([%s]==0);', equ{ZI_buses(i)}(1:end-1), sprintf('f(%d) ', adj{ZI_buses(i)}));
    num_of_incidents = length(adj{ZI_buses(i)});
    for j = 1:num_of_incidents
        incidents = adj{ZI_buses(i)};
        incidents(j) = [];
        incidents(num_of_incidents) = ZI_buses(i);
        equ{adj{ZI_buses(i)}(j)} = sprintf('%s+~any([%s]==0);', equ{adj{ZI_buses(i)}(j)}(1:end-1), sprintf('f(%d) ', incidents));
    end
end

constraint_function = strjoin(equ, '\n');

function f = F(Xv)
    f = zeros(num_buses, 1);
    c_f = f;
    while true
        eval(constraint_function);
        if c_f == f
            break;
        else
            c_f = f;
        end
    end
end

msg.success = true;
for k = 1:num_buses
    [~, index] = max(in_degree);
    if isinf(in_degree(index))
        msg.success = false;
        break;
    end
    X(index) = 1;
    ob = F(X);
    if ~any(ob < 1)
        break;
    end
    in_degree(index) = -inf;
end

placement = find(transpose(X) == 1);
msg.time = toc(timer_val);
end
