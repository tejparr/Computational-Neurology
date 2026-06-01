function G = mp_POMDP_G(ind, dom)
% This function takes the indices of variables in a POMDP model and
% constructs a causal graph, making use of the domain factors as provided
%--------------------------------------------------------------------------
if ~isempty(ind.B)
    G = cell(ind.B(end) + length(ind.B),1);
else
    G = cell(length(ind.D) + length(ind.A),1);
end
% Connect hidden states at t = 1 to outcomes at t = 1
%--------------------------------------------------------------------------
for i = ind.A
    j = i + 1 - ind.A(1);
    if isfield(dom,'A') % If domains of A are specified, use these
        G{i} = [ind.D(dom.A(j).s) ind.E(dom.A(j).u)];
    else                % Otherwise, assume depends on all states
        G{i} = ind.D;
    end
end

% Connect hidden states and actions at t = 1 to states at t = 2
%--------------------------------------------------------------------------
for i = ind.B
    j = i + 1 - ind.B(1);
    if isfield(dom,'B') % If domains of B are specified, use these
        G{i} = [ind.D(j) ind.D(dom.B(j).s) ind.E(dom.B(j).u)];
    else  % Otherwise, assume depends on self states and all actions
        G{i} = [ind.D(j) ind.E];
    end
end

% Equip childless nodes with uninformative outcome
%--------------------------------------------------------------------------
for i = 1:length(ind.B)
    G{length(ind.A) + length(ind.B) + length(ind.D) + length(ind.E) + i} = ind.B(i);
end