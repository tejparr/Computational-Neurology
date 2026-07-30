function [o,s] = pomdp_default_gen(s,u,Qo,pomdp)
% Default generative process for pomdp scheme that assumes model and
% process match.
%--------------------------------------------------------------------------

if nargout > 1 % Advance the states
    S = s(:,end);
    for i = 1:numel(pomdp.B)
        ind          = [{':'}, {S(i)}, num2cell(S(pomdp.dom.B(i).s)'),num2cell(u(pomdp.dom.B(i).u,end)')];
        [~,s(i,end)] = max(pomdp.B{i}(ind{:}));
    end
end

o = zeros(numel(pomdp.A),1);

% Option for reflexive fulfilment of predictions
%--------------------------------------------------------------------------
if isfield(pomdp,'n')
    for i = find(pomdp.n)'
        if ~isempty(Qo)
            [~,o(i)] = max(Qo{i}); % Fulfill prediction
        else
            o(i) = 1;              % Or null
        end
    end
else
    pomdp.n = [];
end

% Otherwise, generate from likelihood
for i = setdiff(1:numel(pomdp.A),find(pomdp.n)')
    ind = [{':'}, num2cell(s(pomdp.dom.A(i).s,end)')];
    [~,o(i)] = max(pomdp.A{i}(ind{:}));
end
