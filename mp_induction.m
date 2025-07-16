function [H,n] = mp_induction(h,B,Q)
% Function to compute inductive priors over paths
%--------------------------------------------------------------------------
% Note: Currently, this assumes the transition probabilities factorise
% over state factors.

H   = ones(numel(Q),1);
ind = find(~cellfun(@isempty,h));

b = cell(size(ind));
k = zeros(size(ind));
for i = 1:length(ind)
    k(i) = h{ind(i)}(1);
    b{i} = sum(B{ind(i)},3:length(size(B{ind(i)})))> 1/8;
end

z = 0;
n = 0;
while z == 0 && n < 64
    q = true(numel(Q),1);
    for j = 1:numel(Q)
        for i = 1:length(ind)
            K = zeros(size(b{i},1),1);
            K(k(i)) = 1;
            w = Q{j}{ind(i)}'*((b{i}')^n)*K >= 1/2;
            q(j) = q(j) && w;
        end
    end
    if sum(q)>0
        H = q > 0;
        z = 1;
    end
    n = n + 1;
end