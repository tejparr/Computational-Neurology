function G = mp_POMDP_B(ind, dom, T)
% This function takes the indices of variables in a POMDP model and
% constructs a causal graph, making use of the domain factors as provided.
% Unlike the mp_POMDP_G routine, this function deals with the full model,
% enabling Bayesian smoothing.
%--------------------------------------------------------------------------

G  =  cell(max(ind.A),1);
Nb =  length(ind.B)/(T-1);
Ne =  length(ind.E)/(T-1);
Na =  length(ind.A)/T;

% Connect states at t = 2 to states at t = 1 and paths from 1 to 2
%--------------------------------------------------------------------------
for i = ind.D
    G{ind.B(i)} = [i,ind.D(dom.B(i).s),ind.E(dom.B(i).u)];
end

% And for observations
%--------------------------------------------------------------------------
for i = 1:Na
    G{ind.A(i)} = ind.D(dom.A(i).s);
end

% And the same for subsequent times
%--------------------------------------------------------------------------
for t = 2:T-1
    for i = 1:Nb
        G{ind.B((t-1)*Nb+i)} = [ind.B((t-2)*Nb+i),ind.B((t-2)*Nb+dom.B(i).s),ind.E((t-1)*Ne+dom.B(i).u)];
    end
end

% And add observations
%--------------------------------------------------------------------------
for t = 2:T
    for i = 1:Na
        G{ind.A((t-1)*Na + i)} = ind.B((t-2)*Nb + dom.A(i).s);
    end
end