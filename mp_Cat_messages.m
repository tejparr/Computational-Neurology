function [Ma, Md, F] = mp_Cat_messages(aM,dM,A)
% Computation of ascending and descending messages from categorical factor
% FORMAT [Ma, Md] = mp_Cat_messages(aM,dM,A)
% aM - ascending messages to factor    (from children)
% dM - descending messages to factor   (from parents and coparents)
% Ma - ascending messages from factor  (to parents and coparents)
% Md - descending messages from factor (from children)
% A  - probability tensor whose first dimension relates to children, with
%      subsequent dimensions relating to (co)parents.
%__________________________________________________________________________
% This function takes messages to a categorical probability factor and
% computes messages from this factor. By default, a belief-propagation
% scheme is assumed. The messages are normalised for numerical stability.
%__________________________________________________________________________

% Compute outgoing descending messages
%--------------------------------------------------------------------------
Md = mp_norm(mp_dot(A,dM));

% Compute outgoing ascending messages
%--------------------------------------------------------------------------
Ma = cell(size(dM));

for i = 1:numel(Ma)
    Ma{i} = mp_norm(mp_dot(A,[{aM},dM(:)'],i+1));
end

% Compute contribution to log marginal likelihood
%--------------------------------------------------------------------------
F = mp_log(aM'*Md);
