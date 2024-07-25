function A  = mp_norm(A)
% normalisation of a probability tensor
%--------------------------------------------------------------------------
A           = rdivide(A,sum(A,1));
A(isnan(A)) = 1/size(A,1);