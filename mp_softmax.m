function s = mp_softmax(a)
% Softmax (normalised exponential) function
%--------------------------------------------------------------------------
s  = exp(a)/sum(exp(a));