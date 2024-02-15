function x = smax(y)
% Format x = smax(y)
% Softmax (normalised exponential) function of the sort used to convert
% unnormalised log probability values into a categorical probability
% distribution.
%--------------------------------------------------------------------------
x = exp(y)*diag(1./sum(exp(y)));

