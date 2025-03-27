function x = cn_smax(y)
% Format x = cn_smax(y)
% Softmax (normalised exponential) function of the sort used to convert
% unnormalised log probability values into a categorical probability
% distribution.
%--------------------------------------------------------------------------
if size(y) == 1

    x =  exp(y)/(exp(y) + 1);
    
else

    x = exp(y)*diag(1./sum(exp(y)));
end
