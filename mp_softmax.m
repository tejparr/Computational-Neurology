function s = mp_softmax(a)
% Softmax (normalised exponential) function
%--------------------------------------------------------------------------
n  = [size(a,1),ones(1,length(size(a))-1)];
s  = exp(a)./repmat(sum(exp(a),1),n);