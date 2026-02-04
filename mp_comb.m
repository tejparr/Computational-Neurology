function A = mp_comb(v,n)
% Function that returns a matrix, A, containing combinations of n variables
% drawn from the vector v.
%--------------------------------------------------------------------------
C      = cell(1, n);
[C{:}] = ndgrid(v);
A      = cell2mat(cellfun(@(x) x(:), C, 'UniformOutput', false));
