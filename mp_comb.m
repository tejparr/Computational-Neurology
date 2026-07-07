function A = mp_comb(v,n)
% FORMAT A = mp_comb(v,n)
% Function that returns a matrix, A, containing combinations of n variables
% drawn from the vector v.
% FORMAT A = mp_comb(v)
% Returns a matrix, A, containing subscript indices for the elements of a
% tensor whose dimensions are given by the vector, v
%--------------------------------------------------------------------------
if nargin>1
    C      = cell(1, n);
    [C{:}] = ndgrid(v);
    A      = cell2mat(cellfun(@(x) x(:), C, 'UniformOutput', false));
else
    n = numel(v);
    V = cell(1, n);
    for k = 1:n
        V{k} = 1:v(k);
    end

    grids = cell(1, n);
    [grids{:}] = ndgrid(V{:});

    A = zeros(prod(v), n);
    for k = 1:n
        A(:,k) = grids{k}(:);
    end
end