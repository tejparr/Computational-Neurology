function C = mp_tensor_con(A, B, ci, ai, bi)
% Format C = mp_tensor_con(A, B, ci, ai, bi)
% Generalized tensor contraction of tensors A and B. The ci, ai, and bi are
% each vectors of scalar values indexing all of the dimensions of each 
% tensor. Any indicies present in ai or bi, but not ci, are summed (i.e.,
% contracted).
%--------------------------------------------------------------------------

% Total number of dimensions before contraction
%--------------------------------------------------------------------------
nT = max([ci(:); ai(:); bi(:)],[],'all');

% Expand A into full dimension space
%--------------------------------------------------------------------------
[~, pA] = sort(ai);             % Find indices to permute A so its axes are ordered by ai
A       = permute(A, pA);       % Perform permutation to reorder dimensions
sA      = ones(1, nT);          % Determine size of expanded shape dimensions
sA(ai)  = size(A);              % Assign non-singleton dimensions in A to their relevant places
A       = reshape(A, sA);       % Reshape the permuted A into the new array dimensions

% Expand B into full dimension space
%--------------------------------------------------------------------------
[~, pB] = sort(bi);             % Find indices to permute B so its axes are ordered by bi
B       = permute(B, pB);       % Perform permutation to reorder dimensions
sB      = ones(1, nT);          % Determine size of expanded shape dimensions
sB(bi)  = size(B);              % Assign non-singleton dimensions in B to their relevant places
B       = reshape(B, sB);       % Reshape the permuted B into the new array dimensions

% Product in expanded dimension space
% -------------------------------------------------------------------------
C = A.*B;

% Sum over contracted dimensions
% -------------------------------------------------------------------------
ix = setdiff(1:nT, ci, 'stable');
for d = sort(ix, 'descend')
    C = sum(C, d);
end

% Remove singleton dimensions
%--------------------------------------------------------------------------
C = squeeze(C);

