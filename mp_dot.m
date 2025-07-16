function C = mp_dot(A,B,j)
% Format C = mp_dot(A,B,j)
% Generalised (tensor) dot product with tensor A, array of vectors B, omit
% dimensions j from sum.
%--------------------------------------------------------------------------

% Input validation
if nargin < 2
    error('At least two input arguments required');
end

if ~iscell(B)
    error('B must be a cell array of vectors');
end


% Handle optional j parameter
if nargin < 3
    J = true(1,numel(B));
else
    J = ~ismember(1:numel(B),j);
end

C = A;

% Alernative fomualtion using builtin tensorprod function, but this appears
% to be slower
%--------------------------------------------------------------------------
% for i = numel(B):-1:1
%     if J(i)
%         C = tensorprod(C,B{i},i,1);
%     end
% end
%--------------------------------------------------------------------------

% Pre-allocate dimensions array
dims = 1:ndims(C);
d    = ndims(C)-numel(B);

for i = 1:numel(B)
    if J(i)
        b = full(B{i});
        b = permute(b, circshift(dims,d+i-1));
        if isscalar(b) && isscalar(C)
            C = C*b;
        else
            C = sum(bsxfun(@times, C, b), i+d);
        end
    end
end

C = squeeze(C);

if size(C,2) > 1 && size(C,1) == 1
    C = C'; % Ensure output is column vector
end



