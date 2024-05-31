function C = mp_dot(A,B,j)
% Format C = mp_dot(A,B,j)
% Generalised (tensor) dot product with tensor A, array of vectors B, omit
% dimensions j from sum.
%--------------------------------------------------------------------------
try J = (1:numel(B))~=j; catch, J = true(1,numel(B)); end

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

for i = 1:numel(B)
    if J(i)
        dims = 1:ndims(C);
        d    = ndims(C)-numel(B);
        ind    = circshift(size(C),1-i-d);
        ind(1) = 1;
        b      = repmat(B{i},ind);
        b      = permute(b,circshift(dims,d+i-1));
        C      = sum(C.*b,i+d);
    end
end

C = squeeze(C);

if size(C,2) > 1
    C = C'; % Ensure output is column vector
end
