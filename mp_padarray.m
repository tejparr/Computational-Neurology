function B = mp_padarray(A, s, v, d)
% FORMAT B = mp_padarray(A, s, v, d)
% Pads array with elements of a scalar value
%   A   - Input array (numeric, logical, etc.)
%   s   - Vector of nonnegative integers specifying pad size per dimension
%   v   - Scalar value used for padding (default = 0)
%   d  - 'both', 'pre', or 'post' (default = 'both')
%--------------------------------------------------------------------------

nd = max(numel(s), ndims(A));
s  = [s zeros(1, nd - numel(s))];
sa = size(A);
sa = [sa ones(1, nd - numel(sa))];

% Direction of padding
%--------------------------------------------------------------------------
switch d
    case 'pre'
        pre = s;
        post = zeros(1, nd);
    case 'post'
        pre = zeros(1, nd);
        post = s;
    case 'both'
        pre = floor(s/2);
        post = s - pre;
    otherwise
        error('Unknown direction: use ''pre'', ''post'', or ''both''.');
end

% Create output array
%--------------------------------------------------------------------------
sb        = sa + pre + post;
B         = repmat(v, sb);
ind       = arrayfun(@(p,n) (p+1):(p+n), pre, sa, 'UniformOutput', false);
B(ind{:}) = A;

