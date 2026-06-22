function lut = mp_lut(LUT)
% Format lut = mp_lut(LUT)
% Function that takes a determinstic look-up table of the form:
% LUT(i,j,k,...) = x, where i,j,k,... and x are categorical indicies, and
% returns a structure containing:
% lut.LUT       - the original look-up table
% lut.Size      - the size of this table
% lut.Cond      - the configurations of variables in the conditioning set consistent with each support index
% lut.Subs      - for each index i,j,k,..., a vector of the value this takes in the corresponding configuration
% lut.Rev       - the reverse mappings such that Rev{a}{b} gives, for dimension a, when it takes value b, the compatible configurations in Sub 
%--------------------------------------------------------------------------

K        = max(LUT(:));
lut.LUT  = LUT;
lut.Size = size(LUT);
ind      = (1:numel(LUT))';
N        = ndims(LUT);

lut.Cond = accumarray( ...
    LUT(:), ...
    ind, ...
    [K 1], ...
    @(x){x}, ...
    {[]} );

subs = cell(1,N);
[subs{:}] = ind2sub(size(LUT),ind);

lut.Rev  = cell(N,1);
lut.Subs = subs;

for i = 1:N

    lut.Rev{i} = accumarray( ...
        subs{i}(:), ...
        ind, ...
        [size(LUT,i) 1], ...
        @(x){x}, ...
        {[]} );
end

