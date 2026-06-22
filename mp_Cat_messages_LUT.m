function [Ma, Md, F, H] = mp_Cat_messages_LUT(aM, dM, lut)
% Computation of ascending and descending messages from categorical factor
% FORMAT [Ma, Md] = mp_Cat_messages_LUT(aM,dM,lut)
% aM  - ascending messages to factor    (from children)
% dM  - descending messages to factor   (from parents and coparents)
% Ma  - ascending messages from factor  (to parents and coparents)
% Md  - descending messages from factor (from children)
% lut - look-up table playing the role of sparse, deterministic, 
%       probability tensor. This is computed from a probability tensor by
%       mp_lut_precompute.m
%__________________________________________________________________________
% This function takes messages to a categorical probability factor and
% computes messages from this factor. By default, a belief-propagation
% scheme is assumed. The messages are normalised for numerical stability.
%__________________________________________________________________________

LUT    = lut.LUT;
Na     = numel(aM);
Nd     = numel(dM);
N      = numel(LUT);

% Build weights for all possible configurations of the conditioning set
%--------------------------------------------------------------------------
w = ones(N,1);

for i = 1:Nd
    w = w.*dM{i}(lut.Subs{i});
end

iw  = w>0;
w   = w(iw);
LUT = LUT(iw);

% Descending messages
%--------------------------------------------------------------------------

Md = accumarray(LUT(:), w, [Na 1]);
Md = mp_norm(Md);

% Ascending messages
%--------------------------------------------------------------------------

B    = w.*aM(LUT(:)); % Joint probabilities for all configurations
Ma   = cell(Nd,1);    % Initialise ascending messages

for i = 1:Nd
    Mi = accumarray( ...
        lut.Subs{i}(iw), ...
        B, ...
        [numel(dM{i}) 1]);

    % Avoid overcounting of messages by dividing by descending message
    %----------------------------------------------------------------------
    Mi    = Mi ./ max(dM{i},realmin); 
    Ma{i} = mp_norm(Mi);
end

% Free energy contribution
%--------------------------------------------------------------------------

F = mp_log(aM' * Md);

% The conditional entropy is zero in this determinstic setting
%--------------------------------------------------------------------------
H = 0;