function a = mp_oh(N,n)
% FORMAT a = mp_oh(N,n)
% Returns one-hot vector of length N with one in position n
%--------------------------------------------------------------------------
a = zeros(N,1); a(n) = 1;