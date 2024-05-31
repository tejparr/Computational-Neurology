function b = mp_log(a)
% Log adjusted to avoid numerical problems when a is very small
%--------------------------------------------------------------------------
b = max(log(double(a)),-32);