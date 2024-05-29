function PSD = psd_ls(x,n,f,dt)
% Format PSD = psd_ls(x,n,f,dt)
% Power-spectral density estimation based upon least squares solution to
% the following autoregressive model of order n for frequencies f:
%
% x(2:end)' = [x(1) 0    0    ...]*W + E
%             [x(2) x(1) 0    ...]
%             [x(3) x(2) x(1) ...]
%             [ .    .    .    . ]
% 
%--------------------------------------------------------------------------

% Matrix form for autoregressive model
%--------------------------------------------------------------------------
X = zeros(length(x)-1,n);     

for i = 1:length(x)-1
    X(i,:) = [x(i:-1:max(1,i+1-n)) zeros(1,n-i)]';
end

% Least squares (ML) estimate for coefficients and noise variance
%--------------------------------------------------------------------------
W = pinv(X'*X)*X'*x(2:end)';  
V = (x(2:end)' - X*W)'*(x(2:end)' - X*W)/(length(x)-1);

% Convert to PSD
%--------------------------------------------------------------------------
s = 0;
for k = 1:n
    s = s + W(k)*exp(-1i*2*pi*f*k*dt);
end

PSD = V./(abs(1 - s).^2);
