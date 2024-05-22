function S = fwt(X,f,w,dt)
% Format S = fwt(X,f,w,dt)
% Time-freq analysis using a Fourier Wavelet Transform of X (an n x t 
% matrix) with frequencies given by the vector f and window given by w.
%--------------------------------------------------------------------------

T = size(X,2);
S = zeros(length(f),size(X,1),size(X,2));
n = size(X,1);

for t = 1:T
    g  = repmat(exp( - w*dt^2*((1:T) - t).^2),[n,1]);          % Gaussian Window
    gX = g(:,g(1,:)>exp(-8)).*X(:,g(1,:)>exp(-8));             % Windowed timeseries
    S(:,:,t) = (gX*exp(1i*2*pi*f.*(0:size(gX,2)-1)'*dt))';     % Local Fourier Transform
end

return

%% Demo
%--------------------------------------------------------------------------
dt = 1/16;
f  = (1:64)/16;

x  = sin(2*pi*(1:128)*dt/32) + sin(2*pi*(1:128)*dt/4) + cos(2*pi*(1:128)*dt);
S  = fwt(x,(1:64)/16,1,dt);
S  = squeeze(S);

subplot(2,1,1)
plot(dt*(1:128),x)
title('Signal')
xlabel('Time')

subplot(2,1,2)
imagesc(1:128*dt, f, abs(S)), axis xy
title('Time-Frequency')
xlabel('Time')
ylabel('Frequency')