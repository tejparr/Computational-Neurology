function [m,H,fx] = cn_extremise(f,x)
% This function extremises a function of the form f(x), returning:
% m  = argmax(f(x))
% H  = f"(x)|_{x = m}
% fx = f(m)
%
% Note that the derivative operator used here requires that complex
% perturbations yield complex numbers in the value returned by the scalar
% function f(x). A common pitfall is to write x'*x without realising that
% this is a conjugate transpose, while it should be written x.'*x to ensure
% a transpose only.
%--------------------------------------------------------------------------

% Newton's method/simple gradient ascent (with adaptive implicit step-size)
%--------------------------------------------------------------------------
% As dt -> infinity, this approaches Newton's method, but as dt -> 0 this approaches simple gradient ascent
dt = 1;                 % implicit step-size
Ni = 64;                % maximum iterations

% Plotting initialisation
%--------------------------------------------------------------------------
cn_figure('Maximise')
subplot(2,1,1)
h1 = bar(zeros(1,Ni),'EdgeColor','none','FaceColor',[0.7 0.7 0.8]);
xlabel('Iteration')
axis square
box off
title('Function to be optimised')

subplot(2,1,2)
h2 = bar(x,'EdgeColor','none','FaceColor',[0.7 0.7 0.8]);
xlabel('Parameter')
axis square
box off
title('Parameters')

% Optimisation loop
%--------------------------------------------------------------------------
for i = 1:Ni
    try
        % Derivatives and regularisation
        %------------------------------------------------------------------
        [fx,g,H] = cn_diff(f,x);   % Calculate relevant derivatives (up to second order)
        [V,D] = eig((H+H')/2);     % Eigendecomposition of (symmetrised) Hessian
        d = diag(D);               % Eigenvalues
        d = min(d,-1e-6);          % Adjust any positive eigenvalues to deal with non-convexities
        H = V*diag(d)*V.';         % Replace H with this corrected version

        % Integration (gradient step using exponential integrator)
        %------------------------------------------------------------------
        M        = [H*dt g;        % Matrix operator
                    zeros(1,length(x)+1)];
        E        = expm(M);        % Matrix exponential
        P        = E(1:end-1,end); % Rate of increase (dx/dt)
        dx       = dt*P;           % Increment
        z        = x + dx;         % Try new value of x
        fz       = f(z);           % Assess actual change in function

        % Assess local deviation from quadratic function
        %------------------------------------------------------------------
        rn = fz - fx;
        rd = dx.'*g + dx.'*H*dx/2; 
        r  = rn/(sign(rd)*(max(abs(rd),1e-4)));

        if i==1
            f0 = fx;
        end

        if i>4 && abs(rn)<1/32                % Convergence critereon
            break
        elseif rn > 0                         % Assess whether to accept new value
            x = z;
            fx = fz;
        end
        dt = max(min(dt*exp(r),8),1e-4);     % Adjust step-size

    catch % Numerical failures
        dt = dt*0.5;
    end

    % Online plotting
    %----------------------------------------------------------------------
    h1.YData(i) = fx-f0;
    h2.YData    = x;
    drawnow limitrate
end
m = x; % return argmax_x(f)
