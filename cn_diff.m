function [fx, dfdx, H] = cn_diff(f,x)
% Derivatives of a function f evaluated at x. First derivatives (gradient)
% are evaluated using a complex-step method which depends upon the
% following:
%
% f(x+i*h)     = f(x) + i*h*f'(x) + O(h^2)
%              =>
% Im(f(x+i*h)) = h*f'(x) + O(h^3)
%              =>
% f'(x)        = Im(f(x+i*h))/h + O(h^2)
%
% The second derivatives (Hessians) are computed using analogous
% complex-step methods based upon the real parts of the same expansion. For
% the diagonal elements of the Hessian, this is:
%
% f(x+i*h)     = f(x) + i*h*f'(x) - (h^2)*f"(x)/2 + O(h^3)
%              =>
% Re(f(x+i*h)) = f(x) - (h^2)*f"(x)/2 + O(h^4)
%              => 
% f"(x)        = 2*(f(x) - Re(f(x+i*h)))/h^2 + O(h^2)
%
% For the off-diagonal elements, let's define the perturbation:
% 
% dx := i*h*e_i+h*e_j
%
% then: 
%
% f(x+dx)      = f(x) + (f'(x))'*dx + dx'*f"(x)*dx/2 + O(dx^3)
%              =>
% Im(f(x+dx))  = (f'(x))'*h*e_i + (h^2)*(e_i)'*f"(x)*(e_j) + O(dx^3)
%              =>
% H(i,j)       = (e_i)'*f"(x)*(e_j)
%              = (Im(f(x+dx)) - (f'(x))'*h*e_i)/h^2 + O(h)
%
% Note that this method relies upon the perturbations yielding complex
% terms in the associated function. This means that operations like x'*x
% should be replaced with x.'*x to avoid introducing conjugate transposes
% that negate the logic of this method.
%--------------------------------------------------------------------------

n = length(x);      % dimension of x
h = 1e-20;          % complex-step step

% Gradients
%--------------------------------------------------------------------------
dfdx = zeros(n,1);

for i = 1:n
    x_h     = x;
    x_h(i)  = x_h(i) + 1i*h;
    dfdx(i) = imag(f(x_h))/h;
end

if nargout > 2

    % Hessian (complex-step on gradient)
    %----------------------------------------------------------------------
    h = 1e-4;           % complex-step step
    H  = zeros(n,n);
    fx = f(x);

    for i = 1:n

        % diagonal terms
        %------------------------------------------------------------------
        xh = x;
        xh(i) = xh(i) + 1i*h;
        H(i,i) = 2*(fx - real(f(xh)))/(h^2);

        % off-diagonal terms
        %------------------------------------------------------------------
        for j = i+1:n

            xj = x;
            xj(i) = xj(i) + 1i*h;
            xj(j) = xj(j) + h;

            xk = x;
            xk(i) = xk(i) + 1i*h;

            Hij = (imag(f(xj)) - imag(f(xk)))/(h^2);

            H(i,j) = Hij;
            H(j,i) = Hij;
        end
    end
end