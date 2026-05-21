function [qM,qP,F] = VariationalLaplace(L,pP,pM,m0,y)
% [qM,qP,F] = VariationalLaplace(L,pP,pM,m0,y)
% Lightweight variational Laplace scheme
% L  = Likelihood function (note that must be compatible with complex step numerical derivatives)
% pP = Prior precision
% pM = Prior mode
% m0 = Parameter initialisation
% y  = Data
% qM = Posterior mode
% qP = Posterior precision
% F  = Evidence Lower Bound/Negative Free energy
%--------------------------------------------------------------------------
f         = @(x) L(y,x) - (x - pM).'*pP*(x - pM)/2;
[qM,H,Lm] = cn_extremise(f,m0);
qP        = pP - H;
F         = vl_Marginal_Likelihood(Lm,pP,pM,qM,qP);

function F = vl_Marginal_Likelihood(Lm,pP,pM,qM,qP)
% Marginal likelihood approximation under the Laplace approximation
% Lm = Log Likelihood function evaluated at posterior mode
% pP = Prior Precision
% pM = Prior Mean
% qP = Posterior Precision
% qM = Posterior Mode
% F  = Evidence Lower Bound/Negative Free energy
%--------------------------------------------------------------------------

F        = Lm ...                                  %  ln p(y|m)
         - (pM - qM).'*pP*(pM - qM)/2 ...          %{ ln p(m)
         + vl_logdet(pP)/2 ...                     %{
         - vl_logdet(qP)/2;                        % -ln q(m)


function ld = vl_logdet(A)
% Log determinant using Cholskey Decomposition
%--------------------------------------------------------------------------
ld = 2*sum(log(diag(chol(A))));