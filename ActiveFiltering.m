function [Y,M,t,a] = ActiveFiltering(f,F,Pf,g,G,Pg,x0,m0,n,s,dt,T,a0,y,yc)
% Format [Y,M,t,a] = ActiveFiltering(f,F,Pf,g,G,Pg,x0,m0,n,s,dt,T,a0,y,yc)
% Generates a data timeseries Y and belief timeseries M with times t
%
% f, F       are the flow functions for model and process
% g, G       are the expectations from the likelihood (as above)
% x0, m0     are the initial conditions for states and beliefs
% Pf, Pg     are precisions for the flow, likelihood, and priors
% n          gives the order of motion to be used
% s          gives the assumed smoothness of the noise process (for model only)
% dt         is the timestep to be used
% T          is the duration of the simulation
% a0         is the initial active state
% y          are data (if not expected to be simulated)
% yc         indexes those data modalities that are controllable
%__________________________________________________________________________

try T = dt*(size(y,2)-1); catch,                 end
try Ng = size(g(m0),1);   catch, Ng = size(g,1); end

t = 0:dt:T;
Nt = length(t);

Nf = size(m0,1);
NF = size(x0,1);

% Pre-allocate all cell arrays
%--------------------------------------------------------------------------
g0 = cell(n,1);
G0 = cell(n,1);
f0 = cell(n,1);
F0 = cell(n,1);
X  = cell(n,1);
Y  = cell(n,1);
M  = cell(n,1);

% Initialize first order states
%-------------------------------------------------------------------------- 
X{1} = zeros(NF,Nt); X{1}(:,1) = x0;
M{1} = zeros(Nf,Nt); M{1}(:,1) = m0;

if ~isempty(a0)
    a = zeros(1,Nt);
    a(1) = a0;
end

% Initialize Y with provided data or zeros
try 
    Y{1} = y; 
    Y{1}(yc,:) = 0; 
catch
    Y{1} = zeros(Ng,Nt); 
end

% Pre-compute initial flow and Jacobian
%--------------------------------------------------------------------------
try 
    f0{1} = f(M{1}(:,1));     
catch
    f0{1} = f;   
end
fJ = AF_dfdx(f,M{1}(:,1));

% Initialize higher order states
%--------------------------------------------------------------------------
for j = 2:n
    X{j} = zeros(NF,Nt);
    if exist('y','var')
        Y{j} = gradient(Y{j-1})/dt;
        Y{j}(yc,:) = 0;
    else
        Y{j} = zeros(Ng,Nt);
    end
    M{j}      = zeros(Nf,Nt); 
    M{j}(:,1) = f0{j-1};
    f0{j}     = fJ*M{j}(:,1);
end
f0{end} = zeros(size(f0{end}));

% Initialize autocorrelations - vectorized computation
%--------------------------------------------------------------------------
k = 0:(n-1);
x = sqrt(2)*s;
r = zeros(1,2*n-1);
r(1+2*k) = cumprod(1-2*k)./(x.^(2*k));

% Pre-compute autocorrelation matrix
%--------------------------------------------------------------------------
A = zeros(n,n);
for z = 1:n
    A(:,z) = r((1:n) + z - 1);
    r      = -r;
end

% Generalised precisions
%--------------------------------------------------------------------------
Pg = (pinv(kron(A,pinv(Pg))));
Pf = (pinv(kron(A,pinv(Pf))));
D  = eye(n);
D  = [D(2:n,:);zeros(1,n)];
D  = sparse(kron(D,eye(Nf)));

% Main time loop
%--------------------------------------------------------------------------
for i = 1:Nt-1
    % Compute derivatives for likelihood and flow functions
    %----------------------------------------------------------------------
    g0{1} = g(M{1}(:,i));         
    G0{1} = G(X{1}(:,i));
    gJ    = AF_dfdx(g,M{1}(:,i)); 
    GJ    = AF_dfdx(G,X{1}(:,i));
    f0{1} = f(M{1}(:,i));
    fJ    = AF_dfdx(f,M{1}(:,i));

    if isempty(a0)
        F0{1} = F(X{1}(:,i));
        FJ    = AF_dfdx(F,X{1}(:,i));
    else
        F0{1} = F(X{1}(:,i),a(i));
        FJ    = AF_dfdx(F,X{1}(:,i),a(i),1);
        FJa   = AF_dfdx(F,X{1}(:,i),a(i),2);
    end
    
    % Subsequent orders of motion
    %----------------------------------------------------------------------
    for k = 2:n
        X{k}(:,i) = F0{k-1};
        G0{k} = GJ*X{k}(:,i);
        g0{k} = gJ*M{k}(:,i);
        f0{k} = fJ*M{k}(:,i);
        F0{k} = FJ*X{k}(:,i);
    end
    f0{end} = zeros(size(f0{end}));

    % Generate data and higher order states
    
    if ~exist('y','var')
        for k = 1:n
            Y{k}(:,i) = G0{k};
        end
    else
        for k = 1:n
            Y{k}(yc,i) = G0{k}(yc);
        end
    end

    % Compute free energy gradients and Hessian
    %----------------------------------------------------------------------
    yY = zeros(Ng*n,1);
    m  = zeros(Nf*n,1);
    for k = 1:n
        yY((k-1)*Ng + (1:Ng),:) = Y{k}(:,i);
        m((k-1)*Nf + (1:Nf),:)  = M{k}(:,i);
    end
    
    DfJ   = kron(eye(n),fJ);
    DgJ   = kron(eye(n),gJ);
    dFdx  = (DfJ - D)'*Pf*(D*m - cat(1,f0{:})) + DgJ'*Pg*(yY - cat(1,g0{:}));
    dFdxx = -(DfJ - D)'*Pf*(DfJ - D) - DgJ'*Pg*DgJ;

    % Update beliefs
    %----------------------------------------------------------------------
    m = m + pinv(dFdxx + D)*(expm(dt*(dFdxx + D)) - eye(Nf*n))*(dFdx + D*m);

    % Reshape and store updated beliefs
    %----------------------------------------------------------------------
    for k = 1:n
        M{k}(:,i+1) = m((k-1)*Nf + (1:Nf),:);
    end
    
    % And action
    %----------------------------------------------------------------------
    if ~isempty(a0)
        dyda = zeros(Ng*n,length(a0));
        for k = 2:n
            dyda((Ng*(k-1)) + (1:Ng),:) = GJ*(FJ^(k-2))*FJa;
        end
        dFda  = -dyda'*Pg*(yY - cat(1,g0{:}));
        dFdaa = -dyda'*Pg*dyda;
        dFdax = -dyda'*Pg*kron(ones(n,1),GJ);
        
        % Update action and states
        %------------------------------------------------------------------
        Jxa = [FJ FJa;dFdax dFdaa];
        Xa          = [X{1}(:,i); a(:,i)] + pinv(Jxa)*(expm(dt*Jxa) - eye(size(Jxa,1)))*[F0{1};dFda];
        X{1}(:,i+1) = Xa(1:NF,1);
        a(:,i+1)    = Xa(NF+1:end,1);
    else
        % Update states without action
        X{1}(:,i+1) = X{1}(:,i) + (expm(dt*FJ) - eye(Nf))*pinv(FJ)*F0{1};
    end
end

function J = AF_dfdx(f,x,a,n)
% Numerical derivative with respect to argument n using finite difference method
%--------------------------------------------------------------------------
if nargin > 2
    if n == 1
        J = zeros(size(f(x,a),1),size(x,1));
        dx = zeros(size(x));
        for i = 1:size(x,1)
            dx(i) = exp(-4);
            J(:,i) = (f(x+dx,a) - f(x-dx,a))/(2*exp(-4));
            dx(i) = 0;
        end
    else
        J = zeros(size(f(x,a),1),size(a,1));
        da = zeros(size(a));
        for i = 1:size(a,1)
            da(i)  = exp(-4);
            J(:,i) = (f(x,a+da) - f(x,a-da))/(2*exp(-4));
            da(i)  = 0;
        end
    end
else
    J = zeros(size(f(x),1),size(x,1));
    dx = zeros(size(x));
    for i = 1:size(x,1)
        dx(i)  = exp(-4);
        J(:,i) = (f(x+dx) - f(x-dx))/(2*exp(-4));
        dx(i)  = 0;
    end
end