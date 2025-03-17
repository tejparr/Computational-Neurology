function POMDP = mp_POMDP(pomdp)
% Format POMDP = mp_POMDP(pomdp)
% This function solves a Partially Observed Markov Decision Process
% generative model using the MessagePassing.m belief-propagation scheme.
% The pomdp structure is formulated as follows:
%
% 
%
% This scheme...
%
% A limitation here is that in cyclic models (e.g., when there are loopy
% conditional dependencies in the space of the transition probabilities)
% the backwards pass necessary for completing the loop is not performed
% during the filtering in the forward direction used for planning. This can
% often be ameliorated through use of outcomes that help resolve any of the
% relevant uncertainty such that precise beliefs about the current state
% can be used to propagate forwards in time. The (slower) alternative would
% be to use a larger model at each time-step so that the full backwards
% sweep and potentially circular iterations can be used to constrain
% inferences.
%--------------------------------------------------------------------------

% Preliminaries
%--------------------------------------------------------------------------
M.nograph = 1;
M.noprint = 1;
M.acyclic = 1;

% Unpack mandatory fields
%--------------------------------------------------------------------------
% This requires either the probability distributions themselves or
% Dirichlet parameters for their priors.

try A = pomdp.A; catch, a = pomdp.a;      end % Likelihood probabilities
try B = pomdp.B; catch, b = pomdp.b;      end % Transition probabilities
try D = pomdp.D; catch, d = pomdp.d;      end % Initial state probabilities

T = pomdp.T;                                  % Time horizon

% Unpack optional fields
%--------------------------------------------------------------------------
if isfield(pomdp,'E'), E = pomdp.E;       end % Prior over paths

% Dirichlet priors (duplicate of above if only Dirichlet priors specified)
if isfield(pomdp,'a'),   a = pomdp.a;     end % likelihood
if isfield(pomdp,'b'),   b = pomdp.b;     end % transitions
if isfield(pomdp,'d'),   d = pomdp.d;     end % initial states
if isfield(pomdp,'e'),   e = pomdp.e;     end % paths

% Domains and controllable paths
if isfield(pomdp,'dom'), dom = pomdp.dom; end % domains
try con = pomdp.con; catch, try con = 1:numel(E); catch, con = 1:numel(e); end,  end % controllable paths

% Planning horizon
try N = pomdp.N; catch, N = 1;            end % assume 1-step ahead unless otherwise specified

% Identify indices of each variable type
%--------------------------------------------------------------------------

try ind.E = 1:numel(E);                catch, ind.E = 1:numel(e);                end % Indices to identify all paths
try ind.D = (1:numel(D)) + ind.E(end); catch, ind.D = (1:numel(d)) + ind.E(end); end % Indices to identify all states
try ind.B = (1:numel(B)) + ind.D(end); catch, ind.B = (1:numel(b)) + ind.D(end); end % Indices to identify all next states
try ind.A = (1:numel(A)) + ind.B(end); catch, ind.A = (1:numel(a)) + ind.B(end); end % Indices to identify all observations

% Construct causal graph
%==========================================================================
G = mp_POMDP_G(ind, dom);       % Create graph from indices and domains
M.G = G;                        % Equip model with graph

% Construct uninformative factors to deal with childless nodes
%--------------------------------------------------------------------------
O = cell(length(ind.B),1);
for i = 1:numel(O)
    try
        O{i} = ones(1,size(B{i},1));
    catch
        O{i} = ones(1,size(b{i},1));
    end
end

% Path combinations
%--------------------------------------------------------------------------
Ne = zeros(size(ind.E));
for k = 1:length(ind.E)
    try Ne(k) = length(E{k}); catch, Ne(k) = length(e{k}); end
end

V = mp_POMDP_comb(Ne(con));  % Determine controllable path combinations
u = zeros(length(Ne),T-1);   % Initialise choice array

% Initialise cell arrays for outputs
%--------------------------------------------------------------------------
pomdp.P = cell(T,1);         % Path probabilities
pomdp.Q = cell(T,numel(D));  % State probabilties              

% Iterate through time
%==========================================================================
for t = 1:T    

    Q = cell(size(V,1),1);  % Posterior marginals conditioned upon controllable paths
    U = cell(size(V,1),1);  % Messages conditioned upon controllable paths

    % Iterate over (controllable) paths
    %----------------------------------------------------------------------
    for k = 1:size(V,1)

        % Construct model
        %------------------------------------------------------------------
        for j = con
            E{j} = mp_POMDP_oh(Ne(j),V(k,j)); % Set controllable path
        end

        M.A = [E(:);D(:);B(:);A(:);O(:)];
        
        try   % If available, use outcomes provided
            Y = pomdp.o(:,t);
        catch % Otherwise, simulate/obtain outcomes from generative function
            if t > 1
                s       = pomdp.s(:,t-1);
                [Y,s]   = pomdp.gen(s,u(:,t-1));
                pomdp.s(:,t) = s;
                pomdp.o(:,t) = Y;
            else
                s       = pomdp.s;
                Y       = pomdp.gen(s);
                pomdp.o(:,t) = Y;
            end
        end
        Y = [Y; ones(length(ind.B),1)]; % To handle childless nodes

        % Solve model
        %------------------------------------------------------------------
        [Qk, ~, Uk] = MessagePassing(M,Y);
        U{k}        = Uk.d(ind.B);  
        Q{k}        = Qk;
    end
    
    % Equip controllable paths with priors and evaluate posteriors
    %----------------------------------------------------------------------
    if isfield(pomdp,'path')
        P = pomdp.path(pomdp,U,N,V);   % If function specified for path priors
    else
        P = mp_path(pomdp,U,N,V);      % Otherwise use default (based on expected free energy)
    end
    pomdp.P{t} = P;                    % Save path probabilities
    
    % Update priors for next step
    %----------------------------------------------------------------------
    for i = 1:numel(D)
        D{i}         = zeros(size(D{i}));
        pomdp.Q{t,i} = zeros(size(D{i}));
        for k = 1:numel(U)
            D{i}         = D{i} + U{k}{i}*P(k);
            pomdp.Q{t,i} = pomdp.Q{t,i} + Q{k}.s{ind.D(i)}*P(k);
        end
    end

    % Select actions
    %----------------------------------------------------------------------
    % Here the actions are selected directly by sampling from the
    % distribution over paths. An alternative option not yet implemented
    % will be to select actions that minimise the difference between
    % anticipated and realised outcomes.
    vi = find(rand<cumsum(P),1,'first');    
    u(:,t) = V(vi,:)';

end
%--------------------------------------------------------------------------
POMDP = pomdp;
POMDP.u = u;

function G = mp_POMDP_G(ind, dom)
% This function takes the indices of variables in a POMDP model and
% constructs a causal graph, making use of the domain factors as provided
%--------------------------------------------------------------------------

G = cell(ind.B(end)+length(ind.B),1);

% Connect hidden states at t = 1 to outcomes at t = 1
%--------------------------------------------------------------------------
for i = ind.A
    j = i + 1 - ind.A(1);
    if isfield(dom,'A') % If domains of A are specified, use these
        G{i} = [ind.D(dom.A(j).s) ind.E(dom.A(j).u)];
    else                % Otherwise, assume depends on all states
        G{i} = ind.D;
    end
end

% Connect hidden states and actions at t = 1 to states at t = 2
%--------------------------------------------------------------------------
for i = ind.B
    j = i + 1 - ind.B(1);
    if isfield(dom,'B') % If domains of B are specified, use these
        G{i} = [ind.D(j) ind.D(dom.B(j).s) ind.E(dom.B(j).u)];
    else  % Otherwise, assume depends on self states and all actions
        G{i} = [ind.D(j) ind.E];
    end
end

% Equip childless nodes with uninformative outcome
%--------------------------------------------------------------------------
for i = 1:length(ind.B)
    G{ind.A(end) + i} = ind.B(i);
end

function U = mp_POMDP_comb(Ne)
% Determine combinations of paths
%--------------------------------------------------------------------------
U = zeros(prod(Ne),length(Ne));

for k = 1:length(Ne)
    bk = ones(1,prod(Ne(1:k-1)));   if isempty(bk), bk = 1; end
    ak = ones(1,prod(Ne(k+1:end))); if isempty(ak), ak = 1; end
    U(:,k) = kron(kron(bk,1:Ne(k)),ak)';
end

function a = mp_POMDP_oh(N,n)
% Returns one-hot vector of length N with one in position n
%--------------------------------------------------------------------------
a = zeros(N,1); a(n) = 1;

function [P,G] = mp_path(pomdp,U,t,V)
% Computes a prior probability for paths based upon beliefs about likely
% outcomes. The inputs are the pomdp structure, the messages from present
% to future states (U), the time-steps (t) left in the planning horizon,
% and the combinations of controllable paths we need to consider.
%--------------------------------------------------------------------------

E = ones(size(V,1),1);
for i = 1:size(V,2)
    E  = E.*pomdp.E{i}(V(:,i));   
end

Ho = zeros(size(E));
HA = zeros(size(E));
C  = zeros(size(E));

for k = 1:size(V,1)
    for j = 1:numel(pomdp.A)
        Qo    = mp_dot(pomdp.A{j},U{k}(pomdp.dom.A(j).s)); % Predictive posterior
        C(k)  = C(k) + Qo'*pomdp.C{j};                     % Expected utility
        Ho(k) = Ho(k) - Qo'*mp_log(Qo);                    % Predictive entropy
        Ha    = -sum(pomdp.A{j}.*mp_log(pomdp.A{j}),1);    % Conditional entropy
        HA(k) = HA(k) + mp_dot(Ha,U{k}(pomdp.dom.A(j).s)); % Ambiguity
    end
end

G = mp_log(E) + Ho + C - HA;
%               __   __
%                'Risk'
%                        __
%                    'Ambiguity'
%               __       __
%       'Expected information gain'
%                    __
%            'Expected utility'
P = mp_softmax(G);

if max(P) > 0.9 % If policy selection definitive, no need for further recursions
    P(P<=0.9) = 0;
    P(P>0.9)  = 1;
    return
elseif t > 1 % Recursive tree search over policies
    % Propagate one-step ahead 
    %----------------------------------------------------------------------
    u     = cell(size(U{k})); % Initialise messages for next time-step
    uk    = U;
    for k = find(P)' % Loop over plausible policies
        for j = 1:numel(U{k}) % And the associated state factors
            B = pomdp.B{j};
            v = cell(length(pomdp.dom.B(j).u),1);
            for i = 1:numel(v)
                v{i} = ones(size(B,pomdp.dom.B(j).u(i)),1);
            end
            % Determine states a step ahead under each policy
            %--------------------------------------------------------------
            u{j} = mp_dot(B,[{ones(size(B,1),1)},U{k}(j),U{k}(pomdp.dom.B(j).s)',v(:)'],[1,(ndims(B)+1-length(pomdp.dom.B(j).u)):ndims(B)]);    
        end
        for l = 1:size(V,1) % Loop over policies for next step
            for j = 1:numel(U{l}) % and over state factors
                if isempty(pomdp.dom.B(j).u)
                    uk{l}{j} = u{j}(:,1);
                else
                    uk{l}{j} = u{j}(:,V(l,pomdp.dom.B(j).u));
                end
            end
        end
        % Recursively evaluate policies
        %------------------------------------------------------------------
        [p,g] = mp_path(pomdp,uk,t-1,V);
        G(k) = G(k) + p'*g; % Accumulate expected free energies along paths
    end
    P = mp_softmax(G);
else
    return
end
