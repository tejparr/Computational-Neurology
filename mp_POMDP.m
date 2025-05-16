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
if isfield(pomdp,'E'),   E = pomdp.E;     end % Prior over paths

% Dirichlet priors (duplicate of above if only Dirichlet priors specified)
if isfield(pomdp,'a'),   a = pomdp.a;     end % likelihood
if isfield(pomdp,'b'),   b = pomdp.b;     end % transitions
if isfield(pomdp,'d'),   d = pomdp.d;     end % initial states
if isfield(pomdp,'e'),   e = pomdp.e;     end % paths

% Domains and controllable paths
if isfield(pomdp,'dom'), dom = pomdp.dom; end % domains
try con = pomdp.con; catch, try con = 1:numel(E); catch, con = 1:numel(e); end,  end % controllable paths

% Planning horizon
try N = pomdp.N;     catch, N = 1;   end % assume 1-step ahead unless otherwise specified

% Option to factorise over time
try fac = pomdp.fac; catch, fac = 0; end % assume no factorisation over time unless specified

% Identify indices of each variable type
%--------------------------------------------------------------------------

try ind.E = 1:numel(E);                catch, ind.E = 1:numel(e);                end % Indices to identify all paths
try ind.D = (1:numel(D)) + ind.E(end); catch, ind.D = (1:numel(d)) + ind.E(end); end % Indices to identify all states
try ind.B = (1:numel(B)) + ind.D(end); catch, ind.B = (1:numel(b)) + ind.D(end); end % Indices to identify all next states
try ind.A = (1:numel(A)) + ind.B(end); catch, ind.A = (1:numel(a)) + ind.B(end); end % Indices to identify all observations

if fac 
% Identify indices for vertical message passing
%--------------------------------------------------------------------------
    jnd.E = [];
    jnd.B = [];
    try jnd.D = (1:numel(D));              catch, jnd.D = (1:numel(d));              end % Indices to identify all states
    try jnd.A = (1:numel(A)) + jnd.D(end); catch, jnd.A = (1:numel(a)) + jnd.D(end); end % Indices to identify all observations

% Identify indices for horizontal message passing
%-------------------------------------------------------------------------
    knd.A = [];
    try knd.E = 1:numel(E);                catch, knd.E = 1:numel(e);                end % Indices to identify all paths
    try knd.D = (1:numel(D)) + knd.E(end); catch, knd.D = (1:numel(d)) + knd.E(end); end % Indices to identify all states
    try knd.B = (1:numel(B)) + knd.D(end); catch, knd.B = (1:numel(b)) + knd.D(end); end % Indices to identify all next states

end

% Construct causal graph
%==========================================================================
if ~fac
    G    = mp_POMDP_G(ind, dom);       % Create graph from indices and domains
    M.G  = G;                          % Equip model with graph
else
    jG   = mp_POMDP_G(jnd, dom);       % Create graph from indices and domains for vertical model
    kG   = mp_POMDP_G(knd, dom);       % Create graph from indices and domains for horizontal model
end

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

    try   % If available, use outcomes provided
        y = pomdp.o(:,t);
    catch % Otherwise, simulate/obtain outcomes from generative function
        if t > 1
            Qo      = cell(size(A)); % Posterior predictive distribution
            for g = 1:numel(Qo)
                Qo{g}      = mp_dot(pomdp.A{g},D(pomdp.dom.A(g).s));
            end
            s       = pomdp.s(:,t-1);
            [y,s]   = pomdp.gen(s,u(:,t-1),Qo,pomdp);
            pomdp.s(:,t) = s;
            pomdp.o(:,t) = y;
        else
            s       = pomdp.s(:,1);
            y       = pomdp.gen(s,[],[],pomdp);
            pomdp.o(:,t) = y;
        end
    end

    if fac
        % Construct and solve vertical model
        %-----------------------------------------------------------
        M.A = [D(:);A(:)];
        M.G = jG;
        jQ  = MessagePassing(M,y); 
        D   = jQ.s;

        if t == 1
            % If mandatory states are specified, ensure their order is optimised
            %--------------------------------------------------------
            if isfield(pomdp,'h')
                pomdp.h = mp_pomdp_order(pomdp.h,pomdp.B,D);
            end
        end    
    end

    % Iterate over (controllable) paths
    %------------------------------------------------------------------
    for k = 1:size(V,1) 

        % Construct model
        %--------------------------------------------------------------
        for j = con
            E{j} = mp_POMDP_oh(Ne(j),V(k,j)); % Set controllable path
        end

        if ~fac
            M.A = [E(:);D(:);B(:);A(:);O(:)];
        
            Y = [y; ones(length(ind.B),1)]; % To handle childless nodes

            % Solve model
            %------------------------------------------------------------
            [Qk, ~, Uk] = MessagePassing(M,Y);
            U{k}        = Uk.d(ind.B);  
            Q{k}        = Qk;
        else
            % Solve horizontal model
            %-----------------------------------------------------------
            M.A = [E(:);D(:);B(:);O(:)];
            M.G = kG;
            Y = ones(length(knd.B),1);    % To handle childless nodes
            [Qk, ~, Uk] = MessagePassing(M,Y);
            U{k}        = Uk.d(knd.B);  
            for j = 1:numel(U{k})
                U{k}{j} = mp_norm(U{k}{j});
            end
            Q{k} = Qk;
        end
    end

    if ~fac && t == 1
        % If mandatory states are specified, ensure their order is optimised
        %------------------------------------------------------------------
        if isfield(pomdp,'h')
            pomdp.h = mp_pomdp_order(pomdp.h,pomdp.B,Q{1}.s(ind.D));
        end
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

    if isfield(pomdp,'h') % If multiple mandatory states specified, then check whether reached. If so, move on to next state.
        pomdp.h = mp_pomdp_h(pomdp.h,D);
    end

    % Select actions
    %----------------------------------------------------------------------
    % Here the actions are selected directly by sampling from the
    % distribution over paths. 
    
    % vi = find(rand<cumsum(P));    
    [~,vi] = max(P);
    u(:,t) = V(vi(1),:)';

end
%--------------------------------------------------------------------------
POMDP = pomdp;
POMDP.u = u;

function G = mp_POMDP_G(ind, dom)
% This function takes the indices of variables in a POMDP model and
% constructs a causal graph, making use of the domain factors as provided
%--------------------------------------------------------------------------
if ~isempty(ind.B)
    G = cell(ind.B(end) + length(ind.B),1);
else
    G = cell(length(ind.D) + length(ind.A),1);
end
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
    G{length(ind.A) + length(ind.B) + length(ind.D) + length(ind.E) + i} = ind.B(i);
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

if isfield(pomdp,'h') % Account for mandatory states (i.e., use inductive inference)
    H = mp_induction(pomdp.h,pomdp.B,U);
    E = E.*H;
end

Ho = -32*ones(size(E));
HA = -32*zeros(size(E));
C  = -32*zeros(size(E));

for k = 1:size(V,1)
    if E(k) > max(E)/8
        for j = 1:numel(pomdp.A)
            Qo    = mp_dot(pomdp.A{j},U{k}(pomdp.dom.A(j).s)); % Predictive posterior
            C(k)  = C(k) + Qo'*pomdp.C{j};                     % Expected utility
            Ho(k) = Ho(k) - Qo'*mp_log(Qo);                    % Predictive entropy
            Ha    = -sum(pomdp.A{j}.*mp_log(pomdp.A{j}),1);    % Conditional entropy
            HA(k) = HA(k) + mp_dot(Ha,U{k}(pomdp.dom.A(j).s)); % Ambiguity
        end
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

    %======================================================================
    % Account for possible observations we might have made at this step and
    % the subsequent belief-update
    %----------------------------------------------------------------------
    % Ou    = mp_POMDP_comb((cellfun(@(x)size(x,1),pomdp.A))'); % Outcome combinations
    %=======================================================================

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

function h = mp_pomdp_h(h,Q)
% Function that checks whether the mandatory states have been reached. If
% so, it moves on to the next state. 
%--------------------------------------------------------------------------
p = 1;
for i = find(~cellfun(@isempty,h))'
    p = p*Q{i}(h{i}(1));
end
if p > 0.9
    for i = find(~cellfun(@isempty,h))'
        h{i}(1) = [];
    end
end

function h = mp_pomdp_order(h,B,Q)
% Function that optimises the order of the mandatory states
%--------------------------------------------------------------------------

% First, find the lengths of the shortest paths between mandatory states
%--------------------------------------------------------------------------
H     = h(~cellfun(@isempty,h));
B     = B(~cellfun(@isempty,h));
Q     = Q(~cellfun(@isempty,h));
L     = zeros(length(H{1})+1);

for i = 1:numel(H)
    [~,d]       = max(Q{i});
    H{i}(end+1) = d;
end

Qi = cell(size(Q));
Hj = cell(size(H));

for i = 1:numel(H{1})
    for j = 1:numel(H{1})
        for k = 1:numel(H)
            Hj{k} = H{k}(j);
            Qi{k} = zeros(size(Q{k}));
            Qi{k}(H{k}(i)) = 1;
        end
        [~,n] = mp_induction(Hj,B,{Qi});
        L(i,j) = n-1;
    end
end

G = L(1:end-1,1:end-1) == 1;

% Prune graph to ensure is bipartite (minimum pruning)
%-----------------------------------------------------

% First, identify and remove triangles:
[g,k]   = sort(diag(G^3),'ascend');
k       = k(g>0);
k(end)  = [];
while ~isempty(k)
    [~,~,j] = intersect(find(G(:,k(1))),k);
    j       = k(min(j));
    G(k(1),j) = 0;
    G(j,k(1)) = 0;
    L(k(1),j) = L(k(1),j) + 1/2;
    L(j,k(1)) = L(j,k(1)) + 1/2;
    if G(k(1),:)*G*G(:,k(1)) == 0
        k(1) = [];
    end
end

% Then duplicate remaining nodes with degree > 2 and distribute their edges
p = sum(ceil(bsxfun(@max,sum(G)-2,0)/2));
G = padarray(G,p*ones(1,2),0,'pre');
L = padarray(L,p*ones(1,2),64,'pre');
for i = 1:numel(H)
    H{i} = padarray(H{i},p,0,'pre');
end

for n = p:-1:1
    [~,k] = sort(sum(G),'descend');
    [~,j] = sort(sum(G*diag(G(:,k(1)))),'descend');

    for i = 1:numel(H)
        H{i}(n) = H{i}(k(1));
    end

    L(n,:)         = L(k(1),:);
    L(:,n)         = L(:,k(1));
    L(k(1),n)      = 1/2;
    L(n,k(1))      = 1/2;
    L(k(1),j(1:2)) = L(k(1),j(1:2)) + 1/2;
    L(j(1:2),k(1)) = L(j(1:2),k(1)) + 1/2;

    G(n,j(1:2))    = G(k(1),j(1:2));
    G(j(1:2),n)    = G(j(1:2),k(1));    
    G(k(1),j(1:2)) = 0;
    G(j(1:2),k(1)) = 0;
end

[C,d] = mp_graph_cluster(G);
rH    = mp_subgoal(H,L,C,d);

% Repackage in original h structure:
%--------------------------------------------------------------------------
h(~cellfun(@isempty,h)) = rH;

function [H,n] = mp_induction(h,B,Q)
% Function to compute inductive priors over paths
%--------------------------------------------------------------------------
% Note: Currently, this assumes the transition probabilities factorise
% over state factors.

H   = ones(numel(Q),1);
ind = find(~cellfun(@isempty,h));

b = cell(size(ind));
k = zeros(size(ind));
for i = 1:length(ind)
    k(i) = h{ind(i)}(1);
    b{i} = sum(B{ind(i)},3:length(size(B{ind(i)})))> 1/8;
end

z = 0;
n = 0;
while z == 0 && n < 64
    q = true(numel(Q),1);
    for j = 1:numel(Q)
        for i = 1:length(ind)
            K = zeros(size(b{i},1),1);
            K(k(i)) = 1;
            w = Q{j}{ind(i)}'*((b{i}')^n)*K >= 1/2;
            q(j) = q(j) && w;
        end
    end
    if sum(q)>0
        H = q > 0;
        z = 1;
    end
    n = n + 1;
end

function [C,D] = mp_graph_cluster(A)
% Examine graph structure based upon adjacency matrix A, returning cluster
% memberships C and allowable starting vertices D
%-------------------------------------------------------------------------
da    = sum(A);                          % Degree of all vertices
sc    = find(da==0);                     % Find singleton clusters
[~,ind] = setdiff(1:size(A,1),sc);       % Indices for recovery of original order
A(sc,:) = [];                            % Remove from graph
A(:,sc) = [];                            % ...
d     = sum(A);                          % Degree of each vertex
L     = diag(d) - A;                     % Graph Laplacian
[E,~] = eig(L);                          % Eigendecomposition
c     = 1;
while any(sum(abs(E(:,c))>0,2)==0)
    c = 1:(c(end)+1);
end
[~,c] = unique((abs(E(:,c))>0)','rows'); % Remove duplicate clusters
C     = cell(length(c)+length(sc),1);    % Initialise cluster memberships
D     = cell(length(c)+length(sc),1);    % Initialise possible starting vertices

for i = 1:length(c)
    C{i} = ind(abs(E(:,c(i))) > 0);   % Determine members of each cluster
end

for i = 1:length(sc)
    C{i+length(c)} = sc(i);
    D{i+length(c)} = sc(i);
end
for i = 1:length(c)
    if any(da(C{i})==1)
        D{i} = C{i}(da(C{i})==1);     % For clusters with roots/leaves, start there
    else
        D{i} = C{i};                  % Otherwise all start locations equally viable
    end
end

function rH = mp_subgoal(H,L,C,d)
% Takes a set of subgoals H, a distance matrix L, graph clusters C, and 
% leaf/root nodes d, and returns a re-ordered list of subgoals rH.
%--------------------------------------------------------------------------

% Pre-allocate reordered states
%--------------------------------------------------------------------------
rH = cell(size(H));
for i = 1:numel(rH)
    rH{i} = zeros(length(H{i})-1,1);
end

% Reorganise into a set of subgoals
%--------------------------------------------------------------------------
L        = L + diag(Inf*ones(length(H{1}),1)); % Preclude point attractors
L(end,:) = Inf;                                % and preclude return to start
j        = length(rH{1});                      % but set this as initial state
r        = 0;                                  % members left in this cluster
Id       = ones(size(L,1),1)*Inf;
for i = 1:numel(d)
    Id(d{i}) = 0; 
end
Ic       = zeros(size(L,1),1);
for i = 1:length(rH{1})
    if r == 0
    % pick cluster (degree 1 node if available)
        [~,j] = min(L(:,j(1)) + Id);
        c     = find(cellfun(@(x)ismember(j(1),x),d));
        Ic    = ones(size(Ic))*Inf;
        Ic(C{c}) = 0;
        r     = length(C{c})-1;
    else
        [~,j] = min(L(:,j(1)) + Ic);
        r     = r - 1; 
    end
    for k = 1:numel(rH)
        rH{k}(i) = H{k}(j(1));
    end
    L(j(1),:) = Inf;
end