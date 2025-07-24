function POMDP = mp_POMDP(pomdp)
% Format POMDP = mp_POMDP(pomdp)
% This function solves a Partially Observed Markov Decision Process
% generative model using the MessagePassing.m belief-propagation scheme.
% Please see DEMO_POMDP_tMaze.m in the Generic Demos folder for an example
% as to how these models should be specified.
%
% In addition to message passing of a sort that might be anticipated for a
% hidden Markov model, this scheme allows for the evaluation of alternative
% trajectories one could pursue. These can either be evaluated in terms of
% their expected free energy as is standard in active inference models, or
% through user-specified functions. In addition, this scheme incorporates a
% form of inductive inference (a recent theoretical development in active
% inference models) to identify plausible paths to evaluate, avoiding the
% need for exhaustive tree searches if we know, a priori, that certain
% states must be visited. If multiple mandatory states are specified, this
% scheme triages these using a graph-clustering method to order them
% efficiently - i.e., uses a form of subgoaling.
%
% Unlike some previous active inference schemes, this routine allows one to
% specify dependencies among the trajectories of hidden state factors
% directly, so that some transitions are dependent upon (non-policy)
% factors. This allows for more complex dynamics for non-hierarchical
% models.
%
% Furthermore, there is more flexibility for user-defined generative
% processes, which may be specified as functions. For hierarhical or
% modular architectures, these functions may include the evaluation of
% another POMDP structure.
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
try con = pomdp.con; catch, try con = 1:numel(E); catch, try con = 1:numel(e); catch, con = []; end, end, end % controllable paths

% Planning horizon
try N = pomdp.N;     catch, N = 1;   end % assume 1-step ahead unless otherwise specified

% Option to factorise over time
try fac = pomdp.fac; catch, fac = 0; end % assume no factorisation over time unless specified

% Identify indices of each variable type
%--------------------------------------------------------------------------

try ind.E = 1:numel(E);                catch, try ind.E = 1:numel(e); catch, ind.E = 0; end,  end % Indices to identify all paths
try ind.D = (1:numel(D)) + ind.E(end); catch, ind.D = (1:numel(d)) + ind.E(end);              end % Indices to identify all states
try ind.B = (1:numel(B)) + ind.D(end); catch, ind.B = (1:numel(b)) + ind.D(end);              end % Indices to identify all next states
try ind.A = (1:numel(A)) + ind.B(end); catch, ind.A = (1:numel(a)) + ind.B(end);              end % Indices to identify all observations

ind.E(ind.E==0) = [];

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

% Prune dependencies as appropriate for transition model
%--------------------------------------------------------------------------
if ~fac
    rG = G;
else
    rG = kG;
end

ib  = cellfun(@(x)isfield(x,'i'),B);  % Find context-dependent Bs

if any(ib)
    for i = find(ib)'
        if ~fac
            rG{ind.B(i)}(2:end-1) = [];
        else
            rG{knd.B(i)}(2:end-1) = [];
        end
    end
end

% Construct uninformative factors to deal with childless nodes
%--------------------------------------------------------------------------
O = cell(length(ind.B),1);
for i = 1:numel(O)
    try
        O{i} = ones(1,B{i}.Nd);
    catch
        try
            O{i} = ones(1,size(B{i},1));
        catch
            O{i} = ones(1,size(b{i},1));
        end
    end
end

% Path combinations
%--------------------------------------------------------------------------
if ~isempty(con)
    Ne = zeros(size(ind.E));

    for k = 1:length(ind.E)
        try Ne(k) = length(E{k}); catch, Ne(k) = length(e{k}); end
    end

    V = mp_POMDP_comb(Ne(con));  % Determine controllable path combinations
    u = zeros(length(Ne),T-1);   % Initialise choice array
else
    Ne = 1;
    u = ones(1,T-1);
    V = 1;
end

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
        Qo      = cell(size(A)); % Posterior predictive distribution
        for g = 1:numel(Qo)
            if isfield(A{g},'f')
                [~,Qo{g}] = pomdp.A{g}.f(ones(pomdp.A{g}.Nd,1),D(pomdp.dom.A(g).s));
            else
                Qo{g}     = mp_dot(pomdp.A{g},D(pomdp.dom.A(g).s));
            end
        end
        if t > 1
            s       = pomdp.s(:,t-1);
            [y,s]   = pomdp.gen(s,u(:,t-1),Qo,pomdp);
            pomdp.s(:,t) = s;
        else
            s       = pomdp.s(:,1);
            y       = pomdp.gen(s,[],[],pomdp);
        end
        if isfield(y,'o') % This indicates there is other information to retain
            pomdp.o(:,t) = y.o;
            pomdp.y(:,t) = rmfield(y,'o');
            y            = y.o;
        else
            pomdp.o(:,t) = y;
        end
    end

    if fac
        % Construct and solve vertical model
        %-----------------------------------------------------------
        if ~isfield(pomdp,'concat')
            M.A = [D(:);A(:)];
            M.G = jG;
            jQ  = MessagePassing(M,y); 
            D   = jQ.s;
        else
            D   = mp_one2one(A,D,pomdp,y);
        end

        if t == 1
            % If mandatory states are specified, ensure their order is optimised
            %--------------------------------------------------------
            if isfield(pomdp,'h') && any(cellfun(@(x) length(x)>1,pomdp.h))
                pomdp.H = mp_pomdp_order(pomdp.h,pomdp.B,D);
            end
        end    
    end

    % Iterate over (controllable) paths
    %------------------------------------------------------------------
    for k = 1:size(V,1) 

        % Construct model
        %--------------------------------------------------------------
        if isempty(con) % HMM
            E = {};
        else
            for j = con
                E{j} = mp_POMDP_oh(Ne(j),V(k,j)); % Set controllable path
            end
        end
        
        % Determine appropriate functions to use given current beliefs
        %------------------------------------------------------------------
        rB = mp_pomdp_rB(D,B,ib,pomdp.dom);

        if ~fac
            if iscell(y) % If probabilistic outcomes are supplied
                y = cellfun(@(x)x.',y,'UniformOutput',false);
                M.A = [E(:);D(:);rB(:);A(:);O(:);y(:)];
                if numel(M.A)>numel(M.G)
                    M.G = [M.G; num2cell(ind.A')];
                end
                Y = [ones(length(ind.B),1);ones(numel(y),1)]; % To handle childless nodes
            else
                M.A = [E(:);D(:);rB(:);A(:);O(:)];
                Y = [y; ones(length(ind.B),1)]; % To handle childless nodes
            end

            % Solve model
            %------------------------------------------------------------
            [Qk, ~, Uk] = MessagePassing(M,Y);
            U{k}        = Uk.d(ind.B);  
            Q{k}        = Qk;
        else
            % Solve horizontal model
            %-----------------------------------------------------------
            M.G = rG;
            M.A = [E(:);D(:);rB(:);O(:)];
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
            pomdp.H = mp_pomdp_order(pomdp.h,pomdp.B,Q{1}.s(ind.D));
        end
    end    

    % Equip controllable paths with priors and evaluate posteriors
    %----------------------------------------------------------------------
    if isfield(pomdp,'path')
        P = pomdp.path(pomdp,U,N,V);   % If function specified for path priors
    elseif ~isempty(con)
        P = mp_path(pomdp,U,N,V);      % Otherwise use default (based on expected free energy)
    else
        P = 1;                         % Unless there are no controllable variables
    end

    % Select actions
    %----------------------------------------------------------------------
    % Here the actions are selected directly by sampling from the
    % distribution over paths. 
    
    % vi = find(rand<cumsum(P));    
    [~,vi] = max(P);
    u(:,t) = V(vi(1),:)';

    if isfield(pomdp,'proprioception')
        if pomdp.proprioception % If actions are also observable
            P        = zeros(size(P));
            P(vi(1)) = 1;
        end
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

    if isfield(pomdp,'H') % If multiple mandatory states specified, then check whether reached. If so, move on to next state.
        pomdp.H = mp_pomdp_h(pomdp.H,D);
        if isfield(pomdp,'hstop') && pomdp.hstop
            if all(cellfun(@isempty,pomdp.H))
                pomdp.T = min(t+2,pomdp.T); % If all mandatory states reached, allow an additional full iteration then break  
            end
            if t == pomdp.T
                pomdp.Q = pomdp.Q{1:t,:};
                pomdp.P = pomdp.P{1:t,:};
                u       = u(:,1:t);
                break
            end
        end
    end
end
%--------------------------------------------------------------------------

% Bayesian smoothing
%--------------------------------------------------------------------------
if isfield(pomdp,'smooth')
    M.A = [pomdp.D(:);pomdp.P(1:end-1);repmat(B(:), T-1, 1);repmat(A(:), T, 1)];
    
    ind.D = 1:numel(D);
    ind.E = max(ind.D) + (1:numel(pomdp.P)-1);
    ind.B = max(ind.E) + (1:numel(B)*(T-1));
    ind.A = max(ind.B) + (1:numel(A)*T);
    M.G   = mp_POMDP_B(ind, dom, T);
    M.acyclic = false;
    BS    = MessagePassing(M,pomdp.o(:));
    pomdp.BS.s = reshape(BS.s([ind.D ind.B]),[],T);
    pomdp.BS.u = BS.s([ind.E]);
end

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

function G = mp_POMDP_B(ind, dom, T)
% This function takes the indices of variables in a POMDP model and
% constructs a causal graph, making use of the domain factors as provided.
% Unlike the mp_POMDP_G routine, this function deals with the full model,
% enabling Bayesian smoothing.
%--------------------------------------------------------------------------

G  =  cell(max(ind.A),1);
Nb =  length(ind.B)/(T-1);
Ne =  length(ind.E)/(T-1);
Na =  length(ind.A)/T;

% Connect states at t = 2 to states at t = 1 and paths from 1 to 2
%--------------------------------------------------------------------------
for i = ind.D
    G{ind.B(i)} = [i,ind.D(dom.B(i).s),ind.E(dom.B(i).u)];
end

% And for observations
%--------------------------------------------------------------------------
for i = 1:Na
    G{ind.A(i)} = ind.D(dom.A(i).s);
end

% And the same for subsequent times
%--------------------------------------------------------------------------
for t = 2:T-1
    for i = 1:Nb
        G{ind.B((t-1)*Nb+i)} = [ind.B((t-2)*Nb+i),ind.B((t-2)*Nb+dom.B(i).s),ind.E((t-1)*Ne+dom.B(i).u)];
    end
end

% And add observations
%--------------------------------------------------------------------------
for t = 2:T
    for i = 1:Na
        G{ind.A((t-1)*Na + i)} = ind.B((t-2)*Nb + dom.A(i).s);
    end
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
% and the combinations (V) of controllable paths we need to consider.
%--------------------------------------------------------------------------

E = ones(size(V,1),1);
for i = 1:size(V,2)
    if size(pomdp.E{i},2)>1 % If time-specific priors are given
        j  = sum(cellfun(@(x)~isempty(x),pomdp.P(:,i)));
        E  = E.*pomdp.E{i}(V(:,i),min(size(pomdp.E{i},2),j+1));  
    else
        E  = E.*pomdp.E{i}(V(:,i));   
    end
end

if isfield(pomdp,'H') % Account for mandatory states (i.e., use inductive inference)
    H = mp_induction(pomdp.H,pomdp.B,U);
    E = E.*H;
end

Ho = -32*ones(size(E));
HA = -32*ones(size(E));
C  = -32*ones(size(E));

for k = 1:size(V,1)
    if E(k) > max(E)/8
        for j = 1:numel(pomdp.A)
            if isfield(pomdp.A{j},'f')
                [~,Qo,~,Ha] = pomdp.A{j}.f(ones(pomdp.A{j}.Nd,1),U{k}(pomdp.dom.A(j).s)); % Predictive posterior and conditional entropy
            else
                Qo     = mp_dot(pomdp.A{j},U{k}(pomdp.dom.A(j).s));   % Predictive posterior
                Ha     = -sum(pomdp.A{j}.*mp_log(pomdp.A{j}),1);      % Conditional entropy
            end
            C(k)  = C(k) + Qo'*pomdp.C{j};                            % Expected utility
            Ho(k) = Ho(k) - Qo'*mp_log(Qo);                           % Predictive entropy
            if isscalar(Ha)
                HA(k) = HA(k) + Ha;
            else
                HA(k) = HA(k) + mp_dot(Ha,U{k}(pomdp.dom.A(j).s));    % Ambiguity
            end
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

if isfield(pomdp,'gamma')
    G = G*pomdp.gamma;
end

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

function rB = mp_pomdp_rB(D,B,ib,dom)
% Where the transitions are specified by multiple alternative function
% handles, this part of the routine selects the functional handle that is
% most probable based upon the prior information available.
%--------------------------------------------------------------------------

rB = B;

for i = find(ib(:))'
    rD = D(dom.B(i).s(B{i}.i.d));
    z  = -ones(numel(B{i}.i.v),1);
    for jf = find(~cellfun(@isempty,B{i}.i.v))
        z(jf) = 1;
        for js = 1:numel(rD)
            z(jf) = z(jf)*rD{js}(B{i}.i.v{jf}(js));
        end
    end
    if any(z>1/length(z)) % If there is evidence for a non-empty criterion
        rB{i}.f = B{i}.f{z>1/length(z)};
    else                  % Otherwise set B to be equal to the function for the empty criterion
        rB{i}.f = B{i}.f{z<0};
    end
end

function D = mp_one2one(A,D,pomdp,y)
% For models in which there is a 1:1 relationship between some of the
% states and observations, one can in principle achieve greater efficiency
% by concatenating the relevant factors and modalities to perform array
% operations. This assumes the same form for the likelihood functions for
% all pairs of state and observation.
%--------------------------------------------------------------------------

tD = D;
for ic    = 1:length(pomdp.concat)
    si    = pomdp.concat(ic).si;
    is    = pomdp.concat(ic).is;
    dim   = cellfun(@length,D(is))';
    iS    = ones(dim);
    ss    = zeros(numel(iS),length(dim));
    subs  = cell(1,length(dim));
    for i = 1:numel(iS)
        [subs{:}] = ind2sub(size(iS),i);
        ss(i,:)   = [subs{:}];
        for j = 1:length(pomdp.concat(ic).is)
            iS(i) = iS(i)*D{pomdp.concat(ic).is(j)}(subs{j});
        end
    end
    if isempty(iS)
        oi    = pomdp.concat(ic).map(si);
        Af = @(x) A{oi(1)}.f(x,{1});
        Nd    = A{oi(1)}.Nd;
        OI    = arrayfun(@(x)mp_POMDP_oh(Nd,x),y(oi),'UniformOutput',false);
        l     = cellfun( Af,OI,'UniformOutput',false);
        l     = vertcat(l{:});
        L     = l(:,1);
    else
        L = num2cell(zeros(length(si),1));
        iS(iS<1/length(iS)) = 0;
        sI = iS;
        for i = find(iS>1/length(iS(:)))'
            si    = pomdp.concat(ic).si;
            oi    = pomdp.concat(ic).map(si,ss(i,:));
            si    = si(~isnan(oi));
            oi    = oi(~isnan(oi));
            io    = [{1} num2cell(ss(i,:))];
            Af    = @(x) A{oi(1)}.f(x,io);
            Nd    = A{oi(1)}.Nd;
            OI    = arrayfun(@(x)mp_POMDP_oh(Nd,x),y(oi),'UniformOutput',false);
            l     = cellfun(Af,OI,'UniformOutput',false);
            l     = vertcat(l{:});
            L(si) = cellfun(@(x,y)y+x*iS(i),l(:,1),L(si),'UniformOutput',false);
            K     = cellfun(@(x,y) x'*y,l(:,1),D(si),'UniformOutput',false);
            sI(i) = prod([K{:}]);
        end
        L  = L(si);
        sI = sI.*iS;
        for i = 1:length(dim)
            di = 1:length(dim);
            di(i) = [];
            D{is(i)} = mp_norm(reshape(sum(sI,di),[size(sI,i),1]));
        end
    end
    tD(si) = cellfun( @(x,y) mp_norm(x.*y) ,D(si),L,'UniformOutput',false);
    D(si) = tD(si);
end