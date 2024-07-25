function [Q, F, U] = MessagePassing(M,Y)
% Message passing scheme for solving generic generative models.
% FORMAT [Q, F, U] = MessagePassing(M,Y)
% M - Generative model
% Y - Data
% Q - Posterior marginals
% F - Log evidence (marginal likelihood)
% U - Final set of messages
%__________________________________________________________________________
% This routine provides a generic inversion scheme for models specified in
% terms of a set of prior and conditional probabilities. The specification
% in terms of factors of a probability distribution allows for application
% of a belief propagation scheme for a range of different models.
% Specification of categorical and Dirichlet factor messages are built in,
% with options to specify custom messages for other sorts of factor.
%
% A key component of the model specification is a field M.G which includes
% a cell structure with one element for every factor in the generative
% model. This element includes a vector of indices for every variable upon
% which that factor depends. Each factor gives the probability for a single
% variable, implying variable j depends upon factor j. An empty element in
% M.G implies the associated factor is a prior.
%__________________________________________________________________________

% Thomas Parr


% Options
%--------------------------------------------------------------------------
try M.nograph; catch, M.nograph = 0;  end  % Plotting
try M.noprint; catch, M.noprint = 0;  end  % Print iterations
try M.acyclic; catch, M.acyclic = 0;  end  % Is graph supplied acyclic?
try M.Nmax;    catch, M.Nmax    = 32; end  % Max iterations
try M.MAP;     catch, M.MAP     = 0;  end  % Use MAP states for learning
try M.net;     catch, M.net     = 0;  end  % Plot network

Ni = M.Nmax; % Maximum number of iterations

% Preliminaries
%--------------------------------------------------------------------------
G  = M.G;    % Causal graph

N  = numel(G);
m  = zeros(1,N);

% Normalise Dirichlet counts if using
%--------------------------------------------------------------------------
try
    a = M.a;
    g = M.g;
    A = cell(1,N);
    for i = 1:length(g)
        if g(i) == -1
            A{i} = mp_norm(M.A{i});
        else
            A{i} = mp_norm(M.a{g(i)});
        end
    end
    if ~M.noprint
        disp('Using Dirichlet priors as specified')
    end

catch
    for i = 1:length(G)
        A{i} = mp_norm(M.A{i});
    end
    if ~M.noprint
        disp('Dirichlet priors not specified: assume fixed parameters')
    end
end

% Find number of levels for each variable
%--------------------------------------------------------------------------
Ln = zeros(numel(A),1);
for i = 1:N
    if isfield(A{i},'f')    % If factor is defined in terms of a function
        Ln(i) = A{i}.Nd;    % Use dimension of descending message from factor
    else
        Ln(i) = size(A{i},1);
    end
end

% Classify variables as hidden states or outcomes
%--------------------------------------------------------------------------
for i = 1:N       % Loop over factors
    for j = 1:N   % Loop over variables
        if ismember(j,G{i})
            m(j) = 1;
        end
    end
end
yi = find(1-m);  % Indices of childless variables

% Convert indexed data to one-hot vectors
%--------------------------------------------------------------------------
if iscell(Y)
    y = Y; % Assume supplied as vectors of evidence (unless generated by other function)
else
    y = cell(1,length(yi));
    for i = 1:numel(Y)
        y{i} = zeros(Ln(yi(i)),1);
        y{i}(Y(i),1) = 1;
    end
end

% Initialise messages
%--------------------------------------------------------------------------
aU = cell(numel(G),numel(G));        % Ascending messages
dU = cell(numel(G),1);               % Descending messages

for i = 1:numel(G)
    for j = 1:numel(G)
        if ismember(i,G{j})
            aU{i,j} = ones(Ln(i),1); % Messages from j to i
        end
    end
    dU{i} = ones(Ln(i),1);           % Messages from i
end

CD = zeros(N,N);
for k = 1:numel(G)
    for j = G{k}
        CD(k,j) = 1;                 % Conditional dependencies
    end
end
CD = logical(CD);

% Identify implicit Dirac delta factors
%--------------------------------------------------------------------------
dirac = cell(N,1);
for i = 1:N
    dirac{i} = find(CD(:,i));
end

% Figure (unless disabled)
%--------------------------------------------------------------------------

if ~M.nograph 
    Fmp = figure('Name','Message Passing','Color','w'); clf
    Ad      = CD + CD';                     % Adjacency matrix
    Lap     = diag(sum(Ad))  - Ad;          % Graph Laplacian 
    [u,~]   = eig(Lap);                     % Eigenvectors 
    u       = u(:,[2 3]);                   % Project to second and third
end

criterion  = zeros(1,4); % For convergence in case of cyclic graphs
Fi        = [];          % Record log marginal likelihood at each iteration 

% Initialise message wavefronts
%----------------------------------------------------------------------
   
ai  = find(~m);           % Ascending messages
pri = zeros(numel(G),1);  % Prior index
for l = 1:numel(G)
    pri(l) = isempty(G{l});
end
di  = find(pri);          % Desending messages

% Iterate through factors
%--------------------------------------------------------------------------
for i = 1:Ni
    Fa = zeros(numel(G),1);
    for j = 1:numel(G)
        
        update = mp_reactive(M,j,ai,di);

        if update
            if isempty(G{j})            % If factor j is a prior...
                dU{j} = A{j};           %... then the descending message is the prior
                AU    = ones(size(aU{j,find(CD(:,j),1,"first")}));
                for k = 1:size(aU,1)
                    if CD(k,j) % if k is a child of j
                        AU = mp_norm(AU.*aU{j,k});
                    end
                end
                Fa(j) = mp_log(A{j}'*AU);

            elseif ~m(j)               % If factor j is a likelihood...
                % Compile messages to factor
                %----------------------------------------------------------
                AU  = y{j + 1 - min(yi)};
                DU = dU(G{j});
                for k = 1:numel(DU)    % Augment with any implicit Dirac nodes
                    if ~isempty(dirac{G{j}(k)})
                        for l = setdiff(dirac{G{j}(k)}',j)
                            DU{k} = DU{k}.*aU{G{j}(k),l};
                        end
                    end
                end

                %  Update messages from factor
                %----------------------------------------------------------
                if isfield(A{j},'f')
                    [Ua, Ud, Fa(j)] = A{j}.f(AU,DU);    % Use arbitrary factor if specified
                else
                    [Ua, Ud, Fa(j)] = mp_Cat_messages(AU,DU,A{j});
                    if M.MAP
                        for k = 1:numel(Ua)
                            Ua{k} = (Ua{k} == max(Ua{k}(:)));
                        end
                        Ud = (Ud == max(Ud(:)));
                    end
                end
                for k = 1:length(G{j})
                    aU{G{j}(k),j} = Ua{k};
                end
                dU{j} = Ud;
            else
                % Compile messages to factor
                %----------------------------------------------------------
                DU = dU(G{j});
                for k = 1:numel(DU) % Augment with any implicit Dirac nodes
                    if ~isempty(dirac{G{j}(k)})
                        for l = setdiff(dirac{G{j}(k)}',j)
                            DU{k} = DU{k}.*aU{G{j}(k),l};
                        end
                    end
                end

                % Product of all ascending messages to factor:
                AU    = ones(size(aU{j,find(CD(:,j),1,"first")}));
                for k = 1:size(aU,1)
                    if CD(k,j) % if k is a child of j
                        AU = mp_norm(AU.*aU{j,k});
                    end
                end

                %  Update messages from factor
                %----------------------------------------------------------
                if isfield(A{j},'f')
                    [Ua, Ud, Fa(j)] = A{j}.f(AU,DU);    % Use arbitrary factor if specified
                else
                    [Ua, Ud, Fa(j)] = mp_Cat_messages(AU,DU,A{j});
                    if M.MAP
                        for k = 1:numel(Ua)
                            Ua{k} = (Ua{k} == max(Ua{k}(:)));
                        end
                        Ud = (Ud == max(Ud(:)));
                    end
                end
                for k = 1:length(G{j})
                    aU{G{j}(k),j} = Ua{k};
                end
                dU{j} = Ud;
            end
        end
    end


% Compute evidence for model
%--------------------------------------------------------------------------
% This works on the principle that we assume all priors are implicitly
% conditioned upon a model. As such, the product of the ascending messages
% from the priors play the role of a marginal likelihood. This is exact when
% the model is acyclic, and approximate when there are cycles.

    if ~exist('F0','var'), F0 = 0; end
    
% If Dirichlet priors are not specified...
%--------------------------------------------------------------------------
    if ~isfield(M,'a'), F = sum(Fa);

% If Dirichlet priors are specified...
%--------------------------------------------------------------------------
% Compute the messages from the Dirichlet prior to the model (a ratio of
% two beta functions) and pass messages from all other Dirichlet
% distributions.

    else
        F  = 0;
        ap = cell(numel(a),1); % Initialise posterior Dirichlet counts
        for k = 1:numel(a)
            
            % Identify associated factors in G (may be more than one)
            %--------------------------------------------------------------
            Gk = find(g==k);

            % Get ascending and descending messages to factor
            %--------------------------------------------------------------
            AUD = ones(size(a{k},1),1); % ascending
            for j = 1:length(Gk)
                for z = 1:size(aU,1)
                    if ismember(Gk(j),G{z})
                        AUD = mp_norm(AUD.*aU{Gk(j),z});
                    end
                end
            end
            nA = size(a{k});
            if length(nA) > 1 % factor k is not a prior
                AUD = repmat(AUD,[1,nA(2:end)]);
                DUD = ones(size(a{k}));    % descending
                for j = 1:length(Gk)
                    si = G{Gk(j)}; 
                    for z = 1:length(si)
                        dA = circshift(1:length(nA),z);
                        DU = permute(dU{si(z)},dA);
                        dA = nA;
                        dA(z+1) = 1;
                        DU = repmat(DU,dA);
                        DUD = DUD.*DU;
                    end
                end
            else
                DUD = ones(size(a{k}));
            end

            % Update posterior and compute contributions to marginal likelihood
            %--------------------------------------------------------------
            if ~length(Gk) == 0
                da    = (a{k}>1/64).*AUD.*DUD;
                ap{k} = a{k} + da;
                F = F - mp_KL_dir(ap{k},a{k});
            else
                F = F + Fa(k);
            end
        end
        Q.a = ap;
    end
    dF = F - F0;
    F0 = F;
    Fi = [Fi F];

    % Determine where messages have reached
    %----------------------------------------------------------------------
    
    CDi = CD^i;
    ai  = find((~m)*CDi);           % Ascending messages
    pri = zeros(numel(G),1);        % Prior index
    for l = 1:numel(G)
        pri(l) = isempty(G{l});
    end
    di  = find(CDi*pri);            % Desending messages

    % Graphics
    %----------------------------------------------------------------------
    if exist('Fmp','var')
        
        subplot(2,2,4)
        title('Conditional Dependencies')
        imagesc(1-CD), colormap gray, hold on
        xlabel('Conditioned upon')
        ylabel('Dependent variable')
        ylims = ylim;
        xlims = xlim;

        % Overlay wavefront of ascending and descending messages
        % originating in roots and leaves of the graph.
        %------------------------------------------------------------------
        for k = 1:length(ai)
            plot([ai(k),ai(k)],ylims,'b')
        end
        for k = 1:length(di)
            plot(xlims,[di(k) di(k)],'r')
        end
        axis square

        subplot(2,2,1), cla
        grid on
        bar(1:length(Fi), Fi, 'cyan')
        xlim([1 length(Fi)+1])
        xlabel('Iteration')
        ylabel('Log Evidence')
        title('Marginal likelihood')
        axis square

        % Compute marginal probabilities for each state
        %----------------------------------------------------------------------
        Q.s = cell(numel(G) - length(yi),1);
        for k = 1:numel(Q.s)
            AU = ones(size(aU{k,find(CD(:,k),1,"first")}));
            for j = 1:size(aU,1)
                if CD(j,k) % if k is a parent of j
                    AU = mp_norm(AU.*aU{k,j});
                end
            end
            if isfield(A{k},'g') % If an alternative rule is given for computing marginals
                Q.s{k} = A{k}.g(dU{k},AU);
            else
                Q.s{k} = mp_softmax(mp_log(dU{k}) + mp_log(AU));
            end
        end

        subplot(2,2,3)
        Lmax = max(Ln);
        PP = ones(numel(Q.s),Lmax);
        for k = 1:numel(Q.s)
            PP(k,:) = [1-Q.s{k}' ones(1,Lmax-Ln(k))];
        end
        imagesc(PP), clim([0 1]); colormap gray
        title('Posterior probabilities (states)')
        xlabel('Level')
        ylabel('Marginal')
        axis square

        if M.net
            Ad      = CD + CD';                     % Adjacency matrix
            Lap     = diag(sum(Ad))  - Ad;          % Graph Laplacian
            [u,~]   = eig(Lap);                     % Eigenvectors
            u       = u(:,[2 3]);                   % Project to second and third
            [ei, ej] = find(CD);                    % Edges
            subplot(2,2,2)
            for k = 1:8
                plot(u(:,1),u(:,2),'.','MarkerSize',16,'Color',[0.5 0.5 0.5]), hold on
                plot([u(ei,1), u(ej,1)]',[u(ei,2), u(ej,2)]','LineWidth',1,'Color',[0.5 0.5 0.5])
                plot((8-k)*u(ei,1)/7+(k-1)*u(ej,1)/7,(8-k)*u(ei,2)/7+(k-1)*u(ej,2)/7,'.r')
                plot((8-k)*u(ej,1)/7+(k-1)*u(ei,1)/7,(8-k)*u(ej,2)/7+(k-1)*u(ei,2)/7,'.b')
                axis square
                axis off
                title('Network diagram')
                hold off
                drawnow
            end
        else
            drawnow
        end
    end

    % Convergence
    %----------------------------------------------------------------------
    % To terminate, we need all messages originating in roots or leaves to
    % have reached a lead or root (respectively) and we need the log
    % evidence to have converged.

    criterion  = [abs(dF)<1e-1, criterion(1:end-1)];
    if isempty(ai) && isempty(di)
        if ~M.noprint
            fprintf('Iteration: %d. Message passing complete if acyclic graph. dF: %g\n', i,dF)
        end
        if  M.acyclic
            U.a = aU;
            U.d = dU;
            % Compute marginal probabilities for each state
            %--------------------------------------------------------------
            Q.s = cell(numel(G) - length(yi),1);
            for k = 1:numel(Q.s)
                AU = ones(size(aU{k,find(CD(:,k),1,"first")}));
                for j = 1:size(aU,1)
                    if CD(j,k) % if k is a parent of j
                        AU = mp_norm(AU.*aU{k,j});
                    end
                end
                if isfield(A{k},'g') % If an alternative rule is given for computing marginals
                    Q.s{k} = A{k}.g(dU{k},AU);
                else
                    Q.s{k} = mp_softmax(mp_log(dU{k}) + mp_log(AU));
                end
            end
            return
        end
        if sum(criterion) == length(criterion)
            U.a = aU;
            U.d = dU;
            if ~M.noprint
                disp('Convergence')
            end

            % Compute marginal probabilities for each state
            %--------------------------------------------------------------
            Q.s = cell(numel(G) - length(yi),1);
            for k = 1:numel(Q.s)
                AU = ones(size(aU{k,find(CD(:,k),1,"first")}));
                for j = 1:size(aU,1)
                    if CD(j,k) % if k is a parent of j
                        AU = mp_norm(AU.*aU{k,j});
                    end
                end
                if isfield(A{k},'g') % If an alternative rule is given for computing marginals
                    Q.s{k} = A{k}.g(dU{k},AU);
                else
                    Q.s{k} = mp_softmax(mp_log(dU{k}) + mp_log(AU));
                end
            end
            return
        end
    else
        if ~M.noprint
            fprintf('Iteration: %d. Asc. messages at: [%s]. Desc. messages at: [%s]. dF: %g\n',i,num2str(unique(ai)),num2str(unique(di')),dF)
        end

    end
end

function [Ma, Md, F] = mp_Cat_messages(aM,dM,A)
% Computation of ascending and descending messages from categorical factor
% FORMAT [Ma, Md] = mp_Cat_messages(aM,dM,A)
% aM - ascending messages to factor    (from children)
% dM - descending messages to factor   (from parents and coparents)
% Ma - ascending messages from factor  (to parents and coparents)
% Md - descending messages from factor (from children)
% A  - probability tensor whose first dimension relates to children, with
%      subsequent dimensions relating to (co)parents.
%__________________________________________________________________________
% This function takes messages to a categorical probability factor and
% computes messages from this factor. By default, a belief-propagation
% scheme is assumed. The messages are normalised for numerical stability.
%__________________________________________________________________________

% Compute outgoing descending messages
%--------------------------------------------------------------------------
Md = mp_norm(mp_dot(A,dM));

% Compute outgoing ascending messages
%--------------------------------------------------------------------------
Ma = cell(size(dM));

for i = 1:numel(Ma)
    Ma{i} = mp_norm(mp_dot(A,[{aM},dM(:)'],i+1));
end

% Compute contribution to log marginal likelihood
%--------------------------------------------------------------------------
F = mp_log(aM'*Md);

function d = mp_KL_dir(a,b)
% KL-Divergence between Dirichlet distributions
%--------------------------------------------------------------------------
a(b==0) = [];
b(b==0) = [];

d = mp_betaln(a) - mp_betaln(b) + ...
    sum((a-b).*max(psi(a)-psi(sum(a)),-32),1);
d = sum(d(:));

function a = mp_betaln(b)
a = max(gammaln(sum(b)),-32) - max(sum(gammaln(b)),-32);

function update = mp_reactive(M,j,ai,di)
% Determines whether update is required in acyclic graph depending upon
% whether any new information has reached this node
%--------------------------------------------------------------------------

if M.acyclic
    if ismember(j,[di(:); ai(:)])
        update = true;
    else
        update = false;
    end
else
    update = true;
end