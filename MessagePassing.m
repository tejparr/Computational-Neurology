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
options  = {'nograph', 'noprint', 'acyclic', 'Nmax', 'MAP', 'net', 'Q'};
defaults = {0, 0, 0, 32, 0, 0, 1};
for i = 1:length(options)
    try
        M.(options{i});
    catch
        M.(options{i}) = defaults{i};
    end
end

Ni = M.Nmax; % Maximum number of iterations

% Preliminaries
%--------------------------------------------------------------------------
G  = M.G;    % Causal graph

N = numel(G);

% Pre-allocate arrays
%--------------------------------------------------------------------------
aU    = cell(N,N);     % Ascending messages
dU    = cell(N,1);     % Descending messages
dirac = cell(N,1);     % Dirac delta factors
Fa    = zeros(N,1);    % Factor contributions to evidence

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
    for i = 1:numel(M.A)
        if ~isfield(M.A{i},'f') && ~issparse(M.A{i}) && size(M.A{i},1)~=1
            A{i} = mp_norm(M.A{i});
        else
            A{i} = M.A{i};
        end
    end
    if ~M.noprint
        disp('Dirichlet priors not specified: assume fixed parameters')
    end
end

% Find number of levels for each variable
%--------------------------------------------------------------------------
Ln = zeros(N,1);
for i = 1:N
    if isfield(A{i},'f')    % If factor is defined in terms of a function
        Ln(i) = A{i}.Nd;    % Use dimension of descending message from factor
    else
        Ln(i) = size(A{i},1);
    end
end

% Classify variables as hidden states or outcomes
%--------------------------------------------------------------------------
m = zeros(1,N);
for i = 1:N
    m(G{i}) = 1;
end
yi = find(1-m);  % Indices of childless variables

% Convert indexed data to one-hot vectors
%--------------------------------------------------------------------------
if iscell(Y)
    y = Y; % Assume supplied as vectors of evidence (unless generated by other function)
else
    y = cell(1,length(yi));
    for i = 1:numel(Y)
        y{i} = sparse(zeros(Ln(yi(i)),1));
        y{i}(Y(i),1) = 1;
    end
end

% Build conditional dependencies matrix
%--------------------------------------------------------------------------
% Create indices for sparse matrix construction
[ii, jj] = deal(cell(N,1));
for k = 1:N
    ii{k} = repmat(k, 1, length(G{k}));
    jj{k} = G{k};
end
CD = sparse([ii{:}], [jj{:}], true, N, N);


% Initialise messages
%--------------------------------------------------------------------------
for i = 1:N
    for j = 1:N
        if CD(j,i)
            aU{i,j} = ones(Ln(i),1); % Messages from j to i
        end
    end
    dU{i} = ones(Ln(i),1);           % Messages from i
end

% Identify implicit Dirac delta factors 
%--------------------------------------------------------------------------
for i = 1:N
    dirac{i} = find(CD(:,i));
end

% Figure (unless disabled)
%--------------------------------------------------------------------------
if ~M.nograph 
    figure('Name','Message Passing','Color','w'); clf
end

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
criterion = zeros(1,4);
Fi        = zeros(1,Ni);

% Main message passing loop
%--------------------------------------------------------------------------
F0 = 0;
for i = 1:Ni
    for j = [di' ai]
        if isempty(G{j})            % If factor j is a prior...
            dU{j} = A{j};           %... then the descending message is the prior
            AU    = ones(size(aU{j,find(CD(:,j),1,"first")}));
            for k = find(CD(:,j))'  % find children of j             
                AU = mp_norm(AU.*aU{j,k});
            end
            Fa(j) = mp_log(A{j}'*AU);

        elseif ~m(j)               % If factor j is a likelihood...
            % Compile messages to factor
            %--------------------------------------------------------------
            AU = y{j + 1 - min(yi)};
            DU = dU(G{j});
            for k = 1:numel(DU)    % Augment with any implicit Dirac nodes
                if ~isempty(dirac{G{j}(k)})
                    for l = mp_setdiff(dirac{G{j}(k)}',j)
                        DU{k} = DU{k}.*aU{G{j}(k),l};
                    end
                end
            end

            %  Update messages from factor
            %--------------------------------------------------------------
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
            %--------------------------------------------------------------
            DU = dU(G{j});
            for k = 1:numel(DU) % Augment with any implicit Dirac nodes
                if ~isempty(dirac{G{j}(k)})
                    for l = mp_setdiff(dirac{G{j}(k)}',j)
                        DU{k} = DU{k}.*aU{G{j}(k),l};
                    end
                end
            end

            % Product of ascending messages
            AU = ones(size(aU{j,find(CD(:,j),1,"first")}));
            for k = find(CD(:,j))' % find children of j
                AU = mp_norm(AU.*aU{j,k});
            end

            %  Update messages from factor
            %--------------------------------------------------------------
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

    % Compute evidence
    if ~isfield(M,'a')
        F = sum(Fa);
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
                    if m(Gk(1))
                        if ismember(Gk(j),G{z})
                            AUD = mp_norm(AUD.*aU{Gk(j),z});
                        end
                    else
                        for v = Gk(:)'
                            AUD = AUD.*y{v + 1 - min(yi)};
                        end
                    end
                end
            end
            
            nA = size(a{k});
            if ~isempty(G{Gk(1)}) % factor k is not a prior
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

            if ~isempty(Gk)
                da = (a{k}>1/64).*AUD.*DUD;
                ap{k} = a{k} + da;
                F = F - mp_KL_dir(ap{k},a{k});
            else
                F = F + Fa(k);
            end
        end
        Q.a = ap;
    end

    % Update convergence criteria
    dF = F - F0;
    F0 = F;
    Fi(i) = F;
    criterion = [abs(dF)<1e-1, criterion(1:end-1)];

    % Determine where messages have reached
    %----------------------------------------------------------------------
   
    aI = find(sum(CD(ai,:),1));
    ai = aI;
    dI = find(sum(CD(:,di),2));
    di = dI;
    
    % Update wavefronts
    ai = unique([ai find(sum(CD(aI(aI>0),:),1))]);
    di = unique([di; find(sum(CD(:,aI(aI>0)),2))]);

    % Graphics
    %----------------------------------------------------------------------
    if ~M.nograph
        
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
        %------------------------------------------------------------------
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
            [u,~]   = eigs(Lap);                    % Eigenvectors
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
    % have reached a leaf or root (respectively) and we need the log
    % evidence to have converged.

    criterion  = [abs(dF)<1e-1, criterion(1:end-1)];
    if isempty(ai) && isempty(di)
        if ~M.noprint
            fprintf('Iteration: %d. Message passing complete if acyclic graph. dF: %g\n', i,dF)
        end
        if M.acyclic || sum(criterion) == length(criterion)
            U.a = aU;
            U.d = dU;
            
            if M.Q
                % Compute marginal probabilities
                %---------------------------------------------------
                Q.s = cell(N - length(yi),1);
                for k = 1:numel(Q.s)
                    AU = ones(size(aU{k,find(CD(:,k),1,"first")}));
                    for j = 1:size(aU,1)
                        if CD(j,k) % if k is a parent of j
                            AU = mp_norm(AU.*aU{k,j});
                        end
                    end
                    if isfield(A{k},'g')
                        Q.s{k} = A{k}.g(dU{k},AU);
                    else
                        Q.s{k} = mp_softmax(mp_log(dU{k}) + mp_log(AU));
                    end
                end
            else
                Q.s = [];
            end
            return
        end
    else
        if ~M.noprint
            fprintf('Iteration: %d. Asc. messages at: [%s]. Desc. messages at: [%s]. dF: %g\n',i,num2str(unique(ai)),num2str(unique(di')),dF)
        end
    end
end

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

function c = mp_setdiff(a,b)
% Faster version of setdiff, avoiding implicit sorting operations.
m = ~ismember(a, b);
c = a(m);