function h = mp_pomdp_order(h,B,Q,OPT)
% Function that optimises the order of the mandatory states h given
% transitions B and beliefs about the initial location Q.
%--------------------------------------------------------------------------

try
    noplot = OPT.noplot;
catch
    noplot = true;
end

% First, find the lengths of the shortest paths between mandatory states
%==========================================================================
Ih = ~cellfun(@isempty,h);     % Identify factors with mandatory states

% Initialise variables with only relevant elements
%--------------------------------------------------------------------------
H     = h(Ih);                 % Mandatory states
B     = B(Ih);                 % Transition probabilities
Q     = Q(Ih);                 % Initial state probabilitie
L     = zeros(length(H{1})+1); % Distance matrix

% Supplement mandatory states with initial state (MAP estimate)
%--------------------------------------------------------------------------
for i = 1:numel(H)
    [~,d]       = max(Q{i});
    H{i}(end+1) = d(1);
end

% Initialise variables to compute distances
%--------------------------------------------------------------------------
Qi = cell(size(Q));
Hj = cell(size(H));

% Loop through pairwise combinations of mandatory (+initial) states and
% compute minimum path lengths inductively
%--------------------------------------------------------------------------
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

% Construct unweighted, undirected, graph based upon immediate neighbours
%--------------------------------------------------------------------------
G = L(1:end-1,1:end-1) == 1;
G = G - diag(diag(G));

% Prune graph to ensure is bipartite (minimum pruning)
%==========================================================================

% Identify triangles
%--------------------------------------------------------------------------
[g,k]   = sort(diag(G^3),'ascend');
k       = k(g>0); % nodes participating in triangles, ranked in ascending 
                  % order by the number of triangles in which they 
                  % participate

% Identify which lines in the triangle are segments of longer, straight,
% lines and preferentially preserve these
%--------------------------------------------------------------------------

if ~noplot
    subplot(4,4,1)
    Gplot = plot(graph(G));
    Gplot.XData = H{2}(1:end-1);
    Gplot.YData = H{1}(1:end-1);
    % Gplot.NodeLabel = {};      
    axis ij
    sp = 2;
end

while ~isempty(k)
    kk = find(G(:,k(1)));
    [~,~,j]  = intersect(find(G*G(:,k(1))),kk);  % nodes on a triangle with k(1)
    [j1,j2] = find(G(kk(j),kk(j)),1);
    cn   = [kk(j(j1)) kk(j(j2)) k(1)];
    nc   = perms(cn);                            % then determine combinations of these nodes
    Z    = mp_momentum_score(nc,H);              % compute momentum scores
    
    % Account for additional connections
    for i = 1:size(nc,1)
        i1 = setdiff(find(G(:,nc(i,1))),nc(i,:));
        i2 = setdiff(find(G(:,nc(i,2))),nc(i,:));
        i3 = setdiff(find(G(:,nc(i,3))),nc(i,:));
        if ~isempty(i1)
            W = max([mp_momentum_score([i1 ones(size(i1))*[nc(i,1) nc(i,2)]],H);0]);
        else
            W = 0;
        end
        if ~isempty(i2)
            W = W + max([max(mp_momentum_score([i2 ones(size(i2))*[nc(i,2) nc(i,1)]],H)); ...
                        max(mp_momentum_score([i2 ones(size(i2))*[nc(i,2) nc(i,3)]],H));0]);
        end
        if ~isempty(i3)
            W  = W + max([mp_momentum_score([i3 ones(size(i3))*[nc(i,3) nc(i,2)]],H);0]);
        end
        Z(i) = Z(i) + W;
    end

    [~,z] = max(Z);                              % find pair of edges with maximum momentum

    % Disconnect the two nodes at ends of this edge
    G(nc(z,1),nc(z,3)) = 0;   
    G(nc(z,3),nc(z,1)) = 0;
    L(nc(z,1),nc(z,3)) = L(nc(z,1),nc(z,3)) + 1/2;  
    L(nc(z,3),nc(z,1)) = L(nc(z,3),nc(z,1)) + 1/2;  

    if ~noplot
        subplot(4,4,sp)
        cla
        Gplot = plot(graph(G));
        Gplot.XData = H{2}(1:end-1);
        Gplot.YData = H{1}(1:end-1);
        Gplot.NodeLabel = {};      
        axis ij
        sp = sp+1;
        sp(sp>16) = 1;
        drawnow
    end
    
    [g,k]   = sort(diag(G^3),'ascend');
    k       = k(g>0);
end

% Then duplicate remaining nodes with degree > 2 and distribute their edges
p = sum(ceil(bsxfun(@max,sum(G)-2,0)/2));
G = padarray(G,p*ones(1,2),0,'pre');
L = padarray(L,p*ones(1,2),64,'pre');
for i = 1:numel(H)
    H{i} = padarray(H{i},p,0,'pre');
end

for n = p:-1:1
    [~,k]    = sort(sum(G),'descend');                  % Sort nodes by degree
    [l,j]    = sort(sum(G*diag(G(:,k(1)))),'descend');  % Find nodes connected to maximum degree node that are not roots/leaves
    [~,~,ui] = unique(l(l>0));

    if length(ui(ui>0))>2 % If there are multiple plausible ways of defining intersection
        cn = j(ui>0);     % Candidate nodes to be connected
        nc = nchoosek(cn,2);     % list permutations of intersections
        nc = [nc(:,1) k(1)*ones(size(nc,1),1) nc(:,2)]; % include node to be duplicated
        
        % Evaluate under conservation of momentum prior (this assumes the
        % state coordinates have an implicit metric meaning)
        Z = mp_momentum_score(nc,H);

        [~,zi] = sort(Z,'descend');
        for iz = 1:length(zi)
            cn(2*(iz-1) + (1:2)) = [nc(zi(iz),1) nc(zi(iz),3)];
        end
        [cni, ~, cnj] = unique(cn,'stable');
        j(ui>0) = cni(cnj(1:length(cni)));
    end

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

if ~noplot
    subplot(4,4,sp)
    Gplot = plot(graph(G));
    Gplot.XData = H{2}(1:end-1);
    Gplot.YData = H{1}(1:end-1);
    [~,u,~] = unique([Gplot.XData; Gplot.YData]','rows','stable');
    if length(u) < length(H{1})-1
        m = setdiff(1:length(H{1})-1,u);
        Gplot.XData(m) = Gplot.XData(m)+1/2;
        Gplot.YData(m) = Gplot.YData(m)+1/2;
    end
    Gplot.NodeLabel = {};
    axis ij
    drawnow
    sp = sp + 1;
    sp(sp>16) = 1;
end

try
    [C,d] = mp_graph_cluster(G);
    i = find(cellfun(@(x)numel(x)<3,C))';
    G([C{i}],:) = 0;
    G(:,[C{i}]) = 0;
    C = C(setdiff(1:numel(C),i));
    d = d(setdiff(1:numel(d),i));
    if ~noplot
        for i = 1:numel(C)
            subplot(4,4,sp)
            Gplot = plot(graph(G(C{i},C{i})));
            Gplot.XData = H{2}(C{i});
            Gplot.YData = H{1}(C{i});
            [~,u,~] = unique([Gplot.XData; Gplot.YData]','rows','stable');
            if length(u) < length(H{1}(C{i}))
                m = setdiff(1:length(H{1}(C{i})),u);
                Gplot.XData(m) = Gplot.XData(m)+1/2;
                Gplot.YData(m) = Gplot.YData(m)+1/2;
            end
            Gplot.NodeLabel = {};
            axis ij
            hold on
        end
        drawnow
    end
    rH    = mp_subgoal(H,L,G,C,d);
catch
    rH = H;
end

% Repackage in original h structure:
%--------------------------------------------------------------------------
h(~cellfun(@isempty,h)) = rH;

function Z = mp_momentum_score(nc,H)
% Compute momentum scores for a set of two part line segments (effectively,
% deviation from a straight line or cosine of angle between segments). One
% could further nuance this by constructing the momenta from the implicit
% metric given in the transition probabilities. However, we will assume for
% simplicity that the states are indexed such that nearby values are near
% to one another.
%--------------------------------------------------------------------------

Z = zeros(size(nc,1),1); % Initialise vector of 'momenta scores'
for ic = 1:size(nc,1)
    X = zeros(numel(H),3);
    for ih = 1:numel(H)
        for ij = 1:3
            X(ih,ij) = H{ih}(nc(ic,ij));
        end
    end
    Z(ic) = (X(:,2)-X(:,1))'*(X(:,3)-X(:,2))/(vecnorm(X(:,2)-X(:,1))*vecnorm(X(:,3)-X(:,2)));
end