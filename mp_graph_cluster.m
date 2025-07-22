function [C,D] = mp_graph_cluster(A)
% Examine graph structure based upon adjacency matrix A, returning cluster
% memberships C and allowable starting vertices D
%-------------------------------------------------------------------------
A     = A - diag(diag(A));               % Ensure no self-connections
da    = sum(A);                          % Degree of all vertices
sc    = find(da==0);                     % Find singleton clusters
[~,ind] = setdiff(1:size(A,1),sc);       % Indices for recovery of original order
A(sc,:) = [];                            % Remove from graph
A(:,sc) = [];                            % ...
d     = sum(A);                          % Degree of each vertex
L     = diag(d) - A;                     % Graph Laplacian
[V,E] = eig(L);                          % Eigendecomposition
V     = V(:,abs(diag(E))<1/512);


[~,~,v] = unique(round(V * 1e6) / 1e6, 'rows');
C       = cell(size(V,2)+length(sc),1);    % Initialise cluster memberships
D       = cell(size(V,2)+length(sc),1);    % Initialise possible starting vertices

% Determine members of each cluster
for i = 1:size(V,2)
    C{i} = ind(v==i);   
end

% Deal with singleton clusters
for i = 1:length(sc)
    C{i} = sc(i);
    D{i} = sc(i);
end

for i = 1:size(V,2)
    if any(da(C{i})==1)
        D{i} = C{i}(da(C{i})==1);     % For clusters with roots/leaves, start there
    else
        D{i} = C{i};                  % Otherwise all start locations equally viable
    end
end
