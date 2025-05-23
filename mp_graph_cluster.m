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