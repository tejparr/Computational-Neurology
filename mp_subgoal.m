function rH = mp_subgoal(H,L,G,C,d)
% Takes a set of subgoals H, a distance matrix L, graph clusters C, and 
% leaf/root nodes d, and returns a re-ordered list of subgoals rH.
%--------------------------------------------------------------------------

% Pre-allocate reordered states
%--------------------------------------------------------------------------
rH = cell(size(H));
for i = 1:numel(rH)
    rH{i} = zeros(length(vertcat(C{:})),1);
end

% Reorganise into a set of subgoals
%--------------------------------------------------------------------------
L        = L + diag(Inf*ones(length(H{1}),1)); % Preclude point attractors
L(end,:) = Inf;                                % and preclude return to start
j        = length(H{1});                       % but set this as initial state
r        = 0;                                  % members left in this cluster
Id       = ones(size(L,1),1)*Inf;              % initialise entry point plausibility
Ic       = zeros(size(L,1),1);                 % initialise cluster plausibility
c        = zeros(length(rH{1}),1);             % initialise sequence of clusters

% Set all entry points to clusters as plausible
%--------------------------------------------------------------------------
for i = 1:numel(d)
    Id(d{i}) = 0; 
end

for i = 1:length(rH{1})
    if r == 0
    % pick cluster (degree 1 node if available)
        [~,j] = min(L(:,j(1)) + Id);
        c(i)  = find(cellfun(@(x)ismember(j(1),x),d));
        Ic    = ones(size(Ic))*Inf; % Initialise all clusters to be implausible
        Ic(C{c(i)}) = 0;            % Except current cluster
        r     = length(C{c(i)})-1;
    else
        k      = find(G(:,j),1);
        G(j,:) = 0;
        j      = k;
        r      = r - 1; 
        c(i)   = c(i-1);
    end
    for k = 1:numel(rH)
        rH{k}(i) = H{k}(j(1));
    end
    L(j(1),:) = Inf;
end

% If clusters made up of ring topologies, ensure ring completion
%--------------------------------------------------------------------------
for i = unique(c)'
    if numel(C{i})==numel(d{i})
         for k = 1:numel(rH)
             lc = find(c==i,1,'last');
             fc = find(c==i,1,'first');
             rH{k} = [rH{k}(1:lc,1);rH{k}(fc,1);rH{k}(lc+1:end,1)];
         end
         c = [c(1:lc);c(fc);c(lc+1:end)];
    end
end