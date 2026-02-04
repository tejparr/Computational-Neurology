function [H,n] = mp_induction(h,B,Q,dom)
% Function to compute inductive priors over paths
% Inputs:
% h are indexed mandatory states for each factor
% B are factor specific transition probabilities
% Q are distributions for next states under each possible action
% dom contains information about dependency structure
%
% Outputs:
% H is a vector that tells us whether each next possible state is on path
% to mandatory states
% n is a scalar giving us the number of moves of the corresponding path
%
% Note: this deals with conditional dependencies in transitions but only
% reliably up to a single factor in the conditioning set. The logic will be
% generalised further in future development.
%--------------------------------------------------------------------------

% Pre-allocation
%--------------------------------------------------------------------------
ind = find(~cellfun(@isempty,h)); % Identify those state factors with associated mandatory states
b = cell(size(ind));              % Initialise cell array to contain matrices for backwards induction
k = zeros(size(ind));             % Initialise cell array to contain vectorised representation of goals
H = ones(numel(Q),1);             % Initialise plausible actions

% Prepare matrices for backwards induction
%--------------------------------------------------------------------------
for i = 1:length(ind)                                                       % Loop over state factors
    k(i) = h{ind(i)}(1);                                                    % Index of mandatory state in each factor
    if isempty(dom(ind(i)).s)                                               % If sparse dependencies between factors, use mean-field-like assumption
        b{i} = sum(B{ind(i)},3:ndims(B{ind(i)})) > 1/8;                     % Plausible state transitions (collapsing over actions)
    else                                                                    % For denser transition structures....
        [sU,iU,Ui] = intersect(dom(ind(i)).u,[dom([dom(ind(i)).s]).u]);     % Find shared actions with states in domain
        if ~isempty(sU)                                                     % If any, loop through actions and augment b tensors for each action
            ib     = 1; % placeholder for future development (to account for domains of multiple states)
            bi     = 3 + (1:length(dom(ind(i)).s));
            bi(ib) = 4;
            bi(1)  = ib+3;
            ui     = max(bi) + (1:length(dom(ind(i)).u));
            iu     = max(ui) + (1:length(dom(dom(ind(i)).s(ib)).u));
            iu(Ui) = ui(iU);
            b{i}   = mp_tensor_con(B{ind(i)},B{dom(ind(i)).s(ib)},...
                1:4,[1 3 bi ui],[2 4 iu])>1/8;
        else                                                                % If no shared actions, then resort to simpler scheme as above
            b{i} = sum(B{ind(i)},...
                3+length(dom(ind(i)).s):ndims(B{ind(i)})) > 1/8;
        end       
    end
end

% Main induction loop
%--------------------------------------------------------------------------
z = 0;                                                                      % initialise flag for completed identification of next state on path
n = 0;                                                                      % initialise number of steps back from goal
while z == 0 && n < 64                                                      % continue either until maximum number of steps or goal reached
    q = true(numel(Q),1);                                                   % initialise within each loop to say 'true' that each of the possible next states is on the path to the mandated state
    for i = 1:length(ind)                                                   % loop over factors with mandated states
        if isempty(dom(ind(i)).s)                                           % In the sparse dependency setting...
            K = zeros(size(b{i},1),1);                                      % Create one-hot vectors...
            K(k(i)) = 1;                                                    % ...for mandated states
            for j = 1:numel(Q)                                              % Loop over possible next steps from current position
                w    = Q{j}{ind(i)}'*((b{i}')^n)*K >= 1/2;                  % Assess whether these steps bring us, with >1/2 probability, to the start of a plausible path to mandated state of length n
                q(j) = q(j) && w;                                           % update for each possible next move
            end
        else
            if n==0
                K = mp_oh(size(b{i},1),k(i))*ones(size(b{i},2),1)';         % Create joint matrix
            else
                K = mp_tensor_con(b{i},K,[1 2],[3 4 1 2],[3 4])>0;
            end
            for j = 1:numel(Q)                                              % Loop over plausible first moves
                w   = mp_dot(K,{Q{j}{ind(i)},Q{j}{dom(ind(i)).s}})...       % Identify those with a high probability of being on a path to mandatory states
                    >= 1/2;
                q(j) = q(j) && w;
            end
        end
    end

    % Stopping criteria
    %----------------------------------------------------------------------
    if sum(q)>0                                                             
        H = q > 0;
        z = 1;
    end
    n = n + 1;
end