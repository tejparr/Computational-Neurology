function MDP = DEMO_POMDP_tMaze
% This function deals with the T-maze demo that serves as an entry-level
% benchmark for Active Inference schemes, and in doing so shows how one may
% use the mp_POMDP.m routine to simulate behaviour
%--------------------------------------------------------------------------

close all

% Setup
%--------------------------------------------------------------------------
rng default     % for reproducibility
T = 3;          % time steps

% Initialise cell arrays for probability tensors
%--------------------------------------------------------------------------
A = cell(2,1);  % Likelihood
B = cell(2,1);  % Transitions
C = cell(2,1);  % Preferences
D = cell(2,1);  % Initial states

% Populate likelihood
%--------------------------------------------------------------------------
A{1}(:,:,1) = [1 0 0 0;     % centre
               0 1 0 0;     % left arm
               0 0 1 0;     % right arm
               0 0 0 1;     % cue shows left
               0 0 0 0];    % cue shows right

A{1}(:,:,2) = [1 0 0 0;     % centre
               0 1 0 0;     % left arm
               0 0 1 0;     % right arm
               0 0 0 0;     % cue shows left
               0 0 0 1];    % cue shows right

A{2}(:,:,1) = [0 1 0 0;     % rewarding
               0 0 1 0;     % aversive
               1 0 0 1];    % neutral

A{2}(:,:,2) = [0 0 1 0;     % rewarding
               0 1 0 0;     % aversive
               1 0 0 1];    % neutral

% Populate initial states
%--------------------------------------------------------------------------
D{1} = zeros(4,1); D{1}(1) = 1;
D{2} = ones(2,1)/2; 

% Populate transition probabilities
%--------------------------------------------------------------------------
B{1}(:,:,1) = [1 0 0 1;
               0 1 0 0;
               0 0 1 0;
               0 0 0 0];

B{1}(:,:,2) = [0 0 0 0;
               1 1 0 1;
               0 0 1 0;
               0 0 0 0];

B{1}(:,:,3) = [0 0 0 0;
               0 1 0 0;
               1 0 1 1;
               0 0 0 0];

B{1}(:,:,4) = [0 0 0 0;
               0 1 0 0;
               0 0 1 0;
               1 0 0 1];

B{2} = eye(2);

% Populate preferences
%--------------------------------------------------------------------------
C{1} = zeros(size(A{1},1),1);
C{2} = [3;-6;0];

% Priors for paths
%--------------------------------------------------------------------------
E{1} = ones(4,1)/4;

% Assemble POMDP
%--------------------------------------------------------------------------
mdp.A = A(:);
mdp.B = B(:);
mdp.C = C(:);
mdp.D = D(:);
mdp.E = E(:);

% Set domains (i.e., parents for each factor)
%--------------------------------------------------------------------------
mdp.dom.B(1).s = []; % Note that it is assumed that B's depend upon self-states
mdp.dom.B(1).u = 1;
mdp.dom.B(2).s = [];
mdp.dom.B(2).u = [];
mdp.dom.A(1).s = [1 2];
mdp.dom.A(1).u = [];
mdp.dom.A(2).s = [1 2];
mdp.dom.A(2).u = [];

% Simulation settings
%--------------------------------------------------------------------------
mdp.T   = T;              % Number of time-steps
mdp.gen = @mdp_tMaze_gen; % Generative process (i.e., simulation environment)
mdp.s   = [1;1];          % True initial states of above
mdp.N   = 2;              % Planning horizon

% Active Inversion of generative model
%--------------------------------------------------------------------------
MDP = mp_POMDP(mdp);

function [o,s] = mdp_tMaze_gen(s,u)
% Function for generative process
%--------------------------------------------------------------------------
if nargout > 1
    if s(1)==1 || s(1)==4
        s(1) = u;
    end    
end
o(1,1) = s(1)*(s(2)==1) + (s(1)==4)*(s(2)==2);
if (s(1)==2 && s(2)==1) || (s(1)==3 && s(2)==2)
    o(2,1) = 1;
elseif s(1)==1 || s(1)==4
    o(2,1) = 3;
else
    o(2,1) = 2;
end
