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

% Visualisation
%--------------------------------------------------------------------------
mp_pomdp_belief_plot(MDP)
mdp_tMaze_plot(MDP)

function [o,s] = mdp_tMaze_gen(s,u,~,~)
% Function for generative process. The arguments are states, actions, and 
% posterior predictions. The final argument would be the pomdp structure
% itself, in case any additional parameters of relevance are needed
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

function mdp_tMaze_plot(MDP)
figure('Color','w','Name','Maze Animation'); clf

% Maze structure
%--------------------------------------------------------------------------
M = [0 0 0 0 0;
     0 1 1 1 0;
     0 0 1 0 0;
     0 0 1 0 0;
     0 0 0 0 0];

% Coordinates for each location
%--------------------------------------------------------------------------
L = [3 3; 2 2; 4 2; 3 4 ;3 4];

% Create continuous trajectories
%--------------------------------------------------------------------------
X = zeros(MDP.T,size(MDP.s,1));
x = zeros(1+(MDP.T-1)*8,size(MDP.s,1));
for t = 1:MDP.T
    X(t,:) = L(MDP.o(1,t),:);
end
for j = 1:size(X,2)
    x(:,j) = interp1(X(:,j),1:1/8:MDP.T);
end

% Animate
%--------------------------------------------------------------------------

for t = 1:MDP.T
    for k = (1+(t-1)*8):(1+t*8)
        % Plot maze
        imagesc(M), colormap gray, axis equal, axis off, hold on
        title('Maze')

        % Plot beliefs about location
        for j = 1:4
            plot(L(j,1),L(j,2),'o','MarkerSize',16,'Color',[1,1 - MDP.Q{t,1}(j),1 - MDP.Q{t,1}(j)])
        end

        % Plot beliefs about context
        plot(L(2,1),L(2,2),'.','Markersize',30,'Color',[1 - MDP.Q{t,2}(1),1,1 - MDP.Q{t,2}(1)])
        plot(L(3,1),L(3,2),'.','Markersize',30,'Color',[1 - MDP.Q{t,2}(2),1,1 - MDP.Q{t,2}(2)])
        text(L(4,1),L(4,2),'\leftarrow', 'Color',[1 -  MDP.Q{t,2}(1),1,1 -  MDP.Q{t,2}(1)],'FontWeight','bold', 'HorizontalAlignment', 'center','Interpreter','Tex')
        text(L(4,1),L(4,2),'\rightarrow','Color',[1 -  MDP.Q{t,2}(2),1,1 -  MDP.Q{t,2}(2)],'FontWeight','bold', 'HorizontalAlignment', 'center','Interpreter','Tex')

        % Plot location
        if t < MDP.T
            plot(x(k,1),x(k,2),'.r','MarkerSize',16) 
            pause(0.1)
            hold off
        else
            plot(x(end,1),x(end,2),'.r','MarkerSize',16)
            pause(0.1)
            break
        end
    end
end
