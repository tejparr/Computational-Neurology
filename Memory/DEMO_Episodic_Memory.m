function pomdp = DEMO_Episodic_Memory
% This demo illustrates some key features of episodic memory and its
% breakdown. It is closely modelled on a previous demo, available in the
% SPM toolbox (DEMO_MDP_Understanding.m), which was focused upon a related
% problem but unpacks the features of this with episodic memory in mind.
%__________________________________________________________________________

% set up and preliminaries
%==========================================================================
rng('default')
close all

% Options
%--------------------------------------------------------------------------
OPTIONS.save = 0; % Option to save animation
cd(fileparts(mfilename('fullpath')))

% Simulation setup
%--------------------------------------------------------------------------
query = 1; % Which move to ask about

% Lesions (defaults all 9/10)
%--------------------------------------------------------------------------
p = [9;       % Attenuation (suppression of visual modalities during recall) [1]
     6;       % Encoding    (precision with which policies and context are predicted by highest level) [6]
     9;       % Retrieval   (precision with which semantic states are predicted by highest level) [5]
     9]/10;   % Retention   (precision of higher level transitions for semantically relevant states) [5]

% FIRST LEVEL MODEL (EPISODE TO ENCODE OR RECALL)
%--------------------------------------------------------------------------

% Linguistic hidden states:
syn      = {'...', 'What', 'did', 'you', 'do', '#1', ' ', '-', 'I', '#2', 'the route home', 'was to the', '#3', 'so I', '#4', 'by going to', '#5'};
sem{1}   = {'first?','second?'};
sem{2}   = {'knew','did not know'};
sem{3}   = {'left','right'};
sem{4}   = {'guessed','checked','found it'};
sem{5}   = {'the left.','the right.','the signpost.'};
words{1} = {' ', 'What', 'did', 'you', 'do', 'first?', 'second?'};
words{2} = {' ', 'I', 'knew', 'did not know', 'the route home', 'was to the', 'right', 'left' 'so I', 'guessed', 'checked', 'found it', 'by going to', 'the right.', 'the left.', 'the signpost.'};

% outcome probabilities: A
%--------------------------------------------------------------------------
% We start by specifying the probabilistic mapping from hidden states
% to outcomes; where outcome can be exteroceptive or interoceptive: The
% exteroceptive outcomes A{1} provide cues about location and context,
% while interoceptive outcome A{2) denotes different levels of reward
%--------------------------------------------------------------------------
for f3 = 1:17
    for f4 = 1:2
        for f5 = 1:2
            for f6 = 1:3
                for f7 = 1:3

                    a1(:,:,1) = [...
                            1 0 0 0;    % start location (junction)
                            0 1 0 0;    % left path
                            0 0 1 0;    % right path
                            0 0 0 1     % signpost right
                            0 0 0 0];   % signpost left
                    a1(:,:,2) = [...
                            1 0 0 0;    % start location (junction)
                            0 1 0 0;    % left path
                            0 0 1 0;    % right path
                            0 0 0 0     % signpost right
                            0 0 0 1];   % signpost left
                    a2(:,:,1) = [...
                            1 0 0 1;    % neutral
                            0 1 0 0;    % home in sight
                            0 0 1 0];   % path ahead flooded
                    a2(:,:,2) = [...
                            1 0 0 1;    % neural
                            0 0 1 0;    % home in sight
                            0 1 0 0];   % path ahead flooded

                    A{1}(:,:,:,f3,f4,f5,f6,f7) = [(f3==1),(f3~=1)]*[1;1-p(1)]*a1 + [(f3==1),(f3~=1)]*[0;p(1)]*repmat([1;zeros(4,1)],1,4,2);
                    A{2}(:,:,:,f3,f4,f5,f6,f7) = [(f3==1),(f3~=1)]*[1;1-p(1)]*a2 + [(f3==1),(f3~=1)]*[0;p(1)]*repmat([1;zeros(2,1)],1,4,2);

                    if f3 == 1 % If behaving
                        
                        % Audition (questioner)
                        A{3}(1,:,:,f3,f4,f5,f6, f7) = ones(1,4,2); % Null
                        
                        % Vocalisation
                        A{4}(1,:,:,f3,f4,f5,f6, f7) = ones(1,4,2); % Null
                        
                    else % If explaining the solution
                        
                        if f3 < 8
                            % Vocalisation
                            A{4}(1,:,:,f3,f4,f5,f6, f7) = ones(1,4,2); % Null
                            
                            for g = 1:length(words{1})
                                if syn{f3}(1) ~= '#'
                                    A{3}(g,:,:,f3,f4,f5,f6,f7) = strcmp(words{1}{g},syn{f3})*ones(1,4,2);
                                elseif syn{f3}(2) == '1'
                                    A{3}(g,:,:,f3,f4,f5,f6,f7) = strcmp(words{1}{g},sem{1}{f4})*ones(1,4,2);
                                end
                            end
                            
                        else
                            % Audition (questioner)
                            A{3}(1,:,:,f3,f4,f5,f6,f7) = ones(1,4,2); % Null
                            for f2 = 1:2
                                for g = 1:length(words{2})
                                    if syn{f3}(1) ~= '#'
                                        if syn{f3}(1) == '-'
                                            A{4}(g,:,f2,f3,f4,f5,f6,f7) = strcmp(words{2}{g},' ')*ones(1,4);
                                        else
                                            A{4}(g,:,f2,f3,f4,f5,f6,f7) = strcmp(words{2}{g},syn{f3})*ones(1,4);
                                        end
                                    elseif syn{f3}(2) == '2'
                                        A{4}(g,:,f2,f3,f4,f5,f6,f7) = strcmp(words{2}{g},sem{2}{f5})*ones(1,4);
                                    elseif syn{f3}(2) == '3'
                                        A{4}(g,:,f2,f3,f4,f5,f6,f7) = strcmp(words{2}{g},sem{3}{f2})*ones(1,4);
                                    elseif syn{f3}(2) == '4'
                                        A{4}(g,:,f2,f3,f4,f5,f6,f7) = strcmp(words{2}{g},sem{4}{f6})*ones(1,4);     
                                    elseif syn{f3}(2) == '5'
                                        A{4}(g,:,f2,f3,f4,f5,f6,f7) = strcmp(words{2}{g},sem{5}{f7})*ones(1,4);         
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


% controlled transitions: B{u}
%--------------------------------------------------------------------------
% Next, we must specify the probabilistic transitions of hidden states
% for each factor. Here, there are four actions taking the agent directly
% to each of the four locations.
%--------------------------------------------------------------------------
B{1}(:,:,1)  = [1 0 0 1; 0 1 0 0;0 0 1 0;0 0 0 0];
B{1}(:,:,2)  = [0 0 0 0; 1 1 0 1;0 0 1 0;0 0 0 0];
B{1}(:,:,3)  = [0 0 0 0; 0 1 0 0;1 0 1 1;0 0 0 0];
B{1}(:,:,4)  = [0 0 0 0; 0 1 0 0;0 0 1 0;1 0 0 1];

% context/semantics, which cannot be changed by action
%--------------------------------------------------------------------------
B{2}  = eye(2);
B{4}  = eye(2);
B{5}  = eye(2);
B{6}  = eye(3);
B{7}  = eye(3);

% Syntactic states
%--------------------------------------------------------------------------
B{3}                = zeros(numel(syn),numel(syn));
B{3}(1,1)           = 1;
B{3}(3:7,2:6)       = eye(5);
B{3}(7,7)           = 1;
B{3}(9:end,8:end-1) = eye(9);
B{3}(17,17)         = 1;


% priors: (utility) C
%--------------------------------------------------------------------------
% Finally, we have to specify the prior preferences in terms of log
% probabilities over outcomes. Here, the agent prefers being home to 
% getting stuck in a flood - and does not like staying at the junction
%--------------------------------------------------------------------------
C{1}  = [-1;0;0;0;0];
C{2}  = [0;3;-9];
C{3}  = zeros(size(A{3},1),1);
C{4}  = zeros(size(A{4},1),1);

% now specify prior beliefs about initial states, in terms of counts. Here
% the hidden states are factorised into location and context:
%--------------------------------------------------------------------------

%   Maze states
%--------------------------------------------------------------------------
D{1} = [1 0 0 0]';      % Location
D{2} = [1 1]'/2;        % Context (left or right)

%   Linguistic states
%--------------------------------------------------------------------------
D{3} = ones(17,1)/17;   % Syntactic states
D{4} = [1 1]'/2;        % #1 (first/second)
D{5} = [1 1]'/2;        % #2 (knew/didn't know)
D{6} = [1 1 1]'/3;      % #4 (guessed/explored/found it)
D{7} = [1 1 1]'/3;      % #5 (the left/the right/the cue)

% allowable policies (of depth T).  These are just sequences of actions
% (with an action for each controllable factor)
%--------------------------------------------------------------------------
V    = [1  1  1  1  2  3  4  4  4  4;
        1  2  3  4  2  3  1  2  3  4];

% Self-generated outcomes
%--------------------------------------------------------------------------
n = zeros(numel(A),1);
n(4,:) = 1;

% MDP Structure - this will be used to generate arrays for multiple trials
%==========================================================================
mdp.V = V;                       % allowable policies
mdp.A = A;                       % observation model
mdp.B = B;                       % transition probabilities
mdp.C = C;                       % preferred outcomes
mdp.D = D;                       % prior over initial states
mdp.E{1} = ones(4,1)/4;          % prior over policies

for i = 1:numel(B)
    mdp.dom.B(i).s = [];
    mdp.dom.B(i).u = [];
end
mdp.dom.B(1).u = 1;

for i = 1:numel(A)
    mdp.dom.A(i).s = 1:numel(D);
    mdp.dom.A(i).u = [];
end

mdp.gen = @mdp_episodic_gen;
mdp.s   = ones(7,1);
mdp.T   = 10;
mdp.n   = n;

% Solve lower level model to check behaviour:
%--------------------------------------------------------------------------
% MDP      = mp_POMDP(mdp);
clear MDP

% SECOND LEVEL MODEL (EXPLAINING BEHAVIOUR)
%--------------------------------------------------------------------------
clear A B C D

% Higher level hidden state priors
%--------------------------------------------------------------------------
D{1} = [1 0 0]'/3;   % Narrative structure (solve maze, query, respond)
D{2} = [1 1 1 1]'/4; % First move
D{3} = [1 1 1 1]'/4; % Second move
D{4} = [1 1]'/2;     % Context
D{5} = [1 1]'/2;     % Query (first/second move)

% Higher level likelihood
%--------------------------------------------------------------------------
for f1 = 1:length(D{1})
    for f2 = 1:length(D{2})
        for f3 = 1:length(D{3})
            for f4 = 1:length(D{4})
                for f5 = 1:length(D{5})
% Initial syntactic state
%--------------------------------------------------------------------------
                    A{1}(1,f1,f2,f3,f4,f5) = f1 == 1;
                    A{1}(2,f1,f2,f3,f4,f5) = f1 == 2;
                    A{1}(8,f1,f2,f3,f4,f5) = f1 == 3;
                    A{1}(numel(syn),f1,f2,f3,f4,f5) = 0;

% Semantic state #1
%--------------------------------------------------------------------------
                    A{2}(f5,f1,f2,f3,f4,f5) = 1;
                    
% Semantic state #2
%--------------------------------------------------------------------------
                    if f5 == 1 % if first move is queried...
                        if f2 == 1 % ...and if stays put...
                            A{3}(2,f1,f2,f3,f4,f5) = 1; %...conclude did not know
                        elseif f2 == 2 %...or if goes to left arm...
                            if f4 == 1 %...and this is the correct context...
                                A{3}(1,f1,f2,f3,f4,f5) = 1; %...conclude did know
                            else %...but if incorrect context...
                                A{3}(2,f1,f2,f3,f4,f5) = 1; %...conclude did not know
                            end
                        elseif f2 == 3 %...or if goes to right arm...
                            if f4 == 2 %...and this is the correct context...
                                A{3}(1,f1,f2,f3,f4,f5) = 1; %...conclude did know
                            else %...but if incorrect context...
                                A{3}(2,f1,f2,f3,f4,f5) = 1; %...conclude did not know
                            end
                        else %...or if goes to cue location...
                            A{3}(2,f1,f2,f3,f4,f5) = 1; %...conclude did not know
                        end
                    else % if second move is queried...
                        if f3 == 1 % ...and if stays put...
                            A{3}(2,f1,f2,f3,f4,f5) = 1; %...conclude did not know
                        elseif f3 == 2 %...or if goes to left arm...
                            if f4 == 1 %...and this is the correct context...
                                A{3}(1,f1,f2,f3,f4,f5) = 1; %...conclude did know
                            else %...but if incorrect context...
                                A{3}(2,f1,f2,f3,f4,f5) = 1; %...conclude did not know
                            end
                        elseif f3 == 3 %...or if goes to right arm...
                            if f4 == 2 %...and this is the correct context...
                                A{3}(1,f1,f2,f3,f4,f5) = 1; %...conclude did know
                            else %...but if incorrect context...
                                A{3}(2,f1,f2,f3,f4,f5) = 1; %...conclude did not know
                            end
                        else %...or if goes to cue location...
                            A{3}(2,f1,f2,f3,f4,f5) = 1; %...conclude did not know
                        end
                    end
% Semantic state #3
%--------------------------------------------------------------------------
                    A{4}(f4,f1,f2,f3,f4,f5) = 1;
                    
% Semantic state #4
%--------------------------------------------------------------------------
                    if f5 == 1 % if first move is queried...
                        if f2 == 1 % ...and if stays put...
                            A{5}(:,f1,f2,f3,f4,f5) = 1; %...no useful information
                        elseif f2 == 2 %...or if goes to left arm...
                            if f4 == 1 %...and this is the correct context...
                                A{5}(3,f1,f2,f3,f4,f5) = 1; %...found it
                            else %...but if incorrect context...
                                A{5}(1,f1,f2,f3,f4,f5) = 1; %...conclude guessed
                            end
                        elseif f2 == 3 %...or if goes to right arm...
                            if f4 == 2 %...and this is the correct context...
                                A{5}(3,f1,f2,f3,f4,f5) = 1; %...found it
                            else %...but if incorrect context...
                                A{5}(1,f1,f2,f3,f4,f5) = 1; %...conclude guessed
                            end
                        else %...or if goes to cue location...
                            A{5}(2,f1,f2,f3,f4,f5) = 1; %...explored
                        end
                    else % if second move is queried...
                        if f3 == 1 % ...and if stays put...
                            A{5}(:,f1,f2,f3,f4,f5) = 1; %...no useful information
                        elseif f3 == 2 %...or if goes to left arm...
                            if f4 == 1 %...and this is the correct context...
                                A{5}(3,f1,f2,f3,f4,f5) = 1; %...found it
                            else %...but if incorrect context...
                                A{5}(1,f1,f2,f3,f4,f5) = 1; %...conclude guessed
                            end
                        elseif f3 == 3 %...or if goes to right arm...
                            if f4 == 2 %...and this is the correct context...
                                A{5}(3,f1,f2,f3,f4,f5) = 1; %...found it
                            else %...but if incorrect context...
                                A{5}(1,f1,f2,f3,f4,f5) = 1; %...conclude guessed
                            end
                        else %...or if goes to cue location...
                            A{5}(2,f1,f2,f3,f4,f5) = 1; %...explored
                        end
                    end
% Semantic state #5
%--------------------------------------------------------------------------
                    if f5 == 1 % if first move is queried...
                        if f2 == 1 % ...and if stays put...
                            A{6}(:,f1,f2,f3,f4,f5) = 1; %...no useful information
                        elseif f2 == 2 %...or if goes to left arm...
                            A{6}(1,f1,f2,f3,f4,f5) = 1; %...left arm
                        elseif f2 == 3 %...or if goes to right arm...
                            A{6}(2,f1,f2,f3,f4,f5) = 1; %...right arm
                        else %...or if goes to cue location...
                            A{6}(3,f1,f2,f3,f4,f5) = 1; %...cue arm
                        end
                    else % if second move is queried...
                        if f2 == 2 % if first move was to left arm (absorbing state)
                            A{6}(1,f1,f2,f3,f4,f5) = 1; %...left arm
                        elseif f2 == 3 % if first move was to right arm (absorbing state)
                            A{6}(2,f1,f2,f3,f4,f5) = 1; %...right arm
                        elseif f3 == 1 % centre of maze
                            A{6}(:,f1,f2,f3,f4,f5) = 1; %...no useful information
                        elseif f3 == 2 %...or if goes to left arm...
                            A{6}(1,f1,f2,f3,f4,f5) = 1; %...left arm
                        elseif f3 == 3 %...or if goes to right arm...
                            A{6}(2,f1,f2,f3,f4,f5) = 1; %...right arm
                        else %...or if goes to cue location...
                            A{6}(3,f1,f2,f3,f4,f5) = 1; %...cue arm
                        end
                    end
% Policy
%--------------------------------------------------------------------------
                    if f2 == 2 || f2 == 3
                        A{7}(V(1,:,1)==f2,f1,f2,f3,f4,f5) = 1;
                    else
                        ind1 = find(V(1,:,1)==f2);
                        ind2 = find(V(2,:,1)==f3);
                        ind  = intersect(ind1,ind2);
                        A{7}(ind,f1,f2,f3,f4,f5) = 1;
                    end
                end
            end
        end
    end
end

for i = [4 7]
    A{i}(A{i}==1) = p(2);
    A{i}(A{i}==0) = (1-p(2))/(size(A{i},1)-1);
end

for i = [2 3 4 6]
    A{i}(A{i}==1) = p(3);
    A{i}(A{i}==0) = (1-p(3))/(size(A{i},1)-1);
end

% Transition probabilities
%--------------------------------------------------------------------------
B{1} = [0 0 0; 1 0 0; 0 1 1];
B{2} = eye(4);
B{3} = eye(4);
B{4} = eye(2);
B{5} = eye(2);

for i = [2 3 4 5]
    B{i}(B{i}==1) = p(4);
    B{i}(B{i}==0) = (1-p(4))/(size(B{i},1)-1);
end

% Preferences
%--------------------------------------------------------------------------
C = cell(numel(A),1);
for g = 1:numel(A)
    C{g} = ones(size(A{g},1),1);
end

% Construct MDP
%--------------------------------------------------------------------------
MDP.mdp = mdp; clear mdp

MDP.A = A;
MDP.B = B;
MDP.C = C;
MDP.D = D;
MDP.con = []; % No controllable elements
MDP.T = 3;


for i = 1:numel(B)
    MDP.dom.B(i).s = [];
    MDP.dom.B(i).u = [];
end

for i = 1:numel(A)
    MDP.dom.A(i).s = 1:numel(D);
    MDP.dom.A(i).u = [];
end

MDP.gen = @mp_episodic_link;
MDP.s   = [1;1;1;2;query];  

pomdp     = mp_POMDP(MDP);


% Plot beliefs and behaviour (animation)
%--------------------------------------------------------------------------
cn_figure('Episodic Memory'); clf
mp_episodic_animation(pomdp,words,OPTIONS)
mp_pomdp_belief_plot(pomdp);
mp_episodic_beliefs_concat(pomdp)

function mp_episodic_beliefs_concat(pomdp)
% Plot beliefs from both levels concatenated together
%--------------------------------------------------------------------------

cn_figure('Beliefs (episodic)')

% Get beliefs about states at higher level
%--------------------------------------------------------------------------
Q = cellfun(@(x)x',pomdp.Q,'UniformOutput',false);
subplot(2,1,1)
imagesc(1 - cell2mat(Q)'), colormap gray
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title('Beliefs (slow)')

% And those at lower level
%--------------------------------------------------------------------------
Nm = length(pomdp.y);

for i = 1:Nm
    Q = cellfun(@(x)x',pomdp.y(i).pomdp.Q,'UniformOutput',false);
    subplot(2,Nm,Nm+i)
    imagesc(1 - cell2mat(Q)'), colormap gray
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    title(['Beliefs (fast - epoch ' num2str(i) ')'])
end

function mp_episodic_animation(MDP,words,OPTIONS)
% If the input is a single level (inverted) MDP, this function plots the
% behaviour elicited from and inferences drawn by this model as an
% animation. If it is a hierarchical MDP, the function is applied
% successively to each of the lower level MDPs.
%--------------------------------------------------------------------------

if isfield(MDP,'y')
    OPTIONS.first = true;
    for i = 1:length(MDP.y)
        mp_episodic_animation(MDP.y(i).pomdp,words,OPTIONS)
        OPTIONS.first = false;
    end
else
    
% Map structure
%--------------------------------------------------------------------------
M = [0 0 0 0 0;
     0 1 0 1 0;
     0 0 1 0 0;
     0 0 1 0 0;
     0 0 0 0 0];
    
% Graphics for different states
%--------------------------------------------------------------------------
I{1} = imread('Graphics\Scene1.png');
I{2} = imread('Graphics\Scene2.png');
I{3} = imread('Graphics\Scene3.png');
I{4} = imread('Graphics\Scene4.png');

%--------------------------------------------------------------------------
% Coordinates for each location
%--------------------------------------------------------------------------
L = [3 3; 2 2; 4 2; 3 4 ;3 4];

% Create continuous trajectories
%--------------------------------------------------------------------------
X = zeros(MDP.T,2);
x = zeros(1+(MDP.T-1)*8,2);
for t = 1:MDP.T
    X(t,:) = L(MDP.o(1,t),:);
end
for j = 1:size(X,2)
    x(:,j) = interp1(X(:,j),1:1/8:MDP.T);
end

% Animate
%--------------------------------------------------------------------------
for t = 1:MDP.T

    % Plot conversation
    %--------------------------------------------------------------------------
    subplot(3,1,2)
    cla
    txt = [];
    for tt = 1:t
        txt = [txt ' ' words{1}{MDP.o(3,tt)}]; %#ok<AGROW>
    end
    axis([0 10 -4 2])
    text(0,1,txt,'FontSize',12)
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    box on
    title('Query')
    hold off

    subplot(3,1,3)
    cla
    txt =  [];
    txt2 = [];
    for tt = 1:t
        if tt < 8
            txt = [txt ' ' words{2}{MDP.o(4,tt)}]; %#ok<AGROW>
        else
            txt2 = [txt2 ' ' words{2}{MDP.o(4,tt)}]; %#ok<AGROW>
        end
    end
    axis([0 10 -4 2])
    text(0,1,txt,'FontSize',12)
    if ~isempty(txt2)
        text(0,0,txt2,'FontSize',12)
    end
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    box on
    title('Response')
    hold off

    % Plot beliefs from observer perspective
    subplot(3,2,2)
    imshow(I{1}*MDP.Q{t,1}(1) ...
        + I{2}*(MDP.Q{t,1}(2)*MDP.Q{t,2}(1) + MDP.Q{t,1}(3)*MDP.Q{t,2}(2))...
        + I{3}*(MDP.Q{t,1}(2)*MDP.Q{t,2}(2) + MDP.Q{t,1}(3)*MDP.Q{t,2}(1))...
        + I{4}*MDP.Q{t,1}(4)), axis equal, axis off

    for k = (1+(t-1)*8):(1+t*8)
        % Plot map
        subplot(3,2,1)
        imagesc(M), colormap gray, axis equal, axis off, hold on

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
            if OPTIONS.save
                F  = getframe(gcf);
                im = frame2im(F);
                [MM,MMM] = rgb2ind(im,256);
                imwrite(MM,MMM,'Graphics/Animation.gif','gif','WriteMode','append','DelayTime',0.1);
            end
            break
        end


        % Animation
        % -----------------------------------------------------------------
        if OPTIONS.save
            F  = getframe(gcf);
            im = frame2im(F);
            [MM,MMM] = rgb2ind(im,256);
            if OPTIONS.first
                imwrite(MM,MMM,'Graphics/EpisodicAnimation.gif','gif','LoopCount',Inf,'DelayTime',0.05);
                OPTIONS.first = false;
            else
                imwrite(MM,MMM,'Graphics/EpisodicAnimation.gif','gif','WriteMode','append','DelayTime',0.05);
            end
        end
    end
end
end


function [o,s] = mp_episodic_link(s,~,Qo,pomdp)
% Link function that acts as a generative process from the perspective of
% the second level, and that generates priors from the perspective of the
% first level.
%--------------------------------------------------------------------------
mdp = pomdp.mdp; % Get MDP structure for first level
V   = mdp.V;

if nargout > 1 % Advance the states
    for i = 1:numel(pomdp.B)
        ind          = [{':'}, s(i,end)'];
        [~,s(i,end)] = max(pomdp.B{i}(ind{:}));
    end
end

% Determine links between outcomes (upper level) and priors (lower level)
%--------------------------------------------------------------------------
linkD       = false(numel(mdp.D),numel(pomdp.A));
linkD(3,1)  = 1;
linkD(4,2)  = 1;
linkD(5,3)  = 1;
linkD(2,4)  = 1;
linkD(6,5)  = 1;
linkD(7,6)  = 1;
linkE       = false(1,numel(pomdp.A));
linkE(end)  = 1;

% Set lower level priors 
%--------------------------------------------------------------------------
if ~isempty(Qo)
    for g = find(sum(linkD,1))
        mdp.D{linkD(:,g)} = Qo{g};
    end
    for g = find(sum(linkE,1))
        dE = zeros(size(mdp.E{linkE(:,g)},1),size(V,1));
        for t = 1:size(V,1)
            for u = 1:size(V,2)
                dE(V(t,u),t) = dE(V(t,u),t) + Qo{g}(u); 
            end
        end
        mdp.E{linkE(:,g)} = dE;
    end
end

% Set initial states
%--------------------------------------------------------------------------
for g = find(sum(linkD,1))
    ind = [{':'}, num2cell(s.')];
    [~,mdp.s(linkD(:,g))] = max(pomdp.A{g}(ind{:}));
end

mdp.smooth = true;

% Solve lower level MDP
%--------------------------------------------------------------------------
MDP = mp_POMDP(mdp);

% Pass messages back to higher level
%--------------------------------------------------------------------------
Q  = MDP.BS.s(:,1)';
D  = MDP.D;
O  = cellfun(@(x,y)mp_softmax(mp_log(x)-mp_log(y)),Q,D,'UniformOutput',false);
y  = cell(size(pomdp.A));
P  = MDP.BS.u;
E  = MDP.E;
R  = zeros(size(V,2),1);

% For paths, translate using V matrix
%--------------------------------------------------------------------------
for u = 1:size(V,2)
    for t = 1:size(V,1)
        R(u) = R(u) + mp_log(P{t,1}(V(t,u))) - mp_log(E{1}(V(t,u)));
    end
end
R = mp_softmax(R);

% Map through link specifications (assign to higher level outcomes)
%--------------------------------------------------------------------------
for g = 1:numel(y)
    if any(linkD(:,g))
        y(g) = O(linkD(:,g));
    end
    if any(linkE(:,g))
        y(g) = {R};
    end
end

o.o = y;
o.pomdp = MDP;

function [o,s] = mdp_episodic_gen(s,u,Qo,pomdp)
% This function gives the generative process that models the dynamics of 
% the environment and the reflexive fulfillment of predictions.
%--------------------------------------------------------------------------

if nargout > 1 % Advance the states
    for i = 1:numel(pomdp.B)
        ind          = [{':'}, {s(i,end)}, num2cell(s(pomdp.dom.B(i).s,end)'),num2cell(u(pomdp.dom.B(i).u,end)')];
        [~,s(i,end)] = max(pomdp.B{i}(ind{:}));
    end
end

o = zeros(numel(pomdp.A),1);
for i = find(pomdp.n)'
    if ~isempty(Qo)
        [~,o(i)] = max(Qo{i}); % Fulfill vocal prediction
    else
        o(i) = 1;              % Or null vocalisation
    end
end
for i = setdiff(1:numel(pomdp.A),find(pomdp.n)')
    ind = [{':'}, num2cell(s(pomdp.dom.A(i).s,end)')];
    [~,o(i)] = max(pomdp.A{i}(ind{:}));
end
if s(3,end)~=1
    o(1) = 1;
    o(2) = 1;
end

