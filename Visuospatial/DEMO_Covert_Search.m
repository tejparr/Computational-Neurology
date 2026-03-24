function MDP = DEMO_Covert_Search
% This demo is designed to illustrate covert attentional sampling. A core
% feature of this demo is that one represents the target for which colour
% should be reported, rather than representing the colours of all possible
% targets. This action-oriented model, combined with an internal selection
% of precise modalities (without overt eye movements), allows for a
% directed and context sensitive covert visual search.
%--------------------------------------------------------------------------

close all
cd(fileparts(mfilename('fullpath')))
OPTIONS.save = 0;

rng default
close all

EXP = 1; % Choose experiment to simulate (1 = static array or 2 = dynamic streams)

%% EXPERIMENT 1
%==========================================================================
% This deals with an experiment in which there is a static display
% (presented after a pause) of all stimuli at once and in which one must
% identify the relevant shaped target based upon a cue (before the pause)
% and report its color
%==========================================================================

e  = 2/7;  % bias towards alignment between fovea and focus of attention [0-1] (<1/7 is aversion for focusing on fovea)
a1 = 8/10; % attentional focus (colour) [0-1]
a2 = 9/10; % attentional focus (shape) [0-1]
a3 = 9/10; % suprise for observing correct shape when unattended [0-1] (in effect, 'bottom-up salience' once primed)
c1 = 3;    % strength of preference for being correct
c2 = 6;    % strength of aversion for being wrong

switch EXP; case 1

T  = 16;   % time-steps to simulate (stimuli appear at time-step 8)

% Experimental setup
%--------------------------------------------------------------------------
% These give the colours and shapes at 6 different locations arranged
% radially around the central fixation (ensure the target index is a unique
% shape).

X.col = [1 2 3 1 3 2];
X.sha = [1 3 3 2 1 1];
X.tar = 4;

% Initial states
%--------------------------------------------------------------------------
D{1} = zeros(7,1); D{1}(7) = 1; % Focus of covert attention (6 radial locations and central fixation)
D{2} = ones(6,1)/6;             % Location of target
D{3} = ones(3,1)/3;             % Color of target (R/G/B)
D{4} = ones(3,1)/3;             % Shape of target (T/S/C)
D{5} = zeros(4,1); D{5}(4) = 1; % Button press (R/G/B/none)
D{6} = zeros(8,1); D{6}(1) = 1; % Stage of task (cue, delay, stimulus)

Ns   = zeros(numel(D),1);
for f = 1:length(Ns)
    Ns(f) = numel(D{f});
end

% Likelihood
%--------------------------------------------------------------------------
A = cell(Ns(2)+3,1);
for f1 = 1:Ns(1)
    for f2 = 1:Ns(2)
        for f3 = 1:Ns(3)
            for f4 = 1:Ns(4)
                for f5 = 1:Ns(5)
                    for f6 = 1:Ns(6)
                        % For each of the possible locations, there is an
                        % outcome modality reporting colour and shape. When the
                        % focus of attention and the target location match, the
                        % colour and shape are predicted with high precision
                        % Otherwise, the probability is distributed across all
                        % shapes and all colours
                        for g = 1:Ns(2)
                            if g==f1
                                if f1==f2 && f6>7
                                    A{g}(f3,f1,f2,f3,f6) = a1;       % Color
                                    A{g}(setdiff(1:Ns(3)+1,f3),f1,f2,f3,f6) = (1-a1)/Ns(3);
                                    A{g+Ns(2)}(f4,f1,f2,f4,f6) = a2; % Shape
                                    A{g+Ns(2)}(setdiff(1:Ns(4)+1,f4),f1,f2,f4,f6) = (1-a2)/Ns(4);
                                else  
                                    A{g}(1:Ns(3)+1,f1,f2,f3,f6) = 1/(Ns(3)+1);   % Color
                                    A{g+Ns(2)}(f4,f1,f2,f4,f6) = 1-a2; % Shape
                                    A{g+Ns(2)}(setdiff(1:Ns(4)+1,f4),f1,f2,f4,f6) = a2/Ns(4);
                                end
                            else
                                A{g}(1:Ns(3)+1,f1,f2,f3,f6) = 1/(Ns(3)+1); % Color
                                A{g+Ns(2)}(f4,f1,f2,f4,f6) = 1-a3; % Shape
                                A{g+Ns(2)}(setdiff(1:Ns(4)+1,f4),f1,f2,f4,f6) = a3/Ns(4);
                            end
                        end

                        % Instruction - this directly reports the shape of the
                        % target when focus is on the fovea
                        if f1==Ns(2)+1
                            if f6<5 % if the cue is present
                                A{2*Ns(2)+1}(f4,f1,f4,f6) = 1;
                            else
                                A{2*Ns(2)+1}(4,f1,f4,f6) = 1;
                            end
                        else
                            A{2*Ns(2)+1}(1:4,f1,f4,f6) = 1/4;
                        end

                        % Proprioception - reports button press
                        A{2*Ns(2)+2}(f5,f5) = 1;

                        % Feedback - this reports correct when the button press
                        % matches the target color, incorrect for alternative
                        % buttons, and neutral for no presses
                        if f5 == 4
                            A{2*Ns(2)+3}(2,f3,f5) = 1; % Neutral (no button pressed)
                        elseif f5 == f3
                            A{2*Ns(2)+3}(1,f3,f5) = 1; % Correct
                        else
                            A{2*Ns(2)+3}(3,f3,f5) = 1; % Incorrect
                        end
                    end
                end
            end
        end
    end
end

% Transitions
%--------------------------------------------------------------------------
B = cell(size(D));
% Controllable focus of attention
B{1} = zeros(Ns(1),Ns(1),Ns(1));
for k = 1:Ns(1)
    B{1}(k,:,k) = 1;
end

% Fixed states (identity transitions)
for s = 2:4
    B{s} = eye(Ns(s));
end

% Controllable but irreversible button press
B{5} = zeros(Ns(5),Ns(5),Ns(5));
for k = 1:Ns(5)
    B{5}(1:Ns(5)-1,1:Ns(5)-1,k) = eye(Ns(5)-1);
    B{5}(k,Ns(5),k) = 1;
end

% Progression of task
B{6} = circshift(eye(size(D{6},1)),1); B{6}(1,end) = 0; B{6}(end,end) = 1;

% Preferences
%--------------------------------------------------------------------------
C = cell(numel(A),1);
for g = 1:numel(A)
    C{g} = zeros(size(A{g},1),1);
end
C{end} = [c1;0;-c2]; % Preference for correctness

% Policies
%--------------------------------------------------------------------------
E{1} = [ones(Ns(1)-1,1)*(1-e)/(Ns(1)-1);e];
E{2} = ones(Ns(5),1)/Ns(5);

% Compile MDP
%--------------------------------------------------------------------------
mdp.A = A;
mdp.B = B;
mdp.C = C;
mdp.D = D;
mdp.E = E;

% Domains of probability distributions
%--------------------------------------------------------------------------
for i = 1:numel(D)
    mdp.dom.B(i).s = [];
    mdp.dom.B(i).u = [];
end

% assign policies to their states
mdp.dom.B(1).u = 1;
mdp.dom.B(5).u = 2;

for i = 1:Ns(2)
    mdp.dom.A(i).s = [1 2 3 6];
    mdp.dom.A(i).u = [];
end
for i = Ns(2)+1:2*Ns(2)
    mdp.dom.A(i).s = [1 2 4 6];
    mdp.dom.A(i).u = [];
end

mdp.dom.A(2*Ns(2)+1).s = [1 4 6];
mdp.dom.A(2*Ns(2)+1).u = [];
mdp.dom.A(2*Ns(2)+2).s = 5;
mdp.dom.A(2*Ns(2)+2).u = [];
mdp.dom.A(2*Ns(2)+3).s = [3 5];
mdp.dom.A(2*Ns(2)+3).u = [];

% Simulation settings and environment (generative process)
%--------------------------------------------------------------------------
mdp.T   = T;
mdp.gen = @mdp_covert_search_gen;
mdp.N   = 1;                % Planning horizon
mdp.fac = 1;                % Option to separate evaluation of likelihood and transitions into 2 steps
mdp.s   = [7 X.tar X.col(X.tar) X.sha(X.tar) 4 1]'; % Initial states
mdp.GP  = X; % for gen process

% Solve MDP
%--------------------------------------------------------------------------
MDP = mp_POMDP(mdp);

mdp_plot_covert(MDP,OPTIONS)
mp_pomdp_belief_plot(MDP);

case 2

%% EXPERIMENT 2
%==========================================================================
% This deals with a dynamic display, treated using exactly the same
% principles as above, in which there are two locations one can attend to
% either the left or right location, following a cue, and must report the
% colour of the relevant shape as before. The relevant stimulus appears at
% one of the two locations amongst a stream of irrelevant stimuli.
%==========================================================================

e  = 2/6;  % bias towards alignment between fovea and focus of attention [0-1] (<1/3 is aversion for focusing on fovea)
a3 = 1;    % successful behaviour in this experiment requires a stronger 'priming' effect than above

% Experimental setup
%--------------------------------------------------------------------------
% These give the colours and shapes at 2 different locations over time
% (ensure the target index is a unique shape).

X.col = [1 2 3 1 3 2;  % Left colours
         2 2 1 3 1 3]; % Right colours
X.sha = [1 3 3 2 1 1;  % Left shapes
         3 3 1 1 3 1]; % Right shapes
X.tar = [1 4];         % [left v right, time index]

% Initial states
%--------------------------------------------------------------------------
D{1} = zeros(3,1);  D{1}(3) = 1; % Focus of covert attention (2 peripheral locations and central fixation)
D{2} = ones(2,1)/2;              % Location of target
D{3} = ones(3,1)/3;              % Color of target (R/G/B)
D{4} = ones(3,1)/3;              % Shape of target (T/S/C)
D{5} = zeros(4,1);  D{5}(4) = 1; % Button press (R/G/B/none)
D{6} = zeros(10,1); D{6}(1) = 1; % Stage of task (cue, delay, distractor, stimulus, distractor)

Ns   = zeros(numel(D),1);
for f = 1:length(Ns)
    Ns(f) = numel(D{f});
end

% Likelihood
%--------------------------------------------------------------------------
A = cell(Ns(2)+3,1);
for f1 = 1:Ns(1)
    for f2 = 1:Ns(2)
        for f3 = 1:Ns(3)
            for f4 = 1:Ns(4)
                for f5 = 1:Ns(5)
                    for f6 = 1:Ns(6)
                        % For each of the possible locations, there is an
                        % outcome modality reporting colour and shape. When the
                        % focus of attention and the target location match, and
                        % if the task stage is consistent with the stimulus then
                        % colour and shape are predicted with high precision
                        % Otherwise, the probability is distributed across all
                        % shapes and all colours
                        for g = 1:Ns(2)
                            if g==f1 % If focus of attention matches element of visual field
                                if f6==9 && f1==f2 % and if focus of attention matches stimulus location at the right time
                                    A{g}(f3,f1,f2,f3,f6) = a1;       % Color
                                    A{g}(setdiff(1:Ns(3)+1,f3),f1,f2,f3,f6) = (1-a1)/Ns(3);
                                    A{g+Ns(2)}(f4,f1,f2,f4,f6) = a2; % Shape
                                    A{g+Ns(2)}(setdiff(1:Ns(4)+1,f4),f1,f2,f4,f6) = (1-a2)/Ns(4);
                                else % or if stimulus either at another location or not currently visible
                                    A{g}(1:Ns(3)+1,f1,f2,f3,f6) = 1/(Ns(3)+1); % Color
                                    A{g+Ns(2)}(f4,f1,f2,f4,f6) = 1-a2; % Shape
                                    A{g+Ns(2)}(setdiff(1:Ns(4)+1,f4),f1,f2,f4,f6) = a2/Ns(4);
                                end
                            else % Otherwise, if non-attended location, imprecise
                                A{g}(1:Ns(3)+1,f1,f2,f3,f6) = 1/(Ns(3)+1); % Color
                                A{g+Ns(2)}(f4,f1,f2,f4,f6) = 1-a3; % Shape
                                A{g+Ns(2)}(setdiff(1:Ns(4)+1,f4),f1,f2,f4,f6) = a3/Ns(4);
                            end
                        end

                        % Instruction - this directly reports the shape of the
                        % target when focus is on the fovea
                        if f1==Ns(2)+1
                            if f6<5 % if the cue is present
                                A{2*Ns(2)+1}(f4,f1,f4,f6) = 1;
                            else
                                A{2*Ns(2)+1}(4,f1,f4,f6) = 1;
                            end
                        else
                            A{2*Ns(2)+1}(1:4,f1,f4,f6) = 1/4;
                        end

                        % Proprioception - reports button press
                        A{2*Ns(2)+2}(f5,f5) = 1;

                        % Feedback - this reports correct when the button press
                        % matches the target color, incorrect for alternative
                        % buttons, and neutral for no presses
                        if f5 == 4
                            A{2*Ns(2)+3}(2,f3,f5) = 1; % Neutral (no button pressed)
                        elseif f5 == f3
                            A{2*Ns(2)+3}(1,f3,f5) = 1; % Correct
                        else
                            A{2*Ns(2)+3}(3,f3,f5) = 1; % Incorrect
                        end
                    end
                end
            end
        end
    end
end

% Transitions
%--------------------------------------------------------------------------
B = cell(size(D));

% Controllable focus of attention
B{1} = zeros(Ns(1),Ns(1),Ns(1));
for k = 1:Ns(1)
    B{1}(k,:,k) = 1;
end

% Fixed states (identity transitions)
for s = 2:4
    B{s} = eye(Ns(s));
end

% Controllable but irreversible button press
B{5} = zeros(Ns(5),Ns(5),Ns(5));
for k = 1:Ns(5)
    B{5}(1:Ns(5)-1,1:Ns(5)-1,k) = eye(Ns(5)-1);
    B{5}(k,Ns(5),k) = 1;
end

% Progression of task
B{6} = circshift(eye(size(D{6},1)),1); B{6}(1,end) = 0; B{6}(end,end) = 1;
B{6}(9,7) = 1; % progression from delay to stimulus
B{6}(8,8) = 1; % progression from pre-stimulus distractor to distractor
B{6}(9,9) = 1; % progression from stimulus to stimulus
B{6} = mp_norm(B{6});

% Preferences
%--------------------------------------------------------------------------
C = cell(numel(A),1);
for g = 1:numel(A)
    C{g} = zeros(size(A{g},1),1);
end
C{end} = [c1;0;-c2]; % Preference for correctness

% Policies
%--------------------------------------------------------------------------
E{1} = [ones(Ns(1)-1,1)*(1-e)/(Ns(1)-1);e];
E{2} = ones(Ns(5),1)/Ns(5);

% Compile MDP
%--------------------------------------------------------------------------
mdp.A = A;
mdp.B = B;
mdp.C = C;
mdp.D = D;
mdp.E = E;

% Domains of probability distributions
%--------------------------------------------------------------------------
for i = 1:numel(D)
    mdp.dom.B(i).s = [];
    mdp.dom.B(i).u = [];
end

% assign policies to their states
mdp.dom.B(1).u = 1;
mdp.dom.B(5).u = 2;

for i = 1:Ns(2)
    mdp.dom.A(i).s = [1 2 3 6];
    mdp.dom.A(i).u = [];
end
for i = Ns(2)+1:2*Ns(2)
    mdp.dom.A(i).s = [1 2 4 6];
    mdp.dom.A(i).u = [];
end

mdp.dom.A(2*Ns(2)+1).s = [1 4 6];
mdp.dom.A(2*Ns(2)+1).u = [];
mdp.dom.A(2*Ns(2)+2).s = 5;
mdp.dom.A(2*Ns(2)+2).u = [];
mdp.dom.A(2*Ns(2)+3).s = [3 5];
mdp.dom.A(2*Ns(2)+3).u = [];

% Simulation settings and environment (generative process)
%--------------------------------------------------------------------------
mdp.dt  = 2;                % Duration of stimulus presentation (in discrete time-steps)
mdp.T   = size(X.sha,2)*mdp.dt + 7;
mdp.gen = @mdp_covert_search_gen;
mdp.N   = 1;                % Planning horizon
mdp.fac = 1;                % Option to separate evaluation of likelihood and transitions into 2 steps
mdp.s   = [3 X.tar(1) X.col(X.tar(1),X.tar(2)) X.sha(X.tar(1),X.tar(2)) 4 1]'; % Initial states
mdp.GP  = X; % for gen process

% Solve MDP
%--------------------------------------------------------------------------
MDP = mp_POMDP(mdp);

mdp_plot_covert(MDP,OPTIONS)
mp_pomdp_belief_plot(MDP);

end

function [o,s] = mdp_covert_search_gen(s,u,~,pomdp)
% Generative process
%--------------------------------------------------------------------------
GP = pomdp.GP;
if nargout > 1 % Advance the states
    if s(5)==4
        s(5) = u(2);
    end
    if isscalar(GP.tar)
        s(6) = min(s(6)+1,8);
    else
        s(6) = s(6)+1;
    end
end
if s(5)==s(3)
    bp = 1;
elseif s(5)==4
    bp = 2;
else 
    bp = 3;
end

if isscalar(GP.tar)
    if s(6)<5
        o   = [4*ones(size(GP.col)),4*ones(size(GP.sha)),GP.sha(GP.tar),s(5),bp]';
    elseif s(6)<8
        o   = [4*ones(size(GP.col)),4*ones(size(GP.sha)),4,s(5),bp]';
    else
        o   = [GP.col,GP.sha,4,s(5),bp]';
    end
else
    if s(6)<5
        o   = [4*ones(1,size(GP.col,1)),4*ones(1,size(GP.sha,1)),GP.sha(GP.tar(1),GP.tar(2)),s(5),bp]';
    elseif s(6)<8
        o   = [4*ones(1,size(GP.col,1)),4*ones(1,size(GP.sha,1)),4,s(5),bp]';
    else
        r   = ceil((s(6)-7)/pomdp.dt);
        o   = [GP.col(1,r),GP.col(2,r),GP.sha(1,r),GP.sha(2,r),4,s(5),bp]';
    end
end

function mdp_plot_covert(pomdp,OPTIONS)
% Plotting function for animation
%--------------------------------------------------------------------------
cn_figure('Animation'); clf

% Setup
%--------------------------------------------------------------------------
GP  = pomdp.GP;               % Parameters of generative process (experimental set-up)
col = {'r','g','b','w'};      % Colour key
sha = {'s','^','o'};          % Shape key
shc = {'S','T','C'};          % Shape cues

if isscalar(GP.tar)
    c   = [cos((1/6:1/6:1)*2*pi); % Coordinates for locations of stimuli
        sin((1/6:1/6:1)*2*pi)];
else
    c = [-1 1;0 0];
end

nc  = size(c,2);              % Number of locations
ns = numel(sha);              % Number of possible shapes     
markerSize = 16^2;            % For shape plots ^2 accounts for difference in scatter versus plot

% Plot initial setup of (dynamic) experimental stimulus display
%--------------------------------------------------------------------------
subplot(3,1,1)
hold on
axis([-1.5 1.5 -1.5 1.5])
axis equal off
title('Stimuli')

% Graphics handles and objects for efficient animation
%--------------------------------------------------------------------------
iStim   = gobjects(nc,1);

for k = 1:nc
    iStim(k) = plot(c(1,k), c(2,k), sha{GP.sha(k)}, ...
        'MarkerEdgeColor', col{end}, ...
        'MarkerFaceColor', col{end}, ...
        'LineStyle','none','MarkerSize',16);
end

iCentre = text(0, 0, shc{pomdp.s(4,1)}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize', 18, ...       
        'FontWeight','bold', ...
        'Color', [0 0 0]);


% Plot initial setup of (dynamic) belief plots
%--------------------------------------------------------------------------
subplot(3,1,2)
hold on
axis([-1.5 1.5 -1.5 1.5])
axis equal off
title('Predictions')

% Graphics handles and objects for efficient animation
%--------------------------------------------------------------------------
hStim   = gobjects(nc, ns);
hCentre = gobjects(1, ns);

for k = 1:nc % Stimuli in peripheral locations
    for j = 1:ns
        hStim(k,j) = scatter(c(1,k), c(2,k), markerSize, ...
            'Marker', sha{j}, ...
            'MarkerEdgeColor', [1 1 1], ...
            'MarkerFaceColor', [1 1 1], ...
            'MarkerFaceAlpha', 0, ...
            'MarkerEdgeAlpha', 0);
    end
end

for j = 1:ns % Stimulus in central location (i.e., cued shape)
    hCentre(j) = text(0, 0, shc{j}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize', 18, ...       
        'FontWeight','bold', ...
        'Color', 'k');
end

% Plot (dynamic) to illustrate buttons for different responses
%--------------------------------------------------------------------------
subplot(3,1,3)
hold on
axis([-2 6 -1 2])
axis off
title('Response')

% And graphics objects (buttons) for efficient animation
%--------------------------------------------------------------------------
xBtn = 1:3;
yBtn = ones(1,3);
hBtn = gobjects(1,3);

for j = 1:3
    hBtn(j) = scatter(xBtn(j), yBtn(j), 600, 's', ...
        'MarkerFaceColor', 0.85*[1 1 1], ...
        'MarkerEdgeColor', col{j}, ...
        'LineWidth', 2);
end

% Animation
%--------------------------------------------------------------------------
O = cell(numel(pomdp.A),1); % Preallocate predictions

for t = 1:pomdp.T           % Time loop
    % Update stimulus display
    if t==5
        iCentre.Color = 'w';
        iCentre.Color = 'w';
    elseif t==8 && isscalar(GP.tar)
        for k = 1:nc
            iStim(k).MarkerEdgeColor = col{GP.col(k)};
            iStim(k).MarkerFaceColor = col{GP.col(k)};
        end
    elseif t>7 && ~isscalar(GP.tar)
        r   = ceil((t-7)/pomdp.dt);
        for k = 1:nc
            iStim(k).MarkerEdgeColor = col{GP.col(k,r)};
            iStim(k).MarkerFaceColor = col{GP.col(k,r)};
            iStim(k).Marker = sha{GP.sha(k,r)};
        end
    end
    
    Q = pomdp.Q(t,:);       % Beliefs (posteriors) at each time-point


    % Compute (posterior) predictions from posteriors and likelihood
    %----------------------------------------------------------------------
    for g = 1:numel(pomdp.A)
        O{g} = mp_dot(pomdp.A{g}, Q(pomdp.dom.A(g).s));
    end

    % Plot peripheral stimuli
    %----------------------------------------------------------------------
    for k = 1:nc
        for j = 1:3
            hStim(k,j).MarkerFaceColor = O{k}(1:3)';
            hStim(k,j).MarkerEdgeColor = O{k}(1:3)';
            hStim(k,j).MarkerFaceAlpha = O{k + nc}(j);
            hStim(k,j).MarkerEdgeAlpha = O{k + nc}(j);
        end
    end

    % Plot central cue
    %----------------------------------------------------------------------
    for j = 1:3
        gray = (1 - O{2*nc + 1}(j)) * [1 1 1];
        hCentre(j).Color = gray;
    end

    % Determine which (if any) button is pressed
    %----------------------------------------------------------------------
    resp = pomdp.o(end-1,t);  

    % Reset all buttons
    %----------------------------------------------------------------------
    for j = 1:3
        hBtn(j).MarkerFaceColor = 0.85*[1 1 1];
    end

    % Depress active button
    %----------------------------------------------------------------------
    if resp >= 1 && resp <= 3
        hBtn(resp).MarkerFaceColor = col{resp,:};
    end
    pause(1)
    drawnow limitrate

    % Animation
    % -----------------------------------------------------------------
    if OPTIONS.save
        F  = getframe(gcf);
        im = frame2im(F);
        [MM,MMM] = rgb2ind(im,256);
        if t==1
            imwrite(MM,MMM,'Graphics/Animation.gif','gif','LoopCount',Inf,'DelayTime',1);
        else
            imwrite(MM,MMM,'Graphics/Animation.gif','gif','WriteMode','append','DelayTime',0.3);
        end
    end
end
