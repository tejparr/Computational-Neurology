function DEMO_Mountain
% This demo is designed to provide a test for a hypothesis space around
% apathy and impulsivity in an active inference context. The idea is that
% one is given the chance to climb a mountain, with the options available
% at each point in time being:
%
% 1. Stay still | 2. Take a step | 3. Take a leap
%
% One could imagine performing this task in an experimental psychology
% setting by allowing an action every 250-500ms, with a single button press
% in this interval correspinding to 2, a double button press to 3, and no
% button press in that interval to 1, thereby increasing the effort one
% must deploy from 1 -> 3.
% 
% Although the fastest possible way to climb the mountain would be a
% series of leaps, we assume there is a finite possibility that in the
% process of leaping, one ends up losing their footing and falling back
% down the hill. In contrast, they are much more sure-footed with their
% steps.
%
% This gives us a space of hypotheses to examine as possible explanations
% for apathy (e.g., deciding to just stay put at a lower altitude) and for
% impulsivity (e.g., taking unnecessary risks by leaping). The parameters
% available to manipulate to examine these hypotheses are detailed below
% along with their ranges.
%--------------------------------------------------------------------------

% For reproducibility
%--------------------------------------------------------------------------
rng default
close all
T = 16;    % Timesteps to simulate

% Hypothesis space: parameters to consider manipulating
%--------------------------------------------------------------------------
b1 = 0.8; % Probability of successful leap [0-1] - this parameter gives the probability a leap is successful and is assumed to be the same in the world and generative model
c  = 1;   % Slope of preferences [>0] - the greater this number, the greater the relative preference for higher altitudes
b2 = 8;   % Confidence in consequence of actions [>0] - when this parameter is low, the (beliefs about) transition probabilities between different courses of action become more mixed
e  = 0.5; % Effort cost [>0] - this determines how differently the participant experiences the effort associated with the 3 behavioural options above

% Specify generative model
%==========================================================================
pomdp = mdp_mountain_model(b1,b2,c,e,T);

% Invert (solve) model
%--------------------------------------------------------------------------
POMDP = mp_POMDP(pomdp);

% Plotting and animations
%--------------------------------------------------------------------------
mp_mountain_animate(POMDP);
mp_pomdp_belief_plot(POMDP);

cn_figure('Paths')
subplot(3,1,1)
imagesc(1-[POMDP.P{:}]), axis tight, colormap gray
title('Prior probabilities for paths')

return

% Recovery Analysis (single subject example)
%==========================================================================

% Create a set of trials
%--------------------------------------------------------------------------
b1 = linspace(0.1,0.9,4); %#ok % Each trial includes a different 'leap success' probability   
y  = cell(numel(b1),1);   % Initialise simulated data

% Model
%--------------------------------------------------------------------------
m0   = log(ones(3,1));        % Explanatory variables (note converted to log scale) log([c;b2;e])
pP   = eye(3)/16;             % Prior precision
pM   = zeros(3,1);            % Prior mean

for i = 1:numel(y)
    % Simulate behaviour for each trial
    %----------------------------------------------------------------------
    pomdp = mdp_mountain_model(b1(i),b2,c,e,T);
    POMDP = mp_POMDP(pomdp);
    y{i}.u   = POMDP.u;        % Sequence of actions
    y{i}.o   = POMDP.o;        % Sequence of observations
    y{i}.b1  = b1(i);          % Probability of successful leap
end

VariationalLaplace(@mdp_mountain_LL,pP,pM,m0,y);
gcf; subplot(2,1,2), hold on
bar(mp_log([b2;c;e]),'EdgeColor','none','FaceColor','r','BarWidth',0.1)

% Recovery Analysis (simulating multiple participants)
%==========================================================================
Np   = 16;              % Number of participants to simulate
tP   = randn(3,Np)*4;   % Initialise array of 'true' parameters
eP.m = zeros(3,Np);     % Initialise array of estimated modes
eP.C = zeros(3,Np);     % Initialise array of estimated variance (diagonals of inverse precisions)

for j = 1:Np
    y  = cell(numel(b1),1);   % Initialise simulated data for this participant
    
    % Set 'true' parameters for simulation
    b2 = tP(1,j);
    c  = tP(2,j);
    e  = tP(3,j);

    % Model
    %--------------------------------------------------------------------------
    m0   = log(ones(3,1));        % Explanatory variables (note converted to log scale) log([b2;c;e])
    pP   = eye(3)/16;             % Prior precision
    pM   = zeros(3,1);            % Prior mean

    for i = 1:numel(y)
        % Simulate behaviour for each trial
        %------------------------------------------------------------------
        pomdp = mdp_mountain_model(b1(i),exp(b2),exp(c),exp(e),T); % Exponentials enforce positivity
        POMDP = mp_POMDP(pomdp);
        y{i}.u   = POMDP.u;        % Sequence of actions
        y{i}.o   = POMDP.o;        % Sequence of observations
        y{i}.b1  = b1(i);          % Probability of successful leap
    end

    [qM,qP,~] = VariationalLaplace(@mdp_mountain_LL,pP,pM,m0,y);
    qC   = inv(qP);
    eP.m(:,j) = qM;
    eP.C(:,j) = diag(qC);
end

cn_figure('Recovery')
for i = 1:3
    subplot(3,1,i)
    errorbar(tP(i,:), eP.m(i,:), 2*sqrt(eP.C(i,:)),'.','MarkerFaceColor',[0.8,0.8,0.9],'MarkerSize',16,'CapSize',0,'LineWidth',2)
    hold on
    plot([-max(abs(tP(i,:)))-2, max(abs(tP(i,:)))+2],[-max(abs(eP.m(i,:)))-2, max(abs(eP.m(i,:)))+2],'--k')
    axis([-max(abs(tP(i,:)))-2, max(abs(tP(i,:)))+2, -max(abs(eP.m(i,:)))-2, max(abs(eP.m(i,:)))+2])
    axis square
    box off

    ax = gca;
    ax.XRuler.Axle.Visible = 'off';
    ax.YRuler.Axle.Visible = 'off';
    xline(0,'k')
    yline(0,'k')
    xlabel('True parameter')
    ylabel('Estimated parameter')
    title(['Parameter ' num2str(i)])
end

function pomdp = mdp_mountain_model(b1,b2,c,e,T)
% Construct generative model based upon parameters as specified at the
% start of this script
%--------------------------------------------------------------------------

% Initial states
%--------------------------------------------------------------------------
D{1} = [1;zeros(7,1)];    % Position on mountain [foothills -> top mountain]

% Sensory mappings (i.e., likelihood probabilities)
%--------------------------------------------------------------------------
A{1} = eye(length(D{1})); % Direct visual mapping to location

% Transition probabilities
%--------------------------------------------------------------------------
B{1}(:,:,1) = eye(length(D{1}));    % Do nothing
B{1}(:,:,2) = zeros(length(D{1}));  % Take a step
B{1}(:,:,3) = zeros(length(D{1}));  % Take a leap

for i = 1:size(B{1},2)
    B{1}(min(i+1,size(B{1},2)),i,2) = 1;
    B{1}(min(i+2,size(B{1},2)),i,3) = b1;
    B{1}(max(i-2,1),i,3) = 1-b1;
end

% Separate out transition matrices between world and model
%--------------------------------------------------------------------------
wB = B;                         % World's transition matrix
W  = mp_softmax(b2*eye(3));     % Mixing weights
B{1} = mp_tensor_con(B{1},W,[1 2 4],[1 2 3],[3 4]);

% Preferences
%--------------------------------------------------------------------------
C{1} = c*(0:size(A{1},1)-1)';

% Conditional dependencies (i.e., domains of conditional distributions)
%--------------------------------------------------------------------------
dom.A(1).s = 1;
dom.A(1).u = [];
dom.B(1).s = [];
dom.B(1).u = 1;

% Compile partially observed Markov Decision Process model
%--------------------------------------------------------------------------
pomdp.dom = dom;
pomdp.A = A;
pomdp.B = B;
pomdp.wB = wB; % world's B
pomdp.C = C;
pomdp.D = D;
pomdp.E{1} = mp_softmax(e*(2:-1:0)');
pomdp.T = T;
pomdp.s = 1;
pomdp.gen = @mdp_mountain_gen; % Generative process (simulation environment)
pomdp.randact = [];            % Sample actions rather than MAP

function [o,s] = mdp_mountain_gen(s,u,~,pomdp)
% Generative process that plays the role of a simulated environment. Here,
% the generative process matches the transition structure of the generative
% model.
%--------------------------------------------------------------------------

if nargout > 1 % Advance the states
    i1 = [{':'},num2cell(s(1)),num2cell(s(pomdp.dom.B(1).s)),num2cell(u(pomdp.dom.B(1).u))];
    s1 = pomdp.wB{1}(i1{:});
    s(1) = find(cumsum(s1)>rand,1,'first');
end

o = s; % Generate outcomes

function mp_mountain_animate(POMDP)
% This script produces an animation of a single run of the task
%--------------------------------------------------------------------------
cn_figure('Animation') % standardise figure

% Extract results of simulation
%--------------------------------------------------------------------------
o = POMDP.o;                % 'True' outcomes
T = numel(o);               % Length of trial
w = linspace(0, 1, 16);     % Precompute interpolation weights for smooth animation

% Mountain scenery
%--------------------------------------------------------------------------
x = 0:256;                               % Horizontal range
y = 200*exp(-((200 - x).^2) / 20000);    % Mountain
xb = [x fliplr(x)];                      % Sky
yb = [zeros(size(y)),256*ones(size(y))]; % Sky

% Plot sky
%--------------------------------------------------------------------------
fill(xb, yb, [0.6 0.6 0.8], 'EdgeColor', 'none');
axis equal
axis([min(xb) max(xb) min(yb) max(yb)])
axis off
hold on

% Add clouds
%--------------------------------------------------------------------------
cl1 = mountain_clouds( 0.1*max(xb), 0.7*max(yb), 32);
cl2 = mountain_clouds( 0.6*max(xb), 0.9*max(yb), 32);
cl3 = mountain_clouds(-0.3*max(xb), 0.9*max(yb), 32);

% Add mountain
%--------------------------------------------------------------------------
xf = [x fliplr(x)];
yf = [zeros(size(y)) fliplr(y)];
fill(xf, yf, [0.6 0.8 0.6], 'EdgeColor', 'none');

% Create participant
%--------------------------------------------------------------------------
hHead  = plot(nan, nan, '.k', 'MarkerSize', 10);
hTrunk = plot([0 0],[4 8],'k','LineWidth',2);
hLegs  = plot([0 2 4],[0 4 0],'k','LineWidth',1);

% Animation loop
%--------------------------------------------------------------------------
for t = 1:T-1

    o1 = o(t);
    o2 = o(t+1);
    j  = zeros(1,16);
    k  = zeros(1,16);
    if abs(o1-o2)>1
        j = cos(linspace(-pi/2,pi/2,16))*32;
        k = repmat(linspace(-pi/4,pi/4,4),[1 4]);
    elseif abs(o1-o2)==1
        k = repmat(linspace(-pi/4,pi/4,4),[1 4]);
    end

    % Vectorized interpolation
    d = round((200*((1 - w)*o1 + w*o2))/8);

    % Clamp indices to valid range
    d = max(1, min(numel(x), d));

    for i = 1:16
        hHead.XData  = x(d(i));
        hHead.YData  = y(d(i)) + j(i) + 8;
        hTrunk.XData = ones(1,2)*x(d(i));
        hTrunk.YData = y(d(i)) + j(i) + [4 8];
        hLegs.XData  = x(d(i)) + [sin(k(i)) 0 -sin(k(i))];
        hLegs.YData  = y(d(i)) + j(i) + [cos(k(i)) 4 cos(k(i))];
        cl1.XData = cl1.XData + 0.5;
        cl2.XData = cl2.XData + 0.5;
        cl3.XData = cl3.XData + 0.5;

        drawnow limitrate
        pause(0.1)
        %==================================================================
        % Uncomment next line to save GIF
        % cn_animation(t + i - 1,1,1/4,'Graphics','Mountain_Animation')
        %==================================================================        
    end
end

function cloud = mountain_clouds(cx, cy, s)
% Function to produce clouds for visualisation. (cx,cy) gives centre of
% cloud, and s allows for scaling.
%--------------------------------------------------------------------------

t = linspace(0, 2*pi, 300);

x = cx + s*(1.6.*cos(t) + 0.05*cos(8*t) + 0.04*sin(4*t));
y = cy + s*(1.0.*sin(t) + 0.05*sin(8*t) + 0.04*cos(4*t));

cloud = fill(x,y,[1 1 1],'EdgeColor','none'); 

function L = mdp_mountain_LL(y,x) %#ok
% Log likelihood for this model when fitting to behaviour
%--------------------------------------------------------------------------
if iscell(y) % If multiple trials in play
    L = 0;   % Initialise log likelihood
    for i = 1:numel(y)
        pomdp = mdp_mountain_model(y{i}.b1,exp(x(1)),exp(x(2)),exp(x(3)),length(y{i}.u));
        pomdp.o = y{i}.o;
        POMDP = mp_POMDP(pomdp);
        P     = [POMDP.P{:}];
        L     = L + sum(mp_log(P(sub2ind(size(P),y{i}.u,1:size(P,2)))));
    end
else     % If only a single trial
    pomdp = mdp_mountain_model(y.b1,exp(x(1)),exp(x(2)),exp(x(3)),length(y.u));
    pomdp.o = y.o;
    POMDP = mp_POMDP(pomdp);
    P     = [POMDP.P{:}];
    L     = sum(mp_log(P(sub2ind(size(P),y.u,1:size(P,2)))));
end


