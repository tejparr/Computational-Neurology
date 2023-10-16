function [Y,M,t,a] = MetronomeModel(OPTIONS,P,y)
% This function specifies and inverts the generative model for our 
% metronome task, taking the parameters of our model and optionally data y
% and returning outstanding simulated data Y, our belief trajectory M,
% timesteps t, and action a. OPTIONS inlcludes option for plotting.
%
% Thomas Parr
%--------------------------------------------------------------------------

% Preliminaries
%--------------------------------------------------------------------------
No   = 4;               % Number of clocks
Nt   = 2056;            % Number of timesteps
dt   = 1/4;             % Length of timestep
tT   = (0:Nt)*dt;       % Time
T    = round(tT(end));  % Final time

%% Generative model
%-------------------------------------------------------------------------- 

% Equations of motion
%--------------------------------------------------------------------------

% Jacobian for pendular dynamics
%--------------------------------------------------------------------------
for i = 1:2:2*No 
    Jf(i:i+1,i:i+1) = [0  i;
                      -i  0]/32;
end

% Transition dynamics
%--------------------------------------------------------------------------
b11  = 0; b22 = 0;
B    = [b11   1-b22;
       1-b11   b22];

% Attractor locations
%--------------------------------------------------------------------------
tgt = [-1 1];

% Combine pendular dynamics with transition and attractor dynamics
%--------------------------------------------------------------------------

smax = @(x) exp(x)/sum(exp(x));             % Softmax function
sig  = @(x) 1/(1+exp(-4*x));                % Sigmoid function
try gamma = P.gamma; catch, gamma = 2; end  % Precision of policy selection

f = @(x) [Jf*x(1:size(Jf,2),1);                                                                 % oscillators as above
          sig(sum(x(1:2:size(Jf,2)))-2)*(smax(gamma*x(end-2:end-1,1))-x(end-4:end-3,1))/4;      % current state
          (1-sig(sum(x(1:2:size(Jf,2)))))*(log(B*x(end-4:end-3,1)+exp(-2))-x(end-2:end-1,1))/4; % next state
          (tgt*x(end-4:end-3,1) - x(end))/4];                                                   % attractor dynamics

% Prediction of data (i.e., mode of likelihood)
%--------------------------------------------------------------------------
g = @(x) [exp(sum(x(1:2:end-5)))/64; x(end)];

% Precisions and orders of motion
%--------------------------------------------------------------------------
Pg      = diag([exp(2) exp(0)]); % Precision of likelihood
Pf      = exp(4)*eye(2*No+5);    % Precision of dynamics
Pf(end) = exp(0);                % Less precise controllable dynamics
n       = 3;                     % Order of generalised motion
s       = 1;                     % Smoothness
m0      = randn(2*No + 5,1)/16;  % Initial beliefs
m0(end-4:end-1) = [1;0;0;1];

%% Generative process
%--------------------------------------------------------------------------
x0      = zeros(2*No + 5,1);     % Initial states (for generating data)
a0      = 0;                     % Initial active states

try   x0(2*OPTIONS.freq) = 4;    % Option to set pendulum frequency
catch, x0(4)   = 4; end          % Non-zero amplitude for second pendulum


% Equations of motion
%--------------------------------------------------------------------------
F = @(x,a) [Jf*x(1:size(Jf,2),1);                                                     % oscillators as above
            sig(sum(x(1:2:size(Jf,2)))-2)*(x(end-2:end-1,1)-x(end-4:end-3,1))/4;      % current state
            (1-sig(sum(x(1:2:size(Jf,2)))))*(B*x(end-4:end-3,1)-x(end-2:end-1,1))/4;  % next state
            a];                                                                       % attractor dynamics

%% Model inversion
%--------------------------------------------------------------------------

if exist('y','var')
    [Y,M,t,a] = ActiveFiltering(f,F,Pf,g,g,Pg,x0,m0,n,s,dt,T,a0,y,2);
else
    [Y,M,t,a] = ActiveFiltering(f,F,Pf,g,g,Pg,x0,m0,n,s,dt,T,a0);
end

if OPTIONS.plot == 0, return, end

% Plotting
%--------------------------------------------------------------------------
% The first plot gives a summary of the behaviour of the variables in the
% model. These include the data presented to our agent and its predictions
% of these data (dashed) in the upper plot, beliefs about the trajectory of
% (imaginary) oscillators in the secon plot, beliefs involved in sequencing
% action in the third plot, and action in the final plot.

figure('Name','Variable Trajectories','Color','w')

subplot(4,1,1)
plot(t,Y{1}), hold on
for i = 1:size(M{1},2)
    pY(:,i) = g(M{1}(:,i));
    pF(:,i) = [sig(sum(M{1}((1:2:(size(M{1},1)-4)),i))-2);(1-sig(sum(M{1}(1:2:(size(M{1},1)-4),i))))];
end
plot(t,pY,'--k'), hold off
title('Data (-) and predictions (--)')
axis tight

subplot(4,1,2)
plot(t,M{1}(1:end-4,:))
title('Beliefs (modes) about oscillators')
axis tight

subplot(4,1,3)
C = zeros(16,length(t),3);
C(:,:,1) = repmat(1-pF(1,:)/4,16,1,1);
C(:,:,2) = repmat(1-pF(2,:)/4 -pF(1,:)/4,16,1,1);
C(:,:,3) = repmat(1-pF(2,:)/4,16,1,1);
image([t(1) t(end)],[-2 2],C), hold on
plot(t,M{1}(end-4:end-1,:)), hold off
title('Beliefs about sequencing')

subplot(4,1,4)
plot(t,a)
title('Action')
axis tight

% Our second figure provides an animated version of the above designed to
% offer some intuition as to the simulated performance of the task and the
% belief mechanics that underwrite this.

figure('Name','Animation','Color','w')

[i,j] = meshgrid(-10:0.1:10,-10:0.1:10);
stim  = exp(-(i.^2 + j.^2)/4);

for k = 1:16:size(Y{1},2)
    subplot(3,2,1)
    imagesc(-stim*Y{1}(1,k)), colormap gray, clim([-max(Y{1}(1,:)) 0])
    axis square, axis off
    title('Stimulus')

    subplot(3,2,2)
    for i = 1:2:2*No
        plot([0 M{1}(i,k)],[0 M{1}(i+1,k)],'k'), hold on
        plot(M{1}(i,k),M{1}(i+1,k),'.k','MarkerSize',8)
    end
    hold off
    xlim([-max(max(M{1}(1:2*No,:))) max(max(M{1}(1:2*No,:)))])
    ylim([-max(max(M{1}(1:2*No,:))) max(max(M{1}(1:2*No,:)))])
    axis square
    axis off
    title('Internal clock')

    subplot(3,1,2)
    plot(tgt,[0 0],'.r','MarkerSize',32), hold on
    if k > 8
        for j = 8:-1:0
            plot(Y{1}(2,k-j),0,'.','Color',[1 1 1]*j/8,'MarkerSize',16)
        end
    end
    hold off
    xlim([-1 1] + tgt)
    axis off
    title('Behaviour')

    subplot(3,2,5)
    imagesc(1-M{1}(2*No+(1:2),k)'), colormap gray, axis off, clim([0 1])
    title('Current target')

    subplot(3,2,6)
    imagesc(1-smax(M{1}(2*No+(3:4),k))'), colormap gray, axis off, clim([0 1])
    title('Planned next target')

    drawnow

    % Animation
    %----------------------------------------------------------------------
    % F  = getframe(gcf);
    % im = frame2im(F);
    % [MM,MMM] = rgb2ind(im,256);
    % if k==1
    %     imwrite(MM,MMM,'\Animation.gif','gif','LoopCount',Inf,'DelayTime',0.1);
    % else
    %     imwrite(MM,MMM,'\Animation.gif','gif','WriteMode','append','DelayTime',0.1);
    % end
end