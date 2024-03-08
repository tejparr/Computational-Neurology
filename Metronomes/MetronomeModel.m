function [Y,M,t,a] = MetronomeModel(OPTIONS,P,y)
% Format [Y,M,t,a] = MetronomeModel(OPTIONS,P,y)
% This function specifies and inverts the generative model for our 
% metronome task, taking the parameters of our model and optionally data y
% and returning outstanding simulated data Y, our belief trajectory M,
% timesteps t, and action a. OPTIONS includes options for plotting.
%
% Thomas Parr
%--------------------------------------------------------------------------

% Preliminaries
%--------------------------------------------------------------------------
try OPTIONS.plot;                catch, OPTIONS.plot = 1; end  % Plotting
try OPTIONS.ani;                 catch, OPTIONS.ani  = 1; end  % Animate
try OPTIONS.save;                catch, OPTIONS.save = 0; end  % Save animation file  
try OPTIONS.int;                 catch, OPTIONS.int  = 0; end  % Integer multiples between frequencies
try Nt = OPTIONS.T;              catch, Nt = 4112;        end  % Number of timesteps
try phi   = P.phi;               catch, phi   = pi/2;     end  % Phase offset of stimuli and action
try gamma = exp(P.gamma);        catch, gamma = 4;        end  % Precision of policy selection
try alpha = exp(P.alpha);        catch, alpha = 4;        end  % Suppression of belief updating
try beta  = exp(P.beta);         catch, beta  = 1;        end  % Log scale parameter for dynamical precision
try thetaP = exp(P.thetaP);      catch, thetaP = 1;       end  % Log scale parameter for likelihood (proprio) precision
try thetaE = exp(P.thetaE);      catch, thetaE = 1;       end  % Log scale parameter for likelihood (extero) precision
try zeta  = 1/(1+exp(-P.zeta));  catch, zeta  = 0.8;      end  % Beliefs about persistence of occluder state
try sigma = exp(P.sigma);        catch, sigma = 1;        end  % Beliefs about smoothness

No   = 4;               % Number of clocks
dt   = 1/4;             % Length of timestep
tT   = (0:Nt)*dt;       % Time
T    = round(tT(end));  % Final time
ut   = 4;               % Scale temporal units to ms

%% Generative model
%-------------------------------------------------------------------------- 

% Equations of motion
%--------------------------------------------------------------------------

% Jacobian for pendular dynamics
%--------------------------------------------------------------------------
if OPTIONS.int % use of integer multiples to create orthogonal oscillator set

    for i = 1:2:2*No
        Jf(i:i+1,i:i+1) = [0  i;
                          -i  0]/32;
    end

else          % or use of oscillators of only small frequency difference relative to one-another

    for i = 1:2:2*No
        Jf(i:i+1,i:i+1) = [0  i;
                          -i  0]/64 + fliplr(diag([1 -1]))/16;
    end

end

% Transition dynamics
%--------------------------------------------------------------------------
B{1} = [0   1;              % Controllable factor (choice of attractor)
        1   0];  
B{2} = [1   1;              % Uncontrollable factor (presence or absence of occluder)
        0   0];

% Attractor locations
%--------------------------------------------------------------------------
tgt = [-1 1];

% Combine pendular dynamics with transition and attractor dynamics
%--------------------------------------------------------------------------

sig  = @(x) 1/(1+exp(-alpha*x));            % Sigmoid function

f = @(x) [Jf*x(1:size(Jf,2),1);                                                                  % oscillators as above
        sig(sum(x(1:2:size(Jf,2)))-2)*(smax(zeta*x(end-6:end-5,1))-x(end-8:end-7,1))/4;          % current state of occluder
        (1-sig(sum(x(1:2:size(Jf,2)))))*(log(B{2}*x(end-8:end-7,1)+exp(-2))-x(end-6:end-5,1))/4; % next state of occluder
        sig(sum(x(1:2:size(Jf,2)))-2)*(smax(gamma*x(end-2:end-1,1))-x(end-4:end-3,1))/4;         % current state
        (1-sig(sum(x(1:2:size(Jf,2)))))*(log(B{1}*x(end-4:end-3,1)+exp(-2))-x(end-2:end-1,1))/4; % next state
        (tgt*x(end-4:end-3,1) - x(end))/4];                                                      % attractor dynamics


% Prediction of data (i.e., mode of likelihood)
%--------------------------------------------------------------------------
g = @(x) [x(end-8)*exp(sum(x(1:2:end-9)*cos(phi) - x(2:2:end-8)*sin(phi)))/64; x(end)];

% Precisions and orders of motion
%--------------------------------------------------------------------------
Pg      = diag([exp(2)*thetaE exp(0)*thetaP]); % Precision of likelihood
Pf      = exp(4)*eye(2*No+9);                  % Precision of dynamics
Pf(end) = exp(0);                              % Less precise controllable dynamics
n       = 3;                                   % Order of generalised motion
s       = sigma;                               % Smoothness
m0      = randn(2*No + 9,1)/16;                % Initial beliefs
m0(end-4:end-1) = [1;0;-2;0];
m0(end-8:end-5) = [1;0;0;-2];

%% Generative process
%--------------------------------------------------------------------------
x0      = zeros(2*No + 9,1);     % Initial states (for generating data)
a0      = 0;                     % Initial active states

try   x0(2*OPTIONS.freq) = 4;    % Option to set pendulum frequency
catch, x0(4)   = 4; end          % Non-zero amplitude for second pendulum


% Equations of motion
%--------------------------------------------------------------------------
F = @(x,a) [Jf*x(1:size(Jf,2),1);                                                                    % oscillators as above
            sig(sum(x(1:2:size(Jf,2)))-2)*(smax(x(end-6:end-5,1))-x(end-8:end-7,1))/4;               % current state of occluder
            (1-sig(sum(x(1:2:size(Jf,2)))))*(log(B{2}*x(end-8:end-7,1)+exp(-2))-x(end-6:end-5,1))/4; % next state of occluder
            sig(sum(x(1:2:size(Jf,2)))-2)*(x(end-2:end-1,1)-x(end-4:end-3,1))/4;                     % current state
            (1-sig(sum(x(1:2:size(Jf,2)))))*(B{1}*x(end-4:end-3,1)-x(end-2:end-1,1))/4;              % next state
            a];                                                                                      % attractor dynamics

G = @(x) [exp(sum(x(1:2:end-9)))/64; x(end)];

%% Model inversion
%--------------------------------------------------------------------------

if exist('y','var')
    [Y,M,t,a] = ActiveFiltering(f,F,Pf*beta,g,G,Pg,x0,m0,n,s,dt,T,a0,y,2);
else
    [Y,M,t,a] = ActiveFiltering(f,F,Pf*beta,g,G,Pg,x0,m0,n,s,dt,T,a0);
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
pY = zeros(size(Y{1}));
pF = zeros(2,size(M{1},2));
subplot(4,1,1)
plot(t*ut,Y{1}), hold on
for i = 1:size(M{1},2)
    pY(:,i) = g(M{1}(:,i));
    pF(:,i) = [sig(sum(M{1}((1:2:(size(M{1},1)-4)),i))-2);(1-sig(sum(M{1}(1:2:(size(M{1},1)-4),i))))];
end
plot(t*ut,pY,'--k'), hold off
title('Data (-) and predictions (--)')
axis tight

subplot(4,1,2)
plot(t*ut,M{1}(1:end-4,:))
title('Beliefs (modes) about oscillators')
axis tight

subplot(4,1,3)
C = zeros(16,length(t),3);
C(:,:,1) = repmat(1-pF(1,:)/4,16,1,1);
C(:,:,2) = repmat(1-pF(2,:)/4 -pF(1,:)/4,16,1,1);
C(:,:,3) = repmat(1-pF(2,:)/4,16,1,1);
image([t(1) t(end)]*ut,[-2 2],C), hold on
plot(t*ut,M{1}(end-4:end-1,:)), hold off
title('Beliefs about sequencing')

subplot(4,1,4)
plot(t*ut,a)
title('Action')
axis tight

% Our second figure offers an interpretation of the belief-updating above
% as a time-frequency analysis using Fourier wavelet transforms of the sort
% sometimes used in electrophysiology research. This is supplemented with a
% power spectral density.

f = (4:64)/2;

figure('Name','Time-Frequency Analysis','Color','w')
S = zeros(length(f),size(M{1},2));
PSD = zeros(length(f),numel(M));
for i = 1:numel(M)
    s = fwt(gradient(M{i}),f,16,ut/1000);
    S = S + squeeze(sum(s,2));
    for j = 1:size(M{i},1)
        PSD(:,i) = PSD(:,i) + psd_ls(gradient(M{i}(j,:)),32,f,dt*ut/1000)';
    end
end

subplot(2,2,1)
plot(t*ut,Y{1}(2,:))
title('Kinematics')
xlabel('Time (ms)')
ylabel('Position')
axis tight

subplot(2,2,2)
imagesc((1:size(M{i},2))*dt*ut, f, abs(S)), axis xy
title('Time-Frequency')
xlabel('Time (ms)')
ylabel('Frequency (Hz)')

subplot(2,2,3)
plot(sum(gradient(M{1}),1)), axis tight
title('(unfiltered) LFPs')
xlabel('Time')
ylabel('Potential')

subplot(2,2,4)
plot(f,log(sum(PSD,2))), axis tight
title('Power Spectral Density')
xlabel('Frequency (Hz)')
ylabel('log power')

if OPTIONS.ani

% Our third figure provides an animated version of the above designed to
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

        subplot(6,2,9)
        imagesc(1-M{1}(2*No+(5:6),k)'), colormap gray, axis off, clim([0 1])
        title('Current target')

        subplot(6,2,11)
        imagesc(1-smax(M{1}(2*No+(7:8),k))'), colormap gray, axis off, clim([0 1])
        title('Planned next target')

        subplot(6,2,10)
        imagesc(1-M{1}(2*No+(1:2),k)'), colormap gray, axis off, clim([0 1])
        title('Current occlusion')

        subplot(6,2,12)
        imagesc(1-smax(M{1}(2*No+(3:4),k))'), colormap gray, axis off, clim([0 1])
        title('Next occlusion')

        drawnow

        % Animation
        % -----------------------------------------------------------------
        if OPTIONS.save
            F  = getframe(gcf);
            im = frame2im(F);
            [MM,MMM] = rgb2ind(im,256);
            if k==1
                imwrite(MM,MMM,'Graphics/Animation.gif','gif','LoopCount',Inf,'DelayTime',0.1);
            else
                imwrite(MM,MMM,'Graphics/Animation.gif','gif','WriteMode','append','DelayTime',0.1);
            end
        end
    end

end