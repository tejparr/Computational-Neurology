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
try OPTIONS.ani;                 catch, OPTIONS.ani  = 0; end  % Animate
try OPTIONS.save;                catch, OPTIONS.save = 0; end  % Save animation file  
try OPTIONS.int;                 catch, OPTIONS.int  = 0; end  % Integer multiples between frequencies
try OPTIONS.lc;                  catch, OPTIONS.lc   = 1; end  % Limit cycles
try freq = OPTIONS.freq;         catch, freq = 5;         end  % Frequency for generative process
try Nt = OPTIONS.T;              catch, Nt = 2048;        end  % Number of timesteps
try phi   = P.phi;               catch, phi   = pi/4;     end  % Phase offset of stimuli and action
try gamma = exp(P.gamma);        catch, gamma = 2;        end  % Precision of policy selection
try alpha = exp(P.alpha);        catch, alpha = 4;        end  % Suppression of belief updating
try beta  = exp(P.beta);         catch, beta  = 1;        end  % Log scale parameter for dynamical precision
try thetaP = exp(P.thetaP);      catch, thetaP = 1;       end  % Log scale parameter for likelihood (proprio) precision
try thetaE = exp(P.thetaE);      catch, thetaE = 4;       end  % Log scale parameter for likelihood (extero) precision
try zeta  =  exp(P.zeta);        catch, zeta  =  0.5;     end  % Beliefs about persistence of occluder state
try sigma = exp(P.sigma);        catch, sigma = 1;        end  % Beliefs about smoothness
try delta = exp(P.delta);        catch, delta = 20;       end  % Base rate for oscillators (assuming non-integers)

No   = 4;               % Number of clocks
dt   = 1/4;             % Length of timestep
tT   = (0:Nt)*dt;       % Time
T    = round(tT(end));  % Final time
ut   = 4;               % Scale temporal units to ms

if OPTIONS.lc
    try rho   = exp(P.rho);           catch, rho   = 4;        end  % Radius if limit cycles used
    try xi    = exp(P.xi)*ut/1000;    catch, xi    = 1/128;    end  % Attraction to limit if limit cycle (default is about 2 a.u./s)
end

%% Generative model
%-------------------------------------------------------------------------- 

% Equations of motion
%--------------------------------------------------------------------------

% Jacobian for pendular dynamics
%--------------------------------------------------------------------------
if OPTIONS.int      % use of integer multiples to create orthogonal oscillator set

    for i = 1:2:2*No
        Jf(i:i+1,i:i+1) = [0  i;
                          -i  0]*(ut*5)/1000;
    end

else                % or use of oscillators of only small frequency difference relative to one-another

    for i = 1:2:2*No
        Jf(i:i+1,i:i+1) = [0  i;
                          -i  0]*(ut*5)/1000 + fliplr(diag([1 -1]))*delta*ut/1000;
    end

end

% Transition dynamics
%--------------------------------------------------------------------------
B{1} = [0   1   1;       % Controllable factor (choice of attractor)
        1   0   0;
        0   0   0];  
B{2} = [1   0;
        0   1];

% Attractor locations
%--------------------------------------------------------------------------
tgt = [-1 1];
sig = @(x) 1/(1+exp(-alpha*x));            % Sigmoid function

if OPTIONS.lc
% This introduces mutual constraints between the oscillators by inducing
% limit cycles in which the combined radius of all oscillators lie on a
% limit cycle of radius rho. This offers the opportunity to do three things.
% First, by mutually constraining the oscillators, a subtle winner-take-all
% effect is introduced. Second, this favours there being an oscillator even
% in the absence of any external stimuli. Third, we can use the magnitude
% of the radius as a way to modulate the sequencing (similar to the alpha
% parameter).

f = @(x) [(rho - norm(x(1:2*No))).*x(1:2*No)*xi + Jf*x(1:2*No);                                     % oscillators with limit cycles at radius rho
        sig(sum(x(1:2:2*No)))*(zeta*x(2*No + (3:4))-x(2*No + (1:2)))/16;                            % current state of occluder
        (1-sig(sum(x(1:2:2*No))+1))*(exp(-x(2*No + (3:4))).*(B{2}*cn_smax(x(2*No + (1:2)))) - 1)/4; % next state of occluder
        sig(sum(x(1:2:2*No)))*(gamma*x(2*No + (8:10))-x(2*No + (5:7)))/4;                           % current state
        (1-sig(sum(x(1:2:2*No))+1))*(exp(-x(2*No + (8:10))).*(B{1}*cn_smax(x(2*No + (5:7)))) - 1)/4;% next state
        ([tgt 0]*cn_smax(x(2*No + (5:7))) - x(end))/4];                                             % attractor dynamics

else


% Combine pendular dynamics with transition and attractor dynamics
%--------------------------------------------------------------------------

f = @(x) [Jf*x(1:2*No);                                                                             % oscillators as above
        sig(sum(x(1:2:2*No)))*(zeta*x(2*No + (3:4))-x(2*No + (1:2)))/16;                            % current state of occluder
        (1-sig(sum(x(1:2:2*No))+1))*(exp(-x(2*No + (3:4))).*(B{2}*cn_smax(x(2*No + (1:2)))) - 1)/4; % next state of occluder
        sig(sum(x(1:2:2*No)))*(gamma*x(2*No + (8:10))-x(2*No + (5:7)))/4;                           % current state
        (1-sig(sum(x(1:2:2*No))+1))*(exp(-x(2*No + (8:10))).*(B{1}*cn_smax(x(2*No + (5:7)))) - 1)/4;% next state
        ([tgt 0]*cn_smax(x(2*No + (5:7))) - x(end))/4];                                             % attractor dynamics

end

% Prediction of data (i.e., mode of likelihood)
%--------------------------------------------------------------------------

g = @(x) [cn_smax(x(2*No+(1:2)))'*[exp(sum(x((1:No)*2 - 1)*cos(phi) - x((1:No)*2)*sin(phi)))/64;0]; x(end)];

% Precisions and orders of motion
%--------------------------------------------------------------------------
Pg      = diag([thetaE thetaP]); % Precision of likelihood
Pf      = diag([exp(4)*ones(1,2*No) exp(4)*ones(1,4) exp(4)*ones(1,6) 1]);
n       = 3;                     % Order of generalised motion
s       = sigma;                 % Smoothness   

% Initial beliefs
%--------------------------------------------------------------------------
m0      = zeros(2*No + 11,1); 
m0(2*No + (1:10)) = -16;
m0(2*No + (1:4))  = log(1/2);
m0(2*No + 7)      = log(1);
m0(2*No + (8:10)) = log(1/3);

%% Generative process
%--------------------------------------------------------------------------
x0      = zeros(3,1);             % Initial states (for generating data)
x0(2)   = 4;                      % Determine amplitude
a0      = 0;                      % Initial active state

% Equations of motion
%--------------------------------------------------------------------------
F = @(x,a) [x(2)*freq/(16*pi);
           -x(1)*freq/(16*pi);
            a];                   % attractor dynamics

G = @(x) [exp(x(1))/64; x(end)];

%% Model inversion
%--------------------------------------------------------------------------

if exist('y','var')
    [Y,M,t,a] = ActiveFiltering(f,F,Pf*beta,g,G,Pg,x0,m0,n,s,dt,T,a0,y,2);
else
    [Y,M,t,a] = ActiveFiltering(f,F,Pf*beta,g,G,Pg,x0,m0,n,s,dt,T,a0);
end

t = t*ut;

if OPTIONS.plot == 0, return, end

% Plotting
%--------------------------------------------------------------------------
% The first plot gives a summary of the behaviour of the variables in the
% model. These include the data presented to our agent and its predictions
% of these data (dashed) in the upper plot, beliefs about the trajectory of
% (imaginary) oscillators in the second plot, beliefs involved in sequencing
% action in the third plot, and action in the final plot.

figure('Name','Variable Trajectories','Color','w')
pY = zeros(size(Y{1}));
pF = zeros(2,size(M{1},2));

subplot(6,1,1)
plot(t,Y{1}), hold on
for i = 1:size(M{1},2)
    pY(:,i) = g(M{1}(:,i));
    pF(:,i) = [sig(sum(M{1}((1:No)*2 - 1,i)));(1-sig(sum(M{1}((1:No)*2 - 1,i))+1))];
end
plot(t,pY,'--k'), hold off
title('Data and predictions')
axis tight
box off
ax = gca;
ax.XAxisLocation = "origin";
ax.TickLength = [0 0];
xlabel('Time (ms)')

subplot(6,1,2)
plot(t,M{1}(1:2*No,:))
title('Beliefs (modes) about oscillators')
axis tight
box off
ax = gca;
ax.XAxisLocation = "origin";
ax.TickLength = [0 0];
xlabel('Time (ms)')

subplot(6,1,3)
C = zeros(16,length(t),3);
C(:,:,1) = repmat(1-pF(1,:)/4,16,1,1);
C(:,:,2) = repmat(1-pF(2,:)/4 -pF(1,:)/4,16,1,1);
C(:,:,3) = repmat(1-pF(2,:)/4,16,1,1);
image([t(1) t(end)],[0 1],C), hold on
plot(t,pF)
title('Beliefs about sequencing')
axis xy
box off
ax = gca;
ax.XAxisLocation = "origin";
ax.TickLength = [0 0];
xlabel('Time (ms)')

subplot(6,1,4:5)
K = 1-[cn_smax(M{1}(2*No+(1:2),:)); cn_smax(M{1}(2*No+(3:4),:)); cn_smax(M{1}(2*No+(5:7),:)); cn_smax(M{1}(2*No+(8:10),:))];
imagesc([t(1) t(end)],[1 10],K), clim([0 1]), colormap gray
title('Targets and occluders')
xlabel('Time (ms)')
ax = gca;
ax.TickLength = [0 0];

subplot(6,1,6)
plot(t,a)
title('Action')
axis tight
box off
ax = gca;
ax.XAxisLocation = "origin";
ax.TickLength = [0 0];
xlabel('Time (ms)')

% Our second figure offers an interpretation of the belief-updating above
% as a time-frequency analysis using Fourier wavelet transforms of the sort
% sometimes used in electrophysiology research. This is supplemented with a
% power spectral density.

f = (4:256)/8;

figure('Name','Time-Frequency Analysis','Color','w')
S = zeros(length(f),size(M{1},2));
PSD = zeros(length(f),numel(M));
for i = 1:numel(M)
    s = cn_fwt(sum(gradient(M{i}),1),f,32,ut/1000);
    S = S + squeeze(s);
    for j = 1:size(M{i},1)
        PSD(:,i) = PSD(:,i) + cn_psd_ls(M{i}(j,:),16,f,dt*ut/1000)';
    end
end

subplot(2,2,1)
plot(t,Y{1}(2,:))
title('Kinematics')
xlabel('Time (ms)')
ylabel('Position')
axis tight
box off
ax = gca;
ax.XAxisLocation = "origin";
ax.TickLength = [0 0];
xlabel('Time (ms)')

subplot(2,2,2)
imagesc((1:size(M{i},2))*dt*ut, f, abs(S)), axis xy, colormap gray, clim([0 15])
title('Time-Frequency')
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
hold on
plot((1:size(M{i},2))*dt*ut,8*mean(abs(S(logical((f>13).*(f<30)),:)),1),'--w') % Plot beta power
ax = gca;
ax.TickLength = [0 0];

subplot(2,2,3)
plot(sum(gradient(M{1}),1)), axis tight
title('(unfiltered) LFPs')
xlabel('Time')
ylabel('Potential')
axis tight
box off
ax = gca;
ax.XAxisLocation = "origin";
ax.TickLength = [0 0];
xlabel('Time (ms)')

subplot(2,2,4)
plot(f,log(sum(PSD,2))), axis tight
title('Power Spectral Density')
xlabel('Frequency (Hz)')
ylabel('log power')
axis tight
box off
ax = gca;
ax.XAxisLocation = "origin";
ax.TickLength = [0 0];

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
        imagesc(1-cn_smax(M{1}(2*No+(5:6),k))'), colormap gray, axis off, clim([0 1])
        title('Current target')

        subplot(6,2,11)
        imagesc(1-cn_smax(M{1}(2*No+(8:9),k))'), colormap gray, axis off, clim([0 1])
        title('Planned next target')

        subplot(6,2,10)
        imagesc(1-cn_smax(M{1}(2*No+(1:2),k))'), colormap gray, axis off, clim([0 1])
        title('Current occlusion')

        subplot(6,2,12)
        imagesc(1-cn_smax(M{1}(2*No+(3:4),k))'), colormap gray, axis off, clim([0 1])
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


