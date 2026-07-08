function POMDP = DEMO_RPM
% This demo illustrates a relatively simple implementation of a generative
% model capable of solving an example of a Raven's Progressive Matrix.
% The matrix here is constructed from randomly generated rules based upon
% two features. These are the shape (here, triangles of different
% orientation) and the scale (i.e., the size of the triangle). The example
% here involves making a series of fixations to determine the rules by
% which subsequent elements are generated, during which the unseen lower
% right cell can be predicted with progressively increasing confidence.
%==========================================================================

close all

% Rules (assuming no rules converge on a single state)
%--------------------------------------------------------------------------
ns                  = 4; % number of possible stimuli in each direction
r                   = perms(1:ns);
r(r(:,1)~=1,:)      = [];
nr                  = size(r,1);
R                   = cell(nr,1);
for i = 1:nr
    R{i} = zeros(ns,ns);
    R{i}(sub2ind([ns,ns],r(i,:),circshift(r(i,:),1))) = 1;
end

% Initial state priors
%--------------------------------------------------------------------------
D{1} = [1;0;0];        % x-fixation
D{2} = D{1};           % y-fixation
D{3} = ones(ns,1)/ns;  % relevant feature x - UL
D{4} = D{3};           % relevant feature y - UL
D{5} = ones(2,1)/2;    % identity of relevant feature x (shape/scale)
D{6} = ones(nr,1)/nr;  % rules x
D{7} = D{6};           % rules y

% Likelihood distributions
%--------------------------------------------------------------------------

% Location outcomes
for f1 = 1:numel(D{1})
    for f2 = 1:numel(D{2})
        A{1}((f1-1)*numel(D{2})+f2,f1,f2) = 1;
    end
end

for f3 = 1:numel(D{3})
    for f4 = 1:numel(D{4})
        for f5 = 1:numel(D{5})
            if f5 == 1
                A{2}(f3,f3,f4,f5) = 1; % shape
                A{3}(f4,f3,f4,f5) = 1; % scale
            else
                A{2}(f4,f3,f4,f5) = 1;
                A{3}(f3,f3,f4,f5) = 1;
            end
        end
    end
end

% Transition probabilities
%--------------------------------------------------------------------------
B{1} = zeros(numel(D{1}),numel(D{1}),numel(D{1}));
B{2} = zeros(numel(D{2}),numel(D{2}),numel(D{2}));

for k = 1:numel(D{1})
    B{1}(k,:,k) = 1;
    B{2}(k,:,k) = 1;
end

B{3} = zeros(numel(D{3}),numel(D{3}),numel(D{1}),numel(D{6}),numel(D{1}));
B{4} = zeros(numel(D{4}),numel(D{4}),numel(D{2}),numel(D{7}),numel(D{2}));

for fx = 1:numel(D{1})
    for fr = 1:numel(D{6})
        for k = 1:numel(D{1})
            n = k-fx;
            B{3}(:,:,fx,fr,k) = R{fr}^n;
            B{4}(:,:,fx,fr,k) = R{fr}^n;
        end
    end
end

for f = 5:7
    B{f} = eye(numel(D{f}));
end

% Preferences
%--------------------------------------------------------------------------
C    = cell(numel(A),1);
C{1} = zeros(9,1); C{1}(end) = -6; % penalise looking at final square
for g = 2:numel(A)
    C{g} = zeros(size(A{g},1),1);
end

% Policies
%--------------------------------------------------------------------------
E{1} = ones(3,1)/3;
E{2} = ones(3,1)/3;

% Conditional dependencies
%--------------------------------------------------------------------------
dom.A(1).s = [1 2];
dom.A(1).u = [];
dom.A(2).s = [3 4 5];
dom.A(2).u = [];
dom.A(3).s = [3 4 5];
dom.A(3).u = [];
dom.B(1).s = [];
dom.B(1).u = 1;
dom.B(2).s = [];
dom.B(2).u = 2;
dom.B(3).s = [1 6];
dom.B(3).u = 1;
dom.B(4).s = [2 7];
dom.B(4).u = 2;

for i = 5:numel(B)
    dom.B(i).s = [];
    dom.B(i).u = [];
end

% Initialise
%--------------------------------------------------------------------------
s = zeros(numel(D),1);
for f = 1:length(s)
    s(f) = find(rand<cumsum(D{f}),1,'first');
end

% Compile POMDP
%--------------------------------------------------------------------------
pomdp.A = A;
pomdp.B = B;
pomdp.C = C;
pomdp.D = D;
pomdp.E = E;
pomdp.T = 2;
pomdp.s = s;
pomdp.dom = dom;
pomdp.gen = @pomdp_default_gen;
pomdp.smooth = 1;

% solve POMDP (using two-step smoothing)
%--------------------------------------------------------------------------
Ns = 4; % Number of saccades
S  = zeros(size(s,1),Ns);
Q  = cell(Ns,size(s,1));
O  = zeros(numel(A),Ns);
for t = 1:Ns+1
    POMDP    = mp_POMDP(pomdp);
    S(:,t:t+1) = POMDP.s;
    O(:,t:t+1) = POMDP.o;
    Q(t:t+1,:) = POMDP.Q;
    pomdp.D  = POMDP.BS.s(:,end);
    pomdp.s  = POMDP.s(:,end);
end

POMDP.Q = Q;
POMDP.o = O;
POMDP.s = S;

RPM_animation(POMDP)
mp_pomdp_belief_plot(POMDP)


function RPM_animation(POMDP)
% The following constructs an animated visualisation of a single trial.
%--------------------------------------------------------------------------

cn_figure('RPM animation')

% Extract states and observations of generative process
%--------------------------------------------------------------------------
s = POMDP.s;
o = POMDP.o;

% Construct RPM
%--------------------------------------------------------------------------
shapes = zeros(3,3);
scales = zeros(3,3);

if s(5,1) == 1
    shapes(1,1) = s(3,1);
    scales(1,1) = s(4,1);
else
    shapes(1,1) = s(4,1);
    scales(1,1) = s(3,1);
end

for i = 1:3
    for j = 1:3
        [~,ind1] = max(POMDP.B{3}(:,s(3),1,s(6),j));
        [~,ind2] = max(POMDP.B{4}(:,s(4),1,s(7),j));
        if s(5,1) == 1
            shapes(j,i) = ind1;
            scales(i,j) = ind2;
        else
            shapes(i,j) = ind2;
            scales(j,i) = ind1;
        end
    end
end

sh = {'<','^','>','v'};
sc = [4 8 12 16];

% Interpolate states for smooth eye movements
%--------------------------------------------------------------------------
X(1,:)  = interp1(0:size(s,2)*4-1,kron(s(1,:),ones(1,4)),1/8:1/8:size(s,2)*4);
X(2,:)  = interp1(0:size(s,2)*4-1,kron(s(2,:),ones(1,4)),1/8:1/8:size(s,2)*4);
SL      = size(X,2)/size(o,2);
O       = kron(o,ones(1,SL));

% Initialise graphics objects
%--------------------------------------------------------------------------
subplot(3,2,4)
H    = gobjects(length(POMDP.Q{1,3}),length(POMDP.Q{1,4}));
for i = 1:size(H,1)
    for j = 1:size(H,2)
        H(i,j) = scatter(0,0,sc(j)^2,[sh{i} 'k'],'MarkerEdgeAlpha',1); hold on
    end
end
axis square, axis off
title('Prediction')

subplot(3,1,1)
h    = gobjects(1,1);
h(1) = plot(X(1,1),X(2,1),'or','MarkerSize',32); hold on

% Plot RPM
%--------------------------------------------------------------------------
for i = 1:3
    for j = 1:3
        if i~=3 || j ~=3
            plot(i,j,[sh{shapes(i,j)} 'k'],'MarkerSize',sc(scales(i,j)))
            hold on
        end
        if j~=1
            plot([1 3],[j j]-0.5,'k')
        end
        if i~=1
            plot([i i]-0.5,[1 3],'k')
        end
    end
end
axis square, axis ij, axis off

% Prepare foveal graphics, including motion suppression
%--------------------------------------------------------------------------
M = sin(linspace(0,pi,16))'*sin(linspace(0,pi,16));
M = M/max(M(:));
N = zeros(size(M));

V = sum(gradient(X).^2,1);
V = V/max(V);
V = 1./(exp(-16*V)+1);

% Loop through frame-by-frame
%--------------------------------------------------------------------------
for i = 1:size(X,2)
    subplot(3,2,3)
    imagesc(M*(1-V(i))+N*V(i)), colormap gray, hold on, clim([0,1])
    plot(8.5,8.5,[sh{O(2,i)} 'k'],'MarkerSize',sc(O(3,i))), hold off
    axis square, axis off
    title('Fovea')

    P1  = mp_dot(squeeze(POMDP.B{3}(:,:,:,:,3)),{POMDP.Q{min(size(POMDP.Q,1),ceil(i/SL)),3},POMDP.Q{min(size(POMDP.Q,1),ceil(i/SL)),1},POMDP.Q{min(size(POMDP.Q,1),ceil(i/SL)),6}});
    P2  = mp_dot(squeeze(POMDP.B{4}(:,:,:,:,3)),{POMDP.Q{min(size(POMDP.Q,1),ceil(i/SL)),4},POMDP.Q{min(size(POMDP.Q,1),ceil(i/SL)),2},POMDP.Q{min(size(POMDP.Q,1),ceil(i/SL)),7}});
    P3  = POMDP.Q{min(size(POMDP.Q,1),ceil(i/SL)),5};
    for h1 = 1:size(H,1)
        for h2 = 1:size(H,2)
            H(h1,h2).MarkerEdgeAlpha = P1(h1)*P2(h2)*P3(1) + P1(h2)*P2(h1)*P3(2);
        end
    end
    
    subplot(3,2,5)
    imagesc(mp_dot(POMDP.B{3}(:,:,1,:,2),POMDP.Q(min(size(POMDP.Q,1),ceil(i/SL)),6))), colormap gray, clim([0,1]), axis square
    title('Horizontal rule')

    subplot(3,2,6)
    imagesc(mp_dot(POMDP.B{4}(:,:,1,:,2),POMDP.Q(min(size(POMDP.Q,1),ceil(i/SL)),7))), colormap gray, clim([0,1]), axis square
    title('Vertical rule')
    h.XData = X(1,i);
    h.YData = X(2,i);
    drawnow
    
    %======================================================================
    % Uncomment the below to save animation
    % cn_animation(i,1,1/8,'Graphics','RPM_Animation')
    %======================================================================
end

% Create static plot for figure
%--------------------------------------------------------------------------
cn_figure('RPM simulation plots')

subplot(3,1,1)
for i = 1:3
    for j = 1:3
        if i~=3 || j ~=3
            plot(i,j,[sh{shapes(i,j)} 'k'],'MarkerSize',sc(scales(i,j)))
            hold on
        end
        if j~=1
            plot([1 3],[j j]-0.5,'k')
        end
        if i~=1
            plot([i i]-0.5,[1 3],'k')
        end
    end
end
X(:,isnan(X(1,:))) = [];
plot(smoothdata(X(1,:)+randn(1,size(X,2))/16),smoothdata(X(2,:)+randn(1,size(X,2))/16),'r')
axis square, axis ij, axis off

for i = 1:4
    subplot(3,4,4+i)
    imagesc(M), colormap gray, hold on, clim([0,1])
    plot(8.5,8.5,[sh{o(2,i)} 'k'],'MarkerSize',sc(o(3,i))), hold off
    axis square, axis off

    subplot(3,4,8+i)
    P1  = mp_dot(squeeze(POMDP.B{3}(:,:,:,:,3)),{POMDP.Q{min(size(POMDP.Q,1),i),3},POMDP.Q{min(size(POMDP.Q,1),i),1},POMDP.Q{min(size(POMDP.Q,1),i),6}});
    P2  = mp_dot(squeeze(POMDP.B{4}(:,:,:,:,3)),{POMDP.Q{min(size(POMDP.Q,1),i),4},POMDP.Q{min(size(POMDP.Q,1),i),2},POMDP.Q{min(size(POMDP.Q,1),i),7}});
    P3  = POMDP.Q{min(size(POMDP.Q,1),i),5};
    for h1 = 1:size(H,1)
        for h2 = 1:size(H,2)
            scatter(0,0,sc(h2)^2,[sh{h1} 'k'],'MarkerEdgeAlpha',P1(h1)*P2(h2)*P3(1) + P1(h2)*P2(h1)*P3(2)); hold on
        end
    end
    axis square, axis off
end