function POMDP = DEMO_Entangled
% This demo illustrates the use of conditional transition matrices combined
% with inductive inference such that one can behave in a world in which our
% effectors (e.g., limbs) directly interact with objects.
%--------------------------------------------------------------------------

% Preliminaries
%--------------------------------------------------------------------------
close all
cd(fileparts(mfilename('fullpath')))
OPTIONS.save = 0;           % Option to save a GIF animation
OPTIONS.frames = 1;         % Option to create summary figure of frames
OPTIONS.trails = 0;         % Plot trails
rng default                 % For reproducibility

% Options for simulation
%--------------------------------------------------------------------------
N = 5;                      % Dimension of arena
T = 10;                     % Number of steps for simulation

% Key variables for generative model and process
%--------------------------------------------------------------------------
[x, y] = meshgrid(1:N,1:N); % Arena layout
ix     = [x(:) y(:)];       % Indices

% Generative model
%==========================================================================
% Initial state priors
%--------------------------------------------------------------------------
D{1} = ones(N^2,1)/N^2;           % Position of an effector (e.g., limb)
D{2} = ones(N^2+1,1)/(N^2+1);     % Position of object to be moved (final state is out of arena)

% Likelihood tensors
%--------------------------------------------------------------------------
A{1} = eye(numel(D{1}));          % Direct visual report of effector
A{2} = eye(numel(D{2}));          % Direct visual report of object

% Preferences
%--------------------------------------------------------------------------
C{1} = zeros(size(A{1},1),1);     % No relevant preferences
C{2} = zeros(size(A{2},1),1);     % No relevant preferences

% Transition probabilities
%--------------------------------------------------------------------------
u      = [0  0; % Possible actions (row vectors)
         -1  0;
          0 -1;
          1  0;
          0  1];

B{1}   = zeros(N^2,N^2,5);
B{2}   = zeros(N^2+1,N^2+1,N^2,5);

for k = 1:size(u,1)
    for f1 = 1:numel(D{1})
        g1 = find(ismember(ix, ix(f1,:) + u(k,:), 'rows'));
        if isempty(g1), g1 = f1; end
        B{1}(g1,f1,k) = 1;
        for f2 = 1:numel(D{2})
            if g1 == f2 % If effector will occupy previous location of object
                h1 = find(ismember(ix, ix(f2,:) + u(k,:), 'rows'));
                if isempty(h1), h1 = N^2+1; end
                B{2}(h1,f2,f1,k) = 1;
            else
                B{2}(f2,f2,f1,k) = 1;
            end
        end
    end
end

% Priors for actions
%--------------------------------------------------------------------------
E{1} = ones(size(u,1))/size(u,1);

% Mandatory target states
%--------------------------------------------------------------------------
H{1} = [];
H{2} = 14;

% Conditional dependencies (i.e., domains of conditional distributions)
%--------------------------------------------------------------------------
dom.A(1).s = 1;
dom.A(1).u = [];
dom.A(2).s = 2;
dom.A(2).u = [];
dom.B(1).s = [];
dom.B(1).u = 1;
dom.B(2).s = 1;
dom.B(2).u = 1;

% Compile POMDP structure
%--------------------------------------------------------------------------
pomdp.A = A;
pomdp.B = B;
pomdp.C = C;
pomdp.D = D;
pomdp.E = E;
pomdp.T = T;
pomdp.N = 1;
pomdp.dom = dom;
pomdp.s = [5 7]';
pomdp.gen = @mdp_entangled_gen;
pomdp.H = H;

% Solve POMDP model
%--------------------------------------------------------------------------
POMDP = mp_POMDP(pomdp);

% Plotting and animations
%--------------------------------------------------------------------------
mp_pomdp_belief_plot(POMDP);

cn_figure('Paths')
subplot(3,1,1)
imagesc(1-[POMDP.P{:}]), axis tight, colormap gray
title('Prior probabilities for paths')

mdp_entangled_animation(POMDP,x,y,OPTIONS);             % Animation
mdp_induction_demo(pomdp.H,B,{POMDP.Q(1,:)},dom.B)      % Illustration of induction

% Assess metrics based upon mutual information
%--------------------------------------------------------------------------
MI = zeros(N^2,N^2);
for i = 1:N^2
    for j = 1:N^2
        cB = mp_tensor_con(B{1}(:,i,:),B{2}(:,j,i,:),[1,4],[1 2 3],[4 5 2 3]);
        cB = cB/(sum(cB(:)));
        MI(i,j) = sum(cB.*(mp_log(cB) - mp_log(sum(cB,2)*sum(cB,1))),'all');
    end
end
L     = diag(sum(MI>0)) - (MI>0);
[U,V] = eig(L);

cn_figure('Dimensionality analysis')
subplot(2,2,1)
imagesc(MI), axis square, colormap gray
title('Mutual information')

subplot(2,2,2)
imagesc(L), axis square, colormap gray
title('Graph Laplacian')

subplot(2,2,3)
bar(diag(V),'EdgeColor','none','FaceColor',[0.4 0.6 0.8])
box off
title('Eigenvalues of Laplacian')

subplot(2,2,4)
plot(U(:,2),U(:,3),'.','Color',[0.4 0.4 0.4],'MarkerSize',16), hold on
for i = 1:size(U,1)
    for j = i+1:size(U,1)
        if MI(i,j)>0
            plot([U(i,2), U(j,2)], [U(i,3), U(j,3)],'LineWidth',1,'Color',[0.5,0.5,0.5])
        end
    end
end
axis equal, box off
title('Effective geometry (degenerate Fiedler vectors)')


return

% Simulate multiple examples from randomly generated start positions
%==========================================================================

% Generate solvable initial configurations
%--------------------------------------------------------------------------
U = zeros(3,16); %#ok
for i = 1:size(U,2)
    r = 1+randi(3,[1,2]);
    U(2,i) = sub2ind([N,N],r(1),r(2)); % Initial blue cube location
    S      = setdiff(1:N^2,U(1,i));
    U(1,i) = S(randi(length(S)));      % Initial red cube location
    S      = setdiff(S,U(2,i));
    U(3,i) = S(randi(length(S)));      % Target location
end

pomdp.T = 64;
pomdp.randact = [];                    % Sample actions rather than MAP

% Solve without full induction scheme (naive induction)
%--------------------------------------------------------------------------
pomdp.dom.B(2).mf = [];                % Add flag to eliminate entangled induction
NI = zeros(2,size(U,2));               % Initialise success/steps
NIPOMDP = [];
for i = 1:size(U,2)
    pomdp.s    = U(1:2,i);
    pomdp.H{2} = U(3,i);
    POMDP      = mp_POMDP(pomdp);
    NI(1,i) = POMDP.s(2,end)==U(3,i);
    if NI(1,i)
        NI(2,i) = find(POMDP.s(2,:)==U(3,i),1,'first');
        if isempty(NIPOMDP)
            NIPOMDP = POMDP;
            NIPOMDP.i = i;
        end
    else
        NI(2,i) = pomdp.T;
    end
end

% Solve with full induction scheme (entangled induction)
%--------------------------------------------------------------------------
pomdp.dom.B = rmfield(pomdp.dom.B,'mf');
EI = zeros(2,size(U,2));               % Initialise success/steps
EIPOMDP = [];
for i = 1:size(U,2)
    pomdp.s    = U(1:2,i);
    pomdp.H{2} = U(3,i);
    POMDP      = mp_POMDP(pomdp);
    EI(1,i) = POMDP.s(2,end)==U(3,i);
    if EI(1,i)
        EI(2,i) = find(POMDP.s(2,:)==U(3,i),1,'first');
    else
        EI(2,i) = pomdp.T;
    end
    if i==NIPOMDP.i
        EIPOMDP = POMDP;
    end
end
                     
cn_figure('Effect of entangled induction')
subplot(3,1,1)
plot(NI(2,:),EI(2,:),'.'), hold on
plot(NI(2,NIPOMDP.i),EI(2,NIPOMDP.i),'or')
plot([0 pomdp.T],[0 pomdp.T],'--k')
axis([0 pomdp.T,0 pomdp.T])
axis square
box off
xlabel('Naive induction')
ylabel('Entangled induction')

OPTIONS.trails = true;
NIPOMDP.o(:,NI(2,NIPOMDP.i)+1:end) = [];
mdp_entangled_animation(NIPOMDP,x,y,OPTIONS);             % Animation
EIPOMDP.o(:,EI(2,NIPOMDP.i)+1:end) = [];
mdp_entangled_animation(EIPOMDP,x,y,OPTIONS);             % Animation

function [o,s] = mdp_entangled_gen(s,u,~,pomdp)
% Generative process that plays the role of a simulated environment. Here,
% the generative process matches the transition structure of the generative
% model.
%--------------------------------------------------------------------------

if nargout > 1 % Advance the states
    i1 = [{':'},num2cell(s(1)),num2cell(s(pomdp.dom.B(1).s)),num2cell(u(pomdp.dom.B(1).u))];
    s1 = pomdp.B{1}(i1{:});
    i2 = [{':'},num2cell(s(2)),num2cell(s(pomdp.dom.B(2).s)),num2cell(u(pomdp.dom.B(2).u))];
    s2 = pomdp.B{2}(i2{:});
    s(1) = find(cumsum(s1)>rand,1,'first');
    s(2) = find(cumsum(s2)>rand,1,'first');
end

o = s; % Generate outcomes

function mdp_entangled_animation(pomdp,x,y,OPT)
% Function to generative animation to illustrate the behaviour of the
% simulation.
%--------------------------------------------------------------------------

cn_figure('Animation')          % Create new figure
o = pomdp.o;                    % Extract the outcomes to be plotted

% Checkerboard pattern for floor
%--------------------------------------------------------------------------
dx = abs(x(1,2)-x(1,1));        % Size of squares
cb = mod(x+y,2);                % Checkerboard
[xe,ye] = meshgrid(...,         % Grid for edges
    min(x(:))-dx/2:max(x(:))+dx/2, ...
    min(y(:))-dx/2:max(y(:))+dx/2);

surf(...                        % Surface plot of board
    xe, ye, zeros(size(xe)), cb*0.8 + 0.2, ...
    'EdgeColor','none')

colormap(gray)                  % Set colours to grayscale
clim([0 1])                     % Colour limits

% Set axis properties
%--------------------------------------------------------------------------
axis equal, hold on
view(45,30), axis off
set(gcf,'Color','k')

% Highlight H'th square
%--------------------------------------------------------------------------
Hidx = pomdp.H{2};
hx = x(Hidx);
hy = y(Hidx);

rectangle( ...
    'Position',[hx-dx/2 hy-dx/2 dx dx], ...
    'EdgeColor','b', ...
    'LineWidth',5)

% Create cubes
%--------------------------------------------------------------------------
C1 = mdp_cube([x(o(1,1)) y(o(1,1))], dx, 'r');
C2 = mdp_cube([x(o(2,1)) y(o(2,1))], dx, 'b');

% Optional trajectory trails
%--------------------------------------------------------------------------
if OPT.trails

    % Store trail coordinates
    trail1x = x(o(1,1))-0.1;
    trail1y = y(o(1,1))-0.1;
    trail2x = x(o(2,1))+0.1;
    trail2y = y(o(2,1))+0.1;

    % Plot handles
    T1 = plot3( ...
        trail1x, trail1y, 0.02*ones(size(trail1x)), ...
        'r-', 'LineWidth',4);

    T2 = plot3( ...
        trail2x, trail2y, 0.02*ones(size(trail2x)), ...
        'b-', 'LineWidth',4);
end

% Shadows
%--------------------------------------------------------------------------
LD = [1 1 2];

V1 = mdp_shadow(C1, LD);
F1 = C1.Faces;

S1 = patch( ...
    'Vertices',V1, ...
    'Faces',F1, ...
    'FaceColor','k', ...
    'FaceAlpha',0.2, ...
    'EdgeColor','none', ...
    'FaceLighting','none');

V2 = mdp_shadow(C2, LD);
F2 = C2.Faces;

S2 = patch( ...
    'Vertices',V2, ...
    'Faces',F2, ...
    'FaceColor','k', ...
    'FaceAlpha',0.2, ...
    'EdgeColor','none', ...
    'FaceLighting','none');

% Lighting for cubes
%--------------------------------------------------------------------------
lighting gouraud
camlight(90,60)
drawnow

if OPT.save
    cn_animation(1);
end

if OPT.frames

    % Select frames to display in summary figure
    %----------------------------------------------------------------------
    frameList = round(linspace(1,(size(o,2)-1)*8,6));

    % Storage for frame snapshots
    %----------------------------------------------------------------------
    snapshots = cell(length(frameList),1);
    snapCount = 1;
end

% Animation
%--------------------------------------------------------------------------
for t = 1:size(o,2)-1

    % Motion vectors
    %----------------------------------------------------------------------
    d1 = [ ...
        x(o(1,t+1))-x(o(1,t)), ...
        y(o(1,t+1))-y(o(1,t)), ...
        0];

    d2 = [ ...
        x(o(2,t+1))-x(o(2,t)), ...
        y(o(2,t+1))-y(o(2,t)), ...
        0];

    for i = 1:8

        % Move cubes
        %------------------------------------------------------------------
        C1.Vertices = C1.Vertices + d1/8;
        C2.Vertices = C2.Vertices + d2/8;

        % Update shadows
        %------------------------------------------------------------------
        S1.Vertices = mdp_shadow(C1,LD);
        S2.Vertices = mdp_shadow(C2,LD);

        % Update trajectory trails
        %------------------------------------------------------------------
        if OPT.trails

            % Current cube centres
            c1 = mean(C1.Vertices,1);
            c2 = mean(C2.Vertices,1);

            % Append coordinates
            trail1x(end+1) = c1(1)-0.1; %#ok
            trail1y(end+1) = c1(2)-0.1; %#ok
            trail2x(end+1) = c2(1)+0.1; %#ok
            trail2y(end+1) = c2(2)+0.1; %#ok

            % Update plotted trails
            set(T1, ...
                'XData',trail1x, ...
                'YData',trail1y, ...
                'ZData',0.02*ones(size(trail1x)));

            set(T2, ...
                'XData',trail2x, ...
                'YData',trail2y, ...
                'ZData',0.02*ones(size(trail2x)));
        end

        if OPT.save
            cn_animation(t+1,1,1/16);
        end

        drawnow

        % Store selected frames
        %------------------------------------------------------------------
        if OPT.frames && ismember((t-1)*8 + i, frameList)

            % Capture current figure
            %------------------------------------------------------------------
            snapshots{snapCount} = getframe(gcf);
            snapCount = snapCount + 1;
        end
    end
end

% Create summary figure with selected frames
%--------------------------------------------------------------------------
if OPT.frames
    cn_figure('Selected frames')
    tiledlayout(3,2,'Padding','compact')

    for k = 1:length(snapshots)
        nexttile
        imshow(snapshots{k}.cdata)
        title(['t = ',num2str(frameList(k)/8,2)])
    end
end

function C = mdp_cube(x,d,c)
% Function to create cubes for animation with position, x, dimension, d,
% and colour c.
%--------------------------------------------------------------------------

s = d/2;
V = [
    -s -s 0;
     s -s 0;
     s  s 0;
    -s  s 0;
    -s -s d;
     s -s d;
     s  s d;
    -s  s d
    ] + [x 0];

F = [
    1 2 3 4;
    5 6 7 8;
    1 2 6 5;
    2 3 7 6;
    3 4 8 7;
    4 1 5 8
    ];

C = patch('Vertices',V,'Faces',F,...
    'FaceColor',c,'FaceAlpha',1,'EdgeColor','k','FaceLighting','gouraud');

function V = mdp_shadow(C,L)
% Function to create shadow for each of the cubes (C) with light direction
% L.
%--------------------------------------------------------------------------
V = C.Vertices;
t = V(:,3)/L(3);
V = V - t.*L;
V(:,3) = 0;

function mdp_induction_demo(h,B,Q,dom)
% This incorporates the same elements as the mp_induction.m routine
% (simplified) but with plotting to enable visualisation of paths.
%--------------------------------------------------------------------------

% Pre-allocation
%--------------------------------------------------------------------------
ind = find(~cellfun(@isempty,h)); % Identify those state factors with associated mandatory states
b = cell(size(ind));              % Initialise cell array to contain matrices for backwards induction
k = zeros(size(ind));             % Initialise cell array to contain vectorised representation of goals

% Prepare matrices for backwards induction
%--------------------------------------------------------------------------
for i = 1:length(ind)                                                       % Loop over state factors
    k(i) = h{ind(i)}(1);                                                    % Index of mandatory state in each factors
    [~,iU,Ui] = intersect(dom(ind(i)).u,[dom([dom(ind(i)).s]).u]);         % Find shared actions with states in domain
    ib     = 1; % placeholder for future development (to account for domains of multiple states)
    bi     = 3 + (1:length(dom(ind(i)).s));
    bi(ib) = 4;
    bi(1)  = ib+3;
    ui     = max(bi) + (1:length(dom(ind(i)).u));
    iu     = max(ui) + (1:length(dom(dom(ind(i)).s(ib)).u));
    iu(Ui) = ui(iU);
    b{i}   = mp_tensor_con(B{ind(i)},B{dom(ind(i)).s(ib)},...
        1:4,[1 3 bi ui],[2 4 iu])>1/8;
end

% Main induction loop
%--------------------------------------------------------------------------
z = 0;                                                                      % initialise flag for completed identification of next state on path
n = 0;                                                                      % initialise number of steps back from goal
K = zeros([size(b{i},[1 2]) 64]);                                           % initialise volume
while z == 0 && n < 64                                                      % continue either until maximum number of steps or goal reached
    q = true(numel(Q),1);                                                   % initialise within each loop to say 'true' that each of the possible next states is on the path to the mandated state
    for i = 1:length(ind)                                                   % loop over factors with mandated states
        if n==0
            K(:,:,n+1) = mp_oh(size(b{i},1),k(i))*ones(size(b{i},2),1)';    % Create joint matrix
        else
            K(:,:,n+1) = mp_tensor_con(b{i},K(:,:,n),[1 2],[3 4 1 2],[3 4])>0;
        end
        for j = 1:numel(Q)                                                  % Loop over plausible first moves
            w   = mp_dot(K(:,:,n+1),{Q{j}{ind(i)},Q{j}{dom(ind(i)).s}})...  % Identify those with a high probability of being on a path to mandatory states
                >= 1/2;
            q(j) = q(j) && w;
        end

    end

    % Stopping criteria
    %----------------------------------------------------------------------
    if sum(q)>0                                                             
        z = 1;
        K = K(:,:,1:n+1);
    end
    n = n + 1;
end

% Plot the volume from the current time that leads to the destination
%--------------------------------------------------------------------------
cn_figure('Induction')

p = patch(isosurface(K, 0.5));
isonormals(K, p);
set(p, 'FaceColor', [0.2 0.6 0.9], ...
       'EdgeColor', 'none');
view(3);
axis([1 size(K,1) 1 size(K,2) 1 size(K,3)])
camlight headlight;
material dull;
lighting phong;

grid on;
box on;

ax = gca;
ax.GridAlpha = 0.25;
ax.LineWidth = 1.2;

xlabel('factor 1')
ylabel('factor 2')
zlabel('steps to goal')
view(300,-15)