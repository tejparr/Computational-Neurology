function MDP = DEMO_Speech_Fluency
% This demo is designed to provide a minimal example of the articulation
% of speech capable of demonstrating fluency. A key aspect of this demo
% is the idea that when one chooses to say a given word, one must also 
% determine its duration. This is implemented through a set of orbits
% whose duration must be selected. The combination of selected word and
% position in the orbit determine the phoneme to be spoken. The chosen 
% word can only change when the final state of the orbit is reached.
% As currently implemented, this has an interesting cycle in the graph
% representing the generative model, in that the current word and orbit
% state jointly predict the next word and orbit state. This potentially has
% an impact on disorders of speech fluency, including stuttering and
% non-fluent aphasias.
%--------------------------------------------------------------------------
close all
rng default 
OPTIONS.save  = 0; % Option to save animation
OPTIONS.start = 1; % Option to start at oribt state 1 for each word (alternative 0 is to start from 8 - wordlength)
cd(fileparts(mfilename('fullpath')))

% Simulation set-up (initial states)
%--------------------------------------------------------------------------
Nw = 16;      % Number of words in vocabulary
s  = [1;      % 1. Silence/alone, 2. silence/company, 3. speaking to myself, 4. speaking to another, 5. listening to another, 6. both speaking
      1;      % Word being spoken [1-Nw]
      8];     % Position in orbit [1-8], where 1 is the start of a word and 8 is a silent state that always progresses to 1

% Key parameters to vary (beliefs):
%--------------------------------------------------------------------------

% Social context parameters (all range [0-1]) - these parameters are also
% used in simulating the environment, unlike the other belief parameters
% below
b1   = 1/10; % Probability that someone else will arrive
b2   = 1/10; % Probability that someone else will leave
b3   = 8/10; % Probability that, if present and I'm silent, someone else will start speaking
b4   = 4/10; % Probability that, if present and I'm silent, someone else will stop speaking
b5   = 4/10; % Probability that, if present and I'm speaking, someone else will start speaking
b6   = 8/10; % Probability that, if present and I'm speaking, someone else will stop speaking

% Orbit parameters
b7   = 1;    % Probability that will progress as expected around orbit [0-1]

% Path probabilities
e1   = 8;    % Confidence that I know what I want to say [<=0] (8 corresponds to a high degree of confidence)
e2   = 1;    % Confidence that I will start speaking     [0-1]

% Likelihood parameters ([0-1])
a1   = 1;    % Confidence in predictions of phonemes given latent states
a2   = 1;    % Confidence in predictions of proprioceptive states given latent states

% Open simplified tables of English phonemes
%-------------------------------------------------------------------------
consonants = readtable('consonants.csv');
vowels     = readtable('vowels.csv');

% Extract factors for phonemes
%-------------------------------------------------------------------------
factor1 = {consonants.Properties.VariableNames{2:end} vowels.Properties.VariableNames{2:end}};
factor2 = [consonants{:,1};vowels{:,1}]';

% List non-zero elements
%-------------------------------------------------------------------------
v         = vowels{:,2:end};
[iv1,iv2] = find(~cellfun(@isempty,v));
iv        = [iv1 iv2];
v         = v(~cellfun(@isempty,v));
c         = consonants{:,2:end};
[ic1,ic2] = find(~cellfun(@isempty,c));
c         = c(~cellfun(@isempty,c));
ic        = [ic1 ic2];

% Generate pseudo-words as vocabulary
%-------------------------------------------------------------------------
words = fluency_vocab(c,v,ic,iv,Nw);      % Generate vocabulary of pseudo-words

% Create plausible syntax
%-------------------------------------------------------------------------
Orb = circshift(eye(Nw),1);               % Create generic orbit
Syn = repmat(Orb,[1 1 6]);                % Create matrix for possible orbits
for i = 1:size(Syn,3)
    r          = randperm(size(Syn,1));
    Syn(:,:,i) = Syn(r,r,i);
end

% Prepare parameters for generative process
%-------------------------------------------------------------------------
par.words = words;
par.I     = [ic;(iv + max(ic))];
par.cv    = [c(:);v(:);{' '}];
par.B     = Syn;

% Construct POMPD model
%=========================================================================

% Initial state distribtions
%-------------------------------------------------------------------------
D = cell(3,1);
D{1} = ones(6,1)/6;   % Silence/alone, silence/company, speaking to myself, speaking to another, listening to another, both speaking
D{2} = ones(Nw,1)/Nw; % Word being spoken
D{3} = ones(8,1)/8;   % Orbit (controllable) for sequencing phonemes (the final state allows for changes in the word)

% Likelihood distributions
%------------------------------------------------------------------------
A{1} = zeros(numel(par.cv),length(D{1}),length(D{2}),length(D{3}));                     % Auditory outcomes
A{2} = zeros(numel(factor1)+1,length(D{1}),length(D{2}),length(D{3}));                  % Proprioceptive (airflow)
A{3} = zeros(numel(factor2)+1,length(D{1}),length(D{2}),length(D{3}));                  % Proprioceptive (pharynx)
A{4} = zeros(2,length(D{1}));                                                           % Vision (other present or not)

if OPTIONS.start==1
    for f1 = 1:length(D{1})
        for f2 = 1:length(D{2})
            for f3 = 1:length(D{3})
                if ismember(f1,[1 2])
                    A{1}(end,f1,f2,f3) = 1; % Silence outcome
                elseif f3 > size(words.indices{f2},1)
                    A{1}(end,f1,f2,f3) = 1; % Silence outcome
                elseif f1 == 6 % if both speaking
                    A{1}(:,f1,f2,f3) = 1/size(A{1},1); % ambiguous auditory outcome
                else
                    ph     = words.indices{f2}(f3,:);
                    [~,id] = ismember(ph,[ic;(iv+max(ic))],'rows');
                    A{1}(id,f1,f2,f3) = 1;
                end
                if ismember(f1,[1 2 5])
                    A{2}(end,f1,f2,f3) = 1;
                    A{3}(end,f1,f2,f3) = 1;
                elseif f3 > size(words.indices{f2},1)
                    A{2}(end,f1,f2,f3) = 1;
                    A{3}(end,f1,f2,f3) = 1;
                else
                    ph     = words.indices{f2}(f3,:);
                    A{2}(ph(1),f1,f2,f3) = 1;
                    A{3}(ph(2),f1,f2,f3) = 1;
                end
            end
        end
    end
else
    for f1 = 1:length(D{1})
        for f2 = 1:length(D{2})
            for f3 = 1:length(D{3})
                if ismember(f1,[1 2])
                    A{1}(end,f1,f2,f3) = 1; % Silence outcome
                elseif f3 < 9 - size(words.indices{f2},1)
                    A{1}(end,f1,f2,f3) = 1; % Silence outcome
                elseif f1 == 6 % if both speaking
                    A{1}(:,f1,f2,f3) = 1/size(A{1},1); % ambiguous auditory outcome
                else
                    ph     = words.indices{f2}(f3 - 8 + size(words.indices{f2},1),:);
                    [~,id] = ismember(ph,[ic;(iv+max(ic))],'rows');
                    A{1}(id,f1,f2,f3) = 1;
                end
                if ismember(f1,[1 2 5])
                    A{2}(end,f1,f2,f3) = 1;
                    A{3}(end,f1,f2,f3) = 1;
                elseif f3 < 9 - size(words.indices{f2},1)
                    A{2}(end,f1,f2,f3) = 1;
                    A{3}(end,f1,f2,f3) = 1;
                else
                    ph     = words.indices{f2}(f3 - 8 + size(words.indices{f2},1),:);
                    A{2}(ph(1),f1,f2,f3) = 1;
                    A{3}(ph(2),f1,f2,f3) = 1;
                end
            end
        end
    end    
end

A{1}(A{1}==1) = a1;
A{1}(A{1}==0) = (1-a1)/size(A{1},1);
A{2}(A{2}==1) = a2;
A{2}(A{2}==0) = (1-a2)/size(A{2},1);
A{3}(A{3}==1) = a2;
A{3}(A{3}==0) = (1-a2)/size(A{3},1);

A{4}(sub2ind(size(A{4}),[1 1 2 2 2 2],[1 3 2 4 5 6])) = 1;

% Transition probabilities
%------------------------------------------------------------------------
B    = cell(size(D));

B{1} = zeros(length(D{1}),length(D{1}),2); 

% Action 1 (stop speaking or stay silent)
%---------------------------------------------------------------------------------------------------
B{1}(:,:,1) = [1-b1       b2       1-b1       b2             0             0;       % Silence/alone
                b1   (1-b2)*(1-b3)  b1   (1-b2)*(1-b5)       b4            b6;      % Silence/company
                0          0        0          0             0             0;       % Speaking to myself
                0          0        0          0             0             0;       % speaking to another
                0    (1-b2)*b3      0    (1-b2)*b5         1-b4           1-b6;     % Listening to another
                0          0        0          0             0             0];      % Both speaking

% Action 2 (start or continue speaking)
%---------------------------------------------------------------------------------------------------
B{1}(:,:,2) = [ 0          0        0          0             0             0;       % Silence/alone
                0          0        0          0             0             0;       % Silence/company
               1-b1        b2      1-b1       b2             0             0;       % Speaking to myself
                b1   (1-b2)*(1-b3)  b1   (1-b2)*(1-b5)       b4            b6;      % speaking to another
                0          0        0          0             0             0;       % Listening to another
                0    (1-b2)*b3      0    (1-b2)*b5          1-b4          1-b6];    % Both speaking

for f1 = 1:length(D{1})
    for f3 = 1:length(D{3}) 
        for u1 = 1:size(Syn,3)
            if f3 < 8
                B{2}(:,:,f1,f3,u1)  = eye(length(D{2}));
            elseif f1 == 5 % If I am listening to someone else
                B{2}(:,:,f1,f3,u1)  = sum(Syn,3)/size(Syn,3);
            else
                B{2}(:,:,f1,f3,u1)  = Syn(:,:,u1);
            end
        end
    end
end
if OPTIONS.start == 1
    for f2 = 1:length(D{2})
        Np           = size(words.indices{f2},1);
        B{3}(:,:,f2) = circshift(eye(8),1,1);
        B{3}(:,Np:7,f2) = 0;
        B{3}(8,Np:7,f2) = 1;
    end
else
    for f2 = 1:length(D{2})
        Np           = size(words.indices{f2},1);
        B{3}(:,:,f2) = circshift(eye(8),1,1);
        B{3}(:,1:8-Np,f2) = 0;
        B{3}(9-Np,1:8-Np,f2) = 1;
    end
end
for f2 = 1:size(B{3},3)
    B{3}(:,:,f2) = b7*B{3}(:,:,f2) + (1-b7)*(circshift(B{3}(:,:,f2),1) + circshift(B{3}(:,:,f2),-1))/2;
end

% Preferences
%-------------------------------------------------------------------------
C = cell(size(A));
for g = 1:numel(A)
    C{g} = zeros(size(A{g},1),1);
end

% Paths
%-------------------------------------------------------------------------
E1      = randn(size(Syn,3),1);
E{1}    = mp_softmax(e1*E1);
E{2}    = [1-e2;e2];

% Assemble POMDP
%-------------------------------------------------------------------------
mdp.A = A;
mdp.B = B;
mdp.C = C;
mdp.D = D;
mdp.E = E;

% Domains
%-------------------------------------------------------------------------
for g = 1:3
    mdp.dom.A(g).s = [1 2 3];
    mdp.dom.A(g).u = [];
end
mdp.dom.A(4).s = 1;
mdp.dom.A(4).u = [];

mdp.dom.B(1).s = [];
mdp.dom.B(1).u = 2;
mdp.dom.B(2).s = [1 3];
mdp.dom.B(2).u = 1;
mdp.dom.B(3).s = 2;
mdp.dom.B(3).u = [];

% Generative process and simulation settings
%-------------------------------------------------------------------------
mdp.T   = 28;
mdp.N   = 1;
mdp.gen = @mdp_fluency_gen; % Generative process (i.e., simulation environment)
mdp.s   = s;                % True initial states of above
mdp.par = par;              % Parameters for generative process

% Active Inversion of generative model
%--------------------------------------------------------------------------
MDP = mp_POMDP(mdp);

figure('Color','w','Name','Animation','WindowStyle','normal'); clf
mdp_fluency_animation(MDP,OPTIONS)

mp_pomdp_belief_plot(MDP,{'Social','Word','Orbit'},{'Audition','Airflow','Pharynx','Vision'})

mdp_fluency_speak(MDP)

return 

% Illustrate effect of social context on parameter sensitivity
%--------------------------------------------------------------------------
e1 = 0:6; %#ok<UNRCH>
e2 = 0.5:0.05:0.8;
g  = zeros(length(e1),length(e2),2);
mdp.T = 32;
for i1 = 1:length(e1)
    for i2 = 1:length(e2)
        mdp.E{1} = mp_softmax(e1(i1)*E1);
        mdp.E{2} = [1-e2(i2);e2(i2)];
        for k = 1:2
            mdp.s(1) = k;
            MDP = mp_POMDP(mdp);
            o   = [0 MDP.o(1,find(MDP.o(1,:)<size(MDP.A{1},1),1,'first'):end) 0];
            o   = diff(o==size(MDP.A{1},1));
            d   = 0;
            for j = find(o>0)
                d = d + find(o(j:end)<0,1,'first')-1;
            end
            g(i1,i2,k) = d/sum(o>0);
        end
    end
end

figure('Color','w','Name','Parameter interactions','WindowStyle','normal'); clf
cond = {'(alone)','(company)'};
K = ones(3,3)/9;
for k = 1:2
    subplot(1,2,k)
    imagesc(e1,e2,imfilter(g(:,:,k),K,'symmetric')), colormap gray, axis square, clim([1 max(g(:))]), colorbar
    xlabel('Confidence in what to say')
    ylabel('Confidence in whether to speak')
    title(['Average duration of pauses ' cond{k}])
end

function words = fluency_vocab(c,v,ic,iv,Nw)
% This function generates a vocubulary of pseudo-words comprising sequences of
% phonemes of varying lengths (up to 7 per word):
%
% Inputs:
% c              - Cell containing consononant phoneme strings
% v              - Cell containing vowel phoneme strings
% ic             - Indices for consonants
% iv             - Indices for vowels
% Nw             - Scalar giving size of vocabulary to generate
%
% Outputs:
% words.strings  - Cell giving strings for pseudowords in vocabulary
% words.indices  - Cell giving indicies for vowels and consonants
%------------------------------------------------------------------------------

words.strings = cell(Nw,1);
words.indices = cell(Nw,1);

for i = 1:Nw
    Np = randi(7);      % Number of phonemes in word i
    if Np == 1          % If word i has only one phoneme, ensure is vowel
        r = randi(length(v));
        V = v{r};
        words.strings{i} = ['\' V '\'];
        words.indices{i} = iv(r,:) + max(ic);
    else                % Otherwise alternate vowels and consonants
        if rand>1/2     % Start with vowel
            r = randi(length(v));
            V = v{r};
            words.strings{i} = ['\' V '\'];
            words.indices{i} = iv(r,:) + max(ic);
            p = 'c';
        else            % Start with consonant
            r = randi(length(c));
            C = c{r};
            words.strings{i} = ['\' C '\'];
            words.indices{i} = ic(r,:);
            p = 'v';
        end
        for j = 2:Np
            switch p
                case 'c'
                    r = randi(length(c));
                    C = c{r};
                    words.strings{i} = [words.strings{i} '\' C '\'];
                    words.indices{i} = [words.indices{i};ic(r,:)];
                    p = 'v'; % Vowel next
                case 'v'
                    r = randi(length(v));
                    V = v{r};
                    words.strings{i} = [words.strings{i} '\' V '\'];
                    words.indices{i} = [words.indices{i};iv(r,:)+max(ic)];
                    p = 'c'; % Consonant next
            end
        end
    end
end

function [o,s] = mdp_fluency_gen(s,u,Qo,pomdp)
% Function for generative process
%--------------------------------------------------------------------------

par = pomdp.par;

if nargout > 1
% Advance the states
%------------------------------------------------------------------------
    if s(3) == length(pomdp.D{3})
        s(3) = 1;
        s(2) = randi(length(pomdp.D{3}));
    elseif s(3) == size(par.words.indices{s(2)},1)
        s(3) = length(pomdp.D{3});
    else
        s(3) = s(3)+1;
    end
    if ismember(s(1),[5 6])
        [~,s(1)] = max(pomdp.B{1}(:,s(1),u(2)));
    end
end

% Outcomes
%--------------------------------------------------------------------------
o = zeros(3,1);

for g = 1:numel(pomdp.A)
    o(g,1) = size(pomdp.A{g},1);
end

if ~isempty(Qo)
        
    % Reflexive generation of proprioceptive outcomes
    %---------------------------------------------------------------------
    [~,o(2)] = max(Qo{2});
    [~,o(3)] = max(Qo{3});

    if ismember([o(2), o(3)],par.I,'rows')  % If both proprioceptive outcomes are compatible with phoneme
        [~,o(1)] = ismember([o(2), o(3)],par.I,'rows');
    elseif ismember(s(1),[5 6])
        try
            ph       = par.words.indices{s(2)}(s(3),:);
            [~,o(1)] = ismember(ph,par.I,'rows');
        catch
            o(1) = length(Qo{1});
        end
    else
        o(1) = length(Qo{1});
    end
end
[~,o(4)] = max(pomdp.A{4}(:,s(1)));

function mdp_fluency_animation(pomdp,OPTIONS)
% This function takes the solved pomdp as an input and produces an animation
% to illustrate the events of the simulation
%-------------------------------------------------------------------------

% Get profile image for visual outcome
Im = imread('Graphics/profile.png');
Im = logical(sum(Im,3));
BG = zeros(size(Im));
BG(1:4,:)       = 1;
BG(end-4:end,:) = 1;
BG(:,1:4)       = 1;
BG(:,end-4:end) = 1;

Jm = [zeros(size(Im,1),size(Im,2)*3) fliplr(Im)];
Lm = imread('Graphics/speech.png');
Lm = logical(sum(Lm,3));
Lm = Lm(1:4:end,1:4:end);

% Coordinates for cycle of 8
C = zeros(8,2);
for i = 1:size(C,1)
    C(i,:) = [sin(pi*(i-1)/4) cos(pi*(i-1)/4)];
end

str = '';
subplot(3,2,1)
ax  = gca;
pos = ax.Position;
title('Audition')
axis off
an = annotation('textbox',pos,'String',str,'FontSize',12,'LineStyle','None');
for i = 1:size(pomdp.o,2)
   
    str = [str '/']; %#ok<AGROW>
    for j = 1:length(pomdp.par.cv{pomdp.o(1,i)})
        cla
        str = [str pomdp.par.cv{pomdp.o(1,i)}(j)]; %#ok<AGROW>
        an.String = str;     
        
        % Plot all observables
        %------------------------------------------------------------------
        subplot(3,2,5)
        imagesc((Im*(pomdp.o(4,i)-1) + BG)<1)
        axis equal, colormap gray
        title('Vision')
        axis off

        subplot(3,4,5)
        plot(0,pomdp.o(2,i),'.r','MarkerSize',16)
        axis equal
        axis ij
        axis([-1 1 0 size(pomdp.A{2},1)])        
        set(gca, 'XTick', [], 'YTick', []);
        title('Airflow')
        
        subplot(3,4,6)
        plot(0,pomdp.o(3,i),'.r','MarkerSize',16)
        axis equal
        axis ij
        axis([-1 1 0 size(pomdp.A{3},1)])
        set(gca, 'XTick', [], 'YTick', []);
        title('Pharynx')
       
        % Plot beliefs
        %------------------------------------------------------------------
        subplot(3,2,2)
        [~,jw] = sort(pomdp.Q{i,2});
        axis([0 10 0 10]); 
        for k = jw'
            word = pomdp.par.words.strings{k};
            word(word=='\') = [];
            text(5,5,word,'Color',ones(3,1)-min(1,pomdp.Q{i,2}(k)),'HorizontalAlignment','center','FontSize',12,'FontWeight','bold')
            hold on
        end
        hold off
        axis off
        title('Inferred word')

        subplot(3,2,4)
        if OPTIONS.start==1
            for k = jw'
                l = size(pomdp.par.words.indices{k},1);
                plot(C([1:l 8 1],1),C([1:l 8 1],2),'k','LineWidth',2,'Color',ones(3,1)-min(1,pomdp.Q{i,2}(k))), hold on
            end
            plot(C(:,1),C(:,2),'ok','MarkerSize',10)
        else
           for k = jw'
                l = size(pomdp.par.words.indices{k},1);
                plot(C([(8-l+1):8 1 (8-l+1)],1),C([(8-l+1):8 1 (8-l+1)],2),'k','LineWidth',2,'Color',ones(3,1)-min(1,pomdp.Q{i,2}(k))), hold on
            end
            plot(C(:,1),C(:,2),'ok','MarkerSize',10)
        end
        for k = 1:size(C,1)
            plot(C(k,1),C(k,2),'.','MarkerSize',16,'Color',[1 0 0]*(max(min(pomdp.Q{i,3}(k),1),0)))
        end
        hold off
        title('Inferred timing')
        axis([-1.5 1.5 -1.5 1.5])
        axis equal
        axis off

        subplot(3,2,6)
        q1 = sum(pomdp.Q{i,1}([2 4 5 6]));
        q2 = sum(pomdp.Q{i,1}([3 4 6]));
        q3 = sum(pomdp.Q{i,1}([5 6]));

        Km = Jm;
        Km(1:size(Im,1),1:size(Im,2)) = Im*q1;
        Km(floor(size(Jm,1)/2)+(1-floor(size(Lm,1)/2):ceil(size(Lm,1)/2)),floor(size(Jm,2)/2)+(1-floor(size(Lm,2)/2):ceil(size(Lm,2)/2))) = fliplr(Lm)*q2 + Lm*q3 - fliplr(Lm).*Lm*q2*q3;
        Km(1:4,:)       = 1;
        Km(end-4:end,:) = 1;
        Km(:,1:4)       = 1;
        Km(:,end-4:end) = 1;
        imagesc(1-Km)
        axis equal, colormap gray
        title('Inferred context')
        axis off
        drawnow

        % Animation
        % -----------------------------------------------------------------
        if OPTIONS.save
            F  = getframe(gcf);
            im = frame2im(F);
            [MM,MMM] = rgb2ind(im,256);
            if i==1
                imwrite(MM,MMM,'Graphics/Animation.gif','gif','LoopCount',Inf,'DelayTime',0.1);
            else
                imwrite(MM,MMM,'Graphics/Animation.gif','gif','WriteMode','append','DelayTime',0.1);
            end
        end
    end
end
