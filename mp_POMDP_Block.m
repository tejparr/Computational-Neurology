function POMDP = mp_POMDP_Block(pomdp,s,noplot)
% POMDP = mp_POMDP_Block(pomdp,s)
% pomdp - Partially Observed Markov Decision Process
% s     - Matrix (factor x trial) of initial states
%
% This function is designed to simulate a block of trials using the
% mp_POMDP inversion routine. This allows one to specify a set of initial
% conditions and to simulate learning through accumulation of Dirichlet
% counts across the block of trials.
%--------------------------------------------------------------------------

if nargin < 3, noplot = false; end

N     = size(s,2); % Number of trials in a block
POMDP = cell(N,1); % Initialise POMDP array


fields = {'a','b','c','d','e'};

for i = 1:N
    pomdp.s  = s(:,i);          % Set initial states
    POMDP{i} = mp_POMDP(pomdp); % Solve POMDP
    
    for k = 1:numel(fields)     % Update Dirichlet counts
        f = fields{k};
        if isfield(pomdp, f)
            pomdp.(f) = POMDP{i}.BS.(f);
        end
    end
end

if ~noplot
    cn_figure('POMDP Learning')

    % Plot changing path probabilities over time
    %----------------------------------------------------------------------
    subplot(3,1,1)
    
    P = cellfun(@(s) [s.P{:}], POMDP, 'UniformOutput', false);
    imagesc(1-[P{:}]);
    colormap gray
    title('Paths (concatenated)')
    ylabel('Actions')
    xlabel('Timestep')

    for k = 1:numel(fields)
        f = fields{k};
        if isfield(pomdp, f)
            D.(f) = cellfun(@(s) pomdp_block_vcat(s.(f)), ...
                POMDP, 'UniformOutput', false);
        end
    end

    % Plot changing Dirichlet parameters over time
    %----------------------------------------------------------------------
    fn = fieldnames(D);
    nf = numel(fn);

    for k = 1:nf
        subplot(nf*3,1,nf+k)
        Df = [D.(fn{k}){:}];
        imagesc(max(Df(:))-Df)
        colormap gray
        title(['Dirichlet counts - ' fn{k}])
        ylabel('Parameter')
        xlabel('Timestep')
    end

end

function c = pomdp_block_vcat(C)
C = cellfun(@(x) x(:), C, 'UniformOutput', false);
c = vertcat(C{:});