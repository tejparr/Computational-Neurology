function mp_pomdp_belief_plot(pomdp,Snames,Onames)
% Function to plot the beliefs generated from a pomdp simulation
%---------------------------------------------------------------

cn_figure('Beliefs')

% Get beliefs about states and observations
%---------------------------------------------------------------
Q = pomdp.Q;
o = pomdp.o;

% Determine number of factors and outcome modalities
%---------------------------------------------------------------
Nf = min(size(Q,2),4);
Ng = min(size(o,1),4);

% Names for above
%---------------------------------------------------------------
if nargin < 2
    Snames = cell(Nf,1);
    for i = 1:Nf
        Snames{i} = ['State ' num2str(i)];
    end
    Onames = cell(Ng,1);
    for i = 1:Ng
        Onames{i} = ['Outcome ' num2str(i)];
    end
end

% Loop through state factors and plot beliefs over time
%----------------------------------------------------------------
for i = 1:Nf
    subplot(Nf+Ng,1,i)
    imagesc(1 - [Q{1:end, i}]), colormap gray
    ylabel(Snames{i})
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
end

% Loop through outcome modalities and plot over time
%-----------------------------------------------------------------
for i = 1:Ng
    subplot(Nf+Ng,1,Nf+i)
    if iscell(pomdp.o)
        O = [pomdp.o{i,:}];
    else
        try
            O = zeros(pomdp.A{i}.Nd,size(o,2));
        catch
            O = zeros(size(pomdp.A{i},1),size(o,2));
        end
        O(sub2ind(size(O),o(i,:),1:size(O,2))) = 1;
    end
    imagesc(1 - O), colormap gray
    ylabel(Onames{i})
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
end
xlabel('Time')