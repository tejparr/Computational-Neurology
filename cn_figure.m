function cn_figure(name)
% Create a figure of standardised size and properties
%--------------------------------------------------------------------------

screenSize = get(0, 'ScreenSize');
Fh          = screenSize(4)*0.8;
Fl          = (screenSize(3) - Fh)/2;
Fb          = (screenSize(4) - Fh)/2;
figure('WindowStyle','normal','Name',name,'Color','w','Position',[Fl, Fb, Fh, Fh]); clf