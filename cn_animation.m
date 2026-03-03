function cn_animation(t,d0,d,folder,name)
% Function to create and save a GIF animation from a figure whose elements
% are iteratively adjusted in a loop. Arguments are:
% t        - current iteration
% folder   - to save into (default 'Graphics')
% name     - of file to save (default 'Animation')
% d0       - initial delay time for first frame (default 1)
% d        - delay for subsequent frames (default 0.3)
%--------------------------------------------------------------------------

if nargin < 2, d0     = 1;           end
if nargin < 3, d      = 0.3;         end
if nargin < 4, folder = 'Graphics';  end
if nargin < 5, name   = 'Animation'; end


filename = [folder '/' name '.gif'];

F  = getframe(gcf);
im = frame2im(F);
[MM,MMM] = rgb2ind(im,256);
if t==1
    imwrite(MM,MMM,filename,'gif','LoopCount',Inf,'DelayTime',d0);
else
    imwrite(MM,MMM,filename,'gif','WriteMode','append','DelayTime',d);
end