function DEMO_Articulation


Hz = 8192;  % Default sampling frequency
f0 = 100;   % Fundamental frequency 100Hz


% Formant frequencies (https://en.wikipedia.org/wiki/Formant)
%--------------------------------------------------------------------------
f1_2 = [850 1610;  % 'a'
        390 2300;  % 'e'
        240 2400;  % 'i'
        360 640;   % 'o'
        250 595];  % 'u


% Generate vowel sound
%--------------------------------------------------------------------------
v  = 1;
T  = 0.25; 
t  = 0:1/Hz:T;
F  = [f1_2(v,1)*ones(1,length(t));
      f1_2(v,2)*ones(1,length(t))];
S0 = exp(sin(2*pi*t*f0));
S  = sin(pi*t/T).*S0.*sum(sin(2*pi*t.*F))/16;

plot(S)

sound(S)