function mdp_fluency_speak(pomdp)
% Function to generate audio associated with simulated speech using 
% phonemes. This uses a standard synthesiser via .NET.
%--------------------------------------------------------------------------

% First map from the pseudophonemes used in the DEMO_Speech_Fluency.m
% function to those recognisable by speech synthesisers.
%--------------------------------------------------------------------------
phoneme_map = containers.Map();

% Nasals
phoneme_map('m')   = 'm';
phoneme_map('n')   = 'n';
phoneme_map('ng')  = 'ŋ';

% Plosives (fortis = voiceless, lenis = voiced)
phoneme_map('p')   = 'p';
phoneme_map('t')   = 't';
phoneme_map('ch')  = 'tʃ';  % as in 'church'
phoneme_map('k')   = 'k';

phoneme_map('b')   = 'b';
phoneme_map('d')   = 'd';
phoneme_map('j')   = 'dʒ';  % as in 'judge'
phoneme_map('g')   = 'g';

% Fricatives (fortis)
phoneme_map('f')   = 'f';
phoneme_map('th')  = 'θ';  % voiceless dental, as in 'think'
phoneme_map('s')   = 's';
phoneme_map('sh')  = 'ʃ';  % as in 'shoe'
phoneme_map('x')   = 'x';  % voiceless velar fricative (as in 'loch')
phoneme_map('h')   = 'h';

% Fricatives (lenis)
phoneme_map('v')   = 'v';
phoneme_map('eth') = 'ð';   % voiced dental, as in 'this'
phoneme_map('z')   = 'z';
phoneme_map('zj')  = 'ʒ';   % as in 'measure'

% Approximants
phoneme_map('l')   = 'l';
phoneme_map('r')   = 'ɹ';   % English 'r' (not trilled)
phoneme_map('y')   = 'j';   % as in 'yes'
phoneme_map('w')   = 'w';

% Vowels
phoneme_map('I')   = 'ɪ';   % as in 'bit'
phoneme_map('i')   = 'iː';  % as in 'beet'
phoneme_map('oo')  = 'ʊ';   % as in 'foot'
phoneme_map('u')   = 'uː';  % as in 'goose'
phoneme_map('owe') = 'oʊ';  % diphthong, as in 'go'

phoneme_map('e')   = 'ɛ';   % as in 'bed'
phoneme_map('ir')  = 'ɜː';  % as in 'bird'
phoneme_map('oe')  = 'əː';  % schwa-ish long (approximate)
phoneme_map('or')  = 'ɔː';  % as in 'saw'

phoneme_map('æ')   = 'æ';   % as in 'cat'
phoneme_map('ar')  = 'ɑː';  % as in 'father'


% Construct phonetic SSML string
%--------------------------------------------------------------------------
phonemeString = '';
for i = 1:size(pomdp.o,2)
    phoneme = pomdp.par.cv{pomdp.o(1,i)};
    if strcmp(phoneme,' ') % Deal with spaces
        phonemeString = [phonemeString, '<break time="200ms"/>']; %#ok<AGROW>
    else                   % Otherwise append phoneme
        ipa = phoneme_map(phoneme);
        phonemeString = [phonemeString, ...
            '<phoneme alphabet="ipa" ph="', ipa, '">', phoneme, '</phoneme> ']; %#ok<AGROW>
    end
end

% Final SSML string
%--------------------------------------------------------------------------
ssml = ['<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">', ...
         phonemeString, ...
        '</speak>'];

% Create a .NET Speech Synthesizer object
%--------------------------------------------------------------------------
NET.addAssembly('System.Speech');
synthesizer       = System.Speech.Synthesis.SpeechSynthesizer;
synthesizer.Rate  = -3;
ms                = System.IO.MemoryStream();

% Set output to memory stream
%--------------------------------------------------------------------------
synthesizer.SetOutputToWaveStream(ms);

% Speak the SSML text
%--------------------------------------------------------------------------
synthesizer.SpeakSsml(ssml);

% Construct audiogram (using temp file)
%--------------------------------------------------------------------------
ms.Position = int64(0);
buffer      = ms.ToArray();
byteArray   = uint8(buffer);

tempFile = [tempname, '.wav'];
fid = fopen(tempFile, 'w');
fwrite(fid, byteArray, 'uint8');
fclose(fid);

[y, Fs] = audioread(tempFile);
delete(tempFile);

% Plotting
%--------------------------------------------------------------------------
t = (0:length(y)-1)/Fs;
cn_figure('Speech waveform');
plot(t, y);
xlabel('Time (s)');
ylabel('Amplitude');
title('Synthetic Speech Waveform');
grid on;

% Play audio
%--------------------------------------------------------------------------
sound(y,Fs);


