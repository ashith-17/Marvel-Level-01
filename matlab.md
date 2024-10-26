```matlab
% Sampling frequency (Fs) in Hz and the length of the signal (L)
Fs = 1000;            % Sampling frequency in Hz
T = 1/Fs;             % Sampling period (s)
L = 1000;             % Length of the signal (samples)
t = (0:L-1)*T;        % Time vector in seconds

% Define frequencies for sine waves
f1 = 50;              % Frequency of first sine wave (Hz)
f2 = 120;             % Frequency of second sine wave (Hz)

% Generate the signal by combining two sine waves and adding noise
signal = 0.7 * sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t) + 0.5 * randn(size(t));

% Compute the Fourier transform of the signal
Y = fft(signal);      % Compute the Fourier transform of the signal
P2 = abs(Y/L);        % Compute the two-sided spectrum (normalize by L)
P1 = P2(1:L/2+1);     % Take only the positive half of the spectrum
P1(2:end-1) = 2*P1(2:end-1);  % Adjust for single-sided spectrum

f = Fs * (0:(L/2)) / L;  % Frequency range for plotting the spectrum

% Plot the original signal in the time domain
subplot(2,1,1);
plot(t, signal);
title('Original Signal (Time Domain)');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the Fourier Transform (frequency spectrum) in the frequency domain
subplot(2,1,2);
plot(f, P1);
title('Single-Sided Amplitude Spectrum of the Signal (Frequency Domain)');
xlabel('Frequency (Hz)');
ylabel('|P1(f)|');
