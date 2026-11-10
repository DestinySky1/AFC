function result = run_probe_iv_afc(input_file, probe_rms)
% Probe-assisted closed-loop AFC with a real 128-tap measured feedback path.
% Independent test signal is added to the loudspeaker during calibration.
% Estimation uses mic, speaker, and probe ONLY; never vTarget or f_path_bes.
% True path/target are used ONLY inside this simulation and for evaluation.
%
% IMPORTANT: requires MATLAB with Signal Processing Toolbox, has NOT been
% executed in the present environment; Python implementation was verified.

if nargin < 1 || isempty(input_file)
    input_file = fullfile(fileparts(mfilename('fullpath')),'data','afc_inputs_10s.mat');
end
if nargin < 2 || isempty(probe_rms), probe_rms = 0.1; end
assert(probe_rms > 0 && isfinite(probe_rms),'probe_rms must be positive');
a = load(input_file,'vTarget','f_path_bes');
source = a.vTarget(:);
true_path = a.f_path_bes(:);
N = numel(source);
L = numel(true_path);

rng(20261102,'twister');
probe = probe_rms * (2*double(rand(N,1)>=0.5)-1); % bounded white probe
microphone = zeros(N,1);
speaker = zeros(N,1);
history = zeros(L,1);
last_error = 0;

% During this calibration the cancellation estimate is initially zero.
% The experiment remains CLOSED LOOP: speaker(n) = mic(n-1) + probe(n),
% and mic(n) = source(n) + h' * speaker_history(n).
for n = 1:N
    microphone(n) = source(n) + true_path.' * history;
    speaker(n) = last_error + probe(n);
    last_error = microphone(n);
    history(2:end) = history(1:end-1);
    history(1) = speaker(n);
end

[raw,condition] = solve_iv(microphone,speaker,probe,L);
mid = floor(N/2);
[w1,~] = solve_iv(microphone(1:mid),speaker(1:mid),probe(1:mid),L);
[w2,~] = solve_iv(microphone(mid+1:end),speaker(mid+1:end),probe(mid+1:end),L);
sigma = sqrt(mean((w1-w2).^2))/2;
choices = unique([0;abs(raw)]);
risk = zeros(size(choices));
for k = 1:numel(choices)
    thr = choices(k);
    risk(k) = L*sigma^2 + sum(min(raw.^2,thr^2)) ...
              - 2*sigma^2*sum(abs(raw)<=thr);
end
[~,which] = min(risk);
threshold = choices(which);
w = sign(raw) .* max(abs(raw)-threshold,0);

residual = true_path - w;
nfce = sum(residual.^2) / sum(true_path.^2);
nfft = 16384;
msg0 = -20*log10(max(abs(fft(true_path,nfft))));
msg = -20*log10(max(abs(fft(residual,nfft))));
asg = msg - msg0;

% Independent colored source, probe OFF after calibration.
rng(20261103,'twister');
heldout = filter(fir1(64,.1,'high'),1,0.1*randn(N,1));
before = filter(1,[1;0;-true_path],heldout);
after = filter(1,[1;0;-(true_path-w)],heldout);
skip = max(2000,3*L);
before_echo = before(skip+1:end)-heldout(skip+1:end);
after_echo = after(skip+1:end)-heldout(skip+1:end);
suppression = 20*log10(sqrt(mean(before_echo.^2)) / ...
                       sqrt(mean(after_echo.^2)));

result = struct('coefficients',w,'raw_coefficients',raw, ...
    'nfce',nfce,'MSG0_db',msg0,'MSG_db',msg,'ASG_db',asg, ...
    'heldout_echo_suppression_db',suppression, ...
    'moment_condition',condition,'estimated_noise_std_per_tap',sigma, ...
    'soft_threshold',threshold,'probe_rms',sqrt(mean(probe.^2)), ...
    'calibration_peak_speaker',max(abs(speaker)), ...
    'source_rms',sqrt(mean(source.^2)));
fprintf('Measured-path probe-IV: NFCE=%.6f, ASG=%+.3f dB, heldout echo reduction=%+.3f dB\n', ...
        nfce,asg,suppression);
fprintf('Probe RMS=%.4f, source RMS=%.4f; original baseline unchanged.\n', ...
        result.probe_rms,result.source_rms);
end

function [w,condition] = solve_iv(m,u,p,L)
N = numel(m);
assert(N >= 20*L,'Not enough calibration samples');
assert(mean(p.^2) > 1e-15,'Nonzero independent probe required');
col = zeros(L,1); row = zeros(L,1); b = zeros(L,1);
for k=0:L-1
    col(k+1) = p(1:N-k).' * u(1+k:N) / N;
    row(k+1) = p(1+k:N).' * u(1:N-k) / N;
    b(k+1) = p(1:N-k-1).' * m(k+2:N) / N;
end
G = toeplitz(col,row);
condition = cond(G);
assert(isfinite(condition) && condition < 1e6,'IV moment poorly conditioned');
w = (G + 1e-5 * mean(p.^2) * eye(L)) \ b;
end
