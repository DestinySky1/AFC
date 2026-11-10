function result = run_structured_iv_afc(input_file, seconds, probe_rms)
% End-to-end MATLAB closed-loop measured-path simulation and IV comparison.
% OBSERVED mic, actual speaker, and probe are the ONLY estimator inputs.
% This MATLAB port is provided for user-side independent testing and has NOT
% been executed by the Python-only container.

if nargin<1 || isempty(input_file)
    input_file=fullfile(fileparts(mfilename('fullpath')),'data','afc_inputs_10s.mat');
end
if nargin<2 || isempty(seconds), seconds=60; end
if nargin<3 || isempty(probe_rms), probe_rms=.05; end
assert(seconds>0 && probe_rms>0,'Positive duration and probe RMS required');
d=load(input_file,'f_path_bes');
h=d.f_path_bes(:);
N=16000*seconds;

randn('seed',42);
s=filter(fir1(64,.1,'high'),1,.1*randn(N,1));
rng(20261102,'twister');
p=probe_rms*(2*double(rand(N,1)>=.5)-1);

% Exact linear closed loop for gain=1, D=1, extra one LS-buffer delay.
% Zero cancellation coefficients during the calibration interval.
feedback_probe=filter([0;h],1,p);
error=filter(1,[1;0;-h],s+feedback_probe);
speaker=[0;error(1:end-1)]+p;
mic=s+filter([0;h],1,speaker);

fit=identify_structured_iv(mic,speaker,p,numel(h));
reference=score(h,fit.reference_coefficients);
optimized=score(h,fit.coefficients);

% Independent target, probe disabled, to measure actual simulated echo RMS.
randn('seed',1234);
heldout=filter(fir1(64,.1,'high'),1,.1*randn(160000,1));
baseline_err=filter(1,[1;0;-h],heldout);
ref_err=filter(1,[1;0;-(h-fit.reference_coefficients)],heldout);
opt_err=filter(1,[1;0;-(h-fit.coefficients)],heldout);
start=2001;
noise_base=baseline_err(start:end)-heldout(start:end);
noise_ref=ref_err(start:end)-heldout(start:end);
noise_opt=opt_err(start:end)-heldout(start:end);
ref_reduction=20*log10(norm(noise_base)/norm(noise_ref));
opt_reduction=20*log10(norm(noise_base)/norm(noise_opt));

result=struct('reference',reference,'optimized',optimized, ...
    'fit',fit,'probe_rms',sqrt(mean(p.^2)), ...
    'training_speaker_max',max(abs(speaker)), ...
    'reference_echo_suppression_db',ref_reduction, ...
    'optimized_echo_suppression_db',opt_reduction);
fprintf('Reference: NFCE %.4f, ASG %+8.4f dB, echo reduction %+7.4f dB\n', ...
    reference.NFCE,reference.ASG,ref_reduction);
fprintf('Optimized: NFCE %.4f, ASG %+8.4f dB, echo reduction %+7.4f dB\n', ...
    optimized.NFCE,optimized.ASG,opt_reduction);
fprintf('Crossfit selected group=%d, local=%d; mic/source oracle NOT passed to estimator\n', ...
    fit.group_width,fit.local_width);
end

function m=score(h,w)
res=h-w;
m.NFCE=sum(res.^2)/sum(h.^2);
f0=fft(h,16384);fr=fft(res,16384);
m.MSG0=-20*log10(max(abs(f0)));
m.MSG=-20*log10(max(abs(fr)));
m.ASG=m.MSG-m.MSG0;
end
