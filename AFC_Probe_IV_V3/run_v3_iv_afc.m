function result=run_v3_iv_afc(input_file, seconds, probe_rms)
% Independent MATLAB reproduction of V2 vs V3 with the same closed-loop data.
% Requires Signal Processing Toolbox (fir1). MATLAB port not executed here.
if nargin<1 || isempty(input_file)
    input_file=fullfile(fileparts(mfilename('fullpath')),'data','afc_inputs_10s.mat');
end
if nargin<2 || isempty(seconds),seconds=60;end
if nargin<3 || isempty(probe_rms),probe_rms=.05;end
assert(seconds>0 && probe_rms>0,'Positive duration and RMS required');
data=load(input_file,'f_path_bes');
h=double(data.f_path_bes(:));L=numel(h);N=round(seconds*16000);
rng(42,'twister');
s=filter(fir1(64,.1,'high'),1,.1*randn(N,1));
rng(20261102,'twister');
p=probe_rms*(2*double(rand(N,1)>=.5)-1);

% Gain 1; D=1 plus one sample of physical speaker-to-mic delay.
probe_feedback=filter([0;h],1,p);
error=filter(1,[1;0;-h],s+probe_feedback);
speaker=[0;error(1:end-1)]+p;
mic=s+filter([0;h],1,speaker);

baseline=identify_structured_iv(mic,speaker,p,L);
optimized=identify_iv_v3(mic,speaker,p,L);
score2=metrics(h,baseline.coefficients);
score3=metrics(h,optimized.coefficients);
result=struct('v2',score2,'v3',score3,'v2_fit',baseline, ...
    'v3_fit',optimized,'probe_rms',sqrt(mean(p.^2)), ...
    'speaker_peak',max(abs(speaker)));
fprintf('V2: NFCE %.5f, ASG %+.3f dB\n',score2.NFCE,score2.ASG);
fprintf('V3: NFCE %.5f, ASG %+.3f dB\n',score3.NFCE,score3.ASG);
fprintf('Paired ASG difference: %+.3f dB\n',score3.ASG-score2.ASG);
end

function out=metrics(h,w)
res=h-w;
out.NFCE=sum(res.^2)/sum(h.^2);
out.MSG0=-20*log10(max(abs(fft(h,16384))));
out.MSG=-20*log10(max(abs(fft(res,16384))));
out.ASG=out.MSG-out.MSG0;
end
