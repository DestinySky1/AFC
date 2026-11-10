function result = identify_structured_iv(mic, speaker, probe, L)
% Structure-aware probe-IV acoustic path identification (MATLAB port).
% Inputs are OBSERVED microphone, speaker, and independent probe only.
% NEVER accepts the true feedback path or near-end target as an input.
% Python version optimized_iv.py has been tested; this MATLAB port has not
% been executed in the present container and needs MATLAB verification.

mic = double(mic(:)); speaker = double(speaker(:)); probe = double(probe(:));
N = numel(mic);
assert(numel(speaker)==N && numel(probe)==N,'Input length mismatch');
assert(N>=max(20*L,1024),'Too few calibration samples');
assert(mean(probe.^2)>1e-15,'An independent nonzero probe is required');
assert(all(isfinite(mic)) && all(isfinite(speaker)) && all(isfinite(probe)), ...
       'Nonfinite sample');

raw = solve_iv(mic,speaker,probe,L);
mid = floor(N/2);
raw1 = solve_iv(mic(1:mid),speaker(1:mid),probe(1:mid),L);
raw2 = solve_iv(mic(mid+1:end),speaker(mid+1:end),probe(mid+1:end),L);
sigma = sqrt(mean((raw1-raw2).^2))/2;
sigma_half = sqrt(2)*sigma;
soft1 = sure_soft(raw1,sigma_half);
soft2 = sure_soft(raw2,sigma_half);
soft_full = sure_soft(raw,sigma);

groups = [-1,16,20,24,28,32,36,40,48,64,L];
groups = unique(groups(groups<=L),'stable');
locals = [4,8,12,16,24];
locals = locals(locals<=L);
best_loss = Inf; best_group = -1; best_local = locals(1);
baseline_loss = Inf;
for i=1:numel(groups)
    group = groups(i);
    for j=1:numel(locals)
        local = locals(j);
        if group==-1 && j>1, continue; end
        if group==-1
            w1=soft1;w2=soft2;
        else
            w1=structured_denoise(raw1,sigma_half,group,local);
            w2=structured_denoise(raw2,sigma_half,group,local);
        end
        loss = sum((w1-raw2).^2)+sum((w2-raw1).^2);
        if group==-1, baseline_loss=loss; end
        if loss < best_loss
            best_loss=loss;best_group=group;best_local=local;
        end
    end
end
if best_group==-1
    w = soft_full;
else
    w = structured_denoise(raw,sigma,best_group,best_local);
end
result = struct('coefficients',w,'reference_coefficients',soft_full, ...
    'raw_coefficients',raw,'group_width',best_group, ...
    'local_width',best_local,'sigma',sigma,'crossfit_loss',best_loss, ...
    'baseline_crossfit_loss',baseline_loss,'samples',N);
end

function w = solve_iv(m,u,p,L)
N = numel(m);
col=zeros(L,1);row=zeros(L,1);b=zeros(L,1);
for k=0:L-1
    col(k+1)=p(1:N-k).' * u(1+k:N) / N;
    row(k+1)=p(1+k:N).' * u(1:N-k) / N;
    b(k+1)=p(1:N-k-1).' * m(k+2:N) / N;
end
G=toeplitz(col,row);
assert(isfinite(cond(G)) && cond(G)<1e6,'Poor probe IV conditioning');
w=(G + 1e-5*mean(p.^2)*eye(L)) \ b;
end

function w = sure_soft(raw,sigma)
a=abs(raw);
values=unique([0;a]);
risk=zeros(size(values));
for k=1:numel(values)
    t=values(k);
    risk(k)=numel(raw)*sigma^2+sum(min(a.^2,t^2))-2*sigma^2*sum(a<=t);
end
[~,i]=min(risk);t=values(i);
w=sign(raw).*max(a-t,0);
end

function w=structured_denoise(raw,sigma,group,local)
evidence=max(abs(raw)-sigma,0).^2;
score=conv(evidence,ones(group,1),'valid');
[~,begin]=max(score);
mask=zeros(size(raw));mask(begin:begin+group-1)=1;
power=conv(raw.^2,ones(local,1)/local,'same');
signal=max(power-sigma^2,0);
shrink=signal./(signal+sigma^2+1e-30);
w=raw.*mask.*shrink;
end
