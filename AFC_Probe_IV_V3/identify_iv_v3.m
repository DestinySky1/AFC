function result = identify_iv_v3(mic, speaker, probe, L)
% V3 exact-sample instrument moments + cross-fitted structure ensemble.
% Only observed microphone, loudspeaker and independent probe are inputs.
% The true feedback path and near-end target are NOT used in identification.
% Python is tested; this MATLAB port awaits execution in MATLAB/Octave.

mic=double(mic(:));speaker=double(speaker(:));probe=double(probe(:));
N=numel(mic);
assert(numel(speaker)==N && numel(probe)==N,'Sample count mismatch');
assert(N>=max(20*L,1024),'Not enough calibration samples');
assert(mean(probe.^2)>1e-15,'Independent nonzero probe required');
assert(all(isfinite(mic)) && all(isfinite(speaker)) && all(isfinite(probe)), ...
    'Input contains nonfinite numbers');

[raw,condition]=solve_exact_iv(mic,speaker,probe,L);
mid=floor(N/2);
[r1,~]=solve_exact_iv(mic(1:mid),speaker(1:mid),probe(1:mid),L);
[r2,~]=solve_exact_iv(mic(mid+1:end),speaker(mid+1:end),probe(mid+1:end),L);
sigma=sqrt(mean((r1-r2).^2))/2;
sig_half=sqrt(2)*sigma;
soft1=sure_soft(r1,sig_half);soft2=sure_soft(r2,sig_half);
soft_full=sure_soft(raw,sigma);
groups=unique([16,20,24,28,32,36,40,48,64,L],'stable');
groups=groups(groups<=L);
locals=[4,8,12,16,24];locals=locals(locals<=L);
num=1+numel(groups)*numel(locals);
W1=zeros(L,num);W2=zeros(L,num);Wfull=zeros(L,num);
widths=zeros(1,num);local_widths=zeros(1,num);
W1(:,1)=soft1;W2(:,1)=soft2;Wfull(:,1)=soft_full;
widths(1)=-1;local_widths(1)=0;
q=1;
for i=1:numel(groups)
    for j=1:numel(locals)
        q=q+1;
        width=groups(i);local=locals(j);
        widths(q)=width;local_widths(q)=local;
        W1(:,q)=denoise(r1,sig_half,width,local);
        W2(:,q)=denoise(r2,sig_half,width,local);
        Wfull(:,q)=denoise(raw,sigma,width,local);
    end
end
loss=sum((W1-r2).^2,1)+sum((W2-r1).^2,1);
temperature=3*sqrt(2*L)*sig_half^2;
prob=exp(-(loss-min(loss))/(2*temperature+1e-30));
prob=prob/sum(prob);
w=Wfull*prob(:);
[~,idx]=min(loss);
result=struct('coefficients',w,'raw_coefficients',raw, ...
    'sigma',sigma,'samples',N,'probe_rms',sqrt(mean(probe.^2)), ...
    'moment_condition',condition,'best_group_width',widths(idx), ...
    'best_local_width',local_widths(idx), ...
    'effective_model_count',1/sum(prob.^2),'temperature_factor',3);
end

function [w,condition]=solve_exact_iv(m,u,p,L)
N=numel(m);
G=zeros(L,L);
for k=0:L-1
    prod=p(1:N-k).*u(1+k:N);
    total=sum(prod);
    prefix=[0;cumsum(prod(1:L))];
    suffix=[0;cumsum(flipud(prod(end-L+1:end)))];
    for j=0:L-1-k
        i=j+k;
        lo=L-1-i;hi=i+1-k;
        G(i+1,j+1)=total-prefix(lo+1)-suffix(hi+1);
    end
    if k>0
        prod=p(1+k:N).*u(1:N-k);
        total=sum(prod);
        prefix=[0;cumsum(prod(1:L))];
        suffix=[0;cumsum(flipud(prod(end-L+1:end)))];
        for i=0:L-1-k
            j=i+k;
            lo=L-1-j;hi=i+1;
            G(i+1,j+1)=total-prefix(lo+1)-suffix(hi+1);
        end
    end
end
b=zeros(L,1);
for i=0:L-1
    prod=p(1:N-i-1).*m(i+2:N);
    lo=L-1-i;
    b(i+1)=sum(prod)-sum(prod(1:lo));
end
G=G/(N-L);b=b/(N-L);
condition=cond(G);
assert(isfinite(condition) && condition<1e6,'Poorly conditioned probe moments');
w=(G+1e-5*mean(p.^2)*eye(L))\b;
end

function w=sure_soft(raw,sigma)
a=abs(raw);choices=unique([0;a]);risk=zeros(numel(choices),1);
for k=1:numel(choices)
    t=choices(k);
    risk(k)=numel(raw)*sigma^2+sum(min(a.^2,t^2)) ...
            -2*sigma^2*sum(a<=t);
end
[~,idx]=min(risk);t=choices(idx);
w=sign(raw).*max(a-t,0);
end

function w=denoise(raw,sigma,group,local)
evidence=max(abs(raw)-sigma,0).^2;
score=conv(evidence,ones(group,1),'valid');
[~,begin]=max(score);
mask=zeros(size(raw));mask(begin:begin+group-1)=1;
full_power=conv(raw.^2,ones(local,1)/local,'full');
start=floor((local-1)/2)+1; % Match NumPy's 'same' centering for even windows.
power=full_power(start:start+numel(raw)-1);
signal=max(power-sigma^2,0);
shrink=signal./(signal+sigma^2+1e-30);
w=raw.*mask.*shrink;
end
