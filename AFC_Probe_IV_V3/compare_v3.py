"""Fair paired V2/V3 experiments using identical simulated recordings.

All path and target ground truth is restricted to the simulation and scoring.
The V3 and V2 estimators see the same mic, speaker, and independent probe.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from afc import accuracy
from demo import load_original
from simulation import frozen_closed_loop,highpass_target
from compare_optimized import echo_reduction
from v3_iv import identify_v3


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--probe-rms',type=float,default=.05)
    p.add_argument('--trials',type=int,default=20)
    p.add_argument('--seed',type=int,default=20500100)
    p.add_argument('--temperature-factor',type=float,default=3.)
    p.add_argument('--use-approx-moments',action='store_true')
    p.add_argument('--enable-crossfit-gain',action='store_true',
                   help='Exploratory option; worsened performance on development seeds')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results_v3')
    a=p.parse_args()
    if a.seconds < 1 or a.trials < 1 or a.probe_rms <= 0 or a.temperature_factor <= 0:
        p.error('Choose positive seconds, trials, probe-rms and temperature')
    _,h=load_original()
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    first=None
    for j in range(a.trials):
        seed=a.seed+j
        source=highpass_target(a.seconds*16000,seed)
        probe=(2*np.random.default_rng(seed+800000).integers(0,2,len(source))-1)*a.probe_rms
        mic,u,_=frozen_closed_loop(source,h,probe=probe)
        fit=identify_v3(mic,u,probe,len(h),temperature_factor=a.temperature_factor,
                        use_exact=not a.use_approx_moments,
                        crossfit_gain=a.enable_crossfit_gain)
        base=accuracy(h,fit.v2_coefficients)
        v3=accuracy(h,fit.coefficients)
        heldout=highpass_target(160000,seed+4000000)
        row={
            'trial':j+1,'seed':seed,
            'v2_nfce':base['nfce'],'v3_nfce':v3['nfce'],
            'v2_asg_db':base['asg_db'],'v3_asg_db':v3['asg_db'],
            'asg_improvement_db':v3['asg_db']-base['asg_db'],
            'v2_heldout_echo_db':echo_reduction(heldout,h,fit.v2_coefficients),
            'v3_heldout_echo_db':echo_reduction(heldout,h,fit.coefficients),
            'effective_model_count':fit.effective_model_count,
            'crossfit_gain':fit.crossfit_gain,
            'best_group_width':fit.best_group_width,
            'best_local_width':fit.best_local_width,
            'moment_condition':fit.moment_condition,
            'estimated_coefficient_noise':fit.sigma_full,
            'calibration_speaker_peak':float(np.max(np.abs(u))),
        }
        rows.append(row)
        if first is None:first=(fit.v2_coefficients,fit.coefficients)
        print(f"Trial {j+1:2d}: V2 {base['asg_db']:+.2f} -> "
              f"V3 {v3['asg_db']:+.2f} dB (delta {row['asg_improvement_db']:+.2f}); "
              f"models~{fit.effective_model_count:.1f} gain={fit.crossfit_gain:.2f}",flush=True)
    with (a.output/'per_trial.csv').open('w',newline='',encoding='utf8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    get=lambda key:np.asarray([r[key] for r in rows],dtype=float)
    delta=get('asg_improvement_db')
    sd=float(np.std(delta,ddof=1)) if a.trials>1 else float('nan')
    # 1.96 is asymptotic, not an exact small-sample confidence interval.
    ci=[float(np.mean(delta)-1.96*sd/np.sqrt(a.trials)),
        float(np.mean(delta)+1.96*sd/np.sqrt(a.trials))]
    summary={
        'seconds':a.seconds,'probe_rms':a.probe_rms,'trials':a.trials,
        'initial_seed':a.seed,'temperature_factor':a.temperature_factor,
        'use_exact_moments':not a.use_approx_moments,
        'crossfit_gain_calibration':a.enable_crossfit_gain,
        'v2_mean_nfce':float(np.mean(get('v2_nfce'))),
        'v3_mean_nfce':float(np.mean(get('v3_nfce'))),
        'v2_mean_asg_db':float(np.mean(get('v2_asg_db'))),
        'v3_mean_asg_db':float(np.mean(get('v3_asg_db'))),
        'v3_min_asg_db':float(np.min(get('v3_asg_db'))),
        'mean_asg_improvement_db':float(np.mean(delta)),
        'asg_delta_approx_95ci':ci,
        'v3_vs_v2_positive_improvement_count':int(np.count_nonzero(delta>0)),
        'v3_positive_asg_count':int(np.count_nonzero(get('v3_asg_db')>0)),
        'v3_asg_over_10_count':int(np.count_nonzero(get('v3_asg_db')>=10)),
        'v2_heldout_echo_mean_db':float(np.mean(get('v2_heldout_echo_db'))),
        'v3_heldout_echo_mean_db':float(np.mean(get('v3_heldout_echo_db'))),
        'limit':'Single measured-FIR simulated feedback path, synthetic colored source, no hardware assessment',
    }
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    if first is not None:
        np.save(a.output/'v2_path_trial01.npy',first[0])
        np.save(a.output/'v3_path_trial01.npy',first[1])
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    t=np.arange(1,a.trials+1)
    ax[0].plot(t,get('v2_asg_db'),'s--',label='V2 structured IV')
    ax[0].plot(t,get('v3_asg_db'),'o-',label='V3 exact / IV ensemble')
    ax[0].axhline(10,color='gray',linestyle=':',label='10 dB target')
    ax[0].set(xlabel='Trial',ylabel='ASG / dB',title='Paired closed-loop ASG')
    ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    ax[1].plot(h,color='black',label='Measured path')
    ax[1].plot(first[0],label='V2',alpha=.8)
    ax[1].plot(first[1],label='V3',alpha=.8)
    ax[1].set(xlabel='Tap',title='First estimated path')
    ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=150);plt.close(fig)
    print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
