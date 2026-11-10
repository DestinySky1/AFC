"""Independent multi-seed closed-loop evaluation of cross-fitted IV-AFC.

The same microphone/speaker/probe observation is passed to both estimators.
The true path and target signal are known to the SIMULATOR and SCORER only.
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
from optimized_iv import identify_structured
from simulation import frozen_closed_loop, highpass_target


def echo_reduction(source: np.ndarray, h: np.ndarray, w: np.ndarray) -> float:
    _, _, original = frozen_closed_loop(source, h)
    _, _, adapted = frozen_closed_loop(source, h, w)
    skip = max(2000, 3 * len(h))
    before = original[skip:] - source[skip:]
    after = adapted[skip:] - source[skip:]
    return float(20 * np.log10(np.sqrt(np.mean(before ** 2)) /
                               np.sqrt(np.mean(after ** 2))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=int, default=60)
    p.add_argument('--probe-rms', type=float, default=0.05)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--seed', type=int, default=20421102,
                   help='Independent from exploratory development seeds 20261102+')
    p.add_argument('--allow-gain-search', action='store_true',
                   help='Cross-fit an additional, source-blind global coefficient gain')
    p.add_argument('--output', type=Path,
                   default=Path(__file__).resolve().parent / 'results_structured_cv')
    args = p.parse_args()
    if not (1 <= args.seconds <= 600 and args.trials > 0 and args.probe_rms > 0):
        p.error('Use positive seconds/trials/probe RMS')
    _, h = load_original()
    n = args.seconds * 16000
    args.output.mkdir(parents=True, exist_ok=True)
    rows=[]
    first_weights=None
    for trial in range(args.trials):
        seed=args.seed+trial
        target=highpass_target(n,seed)
        probe=(2*np.random.default_rng(seed+800000).integers(0,2,n)-1)*args.probe_rms
        mic,u,_=frozen_closed_loop(target,h,probe=probe)
        gain_grid=(.7,.85,1.0,1.15,1.3) if args.allow_gain_search else (1.0,)
        fit=identify_structured(mic,u,probe,len(h),candidate_gains=gain_grid)
        base=accuracy(h,fit.reference.coefficients)
        improved=accuracy(h,fit.coefficients)
        heldout=highpass_target(160000,seed+4000000)
        row={
            'trial':trial+1,
            'seed':seed,
            'baseline_nfce':base['nfce'],
            'optimized_nfce':improved['nfce'],
            'baseline_asg_db':base['asg_db'],
            'optimized_asg_db':improved['asg_db'],
            'asg_improvement_db':improved['asg_db']-base['asg_db'],
            'baseline_heldout_echo_reduction_db':echo_reduction(heldout,h,fit.reference.coefficients),
            'optimized_heldout_echo_reduction_db':echo_reduction(heldout,h,fit.coefficients),
            'selected_group_width':fit.group_width,
            'selected_local_width':fit.local_width,
            'selected_gain':fit.gain,
            'cv_loss':fit.crossfit_loss,
            'baseline_cv_loss':fit.baseline_crossfit_loss,
            'target_rms':float(np.sqrt(np.mean(target**2))),
            'probe_rms':float(np.sqrt(np.mean(probe**2))),
            'training_speaker_peak':float(np.max(abs(u))),
        }
        rows.append(row)
        if first_weights is None:
            first_weights=(fit.reference.coefficients,fit.coefficients)
        print(f"Trial {trial+1:2d}: baseline ASG {base['asg_db']:+.2f} "
              f"-> optimized {improved['asg_db']:+.2f} dB; "
              f"CV group={fit.group_width}, local={fit.local_width}, gain={fit.gain}",flush=True)
    with (args.output/'per_trial.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary={
        'seconds':args.seconds,'requested_probe_rms':args.probe_rms,
        'trials':args.trials,'initial_seed':args.seed,
        'baseline_mean_nfce':float(np.mean([r['baseline_nfce'] for r in rows])),
        'optimized_mean_nfce':float(np.mean([r['optimized_nfce'] for r in rows])),
        'baseline_mean_asg_db':float(np.mean([r['baseline_asg_db'] for r in rows])),
        'optimized_mean_asg_db':float(np.mean([r['optimized_asg_db'] for r in rows])),
        'optimized_min_asg_db':float(np.min([r['optimized_asg_db'] for r in rows])),
        'mean_asg_improvement_db':float(np.mean([r['asg_improvement_db'] for r in rows])),
        'optimized_positive_asg_trials':int(sum(r['optimized_asg_db']>0 for r in rows)),
        'optimized_asg_above_10_trials':int(sum(r['optimized_asg_db']>=10 for r in rows)),
        'optimized_nfce_below_0_1_trials':int(sum(r['optimized_nfce']<=0.1 for r in rows)),
        'baseline_heldout_mean_echo_db':float(np.mean([r['baseline_heldout_echo_reduction_db'] for r in rows])),
        'optimized_heldout_mean_echo_db':float(np.mean([r['optimized_heldout_echo_reduction_db'] for r in rows])),
        'selection_uses_true_path':False,
        'test_limit':'Closed-loop measured-FIR simulation, not an audio-device trial',
    }
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    assert first_weights is not None
    np.save(args.output/'reference_path_trial01.npy',first_weights[0]);
    np.save(args.output/'optimized_path_trial01.npy',first_weights[1]);
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.8))
    ix=np.arange(1,args.trials+1)
    ax[0].plot(ix,[r['baseline_asg_db'] for r in rows],'.--',label='Original probe-IV + SURE')
    ax[0].plot(ix,[r['optimized_asg_db'] for r in rows],'o-',label='Cross-fitted structured IV')
    ax[0].axhline(10,color='gray',linestyle=':',label='10 dB research target')
    ax[0].set_xlabel('Independent trial');ax[0].set_ylabel('ASG / dB')
    ax[0].set_title('Paired closed-loop ASG');ax[0].grid(alpha=.2);ax[0].legend(fontsize=7)
    ax[1].plot(np.arange(len(h)),h,color='black',label='Measured path')
    ax[1].plot(np.arange(len(h)),first_weights[0],label='Reference estimate',alpha=.7)
    ax[1].plot(np.arange(len(h)),first_weights[1],label='Structured estimate',alpha=.8)
    ax[1].set_xlabel('Tap (0-based)');ax[1].set_title('Estimated FIR, trial 1')
    ax[1].grid(alpha=.2);ax[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)
    print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
