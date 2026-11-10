"""Run a reproducible closed-loop AFC experiment against a measured FIR path.

The estimation code sees only microphone, speaker, and injected probe. The
known path and isolated source are used solely by the simulator and scoring.
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
from scipy.io import loadmat

from afc import accuracy, identify_path
from simulation import frozen_closed_loop, highpass_target


ROOT = Path(__file__).resolve().parent
LOCAL_INPUT = ROOT / 'data/afc_inputs_10s.mat'
ARCHIVE_INPUT = (ROOT.parent / 'nlms_review_6jOwn7/NLMS_Diagnosis_Bundle'
                 / 'data/afc_inputs_10s.mat')


def load_original(file: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if file is None:
        file = LOCAL_INPUT if LOCAL_INPUT.is_file() else ARCHIVE_INPUT
    if not file.is_file():
        raise FileNotFoundError('A MATLAB afc_inputs_10s.mat file is required; use --input')
    d = loadmat(file)
    return d['vTarget'].ravel(), d['f_path_bes'].ravel()


def evaluate(source: np.ndarray, h: np.ndarray, probe_rms: float,
             seed: int, *, heldout_source: np.ndarray,
             probe_kind: str = 'binary') -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    if probe_kind == 'binary':
        probe = (2*rng.integers(0,2,len(source))-1)*probe_rms
    elif probe_kind == 'gaussian':
        probe = rng.normal(0, probe_rms, len(source))
    else:
        raise ValueError('Unknown probe kind')
    mic, speaker, _ = frozen_closed_loop(source, h, probe=probe)
    fitted = identify_path(mic, speaker, probe, len(h))
    metrics = accuracy(h, fitted.coefficients)
    raw = accuracy(h, fitted.raw_coefficients)
    _, u0, err0 = frozen_closed_loop(heldout_source, h)
    _, u1, err1 = frozen_closed_loop(heldout_source, h,
                                     coefficients=fitted.coefficients)
    skip = max(2000, len(h) * 3)
    echo_before = err0[skip:] - heldout_source[skip:]
    echo_after = err1[skip:] - heldout_source[skip:]
    before_rms = np.sqrt(np.mean(echo_before**2))
    after_rms = np.sqrt(np.mean(echo_after**2))
    metrics.update({
        'nfce_raw': raw['nfce'], 'asg_raw_db': raw['asg_db'],
        'heldout_echo_suppression_db': float(20 * np.log10(before_rms / after_rms)),
        'probe_rms': float(np.sqrt(np.mean(probe**2))),
        'source_rms': float(np.sqrt(np.mean(source**2))),
        'calibration_peak_speaker': float(np.max(abs(speaker))),
        'heldout_max_speaker_before': float(np.max(abs(u0))),
        'heldout_max_speaker_after': float(np.max(abs(u1))),
        'moment_condition': fitted.moment_condition,
        'estimated_noise_per_tap': fitted.noise_std_per_tap,
        'soft_threshold': fitted.shrinkage_threshold,
    })
    return metrics, fitted.coefficients


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=None,
                   help='Original MATLAB 10-second input/path .mat')
    p.add_argument('--calibration-seconds', type=int, default=10)
    p.add_argument('--probe-rms', type=float, default=0.2)
    p.add_argument('--probe-kind', choices=['binary','gaussian'], default='binary')
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    args = p.parse_args()
    if not (1 <= args.calibration_seconds <= 600) or args.trials < 1 or args.probe_rms <= 0:
        p.error('Choose positive probe RMS, 1-600 calibration seconds and >=1 trials')
    original, h = load_original(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    n = 16000 * args.calibration_seconds
    results = []
    reference_coefficients = None
    for trial in range(args.trials):
        # The first 10-second experiment uses the exactly archived MATLAB
        # colored-noise target. Other records use independent innovations.
        source = (original if trial == 0 and n == len(original) else
                  highpass_target(n, args.seed + trial + 123))
        heldout = highpass_target(160000, args.seed + trial + 10000)
        metrics, weights = evaluate(source, h, args.probe_rms, args.seed + trial,
                                    heldout_source=heldout,
                                    probe_kind=args.probe_kind)
        metrics['trial'] = trial + 1
        metrics['source'] = 'exact MATLAB 10s' if trial == 0 and n == len(original) else 'independent colored noise'
        results.append(metrics)
        if trial == 0:
            reference_coefficients = weights
        print(f"Trial {trial+1:2}: NFCE={metrics['nfce']:.4f} "
              f"ASG={metrics['asg_db']:+.2f} dB "
              f"heldout_echo_suppression={metrics['heldout_echo_suppression_db']:+.2f} dB",
              flush=True)
    assert reference_coefficients is not None
    with (args.output / 'per_trial.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    aggregate = {
        'calibration_seconds': args.calibration_seconds,
        'probe_rms_requested': args.probe_rms,
        'probe_kind': args.probe_kind,
        'trials': args.trials,
        'true_path_taps': len(h),
        'nfce_mean': float(np.mean([r['nfce'] for r in results])),
        'asg_db_mean': float(np.mean([r['asg_db'] for r in results])),
        'asg_db_min': float(np.min([r['asg_db'] for r in results])),
        'heldout_echo_suppression_mean_db': float(np.mean([r['heldout_echo_suppression_db'] for r in results])),
        'positive_asg_trials': int(sum(r['asg_db'] > 0 for r in results)),
        'nfce_below_1_trials': int(sum(r['nfce'] < 1 for r in results)),
        'important_limit': 'Simulated measured FIR; hardware/acoustic calibration not performed',
    }
    (args.output / 'summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    np.save(args.output / 'estimated_path_trial01.npy', reference_coefficients)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].plot(np.arange(len(h)), h, label='Real measured path', linewidth=2)
    axs[0].plot(np.arange(len(h)), reference_coefficients, label='Estimated via probe IV', linewidth=1)
    axs[0].set_xlabel('FIR tap (0-based)')
    axs[0].set_title('Acoustic path reconstruction (trial 1)')
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=.3)
    axs[1].scatter([r['nfce'] for r in results], [r['asg_db'] for r in results], s=32)
    axs[1].axvline(1.0, color='gray', linestyle='--', linewidth=1)
    axs[1].axhline(0.0, color='gray', linestyle='--', linewidth=1)
    axs[1].set_xlabel('Final NFCE (lower is better)')
    axs[1].set_ylabel('ASG / dB (higher is better)')
    axs[1].set_title(f'Probe RMS={args.probe_rms:g}; {args.trials} trials')
    axs[1].grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(args.output / 'afc_validation.png', dpi=160)
    plt.close(fig)
    print('\nSUMMARY', json.dumps(aggregate, ensure_ascii=False))


if __name__ == '__main__':
    main()
