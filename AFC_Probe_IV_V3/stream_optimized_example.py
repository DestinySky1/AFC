"""Full causal sample-by-sample training, activation and closed-loop playback.

This is a measured-FIR simulator, NOT an actual audio device driver.
"""

from __future__ import annotations

import argparse
import numpy as np

from afc import FIRFeedbackPlant, ProbeAFC, accuracy
from demo import load_original
from optimized_iv import identify_structured
from simulation import highpass_target


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=int,default=60)
    parser.add_argument('--probe-rms',type=float,default=.05)
    args=parser.parse_args()
    _,h=load_original()
    seed=20421102
    source=highpass_target(16000*args.seconds,seed)
    plant=FIRFeedbackPlant(h)
    afc=ProbeAFC(len(h),delay=1,gain=1.,probe_rms=args.probe_rms,
                 seed=seed+800000)
    for s in source:
        mic=plant.microphone(float(s))
        _,u,_=afc.process_mic(mic)
        plant.observe_speaker(u)
    model=identify_structured(afc.microphones,afc.loudspeakers,afc.probes,len(h))
    base=accuracy(h,model.reference.coefficients)
    enhanced=accuracy(h,model.coefficients)
    afc.activate(model.to_identification(),fade_samples=1600)
    afc.probe_rms=0.0
    afc.clear_recorded_audio()
    heldout=highpass_target(160000,seed+4000000)
    echo=[]
    for s in heldout:
        mic=plant.microphone(float(s))
        err,u,_=afc.process_mic(mic,record=False)
        plant.observe_speaker(u)
        echo.append(err-s)
    print(f"Baseline IV + SURE ASG: {base['asg_db']:+.3f} dB")
    print(f"Structured IV ASG: {enhanced['asg_db']:+.3f} dB; "
          f"NFCE: {enhanced['nfce']:.4f}")
    print(f"Crossfit chose group={model.group_width},local={model.local_width}")
    print(f"Heldout error-vs-target RMS after startup: "
          f"{np.sqrt(np.mean(np.square(echo[2000:]))):.6f}")
    print('Probe OFF during heldout. No path or target used by estimator.')


if __name__=='__main__':main()
