"""Streaming V3 acoustic feedback canceller in a measured-FIR simulation.

The estimator sees only recorded microphone, actual loudspeaker and probe.
Known acoustic path and target are used by the simulated plant and scorer only.
Real hardware requires a device-specific limiter, delay calibration, and tests.
"""

from __future__ import annotations

import argparse

import numpy as np

from afc import FIRFeedbackPlant,ProbeAFC,accuracy
from demo import load_original
from simulation import highpass_target
from v3_iv import identify_v3


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--probe-rms',type=float,default=.05)
    a=p.parse_args()
    if a.seconds<=0 or a.probe_rms<=0:p.error('Use positive time and RMS')
    _,h=load_original()
    seed=20661102
    source=highpass_target(16000*a.seconds,seed)
    plant=FIRFeedbackPlant(h)
    afc=ProbeAFC(len(h),delay=1,gain=1.,probe_rms=a.probe_rms,seed=seed+800000)
    for s in source:
        microphone=plant.microphone(float(s))
        _,speaker,_=afc.process_mic(microphone)
        plant.observe_speaker(speaker)

    model=identify_v3(np.asarray(afc.microphones),np.asarray(afc.loudspeakers),
                      np.asarray(afc.probes),len(h))
    v2=accuracy(h,model.v2_coefficients)
    improved=accuracy(h,model.coefficients)
    afc.activate(model.to_identification(),fade_samples=1600)
    afc.probe_rms=0.
    afc.clear_recorded_audio()

    heldout=highpass_target(160000,seed+4000000)
    residual=[]
    for s in heldout:
        mic=plant.microphone(float(s))
        error,speaker,_=afc.process_mic(mic,record=False)
        plant.observe_speaker(speaker)
        residual.append(error-s)
    print(f'V2 ASG: {v2["asg_db"]:+.3f} dB; NFCE {v2["nfce"]:.4f}')
    print(f'V3 ASG: {improved["asg_db"]:+.3f} dB; NFCE {improved["nfce"]:.4f}')
    print(f'V3 effective candidates: {model.effective_model_count:.1f}')
    print(f'Heldout residual RMS after startup: {np.sqrt(np.mean(np.square(residual[2000:]))):.6f}')
    print('Probe OFF during heldout; path and target NOT estimator inputs.')


if __name__=='__main__':main()
