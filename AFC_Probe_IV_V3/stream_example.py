"""End-to-end SAMPLE-BY-SAMPLE closed-loop AFC demo (no hardware required).

In a real audio application replace FIRFeedbackPlant.microphone/observe_speaker
with synchronous microphone capture and speaker playback in the audio driver.
"""

from __future__ import annotations

import numpy as np

from afc import FIRFeedbackPlant, ProbeAFC, accuracy
from demo import load_original
from simulation import highpass_target


def run():
    source, true_path = load_original()
    plant = FIRFeedbackPlant(true_path)
    afc = ProbeAFC(len(true_path), delay=1, gain=1.0,
                   probe_rms=0.1, seed=20261102)
    for s in source:
        mic = plant.microphone(float(s))
        _, speaker, _ = afc.process_mic(mic)
        plant.observe_speaker(speaker)

    estimation = afc.identify()
    afc.activate(estimation, fade_samples=1600)
    afc.probe_rms = 0.0
    afc.clear_recorded_audio()

    heldout = highpass_target(160000, 20271102)
    echo = []
    loudspeaker_peak = 0.0
    for s in heldout:
        mic = plant.microphone(float(s))
        err, speaker, _ = afc.process_mic(mic, record=False)
        plant.observe_speaker(speaker)
        echo.append(err - s)
        loudspeaker_peak = max(loudspeaker_peak, abs(speaker))

    metrics = accuracy(true_path, estimation.coefficients)
    print(f"Estimated measured path: NFCE={metrics['nfce']:.4f}, "
          f"ASG={metrics['asg_db']:+.3f} dB")
    print(f"Held-out closed-loop error-vs-source RMS after transient: "
          f"{np.sqrt(np.mean(np.square(echo[2000:]))):.6f}")
    print(f"Maximum held-out speaker sample: {loudspeaker_peak:.5f}")
    print('Probe disabled after calibration; no true path or source used by estimator.')


if __name__ == '__main__':
    run()
