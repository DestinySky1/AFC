"""Run with: python -m unittest discover -s afc_effective -p 'test_*.py' -v"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from afc import FIRFeedbackPlant, ProbeAFC, accuracy, identify_path
from simulation import frozen_closed_loop, highpass_target


ROOT = Path(__file__).resolve().parent
ARCHIVE_INPUT = (ROOT.parent / "nlms_review_6jOwn7/NLMS_Diagnosis_Bundle"
                 / "data/afc_inputs_10s.mat")
LOCAL_INPUT = ROOT / "data/afc_inputs_10s.mat"


class ProbeAFCTest(unittest.TestCase):
    def test_sample_exact_against_vectorized_closed_loop(self):
        rng = np.random.default_rng(33)
        path = np.array([.018, -.012, .005, .009])
        weights = np.array([.010, -.004, .007, 0])
        source = highpass_target(500, 44)
        probe = rng.normal(0, .025, len(source))
        mic, speaker, error = frozen_closed_loop(source, path, weights, probe)
        canceller = ProbeAFC(len(path), delay=1, gain=1.0)
        canceller.weights = weights.copy()
        plant = FIRFeedbackPlant(path)
        em = np.zeros_like(source)
        um = np.zeros_like(source)
        mm = np.zeros_like(source)
        for n, s in enumerate(source):
            mm[n] = plant.microphone(s)
            em[n], um[n], _ = canceller.process_mic(mm[n], probe=float(probe[n]))
            plant.observe_speaker(um[n])
        self.assertLess(np.max(abs(mic - mm)), 1e-13)
        self.assertLess(np.max(abs(error - em)), 1e-13)
        self.assertLess(np.max(abs(speaker - um)), 1e-13)

    def test_perfect_path_makes_error_equal_source_when_probe_off(self):
        rng = np.random.default_rng(1)
        path = rng.normal(0, .01, 12)
        s = highpass_target(4000, 34)
        _, _, err = frozen_closed_loop(s, path, coefficients=path)
        self.assertEqual(np.max(np.abs(s - err)), 0)
        score = accuracy(path, path)
        self.assertEqual(score['nfce'], 0.0)

    def test_reject_no_instrument_or_wrong_alignment(self):
        u = np.ones(3000)
        with self.assertRaises(ValueError):
            identify_path(u, u, np.zeros_like(u), 16)
        with self.assertRaises(ValueError):
            identify_path(u[:-1], u, u, 16)

    def test_identifies_other_feedback_path_with_colored_source(self):
        rng = np.random.default_rng(19)
        h = np.zeros(24)
        h[4:10] = [.018, -.012, .02, -.007, .005, -.009]
        s = highpass_target(48000, 107)
        probe = rng.normal(0, .2, len(s))
        mic, u, _ = frozen_closed_loop(s, h, probe=probe)
        fit = identify_path(mic, u, probe, len(h))
        self.assertLess(accuracy(h, fit.coefficients)['nfce'], 0.5)
        self.assertGreater(accuracy(h, fit.coefficients)['asg_db'], 0.0)

    def test_reidentifies_path_after_closed_loop_path_change(self):
        rng = np.random.default_rng(55)
        old = np.zeros(24)
        old[4:10] = [.018, -.012, .02, -.007, .005, -.009]
        changed = np.roll(old, 3) * 1.2
        target = highpass_target(80000, 402)
        probe = (2 * rng.integers(0, 2, len(target)) - 1) * .1
        # The old cancellation filter remains in the ACTIVE loop while the
        # new path is identified; the estimator is not told either path.
        mic, u, _ = frozen_closed_loop(target, changed, old, probe)
        fitted = identify_path(mic, u, probe, len(old))
        self.assertLess(accuracy(changed, fitted.coefficients)['nfce'], 0.5)
        self.assertGreater(accuracy(changed, fitted.coefficients)['asg_db'], 0.0)

    @unittest.skipUnless(ARCHIVE_INPUT.exists() or LOCAL_INPUT.exists(),
                         "Original MATLAB input missing")
    def test_measured_path_10s_closed_loop(self):
        path = LOCAL_INPUT if LOCAL_INPUT.exists() else ARCHIVE_INPUT
        data = loadmat(path)
        h = data['f_path_bes'].ravel()
        s = data['vTarget'].ravel()
        rng = np.random.default_rng(20261102)
        probe = rng.normal(0, .2, len(s))
        mic, u, _ = frozen_closed_loop(s, h, probe=probe)
        estimate = identify_path(mic, u, probe, len(h))
        measured = accuracy(h, estimate.coefficients)
        self.assertLess(measured['nfce'], 0.5)
        self.assertGreater(measured['asg_db'], 3.0)


if __name__ == "__main__":
    unittest.main()
