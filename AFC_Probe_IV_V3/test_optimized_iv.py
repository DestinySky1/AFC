"""Validation for the new independent, cross-fitted structure-aware AFC."""

from __future__ import annotations

import unittest

import numpy as np

from afc import ProbeAFC, accuracy
from optimized_iv import identify_structured, structured_denoise
from simulation import frozen_closed_loop, highpass_target


class StructuredIVTest(unittest.TestCase):
    def test_localizes_structure_at_an_unseen_delay(self):
        rng=np.random.default_rng(400)
        h=np.zeros(40)
        h[20:29]=[.005,-.014,.019,.012,-.008,.013,-.009,.007,-.004]
        source=highpass_target(64000, 813)
        probe=(2*rng.integers(0,2,len(source))-1)*.14
        mic,u,_=frozen_closed_loop(source,h,probe=probe)
        model=identify_structured(mic,u,probe,len(h))
        self.assertGreater(accuracy(h,model.coefficients)['asg_db'],0)
        self.assertLess(accuracy(h,model.coefficients)['nfce'],1)

    def test_updates_when_path_changes_while_filter_active(self):
        rng=np.random.default_rng(67)
        original=np.zeros(24)
        original[4:10]=[.018,-.012,.02,-.007,.005,-.009]
        changed=np.roll(original,4)*1.2
        source=highpass_target(80000,402)
        probe=(2*rng.integers(0,2,len(source))-1)*.1
        mic,u,_=frozen_closed_loop(source,changed,original,probe)
        model=identify_structured(mic,u,probe,len(original))
        self.assertGreater(accuracy(changed,model.coefficients)['asg_db'],0)
        afc=ProbeAFC(len(original))
        afc.activate(model.to_identification(),fade_samples=100)
        for _ in range(100):afc.process_mic(0.0,probe=0.0,record=False)
        np.testing.assert_allclose(afc.weights,model.coefficients,atol=1e-15)

    def test_no_probe_is_rejected(self):
        with self.assertRaises(ValueError):
            identify_structured(np.zeros(5000),np.zeros(5000),np.zeros(5000),24)

    def test_denoiser_has_no_true_path_argument(self):
        noisy=np.array([.01,.005,.002,-.009,.004,.00001,0,0,0,0])
        modeled=structured_denoise(noisy,.001,group_width=5,local_width=3)
        self.assertEqual(modeled.shape,noisy.shape)
        self.assertTrue(np.all(np.isfinite(modeled)))


if __name__=='__main__':unittest.main()
