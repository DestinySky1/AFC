"""Independent, source-blind numerical checks for V3 IV identification."""

import unittest

import numpy as np

from afc import FIRFeedbackPlant, ProbeAFC
from v3_iv import _exact_iv_moments, identify_v3


class V3IVTest(unittest.TestCase):
    def test_exact_moments_match_valid_row_matrix_product(self):
        random = np.random.default_rng(123)
        n, taps = 301, 12
        mic = random.normal(size=n)
        loudspeaker = random.normal(size=n)
        probe = random.normal(size=n)
        g, b = _exact_iv_moments(mic,loudspeaker,probe,taps)
        z = np.stack([probe[taps-1-i:n-1-i] for i in range(taps)],axis=1)
        x = np.stack([loudspeaker[taps-1-i:n-1-i] for i in range(taps)],axis=1)
        self.assertTrue(np.allclose(g,z.T@x/(n-taps),atol=1e-13))
        self.assertTrue(np.allclose(b,z.T@mic[taps:]/(n-taps),atol=1e-13))

    def test_v3_accepts_unknown_path_delay_and_activates(self):
        rng = np.random.default_rng(42)
        taps, n = 32, 16000
        h = np.zeros(taps)
        h[15:21] = [0.025,-0.035,0.02,0.01,-0.005,0.003]
        afc = ProbeAFC(taps=taps,probe_rms=.1,seed=97)
        plant = FIRFeedbackPlant(h)
        source = rng.normal(0,.03,n)
        for k in range(n):
            mic = plant.microphone(source[k])
            _,speaker,_=afc.process_mic(mic)
            plant.observe_speaker(speaker)
        fit = identify_v3(np.array(afc.microphones),np.array(afc.loudspeakers),
                          np.array(afc.probes),taps,include_v2=False)
        self.assertEqual(fit.coefficients.shape,(taps,))
        self.assertTrue(np.isfinite(fit.coefficients).all())
        self.assertGreater(fit.effective_model_count,0)
        afc.activate(fit.to_identification(),fade_samples=100)
        self.assertTrue(np.isfinite(afc.weights).all())

    def test_zero_probe_rejected(self):
        n,taps=2000,32
        mic=np.zeros(n)
        speaker=np.zeros(n)
        with self.assertRaises(ValueError):
            identify_v3(mic,speaker,np.zeros(n),taps)

    def test_invalid_temperature_rejected(self):
        n,taps=2000,32
        with self.assertRaises(ValueError):
            identify_v3(np.zeros(n),np.zeros(n),np.ones(n),taps,temperature_factor=0)


if __name__=='__main__':
    unittest.main()
