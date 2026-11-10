"""Closed-loop reference simulator, isolated from the real-time estimator."""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter


def highpass_target(n_samples: int, seed: int, source_std: float = 0.1) -> np.ndarray:
    """Synthetic source with MATLAB fir1(64,0.1,'high')-like coloration."""
    rng = np.random.default_rng(seed)
    b = firwin(65, 0.1, pass_zero="highpass", window="hamming", fs=2)
    return lfilter(b, [1.0], rng.normal(0.0, source_std, n_samples))


def frozen_closed_loop(source: np.ndarray, true_path: np.ndarray,
                       coefficients: np.ndarray | None = None,
                       probe: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast exact D=1, gain=1 simulation; returns (mic, speaker, error).

    The speaker injects probe[n] after the one-sample delayed error. The
    acoustic feedback uses one additional speaker-history sample delay.
    """
    source = np.asarray(source, dtype=np.float64).ravel()
    true_path = np.asarray(true_path, dtype=np.float64).ravel()
    if coefficients is None:
        coefficients = np.zeros_like(true_path)
    else:
        coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
    if probe is None:
        probe = np.zeros_like(source)
    else:
        probe = np.asarray(probe, dtype=np.float64).ravel()
    if len(source) != len(probe) or len(true_path) != len(coefficients):
        raise ValueError("Source/probe or true-path/estimate sizes do not match")
    residual = true_path - coefficients
    probe_feedback = lfilter(np.r_[0.0, residual], [1.0], probe)
    error = lfilter([1.0], np.r_[1.0, 0.0, -residual], source + probe_feedback)
    speaker = np.r_[0.0, error[:-1]] + probe
    mic = source + lfilter(np.r_[0.0, true_path], [1.0], speaker)
    return mic, speaker, error
