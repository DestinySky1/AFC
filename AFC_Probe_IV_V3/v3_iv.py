"""V3 experimental IV AFC: exact finite-record moments and model averaging.

Only observed microphone, actual loudspeaker, and independent probe are inputs.
The true feedback FIR / isolated near-end signal are never visible to this code.
V2 is preserved as an explicit fall-back and control arm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from afc import IdentificationResult, _validate_series, _sure_soft_threshold
from optimized_iv import identify_structured, structured_denoise


def _exact_iv_moments(mic: np.ndarray, loudspeaker: np.ndarray,
                      probe: np.ndarray, taps: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact Z'X and Z'm over fully valid FIR rows, without NxT arrays.

    Each valid row has response mic[n], regressors u[n-1-i], and instruments
    probe[n-1-i], with n=taps..N-1 and i=0..taps-1. The Toeplitz bulk
    correlations are corrected by small, *exact* prefix/suffix terms.
    """
    n = len(mic)
    g = np.empty((taps, taps), dtype=np.float64)
    target = np.empty(taps, dtype=np.float64)
    inv = 1.0 / (n - taps)

    for k in range(taps):
        # Lower triangle i=j+k: z[n-1-i]*u[n-1-j].
        products = probe[:n-k] * loudspeaker[k:]
        total = float(np.sum(products))
        start = np.r_[0.0, np.cumsum(products[:taps])]
        finish = np.r_[0.0, np.cumsum(products[-taps:][::-1])]
        for j in range(taps - k):
            i = j + k
            low_trim = taps - 1 - i
            high_trim = i + 1 - k
            g[i, j] = (total - start[low_trim] - finish[high_trim]) * inv
        if k:
            # Upper triangle i<j: z[n-1-i]*u[n-1-(i+k)].
            products = probe[k:] * loudspeaker[:n-k]
            total = float(np.sum(products))
            start = np.r_[0.0, np.cumsum(products[:taps])]
            finish = np.r_[0.0, np.cumsum(products[-taps:][::-1])]
            for i in range(taps - k):
                j = i + k
                low_trim = taps - 1 - j
                high_trim = i + 1
                g[i, j] = (total - start[low_trim] - finish[high_trim]) * inv

    for i in range(taps):
        products = probe[:n-1-i] * mic[i+1:]
        target[i] = (float(np.sum(products))
                     - float(np.sum(products[:taps-1-i]))) * inv
    return g, target


def _exact_iv_raw(mic: np.ndarray, loudspeaker: np.ndarray, probe: np.ndarray,
                  taps: int, ridge_relative: float) -> tuple[np.ndarray, float]:
    g, b = _exact_iv_moments(mic, loudspeaker, probe, taps)
    cond = float(np.linalg.cond(g))
    if not np.isfinite(cond) or cond > 1e6:
        raise ValueError(f"Ill-conditioned exact IV moments: {cond:g}")
    scale = ridge_relative * np.mean(probe * probe)
    return np.linalg.solve(g + scale * np.eye(taps), b), cond


@dataclass(frozen=True)
class EnsembleResult:
    coefficients: np.ndarray
    v2_coefficients: np.ndarray
    samples: int
    probe_rms: float
    sigma_full: float
    moment_condition: float
    effective_model_count: float
    best_group_width: int
    best_local_width: int
    temperature_factor: float
    exact_moments: bool
    crossfit_gain: float

    def to_identification(self) -> IdentificationResult:
        """Use with existing ProbeAFC.activate, without changing its DSP loop."""
        return IdentificationResult(self.coefficients.copy(),
                                    self.coefficients.copy(), self.samples,
                                    self.probe_rms, self.sigma_full,
                                    0.0, self.moment_condition)


def identify_v3(mic: np.ndarray, loudspeaker: np.ndarray, probe: np.ndarray,
                taps: int, *, ridge_relative: float = 1e-5,
                temperature_factor: float = 3.0,
                include_v2: bool = True,
                use_exact: bool = True,
                crossfit_gain: bool = False) -> EnsembleResult:
    """Model-average source-blind IV path estimates using two-way crossfit.

    For each (contiguous support, local Wiener) structure, each half-record
    predicts the other half's *raw IV* coefficients. Source-blind weights are
    proportional to exp(-(CVloss-minloss)/(2*T)), where T is an estimated
    CV noise scale. Final models are refit to full-record data and averaged.
    """
    mic, loudspeaker, probe = _validate_series(mic, loudspeaker, probe, taps)
    if not np.isfinite(temperature_factor) or temperature_factor <= 0:
        raise ValueError('temperature_factor must be positive and finite')
    if ridge_relative < 0:
        raise ValueError('ridge_relative must be nonnegative')
    raw_fn = _exact_iv_raw if use_exact else _iv_approx_raw
    raw, condition = raw_fn(mic, loudspeaker, probe, taps, ridge_relative)
    midpoint = len(mic)//2
    r1, _ = raw_fn(mic[:midpoint], loudspeaker[:midpoint], probe[:midpoint], taps, ridge_relative)
    r2, _ = raw_fn(mic[midpoint:], loudspeaker[midpoint:], probe[midpoint:], taps, ridge_relative)
    sigma = float(np.sqrt(np.mean((r1-r2)**2))/2)
    sigma_half = sigma*np.sqrt(2.0)
    widths = tuple(dict.fromkeys([w for w in (16,20,24,28,32,36,40,48,64,taps) if w <= taps]))
    local_windows = tuple(w for w in (4,8,12,16,24) if w <= taps)

    # Candidate zero is a source-blind ordinary soft-thresholded IV path.
    soft1, _ = _sure_soft_threshold(r1,sigma_half)
    soft2, _ = _sure_soft_threshold(r2,sigma_half)
    soft_full, _ = _sure_soft_threshold(raw,sigma)
    candidates = [(-1,0,soft1,soft2,soft_full)]
    for group in widths:
        for local in local_windows:
            candidates.append((group,local,
                               structured_denoise(r1,sigma_half,group,local),
                               structured_denoise(r2,sigma_half,group,local),
                               structured_denoise(raw,sigma,group,local)))
    losses = np.array([np.sum((c1-r2)**2)+np.sum((c2-r1)**2)
                       for _,_,c1,c2,_ in candidates])
    # A typical cross-validation error fluctuation has scale sqrt(2k)*sigma²;
    # soft model weights retain near-tied choices instead of winner-takes-all.
    temperature = temperature_factor*np.sqrt(2*taps)*sigma_half**2
    posterior = np.exp(-(losses - float(np.min(losses))) / (2*temperature+1e-30))
    posterior /= np.sum(posterior)
    result = sum(float(p)*candidate[4] for p,candidate in zip(posterior,candidates))
    gain = 1.0
    if crossfit_gain:
        pred1 = sum(float(p)*candidate[2] for p,candidate in zip(posterior,candidates))
        pred2 = sum(float(p)*candidate[3] for p,candidate in zip(posterior,candidates))
        denominator = float(pred1@pred1 + pred2@pred2)
        if denominator > 1e-28:
            optimal = float((pred1@r2 + pred2@r1)/denominator)
            # A conservative numerical bound, not a hardware safety limiter.
            gain = float(np.clip(optimal, 0.75, 1.5))
            result *= gain
    if include_v2:
        # This is an independent non-oracle comparator; callers can inspect it.
        v2 = identify_structured(mic,loudspeaker,probe,taps,ridge_relative=ridge_relative)
        v2_coefficients = v2.coefficients
    else:
        v2_coefficients = soft_full
    best = candidates[int(np.argmin(losses))]
    return EnsembleResult(np.asarray(result),v2_coefficients,len(mic),
                          float(np.sqrt(np.mean(probe**2))),sigma,condition,
                          float(1/np.sum(posterior**2)),int(best[0]),int(best[1]),
                          float(temperature_factor),bool(use_exact),gain)


def _iv_approx_raw(mic, loudspeaker, probe, taps, ridge_relative):
    from afc import _iv_raw
    return _iv_raw(mic,loudspeaker,probe,taps,ridge_relative)
