"""Cross-fitted, structure-aware probe-IV acoustic feedback estimator.

This module ONLY accepts mic, actual loudspeaker, and independent probe.
Neither the hidden near-end target nor true acoustic path are used. Model
complexity is selected by cross-fitting two temporally disjoint estimates.

`afc.py` remains unchanged as the frozen reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from afc import IdentificationResult, _iv_raw, _validate_series, identify_path


@dataclass(frozen=True)
class StructuredResult:
    coefficients: np.ndarray
    reference: IdentificationResult
    group_width: int
    local_width: int
    gain: float
    crossfit_loss: float
    baseline_crossfit_loss: float
    sigma_full: float

    def to_identification(self) -> IdentificationResult:
        """Adapt a structured path estimate for `ProbeAFC.activate()`."""
        return replace(self.reference, coefficients=self.coefficients.copy())


def structured_denoise(raw: np.ndarray, sigma: float,
                       group_width: int, local_width: int) -> np.ndarray:
    """Estimate an unknown response cluster, then locally denoise its taps.

    Width 0 means the unchanged raw path; width 128 is effectively a global
    group. Neither the cluster location nor its width is supplied by truth.
    """
    raw = np.asarray(raw, dtype=np.float64).ravel()
    if group_width == 0:
        return raw.copy()
    n = len(raw)
    if not 1 <= group_width <= n or not 1 <= local_width <= n:
        raise ValueError('Invalid structure window lengths')
    evidence = np.maximum(np.abs(raw) - sigma, 0.0) ** 2
    candidate_energy = np.convolve(evidence, np.ones(group_width), mode='valid')
    start = int(np.argmax(candidate_energy))
    mask = np.zeros(n)
    mask[start:start + group_width] = 1.0

    local_power = np.convolve(raw * raw,
                              np.ones(local_width) / local_width, mode='same')
    signal_power = np.maximum(local_power - sigma ** 2, 0.0)
    shrink = signal_power / (signal_power + sigma ** 2 + 1e-30)
    return raw * mask * shrink


def _candidate(raw: np.ndarray, sigma: float, group_width: int,
               local_width: int, reference_soft: np.ndarray | None = None) -> np.ndarray:
    if group_width == -1:
        if reference_soft is None:
            raise ValueError('Reference soft threshold required')
        return reference_soft
    return structured_denoise(raw, sigma, group_width, local_width)


def identify_structured(mic: np.ndarray, loudspeaker: np.ndarray,
                        probe: np.ndarray, taps: int,
                        *, candidate_group_widths: tuple[int, ...] | None = None,
                        candidate_local_widths: tuple[int, ...] = (4, 8, 12, 16, 24),
                        candidate_gains: tuple[float, ...] = (1.0,),
                        ridge_relative: float = 1e-5) -> StructuredResult:
    """Two-way cross-fit selection, followed by full-record IV refitting.

    The non-structured SURE baseline is included in candidate selection.
    Loss: ||T(w_half1) - w_raw_half2||^2 + vice versa.
    Because the held-out raw IV is unbiased under independent probing, its
    noise contributes a candidate-independent constant to this expected loss.
    """
    mic, loudspeaker, probe = _validate_series(mic, loudspeaker, probe, taps)
    if candidate_group_widths is None:
        choices = [k for k in (16, 20, 24, 28, 32, 36, 40, 48, 64)
                   if k <= taps]
        candidate_group_widths = tuple([-1, *choices, taps])
    candidate_local_widths = tuple(k for k in candidate_local_widths if k <= taps)
    if any(k not in (-1, 0) and (k < 1 or k > taps) for k in candidate_group_widths):
        raise ValueError('Out-of-range group width candidate')
    if not candidate_local_widths:
        raise ValueError('Need at least one local smoothing window')
    if not candidate_gains or 1.0 not in candidate_gains or any(g <= 0 for g in candidate_gains):
        raise ValueError('Candidate gains must be positive and include 1.0')

    baseline = identify_path(mic, loudspeaker, probe, taps,
                             ridge_relative=ridge_relative, sparse_shrinkage=True)
    midpoint = len(mic) // 2
    raw1, _ = _iv_raw(mic[:midpoint], loudspeaker[:midpoint], probe[:midpoint],
                       taps, ridge_relative)
    raw2, _ = _iv_raw(mic[midpoint:], loudspeaker[midpoint:], probe[midpoint:],
                       taps, ridge_relative)
    sigma_full = float(np.sqrt(np.mean((raw1 - raw2) ** 2)) / 2.0)
    sigma_half = sigma_full * np.sqrt(2.0)

    # The reference's SURE threshold needs a per-half noise estimate.
    from afc import _sure_soft_threshold
    soft1, _ = _sure_soft_threshold(raw1, sigma_half)
    soft2, _ = _sure_soft_threshold(raw2, sigma_half)

    best = None
    baseline_loss = None
    for group in candidate_group_widths:
        for local in candidate_local_widths:
            if group == -1 and local != candidate_local_widths[0]:
                continue
            fitted1 = _candidate(raw1, sigma_half, group, local, soft1)
            fitted2 = _candidate(raw2, sigma_half, group, local, soft2)
            for gain in candidate_gains:
                loss = (float(np.sum((gain*fitted1 - raw2) ** 2))
                        + float(np.sum((gain*fitted2 - raw1) ** 2)))
                if group == -1 and gain == 1.0:
                    baseline_loss = loss
                if best is None or loss < best[0]:
                    best = (loss, group, local, gain)
    assert best is not None and baseline_loss is not None

    loss, width, local, gain = best
    proposed = gain*_candidate(baseline.raw_coefficients, sigma_full,
                               width, local, baseline.coefficients)
    return StructuredResult(proposed, baseline, width, local, gain,
                            loss, baseline_loss, sigma_full)
