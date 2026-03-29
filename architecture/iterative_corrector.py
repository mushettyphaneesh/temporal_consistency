"""
pipeline/iterative_corrector.py
Step 7: Iterative correction - flicker removal, edge-preserving smoothing,
temporal filtering, and weighted blending.

Minimises: L = Lt + λ·Lspatial over multiple correction passes.
"""

import cv2
import numpy as np
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def _temporal_gaussian_smooth(frames: List[np.ndarray], sigma: float = 1.0) -> List[np.ndarray]:
    """
    Apply 1-D Gaussian smoothing along the temporal axis (frame dimension).
    Each pixel's temporal value is convolved with a Gaussian kernel.
    """
    if len(frames) < 3:
        return frames

    n = len(frames)
    # Build Gaussian weights for a 3-tap kernel
    weights = np.array([np.exp(-0.5 * ((k - 1) / sigma) ** 2) for k in range(3)])
    weights /= weights.sum()

    smoothed = [frames[0].copy()]
    for i in range(1, n - 1):
        blended = (
            weights[0] * frames[i - 1].astype(np.float32) +
            weights[1] * frames[i    ].astype(np.float32) +
            weights[2] * frames[i + 1].astype(np.float32)
        )
        smoothed.append(np.clip(blended, 0, 255).astype(np.uint8))
    smoothed.append(frames[-1].copy())
    return smoothed


def _edge_preserving_filter(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply bilateral filtering to each frame to preserve edges while
    removing noise / flicker artefacts.
    """
    return [
        cv2.bilateralFilter(
            f,
            config.BILATERAL_D,
            config.BILATERAL_SIGMA_COLOR,
            config.BILATERAL_SIGMA_SPACE,
        )
        for f in frames
    ]


def _weighted_blend(
    original: List[np.ndarray],
    smoothed: List[np.ndarray],
    alpha: float,
) -> List[np.ndarray]:
    """
    Blend original frames with smoothed frames:
    result[i] = alpha * original[i] + (1 - alpha) * smoothed[i]
    """
    result = []
    for o, s in zip(original, smoothed):
        blended = cv2.addWeighted(
            o.astype(np.float32), alpha,
            s.astype(np.float32), 1.0 - alpha,
            0,
        )
        result.append(np.clip(blended, 0, 255).astype(np.uint8))
    return result


def _remove_flicker(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Histogram-based luminance normalisation to reduce inter-frame
    brightness flicker.
    For each frame, normalise mean luminance to the global mean.
    """
    if not frames:
        return frames

    # Compute per-frame mean luminance
    lum_means = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lum_means.append(float(gray.mean()))

    global_mean = np.mean(lum_means)
    corrected = []
    for f, lum in zip(frames, lum_means):
        if lum < 1e-3:
            corrected.append(f.copy())
            continue
        scale = global_mean / lum
        # Clamp scale to avoid over-brightening/darkening
        scale = np.clip(scale, 0.75, 1.25)
        c = np.clip(f.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        corrected.append(c)
    return corrected


def iterative_correct(
    frames: List[np.ndarray],
    iterations: int = None,
    lambda_weight: float = None,
) -> List[np.ndarray]:
    """
    Apply multi-pass iterative correction to a sequence of frames.

    Each pass:
        1. Remove luminance flicker
        2. Temporal Gaussian smooth
        3. Edge-preserving bilateral filter
        4. Weighted blend (preserves sharpness)

    Args:
        frames:        Input frame list (aligned V').
        iterations:    Number of correction passes.
        lambda_weight: λ (used for adaptive blend alpha calculation).

    Returns:
        Corrected frame list.
    """
    iterations    = iterations    if iterations    is not None else config.CORRECTION_ITERATIONS
    lambda_weight = lambda_weight if lambda_weight is not None else config.LAMBDA_WEIGHT

    # Blend alpha decreases as λ increases (more smoothing for higher λ)
    alpha = max(0.4, config.BLEND_ALPHA - 0.05 * lambda_weight)

    current = [f.copy() for f in frames]

    for it in range(iterations):
        # Step A: Luminance flicker removal
        current = _remove_flicker(current)

        # Step B: Temporal smoothing
        smoothed = _temporal_gaussian_smooth(current, sigma=config.TEMPORAL_SIGMA)

        # Step C: Edge-preserving filter on smoothed frames
        filtered = _edge_preserving_filter(smoothed)

        # Step D: Weighted blend
        current = _weighted_blend(current, filtered, alpha)

        # Adaptive alpha: reduce smoothing in later iterations
        alpha = max(0.55, alpha + 0.03)

        print(f"  → Correction pass {it + 1}/{iterations} done (blend_alpha={alpha:.2f})")

    return current
