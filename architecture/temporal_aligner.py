"""
pipeline/temporal_aligner.py
Step 6: Motion-guided warping and temporal loss computation.
Implements: Lt = ||v'(i+1) - warp(v'i, mi)||²
"""

import cv2
import numpy as np
from typing import List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp a frame forward using an optical flow field.

    Args:
        frame: HxWx3 BGR numpy array
        flow:  HxW×2 float32 flow array (flow[y,x] = (dx,dy))

    Returns:
        Warped frame (HxWx3 uint8)
    """
    h, w = frame.shape[:2]
    # Build absolute remap coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                   np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[:, :, 0]
    map_y = grid_y + flow[:, :, 1]
    warped = cv2.remap(
        frame,
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def compute_temporal_loss(
    frames: List[np.ndarray],
    flows: List[np.ndarray],
) -> float:
    """
    Compute mean temporal loss over all consecutive frame pairs:
    Lt = mean_i( ||frame[i+1] - warp(frame[i], flow[i])||² )
    """
    if len(frames) < 2 or not flows:
        return 0.0

    total_loss = 0.0
    count = 0
    for i in range(min(len(frames) - 1, len(flows))):
        warped = warp_frame(frames[i], flows[i])
        diff   = frames[i + 1].astype(np.float32) - warped.astype(np.float32)
        total_loss += float(np.mean(diff ** 2))
        count += 1

    return total_loss / count if count > 0 else 0.0


def compute_spatial_loss(frames: List[np.ndarray]) -> float:
    """
    Spatial quality loss: mean gradient magnitude across all frames.
    High gradient → sharp; low gradient → blurry.
    We measure blurriness as inverse sharpness (lower is better for smoothness).
    """
    if not frames:
        return 0.0
    losses = []
    for f in frames:
        gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lap   = cv2.Laplacian(gray, cv2.CV_64F)
        # Variance of Laplacian = sharpness; we want to MAXIMISE it (minimise blur)
        losses.append(float(lap.var()))
    # Return normalised spatial loss (1 - normalised_sharpness) so lower = better
    avg = np.mean(losses)
    return float(1.0 / (1.0 + avg))   # ∈ (0, 1], lower = sharper


def align_temporally(
    frames: List[np.ndarray],
    flows: List[np.ndarray],
    lambda_weight: float = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Apply motion-guided temporal alignment to regenerated frames.

    For each frame i, we blend:
        aligned[i] = α·frame[i] + (1-α)·warp(frame[i-1], flow[i-1])

    This enforces smooth temporal transitions while retaining the content
    of the regenerated frame.

    Args:
        frames:        List of BGR frames (regenerated video V').
        flows:         List of optical flow fields from original video.
        lambda_weight: λ for combined loss L = Lt + λ·Lspatial.

    Returns:
        (aligned_frames, combined_loss)
    """
    lambda_weight = lambda_weight if lambda_weight is not None else config.LAMBDA_WEIGHT

    if len(frames) < 2 or not flows:
        lt = compute_temporal_loss(frames, flows)
        ls = compute_spatial_loss(frames)
        return frames, lt + lambda_weight * ls

    aligned: List[np.ndarray] = [frames[0].copy()]
    alpha = 0.6   # weight for the regenerated frame content

    for i in range(1, len(frames)):
        flow_idx = min(i - 1, len(flows) - 1)
        warped_prev = warp_frame(aligned[i - 1], flows[flow_idx])
        blended = cv2.addWeighted(
            frames[i].astype(np.float32),      alpha,
            warped_prev.astype(np.float32), 1.0 - alpha,
            0,
        )
        aligned.append(np.clip(blended, 0, 255).astype(np.uint8))

    # Compute combined loss
    lt = compute_temporal_loss(aligned, flows)
    ls = compute_spatial_loss(aligned)
    combined_loss = lt + lambda_weight * ls

    print(f"  → Temporal loss: {lt:.4f} | Spatial loss: {ls:.4f} | "
          f"Combined (λ={lambda_weight}): {combined_loss:.4f}")
    return aligned, combined_loss
