"""
pipeline/motion_analyzer.py
Step 3: Optical flow computation using OpenCV Farneback.
Outputs per-consecutive-frame flow fields and dominant motion type.
"""

import cv2
import numpy as np
from typing import List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def compute_optical_flow(
    frames: List[np.ndarray],
) -> Tuple[List[np.ndarray], str]:
    """
    Compute dense optical flow (Farneback) between consecutive frames.

    Args:
        frames: List of BGR numpy frames (HxWx3).

    Returns:
        (flows, motion_type)
        - flows: list of (H, W, 2) float32 flow arrays (len = len(frames)-1)
        - motion_type: string describing dominant motion (e.g. 'panning', 'static')
    """
    if len(frames) < 2:
        print("  ⚠️  Need at least 2 frames for optical flow.")
        return [], "static"

    flows: List[np.ndarray] = []
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i],
            gray_frames[i + 1],
            None,
            config.FLOW_PYR_SCALE,
            config.FLOW_LEVELS,
            config.FLOW_WINSIZE,
            config.FLOW_ITERATIONS,
            config.FLOW_POLY_N,
            config.FLOW_POLY_SIGMA,
            0,
        )
        flows.append(flow)

    motion_type = _classify_motion(flows)
    print(f"  → Computed {len(flows)} flow fields. Motion type: {motion_type}")
    return flows, motion_type


def _classify_motion(flows: List[np.ndarray]) -> str:
    """
    Classify the dominant camera/scene motion from flow fields.
    Returns one of: 'static', 'panning', 'zooming', 'complex'
    """
    if not flows:
        return "static"

    # Average flow vector across all frames
    avg_flows = np.mean([f for f in flows], axis=0)  # (H, W, 2)
    mean_dx = float(avg_flows[:, :, 0].mean())
    mean_dy = float(avg_flows[:, :, 1].mean())
    magnitude = np.sqrt(mean_dx ** 2 + mean_dy ** 2)

    # Average magnitude across all pixels
    per_frame_mags = [
        float(np.sqrt(f[:, :, 0] ** 2 + f[:, :, 1] ** 2).mean())
        for f in flows
    ]
    avg_mag = np.mean(per_frame_mags)

    if avg_mag < 0.5:
        return "static"

    # Check for zoom: flow points radially outward/inward
    # Simplified heuristic: high magnitude with low net directional bias
    h, w = flows[0].shape[:2]
    cx, cy = w / 2, h / 2
    ys, xs = np.mgrid[0:h, 0:w]
    dx_field = avg_flows[:, :, 0]
    dy_field = avg_flows[:, :, 1]

    # Compare flow direction vs radial direction
    radial_x = (xs - cx) / (np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) + 1e-6)
    radial_y = (ys - cy) / (np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) + 1e-6)
    dot = (dx_field * radial_x + dy_field * radial_y).mean()

    if abs(dot) > 0.3:
        return "zooming"
    if abs(mean_dx) > abs(mean_dy) and magnitude > 1.0:
        return "panning_horizontal"
    if abs(mean_dy) > abs(mean_dx) and magnitude > 1.0:
        return "panning_vertical"
    return "complex"


def flow_to_rgb_visualization(flow: np.ndarray) -> np.ndarray:
    """
    Convert an optical flow field to an HSV-based RGB visualization.
    Useful for debugging.
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    angle     = np.arctan2(fy, fx) + np.pi                     # 0..2π
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle / (2 * np.pi) * 180).astype(np.uint8)  # Hue
    hsv[:, :, 1] = 255                                             # Saturation
    hsv[:, :, 2] = magnitude.astype(np.uint8)                     # Value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_flow_statistics(flows: List[np.ndarray]) -> dict:
    """Return summary statistics about the flow fields."""
    if not flows:
        return {"avg_magnitude": 0.0, "max_magnitude": 0.0, "std_magnitude": 0.0}
    mags = [float(np.sqrt(f[:,:,0]**2 + f[:,:,1]**2).mean()) for f in flows]
    return {
        "avg_magnitude": float(np.mean(mags)),
        "max_magnitude": float(np.max(mags)),
        "std_magnitude": float(np.std(mags)),
        "per_frame":     mags,
    }
