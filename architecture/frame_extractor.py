"""
pipeline/frame_extractor.py
Extract frames from video using OpenCV.
Provides both the legacy extract_frames() and new extract_n_frames() for the Flask app.
"""

import cv2
import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def extract_n_frames(video_path: str, n: int = 5) -> List[np.ndarray]:
    """
    Extract exactly `n` evenly-spaced frames from a video.
    Each frame is resized so the longest dimension is at most MAX_FRAME_DIM (512px).

    Args:
        video_path: Path to the input .mp4 file.
        n:          Number of frames to extract (default 5).

    Returns:
        List of BGR numpy arrays (HxWx3 uint8).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    # Compute evenly-spaced frame indices
    if total <= n:
        indices = list(range(total))
    else:
        step = total / n
        indices = [int(step * i + step / 2) for i in range(n)]

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = _resize_max_dim(frame, config.MAX_FRAME_DIM)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be extracted from: {video_path}")

    print(f"  → Extracted {len(frames)} frames (max dim={config.MAX_FRAME_DIM}px)")
    return frames


def _resize_max_dim(frame: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize frame so the longest side is at most max_dim, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    if w >= h:
        new_w = max_dim
        new_h = int(h * max_dim / w)
    else:
        new_h = max_dim
        new_w = int(w * max_dim / h)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─── Legacy API (kept for any remaining references) ──────────────────────────

def extract_frames(
    video_path: str,
    max_frames: int = None,
    target_width: int = None,
    target_height: int = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Legacy: extract up to max_frames frames at fixed (width, height).
    Returns (frames, fps).
    """
    max_frames    = max_frames    or config.FRAME_COUNT * 6
    target_width  = target_width  or 512
    target_height = target_height or 288

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride      = max(1, total_count // max_frames)

    frames: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            resized = cv2.resize(frame, (target_width, target_height),
                                 interpolation=cv2.INTER_AREA)
            frames.append(resized)
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    print(f"  → Extracted {len(frames)} frames (stride={stride}, "
          f"resolution={target_width}×{target_height}, fps={fps:.1f})")
    return frames, fps


def load_video_metadata(video_path: str) -> dict:
    """Return basic metadata about a video without extracting all frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {
        "fps":          fps,
        "frame_count":  frame_count,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": frame_count / fps,
    }
    cap.release()
    return meta
