"""
pipeline/__init__.py
Public API for the Temporal Consistency Studio pipeline.
"""

from .frame_extractor import extract_n_frames, extract_frames, load_video_metadata
from .video_regenerator import regenerate_video

__all__ = [
    "extract_n_frames",
    "extract_frames",
    "load_video_metadata",
    "regenerate_video",
]
