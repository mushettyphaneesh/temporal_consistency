"""
Lightweight Video Quality Metrics

Calculates video quality metrics without heavy dependencies (no PyTorch):
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Warping Error (Optical Flow based)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("Installing scikit-image for SSIM calculation...")
    import os
    os.system("pip install scikit-image")
    from skimage.metrics import structural_similarity as ssim


class VideoQualityMetrics:
    """Calculate quality metrics for video comparison."""
    
    def __init__(self, max_frames: int = 30):
        """
        Initialize metrics calculator.
        
        Args:
            max_frames: Maximum number of frames to analyze (to limit computation)
        """
        self.max_frames = max_frames
    
    def load_video_frames(self, video_path: Path, resize_to: Tuple[int, int] = None) -> List[np.ndarray]:
        """Load frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly if video has more frames than max_frames
        if total_frames > self.max_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        else:
            frame_indices = range(total_frames)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize if specified
                if resize_to:
                    frame = cv2.resize(frame, resize_to)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def calculate_psnr(self, frames1: List[np.ndarray], frames2: List[np.ndarray]) -> Dict:
        """
        Calculate PSNR between two video frame sequences.
        
        Returns:
            Dict with mean, std, min, max PSNR values
        """
        num_frames = min(len(frames1), len(frames2))
        psnr_values = []
        
        for i in range(num_frames):
            # Convert to same size if needed
            h1, w1 = frames1[i].shape[:2]
            h2, w2 = frames2[i].shape[:2]
            
            if (h1, w1) != (h2, w2):
                # Resize to smaller dimension
                target_size = (min(w1, w2), min(h1, h2))
                frame1 = cv2.resize(frames1[i], target_size)
                frame2 = cv2.resize(frames2[i], target_size)
            else:
                frame1, frame2 = frames1[i], frames2[i]
            
            # Calculate PSNR
            psnr = cv2.PSNR(frame1, frame2)
            psnr_values.append(psnr)
        
        return {
            "mean": float(np.mean(psnr_values)),
            "std": float(np.std(psnr_values)),
            "min": float(np.min(psnr_values)),
            "max": float(np.max(psnr_values))
        }
    
    def calculate_ssim(self, frames1: List[np.ndarray], frames2: List[np.ndarray]) -> Dict:
        """
        Calculate SSIM between two video frame sequences.
        
        Returns:
            Dict with mean, std, min, max SSIM values
        """
        num_frames = min(len(frames1), len(frames2))
        ssim_values = []
        
        for i in range(num_frames):
            # Convert to same size if needed
            h1, w1 = frames1[i].shape[:2]
            h2, w2 = frames2[i].shape[:2]
            
            if (h1, w1) != (h2, w2):
                # Resize to smaller dimension
                target_size = (min(w1, w2), min(h1, h2))
                frame1 = cv2.resize(frames1[i], target_size)
                frame2 = cv2.resize(frames2[i], target_size)
            else:
                frame1, frame2 = frames1[i], frames2[i]
            
            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM
            ssim_value = ssim(gray1, gray2)
            ssim_values.append(ssim_value)
        
        return {
            "mean": float(np.mean(ssim_values)),
            "std": float(np.std(ssim_values)),
            "min": float(np.min(ssim_values)),
            "max": float(np.max(ssim_values))
        }
    
    def calculate_warping_error(self, frames: List[np.ndarray]) -> Dict:
        """
        Calculate optical flow warping error (temporal consistency).
        Measures how well consecutive frames can be predicted via optical flow.
        
        Returns:
            Dict with mean warping error
        """
        if len(frames) < 2:
            return {"mean": 0.0, "std": 0.0}
        
        warping_errors = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow (Farneback method)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate flow magnitude (movement)
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Use mean flow magnitude as warping error metric
            warping_error = float(np.mean(flow_magnitude))
            warping_errors.append(warping_error)
        
        return {
            "mean": float(np.mean(warping_errors)),
            "std": float(np.std(warping_errors)),
            "min": float(np.min(warping_errors)),
            "max": float(np.max(warping_errors))
        }
    
    def compare_videos(self, video1_path: Path, video2_path: Path) -> Dict:
        """
        Compare two videos using all available metrics.
        
        Returns:
            Dict with all metric results
        """
        # Load frames
        frames1 = self.load_video_frames(video1_path)
        frames2 = self.load_video_frames(video2_path)
        
        if not frames1 or not frames2:
            return {"error": "Failed to load video frames"}
        
        # Calculate metrics
        results = {
            "psnr": self.calculate_psnr(frames1, frames2),
            "ssim": self.calculate_ssim(frames1, frames2),
            "warping_error_video1": self.calculate_warping_error(frames1),
            "warping_error_video2": self.calculate_warping_error(frames2),
            "frames_analyzed": min(len(frames1), len(frames2))
        }
        
        return results
    
    def analyze_single_video(self, video_path: Path) -> Dict:
        """
        Analyze temporal consistency of a single video.
        
        Returns:
            Dict with warping error metrics
        """
        frames = self.load_video_frames(video_path)
        
        if not frames:
            return {"error": "Failed to load video frames"}
        
        return {
            "warping_error": self.calculate_warping_error(frames),
            "frames_analyzed": len(frames)
        }


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python video_quality_metrics.py <video1> <video2>")
        sys.exit(1)
    
    metrics = VideoQualityMetrics(max_frames=30)
    
    video1 = Path(sys.argv[1])
    video2 = Path(sys.argv[2])
    
    print(f"\nComparing videos:")
    print(f"  Video 1: {video1}")
    print(f"  Video 2: {video2}")
    
    results = metrics.compare_videos(video1, video2)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nPSNR (Peak Signal-to-Noise Ratio):")
    print(f"  Mean: {results['psnr']['mean']:.2f} dB")
    print(f"  Std:  {results['psnr']['std']:.2f} dB")
    print(f"  Range: [{results['psnr']['min']:.2f}, {results['psnr']['max']:.2f}] dB")
    
    print(f"\nSSIM (Structural Similarity Index):")
    print(f"  Mean: {results['ssim']['mean']:.4f}")
    print(f"  Std:  {results['ssim']['std']:.4f}")
    print(f"  Range: [{results['ssim']['min']:.4f}, {results['ssim']['max']:.4f}]")
    
    print(f"\nWarping Error (Temporal Consistency):")
    print(f"  Video 1 Mean: {results['warping_error_video1']['mean']:.2f} px/frame")
    print(f"  Video 2 Mean: {results['warping_error_video2']['mean']:.2f} px/frame")
    
    print(f"\nFrames Analyzed: {results['frames_analyzed']}")
    print("=" * 60)
