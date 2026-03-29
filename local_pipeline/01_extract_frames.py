"""
Step 1 — Frame Extraction
=========================
Extracts N evenly-spaced frames from a video and saves them as JPEG files.
Output goes to:  local_pipeline/frames/<video_stem>/frame_000.jpg  ...

Usage:
    python 01_extract_frames.py --video path/to/video.mp4 --n 8
"""

import argparse
import os
import cv2


def extract_frames(video_path: str, n: int = 8, out_dir: str = None) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps if fps > 0 else 0
    print(f"  Video: {total} frames  |  {fps:.1f} fps  |  {dur:.1f}s")

    positions = (
        list(range(total))
        if total <= n
        else [int(i * (total - 1) / (n - 1)) for i in range(n)]
    )

    stem   = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "frames", stem)
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for idx, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        # Resize to max 512px on the long edge
        h, w = frame.shape[:2]
        scale = min(512 / max(h, w), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        path = os.path.join(out_dir, f"frame_{idx:03d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved.append(path)
        print(f"  Saved frame {idx+1}/{len(positions)}: {path}")

    cap.release()
    print(f"\n✓ {len(saved)} frames saved to: {out_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--n",     type=int, default=8, help="Number of frames to extract (default: 8)")
    parser.add_argument("--out",   default=None, help="Output directory (optional)")
    args = parser.parse_args()

    extract_frames(args.video, args.n, args.out)
