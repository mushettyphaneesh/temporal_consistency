"""
Temporal Consistency Studio — CLI Version
Same pipeline as the web app, runnable from the command line.

Usage:
    python video_prompt.py --video path/to/video.mp4
    python video_prompt.py --video path/to/video.mp4 --generate --duration 8 --aspect 16:9
    python video_prompt.py --video path/to/video.mp4 --feedback "more cinematic, golden hour lighting"
"""

import argparse
import os
import sys
import json
import base64
import time
import re

from google import genai
from google.genai import types
from dotenv import load_dotenv

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    os.system("pip install opencv-python")
    import cv2


def extract_frames(video_path: str, n: int = 5) -> list:
    """Extract n evenly-spaced frames from video, resized to max 512px."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= n:
        positions = list(range(total_frames))
    else:
        positions = [int(i * (total_frames - 1) / (n - 1)) for i in range(n)]

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            scale = min(512 / max(h, w), 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frames.append(frame)

    cap.release()
    print(f"  Extracted {len(frames)} frames")
    return frames


def frames_to_base64(frames: list) -> list:
    """Convert frames to base64-encoded JPEG strings."""
    encoded = []
    for frame in frames:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        encoded.append(base64.b64encode(buffer).decode('utf-8'))
    return encoded


def analyze_video(client: genai.Client, frames_b64: list, feedback: str = None) -> dict:
    """
    Semantic Analysis — Paper Section III (Eq. 2, 3, 4).
    Returns structured JSON with scene analysis and refined prompt P*.
    """
    system_instruction = (
        "You are a video temporal consistency analyzer. Analyze these frames from an AI-generated video. "
        "Return ONLY a JSON object with:\n"
        "- scene_description: one paragraph describing stable scene attributes\n"
        "- detected_objects: list of main objects with consistent properties\n"
        "- temporal_issues: list of flickering, texture instability, color shifts, or motion artifacts\n"
        "- refined_prompt: a single structured paragraph (P*) for video regeneration enforcing "
        "consistent lighting, stable textures, smooth motion, coherent color transitions\n"
        "- motion_type: camera motion description\n"
        "- lighting_cues: dominant lighting conditions"
    )

    if feedback:
        system_instruction += (
            f"\n\nUSER REFINEMENT: Incorporate this feedback into refined_prompt: '{feedback}'"
        )

    contents = []
    for b64 in frames_b64:
        contents.append(
            types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg")
        )
    contents.append(system_instruction)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            response_mime_type="application/json"
        )
    )

    raw = response.text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"refined_prompt": raw, "temporal_issues": [], "detected_objects": []}


def regenerate_video(
    client: genai.Client,
    prompt: str,
    output_path: str = "outputs/regenerated.mp4",
    duration: int = 4,
    aspect_ratio: str = "16:9"
) -> str:
    """
    Video Regeneration — Paper Equation 5.
    Sends P* to Veo with polling every 10 seconds. Timeout: 5 minutes.
    """
    print(f"\n  Sending P* to Veo (veo-2.0-generate-001)...")
    print(f"  Duration: {duration}s | Aspect: {aspect_ratio}")

    operation = client.models.generate_videos(
        model="veo-2.0-generate-001",
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=duration,
            person_generation="allow_all"
        )
    )

    max_polls = 30
    poll_count = 0

    while not operation.done:
        if poll_count >= max_polls:
            raise TimeoutError("Video generation timed out after 5 minutes.")
        poll_count += 1
        print(f"  Polling ({poll_count}/30)...")
        time.sleep(10)
        operation = client.operations.get(operation)

    if operation.error:
        raise RuntimeError(f"Video generation failed: {operation.error}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    generated_video.video.save(output_path)

    print(f"  Video saved → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Temporal Consistency Studio CLI — Analysis-Guided Video Regeneration"
    )
    parser.add_argument("--video", "-v", required=True, help="Path to AI-generated video (.mp4)")
    parser.add_argument("--frames", "-f", type=int, default=5, help="Frames to extract (default: 5)")
    parser.add_argument("--feedback", type=str, default=None, help="Optional user feedback to refine P*")
    parser.add_argument("--output-json", type=str, default=None, help="Save analysis JSON to file")
    parser.add_argument("--generate", "-g", action="store_true", help="Regenerate video with Veo after analysis")
    parser.add_argument("--duration", "-d", type=int, choices=[4, 8], default=4, help="Video duration (4 or 8 seconds)")
    parser.add_argument("--aspect", "-a", type=str, choices=["16:9", "9:16"], default="16:9", help="Aspect ratio")
    parser.add_argument("--video-output", "-o", type=str, default="outputs/regenerated.mp4", help="Output video path")

    args = parser.parse_args()
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found. Add it to .env or set it as environment variable.")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("\n" + "=" * 60)
    print("  TEMPORAL CONSISTENCY STUDIO — CLI")
    print("  Paper: Analysis-Guided Video Regeneration")
    print("=" * 60)

    # Step 1: Extract frames
    print("\n[Step 1] Extracting frames...")
    frames = extract_frames(args.video, n=args.frames)
    frames_b64 = frames_to_base64(frames)

    # Step 2: Semantic analysis (Eq. 2, 3, 4)
    print("\n[Step 2] Running Gemini semantic analysis (Eq. 2, 3, 4)...")
    analysis = analyze_video(client, frames_b64, feedback=args.feedback)

    print("\n── Scene Description ──────────────────────────────────")
    print(analysis.get("scene_description", "N/A"))

    print("\n── Detected Objects ───────────────────────────────────")
    objs = analysis.get("detected_objects", [])
    for i, obj in enumerate(objs, 1):
        print(f"  {i}. {obj}")

    print("\n── Temporal Issues ────────────────────────────────────")
    issues = analysis.get("temporal_issues", [])
    if issues:
        for issue in issues:
            print(f"  ⚠  {issue}")
    else:
        print("  ✓  No temporal issues detected")

    print("\n── Motion Type ────────────────────────────────────────")
    print(f"  {analysis.get('motion_type', 'unknown')}")

    print("\n── Lighting Cues ──────────────────────────────────────")
    print(f"  {analysis.get('lighting_cues', 'unknown')}")

    print("\n── Refined Prompt P* (Eq. 4) ──────────────────────────")
    print(analysis.get("refined_prompt", "N/A"))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n  Analysis saved → {args.output_json}")

    # Step 4: Regenerate video (Eq. 5)
    if args.generate:
        print("\n[Step 4] Regenerating video with Veo (Eq. 5)...")
        refined_prompt = analysis.get("refined_prompt", "")
        if not refined_prompt:
            print("Error: No refined prompt available.")
            sys.exit(1)
        try:
            regenerate_video(
                client=client,
                prompt=refined_prompt,
                output_path=args.video_output,
                duration=args.duration,
                aspect_ratio=args.aspect
            )
        except (TimeoutError, RuntimeError) as e:
            print(f"\nError during video generation: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
