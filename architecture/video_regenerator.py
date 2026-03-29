"""
pipeline/video_regenerator.py
Step 4: Regenerate video using Google Veo 2 API (veo-2.0-generate-001).
100% cloud-based — no local model weights required.
"""

import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def regenerate_video(
    prompt: str,
    output_path: str,
    duration: int = 4,
    aspect_ratio: str = "16:9",
) -> str:
    """
    Regenerate a temporally-consistent video via Google Veo 2.

    Args:
        prompt:       Refined prompt P* from Gemini analysis.
        output_path:  Absolute path where the output .mp4 will be saved.
        duration:     Video duration in seconds (4 or 8).
        aspect_ratio: "16:9" or "9:16".

    Returns:
        output_path on success.

    Raises:
        RuntimeError: If Veo API fails or times out.
        ImportError:  If google-genai SDK is not installed.
    """
    if not _GENAI_AVAILABLE:
        raise ImportError(
            "google-genai SDK is not installed. Run: pip install google-genai"
        )

    if not config.GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file."
        )

    client = genai.Client(api_key=config.GOOGLE_API_KEY)

    print(f"  → Submitting Veo 2 generation job "
          f"(duration={duration}s, aspect={aspect_ratio}) …")

    operation = client.models.generate_videos(
        model=config.VEO_MODEL,
        prompt=prompt,
        config=genai_types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=duration,
            number_of_videos=1,
            person_generation="allow_adult",
            enhance_prompt=True,
        ),
    )

    # Poll until done
    attempt = 0
    while not operation.done:
        attempt += 1
        if attempt > config.VEO_MAX_POLL_ATTEMPTS:
            raise TimeoutError(
                f"Veo 2 timed out after {attempt * config.VEO_POLL_INTERVAL}s. "
                "Try again or reduce the duration."
            )
        print(f"  … Polling Veo 2 ({attempt}/{config.VEO_MAX_POLL_ATTEMPTS}) …")
        time.sleep(config.VEO_POLL_INTERVAL)
        operation = client.operations.get(operation)

    print(f"  → Veo 2 job complete after {attempt} poll(s).")

    # Extract video bytes
    generated_videos = operation.response.generated_videos
    if not generated_videos:
        raise RuntimeError("Veo 2 returned no videos in response.")

    video_obj = generated_videos[0].video
    client.files.download(file=video_obj)
    video_bytes = video_obj.video_bytes

    if not video_bytes:
        raise RuntimeError("Veo 2 returned empty video_bytes after download.")

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(video_bytes)

    size_mb = len(video_bytes) / (1024 * 1024)
    print(f"  → Saved {size_mb:.2f} MB video to {output_path}")
    return output_path
