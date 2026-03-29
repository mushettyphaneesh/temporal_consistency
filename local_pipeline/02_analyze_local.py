"""
Step 2 — Local Visual Analysis with Ollama + LLaVA
====================================================
Sends extracted frames to a locally-running LLaVA model via Ollama
and returns a structured JSON analysis of temporal consistency issues.

Prerequisites:
    1. Install Ollama:  https://ollama.com/download
    2. Pull the model:  ollama pull llava:13b
                        (or llava:7b for lower VRAM)
    3. Ollama must be running: ollama serve

Output: local_pipeline/analysis/<video_stem>_analysis.json

Usage:
    python 02_analyze_local.py --frames local_pipeline/frames/my_video
    python 02_analyze_local.py --frames local_pipeline/frames/my_video --model llava:7b
"""

import argparse
import base64
import json
import os
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

ANALYSIS_PROMPT = """You are a video temporal consistency expert. I will show you frames from an AI-generated video.
Analyze them for temporal consistency issues and return ONLY a JSON object with these exact keys:

{
  "scene_description": "one paragraph describing stable scene attributes: lighting, background, color palette, setting",
  "detected_objects": ["list", "of", "main", "objects", "with", "consistent", "properties"],
  "temporal_issues": ["list", "every", "flickering", "texture instability", "color shift", "motion artifact", "you observe between frames"],
  "motion_type": "camera motion: static | panning | zooming | handheld | etc.",
  "lighting_cues": "dominant lighting conditions and any changes across frames",
  "severity": "low | medium | high",
  "refined_prompt": "A single detailed paragraph prompt for video regeneration that enforces: consistent lighting, stable object textures, smooth motion, coherent color transitions. This is the temporally-enforced P* prompt."
}

Return ONLY the JSON, no markdown fences, no extra text."""


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_frames(frame_dir: str, model: str = "llava:13b") -> dict:
    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not frame_paths:
        raise FileNotFoundError(f"No image files found in: {frame_dir}")

    print(f"  Found {len(frame_paths)} frames in: {frame_dir}")
    print(f"  Sending to Ollama model: {model}")

    images_b64 = [encode_image(p) for p in frame_paths]

    payload = {
        "model":  model,
        "prompt": ANALYSIS_PROMPT,
        "images": images_b64,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve\n"
            "And the model is pulled:\n"
            "  ollama pull llava:13b"
        )

    raw = resp.json().get("response", "").strip()
    print(f"\n  Raw LLaVA response ({len(raw)} chars)")

    # Try to parse JSON (strip markdown fences if model added them)
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        # Extract first {...} block
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
        else:
            result = {
                "scene_description": raw,
                "detected_objects": [],
                "temporal_issues": ["Failed to parse structured output — raw text stored in scene_description"],
                "motion_type": "unknown",
                "lighting_cues": "unknown",
                "severity": "unknown",
                "refined_prompt": raw,
                "_parse_error": True,
            }

    result["_model"] = model
    result["_frames"] = frame_paths
    return result


def save_analysis(result: dict, frame_dir: str) -> str:
    stem     = os.path.basename(frame_dir.rstrip("/\\"))
    out_dir  = os.path.join(os.path.dirname(__file__), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Analysis saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze frames with local LLaVA via Ollama")
    parser.add_argument("--frames", required=True, help="Directory containing extracted frames")
    parser.add_argument("--model",  default="llava:13b", help="Ollama model to use (default: llava:13b)")
    args = parser.parse_args()

    result   = analyze_frames(args.frames, args.model)
    out_path = save_analysis(result, args.frames)

    print("\n=== Analysis Summary ===")
    print(f"  Scene:   {result.get('scene_description', '')[:120]}...")
    print(f"  Issues:  {result.get('temporal_issues', [])}")
    print(f"  Severity:{result.get('severity', '?')}")
    print(f"  Prompt:  {result.get('refined_prompt', '')[:120]}...")
    print(f"\nNext step:")
    print(f"  python 03_refine_prompt.py --analysis {out_path}")
