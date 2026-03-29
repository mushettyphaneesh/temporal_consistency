"""
pipeline/semantic_analyzer.py
Step 2: Semantic analysis using Google Cloud Vision API.
Falls back to colour histogram + basic stats if no API key available.
"""

import base64
import json
import requests
import cv2
import numpy as np
from typing import List, Dict, Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─── Google Vision API helpers ───────────────────────────────────────────────

def _encode_image_b64(frame: np.ndarray) -> str:
    """Encode a BGR numpy frame to base64-encoded JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _call_vision_api(frame: np.ndarray) -> Dict[str, Any]:
    """
    Call Google Cloud Vision API's annotateImage endpoint.
    Returns raw API response dict, or empty dict on failure.
    """
    if not config.GOOGLE_VISION_API_KEY:
        return {}

    url = (
        "https://vision.googleapis.com/v1/images:annotate"
        f"?key={config.GOOGLE_VISION_API_KEY}"
    )
    payload = {
        "requests": [{
            "image": {"content": _encode_image_b64(frame)},
            "features": [
                {"type": "LABEL_DETECTION",      "maxResults": config.MAX_VISION_LABELS},
                {"type": "OBJECT_LOCALIZATION",  "maxResults": config.MAX_VISION_OBJECTS},
                {"type": "IMAGE_PROPERTIES",     "maxResults": 1},
                {"type": "SAFE_SEARCH_DETECTION","maxResults": 1},
            ],
        }]
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json().get("responses", [{}])[0]
    except Exception as e:
        print(f"  ⚠️  Vision API error: {e}")
        return {}


# ─── Fallback semantic extraction (no API) ───────────────────────────────────

def _fallback_analyze(frame: np.ndarray) -> Dict[str, Any]:
    """Simple colour-based fallback when Vision API is unavailable."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    avg_color = rgb.mean(axis=(0, 1)).tolist()
    brightness = float(rgb.mean())
    return {
        "labels":  ["scene"],
        "objects": [],
        "colors":  [{"r": int(avg_color[0]), "g": int(avg_color[1]), "b": int(avg_color[2]),
                     "score": 1.0, "hex": "#{:02x}{:02x}{:02x}".format(
                         int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))}],
        "brightness": brightness,
    }


# ─── Public function ─────────────────────────────────────────────────────────

def analyze_semantics(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Run semantic analysis over a list of frames.
    Samples every N frames (config.VISION_SAMPLE_EVERY_N_FRAMES) to save API calls.

    Returns aggregated semantic descriptor S:
    {
        "labels":      list of label strings (most frequent first),
        "objects":     list of object name strings,
        "colors":      list of dominant color dicts {hex, r, g, b, score},
        "brightness":  average scene brightness 0-255,
        "raw_per_frame": list of per-sampled-frame raw results,
    }
    """
    sample_every = config.VISION_SAMPLE_EVERY_N_FRAMES
    sampled = frames[::sample_every]
    if not sampled:
        sampled = frames[:1]

    using_api = bool(config.GOOGLE_VISION_API_KEY)
    print(f"  → Analyzing {len(sampled)} sampled frames "
          f"({'Vision API' if using_api else 'fallback'})")

    label_counts: Dict[str, float]  = {}
    object_counts: Dict[str, float] = {}
    all_colors: List[Dict]           = []
    brightness_vals: List[float]     = []
    raw_results: List[Dict]          = []

    for frame in sampled:
        if using_api:
            result = _call_vision_api(frame)
        else:
            result = {}

        if result:
            # Labels
            for ann in result.get("labelAnnotations", []):
                desc  = ann.get("description", "")
                score = ann.get("score", 0.0)
                label_counts[desc] = label_counts.get(desc, 0.0) + score

            # Objects
            for obj in result.get("localizedObjectAnnotations", []):
                name  = obj.get("name", "")
                score = obj.get("score", 0.0)
                object_counts[name] = object_counts.get(name, 0.0) + score

            # Colors
            props = result.get("imagePropertiesAnnotation", {})
            for color_info in props.get("dominantColors", {}).get("colors", []):
                c = color_info.get("color", {})
                r, g, b = int(c.get("red", 0)), int(c.get("green", 0)), int(c.get("blue", 0))
                all_colors.append({
                    "r": r, "g": g, "b": b,
                    "hex": f"#{r:02x}{g:02x}{b:02x}",
                    "score": color_info.get("score", 0.0),
                })
            raw_results.append(result)
        else:
            fb = _fallback_analyze(frame)
            for lbl in fb["labels"]:
                label_counts[lbl] = label_counts.get(lbl, 0.0) + 1.0
            all_colors.extend(fb["colors"])
            brightness_vals.append(fb["brightness"])

        # Brightness from frame
        brightness_vals.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))

    # Sort labels and objects by cumulative score
    sorted_labels  = sorted(label_counts.items(),  key=lambda x: -x[1])
    sorted_objects = sorted(object_counts.items(), key=lambda x: -x[1])

    # Deduplicate colors by taking top N unique hex values
    seen_hex: set = set()
    unique_colors = []
    for c in sorted(all_colors, key=lambda x: -x["score"]):
        if c["hex"] not in seen_hex:
            seen_hex.add(c["hex"])
            unique_colors.append(c)
        if len(unique_colors) >= config.MAX_VISION_COLORS:
            break

    return {
        "labels":        [l[0] for l in sorted_labels[:config.MAX_VISION_LABELS]],
        "label_scores":  {l[0]: l[1] for l in sorted_labels[:config.MAX_VISION_LABELS]},
        "objects":       [o[0] for o in sorted_objects[:config.MAX_VISION_OBJECTS]],
        "object_scores": {o[0]: o[1] for o in sorted_objects[:config.MAX_VISION_OBJECTS]},
        "colors":        unique_colors,
        "brightness":    float(np.mean(brightness_vals)) if brightness_vals else 128.0,
        "raw_per_frame": raw_results,
        "num_sampled":   len(sampled),
    }
