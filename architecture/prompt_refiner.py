"""
pipeline/prompt_refiner.py
Step 4: Refine generation prompt using Google Gemini API.
Combines semantic features (S) and technical motion attributes (T) into a
structured, temporally-consistent generation prompt P*.
"""

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def _build_system_instruction() -> str:
    return (
        "You are a video generation prompt engineer specializing in temporal consistency. "
        "Your task is to create a precise, structured video generation prompt that ensures "
        "object consistency, color stability, and smooth motion continuity throughout a video. "
        "The prompt MUST be a single paragraph, max 150 words, written as a direct description "
        "of the video content. Do NOT include instructions or meta-commentary."
    )


def _build_user_message(semantics: dict, motion_info: dict) -> str:
    labels   = semantics.get("labels",  [])[:8]
    objects  = semantics.get("objects", [])[:6]
    colors   = semantics.get("colors",  [])[:4]
    motion   = motion_info.get("type",           "unknown")
    avg_mag  = motion_info.get("avg_magnitude",   0.0)
    brightness = semantics.get("brightness", 128)

    color_desc = ", ".join(
        [f"{c['hex']} (R:{c['r']} G:{c['g']} B:{c['b']})" for c in colors]
    ) if colors else "neutral tones"

    scene_desc  = ", ".join(labels)  if labels  else "general scene"
    object_desc = ", ".join(objects) if objects else "no specific objects"
    motion_desc = {
        "static":             "camera is completely still, no camera movement",
        "panning_horizontal": f"smooth horizontal camera pan (flow magnitude ~{avg_mag:.1f})",
        "panning_vertical":   f"smooth vertical camera pan (flow magnitude ~{avg_mag:.1f})",
        "zooming":            f"smooth zoom motion (flow magnitude ~{avg_mag:.1f})",
        "complex":            f"dynamic camera movement (flow magnitude ~{avg_mag:.1f})",
    }.get(motion, "natural camera movement")

    brightness_desc = (
        "bright, well-lit scene"     if brightness > 180 else
        "dimly lit, atmospheric scene" if brightness < 80  else
        "naturally lit scene"
    )

    return (
        f"Generate a temporally consistent video prompt based on this analysis:\n\n"
        f"SCENE LABELS: {scene_desc}\n"
        f"DETECTED OBJECTS: {object_desc}\n"
        f"DOMINANT COLORS: {color_desc}\n"
        f"LIGHTING: {brightness_desc} (avg brightness: {brightness:.0f}/255)\n"
        f"MOTION: {motion_desc}\n\n"
        f"Requirements for the prompt:\n"
        f"1. All objects must remain visually consistent throughout the video\n"
        f"2. Colors must stay stable (no color flicker or drift)\n"
        f"3. Motion must be smooth and physically plausible: {motion_desc}\n"
        f"4. The video style must be photorealistic and cinematic\n\n"
        f"Write only the final video generation prompt (no labels, no lists)."
    )


def refine_prompt(semantics: dict, motion_info: dict) -> str:
    """
    Use Gemini API to generate a refined, temporally-consistent video prompt.

    Args:
        semantics:   Dict from semantic_analyzer.analyze_semantics()
        motion_info: Dict with keys 'type' and 'avg_magnitude'

    Returns:
        Refined prompt string P*
    """
    if not _GENAI_AVAILABLE or not config.GOOGLE_GEMINI_API_KEY:
        print("  ⚠️  Gemini unavailable — using fallback prompt construction")
        return _fallback_prompt(semantics, motion_info)

    try:
        genai.configure(api_key=config.GOOGLE_GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=config.GEMINI_TEXT_MODEL,
            system_instruction=_build_system_instruction(),
        )
        user_msg = _build_user_message(semantics, motion_info)
        response = model.generate_content(
            user_msg,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=300,
            ),
        )
        prompt = response.text.strip()
        print(f"  → Gemini refined prompt ({len(prompt)} chars)")
        return prompt

    except Exception as e:
        print(f"  ⚠️  Gemini error: {e} — using fallback")
        return _fallback_prompt(semantics, motion_info)


def _fallback_prompt(semantics: dict, motion_info: dict) -> str:
    """
    Build a deterministic prompt without calling any API.
    Used as graceful fallback.
    """
    labels   = semantics.get("labels",  ["a scene"])[:4]
    objects  = semantics.get("objects", [])[:3]
    colors   = semantics.get("colors",  [])[:2]
    motion   = motion_info.get("type", "static")
    brightness = semantics.get("brightness", 128)

    scene    = " and ".join(labels)
    obj_part = f" featuring {', '.join(objects)}" if objects else ""
    col_part = (
        f", with dominant colors {', '.join(c['hex'] for c in colors)}"
        if colors else ""
    )
    motion_part = {
        "static":             "The camera remains perfectly still",
        "panning_horizontal": "The camera pans smoothly from left to right",
        "panning_vertical":   "The camera tilts smoothly upward",
        "zooming":            "The camera zooms in gradually",
        "complex":            "The camera moves dynamically",
    }.get(motion, "The camera moves naturally")

    light_part = (
        "in bright, natural lighting" if brightness > 180 else
        "in moody, atmospheric lighting" if brightness < 80 else
        "in natural, balanced lighting"
    )

    return (
        f"A cinematic, photorealistic video of {scene}{obj_part}{col_part}, "
        f"{light_part}, with temporally consistent colors and objects. "
        f"{motion_part} throughout the entire video. "
        f"No flickering, no color drift, smooth motion continuity from start to finish, "
        f"high detail, 4K quality."
    )
