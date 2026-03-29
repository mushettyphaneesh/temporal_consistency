"""
Step 3 — Prompt Refinement with Local LLM (Ollama)
====================================================
Takes the analysis JSON from step 2 and uses a local text LLM
(Mistral / Llama 3 / Phi-3) to refine the regeneration prompt P*.

The refinement incorporates the detected temporal issues and
enforces consistency constraints in the final prompt.

Prerequisites:
    ollama pull mistral          (fast, ~4GB)
    ollama pull llama3:8b        (better quality, ~5GB)
    ollama pull phi3:mini        (very fast on CPU, ~2GB)

Output: local_pipeline/analysis/<stem>_refined_prompt.txt

Usage:
    python 03_refine_prompt.py --analysis local_pipeline/analysis/my_video_analysis.json
    python 03_refine_prompt.py --analysis local_pipeline/analysis/my_video_analysis.json \
                               --model mistral \
                               --feedback "make it look more cinematic, golden hour"
"""

import argparse
import json
import os
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"


def build_refinement_prompt(analysis: dict, user_feedback: str = "") -> str:
    issues = "\n".join(f"  - {i}" for i in analysis.get("temporal_issues", []))
    objects = ", ".join(analysis.get("detected_objects", []))
    base_prompt = analysis.get("refined_prompt", "")

    feedback_section = ""
    if user_feedback.strip():
        feedback_section = f"""
USER STYLE FEEDBACK (incorporate this into the refined prompt):
  {user_feedback}
"""

    return f"""You are an expert at writing video generation prompts that enforce temporal consistency.

=== Current Analysis ===
Scene description: {analysis.get("scene_description", "")}
Objects: {objects}
Motion type: {analysis.get("motion_type", "unknown")}
Lighting: {analysis.get("lighting_cues", "unknown")}
Severity of issues: {analysis.get("severity", "unknown")}

=== Detected Temporal Issues ===
{issues}

=== Current Draft Prompt (P*) ===
{base_prompt}
{feedback_section}
=== Your Task ===
Rewrite the prompt above into a single, detailed paragraph optimized for AI video generation.
The rewritten prompt MUST:
1. Explicitly describe stable lighting that does NOT change
2. Enumerate objects with exact, locked visual properties (color, texture, material)
3. Describe smooth, consistent motion with no sudden jumps
4. Address each temporal issue above with a concrete constraint
5. Use cinematic language suitable for a video diffusion model
6. Be 3-5 sentences, dense with visual detail

Return ONLY the refined prompt text. No preamble, no explanation, no quotes."""


def refine_prompt(analysis_path: str, model: str = "mistral", user_feedback: str = "") -> str:
    with open(analysis_path) as f:
        analysis = json.load(f)

    prompt_text = build_refinement_prompt(analysis, user_feedback)
    print(f"  Using model: {model}")
    print(f"  Temporal issues to address: {len(analysis.get('temporal_issues', []))}")

    payload = {
        "model":  model,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 512,
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve\n"
            "And a text model is pulled:\n"
            "  ollama pull mistral"
        )

    refined = resp.json().get("response", "").strip()
    # Strip surrounding quotes if model added them
    refined = refined.strip('"\'')
    return refined


def save_prompt(refined: str, analysis_path: str) -> str:
    stem     = os.path.splitext(os.path.basename(analysis_path))[0].replace("_analysis", "")
    out_dir  = os.path.dirname(analysis_path)
    out_path = os.path.join(out_dir, f"{stem}_refined_prompt.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(refined)
    print(f"\n✓ Refined prompt saved to: {out_path}")

    # Also update the analysis JSON with the new prompt
    with open(analysis_path) as f:
        analysis = json.load(f)
    analysis["refined_prompt_v2"] = refined
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine prompt with local LLM via Ollama")
    parser.add_argument("--analysis",  required=True, help="Path to analysis JSON from step 2")
    parser.add_argument("--model",     default="mistral", help="Ollama text model (default: mistral)")
    parser.add_argument("--feedback",  default="", help="Optional user style feedback")
    args = parser.parse_args()

    refined  = refine_prompt(args.analysis, args.model, args.feedback)
    out_path = save_prompt(refined, args.analysis)

    print("\n=== Refined Prompt (P*) ===")
    print(refined)
    print(f"\nNext step:")
    print(f"  python 04_generate_video.py --prompt \"{out_path}\"")
