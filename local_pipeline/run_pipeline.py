"""
run_pipeline.py — Full Local Pipeline (one command)
=====================================================
Runs all 4 steps end-to-end:
  Video → Frames → LLaVA Analysis → Prompt Refinement → AnimateDiff Generation

Usage:
    python run_pipeline.py --video path/to/video.mp4
    python run_pipeline.py --video path/to/video.mp4 --feedback "make it cinematic"
    python run_pipeline.py --video path/to/video.mp4 \\
                           --vision-model llava:7b \\
                           --text-model   mistral \\
                           --frames       16 \\
                           --steps        30 \\
                           --no-gpu
"""

import argparse
import os
import sys

# Add pipeline dir to path so we can import the step modules directly
sys.path.insert(0, os.path.dirname(__file__))

from _01_extract_frames import extract_frames          # noqa: E402
from _02_analyze_local  import analyze_frames, save_analysis  # noqa: E402
from _03_refine_prompt  import refine_prompt, save_prompt     # noqa: E402
from _04_generate_video import generate_video, load_prompt    # noqa: E402

import time


def banner(text: str):
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


def run_pipeline(
    video_path:    str,
    vision_model:  str  = "llava:13b",
    text_model:    str  = "mistral",
    n_frames:      int  = 8,
    num_gen_frames:int  = 16,
    num_steps:     int  = 25,
    guidance:      float = 7.5,
    seed:          int  = 42,
    feedback:      str  = "",
    use_gpu:       bool = True,
):
    t_start = time.time()
    stem = os.path.splitext(os.path.basename(video_path))[0]

    # ── Step 1: Extract Frames ──────────────────────────────
    banner("Step 1/4 — Extracting Frames")
    frames_dir = os.path.join(os.path.dirname(__file__), "frames", stem)
    frame_paths = extract_frames(video_path, n=n_frames, out_dir=frames_dir)

    # ── Step 2: Analyze with LLaVA ─────────────────────────
    banner("Step 2/4 — Analyzing with LLaVA (local)")
    analysis = analyze_frames(frames_dir, model=vision_model)

    out_dir      = os.path.join(os.path.dirname(__file__), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    analysis_path = os.path.join(out_dir, f"{stem}_analysis.json")
    save_analysis(analysis, frames_dir)

    print(f"  Severity: {analysis.get('severity', '?')}")
    print(f"  Issues found: {len(analysis.get('temporal_issues', []))}")

    # ── Step 3: Refine Prompt ───────────────────────────────
    banner("Step 3/4 — Refining Prompt (local LLM)")
    refined = refine_prompt(analysis_path, model=text_model, user_feedback=feedback)
    prompt_path = save_prompt(refined, analysis_path)

    print(f"\n  P* Prompt preview:")
    print(f"  {refined[:200]}...")

    # ── Step 4: Generate Video ──────────────────────────────
    banner("Step 4/4 — Generating Video (AnimateDiff)")
    out_video = os.path.join(os.path.dirname(__file__), "outputs", f"{stem}_regenerated.mp4")
    os.makedirs(os.path.dirname(out_video), exist_ok=True)

    generate_video(
        prompt=refined,
        output_path=out_video,
        num_frames=num_gen_frames,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    # ── Done ────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner(f"✓ Pipeline complete in {elapsed:.0f}s")
    print(f"  Frames:   {frames_dir}")
    print(f"  Analysis: {analysis_path}")
    print(f"  Prompt:   {prompt_path}")
    print(f"  Video:    {out_video}\n")
    return out_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full local temporal consistency pipeline")
    parser.add_argument("--video",          required=True, help="Input video path")
    parser.add_argument("--vision-model",   default="llava:13b",
                        help="Ollama vision model for analysis (default: llava:13b)")
    parser.add_argument("--text-model",     default="mistral",
                        help="Ollama text model for prompt refinement (default: mistral)")
    parser.add_argument("--extract-frames", type=int, default=8,
                        help="Number of frames to extract (default: 8)")
    parser.add_argument("--gen-frames",     type=int, default=16,
                        help="Number of frames to generate (default: 16)")
    parser.add_argument("--steps",          type=int, default=25,
                        help="Diffusion steps (default: 25)")
    parser.add_argument("--guidance",       type=float, default=7.5,
                        help="Guidance scale (default: 7.5)")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--feedback",       default="",
                        help="Optional style feedback to incorporate in prompt")
    parser.add_argument("--no-gpu",         action="store_true",
                        help="Force CPU offload (use on laptops with low VRAM)")
    args = parser.parse_args()

    run_pipeline(
        video_path     = args.video,
        vision_model   = args.vision_model,
        text_model     = args.text_model,
        n_frames       = args.extract_frames,
        num_gen_frames = args.gen_frames,
        num_steps      = args.steps,
        guidance       = args.guidance,
        seed           = args.seed,
        feedback       = args.feedback,
        use_gpu        = not args.no_gpu,
    )
