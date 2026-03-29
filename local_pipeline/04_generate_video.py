"""
Step 4 — Local Video Generation with AnimateDiff
=================================================
Uses HuggingFace diffusers + AnimateDiff to generate a video
from the refined prompt P* entirely on your local GPU/CPU.

Model requirements (approximate VRAM):
  - animatediff-motion-adapter + stable-diffusion-v1-5  →  ~6 GB VRAM (GPU)
  - With CPU offload (USE_CPU_OFFLOAD=True)             →  ~10 GB RAM  (runs on CPU, slow)

The generated video is saved as an MP4 in local_pipeline/outputs/.

Prerequisites:
    pip install diffusers transformers accelerate torch torchvision
    pip install imageio[ffmpeg] imageio-ffmpeg

Usage:
    # Pass the refined prompt text directly:
    python 04_generate_video.py --prompt "A serene forest clearing..."

    # Or pass the path to the .txt file from step 3:
    python 04_generate_video.py --prompt local_pipeline/analysis/my_video_refined_prompt.txt

    # Full options:
    python 04_generate_video.py \\
        --prompt "A serene forest..." \\
        --frames 16 \\
        --steps  25 \\
        --guidance 7.5 \\
        --seed   42 \\
        --output local_pipeline/outputs/my_video.mp4
"""

import argparse
import os
import time


# ─────────────────────────────────────────────────────────────
# Configurable defaults
# ─────────────────────────────────────────────────────────────

# Set to True to offload model weights to CPU RAM (slower but runs on low VRAM laptops)
USE_CPU_OFFLOAD = True

# AnimateDiff motion adapter checkpoint
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"

# Base Stable Diffusion 1.5 checkpoint
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Output directory
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# ─────────────────────────────────────────────────────────────

def load_prompt(prompt_arg: str) -> str:
    """If prompt_arg is a file path that exists, read it. Otherwise use it as-is."""
    if os.path.isfile(prompt_arg):
        with open(prompt_arg, encoding="utf-8") as f:
            return f.read().strip()
    return prompt_arg.strip()


def build_pipeline(use_cpu_offload: bool = USE_CPU_OFFLOAD):
    import torch
    from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
    from diffusers.utils import export_to_video

    print(f"  Loading motion adapter: {MOTION_ADAPTER_ID}")
    adapter = MotionAdapter.from_pretrained(
        MOTION_ADAPTER_ID,
        torch_dtype=torch.float16,
    )

    print(f"  Loading base model: {BASE_MODEL_ID}")
    pipe = AnimateDiffPipeline.from_pretrained(
        BASE_MODEL_ID,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )

    # Use DDIM scheduler for stable, coherent frame generation
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
    )

    if use_cpu_offload:
        print("  CPU offload enabled (low-VRAM mode) — this will be slow on CPU")
        pipe.enable_model_cpu_offload()
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"  Running on device: {device}")

    # Memory optimisation: attention slicing reduces peak VRAM
    pipe.enable_attention_slicing()

    return pipe


def generate_video(
    prompt:    str,
    output_path: str,
    num_frames:  int   = 16,
    num_steps:   int   = 25,
    guidance:    float = 7.5,
    seed:        int   = 42,
    height:      int   = 512,
    width:       int   = 512,
) -> str:
    import torch
    from diffusers.utils import export_to_video

    # Negative prompt enforces temporal stability
    negative_prompt = (
        "flickering, blurry, low quality, deformed, glitch, artifacts, "
        "inconsistent lighting, color shift, temporal inconsistency, watermark, "
        "text, logo, bad anatomy, distorted"
    )

    generator = torch.Generator().manual_seed(seed)

    pipe = build_pipeline()

    print(f"\n  Generating {num_frames} frames ...")
    print(f"  Steps: {num_steps}  |  Guidance: {guidance}  |  Seed: {seed}")
    print(f"  Prompt: {prompt[:100]}...")

    t0 = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance,
        num_inference_steps=num_steps,
        generator=generator,
        height=height,
        width=width,
    )
    elapsed = time.time() - t0
    print(f"  Generation took {elapsed:.1f}s")

    # Export frames → MP4 (16 fps → 1-second-ish clip for 16 frames)
    frames = output.frames[0]
    fps = num_frames // 2  # ~2 seconds at half-frame-rate feels natural

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    export_to_video(frames, output_path, fps=fps)
    print(f"\n✓ Video saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video locally with AnimateDiff")
    parser.add_argument("--prompt",   required=True,
                        help="Prompt text OR path to refined_prompt.txt from step 3")
    parser.add_argument("--frames",   type=int,   default=16,   help="Number of frames (default: 16)")
    parser.add_argument("--steps",    type=int,   default=25,   help="Diffusion steps (default: 25)")
    parser.add_argument("--guidance", type=float, default=7.5,  help="Guidance scale (default: 7.5)")
    parser.add_argument("--seed",     type=int,   default=42,   help="Random seed (default: 42)")
    parser.add_argument("--height",   type=int,   default=512,  help="Frame height (default: 512)")
    parser.add_argument("--width",    type=int,   default=512,  help="Frame width  (default: 512)")
    parser.add_argument("--output",   default=None,             help="Output MP4 path")
    parser.add_argument("--no-offload", action="store_true",    help="Disable CPU offload (requires 6+ GB VRAM)")
    args = parser.parse_args()

    prompt_text = load_prompt(args.prompt)

    if args.output:
        out_path = args.output
    else:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        ts = int(time.time())
        out_path = os.path.join(DEFAULT_OUT_DIR, f"generated_{ts}.mp4")

    use_offload = not args.no_offload

    generate_video(
        prompt=prompt_text,
        output_path=out_path,
        num_frames=args.frames,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        height=args.height,
        width=args.width,
    )
