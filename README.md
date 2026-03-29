# Temporal Consistency Studio

A web application implementing the pipeline from the research paper:

**"Enforcing Temporal Consistency in AI-Generated Videos through Analysis-Guided Regeneration"**  
*Dr. Anil Kumar, Phaneesh Mushetty, Sai Teja Nimmala — CMR College of Engineering and Technology, ICDSCNC 2026*

---

## What This Project Does

AI-generated videos suffer from temporal artifacts — flickering textures, unstable motion, inconsistent lighting across frames. This app implements the paper's retraining-free post-processing pipeline to fix those artifacts:

1. Upload an AI-generated video
2. The system extracts frames and analyzes scene semantics + temporal issues
3. A refined structured prompt P\* is generated (Eq. 4 from the paper)
4. A video generation model regenerates the video using P\* as a conditioning signal (Eq. 5)
5. The temporally consistent output is returned for download

No model retraining. No parameter modification. The base generator stays frozen throughout.

---

## Architecture

```
User Browser
    │
    ▼
Flask App (app.py)
    │
    ├──► Analysis Backend ──► Scene semantics, temporal issues, refined prompt P*
    │
    └──► Video Generation Backend ──► Regenerated temporally consistent video
```

The analysis and generation backends are **swappable**. The app supports two modes:

---

## Backend Modes

### Mode 1 — Google APIs (Cloud, No Local Hardware Required)

Uses Google Gemini for frame analysis and Google Veo for video generation. No GPU needed. Runs on any machine.

| Stage | Model |
|---|---|
| Frame analysis + prompt refinement (Eq. 2–4) | `gemini-2.0-flash` |
| Video regeneration (Eq. 5) | `veo-2.0-generate-001` |

**Setup:**
```bash
pip install flask opencv-python google-genai python-dotenv
```

`.env`:
```
GOOGLE_API_KEY=your_key_here
BACKEND=google
```

### Mode 2 — Local Models (Full Research Pipeline)

Uses CLIP for semantic embeddings, RAFT for optical flow-based temporal alignment, and ModelScope as the base video generator. Requires a GPU.

| Stage | Model | Paper Reference |
|---|---|---|
| Semantic embedding Φ(fᵢ) — Eq. 2 | CLIP (ViT-B/32) | §III |
| Optical flow / motion features | RAFT | §III |
| Video regeneration G(P\*) — Eq. 5 | ModelScope | §III |

**Setup:**
```bash
pip install flask opencv-python torch torchvision transformers diffusers python-dotenv
```

`.env`:
```
BACKEND=local
```

### Mode 3 — Claude API (Anthropic)

Uses Claude's vision capability for frame analysis and prompt refinement. Pair with any video generation backend for the regeneration step.

| Stage | Model |
|---|---|
| Frame analysis + prompt refinement (Eq. 2–4) | `claude-sonnet-4-5` or `claude-opus-4-5` |
| Video regeneration (Eq. 5) | Google Veo or local ModelScope |

**Setup:**
```bash
pip install flask opencv-python anthropic python-dotenv
```

`.env`:
```
ANTHROPIC_API_KEY=your_key_here
BACKEND=claude
VIDEO_BACKEND=google   # or local
GOOGLE_API_KEY=your_key_here   # only if VIDEO_BACKEND=google
```

**Why Claude:** Claude's vision model performs strong scene understanding, can reason about temporal artifacts across multiple frames simultaneously, and produces structured prompt refinements well-suited for video conditioning.

---

## Paper → Code Mapping

| Paper | Equation | Implementation |
|---|---|---|
| Frame decomposition | Eq. 1 | `extract_frames()` in `app.py` |
| Semantic embedding per frame | Eq. 2 | Gemini / CLIP / Claude vision call |
| Global semantic aggregation | Eq. 3 | Averaged embeddings or aggregated Gemini analysis |
| Refined prompt P\* generation | Eq. 4 | `analyze_video()` — returns `refined_prompt` field |
| Video regeneration with frozen G | Eq. 5 | `regenerate_video()` — Veo or ModelScope call |
| Temporal loss minimization | Eq. 6 | Optical flow alignment (local mode) / Gemini-guided (cloud mode) |
| Total objective | Eq. 7 | λ = 0.6, α = 0.4, β = 0.3 (empirically validated) |

---

## Algorithm

```
Input:  AI-generated video V = {f₁, f₂, ..., fₙ}
Output: Temporally consistent video V'

1.  Decompose V into individual frames {f₁, ..., fₙ}
2.  For each frame fᵢ:
      a. Extract semantic embedding sᵢ = Φ(fᵢ)  [CLIP]
      b. Extract motion features mᵢ from (fᵢ, fᵢ₊₁)
3.  Aggregate global representation: S = (1/n) Σ sᵢ
4.  Estimate inter-frame motion properties and technical attributes T
5.  Generate refined prompt: P* = Ψ(S, T)
6.  Regenerate video with frozen model: V' = G(P*)  [ModelScope]
7.  For each consecutive frame pair (v'ᵢ, v'ᵢ₊₁) in V':
      a. Apply motion-guided temporal alignment
      b. Minimize temporal loss: Lₜ = ‖v'ᵢ₊₁ − warp(v'ᵢ, mᵢ)‖²
8.  Apply iterative frame correction to remove residual flicker
9.  Minimize total objective: L = Lₜ + λ · L_spatial
10. Return V'
```

### Results

The proposed framework showed consistent improvement across all five metrics compared to:
- **Blind Video Consistency (BVC)** — optical flow-based alignment and smoothing
- **Rolling Guidance Refinement** — iterative consistency correction

Warping Error was reduced by **28%**, validating the motion-aware alignment stage. FVD drop confirmed the regenerated video distribution moved closer to real video statistics.

---

## Comparison with Existing Methods

| Method | Category | Retraining | Limitation |
|---|---|---|---|
| Video Diffusion with Temporal Attention | Model-level | Yes | High training cost, model-specific |
| Space-Time Transformer Generators | Model-level | Yes | Requires large-scale datasets |
| Motion-Guided Diffusion | Model-level | Yes | Error propagation in complex scenes |
| Blind Video Consistency (BVC) | Post-processing | No | Struggles with large motion changes |
| Deep Video Prior (DVP) | Post-processing | No | Computationally expensive |
| Rolling Guidance Refinement | Post-processing | No | Slow for long videos |
| **Proposed Framework** | **Post-generation** | **No** | Depends on base generator quality |

---

## Key Properties

- **Model-agnostic** — works with any frozen video generation architecture
- **Retraining-free** — no parameter updates to the base generator
- **Three contributions:**
  1. Semantic-guided prompt refinement pipeline (aggregates scene-level features to reduce regeneration ambiguity)
  2. Motion-semantically aware temporal alignment (inhibits inter-frame difference)
  3. Iterative frame correction cycle (eliminates residual flicker while preserving spatial faithfulness)

---

## Dependencies

```bash
pip install torch torchvision opencv-python transformers diffusers
```

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Core deep learning |
| `opencv-python` | Frame extraction and processing |
| `transformers` | CLIP semantic encoder |
| `diffusers` | ModelScope video generation |

---


