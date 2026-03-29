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

## Project Structure

```
temporal-consistency-studio/
├── app.py                  # Flask backend — all routes and API logic
├── video_prompt.py         # CLI version of the same pipeline
├── templates/
│   └── index.html          # Single-page UI
├── static/
│   └── style.css
├── outputs/                # Generated videos saved here
├── uploads/                # Temp storage, cleaned after processing
├── backends/
│   ├── google_backend.py   # Gemini + Veo implementation
│   ├── claude_backend.py   # Claude + Veo/ModelScope implementation
│   └── local_backend.py    # CLIP + RAFT + ModelScope implementation
└── .env
```

---

## Flask Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Main UI |
| `/analyze` | POST | Upload video → extract frames → return analysis JSON |
| `/refine` | POST | Re-analyze with user feedback |
| `/regenerate` | POST | Send P\* to video generator → poll → save output |
| `/download/<filename>` | GET | Stream back generated video |

---

## The 4-Step User Flow

**Step 1 — Upload**  
Drag and drop an AI-generated `.mp4` (up to 100MB). 5 evenly-spaced frames are extracted from the video.

**Step 2 — Analyze**  
Frames are sent to the analysis backend. Returns:
- `scene_description` — stable scene attributes
- `detected_objects` — objects with consistent properties
- `temporal_issues` — identified flickering, color shifts, texture instability
- `refined_prompt` — the structured P\* conditioning signal
- `motion_type` — camera motion description
- `lighting_cues` — dominant lighting conditions

**Step 3 — Refine (optional)**  
User can add feedback to further guide prompt refinement. The backend re-runs analysis with the feedback appended.

**Step 4 — Regenerate**  
P\* is sent to the video generation backend. User selects duration (4s / 8s) and aspect ratio (16:9 / 9:16). The regenerated video is returned for preview and download.

---

## CLI Usage

```bash
# Google backend
python video_prompt.py --video_path myvideo.mp4 --backend google --generate-video --duration 6

# Claude backend
python video_prompt.py --video_path myvideo.mp4 --backend claude --generate-video --duration 4

# Local models
python video_prompt.py --video_path myvideo.mp4 --backend local --generate-video --aspect-ratio 16:9

# Save extracted prompt only
python video_prompt.py --video_path myvideo.mp4 --backend google --output prompt.txt
```

---

## Results (from the paper)

Evaluated on 200 clips (512×512, 64 frames) generated by Stable Video Diffusion v2 using UCF-101 action prompts.

| Metric | Baseline | Proposed Framework | Improvement |
|---|---|---|---|
| PSNR ↑ | — | Higher | Reduced frame-level noise |
| SSIM ↑ | — | Higher | Better structural consistency |
| LPIPS ↓ | — | Lower | Improved perceptual consistency |
| Warping Error ↓ | — | −28% | Better inter-frame alignment |
| FVD ↓ | — | Lower | Closer to real video distribution |

Outperforms baselines: Blind Video Consistency (BVC) and Rolling Guidance Refinement across all five metrics.

---

## Requirements Summary

| Dependency | All Modes | Google Mode | Claude Mode | Local Mode |
|---|---|---|---|---|
| `flask` | ✓ | ✓ | ✓ | ✓ |
| `opencv-python` | ✓ | ✓ | ✓ | ✓ |
| `python-dotenv` | ✓ | ✓ | ✓ | ✓ |
| `google-genai` | | ✓ | optional | |
| `anthropic` | | | ✓ | |
| `torch`, `torchvision` | | | | ✓ |
| `transformers`, `diffusers` | | | | ✓ |

---

## Running the App

```bash
# Install dependencies for your chosen backend
pip install -r requirements-google.txt   # or requirements-claude.txt / requirements-local.txt

# Set up .env with your backend choice and API keys
cp .env.example .env

# Run
python app.py
# → http://localhost:5000
```

---

## Citation

If you use this implementation, please cite the original paper:

```
Dr. Anil Kumar, Phaneesh Mushetty, Sai Teja Nimmala.
"Enforcing Temporal Consistency in AI-Generated Videos through Analysis-Guided Regeneration."
ICDSCNC 2026, CMR College of Engineering and Technology, Hyderabad.
```
