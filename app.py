"""
Temporal Consistency Studio
Flask backend implementing the paper:
"Enforcing Temporal Consistency in AI-Generated Videos through Analysis-Guided Regeneration"

Pipeline: Upload Video → Gemini Analysis (Eq 2,3,4) → Prompt Refinement → Veo Regeneration (Eq 5)
"""

import os
import json
import time
import base64
import uuid
import re
import threading
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    os.system("pip install opencv-python")
    import cv2

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# In-memory job trackers for async operations
generation_jobs = {}
analysis_jobs = {}


# ──────────────────────────────────────────────
# Core Functions (from paper pipeline)
# ──────────────────────────────────────────────

def extract_frames(video_path: str, n: int = 5) -> list:
    """
    Extract n evenly-spaced frames from a video, resized to max 512px.
    Implements frame sampling for Equations 2,3 of the paper.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    # Evenly spaced positions across entire video
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
    return frames


def frames_to_base64(frames: list) -> list:
    """Convert OpenCV frames to base64-encoded JPEG strings."""
    encoded = []
    for frame in frames:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        encoded.append(base64.b64encode(buffer).decode('utf-8'))
    return encoded


# Model fallback chain — tried in order when quota is exhausted
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
]

# Seconds to wait before switching to the next fallback model
# (gives the API time to recover before we hammer a different model)
MODEL_SWITCH_COOLDOWN = 15

# Max retry attempts per model before moving to the next
MAX_ATTEMPTS_PER_MODEL = 3


def _call_gemini_with_retry(contents: list, max_output_tokens: int = 1024) -> str:
    """
    Call Gemini with automatic retry on 429 rate-limit errors.
    Tries each model in GEMINI_MODELS; for each model retries up to
    MAX_ATTEMPTS_PER_MODEL times, respecting the retryDelay from the API.
    Waits MODEL_SWITCH_COOLDOWN seconds before switching models so we
    don't immediately exhaust the next model's quota either.
    """
    last_error = None

    for model_idx, model in enumerate(GEMINI_MODELS):
        # Brief cooldown before switching to a fallback model
        if model_idx > 0:
            print(f"  Waiting {MODEL_SWITCH_COOLDOWN}s before trying {model}...")
            time.sleep(MODEL_SWITCH_COOLDOWN)

        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
                        response_mime_type="application/json"
                    )
                )
                return response.text.strip(), model

            except ClientError as e:
                last_error = e
                error_msg = str(e)

                if '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                    # Try to parse the retry delay suggested by the API
                    delay = 30  # sensible default
                    delay_match = re.search(r'retryDelay[":\s]+(\d+)s', error_msg)
                    if delay_match:
                        delay = int(delay_match.group(1)) + 5  # add 5s buffer

                    # Exponential backoff: multiply delay by attempt number
                    backoff_delay = delay * (attempt + 1)

                    if attempt < MAX_ATTEMPTS_PER_MODEL - 1:
                        print(
                            f"  [{model}] 429 rate-limit — "
                            f"retrying in {backoff_delay}s "
                            f"(attempt {attempt + 1}/{MAX_ATTEMPTS_PER_MODEL})"
                        )
                        time.sleep(backoff_delay)
                        continue
                    else:
                        # All attempts for this model exhausted
                        print(
                            f"  [{model}] quota exhausted after "
                            f"{MAX_ATTEMPTS_PER_MODEL} attempts — "
                            f"trying next model..."
                        )
                        break
                else:
                    raise  # non-429 error: propagate immediately

    raise RuntimeError(
        f"All Gemini models exhausted their quota. Last error: {last_error}. "
        "Please wait a minute then try again, or check your quota at "
        "https://aistudio.google.com/app/apikey"
    )


def analyze_video(frames_b64: list, feedback: str = None) -> dict:
    """
    Semantic Analysis (Paper Section III, Equations 2, 3, 4).

    Sends frames to Gemini to extract:
    - scene_description: stable scene attributes (Eq. 2 - Scene Representation)
    - detected_objects: consistent object properties (Eq. 3 - Object Tracking)
    - temporal_issues: flickering, color shifts, artifacts
    - refined_prompt: P* — the temporally-enforced generation prompt (Eq. 4)
    - motion_type: camera motion
    - lighting_cues: dominant lighting
    """
    system_instruction = (
        "You are a video temporal consistency analyzer. Analyze these frames from an AI-generated video. "
        "Return ONLY a JSON object with:\n"
        "- scene_description: one paragraph describing stable scene attributes (lighting, background, objects, color palette)\n"
        "- detected_objects: list of main objects with their consistent properties\n"
        "- temporal_issues: list of observed flickering, texture instability, color shifts, or motion artifacts between frames\n"
        "- refined_prompt: a single structured paragraph prompt for video regeneration that enforces: "
        "consistent lighting, stable object textures, smooth motion, coherent color transitions. "
        "This is the P* from the paper's Eq. 4.\n"
        "- motion_type: camera motion description (static, panning, zooming, etc.)\n"
        "- lighting_cues: dominant lighting conditions"
    )

    if feedback:
        system_instruction += (
            f"\n\nUSER REFINEMENT FEEDBACK: The user wants the following changes incorporated into refined_prompt: "
            f"'{feedback}'. Adjust the refined_prompt accordingly while maintaining temporal consistency principles."
        )

    contents = []
    for b64 in frames_b64:
        contents.append(
            types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg")
        )
    contents.append(system_instruction)

    raw, used_model = _call_gemini_with_retry(contents)
    print(f"  Analysis completed with model: {used_model}")

    try:
        result = json.loads(raw)
        result['_model_used'] = used_model
        return result
    except json.JSONDecodeError:
        # Fallback: try to extract JSON block
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                result['_model_used'] = used_model
                return result
            except json.JSONDecodeError:
                pass
        # Last resort: return partial structure with raw text
        return {
            "scene_description": raw,
            "detected_objects": [],
            "temporal_issues": ["Unable to parse structured analysis"],
            "refined_prompt": raw,
            "motion_type": "unknown",
            "lighting_cues": "unknown",
            "_model_used": used_model,
            "_parse_error": True
        }


def regenerate_video(
    prompt: str,
    output_path: str,
    duration: int = 5,
    aspect_ratio: str = "16:9"
) -> str:
    """
    Video Regeneration (Paper Equation 5).
    Sends refined prompt P* to Veo and polls every 10 seconds.
    Timeout: 5 minutes (30 polls × 10s).
    """
    operation = client.models.generate_videos(
        model="veo-2.0-generate-001",
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=duration,
            person_generation="allow_all"
        )
    )

    max_polls = 30  # 5-minute timeout
    poll_count = 0

    while not operation.done:
        if poll_count >= max_polls:
            raise TimeoutError("Video generation timed out after 5 minutes.")
        poll_count += 1
        time.sleep(10)
        operation = client.operations.get(operation)

    if operation.error:
        raise RuntimeError(f"Video generation failed: {operation.error}")

    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    generated_video.video.save(output_path)

    return output_path


# ──────────────────────────────────────────────
# Flask Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Step 1 + 2: Upload video → save file → start async Gemini analysis.
    Returns job_id immediately; client polls /analyze_status/<job_id>.
    """
    if not client:
        return jsonify({'error': 'GOOGLE_API_KEY not configured in .env'}), 500

    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    job_id = uuid.uuid4().hex
    analysis_jobs[job_id] = {
        'status': 'running',
        'message': 'Extracting frames…',
        'result': None,
        'error': None
    }

    def run_analysis():
        try:
            analysis_jobs[job_id]['message'] = 'Extracting 5 frames from video…'
            frames = extract_frames(filepath, n=5)
            if not frames:
                analysis_jobs[job_id]['status'] = 'error'
                analysis_jobs[job_id]['error'] = 'Could not extract frames from video'
                return

            frames_b64 = frames_to_base64(frames)
            analysis_jobs[job_id]['message'] = 'Sending frames to Gemini for analysis…'
            result = analyze_video(frames_b64)
            result['video_path'] = filepath
            result['frame_count'] = len(frames)
            analysis_jobs[job_id]['status'] = 'done'
            analysis_jobs[job_id]['result'] = result
        except Exception as e:
            analysis_jobs[job_id]['status'] = 'error'
            analysis_jobs[job_id]['error'] = str(e)

    threading.Thread(target=run_analysis, daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/analyze_status/<job_id>', methods=['GET'])
def analyze_status(job_id):
    """Poll endpoint for async Gemini analysis."""
    job = analysis_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    resp = {'status': job['status'], 'message': job.get('message', ''), 'error': job.get('error')}
    if job['status'] == 'done':
        resp['result'] = job['result']
    return jsonify(resp)


@app.route('/refine', methods=['POST'])
def refine():
    """
    Step 3 (optional): Re-analyze with user feedback to refine P*.
    Returns job_id immediately; client polls /analyze_status/<job_id>.
    """
    if not client:
        return jsonify({'error': 'GOOGLE_API_KEY not configured in .env'}), 500

    data = request.json
    video_path = data.get('video_path')
    feedback = data.get('feedback', '')

    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found. Please re-upload.'}), 400

    job_id = uuid.uuid4().hex
    analysis_jobs[job_id] = {
        'status': 'running',
        'message': 'Re-analyzing with your feedback…',
        'result': None,
        'error': None
    }

    def run_refine():
        try:
            analysis_jobs[job_id]['message'] = 'Sending feedback to Gemini…'
            frames = extract_frames(video_path, n=5)
            frames_b64 = frames_to_base64(frames)
            result = analyze_video(frames_b64, feedback=feedback)
            result['video_path'] = video_path
            analysis_jobs[job_id]['status'] = 'done'
            analysis_jobs[job_id]['result'] = result
        except Exception as e:
            analysis_jobs[job_id]['status'] = 'error'
            analysis_jobs[job_id]['error'] = str(e)

    threading.Thread(target=run_refine, daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/regenerate', methods=['POST'])
def regenerate():
    """
    Step 4: Send P* to Veo (Eq. 5) and start async generation.
    Returns a job_id to poll for status.
    """
    if not client:
        return jsonify({'error': 'GOOGLE_API_KEY not configured in .env'}), 500

    data = request.json
    prompt = data.get('prompt', '').strip()
    duration = int(data.get('duration', 5))
    aspect_ratio = data.get('aspect_ratio', '16:9')

    if not prompt:
        return jsonify({'error': 'No regeneration prompt provided'}), 400

    # Veo API accepts 5–8 seconds only
    duration = max(5, min(8, duration))

    job_id = uuid.uuid4().hex
    output_filename = f"tcs_{job_id[:8]}.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    generation_jobs[job_id] = {
        'status': 'running',
        'progress': 0,
        'output_filename': None,
        'error': None
    }

    def run_generation():
        try:
            regenerate_video(prompt, output_path, duration, aspect_ratio)
            generation_jobs[job_id]['status'] = 'done'
            generation_jobs[job_id]['output_filename'] = output_filename
        except TimeoutError as e:
            generation_jobs[job_id]['status'] = 'error'
            generation_jobs[job_id]['error'] = str(e)
        except Exception as e:
            generation_jobs[job_id]['status'] = 'error'
            generation_jobs[job_id]['error'] = str(e)

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Poll endpoint for async video generation status."""
    job = generation_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    result = {'status': job['status'], 'error': job.get('error')}
    if job['status'] == 'done' and job.get('output_filename'):
        result['video_url'] = f"/download/{job['output_filename']}"
    return jsonify(result)


@app.route('/download/<filename>')
def download(filename):
    """Stream back the generated video."""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=False, mimetype='video/mp4')
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    print("=" * 60)
    print("  Temporal Consistency Studio")
    print("  Implementing: Analysis-Guided Video Regeneration")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, port=5000)
