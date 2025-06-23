import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from pipeline import FrameExtractor, AudioTranscriber, generate_descriptions, BLIPCaptioner, main_pipeline, process_video
from rag import fact_check_wikipedia_only
from download import SimpleDownloader

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom CORS middleware to ensure headers are always set
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    origin = request.headers.get("origin", "*")
    if request.method == "OPTIONS":
        response = Response(status_code=200)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DOWNLOADS_DIR = Path(__file__).parent / "downloads"

# ──────────────────────────────────────────────────────────────
# Endpoint: Upload Video
# ──────────────────────────────────────────────────────────────
@app.post("/upload_video/")
def upload_video(file: UploadFile = File(...)):
    """Upload a video file for analysis."""
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": str(file_path)}

# ──────────────────────────────────────────────────────────────
# Endpoint: Analyze Video (Async, Concurrent)
# ──────────────────────────────────────────────────────────────
@app.post("/analyze_video/")
def analyze_video(filename: str = Form(...), num_frames: int = Form(10)):
    """Run the full pipeline: concurrent frame/audio extraction, analysis, scene generation, fact-checking."""
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found."})

    # Concurrently extract frames and audio
    frame_extractor = FrameExtractor()
    audio_transcriber = AudioTranscriber()
    
    frames_task = frame_extractor.extract_frames(str(video_path), num_frames)
    audio_task = audio_transcriber.transcribe_video(str(video_path))
    frames, audio_text = frames_task, audio_task

    # Concurrently analyze frames (Gemini/BLIP) and audio
    # First frame: Gemini, others: BLIP (simulate Gemini with BLIP if not available)
    blip = BLIPCaptioner()
    def analyze_frame(frame_path, use_gemini=False):
        if use_gemini:
            # Placeholder: Use BLIP for now, replace with Gemini API if available
            return f"[Gemini] {blip.generate(frame_path)}"
        else:
            return blip.generate(frame_path)
    
    tasks = [analyze_frame(frames[0], True)] + [analyze_frame(f) for f in frames[1:]]
    frame_descriptions = tasks
    
    # Combine frame descriptions
    final_output = " ".join([f"Frame {i+1}: {desc}" for i, desc in enumerate(frame_descriptions)])

    # Scene generation (audio+video or video only)
    if audio_text:
        from pipeline import audio_video_scene_generation
        scene = audio_video_scene_generation(audio_text, final_output)
    else:
        from pipeline import scene_generation_from_frame
        scene = scene_generation_from_frame(final_output)

    # Fact-checking (standard + RAG)
    from pipeline import scene_realism_checking1, scene_realism_checking1_rag
    analysis1_task = scene_realism_checking1(scene)
    analysis2_task = scene_realism_checking1_rag(scene)
    analysis1, analysis2 = analysis1_task, analysis2_task

    # Compose result
    return {
        "scene_description": scene,
        "analysis_1": analysis1,
        "analysis_2": analysis2,
        "frame_descriptions": frame_descriptions,
        "audio_text": audio_text
    }

# ──────────────────────────────────────────────────────────────
# Endpoint: Fact-Check (RAG)
# ──────────────────────────────────────────────────────────────
@app.post("/fact_check/")
def fact_check(claim: str = Form(...)):
    """Fact-check a claim using Wikipedia-based RAG."""
    result = fact_check_wikipedia_only(claim)
    return result

# ──────────────────────────────────────────────────────────────
# Endpoint: Content Moderation (Blur NSFW)
# ──────────────────────────────────────────────────────────────
@app.post("/moderate_content/")
def moderate_content(filename: str = Form(...)):
    """Blur/censor NSFW content in a video (placeholder)."""
    # Placeholder: Just return the same file for now
    # TODO: Integrate NSFW detection and blurring
    return {"moderated_file": filename, "status": "No moderation applied (demo)"}

# ──────────────────────────────────────────────────────────────
# Endpoint: Human Feedback
# ──────────────────────────────────────────────────────────────
@app.post("/feedback/")
def feedback(scene: str = Form(...), agree: bool = Form(...)):
    """Accept human feedback (agree/disagree) and reinforce LLM prompt."""
    # TODO: Store feedback, optionally retrain or reinforce prompt
    return {"status": "Feedback received", "agree": agree}

# ──────────────────────────────────────────────────────────────
# Endpoint: Download Video
# ──────────────────────────────────────────────────────────────
@app.post("/download/")
def download_video(url: str = Form(...)):
    try:
        downloader = SimpleDownloader()
        filename = downloader.download(url)
        if filename:
            # Return only the filename, not the full path
            return {"success": True, "filename": os.path.basename(filename)}
        else:
            return {"success": False, "error": "Download failed"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/analyze_downloaded/")
def analyze_downloaded(filename: str = Form(...), num_frames: int = Form(10)):
    """Analyze a previously downloaded video by filename."""
    video_path = DOWNLOADS_DIR / filename
    print(f"Looking for file: {video_path}")
    print(f"Exists: {video_path.exists()}, Path: {video_path}")
    print(f"Dir listing: {list(DOWNLOADS_DIR.iterdir())}")
    print(f"Filename repr: {repr(filename)}")
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found."})
    # Use the new pipeline_workflow for processing
    from pipeline_workflow import run_full_pipeline
    result = run_full_pipeline(str(video_path), num_frames)
    return result

# ──────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "Backend is running"} 