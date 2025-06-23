from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import shutil
from download import SimpleDownloader
from pipeline import process_video

app = FastAPI(title="Fake Video Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

downloader = SimpleDownloader()

def analyze_video(video_path):
    """Process video through the pipeline and return analysis results"""
    try:
        # Process video through pipeline
        result = process_video(video_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to Fake Video Detection API"}

@app.post("/analyze-url")
async def analyze_video_url(url: str):
    """Analyze video from URL"""
    try:
        # Download video
        success = downloader.download(url)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to download video")
            
        # Get downloaded file path
        downloaded_file = list(downloader.download_path.glob('*.mp4'))[-1]
        
        # Analyze video
        result = analyze_video(str(downloaded_file))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-file")
async def analyze_video_file(file: UploadFile = File(...)):
    """Analyze uploaded video file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Analyze video
            result = analyze_video(temp_file.name)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
