# Video Fake Detection Pipeline

This project is a backend service designed to analyze video files and determine their authenticity. It implements a multi-stage pipeline that processes visual and audio data through a series of AI and machine learning models to produce a final verdict on whether a video is real or fake.

## Analysis Workflow

The pipeline processes videos as shown in the diagram below. The core logic involves separating the video into its constituent frames and audio, analyzing each component, and then synthesizing the results for a final conclusion.

```
                                 ┌─────────────────┐
                                 │   INPUT VIDEO   │
                                 │   (MP4/AVI/etc) │
                                 └─────────┬───────┘
                                           │
                           ┌───────────────┴───────────────┐
                           │                               │
                           ▼                               ▼
                  ┌─────────────────┐                ┌─────────────────┐
                  │ FRAME EXTRACTOR │                │ AUDIO EXTRACTOR │
                  │ FrameExtractor  │                │AudioTranscriber │
                  └─────────┬───────┘                └─────────┬───────┘
                            │                                  │
                            ▼                                  ▼
                  ┌─────────────────┐                ┌─────────────────┐
                  │ Extract N Frames│                │ Extract Audio & │
                  │ (default: 10)   │                │ Transcribe to   │
                  │ Save as JPG     │                │ Text (Google)   │
                  └─────────┬───────┘                └─────────┬───────┘
                            │                                  │
                            ▼                                  │
                  ┌─────────────────┐                         │
                  │ BLIP CAPTIONER  │                         │
                  │ BLIPCaptioner   │                         │
                  └─────────┬───────┘                         │
                            │                                  │
                            ▼                                  │
                  ┌─────────────────┐                         │
                  │ Generate Frame  │                         │
                  │ Descriptions    │                         │
                  │ (BLIP Model)    │                         │
                  └─────────┬───────┘                         │
                            │                                  │
                            └─────────┬────────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────┐
                            │ SCENE GENERATOR │
                            │ (Ollama LLM)    │
                            └─────────┬───────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
                     ▼                                 ▼
          ┌─────────────────┐                ┌─────────────────┐
          │ Audio Available?│                │ No Audio Case   │
          │      YES        │                │ scene_generation│
          └─────────┬───────┘                │ _from_frame()   │
                    │                        └─────────┬───────┘
                    ▼                                  │
          ┌─────────────────┐                         │
          │ AUDIO+VIDEO     │                         │
          │ Scene Generator │                         │
          │ audio_video_    │                         │
          │ scene_generation│                         │
          └─────────┬───────┘                         │
                    │                                 │
                    └─────────┬───────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ GENERATED SCENE │
                    │   DESCRIPTION   │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ FAKE DETECTION  │
                    │    ANALYSIS     │
                    └─────────┬───────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────┐                ┌─────────────────┐
    │ ANALYSIS 1      │                │ ANALYSIS 2      │
    │ Standard LLM    │                │ RAG-Enhanced    │
    │ scene_realism_  │                │ scene_realism_  │
    │ checking1()     │                │ checking1_rag() │
    └─────────┬───────┘                └─────────┬───────┘
              │                                  │
              │     ┌─────────────────┐          │
              └────▶│ fact_check_     │◀─────────┘
                    │ wikipedia_only  │
                    │ (RAG System)    │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ FINAL VERDICT   │
                    └─────────────────┘
```

### Key Components

*   📹 **`FrameExtractor`**: Extracts video frames using OpenCV.
*   🎵 **`AudioTranscriber`**: Converts audio to text using the SpeechRecognition library (with Google Speech Recognition).
*   🤖 **`BLIPCaptioner`**: Generates descriptive captions for image frames using the BLIP model.
*   🧠 **`Ollama LLM`**: Powers scene analysis, description generation, and fake detection logic.
*   📚 **`RAG System`**: Implements a Retrieval-Augmented Generation pipeline using Wikipedia for fact-checking and grounding.
*   🔍 **`Fact Checker`**: Provides a multi-layered authenticity verification by combining a standard LLM check with a RAG-enhanced check.

### Analysis Flow Summary

1.  **Input**: The pipeline starts with a video file.
2.  **Extraction**: Frames and audio are extracted concurrently.
3.  **Description**: `BLIP` generates descriptions for the visual frames.
4.  **Transcription**: Google Speech-to-Text transcribes the audio.
5.  **Synthesis**: An LLM generates a cohesive scene description, using the audio transcript if available.
6.  **Verification**: The generated scene undergoes two parallel analyses: a standard check for realism and a RAG-enhanced check for factual consistency.
7.  **Output**: A final verdict is composed from the results of the dual analyses.

## API & Output

The service is exposed via a FastAPI backend. After processing, it returns a JSON object with the final analysis.

### Output Format

```json
{
  "realism_score": 0.0,
  "ai_detection": false,
  "fact_check": false,
  "scene_description": "A descriptive summary of the video's content."
}
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd backend2
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Ollama:**
    -   Install Ollama from [ollama.ai](https://ollama.ai/).
    -   Pull the required model (e.g., `llava`):
        ```bash
        ollama pull llava
        ```
    -   Ensure the Ollama server is running in the background.

## Usage

1.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

2.  **Interact with the API endpoints** (e.g., `/analyze_video/`, `/analyze_downloaded/`) using a tool like `curl` or Postman to get a video analysis.
