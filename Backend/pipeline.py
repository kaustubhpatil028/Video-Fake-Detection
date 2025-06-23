import cv2
import ollama
import moviepy.editor as mp
import speech_recognition as sr
from pathlib import Path
from typing import List
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


# ---------- FRAME EXTRACTOR ----------
class FrameExtractor:
    def __init__(self):
        self.frames_dir = Path("frames")
        self.frames_dir.mkdir(exist_ok=True)

    def extract_frames(self, video_path: str, num_frames: int = 10) -> List[str]:
        video_name = Path(video_path).stem
        output_dir = self.frames_dir / video_name
        output_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"❌ OpenCV failed to open the video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")

        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frame_paths = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = output_dir / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))

        cap.release()
        return frame_paths


# ---------- AUDIO TRANSCRIPTION ----------
class AudioTranscriber:
    def __init__(self):
        self.text_dir = Path("text")
        self.text_dir.mkdir(exist_ok=True)

    def transcribe_video(self, video_path: str) -> str:
        """Extract audio from video and transcribe to text"""
        try:
            # Load the video
            video = mp.VideoFileClip(video_path)
            
            # Extract the audio from the video
            audio_file = video.audio
            audio_path = "temp_audio.wav"
            audio_file.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Initialize recognizer
            r = sr.Recognizer()
            
            # Load the audio file
            with sr.AudioFile(audio_path) as source:
                data = r.record(source)
            
            # Convert speech to text
            text = r.recognize_google(data)
            
            print(f"\n🎵 Transcribed audio: {text[:100]}...")
            
            # Save the text to a file
            output_file = self.text_dir / "transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"✅ Transcript saved to: {output_file}")
            
            # Clean up temporary audio file
            Path(audio_path).unlink(missing_ok=True)
            video.close()
            
            return text.strip()
            
        except Exception as e:
            print(f"⚠️ Audio transcription failed: {e}")
            return ""


# ---------- BLIP DESCRIPTION ----------
class BLIPCaptioner:
    def __init__(self):
        print("🤖 Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✅ BLIP model loaded successfully")

    def generate(self, image_path: str) -> str:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)


# ---------- LLaVA DESCRIPTION ----------
class LlavaCaptioner:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        print("🤖 Loading LLaVA model...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name)
        print("✅ LLaVA model loaded successfully")

    def generate(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=30)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Remove the prompt from the output if present
        if "ASSISTANT:" in output:
            output = output.split("ASSISTANT:", 1)[-1].strip()
        return output


# ---------- HYBRID BLIP + LLaVA FRAME PROCESSING ----------
def generate_descriptions(frames: List[str]) -> str:
    """Generate descriptions for all frames using BLIP and LLaVA (hybrid)."""
    blip = BLIPCaptioner()
    llava = LlavaCaptioner()
    descriptions = []
    llava_indices = [0, 4, 9]  # 1st, 5th, 10th frames (0-based)

    for i, frame_path in enumerate(frames):
        print(f"\n--- Frame {i+1} ---")
        if i in llava_indices:
            # LLaVA description
            try:
                llava_desc = llava.generate(frame_path)
                print(f"[LLaVA] Frame {i+1}: {llava_desc}")
            except Exception as e:
                llava_desc = f"LLaVA error: {e}"
                print(f"[LLaVA] Frame {i+1}: {llava_desc}")
            descriptions.append(f"LLaVA: {llava_desc}")
        # Always do BLIP as well
        try:
            blip_desc = blip.generate(frame_path)
            print(f"[BLIP] Frame {i+1}: {blip_desc}")
        except Exception as e:
            blip_desc = f"BLIP error: {e}"
            print(f"[BLIP] Frame {i+1}: {blip_desc}")
        descriptions.append(f"BLIP: {blip_desc}")

    # Create final output with all frame descriptions
    final_output = ""
    for idx, desc in enumerate(descriptions):
        final_output += f"Frame {idx+1}: {desc}\n"

    print("\n==== Combined LLM Output ====")
    print(final_output)
    print("============================\n")
    return final_output.strip()


def parse_llm_output(text):
    """
    Parse LLM output for verdict, confidence, and summary.
    Expected format:
    Verdict: FAKE/REAL
    Confidence: float (0-100)
    Summary: ...
    """
    verdict = ''
    confidence = 0.0
    summary = ''
    lines = text.strip().split('\n')
    for line in lines:
        if 'verdict' in line.lower():
            if 'fake' in line.lower():
                verdict = 'FAKE'
            elif 'real' in line.lower():
                verdict = 'REAL'
        elif 'confidence' in line.lower():
            try:
                confidence = float(''.join([c for c in line if c.isdigit() or c == '.']))
            except Exception:
                confidence = 0.0
        else:
            summary += line.strip() + ' '
    return verdict, confidence, summary.strip()


# ---------- OLLAMA LLM FUNCTIONS ----------
def analyze_video_content(question):
    """Analyze video content for potential fakes using Ollama, with clear verdict, confidence, and summary."""
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "system", "content": """
You are a highly advanced video authenticity analyst. Given frame-wise scene descriptions, your job is to:
- Output a clear verdict: Is the video FAKE or REAL?
- Output a confidence score (0-100, float, how sure you are about your verdict)
- Give a concise 2-3 line summary of the scene and any key evidence for your verdict.
- Output format:
Verdict: FAKE/REAL
Confidence: <float>
Summary: <2-3 lines>
Only output these three fields, suitable for display in a browser extension.
"""},
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"


def analyze_video_content_with_audio(question):
    """Analyze video content for potential fakes using Ollama, with clear verdict, confidence, and summary."""
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "system", "content": """
You are a multimodal video authenticity analyst. Given frame descriptions and audio transcript, your job is to:
- Output a clear verdict: Is the video FAKE or REAL?
- Output a confidence score (0-100, float, how sure you are about your verdict)
- Give a concise 2-3 line summary of the scene and any key evidence for your verdict.
- Output format:
Verdict: FAKE/REAL
Confidence: <float>
Summary: <2-3 lines>
Only output these three fields, suitable for display in a browser extension.
"""},
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"


def fact_check_video(question):
    """Fact check video content using Ollama, with clear verdict, confidence, and summary."""
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "system", "content": """
You are a video fact-checker. Given a generated scene, output:
- A clear verdict: FAKE or REAL
- A confidence score (0-100, float, how sure you are about your verdict)
- A 2-3 line summary of the scene and the main evidence for your verdict.
- Output format:
Verdict: FAKE/REAL
Confidence: <float>
Summary: <2-3 lines>
Only output these three fields, suitable for display in a browser extension.
"""},
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"


def generate_final_verdict(question):
    """Generate final verdict on video authenticity."""
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "system", "content": """You are a senior video authenticity expert. Your task is to provide a clear, final verdict on video authenticity based on comprehensive analysis.
                
                Format your response as:
                1. Final Verdict (Authentic/Fake/Suspicious)
                2. Confidence Level (High/Medium/Low)
                3. Key Evidence:
                   - Technical analysis findings
                   - Factual inconsistencies
                   - Visual/audio artifacts
                   - Metadata anomalies
                4. Risk Assessment:
                   - Potential impact if fake
                   - Likelihood of manipulation
                   - Distribution risk
                
                Provide clear, actionable recommendations for users."""},
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"


# ---------- SCENE GENERATION FUNCTIONS ----------
def scene_generation_from_frame(final_output):
    """Generates a scene description from the given frame description."""
    prompt = f"I will give you the  video frame descriptions and generate a detailed scene description by combining frame Description is : {final_output}"
    return analyze_video_content(prompt)


def audio_video_scene_generation(final_speech, final_output):
    """Generates a scene description from audio and video."""
    prompt = f"Analyze the audio transcript and video frames to generate a comprehensive scene description. Audio text is : {final_speech}. Video frame description is : {final_output}"
    return analyze_video_content_with_audio(prompt)


def scene_realism_checking1(scene):
    """Checks if the generated scene description meets certain criteria."""
    prompt = f"Analyze this scene description for authenticity video conation animal slike gorilla, and peguin are fake moslt and potential fakery: {scene}"
    return fact_check_video(prompt)

from rag import fact_check_wikipedia_only

def scene_realism_checking1_rag(scene):
    """Checks if the generated scene description meets certain criteria using RAG-based fact checking."""
    # Prepare the prompt for fact checking
    prompt = f"Analyze this scene description for authenticity and potential video conation animal slike gorilla, and peguin are fake moslty  fakery: {scene}"
    
    # Perform RAG-based fact checking with optimized parameters
    result = fact_check_wikipedia_only(
        claim=prompt,
        num_wiki_sentences=3,  # Adjusted for scene analysis
        num_retrieved_chunks=2  # Smaller number of chunks for scene context
    )
    
    # Return the fact checking result
    return result


def is_adult_content_llm(scene_or_frames):
    """
    Use LLM to detect if the scene/frame descriptions indicate adult/NSFW content.
    Returns (is_adult: bool, reason: str)
    """
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "system", "content": """
You are an expert content moderator. Given a video scene or frame description, answer:
- Is there any explicit, adult, or NSFW content? (Answer only YES or NO)
- If YES, provide a 1-line reason (e.g., 'nudity', 'sexual activity', 'suggestive content').
- Output format: YES/NO, then reason (if YES).
- Be concise and suitable for browser extension moderation.
"""},
                {"role": "user", "content": scene_or_frames}
            ]
        )
        text = response['message']['content'].strip().lower()
        if text.startswith('yes'):
            reason = text[3:].strip(' ,:.-') or 'Adult/NSFW content detected.'
            return True, reason
        return False, ''
    except Exception as e:
        return False, f"Moderation error: {e}"


# ---------- MAIN PIPELINE ----------
def main_pipeline(video_path: str, num_frames: int = 10):
    """Complete video fake detection pipeline using BLIP only, with concurrent RAG check and adult moderation."""
    try:
        print("🎬 Starting video analysis...")
        frame_extractor = FrameExtractor()
        frames = frame_extractor.extract_frames(video_path, num_frames)
        print(f"✅ Extracted {len(frames)} frames")
        descriptions = generate_descriptions(frames)
        print("✅ Generated frame descriptions")
        transcriber = AudioTranscriber()
        speech_text = transcriber.transcribe_video(video_path)
        print("✅ Transcribed audio")
        if speech_text:
            scene = audio_video_scene_generation(speech_text, descriptions)
        else:
            scene = scene_generation_from_frame(descriptions)
        print("✅ Generated scene description")
        # Adult content moderation (LLM)
        is_adult, moderation_msg = is_adult_content_llm(scene)
        analysis1 = scene_realism_checking1(scene)
        analysis2 = scene_realism_checking1_rag(scene)
        rag_result = fact_check_wikipedia_only(scene)
        # Parse LLM output
        verdict, llm_conf, llm_summary = parse_llm_output(analysis1)
        # Parse RAG output (try to extract confidence and verdict)
        rag_verdict, rag_conf, rag_summary = parse_llm_output(rag_result['response'] if isinstance(rag_result, dict) else rag_result)
        output = {
            'realism_score': llm_conf,
            'rag_confidence': rag_conf,
            'ai_detection': verdict == 'FAKE',
            'fact_check': verdict == 'REAL',
            'scene_description': f"Verdict: {verdict}\nConfidence: {llm_conf}\nSummary: {llm_summary}",
            'rag_check': f"Verdict: {rag_verdict}\nConfidence: {rag_conf}\nSummary: {rag_summary}",
            'is_adult_content': is_adult,
            'moderation_message': moderation_msg
        }
        return output
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        raise


def process_video(video_path: str):
    """
    Process video through the fake detection pipeline
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Analysis results including:
            - realism_score: float (0-100)
            - ai_detection: bool
            - fact_check: bool
            - scene_description: str
    """
    try:
        # Run the main pipeline
        results = main_pipeline(video_path)
        
        # Extract and format results
        analysis_1 = results['analysis_1']
        analysis_2 = results['analysis_2']
        
        # Calculate scores and detections
        realism_score = 0
        ai_detection = False
        fact_check = False
        
        # Analyze results from both checks
        if analysis_1:
            # Extract realism score (if available)
            if 'realism_score' in analysis_1:
                realism_score = analysis_1['realism_score']
            
            # Check for AI detection
            if 'ai_detected' in analysis_1:
                ai_detection = analysis_1['ai_detected']
            
            # Check fact verification
            if 'verified' in analysis_1:
                fact_check = analysis_1['verified']
        
        if analysis_2:
            # Override with RAG results if they are more confident
            if 'realism_score' in analysis_2:
                realism_score = analysis_2['realism_score']
            
            if 'ai_detected' in analysis_2:
                ai_detection = analysis_2['ai_detected']
            
            if 'verified' in analysis_2:
                fact_check = analysis_2['verified']
        
        return {
            'realism_score': realism_score,
            'ai_detection': ai_detection,
            'fact_check': fact_check,
            'scene_description': results['video_scene']
        }
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        raise


# ---------- MAIN ----------
if __name__ == "__main__":
    # Replace with your video path
    video_path = "downloads/twitter/demo.mp4"  # Change this to your video file
    
    try:
        # Run the complete pipeline
        results = main_pipeline(video_path, num_frames=10)
        
        # Optional: Test basic Q&A

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        print("Make sure you have:")
        print("- Video file at the specified path")
        print("- Ollama running with llava  model")
        print("- All required dependencies installed")
        print("\nRequired packages:")
        print("pip install opencv-python transformers torch pillow ollama moviepy SpeechRecognition")