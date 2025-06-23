from typing import List

# ---------- HYBRID PIPELINE ----------
async def generate_descriptions(frames: List[str]) -> str:
    blip = BLIPCaptioner()
    sem = asyncio.Semaphore(2)
    descriptions = [""] * len(frames)

    async def process_llava(index: int, path: str):
        descriptions[index] = await generate_llava_description(path, index, sem)

    def process_blip(index: int, path: str):
        caption = blip.generate(path)
        print(f"⚡ [BLIP] Frame {index + 1} done")
        descriptions[index] = caption.strip()

    tasks = []

    for i, frame_path in enumerate(frames):
        if i in [0, len(frames) - 1]:
            tasks.append(process_llava(i, frame_path))
        else:
            process_blip(i, frame_path)

    await asyncio.gather(*tasks)

    final_output = ""
    for idx, desc in enumerate(descriptions):
        lines = desc.strip().split('\n')
        final_output += f"The frame {idx + 1} description is {' '.join(lines[:3])}. "

    return final_output.strip()


import cv2
import asyncio
import aiohttp
import base64
import ollama
import moviepy.editor as mp
import speech_recognition as sr
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


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
        try:
            video = mp.VideoFileClip(video_path)
            audio_file = video.audio
            audio_path = "temp_audio.wav"
            audio_file.write_audiofile(audio_path, verbose=False, logger=None)

            r = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                data = r.record(source)

            text = r.recognize_google(data)
            print(f"\n🎵 Transcribed audio: {text[:100]}...")

            output_file = self.text_dir / "transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Transcript saved to: {output_file}")
            Path(audio_path).unlink(missing_ok=True)
            video.close()

            return text.strip()
        except Exception as e:
            print(f"⚠ Audio transcription failed: {e}")
            return ""


# ---------- BLIP DESCRIPTION ----------
class BLIPCaptioner:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate(self, image_path: str) -> str:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)


# ---------- LLaVA DESCRIPTION ----------
async def generate_llava_description(image_path: str, index: int, sem: asyncio.Semaphore) -> str:
    url = "http://localhost:11434/api/generate"
    prompt = "Describe this image in detail. you are a expert in ocr and scene description"

    async with sem:
        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            payload = {
                "model": "llava",
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }

            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    data = await response.json()
                    text = data.get("response", "No response")
                    print(f"🧠 [LLaVA] Frame {index + 1} done: {text.strip()}")
                    return text.strip()
        except Exception as e:
            print(f"[LLaVA ERROR] Frame {index + 1}: {e}")
            return f"[Frame {index + 1}] Error: {e}"


# ---------- HYBRID PIPELINE ----------
async def generate_descriptions(frames: List[str]) -> str:
    blip = BLIPCaptioner()
    sem = asyncio.Semaphore(2)
    descriptions = [""] * len(frames)

    async def process_llava(index: int, path: str):
        descriptions[index] = await generate_llava_description(path, index, sem)

    def process_blip(index: int, path: str):
        caption = blip.generate(path)
        print(f"⚡ [BLIP] Frame {index + 1} done")
        descriptions[index] = caption.strip()

    tasks = []

    for i, frame_path in enumerate(frames):
        if i in [0, len(frames) - 1]:
            tasks.append(process_llava(i, frame_path))
        else:
            process_blip(i, frame_path)

    await asyncio.gather(*tasks)

    final_output = ""
    for idx, desc in enumerate(descriptions):
        lines = desc.strip().split('\n')
        final_output += f"The frame {idx + 1} description is {' '.join(lines[:3])}. "

    return final_output.strip()


# ---------- OLLAMA LLM FUNCTIONS ----------
def answer_question(question):
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

def create_scene_LLM(question):
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

def create_complete_scene_LLM(question):
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

def create_complete_scene_LLM2(question):
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

def create_complete_scene_LLM3(question):
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

def factchecking_LLM(question):
    try:
        response = ollama.chat(
            model='llava',
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"


# ---------- SCENE GENERATION FUNCTIONS ----------
def scene_generation_from_frame(final_output):
    inputFrame = f"Please describe the scene in detail using the following information: {final_output}"
    response = create_scene_LLM(inputFrame)
    return response

def audio_video_scene_generation(final_speech, final_output):
    prompt = f"Generate a scene description for the following audio and video: {final_speech} {final_output}. Include details about the scene, such as location. The scene should be detailed and include sensory details such as sights, sounds, smells."
    scene = create_complete_scene_LLM(prompt)
    return scene

def scene_realism_checking1(scene):
    prompt = f"YOU ARE EXPERT IN VIDEO FAKE DETECTION. Is the following scene description accurate psychologically and emotionally resonant? Is it correct or reliable? Is the video fake or realistic? {scene}"
    response = create_complete_scene_LLM(prompt)
    return response

def scene_realism_checking2(scene):
    prompt = f"YOU ARE EXPERT IN VIDEO FAKE DETECTION. Is the following scene description accurate psychologically and emotionally resonant? Is it correct or reliable? Is the video fake or realistic? {scene}"
    response = create_complete_scene_LLM2(prompt)
    return response

def scene_realism_checking3(scene):
    prompt = f"YOU ARE EXPERT IN VIDEO FAKE DETECTION. Is the following scene description accurate psychologically and emotionally resonant? Is it correct or reliable? Is the video fake or realistic? {scene}"
    response = create_complete_scene_LLM3(prompt)
    return response


# ---------- MAIN PIPELINE ----------
async def main_pipeline(video_path: str, num_frames: int = 10):
    print("🎬 Starting Video Fake Detection Pipeline...")
    print(f"📹 Processing video: {video_path}")

    print("\n🎬 Extracting frames...")
    extractor = FrameExtractor()
    frames = extractor.extract_frames(video_path, num_frames=num_frames)
    print(f"✅ Extracted {len(frames)} frames")

    print("\n🎵 Transcribing audio...")
    transcriber = AudioTranscriber()
    final_speech = transcriber.transcribe_video(video_path)

    print("\n🧠 Generating hybrid descriptions...")
    final_output = await generate_descriptions(frames)
    print("✅ Frame descriptions generated")

    print("\n🎭 Generating video scene...")
    video_scene = scene_generation_from_frame(final_output)
    print("✅ Video scene generated")

    print("\n🎯 Combining audio-video scene...")
    complete_scene = audio_video_scene_generation(final_speech, video_scene)
    print("✅ Complete scene generated")

    print("\n🔍 Running fake detection analysis...")
    output1 = scene_realism_checking1(complete_scene)
    output2 = scene_realism_checking2(complete_scene)
    output3 = scene_realism_checking3(complete_scene)

    print("\n" + "="*60)
    print("🎯 FAKE DETECTION RESULTS")
    print("="*60)
    print(f"\n📝 Video Scene Description:")
    print(video_scene)
    print(f"\n🎵 Audio Transcript:")
    print(final_speech)
    print(f"\n🎭 Complete Scene:")
    print(complete_scene)
    print(f"\n🔍 Analysis 1:")
    print(output1)
    print(f"\n🔍 Analysis 2:")
    print(output2)
    print(f"\n🔍 Analysis 3:")
    print(output3)
    print("="*60)

    return {
        'video_scene': video_scene,
        'audio_transcript': final_speech,
        'complete_scene': complete_scene,
        'analysis_1': output1,
        'analysis_2': output2,
        'analysis_3': output3
    }


# ---------- MAIN ----------
if __name__ == "__main__":
    video_path = "sample2.mp4"

    try:
        results = asyncio.run(main_pipeline(video_path, num_frames=10))

        print("\n🧪 Testing basic Q&A...")
        question = "What is the capital of India?"
        answer = answer_question(question)
        print(f"Q: {question}")
        print(f"A: {answer}")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        print("Make sure you have:")
        print("- Video file at the specified path")
        print("- Ollama running with LLaVA model")
        print("- All required dependencies installed")
