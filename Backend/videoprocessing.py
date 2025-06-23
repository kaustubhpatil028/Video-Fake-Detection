import cv2
import asyncio
import aiohttp
import base64
from pathlib import Path
from typing import List
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

    # Use LLaVA for frame 1, 5, and 10
    llava_indices = [0, 4, 9]

    for i, frame_path in enumerate(frames):
        if i in llava_indices:
            tasks.append(process_llava(i, frame_path))
        else:
            process_blip(i, frame_path)

    await asyncio.gather(*tasks)

    # Format and return
    final_output = ""
    for idx, desc in enumerate(descriptions):
        lines = desc.strip().split('\n')
        final_output += f"\n[Output {idx + 1}]\n" + "\n".join(lines[:3]) + "\n"

    return final_output.strip()


# ---------- MAIN ----------
if __name__ == "__main__":
    video_path = "sample2.mp4"

    extractor = FrameExtractor()
    print("🎬 Extracting frames...")
    frames = extractor.extract_frames(video_path, num_frames=10)
    print(f"✅ Extracted {len(frames)} frames")

    print("🧠 Generating hybrid descriptions...")
    final_result = asyncio.run(generate_descriptions(frames))

    print("\n🎯 Final Combined Output (3 lines each):")
    print("-" * 50)
    print(final_result)
    print("-" * 50)