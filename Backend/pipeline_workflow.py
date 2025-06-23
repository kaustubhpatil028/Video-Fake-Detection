from pipeline import FrameExtractor, AudioTranscriber, BLIPCaptioner
from pipeline import audio_video_scene_generation, scene_generation_from_frame, scene_realism_checking1, scene_realism_checking1_rag, parse_llm_output
import aiohttp
import asyncio
import base64
from pathlib import Path

def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If already in an event loop (e.g., in Jupyter), use nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(coro)

async def analyze_frame_with_llava(image_path: str) -> str:
    url = "http://localhost:11434/api/generate"
    prompt = "Describe this image in scene to check whaerte vidoe fake or not creta discript of 3-4 lines only if it conatin any anmial k=like gorilla, penguin it is mostly fkae."
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
                print(f"🧠 [LLaVA] Frame 5 done")
                return text.strip()
    except Exception as e:
        return f"[Frame 5] Error: {e}"

def run_full_pipeline(video_path, num_frames=10):
    """
    Complete video fake detection pipeline using BLIP for all frames and LLaVA for frame 5, then full analysis as in pipeline.py.
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to extract
    Returns:
        dict: {
            'frame_descriptions': List[str],
            'llava_frame5': str,
            'scene_description': str,
            'llm_verdict': str,
            'llm_confidence': float,
            'llm_summary': str,
            'rag_verdict': str,
            'rag_confidence': float,
            'rag_summary': str
        }
    """
    frame_extractor = FrameExtractor()
    audio_transcriber = AudioTranscriber()
    frames = frame_extractor.extract_frames(video_path, num_frames)
    audio_text = audio_transcriber.transcribe_video(video_path)

    blip = BLIPCaptioner()
    descriptions = []
    for i, frame_path in enumerate(frames):
        try:
            blip_desc = blip.generate(frame_path)
            print(f"[BLIP] Frame {i+1}: {blip_desc}")
        except Exception as e:
            blip_desc = f"BLIP error: {e}"
            print(f"[BLIP] Frame {i+1}: {blip_desc}")
        descriptions.append(blip_desc)
    final_output = "\n".join([f"Frame {idx+1}: {desc}" for idx, desc in enumerate(descriptions)])
    print("\n==== BLIP Output ====")
    print(final_output)
    print("====================\n")

    # Analyze frame 5 (index 4) with LLaVA
    llava_frame5 = None
    if len(frames) >= 5:
        llava_frame5 = _run_async(analyze_frame_with_llava(frames[4]))
        print(f"\n==== LLaVA Output for Frame 5 ====")
        print(llava_frame5)
        print("===============================\n")

    # Scene generation (audio + frames)
    if audio_text:
        scene = audio_video_scene_generation(audio_text, final_output)
    else:
        scene = scene_generation_from_frame(final_output)
    print("\n==== Scene Description ====")
    print(scene)
    print("========================\n")

    # LLM-based realism/fake checking
    analysis1 = scene_realism_checking1(scene)
    # RAG-based fact checking
    analysis2 = scene_realism_checking1_rag(scene)

    # Parse LLM output
    verdict, llm_conf, llm_summary = parse_llm_output(analysis1)
    # Parse RAG output (try to extract confidence and verdict)
    rag_verdict, rag_conf, rag_summary = (None, None, None)
    if isinstance(analysis2, dict) and 'response' in analysis2:
        rag_verdict, rag_conf, rag_summary = parse_llm_output(analysis2['response'])
    elif isinstance(analysis2, str):
        rag_verdict, rag_conf, rag_summary = parse_llm_output(analysis2)

    return {
        'frame_descriptions': descriptions,
        'llava_frame5': llava_frame5,
        'scene_description': scene,
        'llm_verdict': verdict,
        'llm_confidence': llm_conf,
        'llm_summary': llm_summary,
        'rag_verdict': rag_verdict,
        'rag_confidence': rag_conf,
        'rag_summary': rag_summary
    } 