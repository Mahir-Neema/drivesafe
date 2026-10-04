import asyncio
import time
from pathlib import Path
from drivesafe.activities import extract_windows_activity, analyze_window_activity, ExtractWindowsInput, AnalyzeWindowInput
from drivesafe.video import get_video_info

async def benchmark():
    video_path = Path("person-bicycle-car-detection.mp4")
    duration, _, _ = get_video_info(video_path)
    print("Extracting 2-sample window for fast local inference...")
    extract_out = await extract_windows_activity(
        ExtractWindowsInput(
            source_path_str=str(video_path.resolve()),
            duration_seconds=duration,
            window_seconds=10,
            samples_per_window=2,
        )
    )
    first_window = extract_out.windows[0]
    
    print("\nBenchmarking Local Gemma 4 E2B on Ollama...")
    t0 = time.time()
    res = await analyze_window_activity(
        AnalyzeWindowInput(
            window_dict=first_window,
            model_name="gemma4:e2b",
            engine="ollama",
        )
    )
    elapsed = time.time() - t0
    print(f"-> Finished in {elapsed:.2f} seconds!")
    print(f"-> Event: {res.get('event_type')}")
    print(f"-> Summary: {res.get('summary')}")
    print(f"-> Cues: {res.get('observed_cues')}")

if __name__ == "__main__":
    asyncio.run(benchmark())
