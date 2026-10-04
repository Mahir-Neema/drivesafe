"""Temporal Activities for DriveSafe video analysis pipeline."""

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temporalio import activity

from drivesafe.config import CONTEXT_SECONDS, SAMPLES_PER_WINDOW, WINDOW_SECONDS, MODEL_NAME, GOOGLE_CLOUD_MODEL, ENGINE
from drivesafe.gemma import describe_window
from drivesafe.models import FrameSample, VideoWindow, WindowResult, AnalysisSession
from drivesafe.storage import save_session, session_dir
from drivesafe.video import export_clip, get_video_info, make_windows


@dataclass
class WindowFrameDTO:
    timestamp_seconds: float
    image_bytes_base64: str


@dataclass
class WindowDTO:
    start_seconds: float
    end_seconds: float
    frames: list[WindowFrameDTO]


@dataclass
class ExtractWindowsInput:
    source_path_str: str
    duration_seconds: float
    window_seconds: int = 10
    samples_per_window: int = 4


@dataclass
class ExtractWindowsOutput:
    duration_seconds: float
    windows: list[dict[str, Any]]


@dataclass
class AnalyzeWindowInput:
    window_dict: dict[str, Any]
    model_name: str
    engine: str = "gemini_api"


@dataclass
class ExportClipInput:
    source_path_str: str
    start_seconds: float
    end_seconds: float
    output_path_str: str = ""
    clip_name: str = ""
    duration_seconds: float = 0.0
    context_seconds: int = 5


@dataclass
class SaveSessionInput:
    session_dict: dict[str, Any]


def safe_heartbeat(*details: Any) -> None:
    """Send activity heartbeat if executing inside an activity context."""
    try:
        activity.heartbeat(*details)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

@activity.defn
async def extract_windows_activity(input: ExtractWindowsInput) -> ExtractWindowsOutput:
    """Activity 1: Extract and sample video frames into analysis windows."""
    safe_heartbeat("Extracting video metadata...")
    path = Path(input.source_path_str)
    duration, _, _ = get_video_info(path)
    
    safe_heartbeat(f"Sampling frames across {duration:.1f}s video...")
    windows = make_windows(
        path,
        duration,
        window_seconds=input.window_seconds,
        samples_per_window=input.samples_per_window,
    )
    
    serialized_windows = []
    for w in windows:
        serialized_windows.append({
            "start_seconds": w.start_seconds,
            "end_seconds": w.end_seconds,
            "frames": [
                {
                    "timestamp_seconds": f.timestamp_seconds,
                    "image_b64": base64.b64encode(f.image_bytes).decode("ascii"),
                }
                for f in w.frames
            ],
        })
        
    return ExtractWindowsOutput(
        duration_seconds=duration,
        windows=serialized_windows,
    )


@activity.defn
async def analyze_window_activity(input: AnalyzeWindowInput) -> dict[str, Any]:
    """Activity 2: Analyze a single window using Gemma 4 Vision via Gemini API.
    
    Temporal automatically retries this on network errors, Gemini API rate limits, or timeouts.
    """
    w_data = input.window_dict
    frames = []
    for f in w_data["frames"]:
        if "image_b64" in f:
            raw_bytes = base64.b64decode(f["image_b64"])
        elif isinstance(f.get("image_bytes"), list):
            raw_bytes = bytes(f["image_bytes"])
        else:
            raw_bytes = f.get("image_bytes", b"")
        frames.append(FrameSample(timestamp_seconds=f["timestamp_seconds"], image_bytes=raw_bytes))

    window = VideoWindow(
        start_seconds=w_data["start_seconds"],
        end_seconds=w_data["end_seconds"],
        frames=frames,
    )
    
    safe_heartbeat(f"Analyzing {window.start_seconds:.0f}s - {window.end_seconds:.0f}s with {input.model_name} ({input.engine})...")
    result = describe_window(window, engine=input.engine, model_name=input.model_name)
    
    return {
        "start_seconds": result.start_seconds,
        "end_seconds": result.end_seconds,
        "event_type": result.event_type,
        "summary": result.summary,
        "observed_cues": result.observed_cues,
        "needs_review": result.needs_review,
        "review_status": result.review_status,
        "clip_name": result.clip_name,
        "warning": result.warning,
    }


@activity.defn
async def export_clip_activity(input: ExportClipInput) -> str:
    """Activity 3: Extract an event clip around flagged moments using FFmpeg."""
    source_path = Path(input.source_path_str).resolve()
    if input.output_path_str:
        output_path = Path(input.output_path_str)
    else:
        clip_filename = input.clip_name or "event.mp4"
        output_path = source_path.parent / "exports" / clip_filename

    safe_heartbeat(f"Rendering clip {output_path.name}...")
    export_clip(
        path=source_path,
        start=input.start_seconds,
        end=input.end_seconds,
        output=output_path,
        duration=input.duration_seconds,
        context_seconds=input.context_seconds,
    )
    return str(output_path)


@activity.defn
async def save_session_activity(input: SaveSessionInput) -> str:
    """Activity 4: Save session locally and sync vector embeddings to MongoDB Atlas."""
    safe_heartbeat("Saving session and syncing to Atlas...")
    session_dict = input.session_dict
    analysis = AnalysisSession(
        session_id=session_dict["session_id"],
        source_name=session_dict["source_name"],
        duration_seconds=session_dict["duration_seconds"],
        model_name=session_dict["model_name"],
        events=[WindowResult(**item) for item in session_dict.get("events", [])],
        warnings=session_dict.get("warnings", []),
    )
    saved_path = save_session(analysis)
    return str(saved_path)
