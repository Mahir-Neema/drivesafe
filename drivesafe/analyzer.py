from pathlib import Path
from typing import Callable

from drivesafe.config import CONTEXT_SECONDS, MAX_VIDEO_SECONDS, SAMPLES_PER_WINDOW, WINDOW_SECONDS, ENGINE, MODEL_NAME, GOOGLE_CLOUD_MODEL
from drivesafe.gemma import ModelError, describe_window
from drivesafe.models import AnalysisSession, WindowResult
from drivesafe.video import VideoError, export_clip, get_video_info, make_windows


ProgressCallback = Callable[[int, int, str], None]


def analyze_video(
    path: Path,
    session_id: str,
    on_progress: ProgressCallback | None = None,
    engine: str | None = None,
    model_name: str | None = None,
) -> AnalysisSession:
    active_engine = engine or ENGINE
    active_model = model_name or (GOOGLE_CLOUD_MODEL if active_engine in ("gemini_api", "google_cloud") else MODEL_NAME)

    duration, _, _ = get_video_info(path)
    if duration > MAX_VIDEO_SECONDS:
        raise VideoError(f"This MVP supports videos up to {MAX_VIDEO_SECONDS} seconds. This video is {duration:.0f} seconds.")
    windows = make_windows(path, duration, WINDOW_SECONDS, SAMPLES_PER_WINDOW)
    events: list[WindowResult] = []
    warnings: list[str] = []
    for index, window in enumerate(windows, start=1):
        if on_progress:
            on_progress(index - 1, len(windows), f"Reviewing {int(window.start_seconds)}–{int(window.end_seconds)} seconds")
        try:
            result = describe_window(window, engine=active_engine, model_name=active_model)
        except ModelError as exc:
            result = WindowResult(
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                event_type="unclear",
                summary="This section could not be analyzed.",
                observed_cues=[],
                warning=str(exc),
            )
            warnings.append(f"{int(window.start_seconds)}–{int(window.end_seconds)}s: {exc}")
        if result.event_type != "none":
            result.clip_name = f"event-{index:02d}.mp4"
            try:
                export_clip(
                    path,
                    result.start_seconds,
                    result.end_seconds,
                    path.parent / "exports" / result.clip_name,
                    duration,
                    CONTEXT_SECONDS,
                )
            except VideoError as exc:
                result.warning = str(exc)
                warnings.append(str(exc))
            events.append(result)
        if on_progress:
            on_progress(index, len(windows), f"Reviewed {index} of {len(windows)} sections")
    return AnalysisSession(
        session_id=session_id,
        source_name=path.name,
        duration_seconds=duration,
        model_name=f"{active_model} ({active_engine})",
        events=events,
        warnings=warnings,
    )
