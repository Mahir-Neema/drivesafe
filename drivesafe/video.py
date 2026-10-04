import shutil
import subprocess
from pathlib import Path

import cv2

from drivesafe.models import FrameSample, VideoWindow


class VideoError(Exception):
    pass


def get_video_info(path: Path) -> tuple[float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError("This video could not be opened. Try an MP4, MOV, or AVI file.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoError("The video metadata could not be read. Check that the file is not damaged.")
    return frame_count / fps, width, height


def make_windows(path: Path, duration: float, window_seconds: int = 10,
                 samples_per_window: int = 4) -> list[VideoWindow]:
    windows: list[VideoWindow] = []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError("OpenCV could not read the selected video.")
    start = 0.0
    try:
        while start < duration:
            end = min(start + window_seconds, duration)
            span = end - start
            fractions = [0.12 + (0.76 * i / max(samples_per_window - 1, 1))
                         for i in range(samples_per_window)]
            samples: list[FrameSample] = []
            for fraction in fractions:
                timestamp = min(start + span * fraction, max(0.0, duration - 0.04))
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                longest = max(height, width)
                if longest > 1024:
                    scale = 1024 / longest
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if encoded:
                    samples.append(FrameSample(round(timestamp, 2), buffer.tobytes()))
            if not samples:
                raise VideoError(f"Could not sample frames near {int(start)} seconds.")
            windows.append(VideoWindow(start, end, samples))
            start = end
    finally:
        capture.release()
    return windows


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def export_clip(path: Path, start: float, end: float, output: Path,
                duration: float, context_seconds: int = 5) -> Path:
    executable = ffmpeg_path()
    if not executable:
        raise VideoError("FFmpeg is required to export review clips. Install FFmpeg and restart DriveSafe.")
    clip_start = max(0.0, start - context_seconds)
    clip_end = min(duration, end + context_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{clip_start:.2f}", "-i", str(path), "-t", f"{clip_end - clip_start:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-movflags", "+faststart", str(output),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise VideoError("Clip export took too long. Try a shorter source video.") from exc
    if result.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        message = result.stderr.strip().splitlines()
        detail = message[-1] if message else "FFmpeg could not create the clip."
        raise VideoError(detail[:240])
    return output
