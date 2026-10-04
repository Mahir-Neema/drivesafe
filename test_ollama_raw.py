import base64
import json
import urllib.request
from pathlib import Path
from drivesafe.video import make_windows, get_video_info
import cv2
import numpy as np

video_path = Path("person-bicycle-car-detection.mp4")
dur, _, _ = get_video_info(video_path)
windows = make_windows(video_path, dur, 10, 4)
window = windows[0]

ordered_frames = sorted(window.frames, key=lambda frame: frame.timestamp_seconds)
tile_width, tile_height, label_height, gutter = 448, 252, 30, 12
sheet = np.full((2 * (tile_height + label_height) + 3 * gutter, 2 * tile_width + 3 * gutter, 3), 246, dtype=np.uint8)
for index, frame_sample in enumerate(ordered_frames[:4]):
    decoded = cv2.imdecode(np.frombuffer(frame_sample.image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None: continue
    tile = cv2.resize(decoded, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    row, column = divmod(index, 2)
    x = gutter + column * (tile_width + gutter)
    y = gutter + row * (tile_height + label_height + gutter)
    sheet[y:y + tile_height, x:x + tile_width] = tile
encoded, sheet_bytes = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])

prompt = (
    "Review this dashcam contact sheet. "
    "Choose possible_close_encounter, pedestrian_or_cyclist, sudden_visual_change, none, or unclear. "
    "Respond with a JSON object: {\"event_type\": \"...\", \"summary\": \"...\", \"observed_cues\": [\"...\"], \"needs_review\": true/false}"
)

body = {
    "model": "gemma4:e2b",
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 600},
    "messages": [{
        "role": "user",
        "content": prompt,
        "images": [base64.b64encode(sheet_bytes.tobytes()).decode("ascii")],
    }],
}

req = urllib.request.Request(
    "http://127.0.0.1:11434/api/chat",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=120) as resp:
    res = json.loads(resp.read().decode("utf-8"))

print("=== RAW OLLAMA DICT ===")
print(json.dumps(res, indent=2))
print("=== END RAW OLLAMA DICT ===")
