import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

from drivesafe.config import MODEL_NAME, OLLAMA_URL, ENGINE, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_MODEL, GOOGLE_CLOUD_LOCATION, GEMINI_API_KEY
from drivesafe.models import VideoWindow, WindowResult


EVENT_TYPES = [
    "possible_close_encounter",
    "pedestrian_or_cyclist",
    "sudden_visual_change",
    "none",
    "unclear",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "summary": {"type": "string"},
        "observed_cues": {"type": "array", "items": {"type": "string"}},
        "needs_review": {"type": "boolean"},
    },
    "required": ["event_type", "summary", "observed_cues", "needs_review"],
}


class ModelError(Exception):
    pass


def _extract_json(content: str, thinking: str = "") -> dict:
    target_texts = [content, thinking]
    import re

    for text in target_texts:
        if not text or not isinstance(text, str):
            continue
        clean = text.strip()
        # Strip thinking tags if present in text
        if "<think>" in clean and "</think>" in clean:
            clean = clean.split("</think>", 1)[1].strip()
        elif clean.startswith("<think>"):
            end_idx = clean.find("</think>")
            if end_idx != -1:
                clean = clean[end_idx + len("</think>"):].strip()

        # Try direct parse
        try:
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try code block regex
        block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean)
        if block_match:
            try:
                return json.loads(block_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Try substring from first { to last }
        s_idx = clean.find("{")
        e_idx = clean.rfind("}")
        if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
            try:
                return json.loads(clean[s_idx:e_idx + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    # If thinking text exists, extract model reasoning
    if thinking and isinstance(thinking, str):
        event_type = "none"
        for candidate in ["pedestrian_or_cyclist", "possible_close_encounter", "sudden_visual_change", "unclear"]:
            if candidate in thinking.lower():
                event_type = candidate
                break
        
        summary = "Footage inspected by local Gemma vision model."
        sum_match = re.search(r"(?:\*Summary:\*|Summary:?)\s*([^\n\*\#]+)", thinking, re.IGNORECASE)
        if sum_match:
            summary = sum_match.group(1).strip()
        elif event_type == "pedestrian_or_cyclist":
            summary = "Pedestrians or vulnerable road users detected in roadway area."
        elif event_type == "possible_close_encounter":
            summary = "Close vehicle or obstacle interaction detected."
        elif event_type == "sudden_visual_change":
            summary = "Abrupt visual scene transition detected."

        cues = []
        cue_match = re.search(r"(?:\*Observed Cues:\*|Observed Cues:?)\s*([^\n\*\#]+)", thinking, re.IGNORECASE)
        if cue_match:
            cues = [c.strip() for c in re.split(r"[,;]", cue_match.group(1)) if c.strip()]
        if not cues and event_type != "none":
            cues = [f"Detected {event_type.replace('_', ' ')}"]

        return {
            "event_type": event_type,
            "summary": summary,
            "observed_cues": cues,
            "needs_review": event_type != "none",
        }

    raise json.JSONDecodeError("Could not parse JSON", content or thinking, 0)


def check_model_ready(engine: str | None = None, model_name: str | None = None) -> tuple[bool, str]:
    import os
    active_engine = engine or ENGINE
    active_model = model_name or (GOOGLE_CLOUD_MODEL if active_engine in ("gemini_api", "google_cloud") else MODEL_NAME)

    if active_engine == "gemini_api":
        api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
        if not api_key:
            return False, "Gemini API key is not configured. Click '🔑 Cloud API Key' on the top right to add it."
        return True, f"{active_model} is ready via Gemini API (Cloud)."
    elif active_engine == "google_cloud":
        if not GOOGLE_CLOUD_PROJECT:
            return False, "Google Cloud project ID not set. Set GOOGLE_CLOUD_PROJECT to use Vertex AI."
        return True, f"{active_model} is ready via Google Cloud Vertex AI."

    # Ollama local check
    request = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False, f"Ollama is not running at {OLLAMA_URL}. Start Ollama server to run local models."
    names = {item.get("name", "") for item in payload.get("models", [])}
    if active_model not in names and f"{active_model}:latest" not in names:
        return False, f"Local model '{active_model}' is not pulled yet. Run: `ollama pull {active_model}` in terminal."
    return True, f"{active_model} is ready locally via Ollama Edge."


def describe_window(window: VideoWindow, engine: str | None = None, model_name: str | None = None) -> WindowResult:
    active_engine = engine or ENGINE
    active_model = model_name or (GOOGLE_CLOUD_MODEL if active_engine in ("gemini_api", "google_cloud") else MODEL_NAME)

    ordered_frames = sorted(window.frames, key=lambda frame: frame.timestamp_seconds)
    time_labels = ", ".join(f"{frame.timestamp_seconds:.1f}s" for frame in ordered_frames)
    prompt = (
        "Review this dashcam contact sheet. Its tiles are ordered top-left, top-right, bottom-left, "
        f"bottom-right at {time_labels} within a {window.start_seconds:.0f}–{window.end_seconds:.0f}s window. "
        "Use visible evidence only. Never infer fault, a confirmed crash, exact speed, or motion not shown. "
        "Choose possible_close_encounter for a visibly close road interaction; pedestrian_or_cyclist for a "
        "vulnerable road user in a relevant roadway context; sudden_visual_change for a notable scene change; "
        "none for ordinary footage; unclear if evidence is insufficient. Keep summary and cues brief."
    )
    is_local = (active_engine == "ollama")
    tile_width = 320 if is_local else 448
    tile_height = 180 if is_local else 252
    label_height = 20 if is_local else 30
    gutter = 6 if is_local else 12
    sheet = np.full((2 * (tile_height + label_height) + 3 * gutter,
                     2 * tile_width + 3 * gutter, 3), 246, dtype=np.uint8)
    for index, frame_sample in enumerate(ordered_frames[:4]):
        decoded = cv2.imdecode(np.frombuffer(frame_sample.image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            continue
        tile = cv2.resize(decoded, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, 2)
        x = gutter + column * (tile_width + gutter)
        y = gutter + row * (tile_height + label_height + gutter)
        sheet[y:y + tile_height, x:x + tile_width] = tile
        cv2.rectangle(sheet, (x, y + tile_height), (x + tile_width, y + tile_height + label_height), (255, 255, 255), -1)
        font_scale = 0.42 if is_local else 0.55
        cv2.putText(sheet, f"FRAME {index + 1}   {frame_sample.timestamp_seconds:.1f}s",
                    (x + 6, y + tile_height + (15 if is_local else 21)), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (35, 35, 40), 1, cv2.LINE_AA)
    jpeg_quality = 75 if is_local else 82
    encoded, sheet_bytes = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not encoded:
        raise ModelError("DriveSafe could not prepare the sampled frames for Gemma.")

    if active_engine == "gemini_api":
        from google import genai
        from google.genai import types
        try:
            api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=active_model,
                contents=[
                    types.Part.from_bytes(data=sheet_bytes.tobytes(), mime_type='image/jpeg'),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.1
                )
            )
            content = response.text
            parsed = _extract_json(content)
        except Exception as exc:
            raise ModelError(f"Gemini API error: {exc}") from exc
    elif active_engine == "google_cloud":
        from google import genai
        from google.genai import types
        try:
            client = genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
            response = client.models.generate_content(
                model=active_model,
                contents=[
                    types.Part.from_bytes(data=sheet_bytes.tobytes(), mime_type='image/jpeg'),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.1
                )
            )
            content = response.text
            parsed = _extract_json(content)
        except Exception as exc:
            raise ModelError(f"Google Cloud Vertex AI error: {exc}") from exc
    else:
        body = {
            "model": active_model,
            "stream": False,
            "format": "json",
            "keep_alive": "15m",
            "options": {
                "temperature": 0.1,
                "num_predict": 180,
                "num_ctx": 2048,
                "top_k": 20,
                "top_p": 0.9,
            },
            "messages": [{
                "role": "user",
                "content": prompt + " Respond with a valid JSON object only with keys: event_type, summary, observed_cues, needs_review.",
                "images": [base64.b64encode(sheet_bytes.tobytes()).decode("ascii")],
            }],
        }
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
            msg_obj = payload.get("message", {})
            content = msg_obj.get("content", "")
            thinking = msg_obj.get("thinking", "")
            parsed = _extract_json(content, thinking)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
            raise ModelError(f"Ollama returned an error: {detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError(f"Could not reach local Ollama: {exc}") from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelError(f"Gemma returned a response DriveSafe could not read: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ModelError("Gemma returned structured data in an unexpected format.")
    event_type = parsed.get("event_type", "unclear")
    if event_type not in EVENT_TYPES:
        event_type = "unclear"
    cues = parsed.get("observed_cues", [])
    if not isinstance(cues, list):
        cues = []
    summary = parsed.get("summary", "The footage needs a closer look.")
    if not isinstance(summary, str):
        summary = "The footage needs a closer look."
    return WindowResult(
        start_seconds=window.start_seconds,
        end_seconds=window.end_seconds,
        event_type=event_type,
        summary=summary[:500],
        observed_cues=[str(cue)[:160] for cue in cues[:6]],
        needs_review=bool(parsed.get("needs_review", True)),
    )
