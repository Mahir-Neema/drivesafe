import os
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DRIVESAFE_DATA_DIR", Path.home() / ".drivesafe"))
OLLAMA_URL = os.environ.get("DRIVESAFE_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
_ollama_host = urlsplit(OLLAMA_URL).hostname or "localhost"
try:
    _is_loopback = (
        _ollama_host.lower() in ("localhost", "127.0.0.1", "ollama", "host.docker.internal", "0.0.0.0")
        or ip_address(_ollama_host).is_loopback
    )
except ValueError:
    _is_loopback = _ollama_host.lower() in ("localhost", "127.0.0.1", "ollama", "host.docker.internal")
if not _is_loopback:
    raise ValueError("DriveSafe only supports a local Ollama address (localhost or 127.0.0.1).")
MODEL_NAME = os.environ.get("DRIVESAFE_MODEL", "gemma4:e2b")
EMBEDDING_MODEL = os.environ.get("DRIVESAFE_EMBEDDING_MODEL", "nomic-embed-text")
MAX_VIDEO_SECONDS = int(os.environ.get("DRIVESAFE_MAX_VIDEO_SECONDS", 3600)) # Up to 60 min default
MAX_UPLOAD_BYTES = int(os.environ.get("DRIVESAFE_MAX_UPLOAD_BYTES", 2000 * 1024 * 1024)) # 2 GB default
WINDOW_SECONDS = 10
SAMPLES_PER_WINDOW = 4
CONTEXT_SECONDS = 5

# Inference Engine Configuration
ENGINE = os.environ.get("DRIVESAFE_ENGINE", "gemini_api") # 'ollama' or 'google_cloud' or 'gemini_api'

# Google Cloud Vertex AI / Gemini API Configuration (Optional)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "282833341683")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_CLOUD_MODEL = os.environ.get("GOOGLE_CLOUD_MODEL", "gemma-4-26b-a4b-it")

# MongoDB Configuration (Default to local MongoDB, or Atlas URI via DRIVESAFE_MONGO_URI)
MONGO_URI = os.environ.get("DRIVESAFE_MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.environ.get("DRIVESAFE_MONGO_DB", "drivesafe")


def ensure_data_root() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
