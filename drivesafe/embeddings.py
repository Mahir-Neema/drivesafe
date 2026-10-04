import json
import math
import urllib.error
import urllib.request
from typing import List

from drivesafe.config import EMBEDDING_MODEL, OLLAMA_URL


def get_embedding(text: str) -> List[float]:
    """Get vector embeddings for a given text using local Ollama nomic-embed-text model."""
    if not text or not text.strip():
        return []
    body = {
        "model": EMBEDDING_MODEL,
        "prompt": text.strip(),
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("embedding", [])
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        # Fallback to an empty list if embedding fails
        return []


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vector embeddings."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
