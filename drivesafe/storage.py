import json
import shutil
from pathlib import Path
from typing import Any, List

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from drivesafe.config import DATA_ROOT, MONGO_DB_NAME, MONGO_URI
from drivesafe.embeddings import cosine_similarity, get_embedding
from drivesafe.models import AnalysisSession


def get_mongo_collection():
    if not MONGO_URI:
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = client[MONGO_DB_NAME]
        return db["sessions"]
    except ConnectionFailure:
        return None


def session_dir(session_id: str) -> Path:
    if not all(c.isalnum() or c in "-_" for c in session_id):
        raise ValueError("Invalid session id.")
    return DATA_ROOT / "sessions" / session_id


def save_session(session: AnalysisSession) -> Path:
    session_dict = session.to_dict()

    # Generate embeddings for semantic vector search
    for event in session_dict.get("events", []):
        summary_text = event.get("summary", "")
        cues_text = " ".join(event.get("observed_cues", []))
        embed_input = f"{event.get('event_type', '')}: {summary_text}. {cues_text}".strip()
        if embed_input and ("summary_embedding" not in event or not event["summary_embedding"]):
            vector = get_embedding(embed_input)
            if vector:
                event["summary_embedding"] = vector

    # 1. Save to local disk
    folder = session_dir(session.session_id)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "session.json"
    target.write_text(json.dumps(session_dict, indent=2), encoding="utf-8")

    # 2. Sync to MongoDB (Atlas or Local MongoDB)
    collection = get_mongo_collection()
    if collection is not None:
        try:
            collection.update_one(
                {"session_id": session.session_id},
                {"$set": session_dict},
                upsert=True
            )
        except Exception:
            pass

    return target


def delete_session(session_id: str) -> None:
    folder = session_dir(session_id)
    if folder.exists():
        shutil.rmtree(folder)
    
    collection = get_mongo_collection()
    if collection is not None:
        try:
            collection.delete_one({"session_id": session_id})
        except Exception:
            pass


def get_session(session_id: str) -> dict | None:
    """Load a specific session by ID from local disk or MongoDB."""
    try:
        folder = session_dir(session_id)
        json_path = folder / "session.json"
        if json_path.is_file():
            return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    collection = get_mongo_collection()
    if collection is not None:
        try:
            doc = collection.find_one({"session_id": session_id}, {"_id": 0})
            if doc:
                return doc
        except Exception:
            pass
    return None


def get_all_sessions() -> List[dict]:
    """Retrieve all processed sessions from MongoDB and local storage, deduplicated."""
    session_map = {}

    # 1. Fetch from MongoDB
    collection = get_mongo_collection()
    if collection is not None:
        try:
            docs = list(collection.find({}, {"_id": 0}))
            for doc in docs:
                sid = doc.get("session_id")
                if sid:
                    session_map[sid] = doc
        except Exception:
            pass

    # 2. Local disk fallback & merge
    sessions_root = DATA_ROOT / "sessions"
    if sessions_root.is_dir():
        for sdir in sessions_root.iterdir():
            if sdir.is_dir():
                json_file = sdir / "session.json"
                if json_file.is_file():
                    try:
                        data = json.loads(json_file.read_text(encoding="utf-8"))
                        sid = data.get("session_id") or sdir.name
                        if sid not in session_map:
                            session_map[sid] = data
                        if "mtime" not in session_map[sid]:
                            session_map[sid]["mtime"] = json_file.stat().st_mtime
                    except Exception:
                        continue

    results = list(session_map.values())
    # Sort with newest drives first
    results.sort(
        key=lambda s: s.get("mtime", 0) if isinstance(s.get("mtime"), (int, float)) else str(s.get("session_id", "")),
        reverse=True,
    )
    return results


def search_historical_events(query: str, top_k: int = 6) -> List[dict]:
    """
    Perform semantic vector search across all processed video events.
    Uses MongoDB Atlas Vector Search ($vectorSearch) when available,
    with automatic fallback to exact in-memory cosine similarity.
    """
    if not query or not query.strip():
        return []

    query_vector = get_embedding(query)
    if not query_vector:
        return []

    collection = get_mongo_collection()
    
    # 1. Try Native MongoDB Atlas $vectorSearch Aggregation
    if collection is not None:
        for index_name in ("vector_index", "default"):
            try:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": index_name,
                            "path": "events.summary_embedding",
                            "queryVector": query_vector,
                            "numCandidates": 50,
                            "limit": top_k,
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "session_id": 1,
                            "source_name": 1,
                            "events": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                atlas_docs = list(collection.aggregate(pipeline))
                if atlas_docs:
                    atlas_results = []
                    for doc in atlas_docs:
                        sess_id = doc.get("session_id", "")
                        src_name = doc.get("source_name", "")
                        for idx, ev in enumerate(doc.get("events", [])):
                            clip_name = ev.get("clip_name") or f"event-{idx:03d}.mp4"
                            try:
                                clip_path = session_dir(sess_id) / "exports" / clip_name
                                has_clip = clip_path.is_file()
                                clip_str = str(clip_path) if has_clip else ""
                            except Exception:
                                clip_str = ""
                            atlas_results.append({
                                "session_id": sess_id,
                                "source_name": src_name,
                                "start_seconds": ev.get("start_seconds", 0),
                                "end_seconds": ev.get("end_seconds", 0),
                                "event_type": ev.get("event_type", "unclear"),
                                "summary": ev.get("summary", ""),
                                "observed_cues": ev.get("observed_cues", []),
                                "clip_name": clip_name,
                                "clip_path": clip_str,
                                "score": doc.get("score", 0.85),
                            })
                    if atlas_results:
                        return atlas_results[:top_k]
            except (OperationFailure, Exception):
                # Fallback to in-memory cosine similarity
                break

    # 2. In-Memory Cosine Similarity Vector Search across all sessions
    all_sessions = get_all_sessions()
    ranked_events = []

    for sess in all_sessions:
        session_id = sess.get("session_id", "")
        source_name = sess.get("source_name", "Recording")
        events = sess.get("events", [])
        
        for idx, ev in enumerate(events):
            vector = ev.get("summary_embedding")
            if not vector:
                summary_text = ev.get("summary", "")
                cues_text = " ".join(ev.get("observed_cues", []))
                embed_input = f"{ev.get('event_type', '')}: {summary_text}. {cues_text}".strip()
                vector = get_embedding(embed_input)
                ev["summary_embedding"] = vector

            if vector:
                score = cosine_similarity(query_vector, vector)
                if score > 0.35: # Quality relevance threshold
                    clip_name = ev.get("clip_name") or f"event-{idx:03d}.mp4"
                    try:
                        clip_path = session_dir(session_id) / "exports" / clip_name
                        has_clip = clip_path.is_file()
                        clip_str = str(clip_path) if has_clip else ""
                    except Exception:
                        clip_str = ""
                    ranked_events.append({
                        "session_id": session_id,
                        "source_name": source_name,
                        "start_seconds": ev.get("start_seconds", 0),
                        "end_seconds": ev.get("end_seconds", 0),
                        "event_type": ev.get("event_type", "unclear"),
                        "summary": ev.get("summary", ""),
                        "observed_cues": ev.get("observed_cues", []),
                        "clip_name": clip_name,
                        "clip_path": clip_str,
                        "score": score,
                    })

    # Sort descending by relevance similarity
    ranked_events.sort(key=lambda x: x["score"], reverse=True)
    return ranked_events[:top_k]
