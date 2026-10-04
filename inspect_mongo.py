import json
import os
from pathlib import Path
from drivesafe.config import DATA_ROOT, MONGO_URI, MONGO_DB_NAME
from drivesafe.embeddings import get_embedding
from drivesafe.storage import get_all_sessions

def inspect():
    sessions = get_all_sessions()
    print("=" * 60)
    print(f"MongoDB / DriveSafe Storage Inspection")
    print(f"Target Database: {MONGO_DB_NAME} | Collection: sessions")
    print(f"Total Stored Drive Sessions: {len(sessions)}")
    print("=" * 60)

    if not sessions:
        print("No documents found.")
        return

    # Backfill embeddings if needed
    for s in sessions:
        sid = s.get("session_id")
        json_path = DATA_ROOT / "sessions" / sid / "session.json"
        updated = False
        for ev in s.get("events", []):
            if not ev.get("summary_embedding"):
                txt = f"{ev.get('event_type')}: {ev.get('summary')} {' '.join(ev.get('observed_cues', []))}"
                v = get_embedding(txt)
                if v:
                    ev["summary_embedding"] = v
                    updated = True
        if updated and json_path.is_file():
            json_path.write_text(json.dumps(s, indent=2), encoding="utf-8")

    # Sample Document Output
    sample = sessions[0]
    sample_repr = dict(sample)
    if "events" in sample_repr:
        repr_events = []
        for ev in sample_repr["events"]:
            ev_copy = dict(ev)
            vec = ev_copy.get("summary_embedding")
            if vec and isinstance(vec, list):
                ev_copy["summary_embedding"] = f"[Array of {len(vec)} float32 dimensions: {vec[:3]}...]"
            repr_events.append(ev_copy)
        sample_repr["events"] = repr_events

    print("\nSample MongoDB Document Schema:")
    print(json.dumps(sample_repr, indent=2))

    print("\nStored Drives Overview:")
    for i, s in enumerate(sessions[:6]):
        print(f"  [{i+1}] Session ID: {s.get('session_id')} | File: {s.get('source_name')} | Duration: {s.get('duration_seconds')}s | Events: {len(s.get('events', []))}")

if __name__ == "__main__":
    inspect()
