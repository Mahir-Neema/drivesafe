from drivesafe.storage import search_historical_events

def main():
    queries = [
        "cyclist or bicycle riding on the street",
        "close vehicle encounter or sudden stop",
        "pedestrians crossing",
    ]
    for q in queries:
        print(f"\n=======================================================")
        print(f" Query: '{q}'")
        print(f"=======================================================")
        matches = search_historical_events(q, top_k=3)
        print(f"Found {len(matches)} relevant events:")
        for idx, m in enumerate(matches):
            pct = m['score'] * 100
            print(f" [{idx+1}] Match: {pct:.1f}% | Type: {m['event_type']}")
            print(f"     Summary: {m['summary']}")
            print(f"     Clip: {m['clip_name']} (Exists: {bool(m['clip_path'])})")

if __name__ == "__main__":
    main()
