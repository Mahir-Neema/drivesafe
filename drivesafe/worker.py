"""Temporal Worker for DriveSafe."""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporalio.client import Client
from temporalio.worker import Worker

from drivesafe.activities import (
    analyze_window_activity,
    export_clip_activity,
    extract_windows_activity,
    save_session_activity,
)
from drivesafe.temporal_client import TASK_QUEUE, TEMPORAL_HOST
from drivesafe.workflows import VideoAnalysisWorkflow


async def run_worker(target_host: str = TEMPORAL_HOST) -> None:
    """Start the DriveSafe Temporal worker."""
    print(f"[Temporal Worker] Connecting to Temporal server at {target_host}...")
    client = await Client.connect(target_host)
    print(f"[Temporal Worker] Connected! Listening on task queue: '{TASK_QUEUE}'")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[VideoAnalysisWorkflow],
        activities=[
            extract_windows_activity,
            analyze_window_activity,
            export_clip_activity,
            save_session_activity,
        ],
    )

    print("[Temporal Worker] Worker is running. Waiting for workflows...")
    await worker.run()


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else TEMPORAL_HOST
    try:
        asyncio.run(run_worker(host))
    except KeyboardInterrupt:
        print("\n[Temporal Worker] Worker stopped.")
