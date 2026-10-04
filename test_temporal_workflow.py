"""End-to-end test for DriveSafe Temporal Workflow."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from drivesafe.activities import (
    analyze_window_activity,
    export_clip_activity,
    extract_windows_activity,
    save_session_activity,
)
from drivesafe.config import MODEL_NAME, GOOGLE_CLOUD_MODEL, ENGINE
from drivesafe.temporal_client import TASK_QUEUE
from drivesafe.video import get_video_info
from drivesafe.workflows import AnalysisWorkflowInput, VideoAnalysisWorkflow


async def main():
    print("=============================================================")
    print(" DriveSafe Temporal Workflow - End-to-End Test")
    print("=============================================================")

    # Test sample video
    sample_video = Path("sample_dashcam.mp4")
    if not sample_video.exists():
        sample_video = Path("person-bicycle-car-detection.mp4")
    
    if not sample_video.exists():
        print("ERROR: Sample video not found.")
        return

    print(f"[1/4] Found test video: {sample_video.name}")
    duration, width, height = get_video_info(sample_video)
    print(f"      Duration: {duration:.1f}s | Resolution: {width}x{height}")

    print("[2/4] Starting local Temporal test environment...")
    async with await WorkflowEnvironment.start_local() as env:
        print("      Temporal test cluster started successfully!")

        print("[3/4] Registering activities & starting worker...")
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[VideoAnalysisWorkflow],
            activities=[
                extract_windows_activity,
                analyze_window_activity,
                export_clip_activity,
                save_session_activity,
            ],
        ):
            print(f"      Worker listening on task queue: '{TASK_QUEUE}'")
            print("[4/4] Executing VideoAnalysisWorkflow with Temporal durability...")

            model_text = GOOGLE_CLOUD_MODEL if ENGINE in ("gemini_api", "google_cloud") else MODEL_NAME
            input_data = AnalysisWorkflowInput(
                session_id="temporaltest01",
                source_name=sample_video.name,
                source_path_str=str(sample_video.resolve()),
                duration_seconds=duration,
                model_name=model_text,
                window_seconds=10,
                samples_per_window=4,
            )

            # Start workflow handle
            handle = await env.client.start_workflow(
                VideoAnalysisWorkflow.run,
                input_data,
                id=f"workflow-{input_data.session_id}",
                task_queue=TASK_QUEUE,
            )
            print(f"      Workflow started with ID: '{handle.id}'", flush=True)

            # Monitor progress via Temporal query
            while True:
                progress = await handle.query(VideoAnalysisWorkflow.get_progress)
                print(f"      [Query Progress] {progress['percent']}% - {progress['status_message']}", flush=True)
                if progress.get("is_completed") or progress.get("error"):
                    break
                await asyncio.sleep(1.0)

            # Await final workflow result
            result = await handle.result()
            print("\n=============================================================")
            print(" WORKFLOW RESULT SUCCESS!")
            print("=============================================================")
            print(f" Session ID: {result.get('session_id')}")
            print(f" Source:     {result.get('source_name')}")
            print(f" Duration:   {result.get('duration_seconds'):.1f}s")
            print(f" Events Detected: {len(result.get('events', []))}")
            for idx, ev in enumerate(result.get("events", []), 1):
                print(f"   Event {idx}: {ev.get('start_seconds')}s-{ev.get('end_seconds')}s -> {ev.get('event_type')}: {ev.get('summary')}")
            print("=============================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
