"""Run full end-to-end Temporal Workflow with Local Gemma 4 E2B (Ollama Edge)."""

import asyncio
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from drivesafe.activities import (
    analyze_window_activity,
    export_clip_activity,
    extract_windows_activity,
    save_session_activity,
)
from drivesafe.gemma import check_model_ready
from drivesafe.temporal_client import TASK_QUEUE
from drivesafe.video import get_video_info
from drivesafe.workflows import AnalysisWorkflowInput, VideoAnalysisWorkflow


async def run_local_temporal_test():
    video_path = Path("person-bicycle-car-detection.mp4")
    assert video_path.is_file(), f"Video file not found: {video_path}"

    print("\n" + "=" * 65)
    print("  END-TO-END LOCAL TEST: GEMMA 4 E2B (Ollama) + TEMPORAL WORKFLOW")
    print("=" * 65)

    ready, msg = check_model_ready(engine="ollama", model_name="gemma4:e2b")
    print(f"\n[1] Local Model Status: ready={ready} | msg='{msg}'")
    assert ready is True

    duration, width, height = get_video_info(video_path)
    print(f"[2] Video Info: {video_path.name} | Duration: {duration:.1f}s | Resolution: {width}x{height}")

    session_id = "localtemp-" + secrets.token_hex(4)
    print(f"[3] Initializing Temporal Test Environment for Session: {session_id}...")

    t0 = time.time()
    async with await WorkflowEnvironment.start_local() as env:
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
            input_data = AnalysisWorkflowInput(
                session_id=session_id,
                source_name=video_path.name,
                source_path_str=str(video_path.resolve()),
                duration_seconds=duration,
                model_name="gemma4:e2b",
                engine="ollama",
                window_seconds=10,
                samples_per_window=2,  # Optimized local sampling
            )

            print(f"[4] Starting VideoAnalysisWorkflow on Temporal (Task Queue: '{TASK_QUEUE}')...\n")
            handle = await env.client.start_workflow(
                VideoAnalysisWorkflow.run,
                input_data,
                id=f"workflow-{session_id}",
                task_queue=TASK_QUEUE,
            )

            # Poll progress query
            last_step = ""
            while True:
                prog = await handle.query(VideoAnalysisWorkflow.get_progress)
                step_str = f"Step: {prog['current_step']} ({prog['percent']}%) -> {prog['status_message']}"
                if step_str != last_step:
                    print(f"   [Temporal Query] {step_str}")
                    last_step = step_str
                if prog.get("is_completed") or prog.get("error"):
                    break
                await asyncio.sleep(0.8)

            result = await handle.result()
            elapsed = time.time() - t0

            print("\n" + "=" * 65)
            print(f"  LOCAL TEST COMPLETED SUCCESSFULLY in {elapsed:.1f} seconds! [SUCCESS]")
            print("=" * 65)
            print(f"  Session ID : {result.get('session_id')}")
            print(f"  Total Windows Analyzed: {len(result.get('windows', []))}")
            print(f"  Total Events Flagged  : {len(result.get('events', []))}")
            print("\n  Flagged Events Details:")
            for idx, event in enumerate(result.get("events", [])):
                print(f"   [{idx + 1}] Time: {event.get('start_seconds', 0):.0f}s - {event.get('end_seconds', 0):.0f}s | Type: {event.get('event_type')}")
                print(f"       Summary: {event.get('summary')}")
                print(f"       Cues: {event.get('observed_cues')}")
                print(f"       Clip: {event.get('clip_name')}\n")

            return result


if __name__ == "__main__":
    asyncio.run(run_local_temporal_test())
