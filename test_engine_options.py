"""Test engine options: Cloud (Gemini API / Gemma 4 26B) and Local (Ollama / Gemma 4 E2B)."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drivesafe.gemma import check_model_ready, describe_window
from drivesafe.config import MODEL_NAME, GOOGLE_CLOUD_MODEL
from drivesafe.models import FrameSample, VideoWindow
from drivesafe.workflows import AnalysisWorkflowInput, VideoAnalysisWorkflow
from drivesafe.activities import (
    AnalyzeWindowInput,
    extract_windows_activity,
    analyze_window_activity,
    export_clip_activity,
    save_session_activity,
    ExtractWindowsInput,
)
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from drivesafe.temporal_client import TASK_QUEUE
from drivesafe.video import get_video_info


async def main():
    print("==============================================================")
    print("  TESTING INFERENCE ENGINE OPTIONS (Cloud 26B & Local E2B)")
    print("==============================================================")

    # 1. Test check_model_ready for Cloud (Gemini API)
    cloud_ready, cloud_msg = check_model_ready(engine="gemini_api", model_name="gemma-4-26b-a4b-it")
    print(f"[1] Cloud check: ready={cloud_ready}, msg='{cloud_msg}'")
    assert cloud_ready is True, "Cloud model should be ready"

    # 2. Test check_model_ready for Local (Ollama)
    local_ready, local_msg = check_model_ready(engine="ollama", model_name="gemma4:e2b")
    print(f"[2] Local check: ready={local_ready}, msg='{local_msg}'")
    assert "gemma4:e2b" in local_msg or "Ollama" in local_msg, "Expected local model or ollama in msg"

    # 3. Test direct activity with engine="gemini_api"
    video_path = Path("person-bicycle-car-detection.mp4")
    duration, _, _ = get_video_info(video_path)
    extract_out = await extract_windows_activity(
        ExtractWindowsInput(
            source_path_str=str(video_path.resolve()),
            duration_seconds=duration,
            window_seconds=10,
            samples_per_window=4,
        )
    )
    first_window = extract_out.windows[0]

    print("[3] Testing analyze_window_activity with engine='gemini_api' & model='gemma-4-26b-a4b-it'...")
    res = await analyze_window_activity(
        AnalyzeWindowInput(
            window_dict=first_window,
            model_name="gemma-4-26b-a4b-it",
            engine="gemini_api",
        )
    )
    print(f"  [OK] Cloud analysis result -> event: {res['event_type']} | summary: {res['summary'][:60]}...")
    assert "event_type" in res

    # 4. Test full Temporal workflow execution with engine="gemini_api"
    print("[4] Testing Temporal workflow with engine='gemini_api'...")
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
                session_id="test-engine-cloud",
                source_name=video_path.name,
                source_path_str=str(video_path.resolve()),
                duration_seconds=duration,
                model_name="gemma-4-26b-a4b-it",
                engine="gemini_api",
                window_seconds=10,
                samples_per_window=4,
            )

            result = await env.client.execute_workflow(
                VideoAnalysisWorkflow.run,
                input_data,
                id="test-workflow-engine-opt",
                task_queue=TASK_QUEUE,
            )
            assert result["session_id"] == "test-engine-cloud"
            print(f"  [OK] Temporal Workflow completed with {len(result['events'])} events!")

    print("\n==============================================================")
    print("  ALL ENGINE TESTS PASSED SUCCESSFULLY! [SUCCESS]")
    print("==============================================================")


if __name__ == "__main__":
    asyncio.run(main())
