"""Test DriveSafe with Temporal durable workflow using Local Gemma via Ollama and Cloud Gemma 4."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drivesafe.gemma import check_model_ready, describe_window
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


async def test_local_gemma_temporal(model_tag: str):
    print(f"\n==============================================================")
    print(f"  TESTING LOCAL GEMMA VIA OLLAMA ({model_tag}) + TEMPORAL")
    print(f"==============================================================")

    # 1. Check readiness
    ready, status = check_model_ready(engine="ollama", model_name=model_tag)
    print(f"[1] Local Model Readiness Check -> ready={ready}, msg='{status}'")
    if not ready:
        print(f"  [SKIP/WAIT] Model {model_tag} is not ready yet: {status}")
        return False

    video_path = Path("person-bicycle-car-detection.mp4")
    duration, _, _ = get_video_info(video_path)

    # 2. Test direct activity with local Ollama
    print(f"[2] Extracting video frames...")
    extract_out = await extract_windows_activity(
        ExtractWindowsInput(
            source_path_str=str(video_path.resolve()),
            duration_seconds=duration,
            window_seconds=10,
            samples_per_window=4,
        )
    )
    first_window = extract_out.windows[0]

    print(f"[3] Executing analyze_window_activity with Local Ollama ({model_tag})...")
    res = await analyze_window_activity(
        AnalyzeWindowInput(
            window_dict=first_window,
            model_name=model_tag,
            engine="ollama",
        )
    )
    print(f"  [OK] Local Gemma Response:")
    print(f"       - Event Type: {res['event_type']}")
    print(f"       - Summary: {res['summary']}")
    print(f"       - Cues: {res['observed_cues']}")

    # 3. Test full Temporal Workflow Orchestration with Local Model
    print(f"[4] Starting full Temporal Workflow with Local Ollama ({model_tag})...")
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
                session_id=f"testlocal{model_tag.replace(':', '').replace('.', '').replace('-', '')}",
                source_name=video_path.name,
                source_path_str=str(video_path.resolve()),
                duration_seconds=duration,
                model_name=model_tag,
                engine="ollama",
                window_seconds=10,
                samples_per_window=4,
            )

            handle = await env.client.start_workflow(
                VideoAnalysisWorkflow.run,
                input_data,
                id=f"wf-local-{model_tag.replace(':', '-')}",
                task_queue=TASK_QUEUE,
            )

            while True:
                prog = await handle.query(VideoAnalysisWorkflow.get_progress)
                print(f"    [Workflow Progress] Step: {prog['current_step']} ({prog['percent']}%) -> {prog['status_message']}")
                if prog.get("is_completed") or prog.get("error"):
                    break
                await asyncio.sleep(1.0)

            result = await handle.result()
            print(f"  [OK] Temporal Workflow completed successfully with {len(result['events'])} events detected locally by {model_tag}!")
            return True


async def main():
    # Test with gemma3:4b (currently installed)
    success_3 = await test_local_gemma_temporal("gemma3:4b")
    
    # Test with gemma4:e2b (if installed or done downloading)
    ready_4, _ = check_model_ready(engine="ollama", model_name="gemma4:e2b")
    if ready_4:
        success_4 = await test_local_gemma_temporal("gemma4:e2b")
    else:
        print(f"\n[INFO] gemma4:e2b is still downloading or not ready. Tested gemma3:4b successfully.")


if __name__ == "__main__":
    asyncio.run(main())
