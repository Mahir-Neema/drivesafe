"""Comprehensive End-to-End Verification Test for Temporal with Local Gemma 4 E2B & Cloud Gemma 4 26B."""

import asyncio
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from drivesafe.activities import (
    AnalyzeWindowInput,
    ExportClipInput,
    ExtractWindowsInput,
    SaveSessionInput,
    analyze_window_activity,
    export_clip_activity,
    extract_windows_activity,
    save_session_activity,
)
from drivesafe.config import GOOGLE_CLOUD_MODEL, MODEL_NAME
from drivesafe.gemma import check_model_ready, describe_window
from drivesafe.temporal_client import TASK_QUEUE
from drivesafe.video import get_video_info
from drivesafe.workflows import AnalysisWorkflowInput, VideoAnalysisWorkflow


async def test_local_gemma4_e2b(video_path: Path):
    print("\n==============================================================")
    print("  [TEST 1] LOCAL GEMMA 4 E2B (Ollama Edge 2B) + TEMPORAL")
    print("==============================================================")

    # 1. Readiness check
    ready, msg = check_model_ready(engine="ollama", model_name="gemma4:e2b")
    print(f"1. Local Model Status: ready={ready} | msg='{msg}'")
    assert ready is True, f"gemma4:e2b should be ready in Ollama: {msg}"

    duration, _, _ = get_video_info(video_path)

    # 2. Direct Activity Test
    print("2. Extracting sample windows...")
    extract_out = await extract_windows_activity(
        ExtractWindowsInput(
            source_path_str=str(video_path.resolve()),
            duration_seconds=duration,
            window_seconds=10,
            samples_per_window=4,
        )
    )
    assert len(extract_out.windows) > 0

    print("3. Executing analyze_window_activity with Local gemma4:e2b...")
    res = await analyze_window_activity(
        AnalyzeWindowInput(
            window_dict=extract_out.windows[0],
            model_name="gemma4:e2b",
            engine="ollama",
        )
    )
    print(f"   [OK] Local Gemma 4 E2B Response -> Event: '{res['event_type']}' | Summary: '{res['summary']}'")
    assert "event_type" in res
    assert "summary" in res

    # 3. Full Temporal Workflow with Local Ollama
    print("4. Starting Temporal Workflow with Local Ollama gemma4:e2b...")
    session_id = "testlocal" + secrets.token_hex(4)
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
                duration_seconds=min(duration, 20.0), # test first 2 windows for speedy verification
                model_name="gemma4:e2b",
                engine="ollama",
                window_seconds=10,
                samples_per_window=4,
            )

            handle = await env.client.start_workflow(
                VideoAnalysisWorkflow.run,
                input_data,
                id=f"workflow-{session_id}",
                task_queue=TASK_QUEUE,
            )

            while True:
                prog = await handle.query(VideoAnalysisWorkflow.get_progress)
                print(f"   [Temporal Query] Step: {prog['current_step']} ({prog['percent']}%) - {prog['status_message']}")
                if prog.get("is_completed") or prog.get("error"):
                    break
                await asyncio.sleep(1.0)

            result = await handle.result()
            assert result.get("session_id") == session_id
            print(f"   [SUCCESS] Local Gemma 4 E2B Temporal Workflow completed with {len(result.get('events', []))} events!")


async def test_cloud_gemma4_26b(video_path: Path):
    print("\n==============================================================")
    print("  [TEST 2] CLOUD GEMMA 4 26B (Gemini API) + TEMPORAL")
    print("==============================================================")

    # 1. Readiness check
    ready, msg = check_model_ready(engine="gemini_api", model_name="gemma-4-26b-a4b-it")
    print(f"1. Cloud Model Status: ready={ready} | msg='{msg}'")
    assert ready is True, f"Cloud model should be ready: {msg}"

    duration, _, _ = get_video_info(video_path)

    # 2. Full Temporal Workflow with Cloud Gemini API
    print("2. Starting Temporal Workflow with Cloud Gemini API...")
    session_id = "testcloud" + secrets.token_hex(4)
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
                model_name="gemma-4-26b-a4b-it",
                engine="gemini_api",
                window_seconds=10,
                samples_per_window=4,
            )

            handle = await env.client.start_workflow(
                VideoAnalysisWorkflow.run,
                input_data,
                id=f"workflow-{session_id}",
                task_queue=TASK_QUEUE,
            )

            while True:
                prog = await handle.query(VideoAnalysisWorkflow.get_progress)
                print(f"   [Temporal Query] Step: {prog['current_step']} ({prog['percent']}%) - {prog['status_message']}")
                if prog.get("is_completed") or prog.get("error"):
                    break
                await asyncio.sleep(0.8)

            result = await handle.result()
            assert result.get("session_id") == session_id
            print(f"   [SUCCESS] Cloud Gemma 4 26B Temporal Workflow completed with {len(result.get('events', []))} events!")


async def main():
    test_video = Path("person-bicycle-car-detection.mp4")
    if not test_video.is_file():
        test_video = Path("sample_dashcam.mp4")

    print(f"Running comprehensive tests with test video: {test_video.resolve()}")

    await test_local_gemma4_e2b(test_video)
    await test_cloud_gemma4_26b(test_video)

    print("\n==============================================================")
    print("  ALL TESTS PASSED: TEMPORAL + LOCAL GEMMA 4 E2B + CLOUD GEMMA 4 26B!")
    print("==============================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
