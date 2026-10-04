"""Comprehensive verification test suite for DriveSafe with Temporal."""

import asyncio
import base64
import json
import secrets
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to path
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
from drivesafe.config import CONTEXT_SECONDS, DATA_ROOT, MODEL_NAME, GOOGLE_CLOUD_MODEL, ENGINE
from drivesafe.models import AnalysisSession, FrameSample, VideoWindow
from drivesafe.storage import session_dir
from drivesafe.temporal_client import TASK_QUEUE
from drivesafe.video import get_video_info
from drivesafe.workflows import AnalysisWorkflowInput, VideoAnalysisWorkflow


async def test_activities_directly(video_path: Path):
    print("\n--- [TEST 1/3] Direct Activities Verification ---")
    
    # 1. Test extract_windows_activity
    duration, width, height = get_video_info(video_path)
    print(f"[Activity 1] extract_windows_activity -> Video {video_path.name} ({duration:.1f}s)")
    extract_out = await extract_windows_activity(
        ExtractWindowsInput(
            source_path_str=str(video_path.resolve()),
            duration_seconds=duration,
            window_seconds=10,
            samples_per_window=4,
        )
    )
    assert len(extract_out.windows) > 0, "No windows extracted!"
    first_window = extract_out.windows[0]
    assert len(first_window["frames"]) == 4, "Expected 4 frames in window!"
    assert "image_b64" in first_window["frames"][0], "Expected image_b64 in serialized frame!"
    print(f"  [OK] Extracted {len(extract_out.windows)} window(s), 4 frames each (Base64 encoded)")

    # 2. Test analyze_window_activity
    print(f"[Activity 2] analyze_window_activity -> Calling Gemma 4 Vision via Gemini API...")
    model_text = GOOGLE_CLOUD_MODEL if ENGINE in ("gemini_api", "google_cloud") else MODEL_NAME
    analysis_res = await analyze_window_activity(
        AnalyzeWindowInput(
            window_dict=first_window,
            model_name=model_text,
        )
    )
    assert "event_type" in analysis_res, "Missing event_type in result!"
    assert "summary" in analysis_res, "Missing summary in result!"
    print(f"  [OK] Gemma 4 Response -> Event: '{analysis_res['event_type']}' | Summary: '{analysis_res['summary'][:60]}...'")

    # 3. Test export_clip_activity
    test_session_id = "testact" + secrets.token_hex(4)
    test_folder = session_dir(test_session_id)
    test_folder.mkdir(parents=True, exist_ok=True)
    exports_dir = test_folder / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    clip_output = exports_dir / "event-000.mp4"
    
    print(f"[Activity 3] export_clip_activity -> Extracting subclip via FFmpeg...")
    clip_path = await export_clip_activity(
        ExportClipInput(
            source_path_str=str(video_path.resolve()),
            start_seconds=0.0,
            end_seconds=min(duration, 10.0),
            output_path_str=str(clip_output.resolve()),
            duration_seconds=duration,
            context_seconds=2,
        )
    )
    assert Path(clip_path).is_file(), "Exported clip was not created!"
    assert Path(clip_path).stat().st_size > 0, "Exported clip is empty!"
    print(f"  [OK] FFmpeg Clip exported ({Path(clip_path).stat().st_size / 1024:.1f} KB)")

    # 4. Test save_session_activity
    print(f"[Activity 4] save_session_activity -> Saving local JSON & MongoDB Atlas vectors...")
    session_data = {
        "session_id": test_session_id,
        "source_name": video_path.name,
        "duration_seconds": duration,
        "model_name": model_text,
        "events": [analysis_res],
        "warnings": [],
    }
    saved_path = await save_session_activity(SaveSessionInput(session_dict=session_data))
    assert Path(saved_path).is_file(), "Session JSON file was not saved!"
    print(f"  [OK] Session saved successfully at: {saved_path}")


async def test_workflow_orchestration(video_path: Path):
    print("\n--- [TEST 2/3] Temporal Workflow Orchestration Verification ---")
    duration, _, _ = get_video_info(video_path)
    model_text = GOOGLE_CLOUD_MODEL if ENGINE in ("gemini_api", "google_cloud") else MODEL_NAME
    session_id = "testwf" + secrets.token_hex(4)

    print("  Starting Temporal test cluster...")
    async with await WorkflowEnvironment.start_local() as env:
        print("  Registering Worker on task queue 'drivesafe-analysis-queue'...")
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
                model_name=model_text,
                window_seconds=10,
                samples_per_window=4,
            )

            print(f"  Starting VideoAnalysisWorkflow (id='workflow-{session_id}')...")
            handle = await env.client.start_workflow(
                VideoAnalysisWorkflow.run,
                input_data,
                id=f"workflow-{session_id}",
                task_queue=TASK_QUEUE,
            )

            # Poll workflow queries
            steps_observed = []
            while True:
                prog = await handle.query(VideoAnalysisWorkflow.get_progress)
                if prog["current_step"] not in steps_observed:
                    steps_observed.append(prog["current_step"])
                    print(f"    [Workflow Query Update] Step: {prog['current_step']} | Progress: {prog['percent']}% | Msg: {prog['status_message']}")
                
                if prog.get("is_completed") or prog.get("error"):
                    break
                await asyncio.sleep(0.5)

            result = await handle.result()
            assert result.get("session_id") == session_id, "Session ID mismatch in workflow result!"
            assert "events" in result, "Missing events in workflow result!"
            print(f"  [OK] Workflow completed successfully with {len(result['events'])} detected event(s)!")


async def test_streamlit_integration(video_path: Path):
    print("\n--- [TEST 3/3] Streamlit Async Helper Verification ---")
    from app import run_temporal_analysis

    class MockStatus:
        def info(self, msg): pass
        def empty(self): pass

    class MockProgress:
        def progress(self, val): pass
        def empty(self): pass

    duration, _, _ = get_video_info(video_path)
    model_text = GOOGLE_CLOUD_MODEL if ENGINE in ("gemini_api", "google_cloud") else MODEL_NAME
    session_id = "testui" + secrets.token_hex(4)

    input_data = AnalysisWorkflowInput(
        session_id=session_id,
        source_name=video_path.name,
        source_path_str=str(video_path.resolve()),
        duration_seconds=duration,
        model_name=model_text,
        window_seconds=10,
        samples_per_window=4,
    )

    print("  Executing run_temporal_analysis() async helper...")
    res = await run_temporal_analysis(input_data, MockStatus(), MockProgress())
    assert res is not None, "run_temporal_analysis returned None!"
    assert res.get("session_id") == session_id, "Session ID mismatch!"
    print(f"  [OK] Streamlit helper returned valid AnalysisSession dict with {len(res.get('events', []))} event(s)!")


async def main():
    print("==============================================================")
    print("  DRIVESAFE TEMPORAL FULL TEST SUITE")
    print("==============================================================")
    
    test_video = Path("person-bicycle-car-detection.mp4")
    if not test_video.exists():
        test_video = Path("sample_dashcam.mp4")
    
    if not test_video.exists():
        print("ERROR: Test video not found!")
        sys.exit(1)

    print(f"Using test video: {test_video.resolve()}")

    try:
        await test_activities_directly(test_video)
        await test_workflow_orchestration(test_video)
        await test_streamlit_integration(test_video)
        
        print("\n==============================================================")
        print("  ALL 3 TEST SUITES PASSED PERFECTLY! [SUCCESS]")
        print("==============================================================\n")
    except Exception as exc:
        print(f"\n[FAIL] TEST FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
