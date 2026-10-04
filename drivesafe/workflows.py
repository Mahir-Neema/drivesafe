"""Temporal Workflows for durable DriveSafe video analysis."""

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

# Import activities using unsafe imports for workflow sandbox
with workflow.unsafe.imports_passed_through():
    from drivesafe.activities import (
        AnalyzeWindowInput,
        ExportClipInput,
        ExtractWindowsInput,
        ExtractWindowsOutput,
        SaveSessionInput,
        analyze_window_activity,
        export_clip_activity,
        extract_windows_activity,
        save_session_activity,
    )


@dataclass
class AnalysisWorkflowInput:
    session_id: str
    source_name: str
    source_path_str: str
    duration_seconds: float
    model_name: str
    engine: str = "gemini_api"
    window_seconds: int = 10
    samples_per_window: int = 4
    context_seconds: int = 5
    max_concurrent_analysis: int = 3


@dataclass
class WorkflowProgress:
    current_step: str
    completed_windows: int
    total_windows: int
    percent: int
    status_message: str
    events_count: int
    is_completed: bool = False
    is_cancelled: bool = False
    error: str | None = None


@workflow.defn
class VideoAnalysisWorkflow:
    """Durable workflow for processing dashcam video through vision AI."""

    def __init__(self) -> None:
        self.current_step = "initializing"
        self.completed_windows = 0
        self.total_windows = 0
        self.percent = 0
        self.status_message = "Workflow initialized..."
        self.events_count = 0
        self.is_completed = False
        self.is_cancelled = False
        self.error: str | None = None
        self.events: list[dict[str, Any]] = []

    @workflow.query
    def get_progress(self) -> dict[str, Any]:
        """Query method to read real-time analysis progress from Streamlit."""
        return {
            "current_step": self.current_step,
            "completed_windows": self.completed_windows,
            "total_windows": self.total_windows,
            "percent": self.percent,
            "status_message": self.status_message,
            "events_count": self.events_count,
            "is_completed": self.is_completed,
            "is_cancelled": self.is_cancelled,
            "error": self.error,
        }

    @workflow.signal
    def cancel_analysis(self) -> None:
        """Signal method to gracefully cancel workflow execution."""
        self.is_cancelled = True
        self.status_message = "Analysis cancelled by user."

    @workflow.run
    async def run(self, input: AnalysisWorkflowInput) -> dict[str, Any]:
        """Main durable execution pipeline."""
        # Standard retry policy for video/file operations
        standard_retry = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=10),
            maximum_attempts=3,
        )

        # Resilient retry policy for Gemini Vision API calls (survives 429 rate limits & network drops)
        vision_api_retry = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=5,
        )

        try:
            # ---------------------------------------------------------------
            # Step 1: Extract & Sample Video Windows
            # ---------------------------------------------------------------
            self.current_step = "extracting_frames"
            self.status_message = "Extracting and sampling video frames..."
            self.percent = 5

            extract_result: ExtractWindowsOutput = await workflow.execute_activity(
                extract_windows_activity,
                ExtractWindowsInput(
                    source_path_str=input.source_path_str,
                    duration_seconds=input.duration_seconds,
                    window_seconds=input.window_seconds,
                    samples_per_window=input.samples_per_window,
                ),
                start_to_close_timeout=timedelta(minutes=3),
                retry_policy=standard_retry,
            )

            windows = extract_result.windows
            self.total_windows = len(windows)
            self.status_message = f"Sampled {len(windows)} windows. Running Gemma 4 Vision..."
            self.percent = 15

            if self.is_cancelled:
                return {"status": "cancelled"}

            # ---------------------------------------------------------------
            # Step 2: Analyze Windows (with automatic activity retry)
            # ---------------------------------------------------------------
            self.current_step = "analyzing_windows"
            analyzed_results = []

            for idx, window_dict in enumerate(windows, start=1):
                if self.is_cancelled:
                    return {"status": "cancelled"}

                t_start = window_dict["start_seconds"]
                t_end = window_dict["end_seconds"]
                self.status_message = f"Analyzing {t_start:.0f}s - {t_end:.0f}s with Gemma 4..."

                # Execute analysis activity with exponential backoff on flaky API calls
                res_dict: dict[str, Any] = await workflow.execute_activity(
                    analyze_window_activity,
                    AnalyzeWindowInput(
                        window_dict=window_dict,
                        model_name=input.model_name,
                        engine=input.engine,
                    ),
                    start_to_close_timeout=timedelta(minutes=3),
                    retry_policy=vision_api_retry,
                )

                if res_dict.get("event_type") and res_dict["event_type"] != "none":
                    self.events_count += 1

                analyzed_results.append(res_dict)
                self.completed_windows = idx
                self.percent = 15 + int((idx / self.total_windows) * 65)

            # ---------------------------------------------------------------
            # Step 3: Export Clips for Flagged Moments (FFmpeg)
            # ---------------------------------------------------------------
            self.current_step = "exporting_clips"
            self.status_message = "Exporting review clips with FFmpeg..."
            flagged_events = []

            for idx, res in enumerate(analyzed_results):
                if res.get("event_type") and res["event_type"] != "none":
                    clip_name = f"event-{idx:03d}.mp4"

                    try:
                        await workflow.execute_activity(
                            export_clip_activity,
                            ExportClipInput(
                                source_path_str=input.source_path_str,
                                start_seconds=res["start_seconds"],
                                end_seconds=res["end_seconds"],
                                clip_name=clip_name,
                                duration_seconds=input.duration_seconds,
                                context_seconds=input.context_seconds,
                            ),
                            start_to_close_timeout=timedelta(minutes=2),
                            retry_policy=standard_retry,
                        )
                        res["clip_name"] = clip_name
                    except Exception as exc:
                        res["warning"] = f"Failed to export clip: {exc}"

                    flagged_events.append(res)

            self.events = flagged_events
            self.percent = 90

            # ---------------------------------------------------------------
            # Step 4: Save Session and Sync to MongoDB Atlas
            # ---------------------------------------------------------------
            self.current_step = "saving_session"
            self.status_message = "Saving session & syncing vector embeddings to Atlas..."

            final_session_dict = {
                "session_id": input.session_id,
                "source_name": input.source_name,
                "duration_seconds": input.duration_seconds,
                "model_name": input.model_name,
                "events": flagged_events,
                "warnings": [],
            }

            await workflow.execute_activity(
                save_session_activity,
                SaveSessionInput(session_dict=final_session_dict),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=standard_retry,
            )

            # Done!
            self.current_step = "completed"
            self.status_message = f"Analysis complete! {len(flagged_events)} events detected."
            self.percent = 100
            self.is_completed = True

            return final_session_dict

        except Exception as exc:
            self.error = str(exc)
            self.status_message = f"Workflow failed: {exc}"
            raise
