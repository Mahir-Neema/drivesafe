"""Client interface for interacting with DriveSafe Temporal workflows."""

import os
from typing import Any

from temporalio.client import Client

from drivesafe.workflows import AnalysisWorkflowInput, VideoAnalysisWorkflow

TEMPORAL_HOST = os.environ.get("TEMPORAL_HOST", "localhost:7233")
TASK_QUEUE = "drivesafe-analysis-queue"


async def get_temporal_client(target_host: str = TEMPORAL_HOST) -> Client:
    """Connect to a running Temporal cluster / dev server."""
    return await Client.connect(target_host)


async def check_temporal_connected(target_host: str = TEMPORAL_HOST) -> bool:
    """Check if Temporal server is reachable."""
    try:
        client = await Client.connect(target_host)
        # Test basic cluster connectivity
        await client.service_client.check_health()
        return True
    except Exception:
        return False


async def run_analysis_workflow(
    input_data: AnalysisWorkflowInput,
    target_host: str = TEMPORAL_HOST,
) -> dict[str, Any]:
    """Execute the VideoAnalysisWorkflow and wait for results."""
    client = await get_temporal_client(target_host)
    workflow_id = f"drivesafe-{input_data.session_id}"
    
    result = await client.execute_workflow(
        VideoAnalysisWorkflow.run,
        input_data,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    return result
