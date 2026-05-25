from __future__ import annotations

from typing import Any

from app.services import async_jobs as async_job_service


async def handle_demo_echo(job_id: int, payload: dict[str, Any]) -> None:
    await async_job_service.mark_processing(job_id)
    await async_job_service.mark_success(
        job_id,
        {
            "echo": payload,
            "handler": "DEMO_ECHO",
        },
    )


async def handle_stream_job(job_id: int, job_type: str, payload: dict[str, Any]) -> None:
    if job_type == async_job_service.DEMO_ECHO_JOB_TYPE:
        await handle_demo_echo(job_id, payload)
        return
    await async_job_service.mark_failed(job_id, f"unsupported_job_type: {job_type}")
