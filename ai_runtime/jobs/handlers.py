from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.services import async_jobs as async_job_service

JobHandler = Callable[[int, dict[str, Any]], Awaitable[None]]
JOB_HANDLERS: dict[str, JobHandler] = {}


class NonRetryableJobError(RuntimeError):
    """Base error for jobs that should be sent to DLQ without another retry."""


class UnsupportedJobTypeError(NonRetryableJobError):
    def __init__(self, job_type: str) -> None:
        super().__init__(f"unsupported_job_type: {job_type}")
        self.job_type = job_type


def register_job_handler(job_type: str) -> Callable[[JobHandler], JobHandler]:
    def decorator(handler: JobHandler) -> JobHandler:
        JOB_HANDLERS[job_type] = handler
        return handler

    return decorator


@register_job_handler(async_job_service.DEMO_ECHO_JOB_TYPE)
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
    handler = JOB_HANDLERS.get(job_type)
    if handler is None:
        await async_job_service.mark_failed(job_id, f"unsupported_job_type: {job_type}")
        raise UnsupportedJobTypeError(job_type)
    await handler(job_id, payload)
