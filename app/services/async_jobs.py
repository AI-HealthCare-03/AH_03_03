from __future__ import annotations

import logging
from typing import Any

from tortoise.timezone import now

from ai_worker.jobs.redis_stream import enqueue_async_job
from app.models.async_jobs import AsyncJob, AsyncJobStatus

logger = logging.getLogger(__name__)

DEMO_ECHO_JOB_TYPE = "DEMO_ECHO"


async def create_demo_job(request_payload: dict[str, Any]) -> AsyncJob:
    job = await AsyncJob.create(
        job_type=DEMO_ECHO_JOB_TYPE,
        status=AsyncJobStatus.PENDING,
        request_payload=request_payload,
    )
    try:
        stream_id = await enqueue_async_job(
            job_id=int(job.id),
            job_type=job.job_type,
            payload=request_payload,
        )
    except Exception as exc:
        logger.exception("Failed to enqueue async demo job", extra={"job_id": int(job.id)})
        return await mark_failed(int(job.id), f"enqueue_failed: {exc.__class__.__name__}")

    job.stream_id = stream_id
    await job.save(update_fields=["stream_id", "updated_at"])
    return job


async def get_job(job_id: int) -> AsyncJob | None:
    return await AsyncJob.get_or_none(id=job_id)


async def mark_processing(job_id: int) -> AsyncJob | None:
    job = await get_job(job_id)
    if job is None:
        return None
    job.status = AsyncJobStatus.PROCESSING
    job.started_at = job.started_at or now()
    job.error_message = None
    await job.save(update_fields=["status", "started_at", "error_message", "updated_at"])
    return job


async def mark_success(job_id: int, result_payload: dict[str, Any]) -> AsyncJob | None:
    job = await get_job(job_id)
    if job is None:
        return None
    job.status = AsyncJobStatus.SUCCESS
    job.result_payload = result_payload
    job.error_message = None
    job.finished_at = now()
    await job.save(update_fields=["status", "result_payload", "error_message", "finished_at", "updated_at"])
    return job


async def mark_failed(job_id: int, error_message: str) -> AsyncJob | None:
    job = await get_job(job_id)
    if job is None:
        return None
    job.status = AsyncJobStatus.FAILED
    job.error_message = error_message
    job.finished_at = now()
    await job.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
    return job
