from __future__ import annotations

import logging
from typing import Any

from tortoise.timezone import now

from ai_runtime.jobs.redis_stream import AI_JOB_STREAM, JobStream, enqueue_async_job
from app.models.async_jobs import AsyncJob, AsyncJobStatus

logger = logging.getLogger(__name__)

DEMO_ECHO_JOB_TYPE = "DEMO_ECHO"
EXAM_OCR_JOB_TYPE = "exam_ocr.run"
DIET_ANALYZE_IMAGE_JOB_TYPE = "diet.analyze_image"
ANALYSIS_RUN_JOB_TYPE = "analysis.run"


async def create_async_job(
    *,
    job_type: str,
    request_payload: dict[str, Any],
    stream: str | JobStream = AI_JOB_STREAM,
    user_id: int | None = None,
    resource_id: int | None = None,
    idempotency_key: str | None = None,
    max_attempts: int = 3,
    stream_payload: dict[str, Any] | None = None,
) -> AsyncJob:
    job = await AsyncJob.create(
        job_type=job_type,
        status=AsyncJobStatus.PENDING,
        request_payload=request_payload,
    )
    try:
        stream_id = await enqueue_async_job(
            job_id=int(job.id),
            job_type=job.job_type,
            payload=stream_payload or request_payload,
            stream=stream,
            user_id=user_id,
            resource_id=resource_id,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
        )
    except Exception as exc:
        logger.exception("Failed to enqueue async job", extra={"job_id": int(job.id), "job_type": job_type})
        return await mark_failed(int(job.id), f"enqueue_failed: {exc.__class__.__name__}")

    job.stream_id = stream_id
    await job.save(update_fields=["stream_id", "updated_at"])
    return job


async def create_demo_job(request_payload: dict[str, Any]) -> AsyncJob:
    return await create_async_job(
        job_type=DEMO_ECHO_JOB_TYPE,
        request_payload=request_payload,
        stream=AI_JOB_STREAM,
    )


async def create_exam_ocr_job(user_id: int, exam_report_id: int) -> AsyncJob:
    return await create_async_job(
        job_type=EXAM_OCR_JOB_TYPE,
        request_payload={
            "user_id": user_id,
            "exam_report_id": exam_report_id,
            "resource_type": "exam_report",
        },
        stream=AI_JOB_STREAM,
        user_id=user_id,
        resource_id=exam_report_id,
        idempotency_key=f"exam_ocr:{exam_report_id}",
    )


async def create_diet_analyze_image_job(user_id: int, request_payload: dict[str, Any]) -> AsyncJob:
    return await create_async_job(
        job_type=DIET_ANALYZE_IMAGE_JOB_TYPE,
        request_payload={
            **request_payload,
            "user_id": user_id,
            "resource_type": "diet_analysis_request",
        },
        stream=AI_JOB_STREAM,
        user_id=user_id,
        idempotency_key=request_payload.get("idempotency_key"),
        stream_payload={
            "resource_type": "diet_analysis_request",
        },
    )


async def create_analysis_run_job(user_id: int, health_record_id: int, mode: str) -> AsyncJob:
    request_payload = {
        "user_id": user_id,
        "health_record_id": health_record_id,
        "mode": mode,
        "resource_type": "health_record",
    }
    return await create_async_job(
        job_type=ANALYSIS_RUN_JOB_TYPE,
        request_payload=request_payload,
        stream=AI_JOB_STREAM,
        user_id=user_id,
        resource_id=health_record_id,
        stream_payload=request_payload,
    )


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


async def mark_retry_scheduled(job_id: int, error_message: str) -> AsyncJob | None:
    job = await get_job(job_id)
    if job is None:
        return None
    job.status = AsyncJobStatus.PENDING
    job.error_message = error_message
    await job.save(update_fields=["status", "error_message", "updated_at"])
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
