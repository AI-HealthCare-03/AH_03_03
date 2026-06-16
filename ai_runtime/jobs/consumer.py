from __future__ import annotations

import asyncio
import logging
from typing import Any

from tortoise import Tortoise

from ai_runtime.jobs.handlers import NonRetryableJobError, handle_stream_job
from ai_runtime.jobs.redis_stream import (
    ASYNC_JOB_GROUP,
    CONSUMER_JOB_STREAMS,
    DEFAULT_CONSUMER_NAME,
    create_redis_client,
    enqueue_retry,
    ensure_consumer_group,
    is_job_available,
    move_to_dlq,
    parse_stream_job,
)
from app.core.db.databases import TORTOISE_ORM
from app.services import async_jobs as async_job_service

logger = logging.getLogger(__name__)
PENDING_IDLE_MS = 60_000
PENDING_CLAIM_COUNT = 10


async def _initialize_db() -> None:
    if Tortoise._inited:
        return
    await Tortoise.init(config=TORTOISE_ORM)


async def _close_db() -> None:
    if Tortoise._inited:
        await Tortoise.close_connections()


async def process_stream_message(redis_client, stream_name: str, stream_id: str, fields: dict[str, Any]) -> bool:
    job_id: int | None = None
    job = parse_stream_job(fields)
    job_id = job.job_id
    if not is_job_available(job):
        logger.info(
            "Async job is not available yet; leaving message pending",
            extra={"stream": stream_name, "stream_id": stream_id, "job_id": job_id, "available_at": job.available_at},
        )
        return False

    try:
        await handle_stream_job(job.job_id, job.job_type, job.payload)
    except NonRetryableJobError as exc:
        error_message = str(exc)
        await move_to_dlq(
            redis_client,
            job,
            source_stream=stream_name,
            source_stream_id=stream_id,
            error_message=error_message,
        )
        await async_job_service.mark_failed(job.job_id, error_message)
        await redis_client.xack(stream_name, ASYNC_JOB_GROUP, stream_id)
        return True
    except Exception as exc:
        error_message = f"handler_failed: {exc.__class__.__name__}"
        next_attempts = job.attempts + 1
        logger.exception(
            "Failed to process async job stream message",
            extra={
                "stream": stream_name,
                "stream_id": stream_id,
                "job_id": job_id,
                "attempts": next_attempts,
                "max_attempts": job.max_attempts,
            },
        )
        if next_attempts >= job.max_attempts:
            await move_to_dlq(
                redis_client,
                job,
                source_stream=stream_name,
                source_stream_id=stream_id,
                error_message=error_message,
            )
            await async_job_service.mark_failed(job.job_id, error_message)
        else:
            await enqueue_retry(redis_client, job)
            await async_job_service.mark_retry_scheduled(job.job_id, f"retry_scheduled: {error_message}")
        await redis_client.xack(stream_name, ASYNC_JOB_GROUP, stream_id)
        return True

    await redis_client.xack(stream_name, ASYNC_JOB_GROUP, stream_id)
    return True


def _claimed_messages_from_xautoclaim_result(result: Any) -> list[tuple[str, dict[str, Any]]]:
    if not result:
        return []
    if isinstance(result, list | tuple) and len(result) >= 2 and isinstance(result[1], list):
        return result[1]
    if isinstance(result, list):
        return result
    return []


async def recover_pending_messages(
    redis_client,
    consumer_name: str = DEFAULT_CONSUMER_NAME,
    streams: tuple[str, ...] = CONSUMER_JOB_STREAMS,
    min_idle_ms: int = PENDING_IDLE_MS,
    count: int = PENDING_CLAIM_COUNT,
) -> int:
    recovered = 0
    for stream_name in streams:
        result = await redis_client.xautoclaim(
            stream_name,
            ASYNC_JOB_GROUP,
            consumer_name,
            min_idle_ms,
            "0-0",
            count=count,
        )
        for stream_id, fields in _claimed_messages_from_xautoclaim_result(result):
            await process_stream_message(redis_client, stream_name, stream_id, fields)
            recovered += 1
    return recovered


async def run_consumer_forever(stop_event: asyncio.Event, consumer_name: str = DEFAULT_CONSUMER_NAME) -> None:
    await _initialize_db()
    try:
        while not stop_event.is_set():
            redis_client = create_redis_client()
            try:
                await ensure_consumer_group(redis_client)
                logger.info(
                    "AI Worker Redis Stream consumer started", extra={"streams": ",".join(CONSUMER_JOB_STREAMS)}
                )
                while not stop_event.is_set():
                    await recover_pending_messages(redis_client, consumer_name=consumer_name)
                    response = await redis_client.xreadgroup(
                        groupname=ASYNC_JOB_GROUP,
                        consumername=consumer_name,
                        streams={stream_name: ">" for stream_name in CONSUMER_JOB_STREAMS},
                        count=1,
                        block=5000,
                    )
                    if not response:
                        continue
                    for stream_name, messages in response:
                        for stream_id, fields in messages:
                            await process_stream_message(redis_client, str(stream_name), stream_id, fields)
            except Exception:
                logger.exception("AI Worker Redis Stream consumer connection failed; retrying")
                await asyncio.sleep(5)
            finally:
                await redis_client.aclose()
    finally:
        await _close_db()
