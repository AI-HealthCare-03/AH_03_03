from __future__ import annotations

import asyncio
import logging

from tortoise import Tortoise

from ai_runtime.jobs.handlers import handle_stream_job
from ai_runtime.jobs.redis_stream import (
    ASYNC_JOB_GROUP,
    ASYNC_JOB_STREAM,
    DEFAULT_CONSUMER_NAME,
    create_redis_client,
    ensure_consumer_group,
    parse_stream_fields,
)
from app.core.db.databases import TORTOISE_ORM
from app.services import async_jobs as async_job_service

logger = logging.getLogger(__name__)


async def _initialize_db() -> None:
    if Tortoise._inited:
        return
    await Tortoise.init(config=TORTOISE_ORM)


async def _close_db() -> None:
    if Tortoise._inited:
        await Tortoise.close_connections()


async def process_stream_message(stream_id: str, fields: dict) -> None:
    job_id: int | None = None
    try:
        job_id, job_type, payload = parse_stream_fields(fields)
        await handle_stream_job(job_id, job_type, payload)
    except Exception as exc:
        logger.exception("Failed to process async job stream message", extra={"stream_id": stream_id, "job_id": job_id})
        if job_id is not None:
            await async_job_service.mark_failed(job_id, f"handler_failed: {exc.__class__.__name__}")


async def run_consumer_forever(stop_event: asyncio.Event, consumer_name: str = DEFAULT_CONSUMER_NAME) -> None:
    await _initialize_db()
    try:
        while not stop_event.is_set():
            redis_client = create_redis_client()
            try:
                await ensure_consumer_group(redis_client)
                logger.info("AI Worker Redis Stream consumer started", extra={"stream": ASYNC_JOB_STREAM})
                while not stop_event.is_set():
                    response = await redis_client.xreadgroup(
                        groupname=ASYNC_JOB_GROUP,
                        consumername=consumer_name,
                        streams={ASYNC_JOB_STREAM: ">"},
                        count=1,
                        block=5000,
                    )
                    if not response:
                        continue
                    for _stream_name, messages in response:
                        for stream_id, fields in messages:
                            await process_stream_message(stream_id, fields)
                            await redis_client.xack(ASYNC_JOB_STREAM, ASYNC_JOB_GROUP, stream_id)
            except Exception:
                logger.exception("AI Worker Redis Stream consumer connection failed; retrying")
                await asyncio.sleep(5)
            finally:
                await redis_client.aclose()
    finally:
        await _close_db()
