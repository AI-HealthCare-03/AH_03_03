from __future__ import annotations

import asyncio
import logging

from tortoise import Tortoise

from ai_worker.jobs.handlers import handle_stream_job
from ai_worker.jobs.redis_stream import (
    ASYNC_JOB_GROUP,
    ASYNC_JOB_STREAM,
    DEFAULT_CONSUMER_NAME,
    create_redis_client,
    ensure_consumer_group,
    parse_stream_fields,
)
from app.core.db.databases import TORTOISE_ORM

logger = logging.getLogger(__name__)


async def _initialize_db() -> None:
    if Tortoise._inited:
        return
    await Tortoise.init(config=TORTOISE_ORM)


async def _close_db() -> None:
    if Tortoise._inited:
        await Tortoise.close_connections()


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
                            try:
                                job_id, job_type, payload = parse_stream_fields(fields)
                                await handle_stream_job(job_id, job_type, payload)
                            except Exception:
                                logger.exception(
                                    "Failed to process async job stream message", extra={"stream_id": stream_id}
                                )
                            finally:
                                await redis_client.xack(ASYNC_JOB_STREAM, ASYNC_JOB_GROUP, stream_id)
            except Exception:
                logger.exception("AI Worker Redis Stream consumer connection failed; retrying")
                await asyncio.sleep(5)
            finally:
                await redis_client.aclose()
    finally:
        await _close_db()
