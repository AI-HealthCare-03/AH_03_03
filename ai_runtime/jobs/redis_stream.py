from __future__ import annotations

import json
import socket
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.core import config

ASYNC_JOB_STREAM = "ai_health:async_jobs"
ASYNC_JOB_GROUP = "ai_runtime"
DEFAULT_CONSUMER_NAME = f"ai-worker-{socket.gethostname()}"


def create_redis_client() -> Redis:
    return Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=10,
    )


def build_stream_fields(job_id: int, job_type: str, payload: dict[str, Any]) -> dict[str, str]:
    return {
        "job_id": str(job_id),
        "job_type": job_type,
        "payload": json.dumps(payload, ensure_ascii=False, default=str),
    }


def parse_stream_fields(fields: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
    job_id = int(fields["job_id"])
    job_type = str(fields["job_type"])
    payload_raw = fields.get("payload") or "{}"
    payload = json.loads(payload_raw)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return job_id, job_type, payload


async def ensure_consumer_group(redis_client: Redis) -> None:
    try:
        await redis_client.xgroup_create(ASYNC_JOB_STREAM, ASYNC_JOB_GROUP, id="0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def enqueue_async_job(job_id: int, job_type: str, payload: dict[str, Any]) -> str:
    redis_client = create_redis_client()
    try:
        fields = build_stream_fields(job_id=job_id, job_type=job_type, payload=payload)
        stream_id = await redis_client.xadd(ASYNC_JOB_STREAM, fields)
        return str(stream_id)
    finally:
        await redis_client.aclose()
