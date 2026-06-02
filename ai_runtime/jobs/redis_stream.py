from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.core import config


class JobStream(StrEnum):
    AI = "ai_health:jobs:ai"
    SERVICE = "ai_health:jobs:service"
    DLQ = "ai_health:jobs:dlq"


AI_JOB_STREAM = JobStream.AI.value
SERVICE_JOB_STREAM = JobStream.SERVICE.value
DLQ_JOB_STREAM = JobStream.DLQ.value
CONSUMER_JOB_STREAMS = (AI_JOB_STREAM, SERVICE_JOB_STREAM)
ASYNC_JOB_STREAM = AI_JOB_STREAM
ASYNC_JOB_GROUP = "ai_runtime"
DEFAULT_CONSUMER_NAME = f"ai-worker-{socket.gethostname()}"
DEFAULT_MAX_ATTEMPTS = 3
STREAM_MAXLEN = 10_000
RETRY_BASE_DELAY_SECONDS = 30
RETRY_MAX_DELAY_SECONDS = 30 * 60


@dataclass(frozen=True)
class StreamJobMessage:
    job_id: int
    job_type: str
    stream: str
    payload: dict[str, Any]
    user_id: int | None
    resource_id: int | None
    idempotency_key: str | None
    attempts: int
    max_attempts: int
    available_at: datetime
    created_at: datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def serialize_datetime(value: datetime | None) -> str:
    normalized = value or utc_now()
    if normalized.tzinfo is None:
        normalized = normalized.replace(tzinfo=UTC)
    return normalized.astimezone(UTC).isoformat()


def parse_datetime(value: Any) -> datetime:
    if not value:
        return utc_now()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return utc_now()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def create_redis_client() -> Redis:
    return Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=10,
    )


def normalize_stream_name(stream: str | JobStream | None) -> str:
    if isinstance(stream, JobStream):
        return stream.value
    if stream in {"ai", "AI"}:
        return AI_JOB_STREAM
    if stream in {"service", "SERVICE"}:
        return SERVICE_JOB_STREAM
    if stream in {"dlq", "DLQ"}:
        return DLQ_JOB_STREAM
    return str(stream or AI_JOB_STREAM)


def build_stream_fields(
    job_id: int,
    job_type: str,
    payload: dict[str, Any],
    *,
    stream: str | JobStream = AI_JOB_STREAM,
    user_id: int | None = None,
    resource_id: int | None = None,
    idempotency_key: str | None = None,
    attempts: int = 0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    available_at: datetime | None = None,
    created_at: datetime | None = None,
) -> dict[str, str]:
    # Redis Stream payload에는 이미지/PDF bytes나 민감 건강정보 원문을 직접 넣지 않는다.
    # 파일/민감 원문은 DB 또는 안전한 파일 저장소에 두고, payload에는 참조 ID와 최소 메타데이터만 담는다.
    stream_name = normalize_stream_name(stream)
    created_at = created_at or utc_now()
    available_at = available_at or created_at
    return {
        "job_id": str(job_id),
        "job_type": job_type,
        "stream": stream_name,
        "user_id": "" if user_id is None else str(user_id),
        "resource_id": "" if resource_id is None else str(resource_id),
        "payload": json.dumps(payload, ensure_ascii=False, default=str),
        "idempotency_key": idempotency_key or "",
        "attempts": str(max(0, attempts)),
        "max_attempts": str(max(1, max_attempts)),
        "available_at": serialize_datetime(available_at),
        "created_at": serialize_datetime(created_at),
    }


def parse_stream_fields(fields: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
    job = parse_stream_job(fields)
    return job.job_id, job.job_type, job.payload


def parse_stream_job(fields: dict[str, Any]) -> StreamJobMessage:
    job_id = int(fields["job_id"])
    job_type = str(fields["job_type"])
    payload_raw = fields.get("payload") or "{}"
    payload = json.loads(payload_raw)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    stream = normalize_stream_name(str(fields.get("stream") or AI_JOB_STREAM))
    return StreamJobMessage(
        job_id=job_id,
        job_type=job_type,
        stream=stream,
        payload=payload,
        user_id=parse_optional_int(fields.get("user_id")),
        resource_id=parse_optional_int(fields.get("resource_id")),
        idempotency_key=str(fields.get("idempotency_key") or "") or None,
        attempts=max(0, parse_optional_int(fields.get("attempts")) or 0),
        max_attempts=max(1, parse_optional_int(fields.get("max_attempts")) or DEFAULT_MAX_ATTEMPTS),
        available_at=parse_datetime(fields.get("available_at")),
        created_at=parse_datetime(fields.get("created_at")),
    )


async def ensure_consumer_group(redis_client: Redis, streams: tuple[str, ...] = CONSUMER_JOB_STREAMS) -> None:
    for stream in streams:
        try:
            await redis_client.xgroup_create(stream, ASYNC_JOB_GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise


async def enqueue_async_job(
    job_id: int,
    job_type: str,
    payload: dict[str, Any],
    *,
    stream: str | JobStream = AI_JOB_STREAM,
    user_id: int | None = None,
    resource_id: int | None = None,
    idempotency_key: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    available_at: datetime | None = None,
) -> str:
    redis_client = create_redis_client()
    try:
        stream_name = normalize_stream_name(stream)
        fields = build_stream_fields(
            job_id=job_id,
            job_type=job_type,
            payload=payload,
            stream=stream_name,
            user_id=user_id,
            resource_id=resource_id,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            available_at=available_at,
        )
        stream_id = await redis_client.xadd(stream_name, fields, maxlen=STREAM_MAXLEN, approximate=True)
        return str(stream_id)
    finally:
        await redis_client.aclose()


def is_job_available(job: StreamJobMessage, now: datetime | None = None) -> bool:
    return job.available_at <= (now or utc_now())


def next_retry_available_at(attempts: int, now: datetime | None = None) -> datetime:
    delay = min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2 ** max(0, attempts - 1)))
    return (now or utc_now()) + timedelta(seconds=delay)


async def enqueue_retry(redis_client: Redis, job: StreamJobMessage) -> str:
    next_attempts = job.attempts + 1
    fields = build_stream_fields(
        job_id=job.job_id,
        job_type=job.job_type,
        payload=job.payload,
        stream=job.stream,
        user_id=job.user_id,
        resource_id=job.resource_id,
        idempotency_key=job.idempotency_key,
        attempts=next_attempts,
        max_attempts=job.max_attempts,
        available_at=next_retry_available_at(next_attempts),
        created_at=job.created_at,
    )
    return str(await redis_client.xadd(job.stream, fields, maxlen=STREAM_MAXLEN, approximate=True))


async def move_to_dlq(
    redis_client: Redis,
    job: StreamJobMessage,
    *,
    source_stream: str,
    source_stream_id: str,
    error_message: str,
) -> str:
    fields = build_stream_fields(
        job_id=job.job_id,
        job_type=job.job_type,
        payload={
            **job.payload,
            "error_message": error_message,
            "source_stream": source_stream,
            "source_stream_id": source_stream_id,
        },
        stream=DLQ_JOB_STREAM,
        user_id=job.user_id,
        resource_id=job.resource_id,
        idempotency_key=job.idempotency_key,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        available_at=utc_now(),
        created_at=job.created_at,
    )
    fields["failed_at"] = serialize_datetime(utc_now())
    fields["error_message"] = error_message
    fields["source_stream"] = source_stream
    fields["source_stream_id"] = source_stream_id
    return str(await redis_client.xadd(DLQ_JOB_STREAM, fields, maxlen=STREAM_MAXLEN, approximate=True))
