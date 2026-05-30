from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_runtime.jobs import consumer, handlers, redis_stream
from ai_runtime.jobs.redis_stream import (
    AI_JOB_STREAM,
    ASYNC_JOB_GROUP,
    DLQ_JOB_STREAM,
    STREAM_MAXLEN,
    build_stream_fields,
    parse_stream_fields,
    parse_stream_job,
)


class FakeRedis:
    def __init__(self, claimed_messages=None) -> None:
        self.xadds = []
        self.xacks = []
        self.xautoclaim_calls = []
        self.claimed_messages = claimed_messages or []

    async def xadd(self, name, fields, **kwargs):
        self.xadds.append((name, fields, kwargs))
        return "2-0"

    async def xack(self, stream_name, group_name, stream_id):
        self.xacks.append((stream_name, group_name, stream_id))
        return 1

    async def xautoclaim(self, stream_name, group_name, consumer_name, min_idle_ms, start_id, count):
        self.xautoclaim_calls.append((stream_name, group_name, consumer_name, min_idle_ms, start_id, count))
        return ("0-0", self.claimed_messages, [])

    async def aclose(self):
        return None


def test_stream_fields_round_trip_preserves_demo_payload() -> None:
    created_at = datetime(2026, 5, 30, tzinfo=UTC)
    fields = build_stream_fields(
        job_id=7,
        job_type="DEMO_ECHO",
        payload={"message": "hello", "payload": {"source": "test"}},
        stream=AI_JOB_STREAM,
        user_id=3,
        resource_id=17,
        idempotency_key="demo-7",
        attempts=1,
        max_attempts=4,
        created_at=created_at,
        available_at=created_at,
    )

    job_id, job_type, payload = parse_stream_fields(fields)
    job = parse_stream_job(fields)

    assert job_id == 7
    assert job_type == "DEMO_ECHO"
    assert payload == {"message": "hello", "payload": {"source": "test"}}
    assert job.stream == AI_JOB_STREAM
    assert job.user_id == 3
    assert job.resource_id == 17
    assert job.idempotency_key == "demo-7"
    assert job.attempts == 1
    assert job.max_attempts == 4


@pytest.mark.asyncio
async def test_enqueue_async_job_writes_to_named_stream_with_maxlen(monkeypatch) -> None:
    fake_redis = FakeRedis()
    monkeypatch.setattr(redis_stream, "create_redis_client", lambda: fake_redis)

    stream_id = await redis_stream.enqueue_async_job(
        job_id=8,
        job_type="DEMO_ECHO",
        payload={"message": "hello"},
        stream=AI_JOB_STREAM,
        user_id=2,
    )

    assert stream_id == "2-0"
    assert fake_redis.xadds[0][0] == AI_JOB_STREAM
    assert fake_redis.xadds[0][2] == {"maxlen": STREAM_MAXLEN, "approximate": True}
    assert parse_stream_job(fake_redis.xadds[0][1]).user_id == 2


@pytest.mark.asyncio
async def test_demo_echo_handler_marks_processing_and_success(monkeypatch) -> None:
    calls: list[tuple[str, int, dict | None]] = []

    async def fake_mark_processing(job_id: int):
        calls.append(("processing", job_id, None))

    async def fake_mark_success(job_id: int, result_payload: dict):
        calls.append(("success", job_id, result_payload))

    monkeypatch.setattr(handlers.async_job_service, "mark_processing", fake_mark_processing)
    monkeypatch.setattr(handlers.async_job_service, "mark_success", fake_mark_success)

    await handlers.handle_stream_job(3, "DEMO_ECHO", {"message": "ping"})

    assert calls[0] == ("processing", 3, None)
    assert calls[1] == ("success", 3, {"echo": {"message": "ping"}, "handler": "DEMO_ECHO"})


@pytest.mark.asyncio
async def test_unknown_job_type_is_marked_failed_and_raises_non_retryable(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    async def fake_mark_failed(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(handlers.async_job_service, "mark_failed", fake_mark_failed)

    with pytest.raises(handlers.UnsupportedJobTypeError):
        await handlers.handle_stream_job(9, "UNKNOWN", {})

    assert calls == [(9, "unsupported_job_type: UNKNOWN")]


@pytest.mark.asyncio
async def test_consumer_acks_successful_message(monkeypatch) -> None:
    fake_redis = FakeRedis()
    calls: list[tuple[int, str, dict]] = []

    async def fake_handle_stream_job(job_id: int, job_type: str, payload: dict):
        calls.append((job_id, job_type, payload))

    monkeypatch.setattr(consumer, "handle_stream_job", fake_handle_stream_job)

    fields = build_stream_fields(job_id=10, job_type="DEMO_ECHO", payload={"message": "ok"})
    processed = await consumer.process_stream_message(fake_redis, AI_JOB_STREAM, "1-0", fields)

    assert processed is True
    assert calls == [(10, "DEMO_ECHO", {"message": "ok"})]
    assert fake_redis.xacks == [(AI_JOB_STREAM, ASYNC_JOB_GROUP, "1-0")]


@pytest.mark.asyncio
async def test_consumer_retries_failed_message_before_max_attempts(monkeypatch) -> None:
    fake_redis = FakeRedis()
    calls: list[tuple[int, str]] = []

    async def fake_handle_stream_job(job_id: int, job_type: str, payload: dict):
        raise RuntimeError("boom")

    async def fake_mark_retry_scheduled(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(consumer, "handle_stream_job", fake_handle_stream_job)
    monkeypatch.setattr(consumer.async_job_service, "mark_retry_scheduled", fake_mark_retry_scheduled)

    fields = build_stream_fields(
        job_id=11, job_type="DEMO_ECHO", payload={"message": "fail"}, attempts=0, max_attempts=3
    )
    processed = await consumer.process_stream_message(fake_redis, AI_JOB_STREAM, "1-0", fields)

    assert processed is True
    assert fake_redis.xadds[0][0] == AI_JOB_STREAM
    retry_job = parse_stream_job(fake_redis.xadds[0][1])
    assert retry_job.job_id == 11
    assert retry_job.attempts == 1
    assert retry_job.max_attempts == 3
    assert retry_job.available_at > retry_job.created_at
    assert calls == [(11, "retry_scheduled: handler_failed: RuntimeError")]
    assert fake_redis.xacks == [(AI_JOB_STREAM, ASYNC_JOB_GROUP, "1-0")]


@pytest.mark.asyncio
async def test_consumer_moves_to_dlq_when_max_attempts_exceeded(monkeypatch) -> None:
    fake_redis = FakeRedis()
    calls: list[tuple[int, str]] = []

    async def fake_handle_stream_job(job_id: int, job_type: str, payload: dict):
        raise RuntimeError("boom")

    async def fake_mark_failed(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(consumer, "handle_stream_job", fake_handle_stream_job)
    monkeypatch.setattr(consumer.async_job_service, "mark_failed", fake_mark_failed)

    fields = build_stream_fields(
        job_id=12, job_type="DEMO_ECHO", payload={"message": "fail"}, attempts=2, max_attempts=3
    )
    processed = await consumer.process_stream_message(fake_redis, AI_JOB_STREAM, "1-0", fields)

    assert processed is True
    assert fake_redis.xadds[0][0] == DLQ_JOB_STREAM
    dlq_fields = fake_redis.xadds[0][1]
    assert dlq_fields["error_message"] == "handler_failed: RuntimeError"
    assert dlq_fields["source_stream"] == AI_JOB_STREAM
    assert dlq_fields["source_stream_id"] == "1-0"
    assert calls == [(12, "handler_failed: RuntimeError")]
    assert fake_redis.xacks == [(AI_JOB_STREAM, ASYNC_JOB_GROUP, "1-0")]


@pytest.mark.asyncio
async def test_consumer_moves_unknown_job_type_to_dlq(monkeypatch) -> None:
    fake_redis = FakeRedis()
    calls: list[tuple[int, str]] = []

    async def fake_mark_failed(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(consumer.async_job_service, "mark_failed", fake_mark_failed)

    fields = build_stream_fields(job_id=13, job_type="UNKNOWN", payload={})
    processed = await consumer.process_stream_message(fake_redis, AI_JOB_STREAM, "1-0", fields)

    assert processed is True
    assert fake_redis.xadds[0][0] == DLQ_JOB_STREAM
    assert fake_redis.xadds[0][1]["error_message"] == "unsupported_job_type: UNKNOWN"
    assert calls[-1] == (13, "unsupported_job_type: UNKNOWN")
    assert fake_redis.xacks == [(AI_JOB_STREAM, ASYNC_JOB_GROUP, "1-0")]


@pytest.mark.asyncio
async def test_recover_pending_messages_claims_and_processes(monkeypatch) -> None:
    fields = build_stream_fields(job_id=14, job_type="DEMO_ECHO", payload={"message": "pending"})
    fake_redis = FakeRedis(claimed_messages=[("9-0", fields)])
    calls: list[tuple[str, str, dict]] = []

    async def fake_process_stream_message(redis_client, stream_name: str, stream_id: str, message_fields: dict):
        calls.append((stream_name, stream_id, message_fields))
        return True

    monkeypatch.setattr(consumer, "process_stream_message", fake_process_stream_message)

    recovered = await consumer.recover_pending_messages(
        fake_redis,
        consumer_name="test-consumer",
        streams=(AI_JOB_STREAM,),
        min_idle_ms=1,
        count=5,
    )

    assert recovered == 1
    assert fake_redis.xautoclaim_calls == [(AI_JOB_STREAM, ASYNC_JOB_GROUP, "test-consumer", 1, "0-0", 5)]
    assert calls == [(AI_JOB_STREAM, "9-0", fields)]
