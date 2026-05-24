from __future__ import annotations

import pytest

from ai_worker.jobs import consumer, handlers
from ai_worker.jobs.redis_stream import build_stream_fields, parse_stream_fields


def test_stream_fields_round_trip_preserves_demo_payload() -> None:
    fields = build_stream_fields(
        job_id=7,
        job_type="DEMO_ECHO",
        payload={"message": "hello", "payload": {"source": "test"}},
    )

    job_id, job_type, payload = parse_stream_fields(fields)

    assert job_id == 7
    assert job_type == "DEMO_ECHO"
    assert payload == {"message": "hello", "payload": {"source": "test"}}


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
async def test_unknown_job_type_is_marked_failed(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    async def fake_mark_failed(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(handlers.async_job_service, "mark_failed", fake_mark_failed)

    await handlers.handle_stream_job(9, "UNKNOWN", {})

    assert calls == [(9, "unsupported_job_type: UNKNOWN")]


@pytest.mark.asyncio
async def test_consumer_marks_job_failed_when_handler_raises(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    async def fake_handle_stream_job(job_id: int, job_type: str, payload: dict):
        raise RuntimeError("boom")

    async def fake_mark_failed(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(consumer, "handle_stream_job", fake_handle_stream_job)
    monkeypatch.setattr(consumer.async_job_service, "mark_failed", fake_mark_failed)

    fields = build_stream_fields(job_id=11, job_type="DEMO_ECHO", payload={"message": "fail"})
    await consumer.process_stream_message("1-0", fields)

    assert calls == [(11, "handler_failed: RuntimeError")]
