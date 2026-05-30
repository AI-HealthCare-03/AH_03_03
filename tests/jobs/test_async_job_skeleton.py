from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

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
from app.services import async_jobs as async_job_service


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
async def test_create_medication_ocr_job_enqueues_slim_stream_payload(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeAsyncJob:
        id = 31
        job_type = async_job_service.MEDICATION_OCR_JOB_TYPE
        stream_id = None

        async def save(self, update_fields):
            _ = update_fields
            calls.append(("save", {"stream_id": self.stream_id}))

    async def fake_create(**kwargs):
        calls.append(("create", kwargs))
        return FakeAsyncJob()

    async def fake_enqueue_async_job(**kwargs):
        calls.append(("enqueue", kwargs))
        return "31-0"

    monkeypatch.setattr(async_job_service.AsyncJob, "create", fake_create)
    monkeypatch.setattr(async_job_service, "enqueue_async_job", fake_enqueue_async_job)

    job = await async_job_service.create_medication_ocr_job(
        9,
        {
            "source_type": "PRESCRIPTION",
            "upload_path": "/app/var/uploads/medication_ocr/9/source.jpg",
            "image_media_type": "image/jpeg",
        },
    )

    assert job.stream_id == "31-0"
    assert calls[0][0] == "create"
    assert calls[0][1]["request_payload"]["user_id"] == 9
    assert calls[0][1]["request_payload"]["upload_path"].endswith("source.jpg")
    assert calls[1][0] == "enqueue"
    assert calls[1][1]["job_type"] == "medication_ocr.run"
    assert calls[1][1]["payload"] == {"resource_type": "medication_ocr_request"}
    assert calls[1][1]["user_id"] == 9


@pytest.mark.asyncio
async def test_create_diet_analyze_image_job_enqueues_slim_stream_payload(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeAsyncJob:
        id = 32
        job_type = async_job_service.DIET_ANALYZE_IMAGE_JOB_TYPE
        stream_id = None

        async def save(self, update_fields):
            _ = update_fields
            calls.append(("save", {"stream_id": self.stream_id}))

    async def fake_create(**kwargs):
        calls.append(("create", kwargs))
        return FakeAsyncJob()

    async def fake_enqueue_async_job(**kwargs):
        calls.append(("enqueue", kwargs))
        return "32-0"

    monkeypatch.setattr(async_job_service.AsyncJob, "create", fake_create)
    monkeypatch.setattr(async_job_service, "enqueue_async_job", fake_enqueue_async_job)

    job = await async_job_service.create_diet_analyze_image_job(
        10,
        {
            "description": "사진 식단",
            "upload_path": "/app/var/uploads/diet_analysis/10/source.jpg",
            "image_media_type": "image/jpeg",
        },
    )

    assert job.stream_id == "32-0"
    assert calls[0][0] == "create"
    assert calls[0][1]["request_payload"]["user_id"] == 10
    assert calls[0][1]["request_payload"]["upload_path"].endswith("source.jpg")
    assert calls[1][0] == "enqueue"
    assert calls[1][1]["job_type"] == "diet.analyze_image"
    assert calls[1][1]["payload"] == {"resource_type": "diet_analysis_request"}
    assert calls[1][1]["user_id"] == 10


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
async def test_exam_ocr_handler_runs_report_ocr_and_marks_success(monkeypatch) -> None:
    from app.services import exams as exam_service

    calls: list[tuple[str, int, dict | None]] = []

    async def fake_mark_processing(job_id: int):
        calls.append(("processing", job_id, None))

    async def fake_run_exam_ocr_from_report(exam_report_id: int):
        calls.append(("ocr", exam_report_id, None))
        return SimpleNamespace(
            measurements=[SimpleNamespace(id=1), SimpleNamespace(id=2)],
            ocr_provider="paddleocr",
            fallback_used=False,
            provider_message="paddleocr_checkup_ocr",
        )

    async def fake_mark_success(job_id: int, result_payload: dict):
        calls.append(("success", job_id, result_payload))

    monkeypatch.setattr(handlers.async_job_service, "mark_processing", fake_mark_processing)
    monkeypatch.setattr(exam_service, "run_exam_ocr_from_report", fake_run_exam_ocr_from_report)
    monkeypatch.setattr(handlers.async_job_service, "mark_success", fake_mark_success)

    await handlers.handle_stream_job(15, "exam_ocr.run", {"exam_report_id": 33})

    assert calls == [
        ("processing", 15, None),
        ("ocr", 33, None),
        (
            "success",
            15,
            {
                "exam_report_id": 33,
                "measurement_count": 2,
                "ocr_provider": "paddleocr",
                "fallback_used": False,
                "provider_message": "paddleocr_checkup_ocr",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_exam_ocr_handler_rejects_missing_exam_report_id(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    async def fake_mark_failed(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(handlers.async_job_service, "mark_failed", fake_mark_failed)

    with pytest.raises(handlers.NonRetryableJobError):
        await handlers.handle_stream_job(16, "exam_ocr.run", {})

    assert calls == [(16, "missing_exam_report_id")]


@pytest.mark.asyncio
async def test_medication_ocr_handler_runs_job_ocr_and_marks_success(monkeypatch) -> None:
    from app.dtos.medications import MedicationOCRItem, MedicationOCRResponse
    from app.services import medications as medication_service

    calls: list[tuple[str, int, dict | None]] = []

    async def fake_mark_processing(job_id: int):
        calls.append(("processing", job_id, None))

    async def fake_run_medication_ocr_from_job(job_id: int):
        calls.append(("ocr", job_id, None))
        return MedicationOCRResponse(
            source_type="PRESCRIPTION",
            ocr_confidence=0.9,
            items=[MedicationOCRItem(name="테스트약", time_slots=[])],
            message="ok",
            source="paddleocr_medication_ocr",
            fallback_used=False,
        )

    async def fake_mark_success(job_id: int, result_payload: dict):
        calls.append(("success", job_id, result_payload))

    monkeypatch.setattr(handlers.async_job_service, "mark_processing", fake_mark_processing)
    monkeypatch.setattr(medication_service, "run_medication_ocr_from_job", fake_run_medication_ocr_from_job)
    monkeypatch.setattr(handlers.async_job_service, "mark_success", fake_mark_success)

    await handlers.handle_stream_job(21, "medication_ocr.run", {})

    assert calls[0] == ("processing", 21, None)
    assert calls[1] == ("ocr", 21, None)
    assert calls[2][0:2] == ("success", 21)
    assert calls[2][2]["items"][0]["name"] == "테스트약"
    assert calls[2][2]["source"] == "paddleocr_medication_ocr"


@pytest.mark.asyncio
async def test_medication_ocr_handler_rejects_missing_source(monkeypatch) -> None:
    from app.services import medications as medication_service

    calls: list[tuple[int, str] | tuple[str, int]] = []

    async def fake_mark_processing(job_id: int):
        calls.append(("processing", job_id))

    async def fake_run_medication_ocr_from_job(job_id: int):
        _ = job_id
        raise ValueError("medication_ocr_source_missing")

    async def fake_mark_failed(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(handlers.async_job_service, "mark_processing", fake_mark_processing)
    monkeypatch.setattr(medication_service, "run_medication_ocr_from_job", fake_run_medication_ocr_from_job)
    monkeypatch.setattr(handlers.async_job_service, "mark_failed", fake_mark_failed)

    with pytest.raises(handlers.NonRetryableJobError):
        await handlers.handle_stream_job(22, "medication_ocr.run", {})

    assert calls == [("processing", 22), (22, "medication_ocr_source_missing")]


@pytest.mark.asyncio
async def test_diet_analyze_image_handler_runs_job_analysis_and_marks_success(monkeypatch) -> None:
    from app.dtos.diets import (
        DietAnalyzePhotoResultResponse,
        DietAnalyzeResponse,
        DietRecordResponse,
    )
    from app.services import diets as diet_service

    now = datetime.now(UTC)
    calls: list[tuple[str, int, dict | None]] = []

    async def fake_mark_processing(job_id: int):
        calls.append(("processing", job_id, None))

    async def fake_run_diet_analysis_from_job(job_id: int):
        calls.append(("analysis", job_id, None))
        return DietAnalyzeResponse(
            message="식단 분석이 완료되었습니다.",
            diet_record=DietRecordResponse(
                id=1,
                user_id=10,
                meal_type="LUNCH",
                meal_time=now,
                description="사진 식단",
                image_path=None,
                detected_foods=[{"name": "현미밥"}],
                nutrition_summary={"calories": 620},
                diet_score=82.5,
                diet_feedback="ok",
                analysis_method="IMAGE_ANALYSIS",
                is_user_corrected=False,
                memo=None,
                created_at=now,
                updated_at=now,
            ),
            photo_result=DietAnalyzePhotoResultResponse(
                id=2,
                diet_record_id=1,
                detected_foods=[{"name": "현미밥"}],
                confidence_payload={"method": "rule_based_food_detection"},
                raw_output={"source": "rule_based_food_detection"},
                created_at=now,
            ),
            detected_foods=[{"name": "현미밥"}],
            nutrition_summary={"calories": 620},
            diet_score=82.5,
            diet_feedback="ok",
            vision_provider="rule_based_food_detection",
            fallback_used=True,
        )

    async def fake_mark_success(job_id: int, result_payload: dict):
        calls.append(("success", job_id, result_payload))

    monkeypatch.setattr(handlers.async_job_service, "mark_processing", fake_mark_processing)
    monkeypatch.setattr(diet_service, "run_diet_analysis_from_job", fake_run_diet_analysis_from_job)
    monkeypatch.setattr(handlers.async_job_service, "mark_success", fake_mark_success)

    await handlers.handle_stream_job(23, "diet.analyze_image", {})

    assert calls[0] == ("processing", 23, None)
    assert calls[1] == ("analysis", 23, None)
    assert calls[2][0:2] == ("success", 23)
    assert calls[2][2]["diet_record"]["id"] == 1
    assert calls[2][2]["vision_provider"] == "rule_based_food_detection"


@pytest.mark.asyncio
async def test_diet_analyze_image_handler_rejects_missing_user(monkeypatch) -> None:
    from app.services import diets as diet_service

    calls: list[tuple[int, str] | tuple[str, int]] = []

    async def fake_mark_processing(job_id: int):
        calls.append(("processing", job_id))

    async def fake_run_diet_analysis_from_job(job_id: int):
        _ = job_id
        raise ValueError("diet_analysis_user_id_missing")

    async def fake_mark_failed(job_id: int, error_message: str):
        calls.append((job_id, error_message))

    monkeypatch.setattr(handlers.async_job_service, "mark_processing", fake_mark_processing)
    monkeypatch.setattr(diet_service, "run_diet_analysis_from_job", fake_run_diet_analysis_from_job)
    monkeypatch.setattr(handlers.async_job_service, "mark_failed", fake_mark_failed)

    with pytest.raises(handlers.NonRetryableJobError):
        await handlers.handle_stream_job(24, "diet.analyze_image", {})

    assert calls == [("processing", 24), (24, "diet_analysis_user_id_missing")]


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
