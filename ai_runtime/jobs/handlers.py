from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.services import async_jobs as async_job_service

JobHandler = Callable[[int, dict[str, Any]], Awaitable[None]]
JOB_HANDLERS: dict[str, JobHandler] = {}


class NonRetryableJobError(RuntimeError):
    """Base error for jobs that should be sent to DLQ without another retry."""


class UnsupportedJobTypeError(NonRetryableJobError):
    def __init__(self, job_type: str) -> None:
        super().__init__(f"unsupported_job_type: {job_type}")
        self.job_type = job_type


def register_job_handler(job_type: str) -> Callable[[JobHandler], JobHandler]:
    def decorator(handler: JobHandler) -> JobHandler:
        JOB_HANDLERS[job_type] = handler
        return handler

    return decorator


@register_job_handler(async_job_service.DEMO_ECHO_JOB_TYPE)
async def handle_demo_echo(job_id: int, payload: dict[str, Any]) -> None:
    await async_job_service.mark_processing(job_id)
    await async_job_service.mark_success(
        job_id,
        {
            "echo": payload,
            "handler": "DEMO_ECHO",
        },
    )


@register_job_handler(async_job_service.EXAM_OCR_JOB_TYPE)
async def handle_exam_ocr(job_id: int, payload: dict[str, Any]) -> None:
    exam_report_id = _payload_int(payload, "exam_report_id") or _payload_int(payload, "resource_id")
    if exam_report_id is None:
        await async_job_service.mark_failed(job_id, "missing_exam_report_id")
        raise NonRetryableJobError("missing_exam_report_id")

    await async_job_service.mark_processing(job_id)
    from ai_runtime.jobs import exam_ocr_handler

    response = await exam_ocr_handler.run_exam_ocr_from_report(exam_report_id)
    if not response.measurements:
        error_message = _exam_ocr_failure_message(response.provider_message)
        await async_job_service.mark_failed(job_id, error_message)
        raise NonRetryableJobError(error_message)

    await async_job_service.mark_success(
        job_id,
        {
            "exam_report_id": exam_report_id,
            "measurement_count": len(response.measurements),
            "ocr_provider": response.ocr_provider,
            "fallback_used": response.fallback_used,
            "provider_message": response.provider_message,
        },
    )


def _exam_ocr_failure_message(provider_message: str | None) -> str:
    provider_reason = provider_message or "no_measurements"
    if any(token in provider_reason for token in ("disabled", "unavailable", "missing")):
        return f"exam_ocr_service_unavailable:{provider_reason}"
    return f"exam_ocr_no_measurements:{provider_reason}"


@register_job_handler(async_job_service.DIET_ANALYZE_IMAGE_JOB_TYPE)
async def handle_diet_analyze_image(job_id: int, payload: dict[str, Any]) -> None:
    _ = payload
    await async_job_service.mark_processing(job_id)
    from app.services import diets as diet_service

    try:
        response = await diet_service.run_diet_analysis_from_job(job_id)
    except ValueError as exc:
        error_message = str(exc)
        if error_message in {
            "diet_analysis_job_not_found",
            "diet_analysis_user_id_missing",
            "diet_analysis_upload_missing",
            "diet_analysis_service_unavailable",
        }:
            await async_job_service.mark_failed(job_id, error_message)
            raise NonRetryableJobError(error_message) from exc
        raise

    await async_job_service.mark_success(
        job_id,
        response.model_dump(mode="json"),
    )


@register_job_handler(async_job_service.ANALYSIS_RUN_JOB_TYPE)
async def handle_analysis_run(job_id: int, payload: dict[str, Any]) -> None:
    user_id = _payload_int(payload, "user_id")
    health_record_id = _payload_int(payload, "health_record_id") or _payload_int(payload, "resource_id")
    mode_value = payload.get("mode")
    if user_id is None:
        await async_job_service.mark_failed(job_id, "missing_user_id")
        raise NonRetryableJobError("missing_user_id")
    if health_record_id is None:
        await async_job_service.mark_failed(job_id, "missing_health_record_id")
        raise NonRetryableJobError("missing_health_record_id")
    if mode_value in (None, ""):
        await async_job_service.mark_failed(job_id, "missing_analysis_mode")
        raise NonRetryableJobError("missing_analysis_mode")

    from app.models.analysis import AnalysisMode
    from app.models.users import User
    from app.services import analysis as analysis_service
    from app.services import health as health_service

    try:
        mode = AnalysisMode(str(mode_value))
    except ValueError as exc:
        await async_job_service.mark_failed(job_id, f"invalid_analysis_mode:{mode_value}")
        raise NonRetryableJobError(f"invalid_analysis_mode:{mode_value}") from exc

    health_record = await health_service.get_health_record(health_record_id)
    if health_record is None:
        await async_job_service.mark_failed(job_id, "health_record_not_found")
        raise NonRetryableJobError("health_record_not_found")
    if int(health_record.user_id) != user_id:
        await async_job_service.mark_failed(job_id, "health_record_owner_mismatch")
        raise NonRetryableJobError("health_record_owner_mismatch")

    user = await User.get_or_none(id=user_id)
    if user is None:
        await async_job_service.mark_failed(job_id, "user_not_found")
        raise NonRetryableJobError("user_not_found")

    missing_fields = await analysis_service.get_missing_fields_for_mode(user, health_record, mode)
    if missing_fields:
        await async_job_service.mark_failed(job_id, "analysis_required_fields_missing")
        raise NonRetryableJobError("analysis_required_fields_missing")

    await async_job_service.mark_processing(job_id)
    results = await analysis_service.run_analysis(user_id, health_record, mode)
    await async_job_service.mark_success(
        job_id,
        {
            "user_id": user_id,
            "health_record_id": health_record_id,
            "mode": mode.value,
            "analysis_result_ids": [int(result["analysis_result_id"]) for result in results],
            "result_count": len(results),
        },
    )


def _register_service_job_handler(job_type: str, handler: Callable[[int], Awaitable[dict[str, Any]]]) -> None:
    @register_job_handler(job_type)
    async def _handle_service_job(job_id: int, payload: dict[str, Any]) -> None:
        _ = payload
        from app.services import service_jobs as service_job_service

        try:
            await handler(job_id)
        except service_job_service.ServiceJobNonRetryableError as exc:
            await async_job_service.mark_failed(job_id, str(exc))
            raise NonRetryableJobError(str(exc)) from exc


def _register_service_job_handlers() -> None:
    from app.services import service_jobs as service_job_service

    _register_service_job_handler(
        service_job_service.EMAIL_VERIFICATION_SEND_JOB_TYPE,
        service_job_service.handle_email_verification_send,
    )
    _register_service_job_handler(
        service_job_service.PASSWORD_RESET_EMAIL_SEND_JOB_TYPE,
        service_job_service.handle_password_reset_email_send,
    )
    _register_service_job_handler(
        service_job_service.FAMILY_INVITE_EMAIL_SEND_JOB_TYPE,
        service_job_service.handle_family_invite_email_send,
    )
    _register_service_job_handler(
        service_job_service.NOTIFICATION_EMAIL_SEND_JOB_TYPE,
        service_job_service.handle_notification_email_send,
    )
    _register_service_job_handler(
        service_job_service.FAMILY_NOTIFICATION_CREATE_JOB_TYPE,
        service_job_service.handle_family_notification_create,
    )


_register_service_job_handlers()


def _payload_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def handle_stream_job(job_id: int, job_type: str, payload: dict[str, Any]) -> None:
    handler = JOB_HANDLERS.get(job_type)
    if handler is None:
        await async_job_service.mark_failed(job_id, f"unsupported_job_type: {job_type}")
        raise UnsupportedJobTypeError(job_type)
    await handler(job_id, payload)
