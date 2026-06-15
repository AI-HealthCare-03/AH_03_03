from __future__ import annotations

import urllib.parse
from typing import Any

from ai_runtime.jobs.redis_stream import SERVICE_JOB_STREAM
from app.services import async_jobs as async_job_service
from app.services.email_service import EmailConfigurationError, EmailDeliveryError, EmailService

EMAIL_VERIFICATION_SEND_JOB_TYPE = "email.verification.send"
PASSWORD_RESET_EMAIL_SEND_JOB_TYPE = "password_reset.email.send"
FAMILY_INVITE_EMAIL_SEND_JOB_TYPE = "family.invite.email.send"
FAMILY_NOTIFICATION_CREATE_JOB_TYPE = "family.notification.create"


class ServiceJobNonRetryableError(RuntimeError):
    pass


async def enqueue_email_verification_send(*, email: str, code: str) -> None:
    await _create_service_job(
        job_type=EMAIL_VERIFICATION_SEND_JOB_TYPE,
        request_payload={
            "email": email,
            "code": code,
            "resource_type": "email_verification",
        },
    )


async def enqueue_password_reset_email_send(*, email: str, reset_url: str) -> None:
    await _create_service_job(
        job_type=PASSWORD_RESET_EMAIL_SEND_JOB_TYPE,
        request_payload={
            "email": email,
            "reset_url": reset_url,
            "resource_type": "password_reset",
        },
    )


async def enqueue_family_invite_email_send(
    *,
    recipient_email: str,
    inviter_display_name: str,
    invite_code: str,
    invite_url: str,
    expires_at_text: str,
) -> None:
    await _create_service_job(
        job_type=FAMILY_INVITE_EMAIL_SEND_JOB_TYPE,
        request_payload={
            "recipient_email": recipient_email,
            "inviter_display_name": inviter_display_name,
            "invite_code": invite_code,
            "invite_url": invite_url,
            "expires_at_text": expires_at_text,
            "resource_type": "family_invite_email",
        },
    )


async def enqueue_family_notification_create(*, alert_type: str, user_challenge_id: int) -> None:
    await _create_service_job(
        job_type=FAMILY_NOTIFICATION_CREATE_JOB_TYPE,
        request_payload={
            "alert_type": alert_type,
            "user_challenge_id": user_challenge_id,
            "resource_type": "family_notification",
        },
        resource_id=user_challenge_id,
    )


async def handle_email_verification_send(job_id: int) -> dict[str, Any]:
    payload = await _job_payload(job_id)
    email = _required_str(payload, "email")
    code = _required_str(payload, "code")
    await async_job_service.mark_processing(job_id)
    try:
        sent = await EmailService().send_email_verification_code(email, code)
    except EmailConfigurationError as exc:
        raise ServiceJobNonRetryableError("email_configuration_missing") from exc
    except EmailDeliveryError:
        raise
    result = {"sent": sent, "recipient": email, "kind": "email_verification"}
    await async_job_service.mark_success(job_id, result)
    return result


async def handle_password_reset_email_send(job_id: int) -> dict[str, Any]:
    payload = await _job_payload(job_id)
    email = _required_str(payload, "email")
    reset_url = _required_str(payload, "reset_url")
    await async_job_service.mark_processing(job_id)
    try:
        sent = await EmailService().send_password_reset_email(email, reset_url)
    except EmailConfigurationError as exc:
        raise ServiceJobNonRetryableError("email_configuration_missing") from exc
    except EmailDeliveryError:
        raise
    result = {"sent": sent, "recipient": email, "kind": "password_reset"}
    await async_job_service.mark_success(job_id, result)
    return result


async def handle_family_invite_email_send(job_id: int) -> dict[str, Any]:
    payload = await _job_payload(job_id)
    recipient_email = _required_str(payload, "recipient_email")
    await async_job_service.mark_processing(job_id)
    try:
        sent = await EmailService().send_family_invite_email(
            recipient_email,
            inviter_display_name=_required_str(payload, "inviter_display_name"),
            invite_code=_family_invite_code_from_payload(payload),
            invite_url=_required_str(payload, "invite_url"),
            expires_at_text=_required_str(payload, "expires_at_text"),
        )
    except EmailConfigurationError as exc:
        raise ServiceJobNonRetryableError("email_configuration_missing") from exc
    except EmailDeliveryError:
        raise
    result = {"sent": sent, "recipient": recipient_email, "kind": "family_invite"}
    await async_job_service.mark_success(job_id, result)
    return result


async def handle_family_notification_create(job_id: int) -> dict[str, Any]:
    payload = await _job_payload(job_id)
    alert_type = _required_str(payload, "alert_type")
    user_challenge_id = _required_int(payload, "user_challenge_id")
    await async_job_service.mark_processing(job_id)

    from app.services import family as family_service

    if alert_type == "challenge_completed":
        notifications = await family_service.notify_family_challenge_completed(user_challenge_id)
    elif alert_type == "challenge_missed":
        notifications = await family_service.notify_family_challenge_missed(user_challenge_id)
    else:
        raise ServiceJobNonRetryableError("unsupported_family_alert_type")

    result = {"created_count": len(notifications), "alert_type": alert_type}
    await async_job_service.mark_success(job_id, result)
    return result


async def _create_service_job(
    *,
    job_type: str,
    request_payload: dict[str, Any],
    user_id: int | None = None,
    resource_id: int | None = None,
) -> None:
    await async_job_service.create_async_job(
        job_type=job_type,
        request_payload=request_payload,
        stream=SERVICE_JOB_STREAM,
        user_id=user_id,
        resource_id=resource_id,
        idempotency_key=request_payload.get("idempotency_key"),
        stream_payload={"resource_type": request_payload.get("resource_type", "service_job")},
    )


async def _job_payload(job_id: int) -> dict[str, Any]:
    job = await async_job_service.get_job(job_id)
    if job is None:
        raise ServiceJobNonRetryableError("service_job_not_found")
    payload = job.request_payload or {}
    if not isinstance(payload, dict):
        raise ServiceJobNonRetryableError("service_job_payload_invalid")
    return payload


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value in (None, ""):
        raise ServiceJobNonRetryableError(f"missing_{key}")
    return str(value)


def _family_invite_code_from_payload(payload: dict[str, Any]) -> str:
    value = payload.get("invite_code")
    if value not in (None, ""):
        return str(value)

    invite_url = _required_str(payload, "invite_url")
    parsed_url = urllib.parse.urlparse(invite_url)
    parsed_query = urllib.parse.parse_qs(parsed_url.query)
    codes = parsed_query.get("invite_code") or parsed_query.get("code")
    if codes and codes[0]:
        return codes[0]
    raise ServiceJobNonRetryableError("missing_invite_code")


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ServiceJobNonRetryableError(f"missing_{key}") from exc
