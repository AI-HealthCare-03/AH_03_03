from __future__ import annotations

import urllib.parse
from typing import Any

from ai_runtime.jobs.redis_stream import SERVICE_JOB_STREAM
from app.models.notifications import NotificationChannel, NotificationLogStatus, UserFCMToken
from app.services import async_jobs as async_job_service
from app.services import notifications as notification_service
from app.services.email_service import EmailConfigurationError, EmailDeliveryError, EmailService
from app.services.fcm import FCMProviderUnavailableError, FCMService

EMAIL_VERIFICATION_SEND_JOB_TYPE = "email.verification.send"
PASSWORD_RESET_EMAIL_SEND_JOB_TYPE = "password_reset.email.send"
FAMILY_INVITE_EMAIL_SEND_JOB_TYPE = "family.invite.email.send"
FCM_PUSH_SEND_JOB_TYPE = "fcm.push.send"
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


async def enqueue_fcm_push_send(
    *,
    user_id: int,
    title: str,
    body: str,
    data: dict[str, str] | None = None,
    notification_type: str = "PUSH",
    related_type: str | None = None,
    related_id: int | None = None,
) -> None:
    await _create_service_job(
        job_type=FCM_PUSH_SEND_JOB_TYPE,
        request_payload={
            "user_id": user_id,
            "title": title,
            "body": body,
            "data": data or {},
            "notification_type": notification_type,
            "related_type": related_type,
            "related_id": related_id,
            "resource_type": "fcm_push",
        },
        user_id=user_id,
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


async def handle_fcm_push_send(job_id: int) -> dict[str, Any]:
    payload = await _job_payload(job_id)
    user_id = _required_int(payload, "user_id")
    title = _required_str(payload, "title")
    body = _required_str(payload, "body")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    await async_job_service.mark_processing(job_id)

    tokens = await _list_active_fcm_tokens(user_id)
    if not tokens:
        result = {"sent": False, "success_count": 0, "failure_count": 0, "skipped": "no_active_tokens"}
        await async_job_service.mark_success(job_id, result)
        return result

    try:
        send_result = FCMService().send_to_tokens(tokens=tokens, title=title, body=body, data=data)
    except FCMProviderUnavailableError as exc:
        await _record_fcm_push_log(payload, status_value=NotificationLogStatus.FAILED, error_code=type(exc).__name__)
        raise ServiceJobNonRetryableError("fcm_provider_unavailable") from exc
    except Exception as exc:
        await _record_fcm_push_log(
            payload,
            status_value=NotificationLogStatus.FAILED,
            error_code=type(exc).__name__,
            error_message=str(exc),
        )
        raise

    await _record_fcm_push_log(
        payload,
        status_value=NotificationLogStatus.SENT if send_result.success_count > 0 else NotificationLogStatus.FAILED,
        provider_message_id=send_result.provider_message_ids[0] if send_result.provider_message_ids else None,
        error_code=None if send_result.failure_count == 0 else "PARTIAL_FAILURE",
        error_message=None if send_result.failure_count == 0 else f"failure_count={send_result.failure_count}",
    )
    result = {
        "sent": send_result.success_count > 0,
        "success_count": send_result.success_count,
        "failure_count": send_result.failure_count,
    }
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
    codes = parsed_query.get("code")
    if codes and codes[0]:
        return codes[0]
    raise ServiceJobNonRetryableError("missing_invite_code")


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ServiceJobNonRetryableError(f"missing_{key}") from exc


async def _list_active_fcm_tokens(user_id: int) -> list[str]:
    return list(await UserFCMToken.filter(user_id=user_id, is_active=True).values_list("token", flat=True))


async def _record_fcm_push_log(
    payload: dict[str, Any],
    *,
    status_value: NotificationLogStatus,
    provider_message_id: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    user_id = _required_int(payload, "user_id")
    await notification_service.record_notification_log(
        user_id=user_id,
        notification_type=str(payload.get("notification_type") or "PUSH"),
        channel=NotificationChannel.PUSH,
        title=_required_str(payload, "title"),
        status=status_value,
        message_summary=_required_str(payload, "body"),
        related_type=str(payload.get("related_type") or "") or None,
        related_id=_optional_int(payload.get("related_id")),
        provider="firebase_fcm",
        provider_message_id=provider_message_id,
        error_code=error_code,
        error_message=error_message[:255] if error_message else None,
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
