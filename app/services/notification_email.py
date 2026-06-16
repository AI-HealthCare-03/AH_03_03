from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.core import config
from app.models.notifications import NotificationLogStatus
from app.models.users import User
from app.services.email_service import EmailConfigurationError, EmailDeliveryError, EmailService


@dataclass(frozen=True)
class NotificationEmailDeliveryResult:
    status: NotificationLogStatus
    sent: bool
    provider: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    sent_at: datetime | None = None
    failed_at: datetime | None = None


async def deliver_notification_email_to_user(
    *,
    user_id: int,
    title: str,
    message: str,
    action_url: str | None = None,
    email_service: EmailService | None = None,
) -> NotificationEmailDeliveryResult:
    user = await User.get_or_none(id=user_id)
    recipient_email = (getattr(user, "email", None) or "").strip() if user is not None else ""
    if not recipient_email:
        return NotificationEmailDeliveryResult(
            status=NotificationLogStatus.SKIPPED,
            sent=False,
            error_code="recipient_email_missing",
            error_message="recipient_email_missing",
        )

    service = email_service or EmailService()
    try:
        sent = await service.send_notification_email(
            recipient_email,
            title=title,
            message=message,
            action_url=action_url,
        )
    except EmailConfigurationError:
        return NotificationEmailDeliveryResult(
            status=NotificationLogStatus.SKIPPED,
            sent=False,
            error_code="email_configuration_missing",
            error_message="email_configuration_missing",
        )
    except EmailDeliveryError:
        return NotificationEmailDeliveryResult(
            status=NotificationLogStatus.FAILED,
            sent=False,
            provider="smtp",
            error_code="email_delivery_failed",
            error_message="email_delivery_failed",
            failed_at=datetime.now(config.TIMEZONE),
        )

    if not sent:
        return NotificationEmailDeliveryResult(
            status=NotificationLogStatus.SKIPPED,
            sent=False,
            error_code="email_delivery_disabled",
            error_message="email_delivery_disabled",
        )

    return NotificationEmailDeliveryResult(
        status=NotificationLogStatus.SENT,
        sent=True,
        provider="smtp",
        sent_at=datetime.now(config.TIMEZONE),
    )
