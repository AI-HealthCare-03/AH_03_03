import logging
import zoneinfo
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any

from app.core import config
from app.dtos.notifications import (
    NotificationCreateRequest,
    NotificationUpdateRequest,
    ReminderScheduleCreateRequest,
    ReminderScheduleUpdateRequest,
)
from app.models.notifications import (
    Notification,
    NotificationChannel,
    NotificationLog,
    NotificationLogStatus,
    ReminderSchedule,
    ReminderType,
)
from app.repositories import notification_repository
from app.services.notification_email import NotificationEmailDeliveryResult, deliver_notification_email_to_user

logger = logging.getLogger(__name__)

DEFAULT_REMINDER_TIMEZONE = "Asia/Seoul"
MEDICATION_REMINDER_RELATED_TYPE = "medication"
CHALLENGE_REMINDER_RELATED_TYPE = "challenge_daily_reminder"
DIET_REMINDER_RELATED_TYPE = "diet_record_reminder"
MEDICATION_REMINDER_TITLE = "오늘 복약 시간이에요"
MEDICATION_REMINDER_MESSAGE = "복용 예정인 약·영양제가 있습니다. 복용 후 기록을 남기면 복약 패턴을 확인할 수 있어요."
CHALLENGE_REMINDER_TITLE = "오늘의 챌린지를 확인해 주세요"
CHALLENGE_REMINDER_MESSAGE = "작은 실천도 건강 습관을 만드는 데 도움이 됩니다. 오늘 수행 여부를 기록해 주세요."
DIET_REMINDER_TITLE = "오늘 식단을 기록해 보세요"
DIET_REMINDER_MESSAGE = "사진 한 장이나 간단한 메모로 식단을 남기면 식습관을 돌아보는 데 도움이 됩니다."


async def create_notification(user_id: int, request: NotificationCreateRequest) -> Notification:
    notification_data = request.model_dump(exclude={"send_email", "action_url"})
    notification = await notification_repository.create_notification(user_id, notification_data)
    if request.send_email:
        await _enqueue_notification_email_delivery(
            user_id=user_id,
            notification=notification,
            action_url=request.action_url,
        )
    return notification


async def get_notification(notification_id: int) -> Notification | None:
    return await notification_repository.get_notification_by_id(notification_id)


async def list_notifications(
    user_id: int, is_read: bool | None = None, limit: int = 20, offset: int = 0
) -> list[Notification]:
    return await notification_repository.list_notifications_by_user(
        user_id=user_id,
        is_read=is_read,
        limit=limit,
        offset=offset,
    )


async def list_unread_notifications(user_id: int, limit: int = 20, offset: int = 0) -> list[Notification]:
    return await list_notifications(user_id=user_id, is_read=False, limit=limit, offset=offset)


async def mark_notification_as_read(notification_id: int) -> Notification | None:
    data = {"is_read": True, "read_at": datetime.now(config.TIMEZONE)}
    return await notification_repository.update_notification(notification_id, data)


async def mark_all_notifications_as_read(user_id: int) -> list[Notification]:
    notifications = await notification_repository.list_notifications_by_user(user_id=user_id, is_read=False, limit=1000)
    updated = []
    for notification in notifications:
        marked = await mark_notification_as_read(notification.id)
        if marked is not None:
            updated.append(marked)
    return updated


async def update_notification(notification_id: int, request: NotificationUpdateRequest) -> Notification | None:
    return await notification_repository.update_notification(notification_id, request.model_dump(exclude_unset=True))


async def delete_notification(notification_id: int) -> int:
    return await notification_repository.delete_notification(notification_id)


async def list_my_reminder_schedules(
    user_id: int,
    is_active: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ReminderSchedule]:
    return await notification_repository.list_reminder_schedules_by_user(
        user_id=user_id,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )


async def get_reminder_schedule(schedule_id: int) -> ReminderSchedule | None:
    return await notification_repository.get_reminder_schedule_by_id(schedule_id)


async def create_reminder_schedule(user_id: int, request: ReminderScheduleCreateRequest) -> ReminderSchedule:
    return await notification_repository.create_reminder_schedule(user_id, request.model_dump())


async def update_reminder_schedule(
    user_id: int,
    schedule_id: int,
    request: ReminderScheduleUpdateRequest,
) -> ReminderSchedule | None:
    schedule = await notification_repository.get_reminder_schedule_by_id(schedule_id)
    if schedule is None or schedule.user_id != user_id:
        return None
    return await notification_repository.update_reminder_schedule(
        schedule_id,
        request.model_dump(exclude_unset=True),
    )


async def delete_reminder_schedule(user_id: int, schedule_id: int) -> int:
    schedule = await notification_repository.get_reminder_schedule_by_id(schedule_id)
    if schedule is None or schedule.user_id != user_id:
        return 0
    return await notification_repository.delete_reminder_schedule(schedule_id)


async def sync_all_user_reminder_schedules(user_id: int) -> None:
    await sync_medication_reminder_schedules_for_user(user_id)
    await sync_challenge_reminder_schedule_for_user(user_id)
    await sync_diet_reminder_schedule_for_user(user_id)


async def sync_medication_reminder_schedules_for_user(user_id: int) -> None:
    settings = await _get_user_notification_settings(user_id)
    enabled = bool(settings.notification_enabled and settings.medication_reminder_enabled)
    if not enabled:
        await _deactivate_email_reminder_schedules_by_type(
            user_id=user_id,
            reminder_type=ReminderType.MEDICATION,
            related_type=MEDICATION_REMINDER_RELATED_TYPE,
        )
        return

    from app.services import medications as medication_service

    medications = await medication_service.list_medications(user_id=user_id, is_active=True, limit=1000)
    for medication in medications:
        await sync_medication_reminder_schedule(
            user_id=user_id,
            medication_id=int(medication.id),
            reminder_time=medication.reminder_time,
            medication_is_active=bool(medication.is_active),
            settings_enabled=enabled,
        )


async def sync_medication_reminder_schedule(
    *,
    user_id: int,
    medication_id: int,
    reminder_time: time | str | None,
    medication_is_active: bool = True,
    settings_enabled: bool | None = None,
) -> ReminderSchedule | None:
    if settings_enabled is None:
        settings = await _get_user_notification_settings(user_id)
        settings_enabled = bool(settings.notification_enabled and settings.medication_reminder_enabled)
    enabled = bool(settings_enabled and medication_is_active and reminder_time)
    return await _upsert_email_reminder_schedule(
        user_id=user_id,
        reminder_type=ReminderType.MEDICATION,
        related_type=MEDICATION_REMINDER_RELATED_TYPE,
        related_id=medication_id,
        title=MEDICATION_REMINDER_TITLE,
        message=MEDICATION_REMINDER_MESSAGE,
        schedule_time=reminder_time,
        enabled=enabled,
    )


async def deactivate_medication_reminder_schedule(user_id: int, medication_id: int) -> ReminderSchedule | None:
    return await _upsert_email_reminder_schedule(
        user_id=user_id,
        reminder_type=ReminderType.MEDICATION,
        related_type=MEDICATION_REMINDER_RELATED_TYPE,
        related_id=medication_id,
        title=MEDICATION_REMINDER_TITLE,
        message=MEDICATION_REMINDER_MESSAGE,
        schedule_time=None,
        enabled=False,
    )


async def sync_challenge_reminder_schedule_for_user(user_id: int) -> ReminderSchedule | None:
    settings = await _get_user_notification_settings(user_id)
    active_challenge_count = await _count_active_user_challenges_for_reminder(user_id)
    schedule_time = settings.challenge_reminder_time or time(21, 0)
    enabled = bool(settings.notification_enabled and settings.challenge_reminder_enabled and active_challenge_count > 0)
    return await _upsert_email_reminder_schedule(
        user_id=user_id,
        reminder_type=ReminderType.CHALLENGE,
        related_type=CHALLENGE_REMINDER_RELATED_TYPE,
        related_id=None,
        title=CHALLENGE_REMINDER_TITLE,
        message=CHALLENGE_REMINDER_MESSAGE,
        schedule_time=schedule_time,
        enabled=enabled,
    )


async def sync_diet_reminder_schedule_for_user(user_id: int) -> ReminderSchedule | None:
    settings = await _get_user_notification_settings(user_id)
    enabled = bool(settings.notification_enabled and settings.diet_reminder_enabled)
    schedule_time = settings.diet_reminder_time or time(20, 0)
    return await _upsert_email_reminder_schedule(
        user_id=user_id,
        reminder_type=ReminderType.SYSTEM,
        related_type=DIET_REMINDER_RELATED_TYPE,
        related_id=None,
        title=DIET_REMINDER_TITLE,
        message=DIET_REMINDER_MESSAGE,
        schedule_time=schedule_time,
        enabled=enabled,
    )


async def list_my_notification_logs(user_id: int, limit: int = 50, offset: int = 0) -> list[NotificationLog]:
    return await notification_repository.list_notification_logs_by_user(user_id=user_id, limit=limit, offset=offset)


async def process_due_reminder_schedules(now: datetime | None = None, limit: int = 100) -> int:
    now = _aware_datetime(now or datetime.now(config.TIMEZONE))
    schedules = await notification_repository.list_due_reminder_schedules(now, limit=limit)
    created_count = 0
    for schedule in schedules:
        try:
            if await _process_due_reminder_schedule(schedule, now):
                created_count += 1
        except Exception:
            logger.exception("Failed to process reminder schedule", extra={"reminder_schedule_id": schedule.id})
    return created_count


async def _process_due_reminder_schedule(schedule: ReminderSchedule, now: datetime) -> bool:
    trigger_at = _aware_datetime(schedule.next_trigger_at or now)
    if await notification_repository.has_notification_log_for_reminder_since(int(schedule.id), trigger_at):
        await _advance_reminder_schedule(schedule, now)
        return False

    notification = await notification_repository.create_notification(
        int(schedule.user_id),
        {
            "notification_type": _enum_value(schedule.reminder_type),
            "title": schedule.title,
            "message": schedule.message,
            "related_type": schedule.related_type,
            "related_id": schedule.related_id,
        },
    )

    channel = _notification_channel(schedule.channel)
    delivery_result = None
    if channel == NotificationChannel.EMAIL:
        delivery_result = await deliver_notification_email_to_user(
            user_id=int(schedule.user_id),
            title=schedule.title,
            message=schedule.message,
        )

    await record_notification_log(
        user_id=int(schedule.user_id),
        notification_id=int(notification.id),
        reminder_schedule_id=int(schedule.id),
        notification_type=_enum_value(schedule.reminder_type),
        channel=channel,
        title=schedule.title,
        message_summary=_summary(schedule.message),
        related_type=schedule.related_type,
        related_id=schedule.related_id,
        status=delivery_result.status if delivery_result is not None else NotificationLogStatus.SENT,
        provider=delivery_result.provider if delivery_result is not None else None,
        error_code=delivery_result.error_code if delivery_result is not None else None,
        error_message=delivery_result.error_message if delivery_result is not None else None,
        sent_at=delivery_result.sent_at if delivery_result is not None else now,
        failed_at=delivery_result.failed_at if delivery_result is not None else None,
    )

    await _advance_reminder_schedule(schedule, now)
    return True


async def _advance_reminder_schedule(schedule: ReminderSchedule, now: datetime) -> None:
    next_trigger_at = _calculate_next_trigger(schedule, now)
    data: dict[str, Any] = {
        "last_triggered_at": now,
        "next_trigger_at": next_trigger_at,
    }
    if next_trigger_at is None:
        data["is_active"] = False
    await notification_repository.update_reminder_schedule(int(schedule.id), data)


def _calculate_next_trigger(schedule: ReminderSchedule, now: datetime) -> datetime | None:
    if not schedule.schedule_time:
        return None

    parsed_time = _parse_schedule_time(schedule.schedule_time)
    if parsed_time is None:
        return None

    tz = _schedule_timezone(schedule.timezone)
    local_now = _aware_datetime(now).astimezone(tz)
    candidate = datetime.combine(local_now.date(), parsed_time, tzinfo=tz)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return candidate


def calculate_next_trigger_at(
    schedule_time: time | str | None,
    *,
    now: datetime | None = None,
    timezone: str = DEFAULT_REMINDER_TIMEZONE,
) -> datetime | None:
    formatted_time = _format_schedule_time(schedule_time)
    if formatted_time is None:
        return None
    parsed_time = _parse_schedule_time(formatted_time)
    if parsed_time is None:
        return None

    tz = _schedule_timezone(timezone)
    local_now = _aware_datetime(now or datetime.now(tz)).astimezone(tz)
    candidate = datetime.combine(local_now.date(), parsed_time, tzinfo=tz)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return candidate


def _parse_schedule_time(value: str | None) -> time | None:
    if not value:
        return None
    parts = value.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) > 2 else 0
        return time(hour=hour, minute=minute, second=second)
    except (IndexError, ValueError):
        return None


def _format_schedule_time(value: time | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, time):
        return f"{value.hour:02d}:{value.minute:02d}"
    parsed_time = _parse_schedule_time(str(value))
    if parsed_time is None:
        return None
    return f"{parsed_time.hour:02d}:{parsed_time.minute:02d}"


def _schedule_timezone(value: str | None) -> zoneinfo.ZoneInfo:
    try:
        return zoneinfo.ZoneInfo(value or "Asia/Seoul")
    except zoneinfo.ZoneInfoNotFoundError:
        return config.TIMEZONE


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=config.TIMEZONE)
    return value


async def _upsert_email_reminder_schedule(
    *,
    user_id: int,
    reminder_type: ReminderType,
    related_type: str,
    related_id: int | None,
    title: str,
    message: str,
    schedule_time: time | str | None,
    enabled: bool,
) -> ReminderSchedule | None:
    existing = await notification_repository.get_reminder_schedule_by_related(
        user_id=user_id,
        reminder_type=reminder_type,
        channel=NotificationChannel.EMAIL,
        related_type=related_type,
        related_id=related_id,
    )
    formatted_time = _format_schedule_time(schedule_time)
    if not enabled or formatted_time is None:
        if existing is None:
            return None
        return await notification_repository.update_reminder_schedule(
            int(existing.id),
            {
                "is_active": False,
                "next_trigger_at": None,
            },
        )

    data = {
        "reminder_type": reminder_type,
        "channel": NotificationChannel.EMAIL,
        "title": title,
        "message": message,
        "related_type": related_type,
        "related_id": related_id,
        "schedule_time": formatted_time,
        "timezone": DEFAULT_REMINDER_TIMEZONE,
        "is_active": True,
        "next_trigger_at": calculate_next_trigger_at(formatted_time),
    }
    if existing is not None:
        return await notification_repository.update_reminder_schedule(int(existing.id), data)
    return await notification_repository.create_reminder_schedule(user_id, data)


async def _deactivate_email_reminder_schedules_by_type(
    *,
    user_id: int,
    reminder_type: ReminderType,
    related_type: str,
) -> int:
    return await notification_repository.update_reminder_schedules_by_related_type(
        user_id=user_id,
        reminder_type=reminder_type,
        channel=NotificationChannel.EMAIL,
        related_type=related_type,
        data={"is_active": False, "next_trigger_at": None},
    )


async def _get_user_notification_settings(user_id: int):
    from app.services import settings as setting_service

    return await setting_service.get_or_create_user_settings(user_id)


async def _count_active_user_challenges_for_reminder(user_id: int) -> int:
    from app.models.challenges import UserChallenge, UserChallengeStatus

    return await UserChallenge.filter(
        user_id=user_id,
        status=UserChallengeStatus.JOINED,
        canceled_at__isnull=True,
        completed_at__isnull=True,
    ).count()


def _notification_channel(value: NotificationChannel | str) -> NotificationChannel:
    if isinstance(value, NotificationChannel):
        return value
    try:
        return NotificationChannel(str(value))
    except ValueError:
        return NotificationChannel.IN_APP


def _enum_value(value: Enum | str) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _summary(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:255]


async def record_notification_log(
    *,
    user_id: int,
    notification_type: str,
    channel: NotificationChannel = NotificationChannel.IN_APP,
    title: str,
    status: NotificationLogStatus = NotificationLogStatus.PENDING,
    message_summary: str | None = None,
    notification_id: int | None = None,
    reminder_schedule_id: int | None = None,
    related_type: str | None = None,
    related_id: int | None = None,
    provider: str | None = None,
    provider_message_id: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    sent_at: datetime | None = None,
    failed_at: datetime | None = None,
) -> NotificationLog:
    """Record sanitized notification delivery metadata.

    Do not pass raw health values, verification codes, tokens, or full message
    bodies into message_summary/error fields.
    """

    data = {
        "notification_id": notification_id,
        "reminder_schedule_id": reminder_schedule_id,
        "notification_type": notification_type,
        "channel": channel,
        "title": title,
        "message_summary": message_summary,
        "related_type": related_type,
        "related_id": related_id,
        "status": status,
        "provider": provider,
        "provider_message_id": provider_message_id,
        "error_code": error_code,
        "error_message": error_message,
        "sent_at": sent_at,
        "failed_at": failed_at,
    }
    return await notification_repository.create_notification_log(user_id, data)


async def update_notification_log_with_email_result(
    notification_log_id: int,
    delivery_result: NotificationEmailDeliveryResult,
) -> NotificationLog | None:
    return await notification_repository.update_notification_log(
        notification_log_id,
        {
            "status": delivery_result.status,
            "provider": delivery_result.provider,
            "error_code": delivery_result.error_code,
            "error_message": delivery_result.error_message,
            "sent_at": delivery_result.sent_at,
            "failed_at": delivery_result.failed_at,
        },
    )


async def _enqueue_notification_email_delivery(
    *,
    user_id: int,
    notification: Notification,
    action_url: str | None,
) -> None:
    log = await record_notification_log(
        user_id=user_id,
        notification_id=int(notification.id),
        notification_type=notification.notification_type,
        channel=NotificationChannel.EMAIL,
        title=notification.title,
        message_summary=_summary(notification.message),
        related_type=notification.related_type,
        related_id=notification.related_id,
        status=NotificationLogStatus.PENDING,
    )
    try:
        from app.services import service_jobs as service_job_service

        await service_job_service.enqueue_notification_email_send(
            user_id=user_id,
            notification_id=int(notification.id),
            notification_log_id=int(log.id),
            title=notification.title,
            message=notification.message,
            action_url=action_url,
        )
    except Exception:
        logger.exception("Failed to enqueue notification email job", extra={"notification_id": notification.id})
        await notification_repository.update_notification_log(
            int(log.id),
            {
                "status": NotificationLogStatus.FAILED,
                "error_code": "notification_email_enqueue_failed",
                "error_message": "notification_email_enqueue_failed",
                "failed_at": datetime.now(config.TIMEZONE),
            },
        )
