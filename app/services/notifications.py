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
)
from app.repositories import notification_repository

logger = logging.getLogger(__name__)


async def create_notification(user_id: int, request: NotificationCreateRequest) -> Notification:
    return await notification_repository.create_notification(user_id, request.model_dump())


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
        status=NotificationLogStatus.SENT,
        sent_at=now,
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


def _schedule_timezone(value: str | None) -> zoneinfo.ZoneInfo:
    try:
        return zoneinfo.ZoneInfo(value or "Asia/Seoul")
    except zoneinfo.ZoneInfoNotFoundError:
        return config.TIMEZONE


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=config.TIMEZONE)
    return value


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
