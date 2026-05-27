from datetime import datetime

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
