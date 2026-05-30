from typing import Any

from app.models.notifications import Notification, NotificationLog, ReminderSchedule, UserFCMToken


async def create_notification(user_id: int, data: dict[str, Any]) -> Notification:
    return await Notification.create(user_id=user_id, **data)


async def get_notification_by_id(notification_id: int) -> Notification | None:
    return await Notification.get_or_none(id=notification_id)


async def list_notifications_by_user(
    user_id: int, is_read: bool | None = None, limit: int = 20, offset: int = 0
) -> list[Notification]:
    query = Notification.filter(user_id=user_id)
    if is_read is not None:
        query = query.filter(is_read=is_read)
    return await query.order_by("-created_at").offset(offset).limit(limit)


async def list_notifications_by_type(notification_type: str, limit: int = 20, offset: int = 0) -> list[Notification]:
    return (
        await Notification.filter(notification_type=notification_type)
        .order_by("-created_at")
        .offset(offset)
        .limit(limit)
    )


async def update_notification(notification_id: int, data: dict[str, Any]) -> Notification | None:
    notification = await get_notification_by_id(notification_id)
    if notification is None:
        return None
    for key, value in data.items():
        setattr(notification, key, value)
    await notification.save(update_fields=list(data.keys()) if data else None)
    return notification


async def delete_notification(notification_id: int) -> int:
    return await Notification.filter(id=notification_id).delete()


async def create_reminder_schedule(user_id: int, data: dict[str, Any]) -> ReminderSchedule:
    return await ReminderSchedule.create(user_id=user_id, **data)


async def get_reminder_schedule_by_id(schedule_id: int) -> ReminderSchedule | None:
    return await ReminderSchedule.get_or_none(id=schedule_id)


async def list_reminder_schedules_by_user(
    user_id: int,
    is_active: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ReminderSchedule]:
    query = ReminderSchedule.filter(user_id=user_id)
    if is_active is not None:
        query = query.filter(is_active=is_active)
    return await query.order_by("-created_at").offset(offset).limit(limit)


async def update_reminder_schedule(schedule_id: int, data: dict[str, Any]) -> ReminderSchedule | None:
    schedule = await get_reminder_schedule_by_id(schedule_id)
    if schedule is None:
        return None
    for key, value in data.items():
        setattr(schedule, key, value)
    await schedule.save(update_fields=list(data.keys()) if data else None)
    return schedule


async def delete_reminder_schedule(schedule_id: int) -> int:
    return await ReminderSchedule.filter(id=schedule_id).delete()


async def list_due_reminder_schedules(now, limit: int = 100) -> list[ReminderSchedule]:
    return (
        await ReminderSchedule.filter(
            is_active=True,
            next_trigger_at__isnull=False,
            next_trigger_at__lte=now,
        )
        .order_by("next_trigger_at", "id")
        .limit(limit)
    )


async def create_notification_log(user_id: int, data: dict[str, Any]) -> NotificationLog:
    return await NotificationLog.create(user_id=user_id, **data)


async def list_notification_logs_by_user(user_id: int, limit: int = 50, offset: int = 0) -> list[NotificationLog]:
    return await NotificationLog.filter(user_id=user_id).order_by("-created_at").offset(offset).limit(limit)


async def has_notification_log_for_reminder_since(reminder_schedule_id: int, since) -> bool:
    return await NotificationLog.filter(
        reminder_schedule_id=reminder_schedule_id,
        created_at__gte=since,
    ).exists()


async def get_fcm_token_by_token(token: str) -> UserFCMToken | None:
    return await UserFCMToken.get_or_none(token=token)


async def upsert_fcm_token(user_id: int, data: dict[str, Any]) -> UserFCMToken:
    existing = await get_fcm_token_by_token(data["token"])
    if existing is None:
        return await UserFCMToken.create(user_id=user_id, **data)

    existing.user_id = user_id
    for key, value in data.items():
        setattr(existing, key, value)
    await existing.save(
        update_fields=[
            "user_id",
            "platform",
            "device_id",
            "user_agent",
            "is_active",
            "last_seen_at",
            "revoked_at",
            "updated_at",
        ]
    )
    return existing


async def deactivate_fcm_token(user_id: int, token: str, revoked_at) -> int:
    return await UserFCMToken.filter(user_id=user_id, token=token).update(
        is_active=False,
        revoked_at=revoked_at,
        updated_at=revoked_at,
    )


async def deactivate_fcm_tokens_by_user(user_id: int, revoked_at) -> int:
    return await UserFCMToken.filter(user_id=user_id, is_active=True).update(
        is_active=False,
        revoked_at=revoked_at,
        updated_at=revoked_at,
    )
