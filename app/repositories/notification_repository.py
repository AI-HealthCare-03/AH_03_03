from typing import Any

from app.models.notifications import Notification


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
