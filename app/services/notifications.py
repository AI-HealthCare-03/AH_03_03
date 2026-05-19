from datetime import datetime

from app.core import config
from app.dtos.notifications import NotificationCreateRequest, NotificationUpdateRequest
from app.models.notifications import Notification
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
