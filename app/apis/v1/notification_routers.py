from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import ensure_found, ensure_owner
from app.dependencies.security import get_request_user
from app.dtos.notifications import NotificationCreateRequest, NotificationResponse
from app.models.users import User
from app.services import notifications as notification_service

notification_router = APIRouter(prefix="/notifications", tags=["notifications"])


@notification_router.post("", response_model=NotificationResponse, status_code=status.HTTP_201_CREATED)
async def create_notification(request: NotificationCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    return await notification_service.create_notification(user.id, request)


@notification_router.get("", response_model=list[NotificationResponse])
async def list_notifications(
    user: Annotated[User, Depends(get_request_user)],
    is_read: bool | None = None,
    limit: int = 20,
    offset: int = 0,
):
    return await notification_service.list_notifications(user.id, is_read=is_read, limit=limit, offset=offset)


@notification_router.get("/unread", response_model=list[NotificationResponse])
async def list_unread_notifications(user: Annotated[User, Depends(get_request_user)], limit: int = 20, offset: int = 0):
    return await notification_service.list_unread_notifications(user.id, limit=limit, offset=offset)


@notification_router.patch("/{notification_id}/read", response_model=NotificationResponse)
async def mark_notification_as_read(notification_id: int, user: Annotated[User, Depends(get_request_user)]):
    notification = ensure_found(
        await notification_service.get_notification(notification_id),
        "알림을 찾을 수 없습니다.",
    )
    ensure_owner(notification.user_id, user)
    marked = await notification_service.mark_notification_as_read(notification_id)
    return ensure_found(marked, "알림을 찾을 수 없습니다.")


@notification_router.patch("/read-all", response_model=list[NotificationResponse])
async def mark_all_notifications_as_read(user: Annotated[User, Depends(get_request_user)]):
    return await notification_service.mark_all_notifications_as_read(user.id)


@notification_router.delete("/{notification_id}")
async def delete_notification(notification_id: int, user: Annotated[User, Depends(get_request_user)]):
    notification = ensure_found(
        await notification_service.get_notification(notification_id),
        "알림을 찾을 수 없습니다.",
    )
    ensure_owner(notification.user_id, user)
    deleted_count = await notification_service.delete_notification(notification_id)
    return {"deleted_count": deleted_count}
