from datetime import datetime

from pydantic import BaseModel, ConfigDict


class NotificationCreateRequest(BaseModel):
    notification_type: str
    title: str
    message: str
    related_type: str | None = None
    related_id: int | None = None


class NotificationUpdateRequest(BaseModel):
    notification_type: str | None = None
    title: str | None = None
    message: str | None = None
    is_read: bool | None = None
    related_type: str | None = None
    related_id: int | None = None
    read_at: datetime | None = None


class NotificationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    notification_type: str
    title: str
    message: str
    is_read: bool
    related_type: str | None
    related_id: int | None
    read_at: datetime | None
    created_at: datetime
    updated_at: datetime


class NotificationListResponse(BaseModel):
    items: list[NotificationResponse]
    total: int
