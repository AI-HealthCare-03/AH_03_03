from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.notifications import NotificationChannel, NotificationLogStatus, ReminderType

MVP_NOTIFICATION_CHANNELS = {NotificationChannel.IN_APP, NotificationChannel.EMAIL}


def _validate_mvp_notification_channel(channel: NotificationChannel | None) -> NotificationChannel | None:
    if channel is not None and channel not in MVP_NOTIFICATION_CHANNELS:
        raise ValueError("MVP에서는 서비스 내부 알림과 이메일 알림만 사용할 수 있습니다.")
    return channel


class NotificationCreateRequest(BaseModel):
    notification_type: str
    title: str
    message: str
    related_type: str | None = None
    related_id: int | None = None
    send_email: bool = False
    action_url: str | None = Field(default=None, max_length=500)


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


class ReminderScheduleCreateRequest(BaseModel):
    reminder_type: ReminderType
    channel: NotificationChannel = NotificationChannel.IN_APP
    title: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)
    related_type: str | None = Field(default=None, max_length=50)
    related_id: int | None = None
    schedule_time: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    cron_expression: str | None = Field(default=None, max_length=100)
    timezone: str = Field(default="Asia/Seoul", min_length=1, max_length=50)
    is_active: bool = True
    next_trigger_at: datetime | None = None

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, channel: NotificationChannel) -> NotificationChannel:
        return _validate_mvp_notification_channel(channel) or NotificationChannel.IN_APP


class ReminderScheduleUpdateRequest(BaseModel):
    reminder_type: ReminderType | None = None
    channel: NotificationChannel | None = None
    title: str | None = Field(default=None, min_length=1, max_length=100)
    message: str | None = Field(default=None, min_length=1, max_length=1000)
    related_type: str | None = Field(default=None, max_length=50)
    related_id: int | None = None
    schedule_time: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    cron_expression: str | None = Field(default=None, max_length=100)
    timezone: str | None = Field(default=None, min_length=1, max_length=50)
    is_active: bool | None = None
    next_trigger_at: datetime | None = None

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, channel: NotificationChannel | None) -> NotificationChannel | None:
        return _validate_mvp_notification_channel(channel)


class ReminderScheduleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    reminder_type: ReminderType
    channel: NotificationChannel
    title: str
    message: str
    related_type: str | None
    related_id: int | None
    schedule_time: str | None
    cron_expression: str | None
    timezone: str
    is_active: bool
    last_triggered_at: datetime | None
    next_trigger_at: datetime | None
    created_at: datetime
    updated_at: datetime


class NotificationLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    notification_id: int | None
    reminder_schedule_id: int | None
    notification_type: str
    channel: NotificationChannel
    title: str
    message_summary: str | None
    related_type: str | None
    related_id: int | None
    status: NotificationLogStatus
    provider: str | None
    provider_message_id: str | None
    error_code: str | None
    error_message: str | None
    sent_at: datetime | None
    failed_at: datetime | None
    created_at: datetime
