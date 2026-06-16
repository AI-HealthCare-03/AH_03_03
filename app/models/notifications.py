from enum import StrEnum

from tortoise import fields, models


class ReminderType(StrEnum):
    MEDICATION = "MEDICATION"
    CHALLENGE = "CHALLENGE"
    HEALTH_RECORD = "HEALTH_RECORD"
    FAMILY_ALERT = "FAMILY_ALERT"
    SYSTEM = "SYSTEM"


class NotificationChannel(StrEnum):
    IN_APP = "IN_APP"
    EMAIL = "EMAIL"
    SMS = "SMS"
    PUSH = "PUSH"
    KAKAO = "KAKAO"


class NotificationLogStatus(StrEnum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELED = "CANCELED"


class FCMTokenPlatform(StrEnum):
    WEB = "web"
    ANDROID = "android"
    IOS = "ios"


class Notification(models.Model):
    """User-facing notification inbox item.

    Delivery scheduling and provider send history are tracked separately by
    ReminderSchedule and NotificationLog.
    """

    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="notifications")
    notification_type = fields.CharField(max_length=30)
    title = fields.CharField(max_length=100)
    message = fields.TextField()
    is_read = fields.BooleanField(default=False)
    related_type = fields.CharField(max_length=50, null=True)
    related_id = fields.BigIntField(null=True)
    read_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "notifications"
        indexes = (("user_id",), ("user_id", "is_read"), ("notification_type",))


class ReminderSchedule(models.Model):
    """User-owned reminder schedule definition.

    This table describes when an in-app/email/SMS/push/Kakao reminder should be
    triggered. Actual external delivery workers are intentionally out of scope
    for this implementation.
    """

    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="reminder_schedules")
    reminder_type = fields.CharEnumField(enum_type=ReminderType)
    channel = fields.CharEnumField(enum_type=NotificationChannel, default=NotificationChannel.IN_APP)
    title = fields.CharField(max_length=100)
    message = fields.TextField()
    related_type = fields.CharField(max_length=50, null=True)
    related_id = fields.BigIntField(null=True)
    schedule_time = fields.CharField(max_length=8, null=True)
    cron_expression = fields.CharField(max_length=100, null=True)
    timezone = fields.CharField(max_length=50, default="Asia/Seoul")
    is_active = fields.BooleanField(default=True)
    last_triggered_at = fields.DatetimeField(null=True)
    next_trigger_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "reminder_schedules"
        indexes = (
            ("user_id",),
            ("user_id", "is_active"),
            ("reminder_type",),
            ("channel",),
            ("next_trigger_at",),
        )


class NotificationLog(models.Model):
    """Notification delivery attempt history.

    Store only delivery metadata and a sanitized message summary. Do not store
    passwords, tokens, verification codes, raw health values, or full payloads.
    """

    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="notification_logs")
    notification = fields.ForeignKeyField(
        "models.Notification",
        related_name="delivery_logs",
        null=True,
        on_delete=fields.SET_NULL,
    )
    reminder_schedule = fields.ForeignKeyField(
        "models.ReminderSchedule",
        related_name="notification_logs",
        null=True,
        on_delete=fields.SET_NULL,
    )
    notification_type = fields.CharField(max_length=30)
    channel = fields.CharEnumField(enum_type=NotificationChannel, default=NotificationChannel.IN_APP)
    title = fields.CharField(max_length=100)
    message_summary = fields.CharField(max_length=255, null=True)
    related_type = fields.CharField(max_length=50, null=True)
    related_id = fields.BigIntField(null=True)
    status = fields.CharEnumField(enum_type=NotificationLogStatus, default=NotificationLogStatus.PENDING)
    provider = fields.CharField(max_length=50, null=True)
    provider_message_id = fields.CharField(max_length=120, null=True)
    error_code = fields.CharField(max_length=50, null=True)
    error_message = fields.CharField(max_length=255, null=True)
    sent_at = fields.DatetimeField(null=True)
    failed_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "notification_logs"
        indexes = (
            ("user_id",),
            ("notification_id",),
            ("reminder_schedule_id",),
            ("status",),
            ("channel",),
            ("created_at",),
        )


class UserFCMToken(models.Model):
    """FCM registration token for one signed-in user device/browser."""

    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="fcm_tokens", on_delete=fields.CASCADE)
    token = fields.CharField(max_length=512, unique=True)
    platform = fields.CharEnumField(enum_type=FCMTokenPlatform)
    device_id = fields.CharField(max_length=128, null=True)
    user_agent = fields.CharField(max_length=500, null=True)
    is_active = fields.BooleanField(default=True)
    last_seen_at = fields.DatetimeField()
    revoked_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_fcm_tokens"
        indexes = (
            ("user_id",),
            ("user_id", "is_active"),
            ("platform",),
            ("last_seen_at",),
        )
