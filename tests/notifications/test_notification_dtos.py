import pytest
from pydantic import ValidationError

from app.dtos.notifications import ReminderScheduleCreateRequest, ReminderScheduleUpdateRequest
from app.models.notifications import NotificationChannel, ReminderType


def test_reminder_schedule_accepts_mvp_notification_channels() -> None:
    request = ReminderScheduleCreateRequest(
        reminder_type=ReminderType.CHALLENGE,
        channel=NotificationChannel.EMAIL,
        title="챌린지 알림",
        message="오늘 챌린지를 기록해보세요.",
    )

    assert request.channel == NotificationChannel.EMAIL


@pytest.mark.parametrize("channel", [NotificationChannel.SMS, NotificationChannel.KAKAO])
def test_reminder_schedule_rejects_deferred_external_channels(channel: NotificationChannel) -> None:
    with pytest.raises(ValidationError):
        ReminderScheduleCreateRequest(
            reminder_type=ReminderType.CHALLENGE,
            channel=channel,
            title="챌린지 알림",
            message="오늘 챌린지를 기록해보세요.",
        )


def test_reminder_schedule_update_rejects_deferred_external_channels() -> None:
    with pytest.raises(ValidationError):
        ReminderScheduleUpdateRequest(channel=NotificationChannel.SMS)
