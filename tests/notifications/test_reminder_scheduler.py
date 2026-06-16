from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_runtime.jobs import scheduler
from app.core import config
from app.models.notifications import NotificationChannel, NotificationLogStatus, ReminderType
from app.services import notifications as notification_service
from app.services.notification_email import NotificationEmailDeliveryResult


class FakeNotificationRepository:
    def __init__(self, schedules: list[SimpleNamespace]) -> None:
        self.schedules = schedules
        self.notifications: list[SimpleNamespace] = []
        self.logs: list[dict[str, Any]] = []
        self.updated_schedules: list[tuple[int, dict[str, Any]]] = []

    async def list_due_reminder_schedules(self, now: datetime, limit: int = 100) -> list[SimpleNamespace]:
        due = [
            schedule
            for schedule in self.schedules
            if schedule.is_active and schedule.next_trigger_at is not None and schedule.next_trigger_at <= now
        ]
        return due[:limit]

    async def has_notification_log_for_reminder_since(self, reminder_schedule_id: int, since: datetime) -> bool:
        return any(
            log["reminder_schedule_id"] == reminder_schedule_id and log["created_at"] >= since for log in self.logs
        )

    async def create_notification(self, user_id: int, data: dict[str, Any]) -> SimpleNamespace:
        notification = SimpleNamespace(id=len(self.notifications) + 1, user_id=user_id, **data)
        self.notifications.append(notification)
        return notification

    async def create_notification_log(self, user_id: int, data: dict[str, Any]) -> SimpleNamespace:
        log = {
            "id": len(self.logs) + 1,
            "user_id": user_id,
            "created_at": datetime(2026, 5, 31, 9, 0, 0, tzinfo=config.TIMEZONE),
            **data,
        }
        self.logs.append(log)
        return SimpleNamespace(**log)

    async def update_reminder_schedule(self, schedule_id: int, data: dict[str, Any]) -> SimpleNamespace | None:
        self.updated_schedules.append((schedule_id, data))
        for schedule in self.schedules:
            if schedule.id == schedule_id:
                for key, value in data.items():
                    setattr(schedule, key, value)
                return schedule
        return None


def _schedule(
    *,
    schedule_id: int = 1,
    channel: NotificationChannel = NotificationChannel.IN_APP,
    next_trigger_at: datetime | None = None,
    schedule_time: str | None = "09:00",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=schedule_id,
        user_id=7,
        reminder_type=ReminderType.CHALLENGE,
        channel=channel,
        title="챌린지 알림",
        message="오늘 챌린지를 기록해보세요.",
        related_type="challenge",
        related_id=77,
        schedule_time=schedule_time,
        cron_expression=None,
        timezone="Asia/Seoul",
        is_active=True,
        last_triggered_at=None,
        next_trigger_at=next_trigger_at or datetime(2026, 5, 31, 8, 59, 0),
    )


@pytest.mark.asyncio
async def test_due_reminder_creates_notification_and_log(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 5, 31, 9, 0, 0, tzinfo=config.TIMEZONE)
    repository = FakeNotificationRepository([_schedule(next_trigger_at=now - timedelta(minutes=1))])
    monkeypatch.setattr(notification_service, "notification_repository", repository)

    created_count = await notification_service.process_due_reminder_schedules(now=now)

    assert created_count == 1
    assert len(repository.notifications) == 1
    assert repository.notifications[0].title == "챌린지 알림"
    assert len(repository.logs) == 1
    assert repository.logs[0]["status"] == NotificationLogStatus.SENT
    assert repository.updated_schedules[0][1]["last_triggered_at"] == now
    assert repository.updated_schedules[0][1]["next_trigger_at"] > now


@pytest.mark.asyncio
async def test_due_reminder_does_not_create_duplicate_notification(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 5, 31, 9, 0, 0, tzinfo=config.TIMEZONE)
    trigger_at = now - timedelta(minutes=1)
    repository = FakeNotificationRepository([_schedule(next_trigger_at=trigger_at)])
    repository.logs.append(
        {
            "id": 1,
            "user_id": 7,
            "reminder_schedule_id": 1,
            "created_at": now,
            "status": NotificationLogStatus.SENT,
        }
    )
    monkeypatch.setattr(notification_service, "notification_repository", repository)

    created_count = await notification_service.process_due_reminder_schedules(now=now)

    assert created_count == 0
    assert repository.notifications == []
    assert repository.updated_schedules
    assert repository.updated_schedules[0][1]["next_trigger_at"] > now


@pytest.mark.asyncio
async def test_email_channel_reminder_records_skipped_when_email_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 5, 31, 9, 0, 0, tzinfo=config.TIMEZONE)
    repository = FakeNotificationRepository(
        [_schedule(channel=NotificationChannel.EMAIL, next_trigger_at=now - timedelta(minutes=1))]
    )

    async def fake_deliver_notification_email_to_user(**kwargs):
        assert kwargs["user_id"] == 7
        return NotificationEmailDeliveryResult(
            status=NotificationLogStatus.SKIPPED,
            sent=False,
            error_code="email_delivery_disabled",
            error_message="email_delivery_disabled",
        )

    monkeypatch.setattr(notification_service, "notification_repository", repository)
    monkeypatch.setattr(notification_service, "deliver_notification_email_to_user", fake_deliver_notification_email_to_user)

    created_count = await notification_service.process_due_reminder_schedules(now=now)

    assert created_count == 1
    assert repository.logs[0]["status"] == NotificationLogStatus.SKIPPED
    assert repository.logs[0]["channel"] == NotificationChannel.EMAIL
    assert repository.logs[0]["error_code"] == "email_delivery_disabled"


@pytest.mark.asyncio
async def test_scheduler_disabled_does_not_process_reminders(monkeypatch: pytest.MonkeyPatch) -> None:
    processed = False

    async def fake_process_due_reminder_schedules() -> int:
        nonlocal processed
        processed = True
        return 1

    monkeypatch.setattr(notification_service, "process_due_reminder_schedules", fake_process_due_reminder_schedules)
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", False)

    stop_event = asyncio.Event()
    await scheduler.run_scheduler_forever(stop_event, interval_seconds=1)

    assert processed is False
