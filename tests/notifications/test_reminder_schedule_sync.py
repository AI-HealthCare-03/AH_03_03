from datetime import UTC, datetime, time
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.notifications import NotificationChannel, ReminderType
from app.services import notifications as notification_service
from app.services import settings as setting_service


class FakeReminderScheduleRepository:
    def __init__(self) -> None:
        self.schedules: list[SimpleNamespace] = []
        self.next_id = 1

    async def get_reminder_schedule_by_related(
        self,
        *,
        user_id: int,
        reminder_type: ReminderType,
        channel: NotificationChannel,
        related_type: str,
        related_id: int | None,
    ) -> SimpleNamespace | None:
        matches = [
            schedule
            for schedule in self.schedules
            if schedule.user_id == user_id
            and schedule.reminder_type == reminder_type
            and schedule.channel == channel
            and schedule.related_type == related_type
            and schedule.related_id == related_id
        ]
        return matches[-1] if matches else None

    async def create_reminder_schedule(self, user_id: int, data: dict[str, Any]) -> SimpleNamespace:
        schedule = SimpleNamespace(id=self.next_id, user_id=user_id, **data)
        self.next_id += 1
        self.schedules.append(schedule)
        return schedule

    async def update_reminder_schedule(self, schedule_id: int, data: dict[str, Any]) -> SimpleNamespace | None:
        for schedule in self.schedules:
            if schedule.id == schedule_id:
                for key, value in data.items():
                    setattr(schedule, key, value)
                return schedule
        return None

    async def update_reminder_schedules_by_related_type(
        self,
        *,
        user_id: int,
        reminder_type: ReminderType,
        channel: NotificationChannel,
        related_type: str,
        data: dict[str, Any],
    ) -> int:
        updated_count = 0
        for schedule in self.schedules:
            if (
                schedule.user_id == user_id
                and schedule.reminder_type == reminder_type
                and schedule.channel == channel
                and schedule.related_type == related_type
            ):
                for key, value in data.items():
                    setattr(schedule, key, value)
                updated_count += 1
        return updated_count


def _settings(
    *,
    notification_enabled: bool = True,
    medication_reminder_enabled: bool = True,
    challenge_reminder_enabled: bool = True,
    challenge_reminder_time: time | None = None,
    diet_reminder_enabled: bool = False,
    diet_reminder_time: time | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        notification_enabled=notification_enabled,
        medication_reminder_enabled=medication_reminder_enabled,
        challenge_reminder_enabled=challenge_reminder_enabled,
        challenge_reminder_time=challenge_reminder_time,
        diet_reminder_enabled=diet_reminder_enabled,
        diet_reminder_time=diet_reminder_time,
    )


async def _enabled_settings(user_id: int) -> SimpleNamespace:
    return _settings()


@pytest.mark.asyncio
async def test_medication_reminder_schedule_is_created_and_updated_without_duplicates(monkeypatch) -> None:
    repository = FakeReminderScheduleRepository()
    monkeypatch.setattr(notification_service, "notification_repository", repository)
    monkeypatch.setattr(notification_service, "_get_user_notification_settings", _enabled_settings)

    await notification_service.sync_medication_reminder_schedule(
        user_id=7,
        medication_id=10,
        reminder_time=time(8, 30),
    )
    await notification_service.sync_medication_reminder_schedule(
        user_id=7,
        medication_id=10,
        reminder_time="09:40:00",
    )

    assert len(repository.schedules) == 1
    schedule = repository.schedules[0]
    assert schedule.reminder_type == ReminderType.MEDICATION
    assert schedule.channel == NotificationChannel.EMAIL
    assert schedule.related_type == "medication"
    assert schedule.related_id == 10
    assert schedule.schedule_time == "09:40"
    assert schedule.is_active is True
    assert schedule.next_trigger_at is not None


@pytest.mark.asyncio
async def test_medication_reminder_schedule_is_disabled_when_time_or_setting_is_missing(monkeypatch) -> None:
    repository = FakeReminderScheduleRepository()
    monkeypatch.setattr(notification_service, "notification_repository", repository)
    monkeypatch.setattr(notification_service, "_get_user_notification_settings", _enabled_settings)

    await notification_service.sync_medication_reminder_schedule(
        user_id=7,
        medication_id=10,
        reminder_time=time(8, 30),
    )
    await notification_service.sync_medication_reminder_schedule(
        user_id=7,
        medication_id=10,
        reminder_time=None,
    )

    assert repository.schedules[0].is_active is False
    assert repository.schedules[0].next_trigger_at is None

    async def disabled_medication_settings(user_id: int) -> SimpleNamespace:
        return _settings(medication_reminder_enabled=False)

    monkeypatch.setattr(notification_service, "_get_user_notification_settings", disabled_medication_settings)
    await notification_service.sync_medication_reminder_schedule(
        user_id=7,
        medication_id=11,
        reminder_time=time(8, 30),
    )

    assert len(repository.schedules) == 1


@pytest.mark.asyncio
async def test_diet_reminder_schedule_follows_user_setting(monkeypatch) -> None:
    repository = FakeReminderScheduleRepository()
    monkeypatch.setattr(notification_service, "notification_repository", repository)

    async def diet_enabled_settings(user_id: int) -> SimpleNamespace:
        return _settings(diet_reminder_enabled=True)

    monkeypatch.setattr(notification_service, "_get_user_notification_settings", diet_enabled_settings)

    await notification_service.sync_diet_reminder_schedule_for_user(7)

    assert len(repository.schedules) == 1
    schedule = repository.schedules[0]
    assert schedule.reminder_type == ReminderType.SYSTEM
    assert schedule.related_type == "diet_record_reminder"
    assert schedule.related_id is None
    assert schedule.schedule_time == "20:00"
    assert schedule.is_active is True

    async def diet_custom_time_settings(user_id: int) -> SimpleNamespace:
        return _settings(diet_reminder_enabled=True, diet_reminder_time=time(19, 30))

    monkeypatch.setattr(notification_service, "_get_user_notification_settings", diet_custom_time_settings)
    await notification_service.sync_diet_reminder_schedule_for_user(7)

    assert len(repository.schedules) == 1
    assert repository.schedules[0].schedule_time == "19:30"
    assert repository.schedules[0].is_active is True

    monkeypatch.setattr(notification_service, "_get_user_notification_settings", _enabled_settings)
    await notification_service.sync_diet_reminder_schedule_for_user(7)

    assert repository.schedules[0].is_active is False


@pytest.mark.asyncio
async def test_challenge_reminder_schedule_requires_enabled_setting_and_active_challenge(monkeypatch) -> None:
    repository = FakeReminderScheduleRepository()
    monkeypatch.setattr(notification_service, "notification_repository", repository)

    async def challenge_enabled_settings(user_id: int) -> SimpleNamespace:
        return _settings(challenge_reminder_time=time(21, 30))

    async def two_active_challenges(user_id: int) -> int:
        return 2

    monkeypatch.setattr(notification_service, "_get_user_notification_settings", challenge_enabled_settings)
    monkeypatch.setattr(notification_service, "_count_active_user_challenges_for_reminder", two_active_challenges)

    await notification_service.sync_challenge_reminder_schedule_for_user(7)

    assert len(repository.schedules) == 1
    schedule = repository.schedules[0]
    assert schedule.reminder_type == ReminderType.CHALLENGE
    assert schedule.related_type == "challenge_daily_reminder"
    assert schedule.schedule_time == "21:30"
    assert schedule.is_active is True

    async def no_active_challenges(user_id: int) -> int:
        return 0

    monkeypatch.setattr(notification_service, "_count_active_user_challenges_for_reminder", no_active_challenges)
    await notification_service.sync_challenge_reminder_schedule_for_user(7)

    assert repository.schedules[0].is_active is False


def test_calculate_next_trigger_at_uses_today_or_tomorrow_in_kst() -> None:
    before_time = datetime(2026, 6, 18, 14, 0, tzinfo=notification_service.config.TIMEZONE)
    after_time = datetime(2026, 6, 18, 22, 0, tzinfo=notification_service.config.TIMEZONE)

    today_trigger = notification_service.calculate_next_trigger_at("20:00", now=before_time)
    tomorrow_trigger = notification_service.calculate_next_trigger_at("20:00", now=after_time)

    assert today_trigger is not None
    assert tomorrow_trigger is not None
    assert today_trigger.date().isoformat() == "2026-06-18"
    assert tomorrow_trigger.date().isoformat() == "2026-06-19"
    assert today_trigger.hour == 20
    assert today_trigger.minute == 0


@pytest.mark.asyncio
async def test_settings_update_syncs_automatic_reminder_schedules(monkeypatch) -> None:
    synced_user_ids: list[int] = []

    async def fake_update_user_setting(user_id: int, data: dict[str, Any]):
        return SimpleNamespace(user_id=user_id, **data)

    async def fake_get_user_setting_by_user(user_id: int):
        return SimpleNamespace(user_id=user_id)

    async def fake_sync_all_user_reminder_schedules(user_id: int) -> None:
        synced_user_ids.append(user_id)

    monkeypatch.setattr(setting_service.setting_repository, "get_user_setting_by_user", fake_get_user_setting_by_user)
    monkeypatch.setattr(setting_service.setting_repository, "update_user_setting", fake_update_user_setting)
    monkeypatch.setattr(
        notification_service,
        "sync_all_user_reminder_schedules",
        fake_sync_all_user_reminder_schedules,
    )

    from app.dtos.settings import UserSettingUpdateRequest

    updated = await setting_service.update_user_settings(7, UserSettingUpdateRequest(medication_reminder_enabled=False))

    assert updated is not None
    assert synced_user_ids == [7]


@pytest.mark.asyncio
async def test_settings_update_creates_missing_row_and_normalizes_time(monkeypatch) -> None:
    created_payloads: list[tuple[int, dict[str, Any]]] = []
    updated_payloads: list[tuple[int, dict[str, Any]]] = []
    synced_user_ids: list[int] = []

    async def fake_get_user_setting_by_user(user_id: int):
        return None

    async def fake_create_user_setting(user_id: int, data: dict[str, Any]):
        created_payloads.append((user_id, data))
        return SimpleNamespace(user_id=user_id, **data)

    async def fake_update_user_setting(user_id: int, data: dict[str, Any]):
        updated_payloads.append((user_id, data))
        return SimpleNamespace(user_id=user_id, **data)

    async def fake_sync_all_user_reminder_schedules(user_id: int) -> None:
        synced_user_ids.append(user_id)

    monkeypatch.setattr(setting_service.setting_repository, "get_user_setting_by_user", fake_get_user_setting_by_user)
    monkeypatch.setattr(setting_service.setting_repository, "create_user_setting", fake_create_user_setting)
    monkeypatch.setattr(setting_service.setting_repository, "update_user_setting", fake_update_user_setting)
    monkeypatch.setattr(
        notification_service,
        "sync_all_user_reminder_schedules",
        fake_sync_all_user_reminder_schedules,
    )

    from app.dtos.settings import UserSettingUpdateRequest

    updated = await setting_service.update_user_settings(
        7,
        UserSettingUpdateRequest(
            challenge_reminder_time=time(21, 30, tzinfo=UTC),
            diet_reminder_time=time(20, 0, tzinfo=UTC),
        ),
    )

    assert updated is not None
    assert created_payloads
    assert updated_payloads == [
        (
            7,
            {
                "challenge_reminder_time": time(21, 30),
                "diet_reminder_time": time(20, 0),
            },
        )
    ]
    assert synced_user_ids == [7]
