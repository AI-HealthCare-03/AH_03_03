from datetime import time, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.dtos.medications import MedicationCreateRequest, MedicationUpdateRequest
from app.services import medications as medication_service


def test_medication_create_request_normalizes_hh_mm_to_naive_time() -> None:
    request = MedicationCreateRequest(
        name="비타민D",
        medication_type="SUPPLEMENT",
        reminder_time="04:30",
    )

    assert request.reminder_time == time(4, 30)
    assert request.reminder_time.tzinfo is None


def test_medication_create_request_normalizes_hh_mm_ss_to_naive_time() -> None:
    request = MedicationCreateRequest(
        name="비타민D",
        medication_type="SUPPLEMENT",
        reminder_time="04:30:00",
    )

    assert request.reminder_time == time(4, 30)
    assert request.reminder_time.tzinfo is None


def test_medication_create_request_normalizes_blank_time_to_none() -> None:
    request = MedicationCreateRequest(
        name="비타민D",
        medication_type="SUPPLEMENT",
        reminder_time="",
    )

    assert request.reminder_time is None


def test_medication_update_request_strips_timezone_from_time_object() -> None:
    request = MedicationUpdateRequest(reminder_time=time(4, 30, tzinfo=timezone(timedelta(hours=9))))

    assert request.reminder_time == time(4, 30)
    assert request.reminder_time.tzinfo is None


@pytest.mark.asyncio
async def test_create_medication_strips_timezone_before_repository(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_create_medication(user_id: int, data: dict):
        captured.update(user_id=user_id, data=data)
        return SimpleNamespace(id=1, user_id=user_id, **data)

    async def fake_sync_medication_reminder_schedule(**kwargs):
        return None

    monkeypatch.setattr(medication_service.medication_repository, "create_medication", fake_create_medication)
    monkeypatch.setattr(
        medication_service.notification_service,
        "sync_medication_reminder_schedule",
        fake_sync_medication_reminder_schedule,
    )

    request = MedicationCreateRequest(
        name="비타민D",
        medication_type="SUPPLEMENT",
        reminder_time=time(4, 30, tzinfo=timezone(timedelta(hours=9))),
    )

    await medication_service.create_medication(7, request)

    data = captured["data"]
    assert isinstance(data, dict)
    assert data["reminder_time"] == time(4, 30)
    assert data["reminder_time"].tzinfo is None


@pytest.mark.asyncio
async def test_update_medication_strips_timezone_before_repository(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_update_medication(medication_id: int, data: dict):
        captured.update(medication_id=medication_id, data=data)
        return SimpleNamespace(id=medication_id, user_id=7, is_active=True, **data)

    async def fake_sync_medication_reminder_schedule(**kwargs):
        return None

    monkeypatch.setattr(medication_service.medication_repository, "update_medication", fake_update_medication)
    monkeypatch.setattr(
        medication_service.notification_service,
        "sync_medication_reminder_schedule",
        fake_sync_medication_reminder_schedule,
    )

    request = MedicationUpdateRequest(reminder_time=time(4, 30, tzinfo=timezone(timedelta(hours=9))))

    await medication_service.update_medication(11, request)

    data = captured["data"]
    assert isinstance(data, dict)
    assert data["reminder_time"] == time(4, 30)
    assert data["reminder_time"].tzinfo is None
