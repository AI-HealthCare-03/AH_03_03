from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.dtos.challenges import ChallengeLogCreateRequest
from app.models.challenges import UserChallengeStatus
from app.services import challenges as challenge_service


@pytest.mark.asyncio
async def test_create_challenge_log_blocks_second_log_when_daily_goal_is_one(monkeypatch) -> None:
    request = ChallengeLogCreateRequest(log_date=date(2026, 5, 28), is_completed=True, memo="첫 완료")
    counts = iter([0, 1])
    created_payloads: list[dict] = []

    async def fake_get_user_challenge_by_id(user_challenge_id: int):
        assert user_challenge_id == 7
        return SimpleNamespace(id=user_challenge_id, challenge_id=11)

    async def fake_get_challenge(id: int):
        assert id == 11
        return SimpleNamespace(target_metric="minutes", target_value="30")

    async def fake_count_logs_by_date(user_challenge_id: int, log_date: date):
        assert user_challenge_id == 7
        assert log_date == request.log_date
        return next(counts)

    async def fake_create_challenge_log(user_challenge_id: int, data: dict):
        created_payloads.append(data)
        return SimpleNamespace(
            id=1,
            user_challenge_id=user_challenge_id,
            created_at=datetime(2026, 5, 28, 9, 0, 0),
            updated_at=datetime(2026, 5, 28, 9, 0, 0),
            **data,
        )

    monkeypatch.setattr(
        challenge_service.challenge_repository, "get_user_challenge_by_id", fake_get_user_challenge_by_id
    )
    monkeypatch.setattr(challenge_service.Challenge, "get_or_none", fake_get_challenge)
    monkeypatch.setattr(challenge_service.challenge_repository, "count_challenge_logs_by_date", fake_count_logs_by_date)
    monkeypatch.setattr(challenge_service.challenge_repository, "create_challenge_log", fake_create_challenge_log)

    created = await challenge_service.create_challenge_log(7, request)

    assert created.log_date == request.log_date
    assert len(created_payloads) == 1
    with pytest.raises(HTTPException) as exc:
        await challenge_service.create_challenge_log(7, request)
    assert exc.value.status_code == 409
    assert len(created_payloads) == 1


@pytest.mark.asyncio
async def test_create_challenge_log_allows_until_daily_goal_count(monkeypatch) -> None:
    request = ChallengeLogCreateRequest(log_date=date(2026, 5, 28), is_completed=True)
    counts = iter([0, 1, 2])
    created_payloads: list[dict] = []

    async def fake_get_user_challenge_by_id(user_challenge_id: int):
        assert user_challenge_id == 8
        return SimpleNamespace(id=user_challenge_id, challenge_id=12)

    async def fake_get_challenge(id: int):
        assert id == 12
        return SimpleNamespace(target_metric="count", target_value="2")

    async def fake_count_logs_by_date(user_challenge_id: int, log_date: date):
        assert user_challenge_id == 8
        assert log_date == request.log_date
        return next(counts)

    async def fake_create_challenge_log(user_challenge_id: int, data: dict):
        created_payloads.append(data)
        return SimpleNamespace(
            id=len(created_payloads),
            user_challenge_id=user_challenge_id,
            created_at=datetime(2026, 5, 28, 9, 0, 0),
            updated_at=datetime(2026, 5, 28, 9, 0, 0),
            **data,
        )

    monkeypatch.setattr(
        challenge_service.challenge_repository, "get_user_challenge_by_id", fake_get_user_challenge_by_id
    )
    monkeypatch.setattr(challenge_service.Challenge, "get_or_none", fake_get_challenge)
    monkeypatch.setattr(challenge_service.challenge_repository, "count_challenge_logs_by_date", fake_count_logs_by_date)
    monkeypatch.setattr(challenge_service.challenge_repository, "create_challenge_log", fake_create_challenge_log)

    first = await challenge_service.create_challenge_log(8, request)
    second = await challenge_service.create_challenge_log(8, request)

    assert first.id == 1
    assert second.id == 2
    assert len(created_payloads) == 2
    with pytest.raises(HTTPException) as exc:
        await challenge_service.create_challenge_log(8, request)
    assert exc.value.status_code == 409
    assert len(created_payloads) == 2


@pytest.mark.asyncio
async def test_join_challenge_reactivates_canceled_challenge(monkeypatch) -> None:
    fixed_now = datetime(2026, 6, 1, 9, 30, 0)
    existing = SimpleNamespace(
        id=5,
        user_id=42,
        challenge_id=3,
        status=UserChallengeStatus.CANCELED,
        started_at=datetime(2026, 5, 20, 9, 0, 0),
        expected_done_at=datetime(2026, 5, 27, 9, 0, 0),
        completed_at=None,
        canceled_at=datetime(2026, 5, 28, 9, 0, 0),
    )
    updated_payload: dict[str, object] = {}

    async def fake_get_user_challenge_by_user_and_challenge(user_id: int, challenge_id: int):
        assert user_id == 42
        assert challenge_id == 3
        return existing

    async def fake_update_user_challenge(user_challenge_id: int, data: dict):
        assert user_challenge_id == existing.id
        updated_payload.update(data)
        payload = {**existing.__dict__, **data}
        return SimpleNamespace(**payload)

    async def fake_create_user_challenge(user_id: int, challenge_id: int, data: dict):
        raise AssertionError("rejoin must update the canceled row instead of creating a duplicate")

    async def fake_with_progress(user_challenge):
        return user_challenge

    async def fake_sync_challenge_reminders_for_user(user_id: int):
        return None

    monkeypatch.setattr(challenge_service, "_now", lambda: fixed_now)
    monkeypatch.setattr(
        challenge_service.challenge_repository,
        "get_user_challenge_by_user_and_challenge",
        fake_get_user_challenge_by_user_and_challenge,
    )
    monkeypatch.setattr(challenge_service.challenge_repository, "update_user_challenge", fake_update_user_challenge)
    monkeypatch.setattr(challenge_service.challenge_repository, "create_user_challenge", fake_create_user_challenge)
    monkeypatch.setattr(challenge_service, "_with_user_challenge_progress", fake_with_progress)
    monkeypatch.setattr(challenge_service, "_sync_challenge_reminders_for_user", fake_sync_challenge_reminders_for_user)

    rejoined = await challenge_service.join_challenge(user_id=42, challenge_id=3)

    assert rejoined.id == existing.id
    assert rejoined.status == UserChallengeStatus.JOINED
    assert rejoined.started_at == fixed_now
    assert rejoined.expected_done_at == fixed_now + timedelta(days=7)
    assert rejoined.canceled_at is None
    assert rejoined.completed_at is None
    assert updated_payload == {
        "status": UserChallengeStatus.JOINED,
        "started_at": fixed_now,
        "expected_done_at": fixed_now + timedelta(days=7),
        "completed_at": None,
        "canceled_at": None,
    }
