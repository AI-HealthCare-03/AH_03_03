from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.core import config
from app.dtos.challenges import ChallengeLogCreateRequest
from app.models.challenges import UserChallengeStatus
from app.services import challenges as challenge_service


class _AwaitableList:
    def __init__(self, items: list):
        self.items = items

    def only(self, *args):
        return self

    def __await__(self):
        async def _result():
            return self.items

        return _result().__await__()


@pytest.mark.asyncio
async def test_join_challenge_stores_started_and_expected_done_at(monkeypatch) -> None:
    captured_payload: dict = {}
    started_at = datetime(2026, 5, 30, 9, 0, tzinfo=config.TIMEZONE)

    async def fake_get_existing(user_id: int, challenge_id: int):
        return None

    async def fake_create_user_challenge(user_id: int, challenge_id: int, data: dict):
        captured_payload.update(data)
        return SimpleNamespace(
            id=1,
            user_id=user_id,
            challenge_id=challenge_id,
            status=UserChallengeStatus.JOINED,
            completed_at=None,
            canceled_at=None,
            created_at=started_at,
            updated_at=started_at,
            **data,
        )

    async def fake_with_progress(user_challenge):
        return user_challenge

    monkeypatch.setattr(
        challenge_service.challenge_repository,
        "get_user_challenge_by_user_and_challenge",
        fake_get_existing,
    )
    monkeypatch.setattr(challenge_service.challenge_repository, "create_user_challenge", fake_create_user_challenge)
    monkeypatch.setattr(challenge_service, "_now", lambda: started_at)
    monkeypatch.setattr(challenge_service, "_with_user_challenge_progress", fake_with_progress)

    joined = await challenge_service.join_challenge(user_id=10, challenge_id=20)

    assert joined.started_at == started_at
    assert joined.expected_done_at == started_at + timedelta(days=1)
    assert captured_payload["started_at"] == started_at
    assert captured_payload["expected_done_at"] == started_at + timedelta(days=1)


@pytest.mark.asyncio
async def test_completed_log_gets_completed_at(monkeypatch) -> None:
    now = datetime(2026, 5, 30, 10, 0, tzinfo=config.TIMEZONE)
    request = ChallengeLogCreateRequest(log_date=date(2026, 5, 30), is_completed=True)
    captured_payload: dict = {}

    async def fake_get_user_challenge_by_id(user_challenge_id: int):
        return SimpleNamespace(id=user_challenge_id, user_id=7, challenge_id=20)

    async def fake_get_challenge(id: int):
        return SimpleNamespace(target_metric="minutes", target_value="30")

    async def fake_count_logs_by_date(user_challenge_id: int, log_date: date):
        return 0

    async def fake_create_challenge_log(user_challenge_id: int, data: dict):
        captured_payload.update(data)
        return SimpleNamespace(
            id=99,
            user_challenge_id=user_challenge_id,
            created_at=now,
            updated_at=now,
            **data,
        )

    monkeypatch.setattr(
        challenge_service.challenge_repository, "get_user_challenge_by_id", fake_get_user_challenge_by_id
    )
    monkeypatch.setattr(challenge_service.Challenge, "get_or_none", fake_get_challenge)
    monkeypatch.setattr(challenge_service.challenge_repository, "count_challenge_logs_by_date", fake_count_logs_by_date)
    monkeypatch.setattr(challenge_service.challenge_repository, "create_challenge_log", fake_create_challenge_log)
    monkeypatch.setattr(challenge_service, "_now", lambda: now)

    log = await challenge_service.create_challenge_log(5, request)

    assert log.completed_at == now
    assert log.completed_date == date(2026, 5, 30)
    assert captured_payload["completed_at"] == now


@pytest.mark.asyncio
async def test_started_at_without_completed_at_is_not_completed_progress(monkeypatch) -> None:
    started_at = datetime(2026, 5, 30, 9, 0, tzinfo=config.TIMEZONE)
    user_challenge = SimpleNamespace(
        id=1,
        challenge_id=2,
        status=UserChallengeStatus.COMPLETED,
        started_at=started_at,
        expected_done_at=started_at + timedelta(days=1),
        completed_at=None,
    )

    async def fake_get_challenge(id: int):
        return SimpleNamespace(duration_days=7, target_metric="minutes", target_value="30")

    monkeypatch.setattr(challenge_service.Challenge, "get_or_none", fake_get_challenge)
    monkeypatch.setattr(challenge_service.ChallengeLog, "filter", lambda **kwargs: _AwaitableList([]))

    result = await challenge_service._with_user_challenge_progress(user_challenge)

    assert result.is_completed is False
    assert result.progress == 0


@pytest.mark.asyncio
async def test_completed_at_is_only_completed_progress_signal(monkeypatch) -> None:
    completed_at = datetime(2026, 5, 30, 12, 0, tzinfo=config.TIMEZONE)
    user_challenge = SimpleNamespace(
        id=1,
        challenge_id=2,
        status=UserChallengeStatus.JOINED,
        started_at=datetime(2026, 5, 30, 9, 0, tzinfo=config.TIMEZONE),
        expected_done_at=datetime(2026, 5, 31, 9, 0, tzinfo=config.TIMEZONE),
        completed_at=completed_at,
    )

    async def fake_get_challenge(id: int):
        return SimpleNamespace(duration_days=7, target_metric="minutes", target_value="30")

    monkeypatch.setattr(challenge_service.Challenge, "get_or_none", fake_get_challenge)
    monkeypatch.setattr(challenge_service.ChallengeLog, "filter", lambda **kwargs: _AwaitableList([]))

    result = await challenge_service._with_user_challenge_progress(user_challenge)

    assert result.is_completed is True
    assert result.completed_date == date(2026, 5, 30)
    assert result.progress == 100


@pytest.mark.asyncio
async def test_calendar_uses_kst_day_range_without_next_day_in_progress_mix(monkeypatch) -> None:
    started_at = datetime(2026, 5, 30, 23, 30, tzinfo=config.TIMEZONE)
    user_challenge = SimpleNamespace(
        id=1,
        user_id=3,
        challenge_id=2,
        status=UserChallengeStatus.JOINED,
        started_at=started_at,
        expected_done_at=started_at + timedelta(days=1),
        completed_at=None,
        canceled_at=None,
    )

    async def fake_started_between(user_id: int, started_at: datetime, ended_before: datetime):
        if started_at.date() == date(2026, 5, 30):
            return [user_challenge]
        return []

    async def fake_completed_between(user_id: int, completed_at: datetime, ended_before: datetime):
        return []

    monkeypatch.setattr(
        challenge_service.challenge_repository, "list_user_challenges_started_between", fake_started_between
    )
    monkeypatch.setattr(
        challenge_service.challenge_repository, "list_challenge_logs_completed_between", fake_completed_between
    )
    monkeypatch.setattr(
        challenge_service.Challenge, "filter", lambda **kwargs: _AwaitableList([SimpleNamespace(id=2, title="걷기")])
    )
    monkeypatch.setattr(challenge_service, "_now", lambda: datetime(2026, 5, 30, 23, 40, tzinfo=config.TIMEZONE))

    may_30 = await challenge_service.get_challenge_calendar(user_id=3, target_date=date(2026, 5, 30))
    may_31 = await challenge_service.get_challenge_calendar(user_id=3, target_date=date(2026, 5, 31))

    assert may_30["items"][0]["status"] == "IN_PROGRESS"
    assert may_30["items"][0]["started_date"] == date(2026, 5, 30)
    assert may_31["items"] == []


@pytest.mark.asyncio
async def test_calendar_completed_items_are_filtered_by_completed_at(monkeypatch) -> None:
    started_at = datetime(2026, 5, 30, 23, 30, tzinfo=config.TIMEZONE)
    completed_at = datetime(2026, 5, 31, 0, 10, tzinfo=config.TIMEZONE)
    user_challenge = SimpleNamespace(
        id=1,
        user_id=3,
        challenge_id=2,
        status=UserChallengeStatus.JOINED,
        started_at=started_at,
        expected_done_at=started_at + timedelta(days=1),
        completed_at=None,
        canceled_at=None,
    )
    log = SimpleNamespace(
        id=9,
        user_challenge_id=1,
        user_challenge=user_challenge,
        log_date=date(2026, 5, 30),
        is_completed=True,
        completed_at=completed_at,
    )

    async def fake_started_between(user_id: int, started_at: datetime, ended_before: datetime):
        return []

    async def fake_completed_between(user_id: int, completed_at: datetime, ended_before: datetime):
        if completed_at.date() == date(2026, 5, 31):
            return [log]
        return []

    monkeypatch.setattr(
        challenge_service.challenge_repository, "list_user_challenges_started_between", fake_started_between
    )
    monkeypatch.setattr(
        challenge_service.challenge_repository, "list_challenge_logs_completed_between", fake_completed_between
    )
    monkeypatch.setattr(
        challenge_service.Challenge, "filter", lambda **kwargs: _AwaitableList([SimpleNamespace(id=2, title="걷기")])
    )

    may_30 = await challenge_service.get_challenge_calendar(user_id=3, target_date=date(2026, 5, 30))
    may_31 = await challenge_service.get_challenge_calendar(user_id=3, target_date=date(2026, 5, 31))

    assert may_30["items"] == []
    assert may_31["items"][0]["status"] == "COMPLETED"
    assert may_31["items"][0]["completed_date"] == date(2026, 5, 31)
