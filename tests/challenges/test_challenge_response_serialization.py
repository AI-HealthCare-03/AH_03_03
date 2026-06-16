from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from app.apis.v1.challenge_routers import _challenge_log_payload
from app.dtos.challenges import ChallengeActionResponse
from app.services import dashboard as dashboard_service


def test_challenge_action_result_uses_serializable_payload() -> None:
    payload = _challenge_log_payload(
        SimpleNamespace(
            id=7,
            user_challenge_id=3,
            log_date=date(2026, 5, 28),
            is_completed=True,
            memo="오늘 수행 완료",
            created_at=datetime(2026, 5, 28, 9, 0, 0),
            updated_at=datetime(2026, 5, 28, 9, 0, 0),
        )
    )

    response = ChallengeActionResponse(message="ok", result=payload)

    assert response.result["log_date"] == "2026-05-28"
    assert "ChallengeLog" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_dashboard_challenges_are_serialized_before_response(monkeypatch) -> None:
    created_at = datetime(2026, 5, 28, 9, 0, 0)
    active_challenge = _challenge(
        challenge_id=1,
        title="하루 30분 걷기",
        status="ACTIVE",
        created_at=created_at,
    )

    async def fake_list_active_challenges(limit: int):
        assert limit == 10
        return [active_challenge]

    async def fake_list_user_challenges(user_id: int, limit: int):
        assert user_id == 42
        assert limit == 10
        return [_user_challenge(user_challenge_id=2, user_id=user_id, challenge_id=1, created_at=created_at)]

    async def fake_get_challenge(challenge_id: int):
        assert challenge_id == 1
        return active_challenge

    monkeypatch.setattr(dashboard_service.challenge_service, "list_active_challenges", fake_list_active_challenges)
    monkeypatch.setattr(dashboard_service.challenge_service, "list_user_challenges", fake_list_user_challenges)
    monkeypatch.setattr(dashboard_service.challenge_service, "get_challenge", fake_get_challenge)

    result = await dashboard_service.get_dashboard_challenges(user_id=42)

    assert result["active_challenges"][0]["title"] == "하루 30분 걷기"
    assert result["user_challenges"][0]["challenge_title"] == "하루 30분 걷기"
    assert result["user_challenges"][0]["today_completed"] is True
    assert result["user_challenges"][0]["started_at"] == "2026-05-28T09:00:00"


@pytest.mark.asyncio
async def test_dashboard_user_challenge_includes_title_when_challenge_is_outside_active_list(monkeypatch) -> None:
    created_at = datetime(2026, 5, 28, 9, 0, 0)
    joined_challenge = _challenge(
        challenge_id=99,
        title="식사일지 작성 챌린지",
        status="ACTIVE",
        created_at=created_at,
    )

    async def fake_list_active_challenges(limit: int):
        assert limit == 10
        return [_challenge(challenge_id=1, title="하루 30분 걷기", status="ACTIVE", created_at=created_at)]

    async def fake_list_user_challenges(user_id: int, limit: int):
        assert user_id == 42
        assert limit == 10
        return [_user_challenge(user_challenge_id=3, user_id=user_id, challenge_id=99, created_at=created_at)]

    async def fake_get_challenge(challenge_id: int):
        assert challenge_id == 99
        return joined_challenge

    monkeypatch.setattr(dashboard_service.challenge_service, "list_active_challenges", fake_list_active_challenges)
    monkeypatch.setattr(dashboard_service.challenge_service, "list_user_challenges", fake_list_user_challenges)
    monkeypatch.setattr(dashboard_service.challenge_service, "get_challenge", fake_get_challenge)

    result = await dashboard_service.get_dashboard_challenges(user_id=42)

    assert [item["id"] for item in result["active_challenges"]] == [1]
    assert result["user_challenges"][0]["challenge_id"] == 99
    assert result["user_challenges"][0]["challenge_title"] == "식사일지 작성 챌린지"
    assert result["user_challenges"][0]["challenge_status"] == "ACTIVE"


@pytest.mark.asyncio
async def test_dashboard_user_challenge_includes_inactive_challenge_title(monkeypatch) -> None:
    created_at = datetime(2026, 5, 28, 9, 0, 0)
    inactive_challenge = _challenge(
        challenge_id=77,
        title="종료된 수분 섭취 챌린지",
        status="INACTIVE",
        created_at=created_at,
    )

    async def fake_list_active_challenges(limit: int):
        assert limit == 10
        return []

    async def fake_list_user_challenges(user_id: int, limit: int):
        assert user_id == 42
        assert limit == 10
        return [_user_challenge(user_challenge_id=4, user_id=user_id, challenge_id=77, created_at=created_at)]

    async def fake_get_challenge(challenge_id: int):
        assert challenge_id == 77
        return inactive_challenge

    monkeypatch.setattr(dashboard_service.challenge_service, "list_active_challenges", fake_list_active_challenges)
    monkeypatch.setattr(dashboard_service.challenge_service, "list_user_challenges", fake_list_user_challenges)
    monkeypatch.setattr(dashboard_service.challenge_service, "get_challenge", fake_get_challenge)

    result = await dashboard_service.get_dashboard_challenges(user_id=42)

    assert result["active_challenges"] == []
    assert result["user_challenges"][0]["challenge_title"] == "종료된 수분 섭취 챌린지"
    assert result["user_challenges"][0]["challenge_status"] == "INACTIVE"


def _challenge(*, challenge_id: int, title: str, status: str, created_at: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        id=challenge_id,
        title=title,
        description="가벼운 습관 챌린지",
        category="EXERCISE",
        challenge_type="COMMON",
        target_disease="COMMON",
        difficulty="EASY",
        target_metric="minutes",
        target_value="30",
        caution_message=None,
        contraindication_message=None,
        duration_days=7,
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )


def _user_challenge(
    *,
    user_challenge_id: int,
    user_id: int,
    challenge_id: int,
    created_at: datetime,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=user_challenge_id,
        user_id=user_id,
        challenge_id=challenge_id,
        status="JOINED",
        started_at=created_at,
        completed_at=None,
        canceled_at=None,
        completed_days=1,
        progress=14,
        today_completed=True,
        today_completed_count=1,
        daily_goal_count=1,
        duration_days=7,
        created_at=created_at,
        updated_at=created_at,
    )
