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

    async def fake_list_active_challenges(limit: int):
        assert limit == 10
        return [
            SimpleNamespace(
                id=1,
                title="하루 30분 걷기",
                description="가벼운 걷기 습관",
                category="EXERCISE",
                challenge_type="COMMON",
                target_disease="COMMON",
                difficulty="EASY",
                target_metric="minutes",
                target_value="30",
                caution_message=None,
                contraindication_message=None,
                duration_days=7,
                status="ACTIVE",
                created_at=created_at,
                updated_at=created_at,
            )
        ]

    async def fake_list_user_challenges(user_id: int, limit: int):
        assert user_id == 42
        assert limit == 10
        return [
            SimpleNamespace(
                id=2,
                user_id=42,
                challenge_id=1,
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
        ]

    monkeypatch.setattr(dashboard_service.challenge_service, "list_active_challenges", fake_list_active_challenges)
    monkeypatch.setattr(dashboard_service.challenge_service, "list_user_challenges", fake_list_user_challenges)

    result = await dashboard_service.get_dashboard_challenges(user_id=42)

    assert result["active_challenges"][0]["title"] == "하루 30분 걷기"
    assert result["user_challenges"][0]["today_completed"] is True
    assert result["user_challenges"][0]["started_at"] == "2026-05-28T09:00:00"
