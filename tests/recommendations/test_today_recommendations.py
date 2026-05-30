from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.models.analysis import AnalysisType, RiskLevel
from app.models.challenges import UserChallengeStatus
from app.services import recommendations as recommendation_service


async def _empty_list(*args, **kwargs):
    return []


@pytest.mark.asyncio
async def test_today_recommendations_use_latest_analysis_and_user_context(monkeypatch) -> None:
    async def fake_analysis_results(user_id: int):
        assert user_id == 42
        return [
            SimpleNamespace(
                analysis_type=AnalysisType.DIABETES,
                risk_level=RiskLevel.HIGH,
                risk_score=Decimal("0.72"),
            )
        ]

    async def fake_health_record(user_id: int):
        assert user_id == 42
        return SimpleNamespace(
            systolic_bp=128,
            diastolic_bp=76,
            fasting_glucose=104,
            walking_days_per_week=2,
        )

    async def fake_diet_records(user_id: int, limit: int):
        assert user_id == 42
        assert limit == 5
        return [SimpleNamespace(created_at=datetime.now(UTC), diet_score=82)]

    async def fake_user_challenges(user_id: int, limit: int):
        assert user_id == 42
        assert limit == 20
        return [SimpleNamespace(status=UserChallengeStatus.JOINED, today_completed=True)]

    monkeypatch.setattr(recommendation_service.analysis_service, "list_latest_analysis_results", fake_analysis_results)
    monkeypatch.setattr(recommendation_service.health_service, "get_latest_health_record", fake_health_record)
    monkeypatch.setattr(recommendation_service.diet_service, "list_diet_records", fake_diet_records)
    monkeypatch.setattr(recommendation_service.challenge_service, "list_user_challenges", fake_user_challenges)

    result = await recommendation_service.get_today_recommendations(user_id=42)

    assert len(result["items"]) <= 3
    assert result["items"][0]["related_disease"] == AnalysisType.DIABETES.value
    assert result["items"][0]["action_type"] == "movement"
    assert "위험 관리" in result["items"][0]["reason"]


@pytest.mark.asyncio
async def test_today_recommendations_are_deterministic_for_same_inputs(monkeypatch) -> None:
    async def fake_analysis_results(user_id: int):
        return [
            SimpleNamespace(
                analysis_type=AnalysisType.OBESITY, risk_level=RiskLevel.MEDIUM, risk_score=Decimal("0.68")
            ),
            SimpleNamespace(
                analysis_type=AnalysisType.HYPERTENSION,
                risk_level=RiskLevel.MEDIUM,
                risk_score=Decimal("0.41"),
            ),
        ]

    async def fake_health_record(user_id: int):
        return SimpleNamespace(
            systolic_bp=135,
            diastolic_bp=82,
            fasting_glucose=95,
            walking_days_per_week=1,
        )

    monkeypatch.setattr(recommendation_service.analysis_service, "list_latest_analysis_results", fake_analysis_results)
    monkeypatch.setattr(recommendation_service.health_service, "get_latest_health_record", fake_health_record)
    monkeypatch.setattr(recommendation_service.diet_service, "list_diet_records", _empty_list)
    monkeypatch.setattr(recommendation_service.challenge_service, "list_user_challenges", _empty_list)

    first = await recommendation_service.get_today_recommendations(user_id=7)
    second = await recommendation_service.get_today_recommendations(user_id=7)

    assert first == second
    assert [item["priority"] for item in first["items"]] == sorted(item["priority"] for item in first["items"])


@pytest.mark.asyncio
async def test_today_recommendations_return_general_fallback_without_records(monkeypatch) -> None:
    async def fake_health_record(user_id: int):
        return None

    monkeypatch.setattr(recommendation_service.analysis_service, "list_latest_analysis_results", _empty_list)
    monkeypatch.setattr(recommendation_service.health_service, "get_latest_health_record", fake_health_record)
    monkeypatch.setattr(recommendation_service.diet_service, "list_diet_records", _empty_list)
    monkeypatch.setattr(recommendation_service.challenge_service, "list_user_challenges", _empty_list)

    result = await recommendation_service.get_today_recommendations(user_id=1)

    assert 1 <= len(result["items"]) <= 3
    assert all("진단" not in item["title"] for item in result["items"])
    assert all(item["related_disease"] for item in result["items"])
