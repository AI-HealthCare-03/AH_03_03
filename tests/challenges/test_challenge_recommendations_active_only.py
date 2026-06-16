from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from app.dtos.challenges import ChallengeRecommendationCreateRequest
from app.models.analysis import AnalysisMode, AnalysisType, RiskLevel
from app.models.challenges import ChallengeCategory, ChallengeStatus, ChallengeTargetDisease
from app.services import analysis as analysis_service
from app.services import challenges as challenge_service


def _challenge(
    challenge_id: int,
    *,
    title: str,
    category: ChallengeCategory = ChallengeCategory.DIET,
    status: ChallengeStatus = ChallengeStatus.ACTIVE,
    target_disease: ChallengeTargetDisease = ChallengeTargetDisease.GENERAL,
) -> SimpleNamespace:
    now = datetime(2026, 6, 16, 9, 0, 0)
    return SimpleNamespace(
        id=challenge_id,
        title=title,
        description=title,
        category=category,
        challenge_type="COMMON",
        target_disease=target_disease,
        difficulty="EASY",
        target_metric=None,
        target_value=None,
        caution_message=None,
        contraindication_message=None,
        duration_days=7,
        status=status,
        created_at=now,
        updated_at=now,
    )


def _recommendation(
    recommendation_id: int,
    challenge: SimpleNamespace,
    *,
    reason: str = "추천 이유",
) -> SimpleNamespace:
    now = datetime(2026, 6, 16, 9, 0, 0)
    return SimpleNamespace(
        id=recommendation_id,
        user_id=42,
        analysis_result_id=7,
        challenge_id=challenge.id,
        reason=reason,
        priority=1,
        is_selected=False,
        created_at=now,
        updated_at=now,
        challenge=challenge,
    )


@pytest.mark.asyncio
async def test_list_recommendations_excludes_inactive_and_deduplicates(monkeypatch) -> None:
    active = _challenge(1, title="혈압 기록 챌린지", category=ChallengeCategory.BLOOD_PRESSURE)
    inactive = _challenge(99, title="비활성 챌린지", status=ChallengeStatus.INACTIVE)
    fallback = _challenge(2, title="식사일지 작성 챌린지", category=ChallengeCategory.MONITORING)

    async def fake_list_recommendations(user_id: int, limit: int, offset: int):
        assert user_id == 42
        assert limit > 2
        assert offset == 0
        return [
            _recommendation(10, inactive),
            _recommendation(11, active, reason="첫 추천"),
            _recommendation(12, active, reason="중복 추천"),
        ]

    async def fake_list_active_challenges(**kwargs):
        assert kwargs["limit"] == 100
        return [active, fallback]

    monkeypatch.setattr(
        challenge_service.challenge_repository,
        "list_challenge_recommendations",
        fake_list_recommendations,
    )
    monkeypatch.setattr(challenge_service, "list_active_challenges", fake_list_active_challenges)

    recommendations = await challenge_service.list_challenge_recommendations(42, limit=2)

    assert [item.challenge_id for item in recommendations] == [active.id, fallback.id]
    assert len({item.challenge_id for item in recommendations}) == len(recommendations)
    assert all(item.reason for item in recommendations)
    assert inactive.id not in {item.challenge_id for item in recommendations}


@pytest.mark.asyncio
async def test_list_recommendations_fills_active_fallback_when_only_inactive_exists(monkeypatch) -> None:
    inactive = _challenge(99, title="비활성 챌린지", status=ChallengeStatus.INACTIVE)
    fallback = _challenge(2, title="식사일지 작성 챌린지", category=ChallengeCategory.MONITORING)

    async def fake_list_recommendations(user_id: int, limit: int, offset: int):
        return [_recommendation(10, inactive)]

    async def fake_list_active_challenges(**kwargs):
        return [fallback]

    monkeypatch.setattr(
        challenge_service.challenge_repository,
        "list_challenge_recommendations",
        fake_list_recommendations,
    )
    monkeypatch.setattr(challenge_service, "list_active_challenges", fake_list_active_challenges)

    recommendations = await challenge_service.list_challenge_recommendations(42, limit=1)

    assert len(recommendations) == 1
    assert recommendations[0].challenge_id == fallback.id
    assert recommendations[0].id < 0
    assert recommendations[0].reason


@pytest.mark.asyncio
async def test_create_recommendation_replaces_inactive_challenge_with_active_fallback(monkeypatch) -> None:
    inactive = _challenge(99, title="비활성 챌린지", status=ChallengeStatus.INACTIVE)
    active = _challenge(3, title="염분 줄이기 챌린지", category=ChallengeCategory.BLOOD_PRESSURE)
    captured: dict[str, object] = {}

    async def fake_get_challenge_by_id(challenge_id: int):
        assert challenge_id == inactive.id
        return inactive

    async def fake_list_active_challenges(**kwargs):
        return [active]

    async def fake_create_challenge_recommendation(
        user_id: int, analysis_result_id: int, challenge_id: int, data: dict
    ):
        captured.update(
            user_id=user_id,
            analysis_result_id=analysis_result_id,
            challenge_id=challenge_id,
            data=data,
        )
        return _recommendation(20, active, reason=data["reason"])

    monkeypatch.setattr(challenge_service.challenge_repository, "get_challenge_by_id", fake_get_challenge_by_id)
    monkeypatch.setattr(challenge_service, "list_active_challenges", fake_list_active_challenges)
    monkeypatch.setattr(
        challenge_service.challenge_repository,
        "create_challenge_recommendation",
        fake_create_challenge_recommendation,
    )

    recommendation = await challenge_service.create_challenge_recommendation(
        user_id=42,
        analysis_result_id=7,
        challenge_id=inactive.id,
        request=ChallengeRecommendationCreateRequest(challenge_id=inactive.id, analysis_result_id=7),
    )

    assert recommendation.challenge_id == active.id
    assert captured["challenge_id"] == active.id
    assert captured["data"]["reason"]


@pytest.mark.asyncio
async def test_analysis_recommendation_uses_active_deterministic_mapping(monkeypatch) -> None:
    active_challenges = [
        _challenge(1, title="건강 습관 챌린지", category=ChallengeCategory.HABIT),
        _challenge(
            2,
            title="염분 줄이기 챌린지",
            category=ChallengeCategory.BLOOD_PRESSURE,
            target_disease=ChallengeTargetDisease.HYPERTENSION,
        ),
    ]
    captured: dict[str, object] = {}

    async def fake_list_active_challenges(**kwargs):
        return active_challenges

    async def fake_create_challenge_recommendation(user_id: int, analysis_result_id: int, challenge_id: int, request):
        captured.update(user_id=user_id, analysis_result_id=analysis_result_id, challenge_id=challenge_id)
        return SimpleNamespace(id=55)

    monkeypatch.setattr(analysis_service.challenge_service, "list_active_challenges", fake_list_active_challenges)
    monkeypatch.setattr(
        analysis_service.challenge_service,
        "create_challenge_recommendation",
        fake_create_challenge_recommendation,
    )

    result = SimpleNamespace(
        id=7,
        analysis_type=AnalysisType.HYPERTENSION,
        risk_level=RiskLevel.HIGH_CAUTION,
        analysis_mode=AnalysisMode.BASIC,
    )

    recommendation_ids = await analysis_service._create_challenge_recommendations(user_id=42, result=result)

    assert recommendation_ids == [55]
    assert captured["challenge_id"] == 2
