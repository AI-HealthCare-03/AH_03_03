from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.dtos.chatbot import ChatbotAskRequest, ChatbotContextType
from app.models.analysis import AnalysisType, RiskLevel
from app.models.challenges import ChallengeCategory, ChallengeStatus, ChallengeTargetDisease
from app.services import chatbot as chatbot_service


def _analysis_result(*, user_id: int = 42, result_id: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        id=result_id,
        user_id=user_id,
        analysis_type=AnalysisType.HYPERTENSION,
        risk_level=RiskLevel.CAUTION,
        summary="혈압 관리가 필요합니다.",
    )


def _challenge(*, challenge_id: int = 3, status: ChallengeStatus = ChallengeStatus.ACTIVE) -> SimpleNamespace:
    return SimpleNamespace(
        id=challenge_id,
        title="염분 줄이기 챌린지",
        category=ChallengeCategory.BLOOD_PRESSURE,
        target_disease=ChallengeTargetDisease.HYPERTENSION,
        duration_days=7,
        caution_message="무리하지 말고 진행하세요.",
        status=status,
    )


@pytest.mark.asyncio
async def test_analysis_context_loader_uses_target_id_with_owner_check(monkeypatch) -> None:
    async def fake_get_analysis_result(result_id: int):
        assert result_id == 7
        return _analysis_result()

    async def fake_list_analysis_factors(result_id: int):
        assert result_id == 7
        return [
            SimpleNamespace(factor_name="나트륨", factor_value="높음", direction="NEGATIVE"),
            SimpleNamespace(factor_name="활동량", factor_value="낮음", direction="NEGATIVE"),
        ]

    async def fake_list_recommendations(user_id: int, limit: int):
        assert user_id == 42
        assert limit == 3
        return [
            SimpleNamespace(
                challenge_id=3,
                reason="혈압 관리에 도움이 될 수 있습니다.",
                challenge=_challenge(),
            )
        ]

    monkeypatch.setattr(chatbot_service.analysis_service, "get_analysis_result", fake_get_analysis_result)
    monkeypatch.setattr(chatbot_service.analysis_service, "list_analysis_factors", fake_list_analysis_factors)
    monkeypatch.setattr(
        chatbot_service.challenge_service,
        "list_challenge_recommendations",
        fake_list_recommendations,
    )

    context = await chatbot_service.load_analysis_chatbot_context(user_id=42, target_id=7)

    assert context["status"] == "loaded"
    assert "혈압 관리" in context["safe_context_text"]
    assert "나트륨" in context["safe_context_text"]
    assert "염분 줄이기 챌린지" in context["safe_context_text"]
    assert "analysis_result_id" not in str(context)


@pytest.mark.asyncio
async def test_analysis_context_loader_uses_latest_when_target_id_missing(monkeypatch) -> None:
    async def fake_list_results(user_id: int, limit: int):
        assert user_id == 42
        assert limit == 1
        return [_analysis_result(result_id=9)]

    async def fake_list_factors(result_id: int):
        assert result_id == 9
        return []

    async def fake_list_recommendations(user_id: int, limit: int):
        return []

    monkeypatch.setattr(chatbot_service.analysis_service, "list_analysis_results", fake_list_results)
    monkeypatch.setattr(chatbot_service.analysis_service, "list_analysis_factors", fake_list_factors)
    monkeypatch.setattr(chatbot_service.challenge_service, "list_challenge_recommendations", fake_list_recommendations)

    context = await chatbot_service.load_analysis_chatbot_context(user_id=42, target_id=None)

    assert context["status"] == "loaded"
    assert "분석 결과 요약" in context["safe_context_text"]


@pytest.mark.asyncio
async def test_analysis_context_loader_falls_back_for_other_user_target(monkeypatch) -> None:
    async def fake_get_analysis_result(result_id: int):
        return _analysis_result(user_id=99)

    monkeypatch.setattr(chatbot_service.analysis_service, "get_analysis_result", fake_get_analysis_result)

    context = await chatbot_service.load_analysis_chatbot_context(user_id=42, target_id=7)

    assert context["status"] == "fallback"
    assert "확인할 수 없어" in context["safe_context_text"]


@pytest.mark.asyncio
async def test_diet_context_loader_uses_target_record_without_llm_rewrite(monkeypatch) -> None:
    record = SimpleNamespace(
        id=100,
        user_id=42,
        detected_foods=[{"name": "비빔밥"}],
    )

    async def fake_get_diet_record(diet_record_id: int):
        assert diet_record_id == 100
        return record

    async def fake_latest_health_record(user_id: int):
        return None

    async def fake_list_analysis_results(user_id: int, limit: int):
        return []

    async def fake_list_active_challenges(limit: int):
        return [_challenge()]

    def fake_build_context_response(**kwargs):
        assert kwargs["diet_record"] is record
        return {
            "nutrition_findings": [{"issue_key": "sodium_high", "message": "나트륨이 높은 후보가 있어요."}],
            "disease_context": [{"label": "혈압 관리"}],
            "recommended_challenges": [{"challenge_id": 3, "title": "염분 줄이기 챌린지", "reason": "짠맛 줄이기"}],
            "rag_comment": {"summary": "국물과 소스를 줄여보세요."},
        }

    monkeypatch.setattr(chatbot_service.diet_service, "get_diet_record", fake_get_diet_record)
    monkeypatch.setattr(
        chatbot_service.health_repository, "get_latest_health_record_by_user", fake_latest_health_record
    )
    monkeypatch.setattr(
        chatbot_service.analysis_repository,
        "list_analysis_results_by_user",
        fake_list_analysis_results,
    )
    monkeypatch.setattr(chatbot_service.challenge_service, "list_active_challenges", fake_list_active_challenges)
    monkeypatch.setattr(
        chatbot_service.diet_recommendations,
        "build_diet_health_recommendation_context_response",
        fake_build_context_response,
    )

    context = await chatbot_service.load_diet_chatbot_context(user_id=42, target_id=100)

    assert context["status"] == "loaded"
    assert "비빔밥" in context["safe_context_text"]
    assert "염분 줄이기 챌린지" in context["safe_context_text"]


@pytest.mark.asyncio
async def test_challenge_context_loader_excludes_inactive_target(monkeypatch) -> None:
    async def fake_get_challenge(challenge_id: int):
        return _challenge(challenge_id=challenge_id, status=ChallengeStatus.INACTIVE)

    monkeypatch.setattr(chatbot_service.challenge_service, "get_challenge", fake_get_challenge)

    context = await chatbot_service.load_challenge_chatbot_context(user_id=42, target_id=3)

    assert context["status"] == "fallback"
    assert "비활성 챌린지" in context["safe_context_text"]


@pytest.mark.asyncio
async def test_chatbot_service_injects_domain_context_into_graph(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_load_context(user_id: int | None, context_type: str, target_id: int | None):
        assert user_id == 42
        assert context_type == "ANALYSIS"
        assert target_id == 7
        return {
            "context_type": "ANALYSIS",
            "status": "loaded",
            "safe_context_text": "분석 결과 요약 context입니다.",
        }

    async def fake_run_chatbot_graph_async(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            answer="분석 결과를 바탕으로 생활습관을 점검해 보세요.",
            source="rag_llm",
            recommended_actions=[],
            caution_message="이 정보는 진단이 아니며 건강관리 참고용입니다.",
            trace_metadata={},
        )

    monkeypatch.setattr(chatbot_service, "load_chatbot_domain_context", fake_load_context)
    monkeypatch.setattr(chatbot_service, "run_chatbot_graph_async", fake_run_chatbot_graph_async)
    monkeypatch.setattr(chatbot_service.config, "CHATBOT_USE_REAL_LLM", False)

    response, _ = await chatbot_service.ask_chatbot_with_runtime_trace(
        ChatbotAskRequest(
            message="내 결과에서 뭐부터 봐야 해?",
            context_type=ChatbotContextType.ANALYSIS,
            target_id=7,
        ),
        user_id=42,
    )

    assert response.source == "rag_llm"
    assert captured["user_context"]["target_id"] == 7
    assert captured["user_context"]["domain_context"]["safe_context_text"] == "분석 결과 요약 context입니다."


def test_domain_context_helpers_do_not_expose_internal_rag_terms() -> None:
    from ai_runtime.llm.graph.nodes import _context_text_with_domain_context

    rendered = _context_text_with_domain_context(
        "공식 건강정보 context",
        {
            "domain_context": {
                "status": "loaded",
                "safe_context_text": "분석 결과 요약 context입니다. 혈압 관리가 필요합니다.",
            }
        },
    )

    assert "분석 결과 요약" in rendered
    assert all(term not in rendered for term in ["chunk_key", "score", "embedding", "similarity"])
