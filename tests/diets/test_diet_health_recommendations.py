from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ai_runtime.llm.rag.retriever import RagRetrievalResult, RetrievedDocument
from app.apis.v1 import diet_routers
from app.apis.v1.dependencies import get_request_user
from app.core.config import Config
from app.main import app
from app.models.analysis import AnalysisType, RiskLevel
from app.services import diet_recommendations as service


def _challenge(challenge_id: int, title: str) -> SimpleNamespace:
    return SimpleNamespace(id=challenge_id, title=title)


ACTIVE_CHALLENGES = [
    _challenge(1, "염분 빼볼까염 챌린지"),
    _challenge(2, "식사일지 작성 챌린지"),
    _challenge(3, "추가설탕 안녕이당 챌린지"),
    _challenge(4, "탄수화물 체인지 챌린지"),
    _challenge(5, "밥최고NO 채고밥YES 챌린지"),
    _challenge(6, "기름기 쫙빼기 챌린지"),
    _challenge(7, "건강식탁 챌린지"),
    _challenge(8, "Goodbye 야식 챌린지"),
    _challenge(9, "Goodbye 폭식 챌린지"),
    _challenge(10, "2020 식사 챌린지"),
    _challenge(11, "철분 반찬 추가 챌린지"),
    _challenge(12, "식이섬유 먹어유 챌린지"),
    _challenge(13, "비타민C 함께 먹기 챌린지"),
    _challenge(14, "철분 흡수 방해 식품 줄이기 챌린지"),
    _challenge(15, "30일, 금주 챌린지"),
    _challenge(16, "30일, 하루 두 잔 절주 챌린지"),
    _challenge(17, "30일 폭음 피하기 챌린지"),
    _challenge(18, "일정한 삼시세끼 챌린지"),
]


class FakeVectorRetriever:
    def __init__(self, *, documents: list[RetrievedDocument] | None = None, fail: bool = False) -> None:
        self.documents = documents or []
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, **kwargs: Any) -> RagRetrievalResult:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("vector unavailable")
        return RagRetrievalResult(documents=self.documents, strategy="vector")


def _diet(
    nutrition: dict[str, Any],
    *,
    food_name: str = "테스트 식단",
    description: str | None = None,
    meal_type: str | None = "LUNCH",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=100,
        user_id=10,
        meal_type=meal_type,
        description=description,
        memo=None,
        detected_foods=[
            {
                "name": food_name,
                "original_name": food_name,
                "query_name": food_name,
                "match_metadata": {
                    "provider": "mfds",
                    "status": "matched",
                    "nutrition": {
                        **nutrition,
                        "basis_label": nutrition.get("basis_label", "100g 기준"),
                    },
                },
            }
        ],
    )


def _analysis(analysis_type: AnalysisType, risk_level: RiskLevel = RiskLevel.CAUTION) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        analysis_type=analysis_type,
        risk_level=risk_level,
        analyzed_at=datetime(2026, 6, 15, tzinfo=UTC),
    )


def _vector_document(content: str = "vector chunk content") -> RetrievedDocument:
    return RetrievedDocument(
        content=content,
        title="고혈압 식생활",
        source_name="대한고혈압학회",
        url="https://example.test/htn",
        metadata={
            "chunk_key": "rag:hypertension:section:000:chunk:0000",
            "document_key": "rag:hypertension:hypertension.md",
            "source_key": "hypertension",
            "disease_code": "HTN",
            "review_status": "candidate_unreviewed",
            "status": "candidate_unreviewed",
            "retriever_strategy": "vector",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        },
        score=0.88,
    )


def _build(
    *,
    nutrition: dict[str, Any],
    analysis_types: list[AnalysisType],
    food_name: str = "테스트 식단",
    description: str | None = None,
    meal_type: str | None = "LUNCH",
    active_challenges: list[SimpleNamespace] | None = None,
) -> dict[str, Any]:
    return service.build_diet_health_recommendation_response(
        diet_record=_diet(nutrition, food_name=food_name, description=description, meal_type=meal_type),
        analysis_results=[_analysis(item) for item in analysis_types],
        health_record=None,
        active_challenges=active_challenges or ACTIVE_CHALLENGES,
    )


@pytest.mark.parametrize(
    ("name", "nutrition", "analysis_types", "expected_issue", "expected_challenge"),
    [
        (
            "고혈압 + 나트륨 높은 식단",
            {"sodium_mg": "920 mg", "calories_kcal": 350},
            [AnalysisType.HYPERTENSION],
            "sodium_high",
            "염분 빼볼까염 챌린지",
        ),
        (
            "당뇨 + 탄수화물 높은 식단",
            {"carbohydrate_g": "78 g", "sugar_g": 18},
            [AnalysisType.DIABETES],
            "carbohydrate_high",
            "탄수화물 체인지 챌린지",
        ),
        (
            "이상지질혈증 + 지방 높은 식단",
            {"fat_g": 28, "fiber_g": 1.2},
            [AnalysisType.DYSLIPIDEMIA],
            "fat_high",
            "기름기 쫙빼기 챌린지",
        ),
        (
            "비만 + 고열량 식단",
            {"kcal": "650 kcal", "fat": 18},
            [AnalysisType.OBESITY],
            "calorie_high",
            "Goodbye 야식 챌린지",
        ),
        (
            "빈혈 + 철분/단백질 보완 필요",
            {"protein_g": 6, "iron_mg": 0.8},
            [AnalysisType.ANEMIA],
            "iron_support",
            "철분 반찬 추가 챌린지",
        ),
        (
            "지방간 + 음주/당류/지방 관리",
            {"sugar": 20, "fat": 24},
            [AnalysisType.FATTY_LIVER],
            "alcohol_liver_support",
            "30일, 금주 챌린지",
        ),
        (
            "신장질환 의심",
            {"protein_g": 35, "potassium_mg": 900},
            [AnalysisType.CHRONIC_KIDNEY_DISEASE],
            "kidney_caution",
            "식사일지 작성 챌린지",
        ),
        (
            "정상군 + 균형 유지",
            {"calories_kcal": 350, "protein_g": 18, "fiber_g": 4, "sodium_mg": 250},
            [],
            "balanced_support",
            "건강식탁 챌린지",
        ),
        (
            "HTN+DM 복합질환",
            {"sodium_mg": 880, "carbohydrate": 82},
            [AnalysisType.HYPERTENSION, AnalysisType.DIABETES],
            "sodium_high",
            "염분 빼볼까염 챌린지",
        ),
        (
            "OBE+DL 복합질환",
            {"calories": 720, "fat_g": 31},
            [AnalysisType.OBESITY, AnalysisType.DYSLIPIDEMIA],
            "calorie_high",
            "기름기 쫙빼기 챌린지",
        ),
    ],
)
def test_diet_health_recommendation_cases(
    name: str,
    nutrition: dict[str, Any],
    analysis_types: list[AnalysisType],
    expected_issue: str,
    expected_challenge: str,
) -> None:
    description = "맥주와 안주" if "지방간" in name else None
    result = _build(nutrition=nutrition, analysis_types=analysis_types, description=description)
    issue_keys = [item["issue_key"] for item in result["nutrition_findings"]]
    challenge_titles = [item["title"] for item in result["recommended_challenges"]]

    assert expected_issue in issue_keys
    assert expected_challenge in challenge_titles
    assert result["safety_notice"] == service.SAFETY_NOTICE
    assert len(result["recommended_challenges"]) <= 3
    assert "진단이나 처방이 아닌" in result["safety_notice"]
    serialized = str(result)
    assert "나트륨 과다입니다" not in serialized
    assert "단백질이 부족합니다" not in serialized
    assert "먹으면 안 됩니다" not in serialized
    if expected_issue == "kidney_caution":
        assert "의료진 상담" in serialized
        assert "단백질 제한" not in serialized
        assert "칼륨 제한" not in serialized


def test_diet_health_recommendation_keeps_only_active_title_matches() -> None:
    result = _build(
        nutrition={"sodium_mg": 900},
        analysis_types=[AnalysisType.HYPERTENSION],
        active_challenges=[_challenge(2, "식사일지 작성 챌린지")],
    )

    challenge_titles = [item["title"] for item in result["recommended_challenges"]]
    assert challenge_titles == ["식사일지 작성 챌린지"]
    assert "염분 빼볼까염 챌린지" not in challenge_titles


def test_diet_rag_comment_uses_attention_or_higher_disease_only() -> None:
    result = service.build_diet_health_recommendation_response(
        diet_record=_diet({"sugar_g": 18}, food_name="달콤한 간식"),
        analysis_results=[_analysis(AnalysisType.DIABETES, RiskLevel.LOW)],
        health_record=None,
        active_challenges=ACTIVE_CHALLENGES,
    )

    rag_comment = result["rag_comment"]
    assert rag_comment["enabled"] is True
    assert all(item["disease_code"] != "DM" for item in rag_comment["disease_comments"])
    assert all(item["disease_code"] != "DM" for item in rag_comment["evidence_sources"])
    assert any(item["disease_code"] == "DIET_NUTRITION" for item in rag_comment["evidence_sources"])


def test_diet_rag_comment_for_htn_sodium_uses_htn_and_diet_sources() -> None:
    result = _build(nutrition={"sodium_mg": 920}, analysis_types=[AnalysisType.HYPERTENSION])

    rag_comment = result["rag_comment"]
    assert rag_comment["enabled"] is True
    assert rag_comment["fallback_used"] is False
    assert any(item["disease_code"] == "HTN" for item in rag_comment["disease_comments"])
    assert any(item["disease_code"] == "HTN" for item in rag_comment["evidence_sources"])
    assert any(item["disease_code"] == "DIET_NUTRITION" for item in rag_comment["evidence_sources"])
    assert rag_comment["safety_notice"] == service.SAFETY_NOTICE


def test_diet_rag_comment_for_ckd_uses_diet_caution_when_ckd_source_is_disabled() -> None:
    result = _build(
        nutrition={"protein_g": 35, "potassium_mg": 900},
        analysis_types=[AnalysisType.CHRONIC_KIDNEY_DISEASE],
    )

    rag_comment = result["rag_comment"]
    assert rag_comment["enabled"] is True
    assert any(item["disease_code"] == "CKD" for item in rag_comment["disease_comments"])
    assert any(item["disease_code"] == "DIET_CAUTION" for item in rag_comment["evidence_sources"])
    serialized = str(rag_comment)
    assert "의료진 상담" in serialized
    assert "단백질 제한" not in serialized
    assert "칼륨 제한" not in serialized
    assert "인 제한" not in serialized


def test_diet_rag_comment_failure_keeps_rule_based_recommendation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_retrieval(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "retrieve_keyword_rag_matches", fail_retrieval)

    result = _build(nutrition={"sodium_mg": 920}, analysis_types=[AnalysisType.HYPERTENSION])

    assert "sodium_high" in [item["issue_key"] for item in result["nutrition_findings"]]
    assert result["recommended_foods"]
    assert result["rag_comment"]["fallback_used"] is True
    assert result["rag_comment"]["safety_notice"] == service.SAFETY_NOTICE


def test_diet_rag_comment_does_not_use_forbidden_phrases() -> None:
    result = _build(
        nutrition={"sodium_mg": 920, "carbohydrate_g": 80, "fat_g": 28},
        analysis_types=[AnalysisType.HYPERTENSION, AnalysisType.DIABETES, AnalysisType.DYSLIPIDEMIA],
    )

    serialized = str(result["rag_comment"])
    assert "나트륨 과다입니다" not in serialized
    assert "단백질이 부족합니다" not in serialized
    assert "당뇨 식단으로 부적절합니다" not in serialized
    assert "고혈압 식단입니다" not in serialized
    assert "이 음식을 먹으면 안 됩니다" not in serialized


def test_diet_rag_evidence_sources_do_not_expose_internal_review_status() -> None:
    result = _build(nutrition={"sodium_mg": 920}, analysis_types=[AnalysisType.HYPERTENSION])

    evidence_sources = result["rag_comment"]["evidence_sources"]
    assert evidence_sources
    assert any(source["review_status"] == "reference" for source in evidence_sources)

    serialized = json.dumps(result["rag_comment"], ensure_ascii=False)
    assert "candidate_unreviewed" not in serialized
    assert "missing_source" not in serialized
    assert "후보 지식" not in serialized
    assert "status:" not in serialized


def test_public_review_status_maps_internal_values() -> None:
    assert service._public_review_status("candidate_unreviewed") == "reference"
    assert service._public_review_status({"status": "candidate_unreviewed"}) == "reference"
    assert service._public_review_status("missing_source") == "unavailable"
    assert service._public_review_status("reviewed") == "reviewed"
    assert service._public_review_status("approved") == "approved"


def test_diet_rag_strategy_config_default_is_keyword_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIET_RECOMMENDATION_RAG_STRATEGY", raising=False)

    settings = Config(_env_file=None)

    assert settings.DIET_RECOMMENDATION_RAG_STRATEGY == "keyword_only"


@pytest.mark.asyncio
async def test_diet_rag_strategy_keyword_only_does_not_call_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    vector = FakeVectorRetriever(documents=[_vector_document()])
    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_RAG_STRATEGY", "keyword_only")
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")

    result = await service.build_diet_health_recommendation_response_async(
        diet_record=_diet({"sodium_mg": 920}),
        analysis_results=[_analysis(AnalysisType.HYPERTENSION)],
        health_record=None,
        active_challenges=ACTIVE_CHALLENGES,
        vector_retriever=vector,
    )

    assert vector.calls == []
    assert any(item["disease_code"] == "HTN" for item in result["rag_comment"]["evidence_sources"])
    assert result["rag_comment"]["safety_notice"] == service.SAFETY_NOTICE


@pytest.mark.asyncio
async def test_diet_rag_strategy_vector_disabled_does_not_call_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    vector = FakeVectorRetriever(documents=[_vector_document()])
    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_RAG_STRATEGY", "vector_disabled")
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")

    result = await service.build_diet_health_recommendation_response_async(
        diet_record=_diet({"sodium_mg": 920}),
        analysis_results=[_analysis(AnalysisType.HYPERTENSION)],
        health_record=None,
        active_challenges=ACTIVE_CHALLENGES,
        vector_retriever=vector,
    )

    assert vector.calls == []
    assert result["rag_comment"]["enabled"] is True


@pytest.mark.asyncio
async def test_diet_rag_strategy_hybrid_skips_vector_when_keyword_is_sufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = FakeVectorRetriever(documents=[_vector_document()])
    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_RAG_STRATEGY", "keyword_first_vector_fallback")
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")

    result = await service.build_diet_health_recommendation_response_async(
        diet_record=_diet({"sodium_mg": 920}),
        analysis_results=[_analysis(AnalysisType.HYPERTENSION)],
        health_record=None,
        active_challenges=ACTIVE_CHALLENGES,
        vector_retriever=vector,
    )

    assert vector.calls == []
    assert any(item["disease_code"] == "HTN" for item in result["rag_comment"]["evidence_sources"])


@pytest.mark.asyncio
async def test_diet_rag_strategy_hybrid_calls_vector_when_keyword_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = FakeVectorRetriever(documents=[_vector_document(content="chunk content must stay internal")])
    events: list[dict[str, Any]] = []

    def fake_record_langfuse_event(**kwargs: Any) -> None:
        events.append(kwargs)

    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_RAG_STRATEGY", "keyword_first_vector_fallback")
    monkeypatch.setattr(service.config, "RAG_ENABLED", True)
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(service, "retrieve_keyword_rag_matches", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "record_langfuse_event", fake_record_langfuse_event)

    result = await service.build_diet_health_recommendation_response_async(
        diet_record=_diet({"sodium_mg": 920}),
        analysis_results=[_analysis(AnalysisType.HYPERTENSION)],
        health_record=None,
        active_challenges=ACTIVE_CHALLENGES,
        vector_retriever=vector,
    )

    assert len(vector.calls) >= 1
    assert result["rag_comment"]["fallback_used"] is False
    assert any(item["title"] == "고혈압 식생활" for item in result["rag_comment"]["evidence_sources"])
    assert any(item["review_status"] == "reference" for item in result["rag_comment"]["evidence_sources"])
    serialized = str(result["rag_comment"])
    assert "chunk content must stay internal" not in serialized
    assert "rag:hypertension:section:000:chunk:0000" not in serialized
    assert "chunk_key" not in serialized
    assert "score" not in serialized
    assert "embedding" not in serialized
    assert "candidate_unreviewed" not in serialized
    assert "missing_source" not in serialized
    assert "후보 지식" not in serialized

    assert events
    trace_metadata = events[-1]["metadata"]
    assert trace_metadata["keyword_returned_count"] == 0
    assert trace_metadata["vector_returned_count"] >= 1
    assert trace_metadata["fallback_used"] is True
    assert trace_metadata["fallback_reason"] == "keyword_empty_vector_used"
    assert "chunk content must stay internal" not in str(trace_metadata)


@pytest.mark.asyncio
async def test_diet_rag_strategy_vector_failure_keeps_api_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = FakeVectorRetriever(fail=True)
    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_RAG_STRATEGY", "keyword_first_vector_fallback")
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(service, "retrieve_keyword_rag_matches", lambda *args, **kwargs: [])

    result = await service.build_diet_health_recommendation_response_async(
        diet_record=_diet({"sodium_mg": 920}),
        analysis_results=[_analysis(AnalysisType.HYPERTENSION)],
        health_record=None,
        active_challenges=ACTIVE_CHALLENGES,
        vector_retriever=vector,
    )

    assert vector.calls
    assert result["nutrition_findings"]
    assert result["recommended_foods"]
    assert result["safety_notice"] == service.SAFETY_NOTICE
    assert result["rag_comment"]["fallback_used"] is True


@pytest.mark.asyncio
async def test_diet_rag_strategy_embedding_disabled_does_not_call_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = FakeVectorRetriever(documents=[_vector_document()])
    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_RAG_STRATEGY", "keyword_first_vector_fallback")
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_ENABLED", False)
    monkeypatch.setattr(service.config, "RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(service, "retrieve_keyword_rag_matches", lambda *args, **kwargs: [])

    result = await service.build_diet_health_recommendation_response_async(
        diet_record=_diet({"sodium_mg": 920}),
        analysis_results=[_analysis(AnalysisType.HYPERTENSION)],
        health_record=None,
        active_challenges=ACTIVE_CHALLENGES,
        vector_retriever=vector,
    )

    assert vector.calls == []
    assert result["rag_comment"]["fallback_used"] is True
    assert result["rag_comment"]["safety_notice"] == service.SAFETY_NOTICE


def test_diet_rag_rewrite_flag_off_does_not_call_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("LLM must not be called when diet rewrite flag is off")

    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_LLM_REWRITE_ENABLED", False)
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ai_runtime.llm.diet_recommendation_rewriter.call_llm_json", fail_if_called)

    result = _build(nutrition={"sodium_mg": 920}, analysis_types=[AnalysisType.HYPERTENSION])

    assert result["rag_comment"]["rewrite_used"] is False
    assert result["rag_comment"]["fallback_reason"] == "rewrite_disabled"
    assert result["rag_comment"]["safety_notice"] == service.SAFETY_NOTICE


def test_diet_rag_rewrite_without_openai_key_does_not_call_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("LLM must not be called without OpenAI config")

    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_LLM_REWRITE_ENABLED", True)
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", None)
    monkeypatch.setattr("ai_runtime.llm.diet_recommendation_rewriter.call_llm_json", fail_if_called)

    result = _build(nutrition={"sodium_mg": 920}, analysis_types=[AnalysisType.HYPERTENSION])

    assert result["rag_comment"]["rewrite_used"] is False
    assert result["rag_comment"]["fallback_reason"] == "rewrite_disabled"
    assert result["rag_comment"]["safety_notice"] == service.SAFETY_NOTICE


def test_diet_rag_rewrite_success_with_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_call_llm_json(*args: Any, **kwargs: Any) -> str:
        calls.append({"args": args, "kwargs": kwargs})
        return json.dumps(
            {
                "summary": "실제 섭취량이 확정되지 않아 참고용입니다. 혈압 관리 관점에서 저염 식습관을 살펴보세요.",
                "disease_comments": [
                    {
                        "disease_code": "HTN",
                        "label": "혈압 관리",
                        "comment": "나트륨이 높은 후보로 보여 국물과 짠 소스는 참고용으로 주의가 필요합니다.",
                        "basis": "서비스 내 참고 문서 기반",
                    }
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_LLM_REWRITE_ENABLED", True)
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ai_runtime.llm.diet_recommendation_rewriter.call_llm_json", fake_call_llm_json)

    result = _build(nutrition={"sodium_mg": 920}, analysis_types=[AnalysisType.HYPERTENSION])

    assert calls
    assert result["rag_comment"]["rewrite_used"] is True
    assert result["rag_comment"]["fallback_reason"] is None
    assert "저염 식습관" in result["rag_comment"]["summary"]
    assert result["rag_comment"]["safety_notice"] == service.SAFETY_NOTICE


def test_diet_rag_rewrite_llm_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_LLM_REWRITE_ENABLED", True)
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")

    def fail_call(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("mock LLM unavailable")

    monkeypatch.setattr("ai_runtime.llm.diet_recommendation_rewriter.call_llm_json", fail_call)

    result = _build(nutrition={"sodium_mg": 920}, analysis_types=[AnalysisType.HYPERTENSION])

    assert result["rag_comment"]["rewrite_used"] is False
    assert result["rag_comment"]["fallback_reason"] == "llm_rewrite_failed"
    assert any(item["disease_code"] == "HTN" for item in result["rag_comment"]["disease_comments"])
    assert result["nutrition_findings"]
    assert result["recommended_foods"]


def test_diet_rag_rewrite_forbidden_phrase_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_LLM_REWRITE_ENABLED", True)
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")

    def unsafe_call(*args: Any, **kwargs: Any) -> str:
        return json.dumps(
            {
                "summary": "나트륨 과다입니다. 실제 섭취량이 확정되지 않아 참고용입니다.",
                "disease_comments": [
                    {
                        "disease_code": "HTN",
                        "label": "혈압 관리",
                        "comment": "고혈압 식단입니다.",
                        "basis": "서비스 내 참고 문서 기반",
                    }
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("ai_runtime.llm.diet_recommendation_rewriter.call_llm_json", unsafe_call)

    result = _build(nutrition={"sodium_mg": 920}, analysis_types=[AnalysisType.HYPERTENSION])

    assert result["rag_comment"]["rewrite_used"] is False
    assert result["rag_comment"]["fallback_reason"] == "safety_failed"
    serialized = str(result["rag_comment"])
    assert "나트륨 과다입니다" not in serialized
    assert "고혈압 식단입니다" not in serialized


def test_diet_rag_rewrite_ckd_restriction_phrase_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.config, "DIET_RECOMMENDATION_LLM_REWRITE_ENABLED", True)
    monkeypatch.setattr(service.config, "OPENAI_API_KEY", "test-key")

    def unsafe_ckd_call(*args: Any, **kwargs: Any) -> str:
        return json.dumps(
            {
                "summary": "실제 섭취량이 확정되지 않아 참고용입니다.",
                "disease_comments": [
                    {
                        "disease_code": "CKD",
                        "label": "신장 관리",
                        "comment": "단백질 제한하세요. 칼륨 제한하세요. 인 제한하세요.",
                        "basis": "서비스 내 참고 문서 기반",
                    }
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("ai_runtime.llm.diet_recommendation_rewriter.call_llm_json", unsafe_ckd_call)

    result = _build(
        nutrition={"protein_g": 35, "potassium_mg": 900},
        analysis_types=[AnalysisType.CHRONIC_KIDNEY_DISEASE],
    )

    assert result["rag_comment"]["rewrite_used"] is False
    assert result["rag_comment"]["fallback_reason"] == "safety_failed"
    serialized = str(result["rag_comment"])
    assert "의료진 상담" in serialized
    assert "단백질 제한하세요" not in serialized
    assert "칼륨 제한하세요" not in serialized
    assert "인 제한하세요" not in serialized


@pytest.mark.asyncio
async def test_diet_health_recommendation_forbids_other_user_record(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_diet_record(diet_record_id: int) -> SimpleNamespace:
        assert diet_record_id == 100
        return SimpleNamespace(id=100, user_id=99)

    monkeypatch.setattr(service.diet_service, "get_diet_record", fake_get_diet_record)

    with pytest.raises(HTTPException) as exc:
        await service.get_diet_health_recommendations(user_id=10, diet_record_id=100)

    assert exc.value.status_code == 403


def test_diet_health_recommendation_api_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int] = {}

    async def fake_current_user() -> SimpleNamespace:
        return SimpleNamespace(id=10)

    async def fake_get_recommendations(user_id: int, diet_record_id: int) -> dict[str, Any]:
        captured["user_id"] = user_id
        captured["diet_record_id"] = diet_record_id
        return {
            "diet_record_id": diet_record_id,
            "nutrition_findings": [
                {
                    "type": "excess_candidate",
                    "issue_key": "sodium_high",
                    "nutrient": "sodium_mg",
                    "label": "나트륨 주의",
                    "message": "현재 식단 후보에서 나트륨이 높은 음식이 포함된 것으로 보여 주의가 필요합니다.",
                    "basis": "100g 기준",
                }
            ],
            "disease_context": [],
            "recommended_foods": ["채소 반찬"],
            "caution_foods": ["짠 소스"],
            "recommended_challenges": [
                {"challenge_id": 1, "title": "염분 빼볼까염 챌린지", "reason": "나트륨 관리와 연결됩니다."}
            ],
            "safety_notice": service.SAFETY_NOTICE,
        }

    app.dependency_overrides[get_request_user] = fake_current_user
    monkeypatch.setattr(
        diet_routers.diet_recommendation_service, "get_diet_health_recommendations", fake_get_recommendations
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/diets/123/recommendations")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured == {"user_id": 10, "diet_record_id": 123}
    body = response.json()
    assert body["diet_record_id"] == 123
    assert body["nutrition_findings"][0]["issue_key"] == "sodium_high"
    assert body["recommended_challenges"][0]["title"] == "염분 빼볼까염 챌린지"
    assert body["safety_notice"] == service.SAFETY_NOTICE
