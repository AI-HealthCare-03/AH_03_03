from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_worker.cv.food.schemas import (
    CV_CONFIDENCE_THRESHOLD,
    FOOD_DETECTION_PROVIDER_PRIORITY,
    GPT_VISION_FALLBACK_POLICY,
    FoodDetectionCandidateSet,
)
from app.dtos.diets import DietAnalyzeRequest
from app.services import diets as diet_service


def test_nutrition_scoring_result_includes_five_disease_scores() -> None:
    result = diet_service.build_nutrition_scoring_result(
        [
            {"name": "현미밥", "confidence": 0.9},
            {"name": "김치볶음밥", "confidence": 0.8},
        ]
    )

    assert set(result["disease_scores"]) == {"DM", "HTN", "DL", "OBE", "ANEM"}
    assert all(isinstance(score, float) for score in result["disease_scores"].values())
    assert result["scoring_source"] == "nutrition_rule_table"
    assert result["food_score_details"][0]["matched_food_name"] == "현미밥"


def test_nutrition_scoring_handles_unmatched_food_without_error() -> None:
    result = diet_service.build_nutrition_scoring_result([{"name": "매칭안되는테스트음식"}])

    assert result["disease_scores"] == {"DM": None, "HTN": None, "DL": None, "OBE": None, "ANEM": None}
    assert result["food_score_details"][0]["match_status"] == "unmatched"
    assert result["food_score_details"][0]["scores"] is None


def test_food_detection_fallback_policy_schema() -> None:
    candidate_set = FoodDetectionCandidateSet(
        provider="gpt_vision",
        confidence=0.62,
        detected_foods=["현미밥"],
        needs_review=False,
        fallback_reason="cv_confidence_below_threshold",
    )

    assert FOOD_DETECTION_PROVIDER_PRIORITY == ("cv_model", "gpt_vision", "rule_based_food_detection")
    assert CV_CONFIDENCE_THRESHOLD == 0.75
    assert GPT_VISION_FALLBACK_POLICY == "user_confirmation_required"
    assert candidate_set.should_use_nutrition_scorer()


@pytest.mark.asyncio
async def test_run_diet_analysis_adds_nutrition_scoring_and_service_sources(monkeypatch) -> None:
    created_record_payload: dict = {}
    created_photo_payload: dict = {}

    async def fake_create_diet_record(user_id: int, request):
        assert user_id == 10
        created_record_payload.update(request.model_dump())
        return SimpleNamespace(id=1, user_id=user_id, **request.model_dump())

    async def fake_create_diet_photo_result(diet_record_id: int, request):
        assert diet_record_id == 1
        created_photo_payload.update(request.model_dump())
        return SimpleNamespace(id=2, diet_record_id=diet_record_id, **request.model_dump())

    monkeypatch.setattr(diet_service, "create_diet_record", fake_create_diet_record)
    monkeypatch.setattr(diet_service, "create_diet_photo_result", fake_create_diet_photo_result)

    response = await diet_service.run_diet_analysis(
        10,
        DietAnalyzeRequest(meal_type="LUNCH", description="점심 현미밥 닭가슴살 샐러드"),
    )

    assert set(response["disease_scores"]) == {"DM", "HTN", "DL", "OBE", "ANEM"}
    assert response["scoring_source"] == "nutrition_rule_table"
    assert response["food_score_details"]
    assert response["explanation"]["source"] == "rule_based_explanation"
    assert "진단이 아니" in response["explanation"]["safety_notice"]
    assert "의료진 상담" in response["explanation"]["safety_notice"]
    assert created_record_payload["nutrition_summary"]["scoring_source"] == "nutrition_rule_table"
    assert set(created_record_payload["nutrition_summary"]["disease_scores"]) == {"DM", "HTN", "DL", "OBE", "ANEM"}
    assert created_record_payload["nutrition_summary"]["explanation"]["source"] == "rule_based_explanation"
    assert created_photo_payload["is_dummy"] is False
    assert created_photo_payload["confidence_payload"]["method"] == "rule_based_food_detection"
    assert created_photo_payload["raw_output"]["source"] == "rule_based_food_detection"
    assert set(created_photo_payload["raw_output"]["disease_scores"]) == {"DM", "HTN", "DL", "OBE", "ANEM"}
    assert created_photo_payload["raw_output"]["food_score_details"]
    assert created_photo_payload["raw_output"]["scoring_source"] == "nutrition_rule_table"
    assert created_photo_payload["raw_output"]["explanation"]["source"] == "rule_based_explanation"
    assert "rule_stub" not in str(response)
    assert "image_analysis_stub" not in str(response)
    assert "rule_stub" not in str(created_photo_payload)
    assert "image_analysis_stub" not in str(created_photo_payload)


@pytest.mark.asyncio
async def test_run_diet_analysis_keeps_response_when_scorer_fails(monkeypatch) -> None:
    async def fake_create_diet_record(user_id: int, request):
        return SimpleNamespace(id=1, user_id=user_id, **request.model_dump())

    async def fake_create_diet_photo_result(diet_record_id: int, request):
        return SimpleNamespace(id=2, diet_record_id=diet_record_id, **request.model_dump())

    def raise_scoring_error(*args, **kwargs):
        raise RuntimeError("score csv missing")

    monkeypatch.setattr(diet_service, "create_diet_record", fake_create_diet_record)
    monkeypatch.setattr(diet_service, "create_diet_photo_result", fake_create_diet_photo_result)
    monkeypatch.setattr(diet_service, "build_nutrition_scoring_result", raise_scoring_error)

    response = await diet_service.run_diet_analysis(
        10,
        DietAnalyzeRequest(meal_type="LUNCH", description="점심 현미밥 닭가슴살 샐러드"),
    )

    assert response["scoring_source"] == "nutrition_rule_table_unavailable"
    assert response["disease_scores"] == {"DM": None, "HTN": None, "DL": None, "OBE": None, "ANEM": None}
    assert response["explanation"]["source"] == "rule_based_explanation"
