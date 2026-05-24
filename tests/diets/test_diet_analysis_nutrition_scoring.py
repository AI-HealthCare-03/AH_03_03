from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.dtos.diets import DietDummyAnalyzeRequest
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
        DietDummyAnalyzeRequest(meal_type="LUNCH", description="점심 현미밥 닭가슴살 샐러드"),
    )

    assert set(response["disease_scores"]) == {"DM", "HTN", "DL", "OBE", "ANEM"}
    assert response["scoring_source"] == "nutrition_rule_table"
    assert response["food_score_details"]
    assert created_record_payload["nutrition_summary"]["scoring_source"] == "nutrition_rule_table"
    assert created_photo_payload["is_dummy"] is False
    assert created_photo_payload["confidence_payload"]["method"] == "rule_based_food_detection"
    assert created_photo_payload["raw_output"]["source"] == "rule_based_food_detection"
    assert "rule_stub" not in str(created_photo_payload)
    assert "image_analysis_stub" not in str(created_photo_payload)
