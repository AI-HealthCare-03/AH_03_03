from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_runtime.cv.food.schemas import (
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
    assert result["food_score_details"][0]["match_status"] == "needs_user_confirmation"
    assert result["food_score_details"][0]["scores"] is None


@pytest.mark.parametrize("food_name", ["가래떡", "쌀떡", "rice cake"])
def test_nutrition_scoring_normalizes_common_rice_cake_aliases(food_name: str) -> None:
    result = diet_service.build_nutrition_scoring_result([{"name": food_name, "confidence": 0.8}])

    assert result["detected_foods"] == ["가래떡"]
    assert result["food_score_details"][0]["food_name"] == "가래떡"
    assert result["food_score_details"][0]["matched_food_name"] == "가래떡"
    assert result["food_score_details"][0]["match_status"] == "matched"


def test_nutrition_scoring_does_not_overfix_generic_tteok_to_garaetteok() -> None:
    result = diet_service.build_nutrition_scoring_result([{"name": "떡", "confidence": 0.8}])
    detail = result["food_score_details"][0]

    assert result["detected_foods"] == ["떡"]
    assert detail["food_name"] == "떡"
    assert detail["query_name"] == "떡"
    assert detail["needs_user_confirmation"] is True
    assert detail["matched_food_name"] is None
    assert detail["matched_food_code"] is None
    assert detail["match_source"] == "local_stub_unmatched"
    assert detail["match_confidence"] is None
    assert detail["match_status"] == "needs_user_confirmation"
    assert detail["scores"] is None


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
    monkeypatch.setattr(diet_service.config, "DIET_DEMO_FALLBACK_ENABLED", True)

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
    assert created_record_payload["diet_score"] is None
    assert response["diet_score"] is None
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
    monkeypatch.setattr(diet_service.config, "DIET_DEMO_FALLBACK_ENABLED", True)

    response = await diet_service.run_diet_analysis(
        10,
        DietAnalyzeRequest(meal_type="LUNCH", description="점심 현미밥 닭가슴살 샐러드"),
    )

    assert response["scoring_source"] == "nutrition_rule_table_unavailable"
    assert response["disease_scores"] == {"DM": None, "HTN": None, "DL": None, "OBE": None, "ANEM": None}
    assert response["explanation"]["source"] == "rule_based_explanation"


@pytest.mark.asyncio
async def test_run_diet_analysis_rejects_rule_based_without_demo_fallback(monkeypatch) -> None:
    created_record_called = False

    async def fake_create_diet_record(user_id: int, request):
        nonlocal created_record_called
        created_record_called = True
        return SimpleNamespace(id=1, user_id=user_id, **request.model_dump())

    async def fake_create_diet_photo_result(diet_record_id: int, request):
        return SimpleNamespace(id=2, diet_record_id=diet_record_id, **request.model_dump())

    monkeypatch.setattr(diet_service, "create_diet_record", fake_create_diet_record)
    monkeypatch.setattr(diet_service, "create_diet_photo_result", fake_create_diet_photo_result)
    monkeypatch.setattr(diet_service.config, "DIET_VISION_PROVIDER", "rule_based")
    monkeypatch.setattr(diet_service.config, "DIET_GPT_VISION_ENABLED", False)
    monkeypatch.setattr(diet_service.config, "DIET_DEMO_FALLBACK_ENABLED", False)
    monkeypatch.setattr(diet_service.config, "OPENAI_API_KEY", None)

    with pytest.raises(ValueError, match=diet_service.DIET_ANALYSIS_SERVICE_UNAVAILABLE):
        await diet_service.run_diet_analysis(
            10,
            DietAnalyzeRequest(meal_type="LUNCH", description="비빔밥 사진"),
            image_bytes=b"image",
            image_media_type="image/jpeg",
        )

    assert created_record_called is False


@pytest.mark.asyncio
async def test_run_diet_analysis_uses_gpt_vision_foods_when_enabled(monkeypatch) -> None:
    async def fake_create_diet_record(user_id: int, request):
        return SimpleNamespace(id=1, user_id=user_id, **request.model_dump())

    async def fake_create_diet_photo_result(diet_record_id: int, request):
        return SimpleNamespace(id=2, diet_record_id=diet_record_id, **request.model_dump())

    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            assert api_key == "test-key"
            assert model == "gpt-4o"

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            assert analysis_type == "diet"
            assert image_bytes == b"image"
            assert media_type == "image/png"
            return {
                "analysis_status": "success",
                "foods": [
                    {"name": "현미밥", "confidence": 0.91},
                    {"name": "닭가슴살", "confidence": 0.88},
                ],
            }

    monkeypatch.setattr(diet_service, "create_diet_record", fake_create_diet_record)
    monkeypatch.setattr(diet_service, "create_diet_photo_result", fake_create_diet_photo_result)
    monkeypatch.setattr(diet_service, "VisionClient", FakeVisionClient)
    monkeypatch.setattr(diet_service.config, "DIET_VISION_PROVIDER", "gpt_vision")
    monkeypatch.setattr(diet_service.config, "DIET_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(diet_service.config, "DIET_GPT_VISION_MODEL", "gpt-4o")
    monkeypatch.setattr(diet_service.config, "DIET_MFDS_ENABLED", False)
    monkeypatch.setattr(diet_service.config, "OPENAI_API_KEY", "test-key")

    response = await diet_service.run_diet_analysis(
        10,
        DietAnalyzeRequest(meal_type="LUNCH", description="사진 식단"),
        image_bytes=b"image",
        image_media_type="image/png",
    )

    assert response["vision_provider"] == "gpt_vision"
    assert response["fallback_used"] is False
    assert [food["name"] for food in response["detected_foods"]] == ["현미밥", "닭가슴살"]
    assert response["raw_output"]["source"] == "gpt_vision"


@pytest.mark.asyncio
async def test_run_diet_analysis_returns_stable_food_names_from_gpt_aliases(monkeypatch) -> None:
    async def fake_create_diet_record(user_id: int, request):
        return SimpleNamespace(id=1, user_id=user_id, **request.model_dump())

    async def fake_create_diet_photo_result(diet_record_id: int, request):
        return SimpleNamespace(id=2, diet_record_id=diet_record_id, **request.model_dump())

    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            assert api_key == "test-key"
            assert model == "gpt-4o"

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            assert analysis_type == "diet"
            return {
                "analysis_status": "success",
                "foods": [
                    {"name": "rice cake", "confidence": 0.91},
                    {"food_name": "쌀떡", "confidence": 0.87},
                ],
            }

    monkeypatch.setattr(diet_service, "create_diet_record", fake_create_diet_record)
    monkeypatch.setattr(diet_service, "create_diet_photo_result", fake_create_diet_photo_result)
    monkeypatch.setattr(diet_service, "VisionClient", FakeVisionClient)
    monkeypatch.setattr(diet_service.config, "DIET_VISION_PROVIDER", "gpt_vision")
    monkeypatch.setattr(diet_service.config, "DIET_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(diet_service.config, "DIET_GPT_VISION_MODEL", "gpt-4o")
    monkeypatch.setattr(diet_service.config, "DIET_MFDS_ENABLED", False)
    monkeypatch.setattr(diet_service.config, "OPENAI_API_KEY", "test-key")

    response = await diet_service.run_diet_analysis(
        10,
        DietAnalyzeRequest(meal_type="SNACK", description="떡 사진"),
        image_bytes=b"image",
        image_media_type="image/png",
    )

    assert [food["name"] for food in response["detected_foods"]] == ["가래떡", "가래떡"]
    assert [food["original_name"] for food in response["detected_foods"]] == ["rice cake", "쌀떡"]
    assert [food["query_name"] for food in response["detected_foods"]] == ["rice cake", "쌀떡"]
    assert [food["matched_food_name"] for food in response["detected_foods"]] == ["가래떡", "가래떡"]
    assert [food["matched_food_code"] for food in response["detected_foods"]] == [
        "local_stub:garaetteok",
        "local_stub:garaetteok",
    ]
    assert [food["needs_user_confirmation"] for food in response["detected_foods"]] == [False, False]
    assert response["food_score_details"][0]["matched_food_name"] == "가래떡"
    assert response["food_score_details"][1]["matched_food_name"] == "가래떡"


@pytest.mark.asyncio
async def test_run_diet_analysis_rejects_when_gpt_vision_key_missing(monkeypatch) -> None:
    created_record_called = False

    async def fake_create_diet_record(user_id: int, request):
        nonlocal created_record_called
        created_record_called = True
        return SimpleNamespace(id=1, user_id=user_id, **request.model_dump())

    async def fake_create_diet_photo_result(diet_record_id: int, request):
        return SimpleNamespace(id=2, diet_record_id=diet_record_id, **request.model_dump())

    monkeypatch.setattr(diet_service, "create_diet_record", fake_create_diet_record)
    monkeypatch.setattr(diet_service, "create_diet_photo_result", fake_create_diet_photo_result)
    monkeypatch.setattr(diet_service.config, "DIET_VISION_PROVIDER", "gpt_vision")
    monkeypatch.setattr(diet_service.config, "DIET_GPT_VISION_ENABLED", True)
    monkeypatch.setattr(diet_service.config, "DIET_DEMO_FALLBACK_ENABLED", False)
    monkeypatch.setattr(diet_service.config, "OPENAI_API_KEY", None)

    with pytest.raises(ValueError, match=diet_service.DIET_ANALYSIS_SERVICE_UNAVAILABLE):
        await diet_service.run_diet_analysis(
            10,
            DietAnalyzeRequest(meal_type="LUNCH", description="사진 식단"),
            image_bytes=b"image",
            image_media_type="image/png",
        )

    assert created_record_called is False


@pytest.mark.asyncio
async def test_run_diet_analysis_from_job_loads_uploaded_image(monkeypatch, tmp_path) -> None:
    upload_path = tmp_path / "diet.jpg"
    upload_path.write_bytes(b"diet-image")
    now = datetime.now(UTC)
    captured: dict[str, object] = {}

    async def fake_get_job(job_id: int):
        assert job_id == 88
        return SimpleNamespace(
            request_payload={
                "user_id": 10,
                "meal_type": "LUNCH",
                "description": "사진 식단",
                "upload_path": str(upload_path),
                "image_media_type": "image/jpeg",
            }
        )

    async def fake_run_diet_analysis(user_id: int, request, image_bytes=None, image_media_type=None):
        captured["user_id"] = user_id
        captured["description"] = request.description
        captured["image_bytes"] = image_bytes
        captured["image_media_type"] = image_media_type
        return {
            "message": "식단 분석이 완료되었습니다.",
            "diet_record": SimpleNamespace(
                id=1,
                user_id=user_id,
                meal_type=request.meal_type,
                meal_time=now,
                description=request.description,
                image_path=None,
                detected_foods=[{"name": "현미밥"}],
                nutrition_summary={"calories": 620},
                diet_score=82.5,
                diet_feedback="ok",
                analysis_method="IMAGE_ANALYSIS",
                is_user_corrected=False,
                memo=None,
                created_at=now,
                updated_at=now,
            ),
            "photo_result": SimpleNamespace(
                id=2,
                diet_record_id=1,
                detected_foods=[{"name": "현미밥"}],
                confidence_payload={"method": "rule_based_food_detection"},
                raw_output={"source": "rule_based_food_detection"},
                created_at=now,
            ),
            "detected_foods": [{"name": "현미밥"}],
            "nutrition_summary": {"calories": 620},
            "diet_score": 82.5,
            "diet_feedback": "ok",
            "vision_provider": "rule_based_food_detection",
            "fallback_used": True,
        }

    monkeypatch.setattr("app.services.async_jobs.get_job", fake_get_job)
    monkeypatch.setattr(diet_service, "run_diet_analysis", fake_run_diet_analysis)

    response = await diet_service.run_diet_analysis_from_job(88)

    assert response.diet_record.id == 1
    assert captured == {
        "user_id": 10,
        "description": "사진 식단",
        "image_bytes": b"diet-image",
        "image_media_type": "image/jpeg",
    }


def test_diet_analysis_upload_key_uses_user_and_unique_segment() -> None:
    key = diet_service._build_diet_analysis_upload_key(user_id=10, suffix=".jpg")

    assert key.startswith("diet-analysis/10/")
    assert key.endswith("/source.jpg")
    assert ".." not in key


def test_store_diet_analysis_upload_uses_local_storage_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(diet_service.config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(diet_service.config, "LOCAL_STORAGE_ROOT", str(tmp_path))

    stored = diet_service.store_diet_analysis_upload(
        user_id=10,
        image_bytes=b"diet-image",
        image_media_type="image/png",
        filename="meal.png",
    )

    stored_key = stored["upload_path"]
    assert stored_key.startswith("diet-analysis/10/")
    assert stored_key.endswith("/source.png")
    assert (tmp_path / stored_key).read_bytes() == b"diet-image"
    assert stored["image_media_type"] == "image/png"
    assert stored["image_filename"] == "meal.png"


@pytest.mark.asyncio
async def test_run_diet_analysis_from_job_loads_storage_key(monkeypatch, tmp_path) -> None:
    stored_key = "diet-analysis/10/test/source.webp"
    storage_path = tmp_path / stored_key
    storage_path.parent.mkdir(parents=True)
    storage_path.write_bytes(b"stored-diet-image")
    now = datetime.now(UTC)
    captured: dict[str, object] = {}

    monkeypatch.setattr(diet_service.config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(diet_service.config, "LOCAL_STORAGE_ROOT", str(tmp_path))

    async def fake_get_job(job_id: int):
        assert job_id == 89
        return SimpleNamespace(
            request_payload={
                "user_id": 10,
                "meal_type": "LUNCH",
                "description": "사진 식단",
                "upload_path": stored_key,
            }
        )

    async def fake_run_diet_analysis(user_id: int, request, image_bytes=None, image_media_type=None):
        captured["user_id"] = user_id
        captured["description"] = request.description
        captured["image_bytes"] = image_bytes
        captured["image_media_type"] = image_media_type
        return {
            "message": "식단 분석이 완료되었습니다.",
            "diet_record": SimpleNamespace(
                id=1,
                user_id=user_id,
                meal_type=request.meal_type,
                meal_time=now,
                description=request.description,
                image_path=None,
                detected_foods=[{"name": "현미밥"}],
                nutrition_summary={"calories": 620},
                diet_score=82.5,
                diet_feedback="ok",
                analysis_method="IMAGE_ANALYSIS",
                is_user_corrected=False,
                memo=None,
                created_at=now,
                updated_at=now,
            ),
            "photo_result": SimpleNamespace(
                id=2,
                diet_record_id=1,
                detected_foods=[{"name": "현미밥"}],
                confidence_payload={"method": "rule_based_food_detection"},
                raw_output={"source": "rule_based_food_detection"},
                created_at=now,
            ),
            "detected_foods": [{"name": "현미밥"}],
            "nutrition_summary": {"calories": 620},
            "diet_score": 82.5,
            "diet_feedback": "ok",
            "vision_provider": "rule_based_food_detection",
            "fallback_used": True,
        }

    monkeypatch.setattr("app.services.async_jobs.get_job", fake_get_job)
    monkeypatch.setattr(diet_service, "run_diet_analysis", fake_run_diet_analysis)

    response = await diet_service.run_diet_analysis_from_job(89)

    assert response.diet_record.id == 1
    assert captured == {
        "user_id": 10,
        "description": "사진 식단",
        "image_bytes": b"stored-diet-image",
        "image_media_type": "image/webp",
    }
