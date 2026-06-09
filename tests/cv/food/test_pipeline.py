from __future__ import annotations

import pytest

from ai_runtime.cv.food.pipeline import FoodAnalysisPipelineConfig, run_food_analysis_pipeline


@pytest.mark.asyncio
async def test_pipeline_uses_rule_based_provider_when_gpt_disabled() -> None:
    result = await run_food_analysis_pipeline(
        rule_based_foods=[{"name": "현미밥", "confidence": 0.9}],
        image_bytes=None,
        image_media_type=None,
        config=FoodAnalysisPipelineConfig(provider="rule_based", gpt_vision_enabled=False),
    )

    assert result.food_candidate.provider == "rule_based_food_detection"
    assert result.detected_foods[0]["name"] == "현미밥"
    assert result.provider_candidate is None
    assert result.fallback_used is True
    assert result.provider_message == "diet GPT Vision disabled"


@pytest.mark.asyncio
async def test_pipeline_uses_gpt_vision_provider_with_injected_client() -> None:
    class FakeVisionClient:
        def __init__(self, api_key: str, model: str):
            assert api_key == "test-key"
            assert model == "test-model"

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            assert analysis_type == "diet"
            assert image_bytes == b"image"
            assert media_type == "image/png"
            return {
                "analysis_status": "success",
                "foods": [{"name": "쌀떡", "confidence": 0.9}],
            }

    result = await run_food_analysis_pipeline(
        rule_based_foods=[{"name": "일반식", "confidence": 0.7}],
        image_bytes=b"image",
        image_media_type="image/png",
        config=FoodAnalysisPipelineConfig(
            provider="gpt_vision",
            gpt_vision_enabled=True,
            openai_api_key="test-key",
            gpt_vision_model="test-model",
            vision_client_cls=FakeVisionClient,
        ),
    )

    assert result.food_candidate.provider == "gpt_vision"
    assert result.provider_candidate is not None
    assert result.detected_foods[0]["name"] == "가래떡"
    assert result.detected_foods[0]["original_name"] == "쌀떡"
    assert result.fallback_used is False
    assert result.provider_message == "gpt_vision_food_detection"
