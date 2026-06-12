from __future__ import annotations

import pytest

from ai_runtime.cv.food.matcher import FoodMatchResult
from ai_runtime.cv.food.pipeline import FoodAnalysisPipelineConfig, run_food_analysis_pipeline
from ai_runtime.cv.food.schemas import FoodDetectionCandidateSet
from ai_runtime.cv.providers.gpt_vision import PROMPTS, AnalysisType


def test_runtime_diet_prompt_requests_food_candidates_without_nutrition_estimates() -> None:
    prompt = PROMPTS[AnalysisType.DIET]

    assert '"foods"' in prompt
    assert '"name"' in prompt
    assert '"confidence"' in prompt
    assert "완성 음식, 반찬, 음료, 명확한 사이드 메뉴 중심" in prompt
    assert "재료, 소스, 양념, 토핑을 foods에 넣지 마세요" in prompt
    assert "단독 일반 재료명만 반환하지 마세요" in prompt
    assert "foods=[]는 음식, 음료, 식단으로 볼 수 있는 메뉴가 전혀 보이지 않는 경우에만 사용" in prompt
    assert '"nutrition"' not in prompt
    assert '"nutrient_category"' not in prompt
    assert '"search_keyword"' not in prompt
    assert "estimated_amount" not in prompt


@pytest.mark.asyncio
async def test_pipeline_uses_rule_based_provider_when_gpt_disabled() -> None:
    result = await run_food_analysis_pipeline(
        rule_based_foods=[{"name": "쌀떡", "confidence": 0.9}],
        image_bytes=None,
        image_media_type=None,
        config=FoodAnalysisPipelineConfig(provider="rule_based", gpt_vision_enabled=False),
    )

    assert result.food_candidate.provider == "rule_based_food_detection"
    assert result.detected_foods[0]["name"] == "가래떡"
    assert result.provider_candidate is None
    assert result.provider_result.provider_name == "rule_based_food_detection"
    assert result.provider_result.raw_food_names == ["쌀떡"]
    assert result.provider_result.matched_food_names == ["가래떡"]
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
    assert result.provider_result.provider_name == "gpt_vision"
    assert result.provider_result.raw_food_names == ["쌀떡"]
    assert result.provider_result.matched_food_names == ["가래떡"]
    assert result.provider_result.metadata["model"] == "test-model"
    assert result.provider_result.metadata["analysis_status"] == "success"
    assert result.provider_result.metadata["fail_reason"] is None
    assert result.provider_result.metadata["food_count"] == 1
    assert result.detected_foods[0]["name"] == "가래떡"
    assert result.detected_foods[0]["original_name"] == "쌀떡"
    assert result.fallback_used is False
    assert result.provider_message == "gpt_vision_food_detection"


@pytest.mark.asyncio
async def test_pipeline_falls_back_to_rule_based_when_gpt_vision_raises() -> None:
    class FailingVisionClient:
        def __init__(self, api_key: str, model: str):
            pass

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            raise RuntimeError("vision unavailable")

    result = await run_food_analysis_pipeline(
        rule_based_foods=[{"name": "현미밥", "confidence": 0.9}],
        image_bytes=b"image",
        image_media_type="image/png",
        config=FoodAnalysisPipelineConfig(
            provider="gpt_vision",
            gpt_vision_enabled=True,
            openai_api_key="test-key",
            vision_client_cls=FailingVisionClient,
        ),
    )

    assert result.food_candidate.provider == "rule_based_food_detection"
    assert result.provider_candidate is None
    assert result.fallback_provider_result is not None
    assert result.fallback_provider_result.provider_name == "rule_based_food_detection"
    assert result.detected_foods[0]["name"] == "현미밥"
    assert result.fallback_used is True
    assert result.provider_message == "gpt_vision_failed"


@pytest.mark.asyncio
async def test_pipeline_falls_back_to_rule_based_when_gpt_returns_no_foods() -> None:
    class EmptyVisionClient:
        def __init__(self, api_key: str, model: str):
            pass

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            return {"analysis_status": "failed", "foods": []}

    result = await run_food_analysis_pipeline(
        rule_based_foods=[{"name": "샐러드", "confidence": 0.8}],
        image_bytes=b"image",
        image_media_type="image/png",
        config=FoodAnalysisPipelineConfig(
            provider="gpt_vision",
            gpt_vision_enabled=True,
            openai_api_key="test-key",
            vision_client_cls=EmptyVisionClient,
        ),
    )

    assert result.food_candidate.provider == "rule_based_food_detection"
    assert result.detected_foods[0]["name"] == "샐러드"
    assert result.provider_result.unmatched_food_names == []
    assert result.fallback_provider_result is not None
    assert result.provider_message == "gpt_vision_returned_empty_foods"


@pytest.mark.asyncio
async def test_pipeline_matches_raw_food_name_to_scorer_canonical_name() -> None:
    class RiceVisionClient:
        def __init__(self, api_key: str, model: str):
            pass

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            return {
                "analysis_status": "success",
                "foods": [{"name": "흰쌀밥", "confidence": 0.93}],
            }

    result = await run_food_analysis_pipeline(
        rule_based_foods=[],
        image_bytes=b"image",
        image_media_type="image/png",
        config=FoodAnalysisPipelineConfig(
            provider="gpt_vision",
            gpt_vision_enabled=True,
            openai_api_key="test-key",
            vision_client_cls=RiceVisionClient,
        ),
    )

    assert result.detected_foods[0]["original_name"] == "흰쌀밥"
    assert result.detected_foods[0]["name"] == "쌀밥"
    assert result.detected_foods[0]["matched_food_name"] == "쌀밥"
    assert result.detected_foods[0]["needs_user_confirmation"] is False
    assert result.provider_result.raw_food_names == ["흰쌀밥"]
    assert result.provider_result.matched_food_names == ["쌀밥"]


@pytest.mark.asyncio
async def test_pipeline_keeps_unmatched_food_name_without_failing_analysis() -> None:
    class UnknownVisionClient:
        def __init__(self, api_key: str, model: str):
            pass

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            return {
                "analysis_status": "success",
                "foods": [{"name": "미확인테스트음식", "confidence": 0.51}],
            }

    result = await run_food_analysis_pipeline(
        rule_based_foods=[{"name": "현미밥", "confidence": 0.9}],
        image_bytes=b"image",
        image_media_type="image/png",
        config=FoodAnalysisPipelineConfig(
            provider="gpt_vision",
            gpt_vision_enabled=True,
            openai_api_key="test-key",
            vision_client_cls=UnknownVisionClient,
        ),
    )

    assert result.food_candidate.provider == "gpt_vision"
    assert result.detected_foods[0]["name"] == "미확인테스트음식"
    assert result.detected_foods[0]["matched_food_name"] is None
    assert result.detected_foods[0]["needs_user_confirmation"] is True
    assert result.provider_result.unmatched_food_names == ["미확인테스트음식"]
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_pipeline_uses_injected_food_db_matcher_for_detected_foods() -> None:
    class FakeMatcher:
        def match(self, query: str) -> FoodMatchResult:
            return FoodMatchResult(
                original_name=query,
                query_name=query,
                matched_food_name="비빔국수",
                matched_food_code="mfds:001",
                match_source="mfds_matched",
                match_confidence=0.98,
                needs_user_confirmation=True,
            )

    result = await run_food_analysis_pipeline(
        rule_based_foods=[{"name": "비빔국수", "confidence": 0.9}],
        image_bytes=None,
        image_media_type=None,
        config=FoodAnalysisPipelineConfig(provider="rule_based", gpt_vision_enabled=False, food_db_matcher=FakeMatcher()),
    )

    assert result.detected_foods[0]["name"] == "비빔국수"
    assert result.detected_foods[0]["matched_food_name"] == "비빔국수"
    assert result.detected_foods[0]["matched_food_code"] == "mfds:001"
    assert result.detected_foods[0]["match_source"] == "mfds_matched"
    assert result.detected_foods[0]["match_confidence"] == 0.98
    assert result.detected_foods[0]["needs_user_confirmation"] is True


@pytest.mark.asyncio
async def test_pipeline_preserves_gpt_vision_per_food_confidence() -> None:
    class TwoFoodVisionClient:
        def __init__(self, api_key: str, model: str):
            pass

        async def analyze(self, analysis_type: str, image_bytes: bytes, media_type: str):
            return {
                "analysis_status": "success",
                "foods": [
                    {"name": "비빔국수", "confidence": 0.91},
                    {"name": "핫초코", "confidence": 0.42},
                ],
            }

    class EchoMatcher:
        def match(self, query: str) -> FoodMatchResult:
            return FoodMatchResult(
                original_name=query,
                query_name=query,
                matched_food_name=query,
                matched_food_code=f"test:{query}",
                match_source="test_matched",
                match_confidence=0.99,
                needs_user_confirmation=True,
            )

    result = await run_food_analysis_pipeline(
        rule_based_foods=[],
        image_bytes=b"image",
        image_media_type="image/png",
        config=FoodAnalysisPipelineConfig(
            provider="gpt_vision",
            gpt_vision_enabled=True,
            openai_api_key="test-key",
            vision_client_cls=TwoFoodVisionClient,
            food_db_matcher=EchoMatcher(),
        ),
    )

    assert result.food_candidate.confidence == 0.665
    assert result.detected_foods[0]["name"] == "비빔국수"
    assert result.detected_foods[0]["confidence"] == 0.91
    assert result.detected_foods[1]["name"] == "핫초코"
    assert result.detected_foods[1]["confidence"] == 0.42


def test_food_candidate_skips_generic_mfds_lookup_candidates() -> None:
    class CountingMfdsMatcher:
        applies_lookup_gating = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        def match(self, query: str) -> FoodMatchResult:
            self.calls.append(query)
            return FoodMatchResult(original_name=query, query_name=query, matched_food_name=query)

    matcher = CountingMfdsMatcher()
    candidate = FoodDetectionCandidateSet(
        provider="gpt_vision",
        detected_foods=["계란", "고기", "고추장"],
        detected_food_confidences=[0.9, 0.85, 0.8],
        confidence=0.85,
    )

    foods = candidate.to_scorer_foods(matcher)

    assert matcher.calls == []
    assert [food["name"] for food in foods] == ["계란", "고기", "고추장"]
    assert [food["match_source"] for food in foods] == [
        "mfds_skipped_generic",
        "mfds_skipped_generic",
        "mfds_skipped_generic",
    ]
    assert all(food["needs_user_confirmation"] is True for food in foods)


def test_food_candidate_skips_low_confidence_mfds_lookup_candidates() -> None:
    class CountingMfdsMatcher:
        applies_lookup_gating = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        def match(self, query: str) -> FoodMatchResult:
            self.calls.append(query)
            return FoodMatchResult(original_name=query, query_name=query, matched_food_name=query)

    matcher = CountingMfdsMatcher()
    candidate = FoodDetectionCandidateSet(
        provider="gpt_vision",
        detected_foods=["중식 면요리"],
        detected_food_confidences=[0.42],
        confidence=0.42,
    )

    foods = candidate.to_scorer_foods(matcher)

    assert matcher.calls == []
    assert foods[0]["name"] == "중식 면요리"
    assert foods[0]["match_source"] == "mfds_skipped_low_confidence"
    assert foods[0]["matched_food_name"] is None
    assert foods[0]["match_confidence"] is None


def test_food_candidate_limits_mfds_lookup_to_three_unique_queries() -> None:
    class CountingMfdsMatcher:
        applies_lookup_gating = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        def match(self, query: str) -> FoodMatchResult:
            self.calls.append(query)
            return FoodMatchResult(
                original_name=query,
                query_name=query,
                matched_food_name=query,
                matched_food_code=f"mfds:{query}",
                match_source="mfds_matched",
                match_confidence=0.95,
            )

    matcher = CountingMfdsMatcher()
    candidate = FoodDetectionCandidateSet(
        provider="gpt_vision",
        detected_foods=["비빔밥", "짜장면", "김밥", "라면"],
        detected_food_confidences=[0.9, 0.88, 0.86, 0.84],
        confidence=0.87,
    )

    foods = candidate.to_scorer_foods(matcher)

    assert matcher.calls == ["비빔밥", "짜장면", "김밥"]
    assert [food["match_source"] for food in foods] == [
        "mfds_matched",
        "mfds_matched",
        "mfds_matched",
        "mfds_skipped_lookup_limit",
    ]
    assert foods[3]["matched_food_name"] is None


def test_food_candidate_reuses_mfds_lookup_for_duplicate_normalized_query() -> None:
    class CountingMfdsMatcher:
        applies_lookup_gating = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        def match(self, query: str) -> FoodMatchResult:
            self.calls.append(query)
            return FoodMatchResult(
                original_name=query,
                query_name=query,
                matched_food_name="비빔밥",
                matched_food_code="mfds:bibimbap",
                match_source="mfds_matched",
                match_confidence=0.98,
            )

    matcher = CountingMfdsMatcher()
    candidate = FoodDetectionCandidateSet(
        provider="gpt_vision",
        detected_foods=["비빔밥", "비빔밥 "],
        detected_food_confidences=[0.9, 0.8],
        confidence=0.85,
    )

    foods = candidate.to_scorer_foods(matcher)

    assert matcher.calls == ["비빔밥"]
    assert [food["matched_food_name"] for food in foods] == ["비빔밥", "비빔밥"]
    assert [food["match_source"] for food in foods] == ["mfds_matched", "mfds_matched"]
    assert [food["original_name"] for food in foods] == ["비빔밥", "비빔밥"]
