from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_runtime.cv.food.fallback_policy import select_food_detection_candidate
from ai_runtime.cv.food.matcher import FoodDbMatcher
from ai_runtime.cv.food.providers.base import FoodDetectionProviderResult
from ai_runtime.cv.food.providers.gpt_vision import GptVisionFoodDetectionProvider
from ai_runtime.cv.food.providers.rule_based import RuleBasedFoodDetectionProvider
from ai_runtime.cv.food.schemas import CV_CONFIDENCE_THRESHOLD, FoodDetectionCandidateSet
from ai_runtime.cv.providers.gpt_vision import VisionClient


@dataclass(frozen=True)
class FoodAnalysisPipelineConfig:
    provider: str = "rule_based"
    gpt_vision_enabled: bool = False
    gpt_vision_fallback_enabled: bool = False
    confidence_threshold: float = CV_CONFIDENCE_THRESHOLD
    openai_api_key: str | None = None
    gpt_vision_model: str = "gpt-4o"
    vision_client_cls: type[Any] = VisionClient
    food_db_matcher: FoodDbMatcher | None = None


@dataclass(frozen=True)
class FoodAnalysisPipelineResult:
    food_candidate: FoodDetectionCandidateSet
    detected_foods: list[dict[str, Any]]
    provider_candidate: FoodDetectionCandidateSet | None
    fallback_used: bool
    provider_message: str | None
    provider_result: FoodDetectionProviderResult
    fallback_provider_result: FoodDetectionProviderResult | None = None


async def run_food_analysis_pipeline(
    *,
    rule_based_foods: list[dict[str, Any]],
    image_bytes: bytes | None,
    image_media_type: str | None,
    config: FoodAnalysisPipelineConfig,
) -> FoodAnalysisPipelineResult:
    provider_name = str(config.provider or "rule_based").lower()
    if provider_name != "gpt_vision" or not config.gpt_vision_enabled:
        result = await _run_rule_based_provider(
            rule_based_foods=rule_based_foods,
            image_bytes=image_bytes,
            image_media_type=image_media_type,
        )
        food_candidate = _require_candidate(result.candidate)
        return FoodAnalysisPipelineResult(
            food_candidate=food_candidate,
            detected_foods=food_candidate.to_scorer_foods(config.food_db_matcher),
            provider_candidate=None,
            fallback_used=True,
            provider_message="diet GPT Vision disabled",
            provider_result=result,
        )

    provider_result = await GptVisionFoodDetectionProvider(
        api_key=config.openai_api_key,
        model=config.gpt_vision_model,
        vision_client_cls=config.vision_client_cls,
    ).detect(
        image_bytes=image_bytes,
        image_media_type=image_media_type,
    )
    provider_candidate = provider_result.candidate
    if provider_candidate is None:
        fallback_result = await _run_rule_based_provider(
            rule_based_foods=rule_based_foods,
            image_bytes=image_bytes,
            image_media_type=image_media_type,
        )
        food_candidate = _require_candidate(fallback_result.candidate)
        return FoodAnalysisPipelineResult(
            food_candidate=food_candidate,
            detected_foods=food_candidate.to_scorer_foods(config.food_db_matcher),
            provider_candidate=None,
            fallback_used=True,
            provider_message=provider_result.message or fallback_result.message,
            provider_result=provider_result,
            fallback_provider_result=fallback_result,
        )

    food_candidate = select_food_detection_candidate(
        cv_result=provider_candidate,
        rule_based_foods=rule_based_foods,
        gpt_vision_fallback_enabled=config.gpt_vision_fallback_enabled,
        confidence_threshold=config.confidence_threshold,
    )
    fallback_used = provider_result.fallback_used
    provider_message = provider_result.message
    if provider_candidate is None or food_candidate.provider != provider_candidate.provider:
        fallback_used = True
        provider_message = provider_message or "rule_based_food_detection fallback used"

    return FoodAnalysisPipelineResult(
        food_candidate=food_candidate,
        detected_foods=food_candidate.to_scorer_foods(config.food_db_matcher),
        provider_candidate=provider_candidate,
        fallback_used=fallback_used,
        provider_message=provider_message,
        provider_result=provider_result,
    )


async def _run_rule_based_provider(
    *,
    rule_based_foods: list[dict[str, Any]],
    image_bytes: bytes | None,
    image_media_type: str | None,
) -> FoodDetectionProviderResult:
    return await RuleBasedFoodDetectionProvider(rule_based_foods=rule_based_foods).detect(
        image_bytes=image_bytes,
        image_media_type=image_media_type,
    )


def _require_candidate(candidate: FoodDetectionCandidateSet | None) -> FoodDetectionCandidateSet:
    if candidate is None:
        msg = "food detection pipeline did not produce a candidate"
        raise RuntimeError(msg)
    return candidate
