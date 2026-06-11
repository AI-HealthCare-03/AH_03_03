from __future__ import annotations

from typing import Any

from ai_runtime.cv.food.schemas import (
    CV_CONFIDENCE_THRESHOLD,
    FoodDetectionCandidateSet,
)


def select_food_detection_candidate(
    *,
    cv_result: FoodDetectionCandidateSet | None = None,
    rule_based_foods: list[dict[str, Any]] | None = None,
    gpt_vision_fallback_enabled: bool = False,
    confidence_threshold: float = CV_CONFIDENCE_THRESHOLD,
) -> FoodDetectionCandidateSet:
    if cv_result is None or not cv_result.detected_foods:
        return _rule_based_candidate(rule_based_foods or [])

    needs_review = cv_result.needs_review or _is_below_threshold(cv_result.confidence, confidence_threshold)
    fallback_reason = cv_result.fallback_reason
    if needs_review and fallback_reason is None:
        fallback_reason = "cv_confidence_below_threshold"

    if needs_review and gpt_vision_fallback_enabled:
        return FoodDetectionCandidateSet(
            provider="gpt_vision",
            detected_foods=[],
            confidence=None,
            needs_review=True,
            fallback_reason="gpt_vision_candidate_requires_user_confirmation",
            raw_output={
                "fallback_candidate": True,
                "source_provider": cv_result.provider,
                "source_confidence": cv_result.confidence,
                "source_detected_foods": cv_result.detected_foods,
            },
            metadata={
                "policy": "user_confirmation_required",
                "called": False,
            },
        )

    return FoodDetectionCandidateSet(
        provider=cv_result.provider,
        detected_foods=cv_result.detected_foods,
        detected_food_confidences=cv_result.detected_food_confidences,
        confidence=cv_result.confidence,
        needs_review=needs_review,
        fallback_reason=fallback_reason,
        raw_output=cv_result.raw_output,
        raw_provider_status=cv_result.raw_provider_status,
        metadata=cv_result.metadata,
    )


def _rule_based_candidate(rule_based_foods: list[dict[str, Any]]) -> FoodDetectionCandidateSet:
    food_names = [name for food in rule_based_foods if (name := _food_name(food))]
    return FoodDetectionCandidateSet(
        provider="rule_based_food_detection",
        detected_foods=food_names,
        detected_food_confidences=_food_confidences(rule_based_foods),
        confidence=_average_confidence(rule_based_foods),
        needs_review=False,
        raw_output={
            "source": "rule_based_food_detection",
            "foods": rule_based_foods,
        },
        metadata={
            "gpt_vision_called": False,
        },
    )


def _food_name(food: dict[str, Any]) -> str:
    return str(food.get("name") or food.get("food_name") or "").strip()


def _average_confidence(foods: list[dict[str, Any]]) -> float | None:
    values = [float(value) for food in foods if _is_number(value := food.get("confidence"))]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _food_confidences(foods: list[dict[str, Any]]) -> list[float | None]:
    return [round(float(value), 4) if _is_number(value := food.get("confidence")) else None for food in foods if _food_name(food)]


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_below_threshold(confidence: float | None, threshold: float) -> bool:
    return confidence is None or confidence < threshold
