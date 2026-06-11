from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ai_runtime.cv.food.matcher import FoodDbMatcher, match_food_name

FoodDetectionProvider = Literal["cv_model", "gpt_vision", "rule_based_food_detection"]

CV_CONFIDENCE_THRESHOLD = 0.75
GPT_VISION_FALLBACK_POLICY = "user_confirmation_required"
FOOD_DETECTION_PROVIDER_PRIORITY: tuple[FoodDetectionProvider, ...] = (
    "cv_model",
    "gpt_vision",
    "rule_based_food_detection",
)


@dataclass(frozen=True)
class FoodDetectionCandidateSet:
    provider: FoodDetectionProvider
    detected_foods: list[str]
    detected_food_confidences: list[float | None] = field(default_factory=list)
    confidence: float | None = None
    needs_review: bool = False
    fallback_reason: str | None = None
    raw_output: dict[str, Any] = field(default_factory=dict)
    raw_provider_status: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def should_use_nutrition_scorer(self) -> bool:
        return bool(self.detected_foods) and not self.needs_review

    def to_scorer_foods(self, matcher: FoodDbMatcher | None = None) -> list[dict[str, Any]]:
        foods: list[dict[str, Any]] = []
        for index, food_name in enumerate(self.detected_foods):
            match = match_food_name(food_name, matcher=matcher)
            display_name = match.matched_food_name or match.query_name
            if not display_name:
                continue
            food_confidence = self.detected_food_confidences[index] if index < len(self.detected_food_confidences) else None

            food = {
                "name": display_name,
                "original_name": match.original_name,
                "query_name": match.query_name,
                "matched_food_name": match.matched_food_name,
                "matched_food_code": match.matched_food_code,
                "match_source": match.match_source,
                "match_confidence": match.match_confidence,
                "needs_user_confirmation": match.needs_user_confirmation,
                "confidence": food_confidence if food_confidence is not None else self.confidence,
                "provider": self.provider,
            }
            if match.metadata:
                food["match_metadata"] = match.metadata
            foods.append(food)
        return foods
