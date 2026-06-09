from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ai_runtime.cv.food.matcher import match_food_name

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
    confidence: float | None = None
    needs_review: bool = False
    fallback_reason: str | None = None
    raw_output: dict[str, Any] = field(default_factory=dict)
    raw_provider_status: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def should_use_nutrition_scorer(self) -> bool:
        return bool(self.detected_foods) and not self.needs_review

    def to_scorer_foods(self) -> list[dict[str, Any]]:
        foods: list[dict[str, Any]] = []
        for food_name in self.detected_foods:
            match = match_food_name(food_name)
            display_name = match.matched_food_name or match.query_name
            if not display_name:
                continue

            foods.append(
                {
                    "name": display_name,
                    "original_name": match.original_name,
                    "query_name": match.query_name,
                    "matched_food_name": match.matched_food_name,
                    "matched_food_code": match.matched_food_code,
                    "match_source": match.match_source,
                    "match_confidence": match.match_confidence,
                    "needs_user_confirmation": match.needs_user_confirmation,
                    "confidence": self.confidence,
                    "provider": self.provider,
                }
            )
        return foods
