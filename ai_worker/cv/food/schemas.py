from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
    raw_provider_status: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def should_use_nutrition_scorer(self) -> bool:
        return bool(self.detected_foods) and not self.needs_review
