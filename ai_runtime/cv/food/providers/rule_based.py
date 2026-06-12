from __future__ import annotations

from typing import Any

from ai_runtime.cv.food.fallback_policy import select_food_detection_candidate
from ai_runtime.cv.food.providers.base import FoodDetectionProviderResult, build_food_detection_provider_result


class RuleBasedFoodDetectionProvider:
    def __init__(self, *, rule_based_foods: list[dict[str, Any]]) -> None:
        self._rule_based_foods = rule_based_foods

    async def detect(
        self,
        *,
        image_bytes: bytes | None = None,
        image_media_type: str | None = None,
    ) -> FoodDetectionProviderResult:
        candidate = select_food_detection_candidate(
            cv_result=None,
            rule_based_foods=self._rule_based_foods,
        )
        return build_food_detection_provider_result(
            provider_name="rule_based_food_detection",
            candidate=candidate,
            fallback_used=True,
            message="rule_based_food_detection fallback used",
            metadata={"source": "rule_based_food_detection"},
        )
