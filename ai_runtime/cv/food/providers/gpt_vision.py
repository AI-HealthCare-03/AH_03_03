from __future__ import annotations

import logging
from typing import Any

from ai_runtime.cv.food.providers.base import FoodDetectionProviderResult, build_food_detection_provider_result
from ai_runtime.cv.food.schemas import FoodDetectionCandidateSet
from ai_runtime.cv.providers.gpt_vision import AnalysisType, VisionClient

logger = logging.getLogger(__name__)


class GptVisionFoodDetectionProvider:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        vision_client_cls: type[Any] = VisionClient,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._vision_client_cls = vision_client_cls

    async def detect(
        self,
        *,
        image_bytes: bytes | None = None,
        image_media_type: str | None = None,
    ) -> FoodDetectionProviderResult:
        if not image_bytes:
            return build_food_detection_provider_result(
                provider_name="gpt_vision",
                candidate=None,
                fallback_used=True,
                message="image file not provided",
            )
        if not self._api_key:
            return build_food_detection_provider_result(
                provider_name="gpt_vision",
                candidate=None,
                fallback_used=True,
                message="OPENAI_API_KEY missing",
            )

        try:
            client = self._vision_client_cls(api_key=self._api_key, model=self._model)
            raw = await client.analyze(
                analysis_type=AnalysisType.DIET,
                image_bytes=image_bytes,
                media_type=image_media_type or "image/jpeg",
            )
        except Exception:
            logger.exception("Diet GPT Vision provider failed; using rule_based_food_detection fallback")
            return build_food_detection_provider_result(
                provider_name="gpt_vision",
                candidate=None,
                fallback_used=True,
                message="gpt_vision_failed",
            )

        foods = raw.get("foods") if isinstance(raw, dict) else None
        if not isinstance(foods, list):
            return build_food_detection_provider_result(
                provider_name="gpt_vision",
                candidate=None,
                fallback_used=True,
                message="gpt_vision_returned_no_foods",
            )

        detected_foods = [
            str(food.get("name") or food.get("food_name") or "").strip()
            for food in foods
            if isinstance(food, dict) and str(food.get("name") or food.get("food_name") or "").strip()
        ]
        if not detected_foods:
            return build_food_detection_provider_result(
                provider_name="gpt_vision",
                candidate=None,
                fallback_used=True,
                message="gpt_vision_returned_empty_foods",
            )

        candidate = FoodDetectionCandidateSet(
            provider="gpt_vision",
            detected_foods=detected_foods,
            confidence=_average_confidence_from_provider_foods(foods),
            needs_review=False,
            raw_output=raw,
            raw_provider_status=str(raw.get("analysis_status") or "success"),
            metadata={"model": self._model},
        )
        return build_food_detection_provider_result(
            provider_name="gpt_vision",
            candidate=candidate,
            fallback_used=False,
            message="gpt_vision_food_detection",
            metadata={"model": self._model},
        )


def _average_confidence_from_provider_foods(foods: list[Any]) -> float | None:
    values = [float(value) for food in foods if isinstance(food, dict) and _is_number(value := food.get("confidence"))]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
