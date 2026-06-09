from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ai_runtime.cv.food.schemas import FoodDetectionCandidateSet


@dataclass(frozen=True)
class FoodDetectionProviderResult:
    candidate: FoodDetectionCandidateSet | None
    provider_name: str
    fallback_used: bool = False
    message: str | None = None
    confidence: float | None = None
    raw_food_names: list[str] | None = None
    matched_food_names: list[str] | None = None
    unmatched_food_names: list[str] | None = None
    metadata: dict[str, Any] | None = None


class FoodDetectionProvider(Protocol):
    async def detect(
        self,
        *,
        image_bytes: bytes | None = None,
        image_media_type: str | None = None,
    ) -> FoodDetectionProviderResult:
        """Detect food candidates from an image or deterministic fallback input."""


def build_food_detection_provider_result(
    *,
    provider_name: str,
    candidate: FoodDetectionCandidateSet | None,
    fallback_used: bool = False,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> FoodDetectionProviderResult:
    scorer_foods = candidate.to_scorer_foods() if candidate is not None else []
    raw_food_names = [str(food.get("original_name") or food.get("name") or "").strip() for food in scorer_foods]
    matched_food_names = [
        str(food.get("matched_food_name") or "").strip() for food in scorer_foods if food.get("matched_food_name")
    ]
    unmatched_food_names = [
        str(food.get("original_name") or food.get("query_name") or food.get("name") or "").strip()
        for food in scorer_foods
        if not food.get("matched_food_name")
    ]
    return FoodDetectionProviderResult(
        candidate=candidate,
        provider_name=provider_name,
        fallback_used=fallback_used,
        message=message,
        confidence=candidate.confidence if candidate is not None else None,
        raw_food_names=[name for name in raw_food_names if name],
        matched_food_names=[name for name in matched_food_names if name],
        unmatched_food_names=[name for name in unmatched_food_names if name],
        metadata=metadata or {},
    )
