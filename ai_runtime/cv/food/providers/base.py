from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai_runtime.cv.food.schemas import FoodDetectionCandidateSet


@dataclass(frozen=True)
class FoodDetectionProviderResult:
    candidate: FoodDetectionCandidateSet | None
    fallback_used: bool = False
    message: str | None = None


class FoodDetectionProvider(Protocol):
    async def detect(
        self,
        *,
        image_bytes: bytes | None = None,
        image_media_type: str | None = None,
    ) -> FoodDetectionProviderResult:
        """Detect food candidates from an image or deterministic fallback input."""
