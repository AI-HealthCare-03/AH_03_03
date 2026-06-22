from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabelRow:
    row_id: int
    image_path: str
    expected_foods: list[str]
    image_exists: bool = True


@dataclass(frozen=True)
class PredictionRow:
    row_id: int
    image_path: str
    expected_foods: list[str]
    raw_food_names: list[str] = field(default_factory=list)
    allowed_food_names: list[str] = field(default_factory=list)
    canonical_food_names: list[str] = field(default_factory=list)
    unmatched_food_names: list[str] = field(default_factory=list)
    invalid_label_count: int = 0
    constrained_by_allowed_foods: bool = False
    confidence: float | None = None
    api_success: bool = False
    json_parse_success: bool = False
    empty_result: bool = False
    latency_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def expected_canonical_foods(self) -> list[str]:
        from metrics import canonicalize_food_names

        return canonicalize_food_names(self.expected_foods)
