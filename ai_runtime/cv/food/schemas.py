from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ai_runtime.cv.food.matcher import FoodDbMatcher, FoodMatchResult, match_food_name
from ai_runtime.cv.food.normalization import cleanup_food_query, normalize_food_name

FoodDetectionProvider = Literal["cv_model", "gpt_vision", "rule_based_food_detection"]

CV_CONFIDENCE_THRESHOLD = 0.75
GPT_VISION_FALLBACK_POLICY = "user_confirmation_required"
FOOD_DETECTION_PROVIDER_PRIORITY: tuple[FoodDetectionProvider, ...] = (
    "cv_model",
    "gpt_vision",
    "rule_based_food_detection",
)
MFDS_LOOKUP_CONFIDENCE_THRESHOLD = 0.55
MFDS_LOOKUP_MAX_UNIQUE_QUERIES = 3
GENERIC_MFDS_LOOKUP_BLOCKLIST = frozenset(
    {
        "계란",
        "고기",
        "소고기",
        "돼지고기",
        "닭고기",
        "고추장",
        "된장",
        "간장",
        "소스",
        "양념",
        "야채",
        "채소",
    }
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
        lookup_cache: dict[str, FoodMatchResult] = {}
        lookup_count = 0
        use_lookup_gate = bool(getattr(matcher, "applies_lookup_gating", False))
        for index, food_name in enumerate(self.detected_foods):
            food_confidence = self.detected_food_confidences[index] if index < len(self.detected_food_confidences) else None
            if use_lookup_gate:
                match, did_lookup = _gated_match_food_name(
                    food_name=food_name,
                    confidence=food_confidence,
                    matcher=matcher,
                    lookup_cache=lookup_cache,
                    lookup_count=lookup_count,
                )
                if did_lookup:
                    lookup_count += 1
            else:
                match = match_food_name(food_name, matcher=matcher)

            display_name = match.matched_food_name or match.query_name
            if not display_name:
                continue

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


def _gated_match_food_name(
    *,
    food_name: str,
    confidence: float | None,
    matcher: FoodDbMatcher | None,
    lookup_cache: dict[str, FoodMatchResult],
    lookup_count: int,
) -> tuple[FoodMatchResult, bool]:
    query_name = cleanup_food_query(food_name)
    normalized_query = normalize_food_name(query_name)
    if not query_name:
        return _skipped_food_match(food_name=food_name, query_name=query_name, match_source="mfds_skipped_no_query"), False
    if normalized_query in GENERIC_MFDS_LOOKUP_BLOCKLIST:
        return _skipped_food_match(food_name=food_name, query_name=query_name, match_source="mfds_skipped_generic"), False
    if confidence is not None and confidence < MFDS_LOOKUP_CONFIDENCE_THRESHOLD:
        return _skipped_food_match(
            food_name=food_name,
            query_name=query_name,
            match_source="mfds_skipped_low_confidence",
        ), False

    cached = lookup_cache.get(normalized_query)
    if cached is not None:
        return _copy_match_for_query(cached, food_name=food_name, query_name=query_name), False

    if lookup_count >= MFDS_LOOKUP_MAX_UNIQUE_QUERIES:
        return _skipped_food_match(
            food_name=food_name,
            query_name=query_name,
            match_source="mfds_skipped_lookup_limit",
        ), False

    match = match_food_name(food_name, matcher=matcher)
    lookup_cache[normalized_query] = match
    return match, True


def _skipped_food_match(*, food_name: str, query_name: str, match_source: str) -> FoodMatchResult:
    return FoodMatchResult(
        original_name=food_name.strip(),
        query_name=query_name,
        match_source=match_source,
        needs_user_confirmation=True,
    )


def _copy_match_for_query(match: FoodMatchResult, *, food_name: str, query_name: str) -> FoodMatchResult:
    return FoodMatchResult(
        original_name=food_name.strip(),
        query_name=query_name,
        matched_food_name=match.matched_food_name,
        matched_food_code=match.matched_food_code,
        match_source=match.match_source,
        match_confidence=match.match_confidence,
        needs_user_confirmation=match.needs_user_confirmation,
        metadata=match.metadata,
    )
