from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai_runtime.cv.food.normalization import cleanup_food_query, normalize_food_name


@dataclass(frozen=True)
class FoodMatchResult:
    original_name: str
    query_name: str
    matched_food_name: str | None = None
    matched_food_code: str | None = None
    match_source: str = "local_stub_unmatched"
    match_confidence: float | None = None
    needs_user_confirmation: bool = True


class FoodDbMatcher(Protocol):
    def match(self, query: str) -> FoodMatchResult:
        """Match a food query to a public food DB record.

        TODO: Replace or compose this local stub with a MFDS/FoodNara API client.
        The external client should not live in normalization.py; it should satisfy
        this protocol and return the same FoodMatchResult contract.
        """


_LOCAL_ALIAS_MATCHES = {
    "가래떡": ("가래떡", "local_stub:garaetteok", 0.95),
    "쌀떡": ("가래떡", "local_stub:garaetteok", 0.9),
    "흰떡": ("가래떡", "local_stub:garaetteok", 0.88),
    "ricecake": ("가래떡", "local_stub:garaetteok", 0.86),
    "ricecakes": ("가래떡", "local_stub:garaetteok", 0.86),
    "koreanricecake": ("가래떡", "local_stub:garaetteok", 0.86),
    "garaetteok": ("가래떡", "local_stub:garaetteok", 0.95),
    "흰쌀밥": ("쌀밥", "local_stub:white_rice", 0.92),
    "흰밥": ("쌀밥", "local_stub:white_rice", 0.9),
    "백미밥": ("쌀밥", "local_stub:white_rice", 0.9),
    "white밥": ("쌀밥", "local_stub:white_rice", 0.84),
    "whiterice": ("쌀밥", "local_stub:white_rice", 0.88),
    "plainrice": ("쌀밥", "local_stub:white_rice", 0.84),
}


class LocalFallbackFoodDbMatcher:
    def match(self, query: str) -> FoodMatchResult:
        query_name = cleanup_food_query(query)
        match_key = normalize_food_name(query_name)
        matched = _LOCAL_ALIAS_MATCHES.get(match_key)
        if matched is None:
            return FoodMatchResult(
                original_name=query.strip(),
                query_name=query_name,
            )

        matched_food_name, matched_food_code, confidence = matched
        return FoodMatchResult(
            original_name=query.strip(),
            query_name=query_name,
            matched_food_name=matched_food_name,
            matched_food_code=matched_food_code,
            match_source="local_stub_alias",
            match_confidence=confidence,
            needs_user_confirmation=False,
        )


def match_food_name(query: str, matcher: FoodDbMatcher | None = None) -> FoodMatchResult:
    return (matcher or LocalFallbackFoodDbMatcher()).match(query)
