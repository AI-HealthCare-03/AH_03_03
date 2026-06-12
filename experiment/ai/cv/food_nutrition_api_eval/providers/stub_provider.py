from __future__ import annotations

import time
from dataclasses import dataclass

from schemas import NutritionCandidate, NutritionLookupResult

from ai_runtime.cv.food.normalization import normalize_food_name


@dataclass(frozen=True)
class StubNutritionRecord:
    food_name: str
    food_code: str
    energy_kcal: float
    carbohydrate_g: float
    protein_g: float
    fat_g: float
    sodium_mg: float
    serving_size: str
    source: str = "stub_nutrition_db"


STUB_RECORDS = [
    StubNutritionRecord("가리비", "stub:scallop", 90, 3.2, 17.0, 1.0, 360, "100g"),
    StubNutritionRecord("가래떡", "stub:garaetteok", 235, 52.0, 4.0, 0.5, 120, "100g"),
    StubNutritionRecord("BLT샌드위치", "stub:blt_sandwich", 410, 38.0, 18.0, 22.0, 820, "1 serving"),
    StubNutritionRecord("가지", "stub:eggplant", 25, 5.9, 1.0, 0.2, 2, "100g"),
    StubNutritionRecord("가지구이", "stub:grilled_eggplant", 80, 9.5, 2.0, 4.5, 180, "100g"),
    StubNutritionRecord("가츠동", "stub:katsudon", 720, 92.0, 28.0, 25.0, 980, "1 bowl"),
    StubNutritionRecord("쌀밥", "stub:white_rice", 145, 32.0, 2.7, 0.3, 3, "100g"),
]


class StubNutritionProvider:
    provider_name = "stub"

    def __init__(self, records: list[StubNutritionRecord] | None = None) -> None:
        self.records = records or STUB_RECORDS

    def lookup(self, query: str) -> NutritionLookupResult:
        started = time.perf_counter()
        normalized_query = normalize_food_name(query)
        if not normalized_query:
            return NutritionLookupResult(
                query=query,
                normalized_query=normalized_query,
                provider=self.provider_name,
                status="no_query",
                latency_seconds=round(time.perf_counter() - started, 4),
                error_message="empty query",
                needs_user_confirmation=True,
            )

        candidates = self._find_candidates(normalized_query)
        if not candidates:
            return NutritionLookupResult(
                query=query,
                normalized_query=normalized_query,
                provider=self.provider_name,
                status="not_found",
                latency_seconds=round(time.perf_counter() - started, 4),
                candidate_count=0,
                source="stub_nutrition_db",
                needs_user_confirmation=True,
            )

        selected = candidates[0]
        multiple = len(candidates) > 1
        return NutritionLookupResult(
            query=query,
            normalized_query=normalized_query,
            provider=self.provider_name,
            status="multiple_candidates" if multiple else "matched",
            matched_food_name=selected.food_name,
            matched_food_code=selected.food_code,
            candidate_count=len(candidates),
            top_candidates=[
                NutritionCandidate(
                    food_name=candidate.food_name,
                    food_code=candidate.food_code,
                    confidence=1.0 if idx == 0 else 0.75,
                )
                for idx, candidate in enumerate(candidates[:3])
            ],
            energy_kcal=selected.energy_kcal,
            carbohydrate_g=selected.carbohydrate_g,
            protein_g=selected.protein_g,
            fat_g=selected.fat_g,
            sodium_mg=selected.sodium_mg,
            serving_size=selected.serving_size,
            source=selected.source,
            latency_seconds=round(time.perf_counter() - started, 4),
            needs_user_confirmation=multiple,
        )

    def _find_candidates(self, normalized_query: str) -> list[StubNutritionRecord]:
        exact_matches = [record for record in self.records if normalize_food_name(record.food_name) == normalized_query]
        if exact_matches:
            return exact_matches
        return [
            record
            for record in self.records
            if normalized_query in normalize_food_name(record.food_name)
            or normalize_food_name(record.food_name) in normalized_query
        ]
