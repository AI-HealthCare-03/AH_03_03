from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NutritionCandidate:
    food_name: str
    food_code: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NutritionLookupResult:
    query: str
    normalized_query: str
    provider: str
    status: str
    matched_food_name: str | None = None
    matched_food_code: str | None = None
    candidate_count: int = 0
    top_candidates: list[NutritionCandidate] = field(default_factory=list)
    energy_kcal: float | None = None
    carbohydrate_g: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    sodium_mg: float | None = None
    serving_size: str | None = None
    source: str | None = None
    latency_seconds: float = 0.0
    error_message: str | None = None
    cache_hit: bool = False
    needs_user_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["top_candidates"] = [candidate.to_dict() for candidate in self.top_candidates]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NutritionLookupResult:
        top_candidates = [
            NutritionCandidate(
                food_name=str(candidate.get("food_name") or ""),
                food_code=_optional_string(candidate.get("food_code")),
                confidence=_optional_float(candidate.get("confidence")),
            )
            for candidate in payload.get("top_candidates", [])
            if isinstance(candidate, dict)
        ]
        return cls(
            query=str(payload.get("query") or ""),
            normalized_query=str(payload.get("normalized_query") or ""),
            provider=str(payload.get("provider") or ""),
            status=str(payload.get("status") or ""),
            matched_food_name=_optional_string(payload.get("matched_food_name")),
            matched_food_code=_optional_string(payload.get("matched_food_code")),
            candidate_count=int(payload.get("candidate_count") or 0),
            top_candidates=top_candidates,
            energy_kcal=_optional_float(payload.get("energy_kcal")),
            carbohydrate_g=_optional_float(payload.get("carbohydrate_g")),
            protein_g=_optional_float(payload.get("protein_g")),
            fat_g=_optional_float(payload.get("fat_g")),
            sodium_mg=_optional_float(payload.get("sodium_mg")),
            serving_size=_optional_string(payload.get("serving_size")),
            source=_optional_string(payload.get("source")),
            latency_seconds=float(payload.get("latency_seconds") or 0.0),
            error_message=_optional_string(payload.get("error_message")),
            cache_hit=bool(payload.get("cache_hit", False)),
            needs_user_confirmation=bool(payload.get("needs_user_confirmation", False)),
        )


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_float(value: object) -> float | None:
    try:
        raw_value = str(value).strip()
        return float(raw_value) if raw_value else None
    except (TypeError, ValueError):
        return None
