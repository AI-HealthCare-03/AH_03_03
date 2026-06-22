from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from schemas import NutritionLookupResult

from ai_runtime.cv.food.normalization import normalize_food_name


class LocalJsonNutritionCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries = self._load()

    def get(self, query: str) -> NutritionLookupResult | None:
        payload = self._entries.get(normalize_food_name(query))
        if not isinstance(payload, dict):
            return None
        result = NutritionLookupResult.from_dict(payload)
        return replace(result, cache_hit=True)

    def set(self, query: str, result: NutritionLookupResult) -> None:
        self._entries[normalize_food_name(query)] = result.to_dict()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}
