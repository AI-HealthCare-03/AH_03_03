from __future__ import annotations

from typing import Protocol

from schemas import NutritionLookupResult


class NutritionProvider(Protocol):
    provider_name: str

    def lookup(self, query: str) -> NutritionLookupResult:
        """Return normalized nutrition lookup result for a food query."""
