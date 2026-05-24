from __future__ import annotations

from dataclasses import asdict, dataclass

DISEASE_CODES = ("DM", "HTN", "DL", "OBE", "ANEM")


@dataclass(frozen=True)
class FoodNutritionRow:
    food_name: str
    serving_weight_g: float | None = None
    energy_kcal: float | None = None
    carbohydrate_g: float | None = None
    sugar_g: float | None = None
    fat_g: float | None = None
    protein_g: float | None = None
    calcium_mg: float | None = None
    phosphorus_mg: float | None = None
    sodium_mg: float | None = None
    potassium_mg: float | None = None
    magnesium_mg: float | None = None
    iron_mg: float | None = None
    zinc_mg: float | None = None
    cholesterol_mg: float | None = None
    trans_fat_g: float | None = None


@dataclass(frozen=True)
class DiseaseScoreSet:
    dm_score: float
    htn_score: float
    dl_score: float
    obe_score: float
    anem_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DiseaseFoodScoreRecord:
    food_name: str
    serving_weight_g: float | None
    energy_kcal: float | None
    carbohydrate_g: float | None
    sugar_g: float | None
    fat_g: float | None
    protein_g: float | None
    sodium_mg: float | None
    potassium_mg: float | None
    magnesium_mg: float | None
    iron_mg: float | None
    cholesterol_mg: float | None
    trans_fat_g: float | None
    dm_score: float
    htn_score: float
    dl_score: float
    obe_score: float
    anem_score: float

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)
