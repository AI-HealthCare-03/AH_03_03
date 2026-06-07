from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from ai_runtime.cv.food.nutrition.scoring.schemas import DiseaseFoodScoreRecord, DiseaseScoreSet, FoodNutritionRow

REPO_ROOT = Path(__file__).resolve().parents[5]
NUTRITION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_EXCEL_PATH = REPO_ROOT / "etc" / "ai" / "cv" / "food" / "nutrition" / "raw" / "food_nutrition_db.xlsx"
DEFAULT_RULE_PATH = NUTRITION_DIR / "rules" / "disease_score_rules.json"
DEFAULT_SCORE_CSV_PATH = NUTRITION_DIR / "data" / "food_disease_scores.csv"

COLUMN_MAP = {
    "음 식 명": "food_name",
    "음식명": "food_name",
    "중량(g)": "serving_weight_g",
    "에너지(kcal)": "energy_kcal",
    "탄수화물(g)": "carbohydrate_g",
    "당류(g)": "sugar_g",
    "지방(g)": "fat_g",
    "단백질(g)": "protein_g",
    "칼슘(mg)": "calcium_mg",
    "인(mg)": "phosphorus_mg",
    "나트륨(mg)": "sodium_mg",
    "칼륨(mg)": "potassium_mg",
    "마그네슘(mg)": "magnesium_mg",
    "철(mg)": "iron_mg",
    "아연(mg)": "zinc_mg",
    "콜레스테롤(mg)": "cholesterol_mg",
    "트랜스지방(g)": "trans_fat_g",
}

RUNTIME_COLUMNS = [
    "food_name",
    "serving_weight_g",
    "energy_kcal",
    "carbohydrate_g",
    "sugar_g",
    "fat_g",
    "protein_g",
    "calcium_mg",
    "sodium_mg",
    "potassium_mg",
    "magnesium_mg",
    "iron_mg",
    "zinc_mg",
    "cholesterol_mg",
    "trans_fat_g",
    "dm_score",
    "htn_score",
    "dl_score",
    "obe_score",
    "anem_score",
]


def _clean_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped in {"-", "Trace", "trace"}:
            return None
        value = stripped.replace(",", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_score(score: float) -> float:
    return round(max(0.0, min(100.0, score)), 1)


def _linear_penalty(value: float | None, rule: dict[str, float]) -> float:
    if value is None:
        return 0.0
    start = float(rule["start"])
    limit = float(rule["limit"])
    points = float(rule["points"])
    if value <= start:
        return 0.0
    if value >= limit:
        return points
    return ((value - start) / (limit - start)) * points


def _linear_bonus(value: float | None, rule: dict[str, float]) -> float:
    if value is None or value <= 0:
        return 0.0
    target = float(rule["target"])
    points = float(rule["points"])
    if value >= target:
        return points
    return (value / target) * points


def _load_rules(rule_path: Path = DEFAULT_RULE_PATH) -> dict[str, Any]:
    with rule_path.open(encoding="utf-8") as file:
        return json.load(file)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns={column: COLUMN_MAP.get(str(column).strip(), column) for column in df.columns})
    if "food_name" not in normalized.columns:
        raise ValueError("Nutrition source must include a food name column.")

    normalized["food_name"] = normalized["food_name"].astype(str).str.strip()
    normalized = normalized[normalized["food_name"].ne("")].copy()
    for column in set(COLUMN_MAP.values()) - {"food_name"}:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(_clean_number)
        else:
            normalized[column] = None
    return normalized


class DiseaseFoodScorer:
    """Runtime scorer backed by generated CSV data and JSON rules.

    The raw Excel file is only used by explicit build functions below. Service
    runtime code should instantiate this class with food_disease_scores.csv.
    """

    def __init__(
        self,
        score_csv_path: Path = DEFAULT_SCORE_CSV_PATH,
        rule_path: Path = DEFAULT_RULE_PATH,
    ) -> None:
        self.score_csv_path = score_csv_path
        self.rules = _load_rules(rule_path)

    def score_food(self, row: FoodNutritionRow | dict[str, Any]) -> DiseaseScoreSet:
        values = row if isinstance(row, dict) else row.__dict__
        disease_rules = self.rules["diseases"]
        return DiseaseScoreSet(
            dm_score=self._score_for_disease(values, disease_rules["DM"]),
            htn_score=self._score_for_disease(values, disease_rules["HTN"]),
            dl_score=self._score_for_disease(values, disease_rules["DL"]),
            obe_score=self._score_for_disease(values, disease_rules["OBE"]),
            anem_score=self._score_for_disease(values, disease_rules["ANEM"]),
        )

    def load_runtime_scores(self) -> list[DiseaseFoodScoreRecord]:
        df = pd.read_csv(self.score_csv_path)
        return [
            DiseaseFoodScoreRecord(
                food_name=str(row["food_name"]),
                serving_weight_g=_clean_number(row.get("serving_weight_g")),
                energy_kcal=_clean_number(row.get("energy_kcal")),
                carbohydrate_g=_clean_number(row.get("carbohydrate_g")),
                sugar_g=_clean_number(row.get("sugar_g")),
                fat_g=_clean_number(row.get("fat_g")),
                protein_g=_clean_number(row.get("protein_g")),
                calcium_mg=_clean_number(row.get("calcium_mg")),
                sodium_mg=_clean_number(row.get("sodium_mg")),
                potassium_mg=_clean_number(row.get("potassium_mg")),
                magnesium_mg=_clean_number(row.get("magnesium_mg")),
                iron_mg=_clean_number(row.get("iron_mg")),
                zinc_mg=_clean_number(row.get("zinc_mg")),
                cholesterol_mg=_clean_number(row.get("cholesterol_mg")),
                trans_fat_g=_clean_number(row.get("trans_fat_g")),
                dm_score=float(row["dm_score"]),
                htn_score=float(row["htn_score"]),
                dl_score=float(row["dl_score"]),
                obe_score=float(row["obe_score"]),
                anem_score=float(row["anem_score"]),
            )
            for _, row in df.iterrows()
        ]

    def _score_for_disease(self, values: dict[str, Any], rule: dict[str, Any]) -> float:
        score = float(rule.get("base_score", 100))
        for nutrient, penalty_rule in rule.get("penalties", {}).items():
            score -= _linear_penalty(_clean_number(values.get(nutrient)), penalty_rule)
        for nutrient, bonus_rule in rule.get("bonuses", {}).items():
            score += _linear_bonus(_clean_number(values.get(nutrient)), bonus_rule)
        return _clamp_score(score)


def build_score_records_from_excel(
    excel_path: Path = DEFAULT_RAW_EXCEL_PATH,
    rule_path: Path = DEFAULT_RULE_PATH,
) -> list[DiseaseFoodScoreRecord]:
    source_df = pd.read_excel(excel_path)
    normalized_df = _normalize_dataframe(source_df)
    scorer = DiseaseFoodScorer(rule_path=rule_path)
    records: list[DiseaseFoodScoreRecord] = []

    for _, row in normalized_df.iterrows():
        nutrition = FoodNutritionRow(
            food_name=str(row["food_name"]),
            serving_weight_g=_clean_number(row.get("serving_weight_g")),
            energy_kcal=_clean_number(row.get("energy_kcal")),
            carbohydrate_g=_clean_number(row.get("carbohydrate_g")),
            sugar_g=_clean_number(row.get("sugar_g")),
            fat_g=_clean_number(row.get("fat_g")),
            protein_g=_clean_number(row.get("protein_g")),
            calcium_mg=_clean_number(row.get("calcium_mg")),
            phosphorus_mg=_clean_number(row.get("phosphorus_mg")),
            sodium_mg=_clean_number(row.get("sodium_mg")),
            potassium_mg=_clean_number(row.get("potassium_mg")),
            magnesium_mg=_clean_number(row.get("magnesium_mg")),
            iron_mg=_clean_number(row.get("iron_mg")),
            zinc_mg=_clean_number(row.get("zinc_mg")),
            cholesterol_mg=_clean_number(row.get("cholesterol_mg")),
            trans_fat_g=_clean_number(row.get("trans_fat_g")),
        )
        scores = scorer.score_food(nutrition)
        records.append(
            DiseaseFoodScoreRecord(
                food_name=nutrition.food_name,
                serving_weight_g=nutrition.serving_weight_g,
                energy_kcal=nutrition.energy_kcal,
                carbohydrate_g=nutrition.carbohydrate_g,
                sugar_g=nutrition.sugar_g,
                fat_g=nutrition.fat_g,
                protein_g=nutrition.protein_g,
                calcium_mg=nutrition.calcium_mg,
                sodium_mg=nutrition.sodium_mg,
                potassium_mg=nutrition.potassium_mg,
                magnesium_mg=nutrition.magnesium_mg,
                iron_mg=nutrition.iron_mg,
                zinc_mg=nutrition.zinc_mg,
                cholesterol_mg=nutrition.cholesterol_mg,
                trans_fat_g=nutrition.trans_fat_g,
                **scores.to_dict(),
            )
        )
    return records


def write_food_disease_score_csv(
    records: Iterable[DiseaseFoodScoreRecord],
    output_path: Path = DEFAULT_SCORE_CSV_PATH,
    columns: list[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or RUNTIME_COLUMNS
    all_dicts = [record.to_dict() for record in records]
    # columns에 없는 키는 무시, 있는 키만 추출
    df = pd.DataFrame(all_dicts)
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    numeric_columns = [c for c in df.columns if c != "food_name"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce").round(4)
    df.to_csv(output_path, index=False, encoding="utf-8")


def build_food_disease_score_csv(
    excel_path: Path = DEFAULT_RAW_EXCEL_PATH,
    output_path: Path = DEFAULT_SCORE_CSV_PATH,
    rule_path: Path = DEFAULT_RULE_PATH,
) -> tuple[int, int]:
    records = build_score_records_from_excel(excel_path=excel_path, rule_path=rule_path)
    write_food_disease_score_csv(records, output_path=output_path)
    return len(records), len(RUNTIME_COLUMNS)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime food disease score CSV from raw nutrition Excel.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_RAW_EXCEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCORE_CSV_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULE_PATH)
    args = parser.parse_args()

    rows, columns = build_food_disease_score_csv(args.excel, args.output, args.rules)
    print(f"wrote {args.output} rows={rows} columns={columns}")


if __name__ == "__main__":
    _main()
