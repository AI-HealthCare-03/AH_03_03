from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_runtime.cv.food.nutrition.scoring.disease_food_scorer import (
    DEFAULT_SCORE_CSV_PATH,
    DiseaseFoodScorer,
    build_food_disease_score_csv,
)
from ai_runtime.cv.food.nutrition.scoring.schemas import FoodNutritionRow


def test_scores_are_bounded_for_five_disease_groups() -> None:
    scorer = DiseaseFoodScorer()
    scores = scorer.score_food(
        FoodNutritionRow(
            food_name="테스트 음식",
            energy_kcal=500,
            carbohydrate_g=70,
            sugar_g=12,
            fat_g=16,
            protein_g=25,
            sodium_mg=700,
            potassium_mg=500,
            magnesium_mg=80,
            iron_mg=3,
            zinc_mg=2,
            cholesterol_mg=90,
            trans_fat_g=0,
        )
    ).to_dict()

    assert set(scores) == {"dm_score", "htn_score", "dl_score", "obe_score", "anem_score"}
    assert all(0 <= score <= 100 for score in scores.values())


def test_runtime_csv_can_be_loaded_without_raw_excel() -> None:
    records = DiseaseFoodScorer(DEFAULT_SCORE_CSV_PATH).load_runtime_scores()

    assert records
    assert records[0].food_name
    assert 0 <= records[0].dm_score <= 100


def test_build_csv_from_excel_keeps_runtime_columns(tmp_path: Path) -> None:
    excel_path = tmp_path / "food_nutrition_db.xlsx"
    output_path = tmp_path / "food_disease_scores.csv"
    pd.DataFrame(
        [
            {
                "음 식 명": "샘플밥",
                "중량(g)": 200,
                "에너지(kcal)": 320,
                "탄수화물(g)": 65,
                "당류(g)": 0,
                "지방(g)": 1,
                "단백질(g)": 8,
                "칼슘(mg)": 20,
                "인(mg)": 120,
                "나트륨(mg)": 5,
                "칼륨(mg)": 250,
                "마그네슘(mg)": 22,
                "철(mg)": 1.7,
                "아연(mg)": 1.6,
                "콜레스테롤(mg)": 0,
                "트랜스지방(g)": 0,
            }
        ]
    ).to_excel(excel_path, index=False)

    rows, columns = build_food_disease_score_csv(excel_path=excel_path, output_path=output_path)
    result = pd.read_csv(output_path)

    assert (rows, columns) == (1, 18)
    assert result.loc[0, "food_name"] == "샘플밥"
    assert {"dm_score", "htn_score", "dl_score", "obe_score", "anem_score"}.issubset(result.columns)
